"""Bounded accelerator training. A failed pilot never launches full training."""
from __future__ import annotations
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import fcntl
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from sim.config import load_config
from .dataset import DiffusionDataset
from .profiles import profile_spec, validate_task, LEGACY, NATIVE
from .policy import DiffusionPolicy, build_policy, expected_parameters, TRANSFORMER_OPTIONS
from .training import (WeightedCursor, ResourceGuard, atomic_json,
                       save_checkpoint, restore_checkpoint, load_payload,
                       verify_continuation)
from .vendor.model.diffusion.ema_model import EMAModel
from .config import DEFAULT_DATASET, DEFAULT_OUT
from .device import resolve_device, seed_all, synchronize, empty_cache, memory_peaks


def training_config(steps=None, *, device="mps", batch_size=1, accumulation=2, backbone="unet", profile=None):
    steps = 100000 if steps is None else int(steps)
    if not 1 <= steps <= 100000:
        raise ValueError("steps must be between 1 and 100000")
    if device not in ("mps", "cuda:0", "cuda"):
        raise ValueError("training device must be mps or cuda:0; CPU fallback forbidden")
    if any(type(v) is not int or v < 1 for v in (batch_size, accumulation)):
        raise ValueError("batch_size and accumulation must be positive integers")
    if backbone not in {"unet","transformer"}:
        raise ValueError("unknown diffusion backbone")
    config = dict(seed=42,max_steps=steps,max_seconds=86400,batch_size=batch_size,
                accumulation=accumulation,workers=0,device="cuda:0" if device=="cuda" else device,dtype="float32",
                lr=1e-4,betas=[.95,.999],weight_decay=1e-6,clip=1.,
                warmup=500,checkpoint_every=500,keep_checkpoints=2,
                approach_fraction=.35,obs_steps=2,horizon=28,output_steps=25,
                execute_steps=5,image_size=[320,320],unet_dims=[256,512,1024],
                diffusion_train_steps=100,ddim_steps=8,eta=0,ema_power=.75,
                upstream="5ba07ac6661db573af695b419a7947ecb704690f")
    if backbone == "transformer":
        config.pop("unet_dims")
        config.update(backbone="transformer",transformer=dict(TRANSFORMER_OPTIONS),
                      betas=[.9,.95],weight_decay=1e-3,obs_weight_decay=1e-6,
                      warmup=1000,attention_backend="math")
    spec=profile_spec(profile)
    if profile and profile!=LEGACY:
        if backbone!="transformer": raise ValueError("native profile requires Transformer")
        config.update(observation_profile=profile,camera_views=list(spec["views"]),image_size=list(spec["size"]),
                      scene_profile_sha256=spec["scene_sha256"])
    return config


def make_optimizer(model, config):
    if config.get("backbone","unet") == "transformer":
        groups = model.transformer.get_optim_groups(weight_decay=config["weight_decay"])
        groups.append(dict(params=list(model.obs_encoder.parameters()),
                           weight_decay=config["obs_weight_decay"]))
    else:
        groups = model.parameters()
    return torch.optim.AdamW(groups,lr=config["lr"],betas=tuple(config["betas"]),
                             weight_decay=config["weight_decay"],foreach=False)


def make_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda s: min((s+1)/config["warmup"],1.)*.5*(1+math.cos(
            math.pi*max(0,s-config["warmup"])/max(1,config["max_steps"]-config["warmup"]))))


def effective_config(args, frozen):
    config = training_config(args.steps if args.steps is not None else frozen.get("max_steps"),
        device=args.device or frozen.get("device","mps"),
        batch_size=args.batch_size if args.batch_size is not None else frozen.get("batch_size",1),
        accumulation=args.accumulation if args.accumulation is not None else frozen.get("accumulation",2),
        backbone=args.backbone or frozen.get("backbone","unet"),
        profile=getattr(args,"profile",None) or frozen.get("observation_profile"))
    if frozen and frozen != config:
        raise ValueError("launch training config mismatch")
    return config


def validate_benchmark_config(benchmark, config):
    if benchmark.get("backbone","unet") != config.get("backbone","unet"):
        raise ValueError("benchmark backbone mismatch")
    measured = benchmark.get("config")
    if measured is None:
        if config.get("backbone","unet") == "transformer":
            raise ValueError("Transformer benchmark config missing")
        return
    keys = set(config) | set(measured)
    if any(config.get(k)!=measured.get(k) for k in keys-{"batch_size","accumulation"}):
        raise ValueError("benchmark model/optimizer config mismatch")


def pilot_passes(initial, final, updates):
    return (math.isfinite(initial) and math.isfinite(final) and initial > 0
            and final <= initial*.7 and all(updates.values()))


def validate_resume(book, payload, pilot_only=False):
    stage = payload["stage"]
    if pilot_only and stage=="full":
        raise ValueError("pilot-only cannot resume a full-training checkpoint")
    if not book:
        raise ValueError("resume requires the original status.json sidecar")
    info = book.get(stage,{})
    if (float(book.get("budget_seconds",-1)) < float(payload["seconds"])
            or float(info.get("elapsed_seconds",-1)) < float(payload.get("stage_elapsed",0))
            or int(info.get("completed_updates",-1)) < int(payload["step"])):
        raise ValueError("stale status.json would reset a training budget")
    if stage=="pilot" and payload["step"]>0 and not {
            "initial_loss","initial_parameter_hashes"}.issubset(info):
        raise ValueError("stale pilot gate state in status.json")
    return stage


def pilot_needed(stage, book):
    return stage=="pilot" and not bool(book.get("pilot",{}).get("passed"))


def to_device(batch, device="mps"):
    if isinstance(batch,dict):
        return {k:to_device(v,device) for k,v in batch.items()}
    return batch.to(device) if isinstance(batch,torch.Tensor) else batch


def window_batch(ds, index):
    indices = [int(index)] if isinstance(index,(int,np.integer)) else list(index)
    return default_collate([{k:item[k] for k in ("obs","action","valid_mask")}
                            for item in (ds[i] for i in indices)])


def _concat_batches(batches):
    return {k:(_concat_batches([b[k] for b in batches]) if isinstance(batches[0][k],dict)
               else torch.cat([b[k] for b in batches],dim=0)) for k in batches[0]}


def optimizer_update(model,ema,optimizer,scheduler,sampler,ds,config,*,step,windows=None):
    """One optimizer update; normalize over all valid actions across accumulation."""
    read_started=time.perf_counter()
    size,accumulation=config["batch_size"],config["accumulation"]
    batches=[]
    for micro in range(accumulation):
        indices=[sampler.next() for _ in range(size)]
        if windows is None:
            batch=window_batch(ds,indices)
        else:
            offset=(step*accumulation+micro)*size
            batch=_concat_batches([windows[(offset+j)%len(windows)] for j in range(size)])
        batches.append(batch)
    data_seconds=time.perf_counter()-read_started
    total_valid=sum(int(b["valid_mask"].sum()) for b in batches)
    if not total_valid: raise ValueError("update has no valid actions")
    optimizer.zero_grad(set_to_none=True)
    loss_value=0.
    for cpu_batch in batches:
        batch=to_device(cpu_batch,config["device"])
        weight=int(cpu_batch["valid_mask"].sum())/total_valid
        loss=model.loss(batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite diffusion training loss")
        (loss*weight).backward()
        loss_value+=float(loss.detach().item())*weight
        del batch,loss
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["clip"],error_if_nonfinite=True)
    optimizer.step(); scheduler.step(); ema.step(model)
    synchronize(config["device"])
    return dict(loss=loss_value,grad_norm=float(norm.item()),data_seconds=data_seconds,
                windows=size*accumulation)


def initial_budget(benchmark, identity):
    if not benchmark: return 0.
    if benchmark.get("identity") != identity:
        raise ValueError("benchmark dataset identity mismatch")
    seconds=float(benchmark.get("training_seconds",-1))
    if not math.isfinite(seconds) or not 0 <= seconds <= 600:
        raise ValueError("invalid benchmark training budget")
    if benchmark.get("state")!="passed":
        raise ValueError("failed benchmark cannot start formal pilot")
    return seconds


def parameter_marks(model):
    result = {}
    modules = {f"vision:{v}":model.obs_encoder.key_model_map[v]
               for v in model.obs_encoder.rgb_keys}
    modules[model.backbone] = model.noise_model
    for key,module in modules.items():
        p = next(p for p in module.parameters() if p.numel()).detach().cpu().numpy()
        result[key] = hashlib.sha256(p.tobytes()).hexdigest()
    return result


@torch.no_grad()
def fixed_loss(model, windows, deadline):
    model.eval()
    losses = []
    device=next(model.parameters()).device
    for j, batch in enumerate(windows):
        if time.perf_counter() >= deadline:
            raise TimeoutError("pilot 30-minute limit reached during fixed-noise evaluation")
        generator = torch.Generator(device="cpu").manual_seed(424200+j)
        n=batch["action"].shape[0]
        noise = torch.randn((n,28,4),generator=generator).to(device)
        timestep = torch.full((n,),j*8%100,dtype=torch.long,device=device)
        losses.append(float(model.loss(to_device(batch,device),noise,timestep).item()))
    model.train()
    value = float(np.mean(losses))
    if not math.isfinite(value):
        raise RuntimeError("nonfinite fixed-noise pilot loss")
    return value


def train_stage(ds, out, config, book, *, stage, resume=None, stop):
    started = time.perf_counter()
    base_seconds = float(book.get("budget_seconds",0))
    stage_prior = float(book.get(stage,{}).get("elapsed_seconds",0))
    deadline = started+min(86400-base_seconds,
                          max(0,1800-stage_prior) if stage=="pilot" else 86400)
    if deadline <= started:
        raise RuntimeError("cumulative 24-hour training budget exhausted")
    root = out/stage
    root.mkdir(parents=True,exist_ok=True)
    guard = ResourceGuard(Path(__file__).resolve().parents[2])
    device=resolve_device(config["device"])
    seed_all(config["seed"],device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(1)
    model = ema = optimizer = scheduler = None
    step = 0
    checkpoint = None
    stage_info = book.setdefault(stage,{})
    def seconds():
        return base_seconds+time.perf_counter()-started
    def status(**kw):
        book.update(kw,budget_seconds=seconds())
        stage_info["completed_updates"] = step
        stage_info["elapsed_seconds"] = stage_prior+time.perf_counter()-started
        atomic_json(out/"status.json",book)
    try:
        status(state="initializing",stage=stage)
        model = build_policy(ds.normalization,config).to(device,dtype=torch.float32)
        count = sum(p.numel() for p in model.parameters())
        if count != expected_parameters(config):
            raise RuntimeError("architecture parameter count changed")
        print(json.dumps(dict(event="architecture",parameters=count,
                              vision_parameters=sum(p.numel() for p in model.obs_encoder.parameters()),
                              backbone=config.get("backbone","unet"),
                              device=str(device),dtype="float32",batch_size=config["batch_size"],
                              accumulation=config["accumulation"],workers=config["workers"])),flush=True)
        ema = EMAModel(copy.deepcopy(model),power=.75)
        ema.averaged_model.eval()
        optimizer = make_optimizer(model,config)
        scheduler = make_scheduler(optimizer,config)
        sampler = WeightedCursor(ds.phase_sampling_weights(config["approach_fraction"]),config["seed"])
        if resume:
            restored = restore_checkpoint(resume,model,ema,optimizer,scheduler,sampler,
                                          ds.identity,config)
            if restored["stage"] != stage:
                raise ValueError("checkpoint stage mismatch")
            step = restored["step"]
            base_seconds = max(base_seconds,restored["seconds"])
            deadline = min(deadline,started+86400-base_seconds)
            del restored
        # Reserve an entire next recovery checkpoint before starting work.
        recovery_bytes = count*4*4+32*2**20
        guard.check(reserve=recovery_bytes)
        def save():
            nonlocal checkpoint
            guard.check(reserve=recovery_bytes)
            checkpoint = save_checkpoint(root,model,ema,optimizer,scheduler,sampler,
                step=step,seconds=seconds(),identity=ds.identity,stage=stage,config=config,
                stage_elapsed=stage_prior+time.perf_counter()-started)
            status(recovery_checkpoint=str(checkpoint))
        save()
        windows = None
        if stage == "pilot":
            # Deterministic, predeclared positions across the whole valid index.
            ids = np.linspace(0,len(ds)-1,12,dtype=int).tolist()
            windows = [window_batch(ds,i) for i in ids]
            stage_info["window_indices"] = ids
            if "initial_loss" not in stage_info:
                stage_info["initial_loss"] = fixed_loss(model,windows,deadline)
            before = stage_info.setdefault("initial_parameter_hashes",parameter_marks(model))
            status(state="pilot_running")
        else:
            status(state="training")
        # Two actual optimizer executions are reserved for the resume probe.
        limit = 498 if stage=="pilot" else config["max_steps"]
        reason = "step_limit"
        while step < limit:
            if stop["requested"] or time.perf_counter() >= deadline:
                reason = "user_stop" if stop["requested"] else "time_limit"
                break
            guard.check(reserve=recovery_bytes)
            update_started=time.perf_counter()
            metrics=optimizer_update(model,ema,optimizer,scheduler,sampler,ds,config,
                                     step=step,windows=windows)
            step += 1
            row = dict(event="training_progress",policy_type="diffusion",stage=stage,
                       step=step,**metrics,lr=scheduler.get_last_lr()[0],budget_seconds=seconds(),
                       update_seconds=time.perf_counter()-update_started,
                       batch_size=config["batch_size"],accumulation=config["accumulation"],
                       effective_batch=config["batch_size"]*config["accumulation"],
                       sampled_windows=sampler.cursor,device=str(device),**memory_peaks(device))
            with (out/"curves.jsonl").open("a") as log:
                log.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            status()
            if step%config["checkpoint_every"] == 0:
                save()
            if stage == "pilot" and (step%25==0 or step==limit):
                final = fixed_loss(model,windows,deadline)
                marks = parameter_marks(model)
                key = config.get("backbone","unet")
                changed = dict(vision=all(marks[k] != before[k] for k in marks if k.startswith("vision:")),
                               **{key:marks[key]!=before[key]})
                stage_info.update(final_loss=final,parameter_updates=changed,
                                  loss_reduction=1-final/stage_info["initial_loss"])
                status()
                if pilot_passes(stage_info["initial_loss"],final,changed):
                    reason = "pilot_passed"
                    break
        save()
        if stage=="pilot":
            if reason=="pilot_passed":
                def probe_update():
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("pilot30-minute budget exhausted before resume probe")
                    optimizer_update(model,ema,optimizer,scheduler,sampler,ds,config,
                                     step=step,windows=windows)
                stage_info[f"{device.type}_resume_equivalence"] = verify_continuation(
                    checkpoint,model,ema,optimizer,scheduler,sampler,ds.identity,config,
                    probe_update,lambda:ema.averaged_model.predict(to_device(windows[0],device)["obs"]))
                stage_info["executed_optimizer_updates"] = step+2
                if time.perf_counter() >= deadline:
                    raise TimeoutError("pilot30-minute budget exhausted during resume probe")
                save()
            stage_info["passed"] = reason=="pilot_passed"
            status(state=reason if stage_info["passed"] else "pilot_stopped",
                   stop_reason=reason)
            return stage_info["passed"]
        # Latest EMA only, never selected by any rollout metric.
        export = out/"final_ema.pt"
        guard.check(reserve=count*4+32*2**20)
        temporary = export.with_suffix(".pt.tmp")
        torch.save(dict(format_version=1,policy_type="diffusion",stage="full",
                        step=step,seconds=seconds(),identity=ds.identity,config=config,
                        normalization=ds.normalization,
                        ema={k:v.detach().cpu() for k,v in ema.averaged_model.state_dict().items()}),
                   temporary)
        temporary.replace(export)
        status(state="training_complete" if reason!="user_stop" else "stopped",
               stop_reason=reason,final_ema=str(export))
        return True
    except BaseException as exc:
        # Published recovery points are retained, not overwritten with a
        # possibly partially mutated optimizer step.
        stage_info["passed"] = False if stage=="pilot" else None
        status(state="failed",error=f"{type(exc).__name__}: {exc}",
               recovery_checkpoint=str(checkpoint) if checkpoint else None)
        raise
    finally:
        del model,ema,optimizer,scheduler
        empty_cache(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--out")
    parser.add_argument("--training-config",help="frozen launch JSON: dataset, out and full training settings")
    parser.add_argument("--device",choices=("mps","cuda:0","cuda"))
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--accumulation",type=int)
    parser.add_argument("--backbone",choices=("unet","transformer"))
    parser.add_argument("--profile",choices=(LEGACY,NATIVE))
    parser.add_argument("--steps",type=int)
    parser.add_argument("--resume")
    parser.add_argument("--pilot-only",action="store_true")
    args = parser.parse_args(argv)
    from .config import read_launch_config
    launch=read_launch_config(args.training_config) if args.training_config else {}
    frozen=launch.get("training",{})
    config = effective_config(args,frozen)
    for key,default in (("dataset",DEFAULT_DATASET),("out",DEFAULT_OUT)):
        value=getattr(args,key)
        if value and launch.get(key) and Path(value).resolve()!=Path(launch[key]).resolve():
            raise ValueError(f"launch {key} mismatch")
        setattr(args,key,value or launch.get(key,default))
    task=load_config(args.config)
    validate_task(task,config.get("observation_profile"))
    out = Path(args.out).resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(workspace/"runs"):
        raise ValueError("DP output must be inside this independent worktree's runs directory")
    out.mkdir(parents=True,exist_ok=True)
    with (out/".training.lock").open("a+") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resolve_device(config["device"])
        status_path = out/"status.json"
        book = json.loads(status_path.read_text()) if status_path.exists() else {}
        if book and not args.resume:
            raise RuntimeError("existing run must be explicitly resumed; no budget reset")
        stage = "pilot"
        if args.resume:
            payload = load_payload(args.resume)
            stage = validate_resume(book,payload,args.pilot_only)
            del payload
        spec=profile_spec(config.get("observation_profile"))
        ds = DiffusionDataset(args.dataset,profile=config.get("observation_profile"))
        from collections import Counter
        counts = Counter(int(r["target_count"]) for r in ds.records)
        if (len(ds.records)!=300 or counts != {n:50 for n in range(1,7)}
                or sum(len(d["t"]) for d in ds.numeric.values()) != spec["frames"]
                or len({r["layout_fingerprint"] for r in ds.records})!=300):
            raise ValueError("expected immutable 300 trajectories /profile frame count /50 per N")
        if book and (book.get("identity")!=ds.identity or book.get("config")!=config):
            raise ValueError("run config or dataset identity mismatch")
        if ds.identity != spec["identity"]:
            raise ValueError("expected immutable original demonstration identity")
        if not book:
            benchmark_path=out/"benchmark.json"
            benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
            book["budget_seconds"]=initial_budget(benchmark,ds.identity)
            if benchmark:
                validate_benchmark_config(benchmark,config)
                selected=benchmark["selected"]
                if any(config[k]!=selected[k] for k in ("batch_size","accumulation")):
                    raise ValueError("training batch differs from measured selected candidate")
                book["benchmark"]=benchmark
        book.update(identity=ds.identity,config=config,validation=None)
        atomic_json(out/"data_audit.json",ds.audit())
        atomic_json(out/"training_config.json",config)
        stop = {"requested":False}
        def request_stop(_s,_f):
            stop["requested"] = True
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        if stage == "full":
            if not book.get("pilot",{}).get("passed"):
                raise RuntimeError("full training requires recorded successful pilot")
        elif pilot_needed(stage,book):
            passed = train_stage(ds,out,config,book,stage="pilot",resume=args.resume,stop=stop)
            if not passed or args.pilot_only or stop["requested"]:
                return 0 if passed else 2
        if args.pilot_only:
            return 0
        # Fresh initialization inside train_stage; pilot weights are never promoted.
        train_stage(ds,out,config,book,stage="full",
                    resume=args.resume if stage=="full" else None,stop=stop)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(dict(event="training_error",error=f"{type(exc).__name__}: {exc}")),
              flush=True)
        raise
