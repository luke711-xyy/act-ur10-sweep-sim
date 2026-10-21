"""Bounded local MPS training. A failed pilot never launches full training."""
from __future__ import annotations
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
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
from .policy import DiffusionPolicy
from .training import (WeightedCursor, ResourceGuard, atomic_json,
                       save_checkpoint, restore_checkpoint, load_payload,
                       verify_continuation)
from .vendor.model.diffusion.ema_model import EMAModel
from .config import DEFAULT_DATASET, DEFAULT_OUT


def training_config(steps=None):
    steps = 100000 if steps is None else int(steps)
    if not 1 <= steps <= 100000:
        raise ValueError("steps must be between 1 and 100000")
    return dict(seed=42,max_steps=steps,max_seconds=86400,batch_size=1,
                accumulation=2,workers=0,device="mps",dtype="float32",
                lr=1e-4,betas=[.95,.999],weight_decay=1e-6,clip=1.,
                warmup=500,checkpoint_every=500,keep_checkpoints=2,
                approach_fraction=.35,obs_steps=2,horizon=28,output_steps=25,
                execute_steps=5,image_size=[320,320],unet_dims=[256,512,1024],
                diffusion_train_steps=100,ddim_steps=8,eta=0,ema_power=.75,
                upstream="5ba07ac6661db573af695b419a7947ecb704690f")


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


def to_device(batch):
    if isinstance(batch,dict):
        return {k:to_device(v) for k,v in batch.items()}
    return batch.to("mps") if isinstance(batch,torch.Tensor) else batch


def window_batch(ds, index):
    item = ds[index]
    return default_collate([{k:item[k] for k in ("obs","action","valid_mask")}])


def parameter_marks(model):
    result = {}
    modules = {f"vision:{v}":model.obs_encoder.key_model_map[v]
               for v in model.obs_encoder.rgb_keys}
    modules["unet"] = model.unet
    for key,module in modules.items():
        p = next(module.parameters()).detach().cpu().numpy()
        result[key] = hashlib.sha256(p.tobytes()).hexdigest()
    return result


@torch.no_grad()
def fixed_loss(model, windows, deadline):
    model.eval()
    losses = []
    for j, batch in enumerate(windows):
        if time.perf_counter() >= deadline:
            raise TimeoutError("pilot 30-minute limit reached during fixed-noise evaluation")
        generator = torch.Generator(device="cpu").manual_seed(424200+j)
        noise = torch.randn((1,28,4),generator=generator).to("mps")
        timestep = torch.tensor([j*8%100],dtype=torch.long,device="mps")
        losses.append(float(model.loss(to_device(batch),noise,timestep).item()))
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
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    torch.mps.manual_seed(42)
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
        model = DiffusionPolicy(ds.normalization).to("mps",dtype=torch.float32)
        count = sum(p.numel() for p in model.parameters())
        print(json.dumps(dict(event="architecture",parameters=count,
                              vision_parameters=sum(p.numel() for p in model.obs_encoder.parameters()),
                              device="mps",dtype="float32")),flush=True)
        ema = EMAModel(copy.deepcopy(model),power=.75)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,
            betas=(.95,.999),weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda s: min((s+1)/500,1.) * .5*(1+math.cos(
                math.pi*max(0,s-500)/max(1,config["max_steps"]-500))))
        sampler = WeightedCursor(ds.phase_sampling_weights(.35),42)
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
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for micro in range(2):
                if stage == "pilot":
                    batch = windows[(step*2+micro)%12]
                    sampler.next()  # cursor/RNG remain fully recoverable
                else:
                    batch = window_batch(ds,sampler.next())
                batch = to_device(batch)
                loss = model.loss(batch)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("nonfinite diffusion training loss")
                (loss/2).backward()
                loss_value += float(loss.detach().item())/2
                del batch,loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); ema.step(model)
            torch.mps.synchronize()
            step += 1
            row = dict(event="training_progress",policy_type="diffusion",stage=stage,
                       step=step,loss=loss_value,grad_norm=float(norm.item()),
                       lr=scheduler.get_last_lr()[0],budget_seconds=seconds())
            with (out/"curves.jsonl").open("a") as log:
                log.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            status()
            if step%500 == 0:
                save()
            if stage == "pilot" and (step%25==0 or step==limit):
                final = fixed_loss(model,windows,deadline)
                marks = parameter_marks(model)
                changed = dict(vision=all(marks[k] != before[k] for k in marks if k.startswith("vision:")),
                               unet=marks["unet"] != before["unet"])
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
                    optimizer.zero_grad(set_to_none=True)
                    for micro in range(2):
                        sampler.next()
                        batch = to_device(windows[(step*2+micro)%12])
                        loss = model.loss(batch)
                        if not torch.isfinite(loss).item():
                            raise RuntimeError("nonfinite MPS resume probe loss")
                        (loss/2).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step(); scheduler.step(); ema.step(model)
                    torch.mps.synchronize()
                stage_info["mps_resume_equivalence"] = verify_continuation(
                    checkpoint,model,ema,optimizer,scheduler,sampler,ds.identity,config,
                    probe_update,lambda:ema.averaged_model.predict(to_device(windows[0])["obs"]))
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
        torch.mps.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--dataset",default=DEFAULT_DATASET)
    parser.add_argument("--out",default=DEFAULT_OUT)
    parser.add_argument("--steps",type=int)
    parser.add_argument("--resume")
    parser.add_argument("--pilot-only",action="store_true")
    args = parser.parse_args(argv)
    config = training_config(args.steps)
    load_config(args.config)  # fail fast on invalid task config
    out = Path(args.out).resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(workspace/"runs"):
        raise ValueError("DP output must be inside this independent worktree's runs directory")
    out.mkdir(parents=True,exist_ok=True)
    with (out/".training.lock").open("a+") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; CPU fallback is forbidden")
        status_path = out/"status.json"
        book = json.loads(status_path.read_text()) if status_path.exists() else {}
        if book and not args.resume:
            raise RuntimeError("existing run must be explicitly resumed; no budget reset")
        stage = "pilot"
        if args.resume:
            payload = load_payload(args.resume)
            stage = validate_resume(book,payload,args.pilot_only)
            del payload
        ds = DiffusionDataset(args.dataset)
        from collections import Counter
        counts = Counter(int(r["target_count"]) for r in ds.records)
        if (len(ds.records)!=300 or counts != {n:50 for n in range(1,7)}
                or sum(len(d["t"]) for d in ds.numeric.values()) != 203121
                or len({r["layout_fingerprint"] for r in ds.records})!=300):
            raise ValueError("expected original 300 trajectories /203121 frames /50 per N")
        if book and (book.get("identity")!=ds.identity or book.get("config")!=config):
            raise ValueError("run config or dataset identity mismatch")
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
        else:
            passed = train_stage(ds,out,config,book,stage="pilot",resume=args.resume,stop=stop)
            if not passed or args.pilot_only or stop["requested"]:
                return 0 if passed else 2
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
