"""Short, disposable CUDA batch benchmark; no weights are promoted to training."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import resource
import time
import torch
from .train import training_config, optimizer_update, make_optimizer, make_scheduler
from .dataset import DiffusionDataset
from .profiles import profile_spec,LEGACY,NATIVE
from .policy import build_policy, expected_parameters
from .device import resolve_device, seed_all, synchronize, memory_peaks, empty_cache
from .training import WeightedCursor, ResourceGuard, atomic_json
from .vendor.model.diffusion.ema_model import EMAModel


def choose_candidate(rows,total_memory):
    safe=[r for r in rows if r["state"]=="passed" and r["peak_reserved_bytes"] <= .8*total_memory]
    if not safe: raise RuntimeError("no measured candidate has 20% VRAM headroom")
    fastest=max(safe,key=lambda r:r["windows_per_second"])
    equivalent=[r for r in safe if r["batch_size"]*r["accumulation"]==2]
    if equivalent:
        baseline=max(equivalent,key=lambda r:r["windows_per_second"])
        if fastest["windows_per_second"] <= 1.15*baseline["windows_per_second"]:
            return baseline
    return fastest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--backbone",choices=("unet","transformer"),default="unet")
    parser.add_argument("--profile",choices=(LEGACY,NATIVE))
    args=parser.parse_args(argv)
    device=resolve_device("cuda:0")
    ds=DiffusionDataset(args.dataset,profile=args.profile)
    if ds.identity!=profile_spec(args.profile)["identity"]:
        raise ValueError("original dataset identity mismatch")
    out=Path(args.out).resolve()
    workspace=Path(__file__).resolve().parents[2]
    if not out.is_relative_to(workspace/"runs"):raise ValueError("benchmark output must be in this worktree")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"benchmark.json").exists() or (out/"status.json").exists():
        raise FileExistsError("benchmark cannot overwrite or reset a previous budget")
    guard=ResourceGuard(workspace);guard.check()
    start=time.perf_counter()
    report=dict(identity=ds.identity,training_seconds=0.,state="running",candidates=[],
                weights_promoted=False,device=str(device),max_seconds=600)
    base_config=training_config(device="cuda:0",batch_size=1,accumulation=1,backbone=args.backbone,profile=args.profile)
    report.update(backbone=args.backbone,config=base_config)
    model=ema=optimizer=scheduler=None
    try:
        seed_all(42,device);torch.set_num_threads(1)
        model=build_policy(ds.normalization,base_config).to(device,dtype=torch.float32)
        count=sum(p.numel() for p in model.parameters())
        if count!=expected_parameters(base_config):raise RuntimeError("architecture parameter count changed")
        report["parameters"]=count
        ema=EMAModel(copy.deepcopy(model),power=.75)
        ema.averaged_model.eval()
        optimizer=make_optimizer(model,base_config)
        scheduler=make_scheduler(optimizer,base_config)
        sampler=WeightedCursor(ds.phase_sampling_weights(.35),42)
        total=torch.cuda.get_device_properties(device).total_memory
        report["total_vram_bytes"]=total
        # First prove the actual Mac batch/accumulation on CUDA, then ascend.
        for size,accumulation in [(1,2),(1,1),*( (n,1) for n in (2,4,8,16,32,64))]:
            if time.perf_counter()-start > 500:break  # leave a whole update's headroom under 600s
            cfg=dict(base_config,batch_size=size,accumulation=accumulation)
            row=dict(batch_size=size,accumulation=accumulation,effective_batch=size*accumulation)
            try:
                torch.cuda.reset_peak_memory_stats(device)
                optimizer_update(model,ema,optimizer,scheduler,sampler,ds,cfg,step=0)
                synchronize(device)
                measured=time.perf_counter(); metrics=[]
                for step in range(3):
                    if time.perf_counter()-start>580:raise TimeoutError("benchmark 10-minute cap")
                    metrics.append(optimizer_update(model,ema,optimizer,scheduler,sampler,ds,cfg,step=step))
                elapsed=time.perf_counter()-measured
                row.update(state="passed",optimizer_updates=3,seconds_per_update=elapsed/3,
                    windows_per_second=3*size*accumulation/elapsed,
                    data_seconds_per_update=sum(m["data_seconds"] for m in metrics)/3,
                    peak_allocated_bytes=memory_peaks(device)["cuda_allocated_peak_bytes"],
                    peak_reserved_bytes=memory_peaks(device)["cuda_reserved_peak_bytes"],
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                report["candidates"].append(row)
                print(json.dumps(dict(event="batch_benchmark",**row)),flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                row.update(state="oom",error=str(exc));report["candidates"].append(row)
                print(json.dumps(dict(event="batch_benchmark",**row)),flush=True)
                break
        report["selected"]=choose_candidate(report["candidates"],total)
        report["state"]="passed"
    except BaseException as exc:
        report.update(state="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        del model,ema,optimizer,scheduler
        empty_cache(device)
        report["training_seconds"]=time.perf_counter()-start
        atomic_json(out/"benchmark.json",report)
        print(json.dumps(dict(event="benchmark_finished",**report)),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
