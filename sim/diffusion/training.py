"""Deterministic sampling, atomic recovery and non-destructive storage guards."""
from __future__ import annotations
import json
import os
import random
import shutil
import stat
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)


class WeightedCursor:
    def __init__(self, weights, seed):
        self.weights = torch.as_tensor(weights,dtype=torch.float64,device="cpu")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.cursor = 0
    def next(self):
        index = torch.multinomial(self.weights,1,replacement=True,
                                  generator=self.generator).item()
        self.cursor += 1
        return index
    def state_dict(self):
        return dict(rng=self.generator.get_state(),cursor=self.cursor)
    def load_state_dict(self, state):
        self.generator.set_state(state["rng"])
        self.cursor = int(state["cursor"])


class ResourceGuard:
    def __init__(self, root, max_bytes=20*2**30,min_free_bytes=25*2**30):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
    def used(self):
        used = 0
        for path in self.root.rglob("*"):
            try:
                info = path.stat()
            except FileNotFoundError:
                # Workbench records/checkpoints may be atomically replaced
                # between directory discovery and this single measurement.
                continue
            if stat.S_ISREG(info.st_mode):
                used += info.st_size
        return used
    def check(self,reserve=0):
        used = self.used()
        free = shutil.disk_usage(self.root).free
        if used+reserve > self.max_bytes:
            raise RuntimeError(f"artifact budget: {used}+{reserve}>{self.max_bytes}")
        if free-reserve < self.min_free_bytes:
            raise RuntimeError(f"disk floor: {free}-{reserve}<{self.min_free_bytes}")
        return dict(used=used,free=free,reserved=reserve)


def _cpu_tree(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if isinstance(value,dict):
        return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def rng_state():
    state = dict(python=random.getstate(),numpy=np.random.get_state(),
                 torch=torch.get_rng_state())
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG checkpoint requires compatible CUDA runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(root,model,ema,optimizer,scheduler,sampler,*,step,seconds,
                    identity,stage,config,stage_elapsed=0):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    checkpoint = root/f"step_{step:06d}.pt"
    tmp = root/(checkpoint.name+".tmp")
    payload = dict(format_version=1,policy_type="diffusion",stage=stage,step=step,
                   stage_elapsed=float(stage_elapsed),
                   seconds=seconds,identity=identity,config=config,
                   normalization=getattr(model,"normalization",None),
                   model=model.state_dict(),ema=ema.averaged_model.state_dict(),
                   ema_step=ema.optimization_step,
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   sampler=sampler.state_dict(),rng=rng_state())
    try:
        torch.save(_cpu_tree(payload),tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        tmp.replace(checkpoint)
    finally:
        tmp.unlink(missing_ok=True)
    atomic_json(root/"latest.json",dict(checkpoint=checkpoint.name,step=step,stage=stage))
    # Only checkpoints owned by this stage/run are ever pruned.
    for old in sorted(root.glob("step_*.pt"))[:-2]:
        old.unlink()
    return checkpoint


def load_payload(path):
    path = Path(path)
    if path.is_dir():
        pointer = json.loads((path/"latest.json").read_text())
        path = path/pointer["checkpoint"]
    if path.suffix != ".pt":
        raise ValueError("expected a diffusion .pt checkpoint, not ACT")
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("policy_type") != "diffusion" or payload.get("format_version") != 1:
        raise ValueError("checkpoint policy_type must be diffusion")
    return payload


def restore_checkpoint(path,model,ema,optimizer,scheduler,sampler,identity,config):
    state = load_payload(path)
    if state["identity"] != identity:
        raise ValueError("dataset identity mismatch")
    if state["config"] != config:
        raise ValueError("training config mismatch")
    if state.get("normalization") != getattr(model,"normalization",None):
        raise ValueError("normalization mismatch")
    model.load_state_dict(state["model"],strict=True)
    ema.averaged_model.load_state_dict(state["ema"],strict=True)
    ema.optimization_step = state["ema_step"]
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])
    set_rng_state(state["rng"])
    return state


def verify_continuation(path,model,ema,optimizer,scheduler,sampler,identity,config,
                        update,predict):
    """Replay the same real update twice; leave committed model/RNG unchanged."""
    def restore():
        restore_checkpoint(path,model,ema,optimizer,scheduler,sampler,identity,config)
    def snapshot(module):
        return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
    restore()
    first = predict().detach().cpu()
    update()
    expected = snapshot(model)
    expected_ema = snapshot(ema.averaged_model)
    expected_cursor = sampler.cursor
    restore()
    second = predict().detach().cpu()
    torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-6)
    update()
    if sampler.cursor != expected_cursor:
        raise RuntimeError("resume sampler continuation mismatch")
    for module,reference in ((model,expected),(ema.averaged_model,expected_ema)):
        for k,v in module.state_dict().items():
            torch.testing.assert_close(v.detach().cpu(),reference.pop(k),rtol=1e-5,atol=1e-6)
    error = float((first-second).abs().max().item())
    restore()
    return dict(passed=True,output_max_absolute_error=error,
                replayed_optimizer_updates=2,rtol=1e-5,atol=1e-6)
