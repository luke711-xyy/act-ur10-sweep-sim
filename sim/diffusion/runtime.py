"""Two-frame visual cache and synchronous receding-horizon DP execution."""
from collections import deque
from pathlib import Path
import json
import time
import numpy as np
from .dataset import VIEWS
from .profiles import validate_task,validate_profile_config
from .checkpoints import resolve_checkpoint


def inference_resource_guard(workspace, replay_root, video_root):
    from .training import ResourceGuard
    workspace = Path(workspace).resolve()
    for path in (replay_root,video_root):
        if not Path(path).resolve().is_relative_to(workspace/"runs"):
            raise ValueError("DP inference artifacts must stay in this worktree's runs directory")
    guard = ResourceGuard(workspace)
    guard.check(reserve=2*2**30)
    return lambda:guard.check(reserve=128*2**20)


class TwoFrameHistory:
    def __init__(self, views=VIEWS):
        self.views=tuple(views)
        self.frames = deque(maxlen=2)
    def append(self, observation):
        t = float(observation["t"])
        if self.frames and not np.isclose(t-self.frames[-1][0],.04,atol=1e-5,rtol=0):
            raise ValueError("DP online images must be consecutive 40ms frames")
        frame = {v:np.asarray(observation[f"observation.images.{v}"],dtype="f4").copy()
                 for v in self.views}
        frame.update(state=np.asarray(observation["observation.state"],dtype="f4").copy(),
                     environment_state=np.asarray(observation["observation.environment_state"],dtype="f4").copy())
        self.frames.append((t,frame))
    def batch(self):
        if not self.frames:
            raise RuntimeError("no causal DP observations")
        frames = list(self.frames)
        if len(frames)==1:
            frames = frames*2
        return {k:np.stack([f[1][k] for f in frames])[None] for k in frames[-1][1]}


class RollingChunk:
    timeouts = 0
    late_results = 0
    def __init__(self):
        self.reset()
    def reset(self, now=0):
        self.next_query = float(now)
        self.actions = None
        self.issued_at = float(now)
    def query_due(self, now):
        return float(now)+1e-7 >= self.next_query
    def accept(self, issued_at, actions, now):
        values = np.asarray(actions,dtype="f4")
        if values.shape != (25,4) or not np.isfinite(values).all():
            raise ValueError("DP requires finite25x4 actions")
        if abs(float(now)-float(issued_at))>1e-7:
            raise RuntimeError("simulation advanced during paused DP inference")
        self.issued_at = float(issued_at)
        self.actions = values[:5].copy()
        self.next_query = self.issued_at+.2
        return True
    def action_for(self, now):
        i = int(np.floor((float(now)-self.issued_at)/.04+1e-5))
        if self.actions is None or not 0 <= i < 5:
            raise RuntimeError("DP chunk exhausted: query again; holding is forbidden")
        return self.actions[i].copy()


class DiffusionRuntime:
    def __init__(self, checkpoint, cfg, seed=42):
        import torch
        from .device import inference_device, seed_all
        from .policy import build_policy
        from .training import load_payload
        self.device = inference_device()
        self.checkpoint = resolve_checkpoint(checkpoint)
        state = load_payload(self.checkpoint)
        config=state.get("config",{})
        spec=validate_profile_config(config)
        validate_task(cfg,config.get("observation_profile"))
        self.model = build_policy(state["normalization"],state.get("config",{}))
        self.model.load_state_dict(state["ema"],strict=True)
        self.model.to(self.device,dtype=torch.float32).eval()
        print(json.dumps(dict(event="diffusion_checkpoint",checkpoint=self.checkpoint,
                              backbone=state.get("config",{}).get("backbone","unet"),
                              step=state.get("step"),device=str(self.device))),flush=True)
        del state
        seed_all(seed,self.device)
        self.history = TwoFrameHistory(spec["views"])
        self.latencies = []
    def predict(self):
        import torch
        inputs = {k:torch.from_numpy(v).to(self.device) for k,v in self.history.batch().items()}
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.predict(inputs).cpu().numpy()[0]
        self.latencies.append(time.perf_counter()-started)
        print(json.dumps(dict(event="diffusion_query",latency_seconds=self.latencies[-1],
                              device=str(self.device),real_time_200ms_claim=False)),flush=True)
        return result
