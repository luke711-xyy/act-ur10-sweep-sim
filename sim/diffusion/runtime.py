"""Two-frame visual cache and synchronous receding-horizon DP execution."""
from collections import deque
from pathlib import Path
import json
import time
import numpy as np
from .dataset import VIEWS


class TwoFrameHistory:
    def __init__(self):
        self.frames = deque(maxlen=2)
    def append(self, observation):
        t = float(observation["t"])
        if self.frames and not np.isclose(t-self.frames[-1][0],.04,atol=1e-5,rtol=0):
            raise ValueError("DP online images must be consecutive 40ms frames")
        frame = {v:np.asarray(observation[f"observation.images.{v}"],dtype="f4").copy()
                 for v in VIEWS}
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


def resolve_checkpoint(path):
    if not path:
        raise ValueError("DP requires an explicit trained checkpoint")
    p = Path(path).resolve()
    if p.is_dir():
        if (p/"final_ema.pt").is_file():
            p = p/"final_ema.pt"
        elif (p/"latest.json").is_file():
            name = json.loads((p/"latest.json").read_text())["checkpoint"]
            if Path(name).name != name:
                raise ValueError("invalid DP checkpoint pointer")
            p = p/name
    if p.suffix != ".pt" or not p.is_file():
        raise ValueError("diffusion checkpoint required; ACT checkpoints are incompatible")
    return str(p)


class DiffusionRuntime:
    def __init__(self, checkpoint, cfg, seed=42):
        import torch
        from .policy import DiffusionPolicy
        from .training import load_payload
        if (tuple(cfg.act.camera_views)!=VIEWS or tuple(cfg.act.image_size)!=(320,320)
                or int(cfg.act.action_hz)!=25 or int(cfg.act.query_hz)!=5
                or int(cfg.act.execute_steps)!=5 or int(cfg.act.action_dim)!=4):
            raise ValueError("DP requires approved five-view320/25Hz/5Hz/4D contract")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no random/CPU fallback")
        state = load_payload(resolve_checkpoint(checkpoint))
        self.model = DiffusionPolicy(state["normalization"])
        self.model.load_state_dict(state["ema"],strict=True)
        self.model.to("mps",dtype=torch.float32).eval()
        del state
        torch.manual_seed(seed); torch.mps.manual_seed(seed)
        self.history = TwoFrameHistory()
        self.latencies = []
    def predict(self):
        import torch
        inputs = {k:torch.from_numpy(v).to("mps") for k,v in self.history.batch().items()}
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.predict(inputs).cpu().numpy()[0]
        self.latencies.append(time.perf_counter()-started)
        print(json.dumps(dict(event="diffusion_query",latency_seconds=self.latencies[-1],
                              real_time_200ms_claim=False)),flush=True)
        return result
