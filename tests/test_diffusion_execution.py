"""Exercise the shared physical loop; doubles replace only MuJoCo/model compute."""
from types import SimpleNamespace
import numpy as np
import pytest
from sim.config import load_config
from sim.diffusion.runtime import TwoFrameHistory
from sim.diffusion.dataset import VIEWS


def exercise(monkeypatch, fail=False):
    from sim.act import evaluate
    from sim.diffusion import runtime
    calls = []
    class Env:
        instance = None
        def __init__(self,cfg,seed):
            self.cfg=cfg; self.time=0.; self.layout=[None]*6
            self.ee=SimpleNamespace(tcp_yaw=lambda:0.)
            self.pos=np.array([.42,0.,.1]); self.steps=0; self.closed=False
            Env.instance=self
        def reset(self,seed): pass
        def tcp(self): return self.pos.copy()
        def normal_force(self): return 0.
        def collected_mask(self): return np.zeros(6,dtype=bool)
        def partial_collection_overlap_mask(self): return np.zeros(6,dtype=bool)
        def step_control(self,c):
            self.steps+=1; self.time=self.steps/100.
            self.pos[:]=[c.x,c.y,c.z]
        def close(self): self.closed=True
    class Builder:
        def __init__(self,*a,**kw): pass
        def advance_state_history(self,*a,**kw): pass
        def observe(self,env,contact_latched=False):
            return {"t":env.time,"observation.state":np.zeros(36,dtype="f4"),
                    "observation.environment_state":np.array([6],dtype="f4"),
                    **{f"observation.images.{v}":np.zeros((3,320,320),dtype="f4") for v in VIEWS}}
    class Runtime:
        def __init__(self,*a,**kw):
            self.history=TwoFrameHistory(); self.latencies=[]
        def predict(self):
            calls.append((Env.instance.time,[r[0] for r in self.history.frames]))
            if fail: raise RuntimeError("diffusion test failure")
            self.latencies.append(.01)
            actions=np.zeros((25,4),dtype="f4")
            actions[:,0]=np.arange(1,26)*.0001
            return actions
    monkeypatch.setattr(runtime,"DiffusionRuntime",Runtime)
    monkeypatch.setattr(evaluate,"SweepEnv",Env)
    monkeypatch.setattr(evaluate,"ACTObservationBuilder",Builder)
    monkeypatch.setattr(evaluate,"INFERENCE_MAX_FRAMES",7)
    cfg=load_config(overrides=["task.target_count=6","sim.real_time=false"])
    result=evaluate.run_act_episode(cfg,policy_type="diffusion",model_path="fake.pt")
    return result,calls,Env.instance


def test_shared_dp_loop_queries_at_zero_then_200ms_with_40ms_history(monkeypatch):
    result,calls,env=exercise(monkeypatch)
    assert [c[0] for c in calls] == pytest.approx([0,.2])
    assert calls[1][1] == pytest.approx([.16,.2])
    assert result.actions[0,0] == pytest.approx(.4201,abs=1e-6)
    assert result.actions[4,0] == pytest.approx(.4215,abs=1e-6)
    assert result.actions[5,0] == pytest.approx(.4216,abs=1e-6)
    assert result.observations == []
    assert env.closed


def test_shared_dp_loop_propagates_prediction_errors(monkeypatch):
    with pytest.raises(RuntimeError,match="diffusion test failure"):
        exercise(monkeypatch,fail=True)
