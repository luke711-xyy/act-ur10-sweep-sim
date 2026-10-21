from types import SimpleNamespace
import json
import numpy as np
import pytest
from sim.config import load_config


def result():
    return SimpleNamespace(policy_type="diffusion",inference_latencies=[.12,.14],
        wall_seconds=.3,target_count=3,total=6,collected=0,success=False,
        failure_reason="test",trace=[dict(t=.41,command=np.array([.42,0,.1,0]),
                                        phase="approach",contact=False)])


def test_replay_recovery_keeps_diffusion_identity_and_latency(tmp_path):
    from sim.act.replay import persist_replay_input,load_replay_input
    cfg=load_config()
    path=persist_replay_input(cfg,7,result(),tmp_path,model="x.pt")
    recovered=load_replay_input(path,cfg,seed=7,model="x.pt")
    assert recovered.policy_type == "diffusion"
    assert recovered.inference_latencies == [.12,.14]


def test_replay_writer_propagates_budget_failure_and_never_labels_dp_as_act(tmp_path):
    from sim.act.replay import save_inference_replay
    cfg=load_config(overrides=["act.image_size=[16,16]","sim.real_time=false"])
    def no_space():
        raise RuntimeError("disk floor")
    with pytest.raises(RuntimeError,match="disk floor"):
        save_inference_replay(cfg,7,result(),tmp_path,resource_check=no_space)
    assert not (tmp_path/"manifest.jsonl").exists()
    save_inference_replay(cfg,7,result(),tmp_path)
    row=json.loads((tmp_path/"manifest.jsonl").read_text())
    assert row["policy_type"] == "diffusion"
    assert row["planner"] == "diffusion_inference"
    assert row["inference_latencies"] == [.12,.14]
