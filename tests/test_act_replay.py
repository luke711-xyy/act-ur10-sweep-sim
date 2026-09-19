from types import SimpleNamespace

import numpy as np

from sim.act.replay import save_inference_replay
from sim.config import load_config
from sim.web.workbench import WorkbenchState


def test_inference_command_trace_is_saved_as_a_three_camera_episode(tmp_path):
    cfg = load_config(overrides=[
        "sim.real_time=false", "act.image_size=[32,32]",
        "task.target_count=2",
    ])
    result = SimpleNamespace(
        trace=[{
            "t": 0.41,
            "command": np.array([0.42, 0.0, 0.0, 0.0], dtype=np.float32),
            "phase": "approach",
            "contact": False,
            "policy_reference": np.array([0.42, 0.0, 0.03, 0.0], dtype=np.float32),
            "z_owner": "policy",
        }],
        success=False,
        failure_reason="test replay",
        collected=0,
        total=6,
        target_count=2,
        target_collected=0,
        unexpected_collected=0,
        scheduler_queries=1,
        scheduler_timeouts=0,
        peak_force=0.0,
    )

    episode_id = save_inference_replay(
        cfg, seed=17, result=result, root=tmp_path, model="checkpoint"
    )
    assert episode_id is not None
    assert episode_id.startswith("inference_")

    state = WorkbenchState(cfg, dataset_root=tmp_path / "missing",
                           preview_root=tmp_path)
    metadata = state.episode_metadata(episode_id)
    assert metadata["inference_replay"] is True
    assert metadata["episode_kind"] == "inference"
    assert metadata["schema_version"] == 4
    assert metadata["target_count"] == 2
    assert "target_indices" not in metadata
    assert "unexpected_collected" not in metadata
    assert metadata["length"] == 1
    frame = state.load_episode_frame(episode_id, 0)
    assert frame["overhead"].shape == (32, 32, 3)
    assert frame["wrist"].shape == (32, 32, 3)
    assert frame["inspection"].shape == (32, 32, 3)
    signals = state.load_episode_signals(episode_id)
    assert len(signals["fz"]) == 1
    assert len(signals["policy_z"]) == 1
    assert len(signals["applied_z"]) == 1
    assert signals["z_owner"] == ["policy"]
