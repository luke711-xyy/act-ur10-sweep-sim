from types import SimpleNamespace

import numpy as np
import pytest

from sim.act.replay import save_inference_replay
from sim.config import load_config
from sim.web.workbench import WorkbenchState


@pytest.mark.parametrize("image_format", ["png", "webp-lossless"])
def test_inference_command_trace_is_saved_with_five_policy_views(tmp_path, image_format):
    cfg = load_config(overrides=[
        "sim.real_time=false", "act.image_size=[32,32]",
        "task.target_count=2",
    ])
    from sim.environments.sweep_env import SweepEnv
    expected_scene = SweepEnv(cfg, seed=17)
    try:
        expected_scene.reset(seed=17)
        initial_poses = np.concatenate((expected_scene.component_positions(),
                                        expected_scene.component_quats()), axis=1).astype("<f4")
    finally:
        expected_scene.close()
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
        collected=1,
        peak_collected=1,
        final_collected=0,
        first_entry_time=0.405,
        collection_window_seconds=2.0,
        total=6,
        target_count=2,
        target_collected=0,
        unexpected_collected=0,
        scheduler_queries=1,
        scheduler_timeouts=0,
        peak_force=0.0,
    )

    episode_id = save_inference_replay(
        cfg, seed=17, result=result, root=tmp_path, model="checkpoint",
        image_format=image_format,
    )
    assert episode_id is not None
    assert episode_id.startswith("inference_")
    assert "_n2_checkpoint" in episode_id
    assert "_s17" not in episode_id
    import gzip
    import json
    with gzip.open(tmp_path / episode_id / "causal_trace_100hz.json.gz", "rt") as stream:
        causal = json.load(stream)
    assert causal[0]["command"] == pytest.approx([0.42, 0, 0, 0])
    assert causal[0]["policy_reference"] == pytest.approx([0.42, 0, 0.03, 0])

    state = WorkbenchState(cfg, dataset_root=tmp_path / "missing",
                           preview_root=tmp_path)
    metadata = state.episode_metadata(episode_id)
    assert metadata["inference_replay"] is True
    assert metadata["episode_kind"] == "inference"
    assert metadata["schema_version"] == 5
    assert metadata["target_count"] == 2
    import hashlib
    assert metadata["seed"] == 17
    assert metadata["model"] == "checkpoint"
    assert np.array_equal(np.asarray(metadata["initial_object_poses"], dtype="<f4"), initial_poses)
    assert metadata["layout_fingerprint"] == hashlib.sha256(initial_poses.tobytes()).hexdigest()
    assert metadata.get("peak_collected") == 1
    assert metadata.get("final_collected") == 0
    assert metadata.get("first_entry_time") == 0.405
    assert metadata.get("collection_window_seconds") == 2.0
    assert metadata.get("counting_mode") == "peak_simultaneous_full_entry"
    assert "target_indices" not in metadata
    assert "unexpected_collected" not in metadata
    assert metadata["length"] == 1
    frame = state.load_episode_frame(episode_id, 0)
    assert frame["overhead"].shape == (32, 32, 3)
    assert frame["wrist"].shape == (32, 32, 3)
    assert set(frame["images"]) == set(cfg.act.camera_views)
    assert "inspection" not in frame
    signals = state.load_episode_signals(episode_id)
    assert len(signals["fz"]) == 1
    assert len(signals["policy_z"]) == 1
    assert len(signals["applied_z"]) == 1
    assert signals["z_owner"] == ["policy"]
    # The original achievement must not be replaced by re-simulation counts.
    assert state.load_episode_frame_data(episode_id, 0)["fully_collected"] == 0
    listing = next(row for row in state.list_episodes() if row["episode_id"] == episode_id)
    assert listing["collected"] == listing.get("peak_collected") == 1
    assert listing.get("final_collected") == 0
    assert listing.get("collection_summary") == "最高同时入区 1/2 · 最终留区 0/6"
    from sim.act.video import export_inference_video
    video = export_inference_video(tmp_path / episode_id, tmp_path / "videos")
    assert video.name == f"{episode_id}_five_view.mp4"
    assert "_n2_checkpoint" in video.name and "_s17" not in video.name
    assert video.stat().st_size > 100
    state.close()


def test_causal_trace_failure_cannot_publish_an_episode(tmp_path, monkeypatch):
    from sim.act import dataset
    from sim.act.dataset import ActDatasetWriter
    observation = {"images": {"wrist": np.zeros((4, 4, 3), dtype=np.uint8)},
                   "state": np.zeros(36), "environment_state": np.zeros(1)}
    def broken(*args, **kwargs):
        raise OSError("causal trace disk failure")
    monkeypatch.setattr(dataset.gzip, "open", broken)
    with pytest.raises(OSError, match="causal trace"):
        ActDatasetWriter(str(tmp_path)).add_episode(
            "inference_atomic", [observation], np.zeros((1, 4)), False,
            {"episode_kind": "inference"}, causal_trace=[{"t": 0.01}])
    assert not (tmp_path / "inference_atomic").exists()
    assert not (tmp_path / "manifest.jsonl").exists()
    assert not list(tmp_path.glob(".inference_atomic.tmp-*"))


def test_recorded_physics_replay_does_not_resolve_ik_again(tmp_path):
    """A rejected IK attempt can mutate the original state; render that state."""
    import json
    import mujoco
    from sim.controllers.hybrid import Command
    from sim.environments.sweep_env import SweepEnv

    cfg = load_config(overrides=["act.image_size=[32,32]", "task.target_count=4"])
    env = SweepEnv(cfg, seed=2110081)
    try:
        env.reset(seed=2110081)
        env.step_control(Command(0.42, 0.0, float(cfg.end_effector.z_home), 0.0))
        state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        joint_position = env.ee.joint_state().copy()
        object_pose = np.concatenate((env.component_positions(), env.component_quats()), axis=1)
        row = {"t": env.time, "command": [100.0, 0.0, 0.1, 0.0],
               "phase": "approach", "contact": False, "normal_force": 7.25,
               "raw_normal_force": 8.5, "compensated_normal_force_raw": 7.75,
               "wrench": [1, 2, 3, 4, 5, 6],
               "object_collected": [False] * 6, "collected": 0,
               "object_pose": object_pose, "tcp_pose": [*env.tcp(), env.ee.tcp_yaw()],
               "replay_state": {"spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
                                "values": state}}
    finally:
        env.close()
    result = SimpleNamespace(trace=[row], success=False, target_count=4, total=6,
                             collected=0, peak_collected=0, final_collected=0,
                             collection_window_seconds=2.0, first_entry_time=None)
    episode_id = save_inference_replay(cfg, 2110081, result, tmp_path)
    record = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert record["replay_render_mode"] == "recorded_physics_state"
    with np.load(tmp_path / episode_id / "arrays.npz") as arrays:
        np.testing.assert_allclose(arrays["joint_position"][0], joint_position, atol=1e-6)
        np.testing.assert_allclose(arrays["object_pose"][0], object_pose, atol=1e-6)
        assert arrays["t"][0] == pytest.approx(row["t"])
        assert arrays["normal_force"][0] == pytest.approx(7.25)
        assert arrays["raw_normal_force"][0] == pytest.approx(8.5)
        np.testing.assert_array_equal(arrays["wrench"][0], [1, 2, 3, 4, 5, 6])


def test_replay_input_survives_renderer_failure_and_rejects_changed_identity(tmp_path, monkeypatch):
    import sim.act.replay as replay
    cfg = load_config(overrides=["act.image_size=[32,32]", "task.target_count=4"])
    result = SimpleNamespace(trace=[{"t": 0.41, "command": [0.42, 0, 0.1, 0]}],
                             success=False, failure_reason="original timeout", collected=0,
                             peak_collected=0, final_collected=0, target_count=4, total=6,
                             first_entry_time=None, collection_window_seconds=2.0)
    saved = replay.persist_replay_input(cfg, 2110081, result, tmp_path, model="/checkpoint")
    def broken(*args, **kwargs):
        raise OSError("injected renderer failure")
    monkeypatch.setattr(replay, "save_inference_replay", broken)
    with pytest.raises(OSError, match="renderer"):
        replay.recover_replay_input(saved, cfg, seed=2110081, model="/checkpoint")
    assert saved.is_file()
    recovered = replay.load_replay_input(saved, cfg, seed=2110081, model="/checkpoint")
    assert recovered.failure_reason == "original timeout"
    assert recovered.success is False and recovered.collected == 0
    assert recovered.trace[0]["command"] == [0.42, 0, 0.1, 0]
    for changed_cfg, seed, model in [(cfg, 2110082, "/checkpoint"),
                                     (cfg, 2110081, "/different"),
                                     (load_config(overrides=["task.target_count=5"]), 2110081, "/checkpoint")]:
        with pytest.raises(ValueError, match="identity"):
            replay.load_replay_input(saved, changed_cfg, seed=seed, model=model)


def test_inference_capture_renders_the_same_physical_numeric_frames(tmp_path, monkeypatch):
    """The real MuJoCo inference loop must produce usable replay snapshots."""
    from sim.act import evaluate
    import json
    pytest.importorskip("torch")
    class HoldingPolicy:
        def eval(self):
            pass
        def predict_action_chunk(self, _batch):
            raise RuntimeError("no predictions in short capture fixture")
    monkeypatch.setattr(evaluate, "build_act_policy", lambda *a, **k: (
        HoldingPolicy(), SimpleNamespace(output_features={"action": SimpleNamespace(shape=(4,))})))
    monkeypatch.setattr(evaluate, "build_act_processors", lambda *a, **k: (
        lambda x: x, lambda x: x))
    monkeypatch.setattr(evaluate, "INFERENCE_MAX_FRAMES", 2)
    cfg = load_config(overrides=["act.image_size=[32,32]", "task.target_count=4",
                                 f"act.dataset_dir={tmp_path / 'missing'}"])
    result = evaluate.run_act_episode(cfg, seed=2110081, preview=True, capture_replay=True)
    assert result.sampled_frames == 2
    assert result.observations
    assert all(not any(key.startswith("observation.images.") for key in obs)
               for obs in result.observations)
    assert len(result.trace) == 8
    assert all("replay_state" in row for row in result.trace)
    episode_id = save_inference_replay(cfg, 2110081, result, tmp_path)
    record = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert record["frame_count"] == 2
    with np.load(tmp_path / episode_id / "arrays.npz") as arrays:
        for index, original in enumerate(result.trace[3::4]):
            np.testing.assert_allclose(arrays["object_pose"][index], original["object_pose"], atol=1e-7)
            np.testing.assert_allclose(arrays["wrench"][index], original["wrench"], atol=1e-7)
            np.testing.assert_array_equal(arrays["state"][index], original["replay_observation"]["observation.state"])


def test_rejected_commands_still_capture_and_render_the_actual_terminal_state(tmp_path, monkeypatch):
    from sim.act import evaluate
    from sim.environments.sweep_env import SweepEnv
    import mujoco
    pytest.importorskip("torch")
    class RejectedEnv(SweepEnv):
        def step_control(self, command):
            # Real IK uses qpos as scratch and can leave it changed on rejection.
            self.data.qpos[0] += 0.01
            mujoco.mj_forward(self.model, self.data)
            raise RuntimeError("policy and hold rejected")
    class HoldingPolicy:
        def eval(self):
            pass
        def predict_action_chunk(self, _batch):
            raise RuntimeError("no predictions in rejection fixture")
    monkeypatch.setattr(evaluate, "SweepEnv", RejectedEnv)
    monkeypatch.setattr(evaluate, "build_act_policy", lambda *a, **k: (
        HoldingPolicy(), SimpleNamespace(output_features={"action": SimpleNamespace(shape=(4,))})))
    monkeypatch.setattr(evaluate, "build_act_processors", lambda *a, **k: (lambda x: x, lambda x: x))
    monkeypatch.setattr(evaluate, "INFERENCE_MAX_FRAMES", 2)
    cfg = load_config(overrides=["act.image_size=[32,32]", "task.target_count=4",
                                 f"act.dataset_dir={tmp_path / 'missing'}"])
    result = evaluate.run_act_episode(cfg, seed=2110081, preview=True, capture_replay=True)
    assert result.success is False and result.peak_collected == 0
    assert result.sampled_frames == 2 and result.elapsed == pytest.approx(0.4)
    assert len(result.trace) == 8
    assert all(row["control_step_advanced"] is False for row in result.trace)
    episode_id = save_inference_replay(cfg, 2110081, result, tmp_path)
    with np.load(tmp_path / episode_id / "arrays.npz") as arrays:
        np.testing.assert_allclose(arrays["tcp_pose"][-1], result.trace[-1]["tcp_pose"], atol=1e-7)
