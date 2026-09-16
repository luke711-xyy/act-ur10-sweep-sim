"""Behavior tests for the local robot-learning workbench."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim.act.dataset import ActDatasetWriter
from sim.config import load_config
from sim.web.app import create_app
from sim.web.jobs import JobRegistry, build_inference_argv, build_train_argv
from sim.web.workbench import EpisodeNotFound, WorkbenchState


def _episode(root, episode_id="episode_0000"):
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    observations = [{
        "overhead": image,
        "wrist": image + 1,
        "inspection": image + 2,
        "state": np.zeros(7, dtype=np.float32),
        "environment_state": np.zeros(3, dtype=np.float32),
    }]
    ActDatasetWriter(str(root)).add_episode(
        episode_id, observations, np.zeros((1, 4), dtype=np.float32), True,
        {"seed": 4, "failure_reason": ""},
    )


def test_workbench_lists_episode_and_loads_all_three_views(tmp_path):
    _episode(tmp_path)
    state = WorkbenchState(load_config(), dataset_root=tmp_path,
                           preview_root=tmp_path / "previews")

    episodes = state.list_episodes()
    assert episodes[0]["episode_id"] == "episode_0000"
    assert episodes[0]["success"] is True
    frame = state.load_episode_frame("episode_0000", 0)
    assert set(frame) == {"overhead", "wrist", "inspection"}
    assert frame["inspection"].shape == (12, 12, 3)


def test_workbench_rejects_unknown_episode(tmp_path):
    state = WorkbenchState(load_config(), dataset_root=tmp_path,
                           preview_root=tmp_path / "previews")
    with pytest.raises(EpisodeNotFound):
        state.episode_metadata("missing")


def test_job_commands_are_typed_and_use_project_modules(tmp_path):
    train = build_train_argv(tmp_path, "cfg.yaml", "dataset", "out", 10)
    infer = build_inference_argv(tmp_path, "cfg.yaml", 3, "model.pt")
    assert train[1:3] == ["-m", "sim.act.train"]
    assert "--steps" in train and "10" in train
    assert infer[1:3] == ["-m", "sim.act.evaluate"]
    assert "--model" in infer and "model.pt" in infer


def test_job_registry_captures_a_local_job_and_rejects_duplicate_kind(tmp_path):
    import sys
    import time

    registry = JobRegistry(tmp_path)
    job = registry.start("train", [sys.executable, "-c",
                                    "import time; print('workbench-ok'); time.sleep(.2)"])
    assert job["state"] in {"starting", "running"}
    with pytest.raises(RuntimeError):
        registry.start("train", [sys.executable, "-c", "print('duplicate')"])
    for _ in range(50):
        current = registry.list()[0]
        if current["state"] in {"done", "failed"}:
            break
        time.sleep(0.02)
    assert current["state"] == "done"
    assert "workbench-ok" in current["output"]


def test_workbench_html_has_five_operational_areas():
    app = create_app(load_config(overrides=["sim.real_time=false"]))
    index = next(route for route in app.routes if getattr(route, "path", None) == "/")
    html = index.endpoint()
    html = html.body.decode("utf-8") if hasattr(html, "body") else html
    for label in ("Simulation", "Demonstrations", "Training", "Inference", "Parameters",
                  "inspection-only", "Play"):
        assert label in html


def test_http_config_rejects_unknown_parameter_without_running_a_job():
    from fastapi.testclient import TestClient

    client = TestClient(create_app(load_config(overrides=["sim.real_time=false"])))
    response = client.post("/api/config/apply", json={"planner.name": "fixed"})
    assert response.status_code == 422


def test_http_contract_includes_preview_job_and_config_routes():
    app = create_app(load_config())
    paths = {route.path for route in app.routes}
    assert {"/api/preview", "/api/preview/{preview_id}",
            "/api/preview/{preview_id}/frame/{frame_index}",
            "/api/episodes/{episode_id}/signals",
            "/api/jobs/train", "/api/jobs/inference",
            "/api/config/apply"} <= paths


def test_ur10_ik_stays_continuous_for_a_small_tcp_move():
    from sim.controllers.hybrid import Command
    from sim.environments.sweep_env import SweepEnv

    cfg = load_config(overrides=["sim.real_time=false"])
    env = SweepEnv(cfg, seed=0)
    env.reset(seed=0)
    q0 = env.ee.joint_state().copy()
    env.step_control(Command(0.42, 0.006, 0.18, 0.0))
    q1 = env.ee.joint_state().copy()
    env.close()
    assert np.max(np.abs(q1 - q0)) < 0.05


def test_playback_uses_the_live_view_canvas_and_renders_fz():
    app = create_app(load_config())
    index = next(route for route in app.routes if getattr(route, "path", None) == "/")
    html = index.endpoint()
    assert 'id="demoOverheadImage"' not in html
    assert 'id="demoWristImage"' not in html
    assert 'id="demoInspectionImage"' not in html
    assert 'class="demo-views"' not in html
    assert "image('overheadImage',f.overhead)" in html
    assert "image('wristImage',f.wrist)" in html
    assert "image('inspectionImage',f.inspection)" in html
    assert "renderFz" in html
    assert "/api/episodes/${encodeURIComponent(id)}/signals" in html


def test_workbench_persists_and_reads_preview_fz_trace(tmp_path, monkeypatch):
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    observation = {
        "overhead": image,
        "wrist": image,
        "inspection": image,
        "state": np.zeros(7, dtype=np.float32),
        "environment_state": np.zeros(3, dtype=np.float32),
    }
    result = SimpleNamespace(
        observations=[observation],
        actions=np.zeros((1, 4), dtype=np.float32),
        success=True,
        failure_reason="",
        trace=[
            {"t": 0.01, "normal_force": 0.0, "contact": False},
            {"t": 0.02, "normal_force": 0.75, "contact": True},
        ],
    )
    monkeypatch.setattr("sim.web.workbench.run_expert_episode",
                        lambda cfg, seed, collect_observations: result)
    preview_root = tmp_path / "previews"
    state = WorkbenchState(load_config(), dataset_root=tmp_path / "dataset",
                           preview_root=preview_root)

    metadata = state.build_preview(seed=8, count=1)

    signal_path = preview_root / metadata["episode_id"] / "signals.json"
    assert signal_path.is_file()
    assert state.load_episode_signals(metadata["episode_id"]) == {
        "episode_id": metadata["episode_id"],
        "t": [0.01, 0.02],
        "fz": [0.0, 0.75],
        "contact": [False, True],
    }
