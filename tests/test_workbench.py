"""Behavior tests for the local robot-learning workbench."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim.act.dataset import ActDatasetWriter
from sim.act.evaluate import build_arg_parser
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
        "state": np.zeros(42, dtype=np.float32),
        "environment_state": np.zeros(3, dtype=np.float32),
        "phase": "sweep",
    }]
    ActDatasetWriter(str(root)).add_episode(
        episode_id, observations, np.zeros((1, 4), dtype=np.float32), True,
        {"seed": 4, "target_count": 2, "total_count": 6,
         "fps": 25.0, "failure_reason": "", "episode_kind": "expert",
         "split": "train"},
    )


def test_workbench_lists_episode_and_loads_all_three_views(tmp_path):
    _episode(tmp_path)
    state = WorkbenchState(load_config(), dataset_root=tmp_path,
                           preview_root=tmp_path / "previews")

    episodes = state.list_episodes()
    assert episodes[0]["episode_id"] == "episode_0000"
    assert episodes[0]["success"] is True
    assert episodes[0]["target_count"] == 2
    frame = state.load_episode_frame("episode_0000", 0)
    assert set(frame) == {"overhead", "wrist", "inspection"}
    assert frame["inspection"].shape == (12, 12, 3)


def test_workbench_rejects_unknown_episode(tmp_path):
    state = WorkbenchState(load_config(), dataset_root=tmp_path,
                           preview_root=tmp_path / "previews")
    with pytest.raises(EpisodeNotFound):
        state.episode_metadata("missing")


def test_workbench_deletes_exact_episode_and_keeps_sibling(tmp_path):
    _episode(tmp_path, "episode_0000")
    _episode(tmp_path, "episode_0001")
    state = WorkbenchState(load_config(), dataset_root=tmp_path,
                           preview_root=tmp_path / "previews")

    deleted = state.delete_episode("episode_0000")

    assert deleted["episode_id"] == "episode_0000"
    assert not (tmp_path / "episode_0000").exists()
    assert state.episode_metadata("episode_0001")["episode_id"] == "episode_0001"
    with pytest.raises(EpisodeNotFound):
        state.episode_metadata("episode_0000")
    manifest_ids = [record["episode_id"] for record in state.list_episodes()]
    assert manifest_ids == ["episode_0001"]


def test_workbench_serializes_episode_listing_with_delete(tmp_path, monkeypatch):
    """A refresh must not observe a directory midway through deletion."""
    import threading
    import time

    _episode(tmp_path, "episode_0000")
    _episode(tmp_path, "episode_0001")
    state = WorkbenchState(load_config(), dataset_root=tmp_path,
                           preview_root=tmp_path / "previews")

    entered = threading.Event()
    release = threading.Event()
    original_frame_count = __import__("sim.web.workbench", fromlist=["_frame_count"])._frame_count

    def blocked_frame_count(root, record):
        if record["episode_id"] == "episode_0000":
            entered.set()
            assert release.wait(1.0)
        return original_frame_count(root, record)

    monkeypatch.setattr("sim.web.workbench._frame_count", blocked_frame_count)
    listed = {}
    listing_error = []

    def list_worker():
        try:
            listed["episodes"] = state.list_episodes()
        except Exception as exc:  # pragma: no cover - assertion below reports it
            listing_error.append(exc)

    listing_thread = threading.Thread(target=list_worker)
    listing_thread.start()
    assert entered.wait(1.0)

    deleted = {}
    delete_thread = threading.Thread(
        target=lambda: deleted.update(state.delete_episode("episode_0000")))
    delete_thread.start()
    time.sleep(0.05)

    release.set()
    listing_thread.join(1.0)
    delete_thread.join(1.0)
    assert not listing_error
    assert deleted["episode_id"] == "episode_0000"
    assert [item["episode_id"] for item in listed["episodes"]] == [
        "episode_0000", "episode_0001"
    ]


def test_job_commands_are_typed_and_use_project_modules(tmp_path):
    train = build_train_argv(
        tmp_path, "cfg.yaml", "dataset", "out", 10, preview_training=True
    )
    infer = build_inference_argv(tmp_path, "cfg.yaml", 3, "model.pt", 4)
    assert train[1:3] == ["-m", "sim.act.train"]
    assert "--steps" in train and "10" in train
    assert "--preview-training" in train
    assert infer[1:3] == ["-m", "sim.act.evaluate"]
    assert "--model" in infer and "model.pt" in infer
    assert "--target-count" in infer and "4" in infer
    assert "--record-replay" in infer
    assert "--replay-root" in infer
    preview_infer = build_inference_argv(tmp_path, "cfg.yaml", 3, "model.pt", 4,
                                         preview=True)
    assert "--preview" in preview_infer


def test_evaluate_cli_accepts_exact_target_count():
    args = build_arg_parser().parse_args([
        "--config", "cfg.yaml", "--seed", "9", "--model", "checkpoint",
        "--target-count", "4",
    ])
    assert args.seed == 9
    assert args.target_count == 4


def test_evaluate_cli_accepts_preview_mode():
    args = build_arg_parser().parse_args(["--preview"])
    assert args.preview is True


def test_inference_route_passes_target_count_to_job(tmp_path):
    app = create_app(load_config(overrides=["sim.real_time=false"]))
    calls = []

    def fake_start(kind, argv):
        calls.append((kind, argv))
        return {"kind": kind, "argv": argv}

    app.state.jobs.start = fake_start
    route = next(route for route in app.routes
                 if getattr(route, "path", None) == "/api/jobs/inference")
    response = route.endpoint({"model": "checkpoint", "seed": 7,
                               "target_count": 4})
    assert response["kind"] == "inference"
    assert calls[0][0] == "inference"
    assert "--target-count" in calls[0][1]
    assert calls[0][1][calls[0][1].index("--target-count") + 1] == "4"
    assert "--preview" in calls[0][1]


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


def test_job_registry_persists_completed_job_and_output(tmp_path):
    import sys
    import time

    registry = JobRegistry(tmp_path)
    job = registry.start("train", [sys.executable, "-c",
                                    "print('persisted-output')"])
    for _ in range(50):
        current = registry.list()[0]
        if current["state"] in {"done", "failed"}:
            break
        time.sleep(0.02)
    assert current["state"] == "done"

    restored = JobRegistry(tmp_path)
    loaded = restored.list()[0]
    assert loaded["job_id"] == job["job_id"]
    assert loaded["state"] == "done"
    assert "persisted-output" in loaded["output"]
    assert loaded["log_path"]


def test_job_registry_recovers_a_live_detached_job(tmp_path):
    import sys
    import time

    registry = JobRegistry(tmp_path)
    job = registry.start("train", [sys.executable, "-c",
                                    "import time; print('live-output'); time.sleep(.4)"])
    time.sleep(0.08)

    restored = JobRegistry(tmp_path)
    loaded = restored.list()[0]
    assert loaded["job_id"] == job["job_id"]
    assert loaded["state"] in {"starting", "running"}

    for _ in range(60):
        current = restored.list()[0]
        if current["state"] in {"done", "failed", "stopped"}:
            break
        time.sleep(0.02)
    assert current["state"] == "done"
    assert "live-output" in current["output"]


def test_training_stop_request_is_recorded_for_graceful_checkpointing():
    from sim.act.train import TrainingStopRequest

    request = TrainingStopRequest()
    assert request.requested is False
    request(15, None)
    assert request.requested is True


def test_job_registry_runs_callable_with_progress_and_cancellation(tmp_path):
    import time

    registry = JobRegistry(tmp_path)

    def runner(job):
        registry.update(job.job_id, message="preview 1/2",
                        progress={"completed": 1, "total": 2})
        time.sleep(0.02)
        registry.update(job.job_id, message="preview 2/2",
                        progress={"completed": 2, "total": 2,
                                  "latest_episode_id": "preview_goal_2_0001"})

    job = registry.start_callable("preview", runner)
    for _ in range(50):
        current = registry.list()[0]
        if current["state"] == "done":
            break
        time.sleep(0.02)
    assert current["state"] == "done"
    assert current["progress"]["completed"] == 2
    assert "preview 2/2" in current["output"]


def test_stopping_callable_keeps_kind_occupied_until_worker_exits(tmp_path):
    import threading
    import time

    registry = JobRegistry(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def runner(_job):
        started.set()
        release.wait(1.0)

    job = registry.start_callable("preview", runner)
    assert started.wait(1.0)
    stopping = registry.stop(job["job_id"])
    assert stopping["state"] == "stopping"
    with pytest.raises(RuntimeError):
        registry.start_callable("preview", lambda _job: None)

    release.set()
    for _ in range(50):
        current = registry.list()[0]
        if current["state"] == "stopped":
            break
        time.sleep(0.02)
    assert current["state"] == "stopped"


def test_workbench_html_has_five_operational_areas():
    app = create_app(load_config(overrides=["sim.real_time=false"]))
    index = next(route for route in app.routes if getattr(route, "path", None) == "/")
    html = index.endpoint()
    html = html.body.decode("utf-8") if hasattr(html, "body") else html
    for label in ("Simulation", "Demonstrations", "Training", "Inference", "Parameters",
                  "inspection-only", "Play", "Frame inspector", "Signal group"):
        assert label in html
    for element_id in ("episodeTargetFilter", "episodeStatusFilter", "previewOutcome",
                       "previewCount", "previewFailureMode", "inferTarget",
                       "inferRandomize"):
        assert f'id="{element_id}"' in html
    assert "Inference replay loaded" in html
    assert 'id="inferPreview"' in html
    assert "function deleteEpisode(" in html
    assert "oncontextmenu=" in html
    assert "['done','failed'].includes(j.state)" in html
    assert "function filteredEpisodes()" in html
    assert "function toggleFailureMode()" in html
    assert "wrong_count · 错误数量" in html
    assert "lose_contact" not in html
    assert "previewJobId" in html
    assert "后台生成示教中" in html


def test_http_config_rejects_unknown_parameter_without_running_a_job():
    from fastapi.testclient import TestClient

    client = TestClient(create_app(load_config(overrides=["sim.real_time=false"])))
    response = client.post("/api/config/apply", json={"planner.name": "fixed"})
    assert response.status_code == 422


def test_http_contract_includes_preview_job_and_config_routes():
    app = create_app(load_config())
    paths = {route.path for route in app.routes}
    assert {"/api/preview", "/api/preview/{preview_id}",
            "/api/preview/fill", "/api/preview/fill-failure-mix",
            "/api/preview/{preview_id}/frame/{frame_index}",
            "/api/episodes/{episode_id}/signals",
            "/api/episodes/{episode_id}/frame/{frame_index}/data",
            "/api/jobs/train", "/api/jobs/inference",
            "/api/config/apply"} <= paths


def test_failure_mix_default_keeps_target_six_physically_valid():
    from fastapi.testclient import TestClient

    app = create_app(load_config(overrides=["sim.real_time=false"]))
    calls = []

    def fake_fill_failure_mix(**kwargs):
        calls.append(kwargs)
        return {"target_count": kwargs["target_count"],
                "counts": {}, "generated": []}

    app.state.workbench.fill_failure_mix = fake_fill_failure_mix
    response = TestClient(app).post(
        "/api/preview/fill-failure-mix",
        json={"target_counts": [5, 6]},
    )
    assert response.status_code == 200
    assert calls[0]["wrong_over_count"] == 2
    assert calls[1]["wrong_over_count"] == 0


def test_success_batch_counts_only_quality_gated_records(tmp_path):
    state = WorkbenchState(load_config(), dataset_root=tmp_path / "dataset",
                           preview_root=tmp_path / "previews")
    calls = []

    def fake_build_preview(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"success": False, "failure_reason": "quality gate rejected"}
        return {"success": True, "episode_id": "preview_goal_2_0001",
                "target_count": 2, "failure_mode": ""}

    state.build_preview = fake_build_preview
    records = state.build_previews(seed=10, target_count=2,
                                   outcome="success", count=1)
    assert len(records) == 1
    assert records[0]["success"] is True
    assert len(calls) == 2
    assert calls[0]["persist_failed"] is False


def test_preview_batch_route_starts_background_job_instead_of_blocking(tmp_path):
    app = create_app(load_config(overrides=["sim.real_time=false"]))
    calls = []

    def fake_start_callable(kind, runner):
        calls.append((kind, runner))
        return {
            "job_id": "preview-job-1",
            "kind": kind,
            "state": "running",
            "progress": {"completed": 0, "total": 15},
        }

    app.state.jobs.start_callable = fake_start_callable
    route = next(route for route in app.routes
                 if getattr(route, "path", None) == "/api/preview")

    response = route.endpoint({"seed": 10, "target_count": 3,
                               "outcome": "success", "count": 15})

    assert response["job_id"] == "preview-job-1"
    assert response["kind"] == "preview"
    assert calls and calls[0][0] == "preview"


def test_failure_quality_gate_rejects_touchdown_only_or_infeasible_rollouts():
    short = SimpleNamespace(
        planner_status="feasible", observations=[None] * 99, collected=0)
    assert not WorkbenchState._failure_quality_ok(
        short, target_count=1, failure_mode="wrong_count_under")

    infeasible = SimpleNamespace(
        planner_status="infeasible", observations=[None] * 400, collected=0)
    assert not WorkbenchState._failure_quality_ok(
        infeasible, target_count=1, failure_mode="wrong_count_under")

    meaningful = SimpleNamespace(
        planner_status="feasible", observations=[None] * 400, collected=0)
    assert WorkbenchState._failure_quality_ok(
        meaningful, target_count=1, failure_mode="wrong_count_under")


def test_stall_quality_gate_requires_missed_parts_at_real_side_wall():
    cfg = load_config()
    common = dict(
        planner_status="feasible", observations=[None] * 200,
        collected=1, target_count=2, target_indices=[0, 1],
        final_collected_mask=np.array([True, False]),
    )
    middle = SimpleNamespace(
        **common,
        final_object_positions=np.array([[0.0, 0.0, 0.004],
                                         [-0.02, 0.02, 0.004]]),
    )
    assert not WorkbenchState._failure_quality_ok(
        middle, target_count=2, failure_mode="stall_outside_tray", cfg=cfg)

    side_wall = SimpleNamespace(
        **common,
        final_object_positions=np.array([[-0.38, 0.0, 0.004],
                                         [-0.37, 0.17, 0.004]]),
    )
    assert WorkbenchState._failure_quality_ok(
        side_wall, target_count=2, failure_mode="stall_outside_tray", cfg=cfg)


def test_wrong_count_clones_use_distinct_success_sources(tmp_path):
    preview_root = tmp_path / "previews"
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    for index in range(4):
        observation = {
            "overhead": image + index,
            "wrist": image + index,
            "inspection": image + index,
            "state": np.full(42, index, dtype=np.float32),
            "environment_state": np.array([6.0, 2.0, 0.0], dtype=np.float32),
            "phase": "sweep",
        }
        ActDatasetWriter(str(preview_root)).add_episode(
            f"preview_goal_2_{index + 1:04d}", [observation],
            np.zeros((1, 4), dtype=np.float32), True,
            {"preview": True, "seed": index, "target_count": 2,
             "total_count": 6, "fps": 25.0, "failure_reason": "",
             "episode_kind": "expert_preview", "split": "pilot"},
        )

    state = WorkbenchState(load_config(), dataset_root=tmp_path / "dataset",
                           preview_root=preview_root)
    first = state._clone_success_as_wrong_count(
        seed=10, target_count=1, failure_mode="wrong_count_over", source_slot=0)
    second = state._clone_success_as_wrong_count(
        seed=11, target_count=1, failure_mode="wrong_count_over", source_slot=1)

    assert first["derived_from"] != second["derived_from"]
    assert first["episode_id"] != second["episode_id"]


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
    assert "refreshPreviewProgress" in html


def test_workbench_discards_stale_async_frame_and_signal_responses():
    """Frame scrubbing must not let an older network response win the race."""
    app = create_app(load_config())
    index = next(route for route in app.routes
                 if getattr(route, "path", None) == "/")
    html = index.endpoint()
    html = html.body.decode("utf-8") if hasattr(html, "body") else html
    assert "AbortController" in html
    assert "seekSerial" in html
    assert "signalSerial" in html
    assert "current?.episode_id!==episodeId" in html
    assert "if(!timer&&!current)" in html


def test_dashboard_reports_plain_text_http_errors_without_json_parse_crash():
    """A proxy/server 5xx body may be plain text, not JSON."""
    app = create_app(load_config())
    index = next(route for route in app.routes
                 if getattr(route, "path", None) == "/")
    html = index.endpoint()
    html = html.body.decode("utf-8") if hasattr(html, "body") else html
    assert "const body=await r.text()" in html
    assert "JSON.parse(body)" in html
    assert "HTTP ${r.status}" in html


def test_job_output_refresh_preserves_manual_scroll_position():
    app = create_app(load_config())
    index = next(route for route in app.routes
                 if getattr(route, "path", None) == "/")
    html = index.endpoint()
    html = html.body.decode("utf-8") if hasattr(html, "body") else html
    assert "outputNode.scrollTop" in html
    assert "outputNode.scrollHeight" in html


def test_workbench_persists_and_reads_preview_fz_trace(tmp_path, monkeypatch):
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    observation = {
        "overhead": image,
        "wrist": image,
        "inspection": image,
        "state": np.zeros(42, dtype=np.float32),
        "environment_state": np.zeros(3, dtype=np.float32),
        "t": 0.0,
        "normal_force": 0.75,
        "contact": True,
        "phase": "sweep",
        "policy_mask": True,
    }
    result = SimpleNamespace(
        observations=[observation],
        actions=np.zeros((1, 4), dtype=np.float32),
        success=True,
        failure_reason="",
        planner_status="feasible",
        planner_failure_reason="",
        planner_strategy="capture_lane",
        planner_turn_count=2,
        planner_score=0.8,
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

    metadata = state.build_preview(seed=8, target_count=1)

    signal_path = preview_root / metadata["episode_id"] / "signals.json"
    assert signal_path.is_file()
    signals = state.load_episode_signals(metadata["episode_id"])
    assert metadata["planner_strategy"] == "capture_lane"
    assert metadata["planner_status"] == "feasible"
    assert metadata["planner_turn_count"] == 2
    assert signals["episode_id"] == metadata["episode_id"]
    assert signals["t"] == [0.0]
    assert signals["fz"] == [0.75]
    assert signals["contact"] == [True]


def test_partial_preview_keeps_the_concentrated_six_part_layout(tmp_path, monkeypatch):
    seen_spawn_modes = []
    result = SimpleNamespace(
        observations=[{
            "overhead": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
                "state": np.zeros(42, dtype=np.float32),
                "environment_state": np.zeros(3, dtype=np.float32),
                "phase": "sweep",
        }],
        actions=np.zeros((1, 4), dtype=np.float32),
        success=False,
        failure_reason="no_single_capture_lane",
        planner_status="infeasible",
        planner_failure_reason="no_single_capture_lane",
        planner_strategy="no_capture_lane",
        planner_turn_count=0,
        planner_score=float("inf"),
        trace=[],
        target_indices=[],
        collected=0,
        target_collected=0,
        unexpected_collected=0,
    )

    def fake_rollout(cfg, seed, collect_observations):
        seen_spawn_modes.append(str(cfg.components.spawn_mode))
        return result

    monkeypatch.setattr("sim.web.workbench.run_expert_episode", fake_rollout)
    state = WorkbenchState(load_config(), dataset_root=tmp_path / "dataset",
                           preview_root=tmp_path / "previews")

    for target_count in (1, 3, 6):
        state.build_preview(seed=target_count, target_count=target_count)

    # Preview generation now retries a failed physical layout before
    # persisting it, while every retry must keep the same six-part cluster
    # contract.
    assert len(seen_spawn_modes) == 18
    assert seen_spawn_modes == ["cluster"] * 18


def test_workbench_serializes_mujoco_render_requests(monkeypatch):
    import asyncio
    import threading
    import time

    class FakeEnv:
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def __init__(self, cfg, seed):
            self.layout = []
            self.time = 0.0

        def reset(self, seed):
            return None

        def render_rgb(self, camera, size):
            self._render()
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def render_wrist_rgb(self, size):
            self._render()
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def _render(self):
            with self.state_lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            time.sleep(0.01)
            with self.state_lock:
                type(self).active -= 1

        def collected_mask(self):
            return np.zeros(0, dtype=bool)

        def tcp(self):
            return np.zeros(3)

        def normal_force(self):
            return 0.0

        def close(self):
            return None

    monkeypatch.setattr("sim.environments.sweep_env.SweepEnv", FakeEnv)
    app = create_app(load_config(overrides=["sim.real_time=false"]))
    frames_endpoint = next(route.endpoint for route in app.routes
                           if getattr(route, "path", None) == "/api/frames")

    async def render_twice():
        return await asyncio.gather(frames_endpoint(), frames_endpoint())

    results = asyncio.run(render_twice())

    assert len(results) == 2
    assert FakeEnv.max_active == 1


def test_frame_render_route_keeps_mujoco_on_the_event_loop_thread():
    import inspect

    app = create_app(load_config(overrides=["sim.real_time=false"]))
    frames_endpoint = next(route.endpoint for route in app.routes
                           if getattr(route, "path", None) == "/api/frames")

    assert inspect.iscoroutinefunction(frames_endpoint)
