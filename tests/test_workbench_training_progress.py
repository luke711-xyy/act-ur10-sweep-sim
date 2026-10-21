"""Read-only training telemetry must survive refresh, partial writes and resume."""

import json
import os

import pytest


def test_progress_endpoint_has_an_explicit_empty_state(monkeypatch):
    from fastapi.testclient import TestClient
    from sim.config import load_config
    from sim.web.app import create_app

    app = create_app(load_config())
    monkeypatch.setattr(app.state.jobs, "list", lambda: [])
    with TestClient(app) as client:
        response = client.get("/api/training/progress")
    assert response.status_code == 200
    assert response.json()["state"] == "idle"
    assert response.json()["points"] == []


def fixture_run(tmp_path):
    root = tmp_path / "runs" / "dp"
    root.mkdir(parents=True)
    (root / "training_config.json").write_text(json.dumps({"max_steps": 100000}))
    job = {"job_id": "live", "kind": "train", "state": "running",
           "started_at": 100, "pid": os.getpid(),
           "argv": ["python", "-m", "sim.diffusion.pipeline", "--out", "runs/dp"]}
    return root, job


def append(path, rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_full_training_is_not_mixed_with_pilot_and_rates_use_real_step_deltas(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    path = root / "curves.jsonl"
    append(path, [
        {"stage": "pilot", "step": 75, "loss": 99, "budget_seconds": 300},
        {"stage": "full", "step": 100, "loss": 2, "budget_seconds": 600},
        {"stage": "full", "step": 150, "loss": 0, "budget_seconds": 700},
        {"stage": "full", "step": 200, "loss": 1, "budget_seconds": 900},
    ])
    os.utime(path, (1000, 1000))
    data = TrainingProgress(tmp_path).snapshot([job], now=1005)
    assert data["step"] == 200
    assert data["max_steps"] == 100000
    assert data["stage"] == "full"
    assert [p["step"] for p in data["points"]] == [100, 150, 200]
    assert data["loss"] == 1
    assert data["mean_loss"] == 1
    assert data["seconds_per_update"] == 3
    assert data["updates_per_minute"] == 20
    assert data["freshness"] == "live"


def test_partial_lines_are_retried_without_duplicates_and_stale_speed_is_not_current(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    path = root / "curves.jsonl"
    append(path, [{"stage": "full", "step": 1, "loss": 0, "budget_seconds": 10}])
    with path.open("a") as stream:
        stream.write('{"stage":"full","step":2,"loss":2,"budget_seconds":12')
    os.utime(path, (100, 100))
    reader = TrainingProgress(tmp_path)
    first = reader.snapshot([job], now=101)
    assert first["step"] == 1
    assert first["loss"] == 0
    assert first["seconds_per_update"] is None
    with path.open("a") as stream:
        stream.write('}\n')
    os.utime(path, (102, 102))
    second = reader.snapshot([job], now=103)
    assert second["step"] == 2
    assert len(second["points"]) == 2
    assert second["seconds_per_update"] == 2
    stale = reader.snapshot([job], now=200)
    assert stale["freshness"] == "stale"
    assert stale["seconds_per_update"] is None
    assert stale["loss"] == 2


def test_invalid_metrics_do_not_become_zero_or_crash_json_and_restart_discards_old_series(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    path = root / "curves.jsonl"
    append(path, [{"stage": "full", "step": 10, "loss": 5},
                  {"stage": "full", "step": 11, "loss": float("nan")},
                  {"stage": "full", "step": 12, "loss": None}])
    reader = TrainingProgress(tmp_path)
    data = reader.snapshot([job])
    assert data["loss"] == 5
    assert data["invalid_rows"] == 2
    json.dumps(data, allow_nan=False)
    path.write_text(json.dumps({"stage": "full", "step": 1, "loss": 3}) + "\n")
    data = reader.snapshot([job])
    assert data["step"] == 1
    assert len(data["points"]) == 1
    assert data["mean_loss"] == 3


def test_dead_process_is_not_reported_as_running_from_a_stale_job_record(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    append(root / "curves.jsonl", [{"stage": "full", "step": 4, "loss": 1}])
    job["pid"] = 2147483647
    data = TrainingProgress(tmp_path).snapshot([job])
    assert data["state"] == "exited"
    assert data["freshness"] == "stopped"
    assert data["seconds_per_update"] is None


def test_only_job_owned_local_paths_are_read_and_active_job_wins(tmp_path):
    from sim.web.training_progress import TrainingProgress

    _, job = fixture_run(tmp_path)
    older = {**job, "job_id": "old", "state": "done", "started_at": 200}
    data = TrainingProgress(tmp_path).snapshot([older, job])
    assert data["job_id"] == "live"
    job["argv"][-1] = "../outside"
    with pytest.raises(ValueError, match="workspace"):
        TrainingProgress(tmp_path).snapshot([job])


def test_history_is_bounded_for_browser_but_preserves_first_latest_and_loss_spike(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    append(root / "curves.jsonl", [
        {"stage": "full", "step": i, "loss": 50 if i == 1501 else 1,
         "budget_seconds": i * 2} for i in range(1, 4001)])
    data = TrainingProgress(tmp_path).snapshot([job])
    assert len(data["points"]) <= 1200
    assert data["points"][0]["step"] == 1
    assert data["points"][-1]["step"] == 4000
    assert max(p["loss"] for p in data["points"]) == 50
    assert data["mean_loss"] == 1


def test_act_logs_remain_readable_without_diffusion_metadata(tmp_path):
    from sim.web.training_progress import TrainingProgress

    path = tmp_path / "runs" / ".workbench_jobs" / "act.log"
    path.parent.mkdir(parents=True)
    append(path, [{"step": 100, "loss": 2, "elapsed_s": 10},
                  {"step": 200, "loss": 1, "elapsed_s": 30}])
    job = {"job_id": "act", "kind": "train", "state": "done", "argv":
           ["python", "-m", "sim.act.train", "--out", "runs/act", "--steps", "1000"],
           "log_path": str(path)}
    data = TrainingProgress(tmp_path).snapshot([job])
    assert data["policy_type"] == "act"
    assert data["step"] == 200
    assert data["max_steps"] == 1000
    assert data["seconds_per_update"] is None  # finished runs have no live speed
    assert data["loss"] == 1


def test_completed_training_is_not_live_while_pipeline_runs_acceptance(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    path = root / "curves.jsonl"
    append(path, [{"stage": "full", "step": 1, "loss": 1, "budget_seconds": 2},
                  {"stage": "full", "step": 2, "loss": 0.5, "budget_seconds": 4}])
    (root / "status.json").write_text(json.dumps({"state": "training_complete", "stage": "full"}))
    os.utime(path, (100, 100))
    data = TrainingProgress(tmp_path).snapshot([job], now=101)
    assert data["freshness"] == "stopped"
    assert data["seconds_per_update"] is None


def test_progress_api_serves_live_metrics_without_starting_or_stopping_jobs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from sim.config import load_config
    from sim.web.app import create_app
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    append(root / "curves.jsonl", [{"stage": "full", "step": 7, "loss": 0.25}])
    app = create_app(load_config())
    app.state.training_progress = TrainingProgress(tmp_path)
    monkeypatch.setattr(app.state.jobs, "list", lambda: [job])
    with TestClient(app) as client:
        data = client.get("/api/training/progress").json()
        assert data["job_id"] == "live"
        assert data["step"] == 7
        assert data["loss"] == 0.25
        assert client.get("/training-monitor.js").status_code == 200


def test_a_later_poll_cannot_mutate_an_earlier_response(tmp_path):
    from sim.web.training_progress import TrainingProgress

    root, job = fixture_run(tmp_path)
    path = root / "curves.jsonl"
    append(path, [{"stage": "full", "step": 1, "loss": 1}])
    reader = TrainingProgress(tmp_path)
    first = reader.snapshot([job])
    append(path, [{"stage": "full", "step": 2, "loss": 2}])
    second = reader.snapshot([job])
    assert len(first["points"]) == 1
    assert first["step"] == 1
    assert len(second["points"]) == 2
