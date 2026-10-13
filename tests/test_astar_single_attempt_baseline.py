"""A* baseline cannot use physical outcomes to choose another plan."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_astar_baseline.py"
    assert path.is_file(), "single-attempt evaluator is missing"
    spec = importlib.util.spec_from_file_location("astar_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paired_act(tmp_path):
    import hashlib
    import json
    act = tmp_path / "act"
    act.mkdir()
    plan = [{"key": f"k{i}", "seed": 2120000+i, "layout_fingerprint": f"f{i}",
             "target_count": i % 6 + 1} for i in range(120)]
    (act / "episode_plan.json").write_text(json.dumps(plan))
    config = act / "evaluation_config.yaml"
    config.write_text("planner: {expert_planning_mode: coverage_astar_only}\n")
    (act / "metadata.json").write_text(json.dumps({"source_fingerprints": {},
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "episode_plan_sha256": hashlib.sha256((act / "episode_plan.json").read_bytes()).hexdigest()}))
    (act / "status.json").write_text(json.dumps({"state": "complete", "completed": 120, "planned": 120}))
    (act / "summary.json").write_text(json.dumps({"complete": True, "completed": 120, "planned": 120}))
    (act / "episodes.jsonl").write_text("".join(json.dumps(p)+"\n" for p in plan))
    return act


@pytest.mark.parametrize("success", [True, False])
def test_exactly_one_physical_execution_and_no_failure_retry(monkeypatch, success):
    module = runner()
    events = []
    env = SimpleNamespace(table_top_z=0.0)
    plan = SimpleNamespace(feasible=True, target_indices=np.array([0]))
    rejected = SimpleNamespace(feasible=True, target_indices=np.array([1]))
    result = SimpleNamespace(success=success, collected=1 if success else 0)
    def candidates(actual_env, cfg, max_candidates):
        assert actual_env is env and max_candidates == 1
        events.append("plan")
        return [plan, rejected]  # Even an oversized shortlist grants no retry.
    def path(cfg, actual_plan, **kwargs):
        assert actual_plan is plan
        return np.zeros((2, 4)), ["approach", "sweep"]
    def execute(actual_env, cfg, waypoints, **kwargs):
        assert actual_env is env
        assert kwargs["collect_observations"] is False
        assert kwargs["recovery_planner"] is None
        assert kwargs["max_recovery_rounds"] == 0
        events.append("execute")
        return result
    monkeypatch.setattr(module, "expert_plan_candidates", candidates)
    monkeypatch.setattr(module, "_expert_execution_path", path)
    monkeypatch.setattr(module, "run_action_path", execute)
    monkeypatch.setattr(module, "_annotate_expert_result", lambda r, p, n: r)
    assert module.execute_once(env, object()) is result
    assert events == ["plan", "execute"]


def test_no_geometric_plan_is_scored_without_a_safe_probe(monkeypatch):
    module = runner()
    monkeypatch.setattr(module, "expert_plan_candidates", lambda *a, **k: [])
    monkeypatch.setattr(module, "run_action_path", lambda *a, **k: pytest.fail("no fallback rollout"))
    assert module.execute_once(object(), object()) is None


def test_plan_validation_requires_twenty_unique_trials_per_goal():
    module = runner()
    plan = [{"key": f"k{i}", "seed": 2120000+i, "layout_fingerprint": f"f{i}",
             "target_count": i % 6 + 1} for i in range(120)]
    module.validate_plan(plan)
    plan[1]["seed"] = plan[0]["seed"]
    with pytest.raises(ValueError, match="unique"):
        module.validate_plan(plan)


def test_ledger_keeps_failures_and_resumes_without_more_opportunities(tmp_path, monkeypatch, paired_act):
    import json
    module = runner()
    act = paired_act
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module, "_source_fingerprints", lambda p: {})
    from contextlib import nullcontext
    monkeypatch.setattr(module, "inference_lease", nullcontext)
    calls = []
    def trial(cfg, spec):
        calls.append(spec["seed"])
        return {**spec, "success": spec["seed"] != 2120000,
                "final_collected": 0 if spec["seed"] == 2120000 else spec["target_count"],
                "physical_attempts": 1}
    monkeypatch.setattr(module, "evaluate_trial", trial)
    out = tmp_path / "baseline"
    module.run_batch(act, out)
    module.run_batch(act, out)
    rows = [json.loads(s) for s in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == len(calls) == 120
    assert rows[0]["success"] is False
    assert json.loads((out / "summary.json").read_text())["successes"] == 119
    # An attempt marker without a scored row is not permission to retry.
    (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:-1])+"\n")
    with pytest.raises(RuntimeError, match="automatic retry is forbidden"):
        module.run_batch(act, out)
    assert len(calls) == 120


def test_valid_but_changed_plan_rejected_before_any_attempt(tmp_path, paired_act, monkeypatch):
    import json
    from contextlib import nullcontext
    module = runner()
    monkeypatch.setattr(module, "inference_lease", nullcontext)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module, "_source_fingerprints", lambda p: {})
    monkeypatch.setattr(module, "evaluate_trial", lambda *a: pytest.fail("changed plan executed"))
    plan = json.loads((paired_act / "episode_plan.json").read_text())
    plan[0]["seed"] = 3000000
    (paired_act / "episode_plan.json").write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="ACT plan changed"):
        module.run_batch(paired_act, tmp_path / "baseline")
    assert not (tmp_path / "baseline/attempts").exists()


@pytest.mark.parametrize("reason", ["incomplete", "active"])
def test_cannot_execute_before_act_is_complete_even_without_wait_flag(tmp_path, paired_act, monkeypatch, reason):
    import json
    from contextlib import nullcontext
    module = runner()
    monkeypatch.setattr(module, "inference_lease", nullcontext)
    monkeypatch.setattr(module, "_source_fingerprints", lambda p: {})
    monkeypatch.setattr(module, "evaluate_trial", lambda *a: pytest.fail("ACT still active"))
    command = f"python -m sim.act.batch_evaluate --out {paired_act}" if reason == "active" else ""
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: command)
    if reason == "incomplete":
        (paired_act / "status.json").write_text(json.dumps({"state": "running", "completed": 119}))
    with pytest.raises(RuntimeError, match="ACT batch is not complete"):
        module.run_batch(paired_act, tmp_path / "baseline")
    assert not (tmp_path / "baseline/attempts").exists()


def test_completed_ledger_rebuilds_interrupted_final_summary(tmp_path, paired_act, monkeypatch):
    import json
    module = runner()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module, "_source_fingerprints", lambda p: {})
    from contextlib import nullcontext
    monkeypatch.setattr(module, "inference_lease", nullcontext)
    monkeypatch.setattr(module, "evaluate_trial", lambda cfg, spec: {**spec, "success": False, "physical_attempts": 0})
    out = tmp_path / "baseline"
    module.run_batch(paired_act, out)
    (out / "summary.json").write_text("{")
    monkeypatch.setattr(module, "evaluate_trial", lambda *a: pytest.fail("completed layout rerun"))
    module.run_batch(paired_act, out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is True and summary["completed"] == 120
    assert summary["physical_attempts"] == 0 and summary["no_plan_failures"] == 120


def test_waiter_tolerates_transient_summary_write(tmp_path, paired_act, monkeypatch):
    import json
    from contextlib import nullcontext
    module = runner()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module, "_source_fingerprints", lambda p: {})
    monkeypatch.setattr(module, "inference_lease", nullcontext)
    (paired_act / "summary.json").write_text("{")
    waited = []
    def sleep(seconds):
        waited.append(seconds)
        (paired_act / "summary.json").write_text(json.dumps({"complete": True, "completed": 120, "planned": 120}))
    monkeypatch.setattr(module.time, "sleep", sleep)
    called = []
    def trial(cfg, spec):
        called.append(spec["seed"])
        return {**spec, "success": False, "physical_attempts": 1}
    monkeypatch.setattr(module, "evaluate_trial", trial)
    module.main(["--act-batch", str(paired_act), "--out", str(tmp_path / "baseline"), "--wait-act"])
    assert waited == [30] and len(called) == 120


@pytest.mark.parametrize("alias", ["same", "child", "parent"])
def test_output_cannot_overwrite_frozen_act_batch(tmp_path, paired_act, alias):
    module = runner()
    output = {"same": paired_act, "child": paired_act / "baseline", "parent": tmp_path}[alias]
    before = (paired_act / "status.json").read_bytes()
    with pytest.raises(ValueError, match="output must be separate"):
        module.main(["--act-batch", str(paired_act), "--out", str(output), "--wait-act"])
    assert (paired_act / "status.json").read_bytes() == before


def test_competing_runner_cannot_publish_any_output_state(tmp_path, paired_act, monkeypatch):
    import fcntl
    module = runner()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module, "_source_fingerprints", lambda p: {})
    out = tmp_path / "baseline"
    out.mkdir()
    (out / "status.json").write_text('{"state":"running","pid":123}')
    before = (out / "status.json").read_bytes()
    with (out / "batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.main(["--act-batch", str(paired_act), "--out", str(out)])
        assert not (out / "metadata.json").exists()
        assert (out / "status.json").read_bytes() == before


def test_single_attempt_uses_same_2000_frame_budget_as_act(monkeypatch):
    import hashlib
    from sim.config import load_config
    module = runner()
    cfg = load_config()
    positions = np.zeros((6, 3))
    quats = np.tile([1, 0, 0, 0], (6, 1))
    fingerprint = hashlib.sha256(np.concatenate((positions, quats), axis=1).astype("<f4").tobytes()).hexdigest()
    env = SimpleNamespace(reset=lambda seed: None, component_positions=lambda: positions,
                          component_quats=lambda: quats, collected_mask=lambda: np.zeros(6, bool),
                          close=lambda: None)
    monkeypatch.setattr(module, "SweepEnv", lambda c, seed: env)
    def once(actual, local):
        assert local.episode.max_frames == 2000
        assert local.episode.max_time == 2000 / local.act.action_hz
        return None
    monkeypatch.setattr(module, "execute_once", once)
    row = module.evaluate_trial(cfg, {"key": "one", "seed": 2, "target_count": 3,
                                     "layout_fingerprint": fingerprint})
    assert row["success"] is False and row["final_collected"] == 0
