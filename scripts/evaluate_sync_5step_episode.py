"""Run one episode with a process-local synchronous 25-prediction/5-action ACT schedule."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from scripts.sync_5step_adapter import (
    SCHEDULE_RECORD,
    adapt_config,
    install_driver_adapter,
)


def _is_known_ik_failure(error):
    return (isinstance(error, RuntimeError)
            and str(error).startswith("UR10 IK target is unreachable:"))


def _trace_frames(error):
    current = error.__traceback__
    while current is not None:
        yield current.tb_frame.f_locals
        current = current.tb_next


def _find_rollout_locals(error):
    selected = {}
    for values in _trace_frames(error):
        if "env" in values and "trace" in values and "actions" in values:
            selected = values
    return selected


def _find_context(error, fallback=None):
    if isinstance(getattr(error, "hybrid_context", None), dict):
        return error.hybrid_context
    for values in _trace_frames(error):
        context = values.get("context")
        if isinstance(context, dict) and "trace" in context and "actions" in context:
            return context
    return fallback if isinstance(fallback, dict) else {}


def _safe_count(env):
    try:
        return int(np.asarray(env.collected_mask(), dtype=bool).sum())
    except BaseException:
        return None


def capture_partial_snapshot(error, cfg, context=None):
    """Capture actual partial state before Hybrid closes the failed episode."""
    rollout = _find_rollout_locals(error)
    context = _find_context(error, context)
    env = rollout.get("env") or context.get("env")
    if env is None:
        for values in _trace_frames(error):
            candidate = values.get("env")
            if candidate is not None and hasattr(candidate, "collected_mask"):
                env = candidate
                break
    final_count = _safe_count(env) if env is not None else None
    final_mask = None
    if env is not None:
        try:
            final_mask = np.asarray(env.collected_mask(), dtype=bool).copy()
        except BaseException:
            pass

    driver_trace = list(rollout.get("trace") or [])
    prefix_trace = list(context.get("trace") or [])
    trace = prefix_trace + driver_trace
    counts = [int(row["collected"]) for row in trace
              if isinstance(row, dict) and type(row.get("collected")) is int]
    timers = [rollout.get("collection_timer"), context.get("collection_timer")]
    peaks = [int(timer.peak_collected) for timer in timers
             if timer is not None and getattr(timer, "peak_collected", None) is not None]
    if final_count is not None:
        peaks.append(final_count)
    peaks.extend(counts)
    peak_count = max(peaks) if peaks else None

    try:
        component_positions = np.asarray(env.component_positions(), dtype=np.float32).copy()
        component_quats = np.asarray(env.component_quats(), dtype=np.float32).copy()
        final_poses = np.concatenate((component_positions, component_quats), axis=1)
    except BaseException:
        final_poses = None
    initial_poses = context.get("sync_initial_object_poses")
    if initial_poses is None:
        for values in _trace_frames(error):
            candidate = values.get("initial")
            if candidate is not None:
                initial_poses = candidate
                break
    if initial_poses is not None:
        try:
            initial_poses = np.asarray(initial_poses, dtype=np.float32).copy()
        except (TypeError, ValueError):
            initial_poses = None

    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "known_ik_failure": _is_known_ik_failure(error),
        "target_count": int(cfg.task.target_count),
        "peak_collected": peak_count,
        "final_collected": final_count,
        "final_collected_mask": final_mask,
        "trace": trace,
        "actions": list(context.get("actions") or []) + list(rollout.get("actions") or []),
        "initial_object_poses": initial_poses,
        "final_object_poses": final_poses,
        "elapsed": float(env.time) if env is not None else None,
        "peak_force": float(rollout.get("peak_force", context.get("peak_force", 0.0))),
        "sampled_frames": int(rollout.get("sampled_frames", 0)),
        "scheduler_queries": int(rollout.get("scheduler_queries", 0)),
        "scheduler_timeouts": int(getattr(rollout.get("scheduler"), "timeouts", 0)),
        "scheduler_late_results": int(getattr(rollout.get("scheduler"), "late_results", 0)),
        "first_entry_time": getattr(rollout.get("collection_timer") or
                                      context.get("collection_timer"), "first_entry_time", None),
        "context": context,
    }


def _write_incident(root, seed, error, snapshot):
    if not root:
        return
    root = Path(root).resolve()
    incident_dir = root / "incidents"
    incident_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "exception_type": type(error).__name__,
        "exception": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "quantity_status": "known" if snapshot.get("final_collected") is not None else "unknown",
        "peak_collected": snapshot.get("peak_collected"),
        "final_collected": snapshot.get("final_collected"),
        "simulation_time": snapshot.get("elapsed"),
        "trace_rows": len(snapshot.get("trace", [])),
        "last_trace_time": (snapshot["trace"][-1].get("t")
                            if snapshot.get("trace") else None),
        "schedule": SCHEDULE_RECORD,
    }
    path = incident_dir / f"seed_{int(seed)}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _known_ik_result(error, snapshot, cfg, model_path):
    from sim.act import evaluate, hybrid_trial4
    from sim.act.rollout import ActRolloutResult

    target = int(cfg.task.target_count)
    effective = hybrid_trial4.formal.candidate_config(cfg, target)
    module = hybrid_trial4.runtime()
    profile = hybrid_trial4.configure_admittance(module, effective)
    final_count = snapshot["final_collected"]
    peak_count = snapshot["peak_collected"]
    if final_count is None or peak_count is None:
        raise RuntimeError("known IK failure has an unreadable physical collection count") from error
    mask = snapshot["final_collected_mask"]
    if mask is None:
        raise RuntimeError("known IK failure has no readable final collection mask") from error
    trace = snapshot["trace"]
    actions = np.asarray(snapshot["actions"], dtype=np.float32).reshape(-1, 4)
    result = ActRolloutResult(
        success=False,
        failure_reason=f"known IK execution failure: {error}",
        observations=[],
        actions=actions,
        trace=trace,
        collected=int(peak_count),
        total=len(mask),
        target_count=target,
        target_collected=int(peak_count),
        unexpected_collected=max(0, int(peak_count) - target),
        final_collected_mask=np.asarray(mask, dtype=bool),
        elapsed=float(snapshot["elapsed"] or 0.0),
        scheduler_queries=int(snapshot["scheduler_queries"]),
        scheduler_timeouts=int(snapshot["scheduler_timeouts"]),
        scheduler_late_results=int(snapshot["scheduler_late_results"]),
        peak_force=float(snapshot["peak_force"]),
        sampled_frames=int(snapshot["sampled_frames"]),
        termination_reason="known_execution_failure:ik_unreachable",
        peak_collected=int(peak_count),
        final_collected=int(final_count),
        first_entry_time=snapshot["first_entry_time"],
        collection_window_seconds=2.0,
    )
    result.physical_attempts = 1
    result.planner_attempts = 1
    result.planner_strategy = "hybrid_astar_act_trial4_dinov3_frozen"
    result.failure_mode = "ik_unreachable"
    result.control_profile = profile
    result.inference_config = effective
    result.model_checkpoint = evaluate._resolve_model_path(model_path)
    result.initial_object_poses = snapshot["initial_object_poses"]
    if result.initial_object_poses is None:
        raise RuntimeError("known IK failure has no reset-pose snapshot") from error
    result.final_object_positions = (snapshot["final_object_poses"][:, :3]
                                     if snapshot["final_object_poses"] is not None
                                     else np.zeros((0, 3), dtype=np.float32))
    result.peak_collected = int(peak_count)
    result.final_collected = int(final_count)
    return result


def _install_episode_adapter():
    from sim.act import evaluate, hybrid_trial4

    original_run = evaluate.run_inference_episode
    module = hybrid_trial4.runtime()
    original_prepare = module.prepare_contact

    def on_driver_error(error, cfg, context):
        error.sync_partial_snapshot = capture_partial_snapshot(error, cfg, context)

    original_build = install_driver_adapter(hybrid_trial4, on_driver_error=on_driver_error)

    def prepare_with_snapshot(env, cfg, plan):
        initial = np.concatenate((env.component_positions(), env.component_quats()), axis=1)
        try:
            context = original_prepare(env, cfg, plan)
        except BaseException as error:
            if _is_known_ik_failure(error):
                context = _find_context(error)
                context.setdefault("env", env)
                context["sync_initial_object_poses"] = np.asarray(initial, dtype=np.float32).copy()
                error.sync_partial_snapshot = capture_partial_snapshot(error, cfg, context)
            raise
        context["sync_initial_object_poses"] = np.asarray(initial, dtype=np.float32).copy()
        return context

    module.prepare_contact = prepare_with_snapshot

    def run_with_sync_schedule(cfg, seed=0, model_path=None, preview=False, capture_replay=False):
        adapted = adapt_config(cfg)
        try:
            return original_run(adapted, seed=seed, model_path=model_path,
                                preview=False, capture_replay=capture_replay)
        except BaseException as error:
            snapshot = getattr(error, "sync_partial_snapshot", None)
            if snapshot is None:
                snapshot = capture_partial_snapshot(error, adapted)
                error.sync_partial_snapshot = snapshot
            if _is_known_ik_failure(error):
                try:
                    return _known_ik_result(error, snapshot, adapted, model_path)
                except BaseException as conversion_error:
                    _write_incident(os.environ.get("SWEEP_SYNC_EVAL_ROOT"), seed,
                                    conversion_error, snapshot)
                    raise
            _write_incident(os.environ.get("SWEEP_SYNC_EVAL_ROOT"), seed, error, snapshot)
            raise

    evaluate.run_inference_episode = run_with_sync_schedule
    return evaluate, hybrid_trial4, module, original_run, original_build, original_prepare


def main(argv=None):
    from sim.act import evaluate
    from sim.act.inference_lock import inference_lease

    _install_episode_adapter()
    with inference_lease(on_wait=lambda: print(json.dumps({
            "event": "waiting_for_inference_slot"}), flush=True)):
        return evaluate._run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
