"""Paired, render-free A* baseline: one opportunity, at most one execution.

Kept outside sim so adding this tool cannot change a running ACT batch's
frozen simulation fingerprint. It calls the existing expert primitives, never
the best-of-many demonstration collector or an ACT policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from sim.act.batch_evaluate import _source_fingerprints
from sim.act.collect_dataset import _expert_config
from sim.act.expert import expert_plan_candidates
from sim.act.inference_lock import inference_lease
from sim.act.rollout import _annotate_expert_result, _expert_execution_path, run_action_path
from sim.config import load_config
from sim.environments.sweep_env import SweepEnv


def execute_once(env, cfg):
    """Only geometric costs may select the plan; never inspect a probe outcome."""
    plans = expert_plan_candidates(env, cfg, max_candidates=1)
    if not plans or not plans[0].feasible:
        return None
    plan = plans[0]
    path, phases = _expert_execution_path(cfg, plan, table_top_z=env.table_top_z)
    result = run_action_path(
        env, cfg, path, collect_observations=False, phases=phases,
        target_indices=np.asarray(plan.target_indices, dtype=int),
        recovery_planner=None, max_recovery_rounds=0,
        contact_z_mode="expert_admittance",
    )
    return _annotate_expert_result(result, plan, 1)


def validate_plan(plan):
    if len(plan) != 120 or Counter(p["target_count"] for p in plan) != Counter({i: 20 for i in range(1, 7)}):
        raise ValueError("baseline requires exactly twenty trials per goal, 120 total")
    if any(len({p[k] for p in plan}) != 120 for k in ("key", "seed", "layout_fingerprint")):
        raise ValueError("baseline requires unique keys, seeds and layouts")


def evaluate_trial(cfg, spec):
    local = _expert_config(cfg, spec["target_count"])
    # Match ACT's hard physical budget; retain expert path/control and hold.
    local.set_path("episode.max_frames", 2000)
    local.set_path("episode.max_time", 2000 / float(local.act.action_hz))
    if local.get_path("planner.expert_planning_mode") != "coverage_astar_only":
        raise ValueError("baseline must use the current coverage A* planner")
    env = SweepEnv(local, seed=spec["seed"])
    try:
        env.reset(seed=spec["seed"])
        poses = np.concatenate((env.component_positions(), env.component_quats()), axis=1).astype("<f4")
        fingerprint = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
        if fingerprint != spec["layout_fingerprint"]:
            raise RuntimeError("A* initial layout does not match its paired ACT trial")
        result = execute_once(env, local)
        final_count = int(env.collected_mask().sum())
        return {
            "key": spec["key"], "seed": spec["seed"],
            "target_count": spec["target_count"], "layout_fingerprint": fingerprint,
            "model_label": "astar_single_attempt", "model_path": None,
            "success": bool(result is not None and result.success),
            "final_collected": final_count,
            "reason": str(result.failure_reason) if result is not None else "no feasible A* plan",
            "planning_calls": 1, "physical_attempts": int(result is not None),
            "planner_strategy": str(result.planner_strategy) if result is not None else "coverage_astar",
            "elapsed": float(result.elapsed) if result is not None else 0.0,
            "native_success_contract": "expert any exact N, no partial overlap, delivery/contact and one-second stable hold",
            "frame_budget": 2000,
            "media_retained": False,
        }
    finally:
        env.close()


def act_complete(act_root):
    """Completion is mandatory; partial JSON writes only postpone readiness."""
    try:
        status = json.loads((act_root / "status.json").read_text())
        summary = json.loads((act_root / "summary.json").read_text())
    except (OSError, ValueError):
        return False
    if status.get("state") == "needs_attention":
        raise RuntimeError("ACT batch needs attention; do not start baseline")
    if not (summary.get("complete") is True and summary.get("completed") == 120
            and status.get("state") == "complete" and status.get("completed") == 120):
        return False
    commands = subprocess.check_output(["ps", "-Ao", "args="], text=True).splitlines()
    stages = (" -m sim.act.batch_evaluate ", " -m sim.act.evaluate ", " -m sim.act.replay ", "ffmpeg")
    return not any(str(act_root) in cmd and any(stage in cmd for stage in stages) for cmd in commands)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish_summary(root, rows):
    write_json(root / "summary.json", {
        "completed": len(rows), "planned": 120, "complete": len(rows) == 120,
        "successes": sum(r["success"] for r in rows),
        "physical_attempts": sum(r["physical_attempts"] for r in rows),
        "no_plan_failures": sum(r["physical_attempts"] == 0 for r in rows),
        "by_target": {str(t): {"episodes": sum(r["target_count"] == t for r in rows),
            "successes": sum(r["success"] for r in rows if r["target_count"] == t)} for t in range(1, 7)},
    })


def run_batch(act_root, root):
    act_root, root = Path(act_root).resolve(), Path(root).resolve()
    plan = json.loads((act_root / "episode_plan.json").read_text())
    validate_plan(plan)
    metadata = json.loads((act_root / "metadata.json").read_text())
    config = act_root / "evaluation_config.yaml"
    identity = {
        "kind": "astar_single_attempt_v1", "act_batch": str(act_root),
        "plan_sha256": hashlib.sha256((act_root / "episode_plan.json").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "sources_sha256": metadata["source_fingerprints"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_attempt_limit": 1, "candidate_physical_screening": False,
        "opportunity_limit": 1, "no_plan_is_failed_opportunity": True,
        "frame_budget": 2000,
    }
    if identity["config_sha256"] != metadata["config_sha256"]:
        raise RuntimeError("paired ACT config changed")
    if identity["plan_sha256"] != metadata["episode_plan_sha256"]:
        raise RuntimeError("paired ACT plan changed")
    if not act_complete(act_root):
        raise RuntimeError("ACT batch is not complete; baseline execution is forbidden")
    act_rows = [json.loads(s) for s in (act_root / "episodes.jsonl").read_text().splitlines() if s.strip()]
    by_key = {s["key"]: s for s in plan}
    if len(act_rows) != 120 or len({r["key"] for r in act_rows}) != 120:
        raise RuntimeError("ACT ledger is not complete")
    if any(r["key"] not in by_key or any(r[k] != by_key[r["key"]][k]
            for k in ("seed", "target_count", "layout_fingerprint")) for r in act_rows):
        raise RuntimeError("ACT ledger does not match frozen plan")
    if _source_fingerprints(PROJECT) != identity["sources_sha256"]:
        raise RuntimeError("simulation source changed before baseline")
    cfg = load_config(str(config))
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "metadata.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != identity:
            raise RuntimeError("baseline identity changed; refuse to mix runs")
    else:
        write_json(snapshot, identity)
    ledger = root / "episodes.jsonl"
    rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
    completed = {r["key"] for r in rows}
    if len(completed) != len(rows) or not completed <= by_key.keys():
        raise RuntimeError("invalid baseline ledger")
    for row in rows:
        if any(row[k] != by_key[row["key"]][k] for k in ("seed", "target_count", "layout_fingerprint")):
            raise RuntimeError("baseline ledger no longer matches paired plan")
    markers = root / "attempts"
    markers.mkdir(exist_ok=True)
    with (root / "batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_summary(root, rows)
        for spec in plan:
            if spec["key"] in completed:
                continue
            if _source_fingerprints(PROJECT) != identity["sources_sha256"]:
                raise RuntimeError("simulation source changed during baseline")
            if (hashlib.sha256((act_root / "episode_plan.json").read_bytes()).hexdigest() != identity["plan_sha256"]
                    or hashlib.sha256(config.read_bytes()).hexdigest() != identity["config_sha256"]
                    or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != identity["runner_sha256"]):
                raise RuntimeError("baseline frozen input or runner changed")
            marker = markers / f"seed_{spec['seed']}.json"
            # An interrupted attempt is not a free second opportunity.
            if marker.exists():
                raise RuntimeError("unfinished attempt requires a user decision; automatic retry is forbidden")
            with marker.open("x") as stream:
                stream.write(json.dumps({"key": spec["key"], "started_at": time.time()}))
                stream.flush()
                os.fsync(stream.fileno())
            write_json(root / "status.json", {"state": "running", "completed": len(rows),
                "planned": 120, "current": spec["key"], "pid": os.getpid()})
            with inference_lease():
                row = evaluate_trial(cfg, spec)
            with ledger.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            completed.add(spec["key"])
            publish_summary(root, rows)
            print(json.dumps({"event": "astar_trial_finished", "completed": len(rows), "seed": spec["seed"]}), flush=True)
        write_json(root / "status.json", {"state": "complete", "completed": len(rows), "planned": 120})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-batch", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wait-act", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.act_batch).resolve()
    try:
        if args.wait_act:
            while not act_complete(root):
                write_json(Path(args.out) / "status.json", {"state": "waiting_for_act",
                    "act_batch": str(root), "pid": os.getpid()})
                print(json.dumps({"event": "waiting_for_act_batch"}), flush=True)
                time.sleep(30)
        run_batch(root, Path(args.out))
    except BaseException as exc:
        destination = Path(args.out)
        write_json(destination / "status.json", {
            "state": "needs_attention", "error": f"{type(exc).__name__}: {exc}",
            "automatic_retry_forbidden": True,
        })
        raise


if __name__ == "__main__":
    main()
