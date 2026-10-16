"""Prepare and run the approved 120-layout sync-5 ACT evaluation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.sync_5step_adapter import SCHEDULE_RECORD, adapt_config


SOURCE_BATCH = Path(
    "/home/luke/luke－projects/UR１０SWEEP/runs/"
    "hybrid_dinov3_200k_eval120_seed2120000_handoff2s_20261016"
)
DEFAULT_OUT = Path(
    "/home/luke/luke－projects/UR１０SWEEP/runs/"
    "hybrid_dinov3_200k_eval120_sync5_seed2120000_20261016"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_source(source: Path, metadata: dict, plan: list[dict]) -> None:
    if metadata.get("state") not in (None, "complete"):
        raise RuntimeError("the source 200K batch is not complete")
    if len(plan) != 120:
        raise RuntimeError("the approved source plan must contain exactly 120 episodes")
    seeds = [spec.get("seed") for spec in plan]
    if len(set(seeds)) != 120 or set(seeds) != set(range(2120000, 2120120)):
        raise RuntimeError("source plan seeds differ from the approved 2120000–2120119 range")
    if {spec.get("checkpoint_step") for spec in plan} != {200000}:
        raise RuntimeError("source plan does not exclusively use the confirmed 200K checkpoint")
    if {spec.get("model_path") for spec in plan} != {
            model["path"] for model in metadata.get("models", []) if model.get("step") == 200000}:
        raise RuntimeError("source plan checkpoint path differs from its frozen metadata")
    if {spec.get("target_count") for spec in plan} != set(range(1, 7)):
        raise RuntimeError("source plan does not cover target counts 1–6")
    if any(sum(spec["target_count"] == target for spec in plan) != 20
           for target in range(1, 7)):
        raise RuntimeError("source plan must retain exactly 20 layouts per target count")
    if len({spec.get("layout_fingerprint") for spec in plan}) != 120:
        raise RuntimeError("source plan has duplicate initial-layout fingerprints")
    for model in metadata.get("models", []):
        if model.get("step") != 200000:
            continue
        for filename, signature in model["checkpoint_files"].items():
            item = Path(model["path"]) / filename
            if not item.is_file() or [item.stat().st_size, item.stat().st_mtime_ns] != signature:
                raise RuntimeError(f"the pinned 200K checkpoint file changed: {item}")
    status_path = source / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("state") != "complete" or status.get("completed") != 120:
        raise RuntimeError("the source batch does not have a complete 120/120 receipt")
    if (source / "completion_receipt.json").is_file():
        receipt = json.loads((source / "completion_receipt.json").read_text())
        if receipt.get("planned") != 120 or receipt.get("completed") != 120:
            raise RuntimeError("the source batch completion receipt is inconsistent")


def _copy_dashboard_assets(source: Path, root: Path) -> None:
    monitor = (source / "live_monitor.py").read_text(encoding="utf-8")
    replacements = (
        ("200K Hybrid A* → ACT · 120 次评测", "200K Sync-5 Hybrid A* → ACT · 120 次评测"),
        ("200K · Hybrid A* → ACT · 冻结 DINOv3", "200K · 同步五步块 Hybrid A* → ACT · 冻结 DINOv3"),
        ("Hybrid A* -> ACT | frozen DINOv3 | step 200K", "DINOv3 ACT | synchronous 5-step chunks | step 200K"),
        ("120 条独立布局评测 · 种子 2120000–2120119 · 目标 1→6 循环 20 轮 · 所有视频保留",
         "同步预测25步、执行前5步 · 120条原布局计划 · 每回合一段视频"),
    )
    for old, new in replacements:
        monitor = monitor.replace(old, new)
    (root / "live_monitor.py").write_text(monitor, encoding="utf-8")


def prepare_batch(root: Path, source: Path, project: Path) -> dict:
    if root.exists():
        raise FileExistsError(f"new evaluation output already exists: {root}")
    old_metadata = json.loads((source / "metadata.json").read_text())
    plan_bytes = (source / "episode_plan.json").read_bytes()
    layout_bytes = (source / "layout_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    validate_frozen_source(source, old_metadata, plan)

    root.mkdir(parents=True, exist_ok=False)
    for child in ("logs", "videos", "replays", "incidents"):
        (root / child).mkdir()
    (root / "episode_plan.json").write_bytes(plan_bytes)
    (root / "layout_plan.json").write_bytes(layout_bytes)

    from sim.act.batch_evaluate import _source_fingerprints
    from sim.config import load_config, save_config

    base_cfg = load_config(str(source / "evaluation_config.yaml"))
    eval_cfg = adapt_config(base_cfg)
    save_config(eval_cfg, str(root / "evaluation_config.yaml"))
    metadata = dict(old_metadata)
    metadata.update({
        "created_at": time.time(),
        "project": str(project.resolve()),
        "source_batch": str(source.resolve()),
        "config_sha256": sha256(root / "evaluation_config.yaml"),
        "source_fingerprints": _source_fingerprints(project),
        "episode_plan_sha256": sha256(root / "episode_plan.json"),
        "checkpoint_selection": "fixed to the source batch's confirmed step_200000 checkpoint",
        "preview": False,
        "schedule_adapter": SCHEDULE_RECORD,
        "schedule_adapter_sha256": sha256(project / "scripts" / "sync_5step_adapter.py"),
        "episode_entrypoint_sha256": sha256(project / "scripts" / "evaluate_sync_5step_episode.py"),
        "batch_entrypoint_sha256": sha256(project / "scripts" / "run_sync_5step_batch.py"),
        "source_plan_sha256": sha256(source / "episode_plan.json"),
        "layout_seed_mode": "independent_per_episode",
        "layout_plan_source": "copied byte-for-byte from the approved completed 120-layout plan",
        "image_retention": "one MP4 per completed physical attempt; retain source replay per existing batch policy",
    })
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from sim.act.batch_evaluate import write_artifacts
    write_artifacts(root, [], len(plan))
    _copy_dashboard_assets(source, root)
    (root / "status.json").write_text(json.dumps({
        "state": "prepared", "completed": 0, "planned": 120,
        "schedule": SCHEDULE_RECORD, "source_batch": str(source.resolve()),
        "updated_at": time.time(),
    }, indent=2), encoding="utf-8")
    return metadata


def _episode_command(root: Path, spec: dict) -> list[str]:
    return [sys.executable, "-u", "-m", "scripts.evaluate_sync_5step_episode",
            "--config", str(root / "evaluation_config.yaml"),
            "--model", spec["model_path"], "--seed", str(spec["seed"]),
            "--target-count", str(spec["target_count"]), "--record-replay",
            "--replay-root", str(root / "replays"),
            "--video-root", str(root / "videos"),
            "--replay-image-format", "webp-lossless"]


def _annotate_physical_row():
    from sim.act import batch_evaluate

    original = batch_evaluate.parse_episode_output

    def parse_with_schedule(*args, **kwargs):
        physical = original(*args, **kwargs)
        physical["act_chunk_schedule"] = SCHEDULE_RECORD
        if physical.get("success") is True:
            physical["failure_type"] = "none"
        elif physical.get("termination_reason") == "no_plan":
            physical["failure_type"] = "no_feasible_astar_plan"
        elif str(physical.get("reason", "")).startswith("known IK execution failure:"):
            physical["failure_type"] = "ik_unreachable"
        elif physical.get("termination_reason") == "exact target count reached":
            physical["failure_type"] = "strict_success_contract_violation"
        else:
            physical["failure_type"] = "policy_or_task_failure"
        return physical

    batch_evaluate.parse_episode_output = parse_with_schedule


def _write_final_report(root: Path):
    rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    errors = [row["peak_collected"] - row["target_count"]
              for row in rows if type(row.get("peak_collected")) is int]
    failure_types = {}
    for row in rows:
        name = row.get("failure_type", "unclassified")
        failure_types[name] = failure_types.get(name, 0) + 1
    report = {
        "protocol": "DINOv3 ACT + synchronous five-step chunk schedule",
        "checkpoint_step": 200000,
        "episodes": len(rows),
        "planned": 120,
        "strict_successes": sum(row.get("success") is True for row in rows),
        "strict_success_rate": (sum(row.get("success") is True for row in rows) / len(rows)
                                if rows else None),
        "peak_count_exact_matches": sum(
            type(row.get("peak_collected")) is int
            and row["peak_collected"] == row["target_count"] for row in rows),
        "final_count_exact_matches": sum(
            type(row.get("final_collected")) is int
            and row["final_collected"] == row["target_count"] for row in rows),
        "peak_count_error_mean_signed": (sum(errors) / len(errors) if errors else None),
        "peak_count_error_mean_absolute": (sum(abs(value) for value in errors) / len(errors)
                                            if errors else None),
        "failure_types": failure_types,
        "density_heatmap": str(root / "success_density.png"),
        "videos_directory": str(root / "videos"),
        "episode_ledger": str(root / "episodes.jsonl"),
        "schedule": SCHEDULE_RECORD,
    }
    (root / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def _assert_runtime_idle():
    processes = subprocess.check_output(["ps", "-Ao", "pid=,comm=,args="], text=True)
    blockers = []
    for line in processes.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, command, args = fields
        if (int(pid) != os.getpid()
                and (command.startswith("python") or command in {"torchrun", "accelerate"})
                and any(token in args for token in (
                    "sim.act.train", "sim.act.evaluate", "run_sync_5step_batch"))):
            blockers.append(line.strip())
    if blockers:
        raise RuntimeError("inference or training processes are still active: " + " | ".join(blockers))
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    if not raw:
        raise RuntimeError("CUDA device status is unavailable")
    utilization, memory = (int(value.strip()) for value in raw[0].split(",", 1))
    if utilization >= 5 or memory >= 1024:
        raise RuntimeError(f"inference GPU is not idle: utilization={utilization}% memory={memory} MiB")


def _start_dashboard(root: Path, port: int):
    output = (root / "dashboard.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "live_monitor.py"), "--root", str(root),
         "--port", str(port)],
        cwd=root, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"live density dashboard exited early; see {root / 'dashboard.log'}")
        if "Live batch density" in (root / "dashboard.log").read_text(encoding="utf-8", errors="replace"):
            return process
        time.sleep(0.2)
    raise TimeoutError(f"live density dashboard did not start; see {root / 'dashboard.log'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-batch", type=Path, default=SOURCE_BATCH)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.out.resolve()
    source = args.source_batch.resolve()
    project = Path(__file__).resolve().parents[1]
    if root.exists():
        metadata = json.loads((root / "metadata.json").read_text())
        if (metadata.get("source_batch") != str(source)
                or metadata.get("schedule_adapter") != SCHEDULE_RECORD
                or sha256(root / "episode_plan.json") != metadata.get("episode_plan_sha256")
                or (root / "episode_plan.json").read_bytes() != (source / "episode_plan.json").read_bytes()
                or sha256(root / "evaluation_config.yaml") != metadata.get("config_sha256")
                or sha256(project / "scripts" / "sync_5step_adapter.py") !=
                    metadata.get("schedule_adapter_sha256")
                or sha256(project / "scripts" / "evaluate_sync_5step_episode.py") !=
                    metadata.get("episode_entrypoint_sha256")
                or sha256(project / "scripts" / "run_sync_5step_batch.py") !=
                    metadata.get("batch_entrypoint_sha256")
                or metadata.get("source_fingerprints") !=
                    __import__("sim.act.batch_evaluate", fromlist=["_source_fingerprints"])
                    ._source_fingerprints(project)):
            raise RuntimeError("prepared batch no longer matches its frozen plan, config, or source")
        validate_frozen_source(
            source, json.loads((source / "metadata.json").read_text()),
            json.loads((source / "episode_plan.json").read_text()))
    else:
        metadata = prepare_batch(root, source, project)
    print(json.dumps({"event": "sync5_batch_prepared", "root": str(root),
                      "planned": metadata["planned"], "schedule": SCHEDULE_RECORD,
                      "source_batch": str(source)}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return 0

    _assert_runtime_idle()
    dashboard = _start_dashboard(root, args.port)
    print(json.dumps({"event": "dashboard_ready", "url": f"http://127.0.0.1:{args.port}/",
                      "pid": dashboard.pid}), flush=True)
    os.environ["SWEEP_SYNC_EVAL_ROOT"] = str(root)
    from sim.act import batch_evaluate

    batch_evaluate.build_episode_command = _episode_command
    _annotate_physical_row()
    batch_evaluate.run_batch(root, metadata, pipeline_media=True)
    report = _write_final_report(root)
    print(json.dumps({"event": "sync5_batch_complete", **report},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
