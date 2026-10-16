"""Resumable evaluation with optional bounded CPU media overlap."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_models(model_root: str | Path) -> list[dict]:
    """Snapshot ready checkpoints once, never chase the trainer's latest pointer."""
    root = Path(model_root).resolve()
    try:
        pointer = root / (root / "latest_checkpoint.txt").read_text().strip()
        published_step = int(pointer.name.removeprefix("step_"))
        if pointer.resolve().parent != root / "checkpoints" or not pointer.is_dir():
            published_step = 0
    except (OSError, ValueError):
        published_step = 0
    models = []
    for step in (100000, 200000):
        path = root / "checkpoints" / f"step_{step:06d}"
        required = ["model.safetensors", "config.json", "training_state.pt",
                    "policy_preprocessor.json", "policy_postprocessor.json"]
        ready = published_step >= step
        try:
            for processor in ("policy_preprocessor.json", "policy_postprocessor.json"):
                document = json.loads((path / processor).read_text())
                for entry in document["steps"]:
                    if entry.get("state_file"):
                        name = entry["state_file"]
                        if (path / name).resolve().parent != path:
                            raise ValueError("processor state is outside checkpoint")
                        required.append(name)
            ready = ready and all((path / name).is_file() and (path / name).stat().st_size > 0
                                  for name in required)
        except (OSError, ValueError, KeyError, TypeError):
            ready = False
        if ready:
            saved_type = json.loads((path / 'config.json').read_text()).get('type')
            prefix = 'hybrid_dinov3' if saved_type == 'act_dinov3' else 'five_view'
            models.append({"label": f"{prefix}_{step // 1000}k", "step": step,
                           "path": str(path),
                           "weights_sha256": _sha256(path / "model.safetensors"),
                           "checkpoint_files": {
                               p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                               for p in sorted(path.iterdir()) if p.is_file()}})
        elif step == 100000:
            raise FileNotFoundError("the requested 100k checkpoint is not ready")
    return models


def make_episode_plan(layouts: list[dict], models: list[dict]) -> list[dict]:
    """Assign one unique layout to each test, with 20 tests per goal/model."""
    if not models or len(layouts) != 120 * len(models):
        raise ValueError("independent evaluation requires 120 layouts per checkpoint")
    for field in ("seed", "layout_fingerprint", "layout_index"):
        if len({layout[field] for layout in layouts}) != len(layouts):
            raise ValueError(f"independent evaluation requires unique {field} values")
    plan = []
    for index, layout in enumerate(layouts):
        target = (index // len(models)) % 6 + 1
        model = models[index % len(models)]
        plan.append({**layout, "target_count": target,
                     "model_label": model["label"],
                     "model_path": model.get("path", ""),
                     "checkpoint_step": model.get("step"),
                     "key": f"{model['label']}:n{target}:l{layout['layout_index']:02d}"})
    return plan


def build_episode_command(root: Path, spec: dict) -> list[str]:
    return [sys.executable, "-u", "-m", "sim.act.evaluate", "--config",
            str(root / "evaluation_config.yaml"), "--model", spec["model_path"],
            "--seed", str(spec["seed"]), "--target-count", str(spec["target_count"]),
            "--preview", "--record-replay", "--replay-root", str(root / "replays"),
            "--video-root", str(root / "videos"), "--replay-image-format", "webp-lossless"]


def unexecuted_hybrid_no_plan(physical: dict) -> bool:
    """Only a measured, unexecuted Hybrid planning failure needs no media."""
    return (physical.get('inference_strategy') == 'hybrid_astar_act_trial4_dinov3_frozen'
            and physical.get('media_not_applicable_no_plan') is True
            and physical.get('termination_reason') == 'no_plan'
            and physical.get('success') is False
            and type(physical.get('physical_attempts')) is int and physical['physical_attempts'] == 0
            and type(physical.get('planning_calls')) is int and physical['planning_calls'] == 1
            and not physical.get('replay_error') and not physical.get('video_error'))


def effective_replay_config(cfg, physical):
    """Reapply the exact inference overlay when replaying an already measured result."""
    if physical.get('inference_strategy') != 'hybrid_astar_act_trial4_dinov3_frozen':
        return cfg
    from .hybrid_trial4 import configure_admittance, formal, runtime
    effective = formal.candidate_config(cfg, int(cfg.task.target_count))
    profile = configure_admittance(runtime(), effective)
    if physical.get('hybrid_control_profile') != profile:
        raise RuntimeError('measured Hybrid control profile differs from the replay implementation')
    return effective


def parse_episode_output(output: str, returncode: int, *, require_media=True) -> dict:
    result = None
    for line in reversed(output.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and {"success", "target_count", "peak_collected",
                "final_collected", "model_checkpoint"} <= candidate.keys():
            result = candidate
            break
    if result is None or returncode not in (0, 1):
        raise RuntimeError("evaluator exited without a valid physical result")
    if require_media and not unexecuted_hybrid_no_plan(result) and (not result.get("video_path") or result.get("video_error")
                          or not result.get("replay_episode_id") or result.get("replay_error")):
        raise RuntimeError("physical result exists but video/replay export did not finish")
    return result


def summarize_rows(rows: list[dict]) -> dict:
    models = {}
    for label in dict.fromkeys(row["model_label"] for row in rows):
        selected = [row for row in rows if row["model_label"] == label]
        peak, final = np.zeros((7, 6), dtype=int), np.zeros((7, 6), dtype=int)
        by_target = {}
        for target in range(1, 7):
            group = [row for row in selected if row["target_count"] == target]
            for row in group:
                for values, field in ((peak, "peak_collected"), (final, "final_collected")):
                    actual = int(row[field])
                    if not 0 <= actual <= 6:
                        raise ValueError("collection count is outside 0..6")
                    values[actual, target - 1] += 1
            successes = sum(bool(row["success"]) for row in group)
            by_target[str(target)] = {"episodes": len(group), "successes": successes,
                                      "success_rate": successes / len(group) if group else None}
        models[label] = {"episodes": len(selected),
                         "successes": sum(bool(row["success"]) for row in selected),
                         "exact_peak_matches": sum(int(row["peak_collected"]) == row["target_count"]
                                                   for row in selected),
                         "peak_count_matrix": peak.tolist(), "final_count_matrix": final.tolist(),
                         "by_target": by_target}
    return models


def write_artifacts(root: Path, rows: list[dict], planned: int) -> None:
    models = summarize_rows(rows)
    summary = {"completed": len(rows), "planned": planned,
               "complete": len(rows) == planned, "models": models,
               "density_scoring": "peak simultaneous full-entry count; strict success is separate",
               "updated_at": time.time()}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "density_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "target", "actual", "peak_frequency", "final_frequency", "tested"])
        for label, stats in models.items():
            for target in range(1, 7):
                for actual in range(7):
                    writer.writerow([label, target, actual,
                                     stats["peak_count_matrix"][actual][target - 1],
                                     stats["final_count_matrix"][actual][target - 1],
                                     stats["by_target"][str(target)]["episodes"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for field, filename, ylabel in (
            ("peak_count_matrix", "count_density.png", "Peak simultaneous fully collected count"),
            ("final_count_matrix", "final_count_density.png", "Final retained fully collected count")):
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6.5), squeeze=False)
        for axis, (label, stats) in zip(axes[0], models.items()):
            counts = np.array(stats[field], dtype=float)
            totals = counts.sum(axis=0)
            percentages = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
            plot = axis.imshow(percentages, origin="lower", cmap="magma", vmin=0, vmax=1,
                               aspect="auto", extent=(0.5, 6.5, -0.5, 6.5))
            for target in range(1, 7):
                for actual in range(7):
                    value = percentages[actual, target - 1]
                    text = f"{value:.0%}" if totals[target - 1] else "--"
                    axis.text(target, actual, text, ha="center", va="center",
                              color="black" if value > 0.65 else "white", fontsize=11)
                axis.add_patch(Rectangle((target - 0.5, target - 0.5), 1, 1,
                                         fill=False, edgecolor="#ddf4fa", linewidth=2))
            axis.set(xticks=range(1, 7), yticks=range(7), xlabel="Target count", ylabel=ylabel,
                     title=f"{label} | strict successes {stats['successes']}/{stats['episodes']}\n"
                           f"Exact peak-count matches {stats['exact_peak_matches']}/{stats['episodes']}")
            fig.colorbar(plot, ax=axis, label="Frequency within each tested target", format="%.1f")
        status = "FINAL" if len(rows) == planned else "PROVISIONAL"
        title = ('Hybrid A*→ACT + frozen DINOv3' if all(label.startswith('hybrid_dinov3_') for label in models)
                 else 'Five-view ACT')
        fig.suptitle(f"{title} count density | {status} {len(rows)}/{planned}", fontsize=16)
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)


def _source_fingerprints(project: Path) -> dict:
    return {str(p.relative_to(project)): _sha256(p) for p in sorted((project / "sim").rglob("*.py"))}


def validate_media_revision(root: Path, metadata: dict, revision: dict) -> dict:
    """Accept an explicit, audited media repair without rewriting frozen truth."""
    before, after = revision.get("before_sources", {}), revision.get("after_sources", {})
    allowed = {"sim/act/evaluate.py", "sim/act/replay.py", "sim/act/batch_evaluate.py"}
    changed = {name for name in before if before[name] != after.get(name)}
    if (revision.get("kind") != "recorded_state_replay_v1" or not revision.get("name")
            or revision.get("metadata_sha256") != _sha256(root / "metadata.json")
            or before != metadata["source_fingerprints"] or set(before) != set(after)
            or not changed or not changed <= allowed):
        raise RuntimeError("media revision does not match the frozen source/metadata contract")
    return after


def prepare_batch(root: Path, *, project: Path, config: Path, dataset: Path,
                  model_root: Path, seed_start: int) -> dict:
    from ..config import load_config, save_config
    from ..environments.sweep_env import SweepEnv
    models = select_models(model_root)
    cfg = load_config(str(config))
    records = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    train_seeds = {int(row["seed"]) for row in records}
    seen = {row["layout_fingerprint"] for row in records}
    layouts = []
    for index, seed in enumerate(range(seed_start, seed_start + 120 * len(models))):
        if seed in train_seeds:
            raise ValueError("requested held-out seed occurs in the training dataset")
        episode_cfg = cfg.copy()
        episode_cfg.set_path("task.target_count", (index // len(models)) % 6 + 1)
        env = SweepEnv(episode_cfg, seed=seed)
        try:
            env.reset(seed=seed)
            poses = np.concatenate((env.component_positions(), env.component_quats()), axis=1).astype("<f4")
            fingerprint = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
        finally:
            env.close()
        if fingerprint in seen:
            raise ValueError("evaluation layout is not unique or held out from training")
        seen.add(fingerprint)
        layouts.append({"layout_index": index, "seed": seed, "layout_fingerprint": fingerprint,
                        "initial_object_poses": poses.tolist()})
    root.mkdir(parents=True, exist_ok=False)
    for child in ("logs", "videos", "replays"):
        (root / child).mkdir()
    save_config(cfg, str(root / "evaluation_config.yaml"))
    plan = make_episode_plan(layouts, models)
    metadata = {"created_at": time.time(), "project": str(project), "dataset": str(dataset),
                "training_episode_count": len(records), "models": models,
                "checkpoint_selection": "fixed at batch preparation; 200k included only if already ready",
                "config_sha256": _sha256(root / "evaluation_config.yaml"),
                "dataset_manifest_sha256": _sha256(dataset / "manifest.jsonl"),
                "source_fingerprints": _source_fingerprints(project),
                "planned": len(plan), "layouts": len(layouts), "targets": list(range(1, 7)),
                "layout_seed_mode": "independent_per_episode", "episodes_per_target_per_model": 20,
                "preview": True, "counting_mode": "peak_simultaneous_full_entry",
                "max_frames": 2000, "first_entry_window_seconds": 2.0,
                "image_retention": "delete only after complete MP4 verification; keep telemetry and causal trace"}
    (root / "episode_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (root / "layout_plan.json").write_text(json.dumps({"test": layouts}, indent=2), encoding="utf-8")
    metadata["episode_plan_sha256"] = _sha256(root / "episode_plan.json")
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_artifacts(root, [], len(plan))
    return metadata


def _write_status(root: Path, **values):
    values.update({"pid": os.getpid(), "updated_at": time.time()})
    temporary = root / "status.json.tmp"
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    os.replace(temporary, root / "status.json")


def recover_episode_media(root: Path, physical: dict) -> None:
    from .video import export_inference_video
    from .media_retention import finalize_inference_video
    episode = root / "replays" / physical["replay_episode_id"]
    manifest = episode.parent / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    record = next(row for row in records if row["episode_id"] == episode.name)
    # Manifest publication can precede the final stdout line. It is the durable
    # source of playback state when the encoder's images have been pruned.
    candidate = record.get("video_path") or physical.get("video_path")
    video = Path(candidate) if candidate else None
    if video is None or not video.is_file():
        if record.get("frame_images_deleted"):
            raise RuntimeError("published video is missing after cleanup; refusing to resample")
        print(json.dumps({"event": "media_phase", "phase": "encoding",
                          "replay_episode_id": episode.name, "at": time.time()}), flush=True)
        video = export_inference_video(episode, root / "videos")
    print(json.dumps({"event": "media_phase", "phase": "validation_and_cleanup",
                      "replay_episode_id": episode.name, "at": time.time()}), flush=True)
    finalize_inference_video(episode, video)
    physical.update(video_path=str(video), video_error="", replay_error="",
                    frame_images_deleted=True, image_cleanup_error="")
    print(json.dumps({"event": "media_phase", "phase": "complete",
                      "replay_episode_id": episode.name, "at": time.time()}), flush=True)


def validate_episode_identity(root: Path, spec: dict, physical: dict, *,
                              allow_legacy_first_frame: bool = False) -> dict:
    """Verify measured replay provenance, never relabel an old log as a new test."""
    if unexecuted_hybrid_no_plan(physical):
        poses = np.asarray(physical.get('initial_object_poses'), dtype='<f4')
        fingerprint = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
        if (poses.shape != (6, 7) or not np.isfinite(poses).all()
                or fingerprint != spec['layout_fingerprint']
                or physical.get('seed') != spec['seed']
                or physical.get('target_count') != spec['target_count']
                or physical.get('model_checkpoint') != spec['model_path']):
            raise RuntimeError('measured no-plan identity differs from the frozen plan')
        return dict(seed=spec['seed'], source='unexecuted_initial_pose_snapshot',
                    measured_pose_fingerprint=fingerprint,
                    planned_layout_fingerprint=spec['layout_fingerprint'])
    records = [json.loads(line) for line in (root / "replays" / "manifest.jsonl").read_text().splitlines()
               if line.strip()]
    matches = [record for record in records if record["episode_id"] == physical.get("replay_episode_id")]
    if len(matches) != 1:
        raise RuntimeError("measured replay identity is missing or ambiguous")
    record = matches[0]
    if any(record.get(measured) != spec[planned] for measured, planned in (
            ("seed", "seed"), ("target_count", "target_count"), ("model", "model_path"))):
        raise RuntimeError("measured replay identity does not match seed/checkpoint/goal")
    if "initial_object_poses" in record:
        poses = np.asarray(record["initial_object_poses"], dtype="<f4")
        fingerprint = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
        if fingerprint != record.get("layout_fingerprint") or fingerprint != spec["layout_fingerprint"]:
            raise RuntimeError("measured replay identity does not match the initial layout")
        source = "initial_replay_pose_snapshot"
    else:
        if not allow_legacy_first_frame:
            raise RuntimeError("measured replay identity requires an initial pose snapshot")
        # The first already-measured rollout predates pose snapshots. Its seed
        # must still match exactly; accept only its actual first replay frame,
        # within 5e-4 per position/quaternion scalar of the reset pose. Record
        # the observed frame hash separately, not as an invented initial hash.
        from ..config import load_config
        cfg = load_config(str(root / "evaluation_config.yaml"))
        arrays = (root / "replays" / record["arrays"]).resolve()
        episode = (root / "replays" / record["episode_id"]).resolve()
        if episode.parent != (root / "replays").resolve() or arrays.parent != episode:
            raise RuntimeError("measured replay identity arrays are outside the episode")
        with np.load(arrays, allow_pickle=False) as values:
            poses = np.asarray(values["object_pose"][0], dtype="<f4")
            first_time = float(values["t"][0])
        planned = np.asarray(spec["initial_object_poses"], dtype="<f4")
        settle_time = float(cfg.episode.settle_time)
        first_frame_deadline = settle_time + 1.0 / float(cfg.act.action_hz)
        if (not np.isfinite(first_time) or first_time < settle_time - 1e-6 or
                first_time > first_frame_deadline + 1e-6 or poses.shape != planned.shape or
                not np.allclose(poses, planned, rtol=0.0, atol=0.0005)):
            raise RuntimeError("measured replay identity lacks a matching initial frame")
        fingerprint = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
        source = "legacy_first_replay_frame"
    return {"seed": record["seed"], "source": source,
            "measured_pose_fingerprint": fingerprint,
            "planned_layout_fingerprint": spec["layout_fingerprint"]}


def run_batch(root: Path, metadata: dict, *, pipeline_media: bool = False,
              media_revision: dict | None = None) -> None:
    project = Path(metadata["project"])
    if media_revision is not None:
        validate_media_revision(root, metadata, media_revision)
    if (metadata.get("layout_seed_mode") != "independent_per_episode" or
            not metadata.get("episode_plan_sha256")):
        raise RuntimeError("independent sampling protocol is required; legacy batches cannot resume")
    if _sha256(root / "episode_plan.json") != metadata["episode_plan_sha256"]:
        raise RuntimeError("frozen episode plan changed")
    plan = json.loads((root / "episode_plan.json").read_text())
    for field in ("seed", "layout_fingerprint", "key"):
        if len({spec[field] for spec in plan}) != len(plan):
            raise RuntimeError(f"independent sampling requires unique {field} values")
    labels = {spec["model_label"] for spec in plan}
    if not labels or len(plan) != 120 * len(labels) or any(
            sum(spec["model_label"] == label and spec["target_count"] == target
                for spec in plan) != 20 for label in labels for target in range(1, 7)):
        raise RuntimeError("independent sampling requires 20 tests per target/checkpoint")
    results = root / "episodes.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()] if results.exists() else []
    completed = {row["key"] for row in rows}
    if len(completed) != len(rows):
        raise RuntimeError("duplicate scored episodes in batch results")
    if not completed <= {spec["key"] for spec in plan}:
        raise RuntimeError("ledger contains an episode outside the frozen plan")
    by_key = {spec["key"]: spec for spec in plan}
    for row in rows:
        if any(row.get(field) != by_key[row["key"]][field] for field in (
                "seed", "layout_fingerprint", "target_count", "model_label", "model_path")):
            raise RuntimeError("scored ledger does not match the frozen plan")
    with (root / "batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        media_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-media") if pipeline_media else None
        pending = None

        def commit_episode(spec, physical, identity):
            # A durable append already succeeded even if derived plots/CSV
            # failed afterward. Draining the same pending future is idempotent.
            if spec["key"] in completed:
                return
            if not unexecuted_hybrid_no_plan(physical) and (not physical.get("video_path") or physical.get("video_error") or
                    not physical.get("frame_images_deleted") or physical.get("image_cleanup_error")):
                raise RuntimeError("video export or intermediate image cleanup needs attention")
            row = {**spec, **physical, "layout_identity_verification": identity}
            revision_name = (media_revision["name"] if media_revision is not None
                             else metadata.get("active_execution_revision"))
            if revision_name:
                row["execution_revision"] = revision_name
            row.pop("initial_object_poses", None)
            with results.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            completed.add(spec["key"])
            write_artifacts(root, rows, len(plan))
            print(json.dumps({"event": "episode_finished", "completed": len(rows), "planned": len(plan),
                              "key": spec["key"], "success": physical["success"],
                              "peak_collected": physical["peak_collected"],
                              "final_collected": physical["final_collected"],
                              "video_path": physical["video_path"]}), flush=True)

        def finish_pending(current=None):
            nonlocal pending
            if pending is None:
                return
            future, previous_spec, previous_physical, previous_identity = pending
            _write_status(root, state="finishing_media", completed=len(rows), planned=len(plan),
                          current=current, media_pending=previous_spec["key"])
            future.result()
            commit_episode(previous_spec, previous_physical, previous_identity)
            pending = None

        try:
            for spec in plan:
                if spec["key"] in completed:
                    previous = next(row for row in rows if row['key'] == spec['key'])
                    if not unexecuted_hybrid_no_plan(previous) and not Path(previous['video_path']).is_file():
                        raise RuntimeError("a previously completed batch video is missing")
                    continue
                if pending is not None and pending[0].done():
                    finish_pending(spec["key"])
                expected_sources = (validate_media_revision(root, metadata, media_revision)
                                    if media_revision is not None else metadata["source_fingerprints"])
                if _source_fingerprints(project) != expected_sources:
                    raise RuntimeError("inference source changed; stop rather than mix evaluation protocols")
                if _sha256(root / "evaluation_config.yaml") != metadata["config_sha256"]:
                    raise RuntimeError("frozen evaluation config changed")
                for model in metadata["models"]:
                    for name, identity in model["checkpoint_files"].items():
                        item = Path(model["path"]) / name
                        if not item.is_file() or [item.stat().st_size, item.stat().st_mtime_ns] != identity:
                            raise RuntimeError("frozen checkpoint files changed")
                while True:
                    processes = subprocess.check_output(["ps", "-Ao", "pid=,args="], text=True)
                    active = [line.strip() for line in processes.splitlines()
                              if " -m sim.act.evaluate " in line]
                    if not active:
                        break
                    _write_status(root, state="waiting_for_manual_inference", completed=len(rows),
                                  planned=len(plan), current=spec["key"], active=active)
                    time.sleep(15)
                if shutil.disk_usage(root).free < 4 * 1024**3:
                    raise RuntimeError("less than 4 GiB free; preserve training and existing artifacts")
                log = root / "logs" / (spec["key"].replace(":", "_") + ".log")
                recovering = log.exists()
                if recovering:
                    # Recovery may finish export, never repeat a measured rollout
                    # just because encoding failed (which would resample its outcome).
                    physical = parse_episode_output(log.read_text(), 1, require_media=False)
                    if not physical.get("replay_episode_id") and not unexecuted_hybrid_no_plan(physical):
                        saved = physical.get("replay_input_path")
                        if not saved:
                            raise RuntimeError("unfinished replay needs attention; refusing to resample")
                        saved = Path(saved).resolve()
                        if saved.parent != (root / "replays" / "replay_inputs").resolve():
                            raise RuntimeError("saved replay input is outside this batch")
                        from ..config import load_config
                        from .replay import load_replay_input, recover_replay_input
                        cfg = load_config(str(root / "evaluation_config.yaml"))
                        cfg.set_path("task.target_count", spec["target_count"])
                        cfg = effective_replay_config(cfg, physical)
                        measured = load_replay_input(saved, cfg, seed=spec["seed"], model=spec["model_path"])
                        if any(getattr(measured, name, None) != physical[name] for name in (
                                "success", "target_count", "peak_collected", "final_collected")):
                            raise RuntimeError("saved replay input physical outcome differs from measured log")
                        physical.update(replay_episode_id=recover_replay_input(
                            saved, cfg, seed=spec["seed"], model=spec["model_path"]), replay_error="")
                        with log.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({**physical, "event": "replay_recovered"}) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                else:
                    environment = os.environ.copy()
                    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                       VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1",
                                       PYTHONUNBUFFERED="1")
                    with log.open("w", encoding="utf-8") as stream:
                        command = build_episode_command(root, spec)
                        if pipeline_media:
                            command.append("--defer-video")
                        child = subprocess.Popen(command, cwd=project,
                                                 env=environment, stdout=stream, stderr=subprocess.STDOUT)
                        _write_status(root, state="running", completed=len(rows), planned=len(plan),
                                      current=spec["key"], child_pid=child.pid, log_path=str(log),
                                      media_pending=pending[1]["key"] if pending else None)
                        print(json.dumps({"event": "episode_started", "completed": len(rows),
                                          "planned": len(plan), "key": spec["key"], "pid": child.pid}), flush=True)
                        code = child.wait()
                    physical = parse_episode_output(log.read_text(), code, require_media=not pipeline_media)
                if not unexecuted_hybrid_no_plan(physical) and (not physical.get("replay_episode_id") or physical.get("replay_error")):
                    raise RuntimeError("unfinished replay needs attention; refusing to resample")
                identity = validate_episode_identity(root, spec, physical,
                    allow_legacy_first_frame=physical.get("replay_episode_id") in
                    metadata.get("legacy_identity_episode_ids", []))
                if physical["model_checkpoint"] != spec["model_path"] or physical["target_count"] != spec["target_count"]:
                    raise RuntimeError("evaluated checkpoint or target does not match the frozen plan")
                if unexecuted_hybrid_no_plan(physical):
                    finish_pending(spec['key'])
                    commit_episode(spec, physical, identity)
                elif pipeline_media:
                    # At most one export plus one physical episode. Do not queue
                    # a second image set while an earlier encoder is unfinished.
                    finish_pending(spec["key"])
                    pending = (media_pool.submit(recover_episode_media, root, physical),
                               spec, physical, identity)
                    print(json.dumps({"event": "media_started", "key": spec["key"],
                                      "replay_episode_id": physical["replay_episode_id"]}), flush=True)
                else:
                    if recovering:
                        recover_episode_media(root, physical)
                    commit_episode(spec, physical, identity)
                _write_status(root, state="running", completed=len(rows), planned=len(plan), current=None,
                              media_pending=pending[1]["key"] if pending else None)
            finish_pending()
            write_artifacts(root, rows, len(plan))
            _write_status(root, state="complete", completed=len(rows), planned=len(plan), current=None)
        except BaseException as exc:
            # A later replay failure must not discard the previous already
            # measured/exported episode. Never score the broken current one.
            if pending is not None:
                try:
                    finish_pending()
                except BaseException as media_exc:
                    print(json.dumps({"event": "pending_media_failed",
                                      "error": f"{type(media_exc).__name__}: {media_exc}"}), flush=True)
            _write_status(root, state="needs_attention", completed=len(rows), planned=len(plan),
                          current=spec["key"] if "spec" in locals() else None,
                          media_pending=pending[1]["key"] if pending else None,
                          error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if media_pool is not None:
                media_pool.shutdown(wait=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="runs/act_dataset_five_view_rudra_300_smooth320_v1")
    parser.add_argument("--model-root", default="runs/act_model_five_view_rudra_300_smooth320_v1")
    parser.add_argument("--seed-start", type=int, default=2110000,
                        help="first of the unique consecutive held-out episode seeds")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pipeline-media", action="store_true",
                        help="overlap one CPU video export with the next single inference")
    parser.add_argument("--media-revision", default=None,
                        help="explicit audited replay repair; frozen metadata remains unchanged")
    args = parser.parse_args(argv)
    root, project = Path(args.out).resolve(), Path(__file__).resolve().parents[2]
    if (root / "metadata.json").exists():
        metadata = json.loads((root / "metadata.json").read_text())
    else:
        metadata = prepare_batch(root, project=project, config=Path(args.config).resolve(),
                                 dataset=Path(args.dataset).resolve(), model_root=Path(args.model_root).resolve(),
                                 seed_start=args.seed_start)
    print(json.dumps({"event": "batch_ready", "root": str(root),
                      "planned": metadata["planned"], "models": [m["label"] for m in metadata["models"]]}), flush=True)
    if not args.prepare_only:
        revision = None
        if args.media_revision:
            path = Path(args.media_revision).resolve()
            if root not in path.parents:
                raise ValueError("media revision must be stored inside the batch evidence directory")
            revision = json.loads(path.read_text())
        run_batch(root, metadata, pipeline_media=args.pipeline_media, media_revision=revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
