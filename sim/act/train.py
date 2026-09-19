"""Train the pinned LeRobot ACT policy on the local MuJoCo dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import time
from pathlib import Path

# Prevent Apple Silicon's native vision/BLAS imports from creating an
# oversized inherited thread pool in the long-running training subprocess.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np


class TrainingStopRequest:
    """Small signal-safe flag used to finish the current optimizer step."""

    def __init__(self):
        self.requested = False

    def __call__(self, _signum, _frame):
        self.requested = True


def _jsonable_stats(stats):
    return {
        feature: {
            name: np.asarray(value).tolist()
            for name, value in feature_stats.items()
        }
        for feature, feature_stats in stats.items()
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured training batch size for a benchmark or run",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable MPS/CUDA float16 autocast (default follows act.use_amp)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a resumable checkpoint every N optimizer steps (0 disables periodic saves)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=None,
        help="Keep only the newest N periodic checkpoints (0 keeps all)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint directory containing training_state.pt",
    )
    parser.add_argument(
        "--trackio-project",
        default=None,
        help="Trackio project name; requires --trackio-space for HF visibility",
    )
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="Hugging Face Space ID used for the live Trackio dashboard",
    )
    parser.add_argument(
        "--trackio-private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Create/use a private Trackio Space (default follows config)",
    )
    parser.add_argument(
        "--trackio-static",
        action="store_true",
        help="Log locally and publish to a public static HF Space via trackio sync",
    )
    parser.add_argument(
        "--no-trackio",
        action="store_true",
        help="Disable remote/local Trackio logging for this run",
    )
    parser.add_argument(
        "--small-sample-gate",
        action="store_true",
        help=(
            "allow the explicit six-episode one-per-target overfit gate; "
            "formal training otherwise requires exactly 120 records"
        ),
    )
    parser.add_argument(
        "--preview-training",
        action="store_true",
        help=(
            "accept the promoted 90-record successful preview set "
            "(15 records per target) instead of the formal 120-record set"
        ),
    )
    return parser


def validate_small_sample_gate_manifest(root: str | Path) -> dict:
    """Validate the deliberately narrow six-episode overfit-gate dataset."""
    root = Path(root)
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    valid = [
        record for record in records
        if int(record.get("schema_version", 0)) == 4
        and int(record.get("action_dim", 0)) == 4
        and int(record.get("state_dim", 0)) == 42
        and str(record.get("episode_kind", "")) == "expert"
        and str(record.get("split", "")) == "train"
        and bool(record.get("success", False))
        and 1 <= int(record.get("frame_count", 0)) <= 500
        and "target_indices" not in record
    ]
    counts = {
        target: sum(int(record.get("target_count", 0)) == target for record in valid)
        for target in range(1, 7)
    }
    if len(records) != 6 or len(valid) != 6 or any(
            value != 1 for value in counts.values()):
        raise ValueError(
            "small-sample gate requires exactly six schema-v4 successful "
            f"experts, one per target; found total={len(records)}, counts={counts}"
        )
    if ({str(record.get("layout_id", "")) for record in valid}
            != {"paired_000"} or len({int(record.get("seed", -1))
                                      for record in valid}) != 1):
        raise ValueError(
            "small-sample gate requires the approved paired_000 same-seed layout"
        )
    return {"total": 6, "per_target": counts}


def tracking_settings(project, space_id, private=True, static=False):
    if (project is None) != (space_id is None):
        raise ValueError("Trackio project and space must be provided together")
    if project is None:
        return None
    if static:
        return {
            "project": str(project),
            "space_id": str(space_id),
            "private": False,
            "static": True,
        }
    return {
        "project": str(project),
        "space_id": str(space_id),
        "private": bool(private),
        "static": False,
    }


def _numeric_metrics(metrics):
    output = {}
    for name, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        output[name] = float(value)
    return output


def _cpu_state(value):
    """Move tensors nested in an optimizer state to CPU before serialization."""
    if hasattr(value, "detach"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _device_state(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_state(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_state(item, device) for item in value)
    return value


def _checkpoint_dir(out_root: Path, step: int) -> Path:
    return out_root / "checkpoints" / f"step_{int(step):06d}"


def _prune_checkpoints(out_root: Path, keep_checkpoints: int) -> None:
    if keep_checkpoints <= 0:
        return
    root = out_root / "checkpoints"
    checkpoints = sorted(
        (path for path in root.glob("step_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for stale in checkpoints[:-keep_checkpoints]:
        shutil.rmtree(stale)


def save_checkpoint(policy, preprocessor, postprocessor, optimizer, out_root: Path,
                    step: int, dataset_root: str, device: str,
                    keep_checkpoints: int = 3,
                    elapsed_seconds: float = 0.0) -> Path:
    """Save model, processors, optimizer and progress in one resumable directory."""
    import torch

    checkpoint_root = _checkpoint_dir(out_root, step)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_root)
    preprocessor.save_pretrained(checkpoint_root, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(checkpoint_root, config_filename="policy_postprocessor.json")
    torch.save({
        "step": int(step),
        "optimizer": _cpu_state(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "dataset": str(dataset_root),
        "device": str(device),
        "schema_version": 4,
        "state_dim": 42,
        "action_dim": 4,
        "elapsed_seconds": float(elapsed_seconds),
    }, checkpoint_root / "training_state.pt")
    (out_root / "latest_checkpoint.txt").write_text(
        str(checkpoint_root.relative_to(out_root)), encoding="utf-8"
    )
    _prune_checkpoints(out_root, keep_checkpoints)
    return checkpoint_root


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torch

    from ..config import load_config
    from .collect_dataset import (
        validate_preview_training_manifest,
        validate_training_manifest,
    )
    from .dataset import ActDataset
    from .policy import build_act_policy, build_act_processors

    cfg = load_config(args.config)
    dataset_root = args.dataset or str(cfg.act.dataset_dir)
    out_root = Path(args.out or str(cfg.act.model_dir))
    out_root.mkdir(parents=True, exist_ok=True)
    if args.preview_training:
        manifest_summary = validate_preview_training_manifest(dataset_root)
    elif args.small_sample_gate:
        manifest_summary = validate_small_sample_gate_manifest(dataset_root)
    else:
        manifest_summary = validate_training_manifest(dataset_root)
    dataset = ActDataset(
        dataset_root,
        chunk_size=int(cfg.act.chunk_size),
        image_stat_samples=int(cfg.act.get("image_stat_samples", 100)),
    )
    if len(dataset) == 0:
        raise ValueError("ACT training dataset has no valid behavior-cloning frames")
    dataset_stats = dataset.stats
    batch_size = int(args.batch_size or cfg.act.batch_size)
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    sampling_weights = dataset.phase_sampling_weights(
        float(cfg.act.get("approach_sample_fraction", 0.35))
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sampling_weights, dtype=torch.double),
        num_samples=len(dataset), replacement=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, drop_last=True)
    policy, policy_cfg = build_act_policy(cfg, pretrained_path=args.resume)
    preprocessor, postprocessor = build_act_processors(policy_cfg, dataset_stats)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=float(policy_cfg.optimizer_lr),
        weight_decay=float(policy_cfg.optimizer_weight_decay),
    )
    max_steps = int(args.steps or cfg.act.max_steps)
    device_type = str(policy_cfg.device).split(":", 1)[0]
    amp_enabled = bool(
        cfg.act.get("use_amp", False) if args.amp is None else args.amp
    ) and device_type in {"mps", "cuda"}
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    checkpoint_every = int(
        cfg.act.get("checkpoint_every_steps", 500)
        if args.checkpoint_every is None else args.checkpoint_every
    )
    keep_checkpoints = int(
        cfg.act.get("keep_checkpoints", 3)
        if args.keep_checkpoints is None else args.keep_checkpoints
    )
    if keep_checkpoints < 0:
        raise ValueError("keep-checkpoints must be non-negative")
    start_step = 0
    elapsed_before_resume = 0.0
    if args.resume:
        resume_state_path = Path(args.resume) / "training_state.pt"
        if not resume_state_path.exists():
            raise FileNotFoundError(
                f"checkpoint is missing training_state.pt: {resume_state_path}"
            )
        resume_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        checkpoint_contract = {
            "schema_version": int(resume_state.get("schema_version", 0)),
            "state_dim": int(resume_state.get("state_dim", 0)),
            "action_dim": int(resume_state.get("action_dim", 0)),
        }
        required_contract = {
            "schema_version": 4, "state_dim": 42, "action_dim": 4,
        }
        if checkpoint_contract != required_contract:
            raise ValueError(
                "resume checkpoint is not compatible with schema v4: "
                f"found {checkpoint_contract}, required {required_contract}"
            )
        optimizer.load_state_dict(resume_state["optimizer"])
        for state in optimizer.state.values():
            state.update(_device_state(state, policy_cfg.device))
        start_step = int(resume_state["step"])
        elapsed_before_resume = max(
            0.0, float(resume_state.get("elapsed_seconds", 0.0))
        )
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if "numpy_rng_state" in resume_state:
            np.random.set_state(resume_state["numpy_rng_state"])
        if "python_rng_state" in resume_state:
            random.setstate(resume_state["python_rng_state"])
    tracking_project = (args.trackio_project if args.trackio_project is not None
                        else cfg.act.get("tracking_project"))
    tracking_space = (args.trackio_space if args.trackio_space is not None
                      else cfg.act.get("tracking_space"))
    tracking_private = (bool(cfg.act.get("tracking_private", True))
                        if args.trackio_private is None else bool(args.trackio_private))
    tracking = None if args.no_trackio else tracking_settings(
        tracking_project, tracking_space, tracking_private,
        static=args.trackio_static,
    )
    tracker = None
    if tracking is not None:
        try:
            import trackio
        except ImportError as exc:
            raise RuntimeError(
                "Remote Trackio requested but trackio is not installed; "
                "install the ACT extras first"
            ) from exc
        tracker = trackio
        tracking_init = {key: value for key, value in tracking.items()
                         if key not in {"space_id", "private", "static"}}
        if not tracking["static"]:
            tracking_init.update({
                "space_id": tracking["space_id"],
                "private": tracking["private"],
            })
        tracker.init(
            **tracking_init,
            config={
                "dataset": str(dataset_root),
                "dataset_valid_frames": int(len(dataset)),
                "dataset_manifest": manifest_summary,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "checkpoint_every_steps": checkpoint_every,
                "keep_checkpoints": keep_checkpoints,
                "resume_step": start_step,
                "chunk_size": int(policy_cfg.chunk_size),
                "action_dim": int(policy_cfg.output_features["action"].shape[0]),
                "state_dim": int(
                    policy_cfg.input_features["observation.state"].shape[0]
                ),
                "schema_version": 4,
                "device": str(policy_cfg.device),
            },
        )
    iterator = iter(loader)
    started = time.monotonic()
    time_budget_seconds = float(cfg.act.train_hours) * 3600.0
    if elapsed_before_resume >= time_budget_seconds:
        raise ValueError(
            "resume checkpoint has already exhausted the cumulative training "
            f"time budget ({elapsed_before_resume:.1f} >= {time_budget_seconds:.1f} s)"
        )
    completed_steps = start_step
    stop_request = TrainingStopRequest()
    previous_handlers = {}
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[stop_signal] = signal.getsignal(stop_signal)
        signal.signal(stop_signal, stop_request)
    try:
        for step in range(start_step, max_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            for key in ("observation.images.overhead", "observation.images.wrist"):
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device_type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss, metrics = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite ACT loss at step {step}; disable AMP or inspect the input pipeline"
                )
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            completed_steps = step + 1
            elapsed_seconds = (
                elapsed_before_resume + time.monotonic() - started
            )
            if checkpoint_every > 0 and completed_steps % checkpoint_every == 0:
                checkpoint_root = save_checkpoint(
                    policy, preprocessor, postprocessor, optimizer, out_root,
                    completed_steps, dataset_root, str(policy_cfg.device),
                    keep_checkpoints, elapsed_seconds=elapsed_seconds,
                )
                print(json.dumps({
                    "checkpoint_step": completed_steps,
                    "checkpoint": str(checkpoint_root),
                }))
            if step % 100 == 0:
                record = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    **_numeric_metrics(metrics),
                    "elapsed_s": elapsed_seconds,
                    "device": str(policy_cfg.device),
                    "amp": amp_enabled,
                }
                print(json.dumps(record))
                if tracker is not None:
                    tracker.log({
                        key: value for key, value in record.items()
                        if key != "device"
                    })
            if stop_request.requested:
                break
            if elapsed_seconds >= time_budget_seconds:
                break
    finally:
        if tracker is not None:
            tracker.finish()
        if stop_request.requested and checkpoint_every > 0 and completed_steps > start_step:
            try:
                checkpoint_root = save_checkpoint(
                    policy, preprocessor, postprocessor, optimizer, out_root,
                    completed_steps, dataset_root, str(policy_cfg.device),
                    keep_checkpoints, elapsed_seconds=(
                        elapsed_before_resume + time.monotonic() - started
                    ),
                )
                print(json.dumps({
                    "checkpoint_step": completed_steps,
                    "checkpoint": str(checkpoint_root),
                    "interrupted": True,
                }), flush=True)
            except Exception as exc:  # preserve the original stop path
                print(f"interrupted checkpoint failed: {exc}", flush=True)
        for stop_signal, handler in previous_handlers.items():
            signal.signal(stop_signal, handler)
    policy.save_pretrained(out_root)
    preprocessor.save_pretrained(out_root, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(out_root, config_filename="policy_postprocessor.json")
    (out_root / "dataset_stats.json").write_text(
        json.dumps(_jsonable_stats(dataset_stats), indent=2), encoding="utf-8"
    )
    (out_root / "training_summary.json").write_text(
        json.dumps({
            "steps": completed_steps,
            "interrupted": bool(stop_request.requested),
            "checkpoint_every_steps": checkpoint_every,
            "keep_checkpoints": keep_checkpoints,
            "dataset": str(dataset_root),
            "device": str(policy_cfg.device),
            "amp": amp_enabled,
            "action_dim": int(policy_cfg.output_features["action"].shape[0]),
            "state_dim": int(
                policy_cfg.input_features["observation.state"].shape[0]
            ),
            "schema_version": 4,
            "elapsed_seconds": (
                elapsed_before_resume + time.monotonic() - started
            ),
            "time_budget_seconds": time_budget_seconds,
            "sampling": {
                "approach_descent_fraction": float(
                    cfg.act.get("approach_sample_fraction", 0.35)
                ),
                "contact_build_sweep_fraction": float(
                    1.0 - cfg.act.get("approach_sample_fraction", 0.35)
                ),
            },
            "chunk_size": int(policy_cfg.chunk_size),
            "execute_steps": int(policy_cfg.n_action_steps),
            "preprocessor": "lerobot.make_act_pre_post_processors",
        }, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
