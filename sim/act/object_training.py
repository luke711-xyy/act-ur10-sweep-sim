"""v5 training utilities: modality-aware statistics and resumable batches."""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


NORMALIZER_VERSION = 1
FLOAT_FEATURES = (
    "observation.robot_state",
    "observation.task_state",
    "observation.object_tokens",
    "observation.bev",
    "action",
)
IMAGE_FEATURES = (
    "observation.images.overhead",
    "observation.images.wrist",
)


def _array_stats(
    values: np.ndarray,
    *,
    channel_axis: int | None = None,
    reduce_axes: tuple[int, ...] | None = None,
) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot compute statistics from an empty array")
    if channel_axis is not None:
        values = np.moveaxis(values, int(channel_axis), -1)
    if reduce_axes is None:
        reduce_axes = tuple(range(values.ndim - 1))
    mean = np.mean(values, axis=reduce_axes)
    std = np.std(values, axis=reduce_axes)
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "min": np.min(values, axis=reduce_axes).tolist(),
        "max": np.max(values, axis=reduce_axes).tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [int(values.shape[0])],
    }


class ObjectACTPreprocessor:
    """LeRobot-style MEAN_STD preprocessing for the custom v5 keys."""

    def __init__(self, stats: Mapping[str, Mapping[str, object]], *, device: str = "cpu"):
        self.stats = {
            feature: {
                name: np.asarray(value, dtype=np.float32)
                for name, value in feature_stats.items()
            }
            for feature, feature_stats in stats.items()
        }
        self.device = str(device)
        required = set(FLOAT_FEATURES + IMAGE_FEATURES)
        missing = sorted(required - set(self.stats))
        if missing:
            raise ValueError(f"v5 normalization stats are missing {missing}")

    def _normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        stats = self.stats[key]
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=value.device)
        std = torch.as_tensor(stats["std"], dtype=torch.float32, device=value.device).clamp_min(1e-8)
        return (value.to(dtype=torch.float32) - mean) / std

    def unnormalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Invert the training normalizer for policy outputs."""
        if key not in self.stats:
            raise KeyError(f"no normalization statistics for {key!r}")
        stats = self.stats[key]
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=value.device)
        std = torch.as_tensor(stats["std"], dtype=torch.float32, device=value.device).clamp_min(1e-8)
        return value.to(dtype=torch.float32) * std + mean

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            tensor = value.to(self.device) if hasattr(value, "to") else torch.as_tensor(value, device=self.device)
            if key == "observation.task_state" and tensor.ndim >= 2:
                # The v5 task state stores target_count / 6.  Preserve the
                # physical cardinality for the permutation-invariant
                # selector before normalizing the full task-state history.
                result["objectact.target_count"] = torch.round(
                    tensor[:, 1] * 6.0
                ).to(dtype=torch.long).clamp(min=1, max=6).detach().clone()
            if key in IMAGE_FEATURES:
                tensor = tensor.to(dtype=torch.float32)
                if value.dtype == torch.uint8 or float(tensor.detach().amax().item()) > 1.5:
                    tensor = tensor / 255.0
                result[key] = self._normalize(key, tensor)
            elif key in FLOAT_FEATURES:
                result[key] = self._normalize(key, tensor)
            else:
                result[key] = tensor
        return result

    def save(self, path: str | Path) -> None:
        payload = {
            "normalizer_version": NORMALIZER_VERSION,
            "stats": {
                feature: {name: value.tolist() for name, value in feature_stats.items()}
                for feature, feature_stats in self.stats.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "ObjectACTPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("normalizer_version", 0)) != NORMALIZER_VERSION:
            raise ValueError("unsupported ObjectACT normalizer version")
        return cls(payload["stats"], device=device)


def compute_object_dataset_stats(dataset, *, max_samples: int | None = None) -> dict[str, dict[str, list]]:
    """Compute stable normalization statistics from v5 samples only."""

    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if limit <= 0:
        raise ValueError("cannot compute v5 statistics from an empty dataset")
    collected: dict[str, list[np.ndarray]] = {key: [] for key in FLOAT_FEATURES + IMAGE_FEATURES}
    for index in range(limit):
        sample = dataset[index]
        for key in collected:
            if key not in sample:
                raise ValueError(f"v5 sample is missing {key}")
            value = np.asarray(sample[key])
            if key in IMAGE_FEATURES:
                value = value.astype(np.float32) / 255.0
            collected[key].append(value)
    stats: dict[str, dict[str, list]] = {}
    for key, values in collected.items():
        stacked = np.stack(values, axis=0)
        if key in IMAGE_FEATURES:
            stats[key] = _array_stats(stacked, channel_axis=1)
            stats[key]["mean"] = np.asarray(stats[key]["mean"], dtype=np.float32).reshape(3, 1, 1).tolist()
            stats[key]["std"] = np.asarray(stats[key]["std"], dtype=np.float32).reshape(3, 1, 1).tolist()
            stats[key]["min"] = np.asarray(stats[key]["min"], dtype=np.float32).reshape(3, 1, 1).tolist()
            stats[key]["max"] = np.asarray(stats[key]["max"], dtype=np.float32).reshape(3, 1, 1).tolist()
        elif key == "observation.bev":
            stats[key] = _array_stats(stacked, channel_axis=1)
            for name in ("mean", "std", "min", "max"):
                stats[key][name] = np.asarray(stats[key][name], dtype=np.float32).reshape(6, 1, 1).tolist()
        elif key == "action":
            stats[key] = _array_stats(stacked, reduce_axes=(0, 1))
        else:
            stats[key] = _array_stats(stacked, reduce_axes=(0,))
    return stats


def _cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def save_objectact_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    preprocessor: ObjectACTPreprocessor,
    out_root: str | Path,
    *,
    step: int,
    config_payload: Mapping[str, object],
    dataset_root: str,
    elapsed_seconds: float = 0.0,
) -> Path:
    """Save a schema-v5 checkpoint that cannot be mistaken for ordinary ACT."""

    root = Path(out_root)
    checkpoint = root / "checkpoints" / f"step_{int(step):06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint / "model.pt")
    torch.save(
        {
            "step": int(step),
            # The selector has a separate teacher-forcing schedule which is
            # intentionally not a torch parameter.  Persist its cursor so a
            # resumed run does not silently restart teacher forcing.
            "selection_step": int(getattr(model, "selection_step", step)),
            "optimizer": _cpu_state(optimizer.state_dict()),
            "schema_version": 5,
            "robot_state_dim": 36,
            "task_state_dim": 6,
            "object_token_dim": 29,
            "object_slots": 6,
            "bev_shape": [6, 128, 160],
            "action_dim": 4,
            "dataset": str(dataset_root),
            "elapsed_seconds": float(elapsed_seconds),
        },
        checkpoint / "training_state.pt",
    )
    (checkpoint / "objectact_config.json").write_text(
        json.dumps(dict(config_payload), indent=2, default=str), encoding="utf-8"
    )
    preprocessor.save(checkpoint / "normalizer.json")
    (root / "latest_checkpoint.txt").write_text(
        str(checkpoint.relative_to(root)), encoding="utf-8"
    )
    return checkpoint


def load_objectact_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict:
    """Read and validate v5 checkpoint metadata before any weights are loaded."""

    checkpoint = Path(path)
    state_path = checkpoint / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    required = {
        "schema_version": 5,
        "robot_state_dim": 36,
        "task_state_dim": 6,
        "object_token_dim": 29,
        "object_slots": 6,
        "bev_shape": [6, 128, 160],
        "action_dim": 4,
    }
    for key, expected in required.items():
        actual = state.get(key)
        matches = list(actual) == expected if isinstance(expected, list) and actual is not None else actual == expected
        if not matches:
            raise ValueError(
                f"ObjectACT checkpoint contract mismatch for {key}: "
                f"found {actual!r}, expected {expected!r}"
            )
    if not (checkpoint / "model.pt").exists():
        raise FileNotFoundError(checkpoint / "model.pt")
    return state


def prune_objectact_checkpoints(
    out_root: str | Path, *, keep_latest: int = 3, milestones: set[int] | None = None
) -> None:
    """Retain milestones plus the newest periodic checkpoints."""

    root = Path(out_root) / "checkpoints"
    if not root.exists():
        return
    milestone_names = {f"step_{int(step):06d}" for step in (milestones or set())}
    candidates = sorted((path for path in root.glob("step_*") if path.is_dir()), key=lambda path: path.name)
    keep_names = milestone_names | {path.name for path in candidates[-max(0, int(keep_latest)):]}
    for path in candidates:
        if path.name not in keep_names:
            import shutil

            shutil.rmtree(path)


def build_objectact_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the schema-v5 ObjectACT-BEV policy")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2_500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--expected-per-target", type=int, default=20)
    parser.add_argument("--stats-samples", type=int, default=4000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-amp", action="store_true")
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
        help="Log locally and publish a public static HF Space via trackio sync",
    )
    parser.add_argument(
        "--no-trackio",
        action="store_true",
        help="Disable remote/local Trackio logging for this run",
    )
    return parser


def _objectact_config_from_sim_config(cfg):
    from .object_policy import ObjectACTConfig

    image_size = tuple(int(value) for value in cfg.act.image_size)
    device = str(cfg.act.get("device", "mps"))
    return ObjectACTConfig(
        image_size=image_size,
        chunk_size=int(cfg.act.get("chunk_size", 25)),
        temporal_ensemble_coeff=float(cfg.act.get("temporal_ensemble_coeff", 0.01)),
        pretrained_backbone_weights=cfg.act.get(
            "objectact_pretrained_weights", "ResNet18_Weights.IMAGENET1K_V1"
        ),
        device=device,
    )


class _TrainingStop:
    def __init__(self):
        self.requested = False

    def __call__(self, _signal, _frame):
        self.requested = True


def _safe_trackio_call(
    tracker,
    method: str,
    *args,
    failures: list[str] | None = None,
    **kwargs,
) -> bool:
    """Keep optional dashboard failures from terminating a training run.

    Trackio logging and static-Space uploads are observability side effects;
    they must never prevent a checkpoint or the local training loop from
    completing when the network is transiently unavailable.
    """

    try:
        getattr(tracker, method)(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the dashboard is best-effort
        message = f"{method}: {type(exc).__name__}: {exc}"
        if failures is not None:
            failures.append(message)
        print(f"WARNING: Trackio {message}; continuing training", flush=True)
        return False
    return True


def train_objectact(args=None) -> dict:
    """Run the resumable v5 behavior-cloning loop.

    This entry point deliberately imports no planner and only consumes the
    v5 manifest.  It is therefore safe to run in a test process where every
    A* entry point is replaced with an exception.
    """

    args = build_objectact_train_parser().parse_args(args) if args is not None else build_objectact_train_parser().parse_args()
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from ..config import load_config
    from .object_dataset import ObjectActDataset, validate_v5_manifest
    from .generate_objectact_dataset import validate_objectact_training_manifest
    from .object_policy import ObjectACTPolicy
    from .train import _numeric_metrics, tracking_settings

    cfg = load_config(args.config)
    dataset_root = Path(args.dataset or str(cfg.act.get("objectact_dataset_dir", cfg.act.dataset_dir)))
    out_root = Path(args.out or str(cfg.act.get("objectact_model_dir", "runs/objectact_model")))
    records = [
        json.loads(line) for line in (dataset_root / "manifest_v5.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if int(args.expected_per_target) == 20:
        manifest_summary = validate_objectact_training_manifest(dataset_root)
    else:
        manifest_summary = validate_v5_manifest(
            records, expected_per_target=int(args.expected_per_target)
        )
    dataset = ObjectActDataset(str(dataset_root), chunk_size=25)
    if len(dataset) == 0:
        raise ValueError("ObjectACT dataset has no valid expert frames")
    stats = compute_object_dataset_stats(dataset, max_samples=int(args.stats_samples))
    preprocessor = ObjectACTPreprocessor(stats, device="cpu")
    weights = dataset.phase_sampling_weights(0.35)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), sampler=sampler, drop_last=True, num_workers=0)
    policy_config = _objectact_config_from_sim_config(cfg)
    policy = ObjectACTPolicy(policy_config)
    device = str(policy.config.device)
    policy.to(device)
    preprocessor.device = device
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    start_step = 0
    elapsed_before = 0.0
    if args.resume:
        state = load_objectact_checkpoint(args.resume, map_location="cpu")
        policy.load_state_dict(torch.load(Path(args.resume) / "model.pt", map_location=device, weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        # Checkpoints written before this field existed are interpreted as if
        # the schedule had advanced with the global training step.
        policy.selection_step = int(state.get("selection_step", start_step))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        normalizer_path = Path(args.resume) / "normalizer.json"
        if normalizer_path.exists():
            preprocessor = ObjectACTPreprocessor.load(normalizer_path, device="cpu")
            preprocessor.device = device

    tracking_project = (
        args.trackio_project
        if args.trackio_project is not None
        else cfg.act.get("tracking_project")
    )
    tracking_space = (
        args.trackio_space
        if args.trackio_space is not None
        else cfg.act.get("tracking_space")
    )
    tracking_private = (
        bool(cfg.act.get("tracking_private", True))
        if args.trackio_private is None
        else bool(args.trackio_private)
    )
    tracking = None if args.no_trackio else tracking_settings(
        tracking_project,
        tracking_space,
        tracking_private,
        static=bool(args.trackio_static),
    )
    tracker = None
    tracking_failures: list[str] = []
    last_static_sync_elapsed = -60.0
    if tracking is not None:
        try:
            import trackio
        except ImportError as exc:
            raise RuntimeError(
                "Remote Trackio requested but trackio is not installed; "
                "install the ACT extras first"
            ) from exc
        tracker = trackio
        tracking_init = {
            key: value
            for key, value in tracking.items()
            if key not in {"space_id", "private", "static"}
        }
        if not tracking["static"]:
            tracking_init.update({
                "space_id": tracking["space_id"],
                "private": tracking["private"],
            })
        tracker.init(
            **tracking_init,
            config={
                "schema_version": 5,
                "dataset": str(dataset_root),
                "dataset_manifest": manifest_summary,
                "dataset_valid_frames": int(len(dataset)),
                "batch_size": int(args.batch_size),
                "max_steps": int(args.steps),
                "checkpoint_every_steps": int(args.checkpoint_every),
                "keep_checkpoints": int(args.keep_checkpoints),
                "resume_step": int(start_step),
                "device": device,
                "action_dim": 4,
                "robot_state_dim": 36,
                "task_state_dim": 6,
                "object_slots": 6,
                "object_token_dim": 29,
                "bev_shape": [6, 128, 160],
            },
        )
    out_root.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    stop = _TrainingStop()
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_handlers:
        signal.signal(sig, stop)
    started = time.monotonic()
    completed = start_step
    max_seconds = float(cfg.act.get("train_hours", 24.0)) * 3600.0
    milestones = {2500, 10000, 40000, 80000}
    try:
        policy.train()
        for step in range(start_step, int(args.steps)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite ObjectACT loss at step {step}")
            loss.backward()
            optimizer.step()
            completed = step + 1
            elapsed = elapsed_before + time.monotonic() - started
            if int(args.checkpoint_every) > 0 and completed % int(args.checkpoint_every) == 0:
                save_objectact_checkpoint(
                    policy, optimizer, preprocessor, out_root,
                    step=completed, config_payload=asdict(policy_config),
                    dataset_root=str(dataset_root), elapsed_seconds=elapsed,
                )
                prune_objectact_checkpoints(out_root, keep_latest=int(args.keep_checkpoints), milestones=milestones)
                if tracker is not None:
                    _safe_trackio_call(
                        tracker,
                        "log",
                        {"checkpoint_step": int(completed)},
                        failures=tracking_failures,
                    )
            if tracker is not None and completed % 100 == 0:
                progress = {
                    "step": int(completed),
                    "loss": float(loss.detach().cpu()),
                    **_numeric_metrics(metrics),
                    "elapsed_s": float(elapsed),
                }
                print(json.dumps({
                    "step": int(completed),
                    "loss": progress["loss"],
                    "metrics": {
                        key: value for key, value in progress.items()
                        if key not in {"step", "loss", "elapsed_s"}
                    },
                    "elapsed_s": progress["elapsed_s"],
                    "device": device,
                }, ensure_ascii=False), flush=True)
                _safe_trackio_call(
                    tracker,
                    "log",
                    progress,
                    failures=tracking_failures,
                )
            elif tracker is None and completed % 100 == 0:
                print(json.dumps({
                    "step": int(completed),
                    "loss": float(loss.detach().cpu()),
                    "metrics": _numeric_metrics(metrics),
                    "elapsed_s": float(elapsed),
                    "device": device,
                }, ensure_ascii=False), flush=True)
            if (
                tracker is not None
                and tracking is not None
                and bool(tracking["static"])
                and elapsed - last_static_sync_elapsed >= 60.0
            ):
                # Public static Spaces are free but read-only.  Keep the
                # training loop local and push a snapshot asynchronously so a
                # slow HF upload cannot pause MPS optimization.
                _safe_trackio_call(
                    tracker,
                    "sync",
                    project=str(tracking["project"]),
                    space_id=str(tracking["space_id"]),
                    force=True,
                    run_in_background=True,
                    sdk="static",
                    failures=tracking_failures,
                )
                last_static_sync_elapsed = float(elapsed)
            if stop.requested or elapsed >= max_seconds:
                break
    finally:
        if tracker is not None:
            _safe_trackio_call(tracker, "finish", failures=tracking_failures)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    elapsed = elapsed_before + time.monotonic() - started
    save_objectact_checkpoint(
        policy, optimizer, preprocessor, out_root,
        step=completed, config_payload=asdict(policy_config),
        dataset_root=str(dataset_root), elapsed_seconds=elapsed,
    )
    prune_objectact_checkpoints(out_root, keep_latest=int(args.keep_checkpoints), milestones=milestones)
    summary = {
        "schema_version": 5,
        "steps": completed,
        "interrupted": bool(stop.requested),
        "dataset": str(dataset_root),
        "manifest": manifest_summary,
        "device": device,
        "max_steps": int(args.steps),
        "checkpoint_every": int(args.checkpoint_every),
        "tracking": tracking,
        "tracking_failures": tracking_failures,
        "elapsed_seconds": elapsed,
    }
    (out_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if tracker is not None and tracking is not None and bool(tracking["static"]):
        # The final snapshot is best-effort and deliberately happens only
        # after the local checkpoint and summary are durable.  A transient HF
        # or proxy failure must not erase the completed training result.
        _safe_trackio_call(
            tracker,
            "sync",
            project=str(tracking["project"]),
            space_id=str(tracking["space_id"]),
            force=True,
            sdk="static",
            failures=tracking_failures,
        )
        # Persist the warning list after the final sync attempt as well.
        summary["tracking_failures"] = tracking_failures
        (out_root / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


def main(argv=None) -> int:
    summary = train_objectact(argv)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
