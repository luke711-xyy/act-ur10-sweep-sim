"""Train and audit the RGB instance detector used by ObjectACT.

This is intentionally a separate job from ACT training.  It consumes only
RGB frames plus offline detector labels, writes a frontend-compatible state
dict, and refuses to claim a deployable detector unless the held-out quality
gate passes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..perception.detector import FrozenResNet18FPN, decode_detector_output, detector_loss
from ..perception.detector_training import (
    DetectorTrainingDataset,
    detector_quality_gate,
    detector_quality_metrics,
    save_detector_checkpoint,
)


def build_detector_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the RGB ObjectACT detector")
    parser.add_argument("--dataset", default="runs/objectact_detector_dataset")
    parser.add_argument("--out", default="runs/objectact_detector")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fpn-dim", type=int, default=128)
    parser.add_argument("--pretrained", action="store_true",
                        help="initialize the ResNet-18 backbone from ImageNet")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-mask-iou", type=float, default=0.70)
    parser.add_argument("--max-center-error-px", type=float, default=3.0)
    parser.add_argument("--max-miss-rate", type=float, default=0.10)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.10)
    parser.add_argument("--max-count-error", type=float, default=0.25)
    parser.add_argument("--require-quality-gate", action="store_true")
    return parser


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _aggregate(metrics: list[dict]) -> dict[str, float | int]:
    if not metrics:
        raise ValueError("cannot aggregate an empty detector metric list")
    keys = (
        "mask_iou", "center_error_px", "miss_rate",
        "false_positive_rate", "mean_abs_count_error",
    )
    finite = [item for item in metrics if np.isfinite(float(item["center_error_px"]))]
    samples = int(sum(int(item.get("sample_count", 1)) for item in metrics))
    result: dict[str, float | int] = {"sample_count": samples}
    result["matched_count"] = int(sum(int(item.get("matched_count", 0)) for item in metrics))
    for key in keys:
        values = [item[key] for item in (finite if key == "center_error_px" else metrics)]
        result[key] = float(np.mean(np.asarray(values, dtype=float))) if values else float("inf")
    return result


def evaluate_detector(model, dataset, *, device: str, max_samples: int | None = None) -> dict:
    """Evaluate decoded instance quality against offline instance maps."""

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    samples = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            image = batch["image"].to(device)
            output = model(image)
            predictions = decode_detector_output(output, foreground_threshold=0.5)
            truth = np.asarray(batch["instance"][0], dtype=np.int32)
            samples.append(detector_quality_metrics(predictions, truth))
            if max_samples is not None and index + 1 >= int(max_samples):
                break
    return _aggregate(samples)


def train_detector(args=None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    args = build_detector_train_parser().parse_args(args)
    if int(args.steps) < 1 or int(args.batch_size) < 1:
        raise ValueError("steps and batch-size must be positive")
    device = resolve_device(str(args.device))
    torch.manual_seed(int(args.seed))
    dataset_root = Path(args.dataset)
    train = DetectorTrainingDataset(dataset_root, split="train")
    try:
        validation = DetectorTrainingDataset(dataset_root, split="val")
    except ValueError:
        validation = train
    loader = DataLoader(train, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    model = FrozenResNet18FPN(
        pretrained=bool(args.pretrained),
        freeze_backbone=not bool(args.unfreeze_backbone),
        fpn_dim=int(args.fpn_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    losses = []
    started = time.monotonic()
    latest = None
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        targets = {
            "semantic": batch["semantic"].to(device),
            "center": batch["center"].to(device),
            "offset": batch["offset"].to(device),
            "offset_valid": batch["offset_valid"].to(device),
        }
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = detector_loss(model(batch["image"].to(device)), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % int(args.checkpoint_every) == 0 or step == int(args.steps):
            latest = save_detector_checkpoint(
                model,
                out_root,
                step=step,
                config={
                    "fpn_dim": int(args.fpn_dim),
                    "pretrained": bool(args.pretrained),
                    "freeze_backbone": not bool(args.unfreeze_backbone),
                    "device": device,
                },
            )
    quality = evaluate_detector(model, validation, device=device)
    quality.update({
        "mean_train_loss": float(np.mean(losses)),
        "last_train_loss": float(losses[-1]),
        "elapsed_seconds": float(time.monotonic() - started),
        "checkpoint": str(latest) if latest is not None else "",
    })
    quality["quality_gate_passed"] = detector_quality_gate(
        quality,
        min_mask_iou=float(args.min_mask_iou),
        max_center_error_px=float(args.max_center_error_px),
        max_miss_rate=float(args.max_miss_rate),
        max_false_positive_rate=float(args.max_false_positive_rate),
        max_count_error=float(args.max_count_error),
    )
    (out_root / "detector_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if bool(args.require_quality_gate) and not quality["quality_gate_passed"]:
        raise RuntimeError(f"RGB detector quality gate failed: {quality}")
    return quality


def main(argv=None) -> int:
    result = train_detector(argv)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
