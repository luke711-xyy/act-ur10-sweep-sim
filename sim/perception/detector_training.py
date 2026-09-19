"""Training data, metrics, and gates for the RGB object detector.

The detector is allowed to use simulator segmentation only while producing
offline supervision.  This module deliberately stores image-derived inputs
and dense labels in a separate manifest; no simulator truth is exposed to the
ObjectACT dataset or runtime observation contract.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np

from .detector import ImageInstancePrediction


DETECTOR_DATASET_SCHEMA = 1


def instance_maps_from_geom_ids(
    segmentation: np.ndarray,
    *,
    geom_to_component: dict[int, int],
    component_class_indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a MuJoCo geom-id image into grouped instance/class maps.

    The conversion is an offline-labeling operation.  The returned maps are
    written to detector supervision files and are never consumed by the policy
    observation builder at runtime.
    """

    values = np.asarray(segmentation, dtype=np.int32)
    if values.ndim != 2:
        raise ValueError("segmentation must be a 2-D geom-id image")
    classes = [int(value) for value in component_class_indices]
    if any(value < 1 or value > 3 for value in classes):
        raise ValueError("component classes must be grouped indices 1..3")
    instance_map = np.zeros(values.shape, dtype=np.int32)
    class_map = np.zeros(values.shape, dtype=np.int64)
    for geom_id, component_index in geom_to_component.items():
        component_index = int(component_index)
        if component_index < 0 or component_index >= len(classes):
            raise ValueError("geom_to_component references an unknown component")
        pixels = values == int(geom_id)
        instance_map[pixels] = component_index + 1
        class_map[pixels] = classes[component_index]
    return instance_map, class_map


def dense_targets_from_instance_map(
    instance_map: np.ndarray, class_map: np.ndarray
) -> dict[str, np.ndarray]:
    """Create semantic/centre/offset targets from an offline instance map.

    ``class_map`` uses zero for background and 1--3 for the grouped detector
    classes.  Offsets point from each foreground pixel to the instance centre,
    matching :func:`decode_detector_output`'s ``row + offset`` convention.
    """

    instances = np.asarray(instance_map)
    classes = np.asarray(class_map)
    if instances.ndim != 2 or classes.shape != instances.shape:
        raise ValueError("instance_map and class_map must be same-shaped 2-D arrays")
    if not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("instance_map must contain integer ids")
    if np.any(instances < 0) or np.any(classes < 0) or np.any(classes > 3):
        raise ValueError("instance and class labels are outside the declared range")
    foreground = instances > 0
    if np.any(foreground & (classes == 0)):
        raise ValueError("every foreground instance pixel needs a non-zero class")

    height, width = instances.shape
    center = np.zeros((1, height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    offset_valid = foreground[None, ...].astype(np.float32)
    for instance_id in np.unique(instances[foreground]):
        rows, cols = np.nonzero(instances == instance_id)
        if rows.size == 0:
            continue
        centre = np.asarray([rows.mean(), cols.mean()], dtype=np.float32)
        center[0, int(np.rint(centre[0])), int(np.rint(centre[1]))] = 1.0
        offset[0, rows, cols] = centre[0] - rows
        offset[1, rows, cols] = centre[1] - cols
    return {
        "semantic": classes.astype(np.int64, copy=False),
        "center": center,
        "offset": offset,
        "offset_valid": offset_valid,
    }


def _instance_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = int(np.count_nonzero(prediction & truth))
    union = int(np.count_nonzero(prediction | truth))
    return float(intersection / union) if union else 0.0


def detector_quality_metrics(
    predictions: Iterable[ImageInstancePrediction],
    truth_instance_map: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Match predicted masks to offline instances and return audit metrics."""

    truth = np.asarray(truth_instance_map)
    if truth.ndim != 2 or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("truth_instance_map must be a 2-D integer array")
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    predictions = list(predictions)
    truth_ids = [int(value) for value in np.unique(truth) if int(value) > 0]
    pairs: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        mask = np.asarray(prediction.mask, dtype=bool)
        if mask.shape != truth.shape:
            raise ValueError("prediction mask shape does not match truth")
        for truth_id in truth_ids:
            pairs.append((_instance_iou(mask, truth == truth_id), prediction_index, truth_id))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matched: list[tuple[float, int, int]] = []
    for iou, prediction_index, truth_id in pairs:
        if iou < float(iou_threshold):
            break
        if prediction_index in used_predictions or truth_id in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_id)
        matched.append((iou, prediction_index, truth_id))

    center_errors = []
    for _, prediction_index, truth_id in matched:
        rows, cols = np.nonzero(truth == truth_id)
        expected = np.asarray([rows.mean(), cols.mean()], dtype=float)
        observed = np.asarray(predictions[prediction_index].center_rc, dtype=float).reshape(2)
        center_errors.append(float(np.linalg.norm(observed - expected)))
    truth_count = len(truth_ids)
    prediction_count = len(predictions)
    return {
        "sample_count": 1,
        "matched_count": len(matched),
        "mask_iou": float(np.mean([item[0] for item in matched])) if matched else 0.0,
        "center_error_px": float(np.mean(center_errors)) if center_errors else float("inf"),
        "miss_rate": float((truth_count - len(matched)) / max(truth_count, 1)),
        "false_positive_rate": float((prediction_count - len(matched)) / max(prediction_count, 1)),
        "mean_abs_count_error": float(abs(prediction_count - truth_count)),
    }


def detector_quality_gate(
    metrics: dict[str, float | int],
    *,
    min_mask_iou: float,
    max_center_error_px: float,
    max_miss_rate: float,
    max_false_positive_rate: float,
    max_count_error: float,
) -> bool:
    """Return whether all declared held-out detector thresholds pass."""

    return bool(
        float(metrics.get("mask_iou", 0.0)) >= float(min_mask_iou)
        and float(metrics.get("center_error_px", float("inf")) <= float(max_center_error_px))
        and float(metrics.get("miss_rate", 1.0)) <= float(max_miss_rate)
        and float(metrics.get("false_positive_rate", 1.0)) <= float(max_false_positive_rate)
        and float(metrics.get("mean_abs_count_error", float("inf"))) <= float(max_count_error)
    )


def save_detector_checkpoint(
    model,
    root: str | Path,
    *,
    step: int,
    config: dict,
) -> Path:
    """Save a frontend-compatible state dict and its auditable metadata."""

    import torch

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    step = int(step)
    path = root / f"detector_step_{step:06d}.pt"
    torch.save(model.state_dict(), path)
    metadata = {
        "schema_version": DETECTOR_DATASET_SCHEMA,
        "step": step,
        "config": dict(config),
        "checkpoint": path.name,
    }
    (root / "detector_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "latest_checkpoint.txt").write_text(path.name + "\n", encoding="utf-8")
    return path


def load_detector_checkpoint(path: str | Path, *, device: str = "cpu") -> dict:
    """Read a detector state dict plus metadata, accepting a file or output dir."""

    import torch

    path = Path(path)
    if path.is_dir():
        latest = path / "latest_checkpoint.txt"
        if not latest.exists():
            raise FileNotFoundError(latest)
        path = path / latest.read_text(encoding="utf-8").strip()
    state = torch.load(path, map_location=device, weights_only=True)
    metadata_path = path.parent / "detector_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {"schema_version": DETECTOR_DATASET_SCHEMA, "step": -1, "config": {}}
    if int(metadata.get("schema_version", 0)) != DETECTOR_DATASET_SCHEMA:
        raise ValueError("unsupported detector checkpoint schema")
    return {**metadata, "state_dict": state, "path": str(path)}


def verified_detector_checkpoint(path: str | Path) -> Path:
    """Resolve a detector checkpoint only when its held-out gate passed."""

    path = Path(path)
    if path.is_dir():
        latest = path / "latest_checkpoint.txt"
        if not latest.exists():
            raise FileNotFoundError(latest)
        path = path / latest.read_text(encoding="utf-8").strip()
    if not path.exists():
        raise FileNotFoundError(path)
    quality_path = path.parent / "detector_quality.json"
    if not quality_path.exists():
        raise RuntimeError(f"RGB detector quality gate metadata is missing for {path}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not bool(quality.get("quality_gate_passed", False)):
        raise RuntimeError(f"RGB detector quality gate did not pass for {path}")
    return path


class DetectorDatasetWriter:
    """Atomically write RGB frames and detector supervision sidecars."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"

    def add_frame(
        self,
        frame_id: str,
        image: np.ndarray,
        instance_map: np.ndarray,
        class_map: np.ndarray,
        *,
        split: str,
        metadata: dict | None = None,
    ) -> Path:
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        instance_map = np.asarray(instance_map, dtype=np.int32)
        class_map = np.asarray(class_map, dtype=np.int64)
        if instance_map.shape != image.shape[:2] or class_map.shape != image.shape[:2]:
            raise ValueError("label maps must match image height and width")
        targets = dense_targets_from_instance_map(instance_map, class_map)
        frame_dir = self.root / str(frame_id)
        if frame_dir.exists():
            raise FileExistsError(frame_dir)
        temporary = self.root / f".{frame_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        created = False
        committed = False
        try:
            from PIL import Image

            Image.fromarray(image).save(temporary / "image.png")
            np.savez_compressed(
                temporary / "targets.npz",
                instance=instance_map,
                semantic=targets["semantic"],
                center=targets["center"],
                offset=targets["offset"],
                offset_valid=targets["offset_valid"],
            )
            record = {
                **(metadata or {}),
                "frame_id": str(frame_id),
                "schema_version": DETECTOR_DATASET_SCHEMA,
                "split": str(split),
                "image": str(Path(str(frame_id)) / "image.png"),
                "targets": str(Path(str(frame_id)) / "targets.npz"),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            }
            if any("truth" in str(key).lower() or "mujoco" in str(key).lower() for key in record):
                raise ValueError("detector manifest must not expose simulator truth fields")
            temporary.rename(frame_dir)
            created = True
            previous = self.manifest.read_text(encoding="utf-8") if self.manifest.exists() else ""
            manifest_tmp = self.root / f".{self.manifest.name}.tmp-{uuid.uuid4().hex}"
            try:
                manifest_tmp.write_text(previous + json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
                manifest_tmp.replace(self.manifest)
            finally:
                manifest_tmp.unlink(missing_ok=True)
            committed = True
            return frame_dir
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            if created and not committed and frame_dir.exists():
                shutil.rmtree(frame_dir, ignore_errors=True)


class DetectorTrainingDataset:
    """Torch-compatible reader for the detector manifest."""

    def __init__(self, root: str | Path, *, split: str = "train"):
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        self.records = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("schema_version", 0)) != DETECTOR_DATASET_SCHEMA:
                raise ValueError("unsupported detector dataset schema")
            if str(record.get("split", "train")) == str(split):
                self.records.append(record)
        if not self.records:
            raise ValueError(f"detector dataset has no records for split {split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        import torch
        record = self.records[int(index)]
        from PIL import Image

        image = np.asarray(Image.open(self.root / record["image"]).convert("RGB"), dtype=np.float32)
        targets = np.load(self.root / record["targets"])
        return {
            "image": torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float(),
            "semantic": torch.from_numpy(np.asarray(targets["semantic"], dtype=np.int64)),
            "center": torch.from_numpy(np.asarray(targets["center"], dtype=np.float32)),
            "offset": torch.from_numpy(np.asarray(targets["offset"], dtype=np.float32)),
            "offset_valid": torch.from_numpy(np.asarray(targets["offset_valid"], dtype=np.float32)),
            "instance": np.asarray(targets["instance"], dtype=np.int32),
        }
