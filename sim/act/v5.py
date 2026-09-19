"""Schema-v5 contracts shared by ObjectACT-BEV data, training, and runtime.

The module intentionally contains no simulator access.  It is the boundary
that prevents MuJoCo bookkeeping from silently becoming a policy feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = 5
ROBOT_STATE_DIM = 36
TASK_STATE_DIM = 6
OBJECT_SLOTS = 6
OBJECT_TOKEN_DIM = 29
BEV_CHANNELS = 6
BEV_HEIGHT = 128
BEV_WIDTH = 160
BEV_PACKED_WIDTH = (BEV_WIDTH + 7) // 8


@dataclass(frozen=True)
class V5ObservationSpec:
    """The fixed tensor shapes used by the ObjectACT-BEV policy."""

    robot_state_dim: int = ROBOT_STATE_DIM
    task_state_dim: int = TASK_STATE_DIM
    object_slots: int = OBJECT_SLOTS
    object_token_dim: int = OBJECT_TOKEN_DIM
    bev_shape: tuple[int, int, int] = (BEV_CHANNELS, BEV_HEIGHT, BEV_WIDTH)


def pack_instance_bev(masks: np.ndarray) -> np.ndarray:
    """Pack six boolean instance masks along the BEV width axis."""

    values = np.asarray(masks, dtype=bool)
    expected = (OBJECT_SLOTS, BEV_HEIGHT, BEV_WIDTH)
    if values.shape != expected:
        raise ValueError(
            f"instance BEV masks must have shape {expected}, got {values.shape}"
        )
    return np.packbits(values, axis=-1, bitorder="big")


def unpack_instance_bev(packed: np.ndarray) -> np.ndarray:
    """Restore packed instance masks without changing their slot order."""

    values = np.asarray(packed, dtype=np.uint8)
    expected = (OBJECT_SLOTS, BEV_HEIGHT, BEV_PACKED_WIDTH)
    if values.shape != expected:
        raise ValueError(
            f"packed instance BEV must have shape {expected}, got {values.shape}"
        )
    return np.unpackbits(values, axis=-1, count=BEV_WIDTH, bitorder="big").astype(bool)


def _validate_sidecar_arrays(
    object_tokens: np.ndarray,
    object_valid: np.ndarray,
    instance_bev: np.ndarray,
    task_state: np.ndarray,
) -> int:
    tokens = np.asarray(object_tokens, dtype=np.float32)
    valid = np.asarray(object_valid, dtype=bool)
    masks = np.asarray(instance_bev, dtype=bool)
    task = np.asarray(task_state, dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[1:] != (OBJECT_SLOTS, OBJECT_TOKEN_DIM):
        raise ValueError("object_tokens sidecar must have shape (T, 6, 29)")
    frames = tokens.shape[0]
    if valid.shape != (frames, OBJECT_SLOTS):
        raise ValueError("object_valid sidecar must have shape (T, 6)")
    if masks.shape != (frames, OBJECT_SLOTS, BEV_HEIGHT, BEV_WIDTH):
        raise ValueError("instance_bev sidecar must have shape (T, 6, 128, 160)")
    if task.shape != (frames, TASK_STATE_DIM):
        raise ValueError("task_state sidecar must have shape (T, 6)")
    return frames


def write_v5_sidecar(
    path: str | Path,
    object_tokens: np.ndarray,
    object_valid: np.ndarray,
    instance_bev: np.ndarray,
    task_state: np.ndarray,
    selection_target: np.ndarray | None = None,
    visual_count: np.ndarray | None = None,
) -> None:
    """Write predicted perception fields without storing a dense float BEV."""

    frames = _validate_sidecar_arrays(object_tokens, object_valid, instance_bev, task_state)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "object_tokens": np.asarray(object_tokens, dtype=np.float32),
        "object_valid": np.asarray(object_valid, dtype=bool),
        "instance_bev_packed": np.packbits(
            np.asarray(instance_bev, dtype=bool), axis=-1, bitorder="big"
        ),
        "task_state": np.asarray(task_state, dtype=np.float32),
    }
    if selection_target is not None:
        labels = np.asarray(selection_target, dtype=bool)
        if labels.shape != (frames, OBJECT_SLOTS):
            raise ValueError("selection_target sidecar must have shape (T, 6)")
        payload["selection_target"] = labels
    if visual_count is not None:
        counts = np.asarray(visual_count, dtype=np.int16)
        if counts.shape != (frames,):
            raise ValueError("visual_count sidecar must have shape (T,)")
        payload["visual_count"] = counts
    np.savez_compressed(Path(path), **payload)


def read_v5_sidecar(path: str | Path) -> dict[str, np.ndarray]:
    """Read a v5 sidecar and unpack its instance masks."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        schema = int(np.asarray(arrays["schema_version"]).reshape(-1)[0])
        if schema != SCHEMA_VERSION:
            raise ValueError(f"expected schema v5 sidecar, got schema v{schema}")
        tokens = np.asarray(arrays["object_tokens"], dtype=np.float32)
        valid = np.asarray(arrays["object_valid"], dtype=bool)
        packed = np.asarray(arrays["instance_bev_packed"], dtype=np.uint8)
        task = np.asarray(arrays["task_state"], dtype=np.float32)
        if tokens.ndim != 3 or tokens.shape[1:] != (OBJECT_SLOTS, OBJECT_TOKEN_DIM):
            raise ValueError("object_tokens sidecar must have shape (T, 6, 29)")
        frames = tokens.shape[0]
        if packed.shape != (frames, OBJECT_SLOTS, BEV_HEIGHT, BEV_PACKED_WIDTH):
            raise ValueError("instance_bev_packed sidecar has an invalid shape")
        if valid.shape != (frames, OBJECT_SLOTS):
            raise ValueError("object_valid sidecar must have shape (T, 6)")
        if task.shape != (frames, TASK_STATE_DIM):
            raise ValueError("task_state sidecar must have shape (T, 6)")
        result: dict[str, np.ndarray] = {
            "object_tokens": tokens,
            "object_valid": valid,
            "instance_bev": np.unpackbits(
                packed, axis=-1, count=BEV_WIDTH, bitorder="big"
            ).astype(bool),
            "task_state": task,
        }
        if "selection_target" in arrays.files:
            labels = np.asarray(arrays["selection_target"], dtype=bool)
            if labels.shape != (frames, OBJECT_SLOTS):
                raise ValueError("selection_target sidecar must have shape (T, 6)")
            result["selection_target"] = labels
        if "visual_count" in arrays.files:
            counts = np.asarray(arrays["visual_count"], dtype=np.int16)
            if counts.shape != (frames,):
                raise ValueError("visual_count sidecar must have shape (T,)")
            result["visual_count"] = counts
        return result


def _require_shape(observation: Mapping[str, object], key: str, shape: tuple[int, ...]) -> None:
    if key not in observation:
        raise ValueError(f"schema v5 observation is missing {key}")
    actual = np.asarray(observation[key]).shape
    if actual != shape:
        raise ValueError(f"{key} must have shape {shape}, got {actual}")


def validate_v5_observation(observation: Mapping[str, object]) -> None:
    """Validate a policy observation and reject known truth/v4 fields."""

    if "observation.state" in observation or "observation.environment_state" in observation:
        raise ValueError("schema v5 must not use v4 state/environment fields")
    for key in observation:
        lowered = str(key).lower()
        if "mujoco" in lowered or "truth" in lowered or "ground_truth" in lowered:
            raise ValueError(f"schema v5 policy observation contains truth field {key}")

    _require_shape(observation, "observation.robot_state", (ROBOT_STATE_DIM,))
    _require_shape(observation, "observation.task_state", (TASK_STATE_DIM,))
    _require_shape(observation, "observation.object_tokens", (OBJECT_SLOTS, OBJECT_TOKEN_DIM))
    _require_shape(observation, "observation.object_valid", (OBJECT_SLOTS,))
    _require_shape(observation, "observation.instance_bev", (OBJECT_SLOTS, BEV_HEIGHT, BEV_WIDTH))

    if np.asarray(observation["observation.object_valid"]).dtype != np.bool_:
        raise ValueError("observation.object_valid must be boolean")
    if np.asarray(observation["observation.instance_bev"]).dtype != np.bool_:
        raise ValueError("observation.instance_bev must be boolean")
