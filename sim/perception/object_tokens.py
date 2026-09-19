"""Predicted object tracks converted to the schema-v5 token contract.

This module is deliberately independent of MuJoCo and of any detector model.
It accepts detector/tracker outputs, keeps the slot order deterministic, and
encodes shape symmetries so that an equivalent rotation of a round, hexagonal,
or elongated part does not create an artificial discontinuity for ACT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from ..model.geometries import GEOMETRY_NAMES
from ..act.v5 import OBJECT_SLOTS, OBJECT_TOKEN_DIM


CLASS_NAMES = ("nut", "round", "fastener")
GEOMETRY_TO_CLASS = {
    "hex_nut": "nut",
    "cylinder": "round",
    "washer": "round",
    "screw": "fastener",
    "bolt": "fastener",
}
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_TO_HARMONIC = {"nut": 6, "round": 0, "fastener": 2}


# Keep this mapping as the single source of truth for the 29-dimensional
# layout.  Every slice is half-open and the scalar fields have length one so
# callers can concatenate them without special cases.
OBJECT_TOKEN_SLICES: Mapping[str, slice] = {
    "class_probs": slice(0, 3),
    "xy": slice(3, 5),
    "extent": slice(5, 7),
    "symmetry_yaw": slice(7, 9),
    "previous_xy": slice(9, 11),
    "previous_extent": slice(11, 13),
    "previous_symmetry_yaw": slice(13, 15),
    "velocity_xy": slice(15, 17),
    "angular_velocity": slice(17, 18),
    "relative_brush": slice(18, 20),
    "relative_tray": slice(20, 22),
    "tray_overlap": slice(22, 23),
    "full_inside": slice(23, 24),
    "confidence": slice(24, 25),
    "overhead_visibility": slice(25, 26),
    "wrist_visibility": slice(26, 27),
    "staleness": slice(27, 28),
    "valid": slice(28, 29),
}


@dataclass(frozen=True)
class TrackedObject:
    """One detector/tracker result at a policy observation time.

    All positions and extents are in the fixed table XY frame.  The optional
    previous fields are filled from the current detection on a first sighting;
    this avoids fabricating a large velocity at track birth.
    """

    track_id: int
    class_probs: np.ndarray
    xy: np.ndarray
    extent: np.ndarray
    yaw: float
    previous_xy: np.ndarray | None = None
    previous_extent: np.ndarray | None = None
    previous_yaw: float | None = None
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    angular_velocity: float = 0.0
    tray_overlap: float = 0.0
    full_inside: bool = False
    confidence: float = 1.0
    overhead_visibility: float = 1.0
    wrist_visibility: float = 0.0
    staleness: float = 0.0
    valid: bool = True


def class_probabilities_for_geometry(geometry: str) -> np.ndarray:
    """Return the grouped detector target for a known simulator geometry."""

    if geometry not in GEOMETRY_NAMES:
        raise ValueError(f"unknown geometry {geometry!r}; expected one of {GEOMETRY_NAMES}")
    probabilities = np.zeros(3, dtype=np.float32)
    probabilities[CLASS_TO_INDEX[GEOMETRY_TO_CLASS[geometry]]] = 1.0
    return probabilities


def _as_vector(value: np.ndarray | Iterable[float], size: int, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{field_name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{field_name} must contain finite values")
    return result


def _normalise_class_probs(class_probs: np.ndarray | Iterable[float]) -> np.ndarray:
    probabilities = _as_vector(class_probs, 3, "class_probs")
    if np.any(probabilities < 0.0):
        raise ValueError("class_probs must be non-negative")
    total = float(probabilities.sum())
    if total <= 1e-12:
        raise ValueError("class_probs must have a positive sum")
    return (probabilities / total).astype(np.float32)


def symmetry_aware_yaw(
    yaw: float,
    *,
    class_probs: np.ndarray | Iterable[float] | None = None,
    class_name: str | None = None,
) -> np.ndarray:
    """Encode yaw while respecting the predicted planar shape symmetry.

    A nut uses a six-fold harmonic, an elongated fastener a two-fold harmonic,
    and a round object has no meaningful planar yaw so its embedding is zero.
    With soft detector probabilities the non-round harmonics are mixed; the
    class-probability fields remain separate in the token.
    """

    if not np.isfinite(float(yaw)):
        raise ValueError("yaw must be finite")
    if class_probs is not None:
        probabilities = _normalise_class_probs(class_probs)
        values = np.zeros(2, dtype=np.float64)
        for probability, class_label in zip(probabilities, CLASS_NAMES):
            harmonic = CLASS_TO_HARMONIC[class_label]
            if harmonic:
                values += float(probability) * np.array(
                    [np.cos(harmonic * yaw), np.sin(harmonic * yaw)], dtype=float
                )
        return values.astype(np.float32)
    if class_name not in CLASS_TO_HARMONIC:
        raise ValueError(f"class_name must be one of {CLASS_NAMES}")
    harmonic = CLASS_TO_HARMONIC[class_name]
    if harmonic == 0:
        return np.zeros(2, dtype=np.float32)
    return np.asarray([np.cos(harmonic * yaw), np.sin(harmonic * yaw)], dtype=np.float32)


def _bounded_scalar(value: float, field_name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return float(np.clip(value, 0.0, 1.0))


def track_to_token(
    track: TrackedObject,
    *,
    brush_xy: np.ndarray | Iterable[float],
    tray_mouth_xy: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Convert one predicted track to exactly one 29-D float32 token."""

    if not bool(track.valid):
        return np.zeros(OBJECT_TOKEN_DIM, dtype=np.float32)
    current_xy = _as_vector(track.xy, 2, "xy")
    current_extent = _as_vector(track.extent, 2, "extent")
    previous_xy = current_xy if track.previous_xy is None else _as_vector(
        track.previous_xy, 2, "previous_xy"
    )
    previous_extent = current_extent if track.previous_extent is None else _as_vector(
        track.previous_extent, 2, "previous_extent"
    )
    previous_yaw = float(track.yaw if track.previous_yaw is None else track.previous_yaw)
    if not np.isfinite(previous_yaw):
        raise ValueError("previous_yaw must be finite")
    velocity_xy = _as_vector(track.velocity_xy, 2, "velocity_xy")
    brush = _as_vector(brush_xy, 2, "brush_xy")
    tray = _as_vector(tray_mouth_xy, 2, "tray_mouth_xy")
    probabilities = _normalise_class_probs(track.class_probs)
    token = np.zeros(OBJECT_TOKEN_DIM, dtype=np.float32)
    fields = {
        "class_probs": probabilities,
        "xy": current_xy,
        "extent": current_extent,
        "symmetry_yaw": symmetry_aware_yaw(float(track.yaw), class_probs=probabilities),
        "previous_xy": previous_xy,
        "previous_extent": previous_extent,
        "previous_symmetry_yaw": symmetry_aware_yaw(previous_yaw, class_probs=probabilities),
        "velocity_xy": velocity_xy,
        "angular_velocity": [float(track.angular_velocity)],
        "relative_brush": current_xy - brush,
        "relative_tray": current_xy - tray,
        "tray_overlap": [_bounded_scalar(track.tray_overlap, "tray_overlap")],
        "full_inside": [1.0 if bool(track.full_inside) else 0.0],
        "confidence": [_bounded_scalar(track.confidence, "confidence")],
        "overhead_visibility": [_bounded_scalar(track.overhead_visibility, "overhead_visibility")],
        "wrist_visibility": [_bounded_scalar(track.wrist_visibility, "wrist_visibility")],
        "staleness": [_bounded_scalar(track.staleness, "staleness")],
        "valid": [1.0],
    }
    for name, value in fields.items():
        token[OBJECT_TOKEN_SLICES[name]] = np.asarray(value, dtype=np.float32)
    return token


def pack_object_tokens(
    tracks: Iterable[TrackedObject],
    *,
    brush_xy: np.ndarray | Iterable[float],
    tray_mouth_xy: np.ndarray | Iterable[float],
    slots: int = OBJECT_SLOTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack tracks into stable slots and return ``(tokens, valid)``.

    If a detector produces more tracks than the task maximum, the highest
    confidence tracks are retained.  The retained set is then sorted by
    persistent track id, making output independent of detector iteration order.
    """

    if int(slots) != OBJECT_SLOTS:
        raise ValueError(f"schema v5 requires exactly {OBJECT_SLOTS} object slots")
    _as_vector(brush_xy, 2, "brush_xy")
    _as_vector(tray_mouth_xy, 2, "tray_mouth_xy")
    candidates = select_tracks_for_slots(tracks, slots=slots)
    tokens = np.zeros((OBJECT_SLOTS, OBJECT_TOKEN_DIM), dtype=np.float32)
    valid = np.zeros(OBJECT_SLOTS, dtype=bool)
    for index, track in enumerate(candidates):
        tokens[index] = track_to_token(
            track, brush_xy=brush_xy, tray_mouth_xy=tray_mouth_xy
        )
        valid[index] = True
    return tokens, valid


def select_tracks_for_slots(
    tracks: Iterable[TrackedObject], *, slots: int = OBJECT_SLOTS
) -> list[TrackedObject]:
    """Select and stably order the tracks shared by tokens and instance masks."""

    if int(slots) != OBJECT_SLOTS:
        raise ValueError(f"schema v5 requires exactly {OBJECT_SLOTS} object slots")
    candidates = [track for track in tracks if bool(track.valid)]
    ids = [int(track.track_id) for track in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("tracked object ids must be unique")
    candidates.sort(key=lambda item: (-_bounded_scalar(item.confidence, "confidence"), int(item.track_id)))
    candidates = candidates[:OBJECT_SLOTS]
    candidates.sort(key=lambda item: int(item.track_id))
    return candidates
