"""Camera-prediction to table BEV conversion for ObjectACT-BEV.

The public functions in this module consume only detector masks, camera
calibration, and the current tool pose.  They never access an environment or
MuJoCo state.  That separation is intentional: the same code must run for
expert export, policy training, and checkpoint inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from .camera import PinholeCamera
from .object_tokens import OBJECT_SLOTS, TrackedObject


@dataclass(frozen=True)
class BEVSpec:
    """Fixed 1.0 m x 0.8 m table map at 6.25 mm per cell."""

    x_min: float = -0.5
    x_max: float = 0.5
    y_min: float = -0.4
    y_max: float = 0.4
    resolution: float = 0.00625

    @property
    def width(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def to_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.atleast_2d(np.asarray(xy, dtype=float))
        if values.shape[-1] != 2:
            raise ValueError(f"xy must have final dimension 2, got {values.shape}")
        ix = np.floor((values[:, 0] - self.x_min) / self.resolution).astype(int)
        iy = np.floor((values[:, 1] - self.y_min) / self.resolution).astype(int)
        valid = (
            np.isfinite(values).all(axis=1)
            & (ix >= 0)
            & (ix < self.width)
            & (iy >= 0)
            & (iy < self.height)
        )
        return iy, ix, valid

    def to_xy(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        x = self.x_min + (np.asarray(ix, dtype=float) + 0.5) * self.resolution
        y = self.y_min + (np.asarray(iy, dtype=float) + 0.5) * self.resolution
        return np.stack([x, y], axis=-1)

    def centres(self) -> np.ndarray:
        iy, ix = np.meshgrid(
            np.arange(self.height), np.arange(self.width), indexing="ij"
        )
        return self.to_xy(iy, ix)


@dataclass(frozen=True)
class PredictedInstance:
    """One fused detector instance represented in the fixed table frame."""

    track_id: int
    mask_bev: np.ndarray
    class_probs: np.ndarray
    confidence: float
    xy: np.ndarray
    extent: np.ndarray
    yaw: float
    overhead_visibility: float = 1.0
    wrist_visibility: float = 0.0
    tray_overlap: float = 0.0
    full_inside: bool = False


def _vector(value: np.ndarray | Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _probabilities(value: np.ndarray | Iterable[float]) -> np.ndarray:
    probs = _vector(value, 3, "class_probs")
    if np.any(probs < 0.0) or float(probs.sum()) <= 1e-12:
        raise ValueError("class_probs must be non-negative with a positive sum")
    return (probs / float(probs.sum())).astype(np.float32)


def _bounded(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(value, 0.0, 1.0))


def _principal_yaw(points_xy: np.ndarray) -> float:
    if len(points_xy) < 3:
        return 0.0
    centred = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    covariance = (centred.T @ centred) / max(len(points_xy), 1)
    values, vectors = np.linalg.eigh(covariance)
    if float(values[-1]) <= 1e-12:
        return 0.0
    direction = vectors[:, int(np.argmax(values))]
    return float(np.arctan2(direction[1], direction[0]))


def project_image_instance(
    mask: np.ndarray,
    *,
    camera: PinholeCamera,
    plane_z: float,
    bev_spec: BEVSpec,
    class_probs: np.ndarray | Iterable[float],
    confidence: float,
    overhead_visibility: float = 1.0,
    wrist_visibility: float = 0.0,
) -> PredictedInstance:
    """Back-project one image mask and rasterise it into the table BEV."""

    image_mask = np.asarray(mask, dtype=bool)
    if image_mask.ndim != 2:
        raise ValueError("instance mask must be a 2-D boolean image")
    rows, cols = np.nonzero(image_mask)
    if len(rows) == 0:
        raise ValueError("cannot project an empty instance mask")
    points = camera.pixels_to_plane(rows, cols, plane_z=float(plane_z))
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("instance mask has no valid intersection with the table plane")
    iy, ix, valid = bev_spec.to_index(points[:, :2])
    if not np.any(valid):
        raise ValueError("instance mask projects outside the fixed table BEV")
    mask_bev = np.zeros(bev_spec.shape, dtype=bool)
    mask_bev[iy[valid], ix[valid]] = True
    points_xy = points[valid, :2]
    extent = np.maximum(np.ptp(points_xy, axis=0), bev_spec.resolution)
    return PredictedInstance(
        track_id=-1,
        mask_bev=mask_bev,
        class_probs=_probabilities(class_probs),
        confidence=_bounded(confidence, "confidence"),
        xy=np.mean(points_xy, axis=0),
        extent=extent,
        yaw=_principal_yaw(points_xy),
        overhead_visibility=_bounded(overhead_visibility, "overhead_visibility"),
        wrist_visibility=_bounded(wrist_visibility, "wrist_visibility"),
    )


def _geometry_from_bev_mask(mask: np.ndarray, spec: BEVSpec) -> tuple[np.ndarray, np.ndarray, float]:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("cannot derive geometry from an empty BEV mask")
    points = spec.to_xy(rows, cols)
    return (
        np.mean(points, axis=0),
        np.maximum(np.ptp(points, axis=0), spec.resolution),
        _principal_yaw(points),
    )


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def fuse_projected_instances(
    instances: Iterable[PredictedInstance],
    *,
    spec: BEVSpec,
    center_distance: float = 0.04,
    mask_iou: float = 0.10,
) -> list[PredictedInstance]:
    """Fuse overhead/wrist detections that describe the same projected part."""

    if float(center_distance) <= 0.0 or not 0.0 <= float(mask_iou) <= 1.0:
        raise ValueError("invalid camera-fusion thresholds")
    ordered = sorted(
        list(instances),
        key=lambda item: (-float(item.confidence), float(item.xy[0]), float(item.xy[1])),
    )
    groups: list[list[PredictedInstance]] = []
    for instance in ordered:
        _check_instance(instance, spec)
        for group in groups:
            representative = group[0]
            distance = float(np.linalg.norm(np.asarray(instance.xy) - np.asarray(representative.xy)))
            if distance <= float(center_distance) or _mask_iou(instance.mask_bev, representative.mask_bev) >= float(mask_iou):
                group.append(instance)
                break
        else:
            groups.append([instance])

    fused: list[PredictedInstance] = []
    for group in groups:
        weights = np.asarray([max(float(item.confidence), 1e-6) for item in group], dtype=float)
        mask = np.logical_or.reduce([np.asarray(item.mask_bev, dtype=bool) for item in group])
        class_probs = np.average(
            np.stack([_probabilities(item.class_probs) for item in group], axis=0),
            axis=0,
            weights=weights,
        )
        centre, extent, yaw = _geometry_from_bev_mask(mask, spec)
        fused.append(
            PredictedInstance(
                track_id=-1,
                mask_bev=mask,
                class_probs=_probabilities(class_probs),
                confidence=float(1.0 - np.prod([1.0 - float(item.confidence) for item in group])),
                xy=centre,
                extent=extent,
                yaw=yaw,
                overhead_visibility=max(float(item.overhead_visibility) for item in group),
                wrist_visibility=max(float(item.wrist_visibility) for item in group),
                tray_overlap=max(float(item.tray_overlap) for item in group),
                full_inside=any(bool(item.full_inside) for item in group),
            )
        )
    return sorted(fused, key=lambda item: (float(item.xy[0]), float(item.xy[1])))


def annotate_tray_features(
    instance: PredictedInstance, *, spec: BEVSpec, tray_bounds: Sequence[float]
) -> PredictedInstance:
    """Add detector-derived tray overlap/full-inclusion features to an instance."""

    _check_instance(instance, spec)
    centres = spec.centres()
    inside = rectangle_signed_distance(centres.reshape(-1, 2), tray_bounds).reshape(spec.shape) >= 0.0
    mask = np.asarray(instance.mask_bev, dtype=bool)
    total = int(np.count_nonzero(mask))
    overlap = float(np.count_nonzero(mask & inside) / total) if total else 0.0
    return replace(instance, tray_overlap=overlap, full_inside=bool(total and np.all(inside[mask])))


def pack_instance_masks(
    instances: Iterable[PredictedInstance], *, spec: BEVSpec, slots: int = OBJECT_SLOTS
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Pack detector masks in the same deterministic order as object tokens."""

    if int(slots) != OBJECT_SLOTS:
        raise ValueError(f"schema v5 requires exactly {OBJECT_SLOTS} object slots")
    candidates = list(instances)
    ids = [int(item.track_id) for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("instance track ids must be unique before mask packing")
    for item in candidates:
        _check_instance(item, spec)
    candidates.sort(key=lambda item: (-_bounded(item.confidence, "confidence"), int(item.track_id)))
    candidates = candidates[:OBJECT_SLOTS]
    candidates.sort(key=lambda item: int(item.track_id))
    masks = np.zeros((OBJECT_SLOTS, *spec.shape), dtype=bool)
    valid = np.zeros(OBJECT_SLOTS, dtype=bool)
    track_ids: list[int] = []
    for index, item in enumerate(candidates):
        masks[index] = np.asarray(item.mask_bev, dtype=bool)
        valid[index] = True
        track_ids.append(int(item.track_id))
    return masks, valid, track_ids


def rectangle_signed_distance(points: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """Signed distance to an axis-aligned rectangle, positive inside."""

    values = np.atleast_2d(np.asarray(points, dtype=float))
    if values.shape[-1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(bounds) != 4:
        raise ValueError("bounds must be (x_min, x_max, y_min, y_max)")
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("rectangle bounds must be increasing")
    inside = (
        (values[:, 0] >= x_min)
        & (values[:, 0] <= x_max)
        & (values[:, 1] >= y_min)
        & (values[:, 1] <= y_max)
    )
    distance_x = np.maximum(np.maximum(x_min - values[:, 0], 0.0), values[:, 0] - x_max)
    distance_y = np.maximum(np.maximum(y_min - values[:, 1], 0.0), values[:, 1] - y_max)
    outside_distance = np.hypot(distance_x, distance_y)
    inside_distance = np.minimum.reduce(
        [values[:, 0] - x_min, x_max - values[:, 0], values[:, 1] - y_min, y_max - values[:, 1]]
    )
    return np.where(inside, inside_distance, -outside_distance)


def _oriented_rectangle_mask(
    spec: BEVSpec,
    centre: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> np.ndarray:
    grid = spec.centres()
    delta = grid - centre[None, None, :]
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    local_x = c * delta[..., 0] + s * delta[..., 1]
    local_y = -s * delta[..., 0] + c * delta[..., 1]
    return (np.abs(local_x) <= max(float(size[0]) / 2.0, spec.resolution / 2.0)) & (
        np.abs(local_y) <= max(float(size[1]) / 2.0, spec.resolution / 2.0)
    )


def _check_instance(instance: PredictedInstance, spec: BEVSpec) -> None:
    if np.asarray(instance.mask_bev).shape != spec.shape:
        raise ValueError(f"instance mask must have shape {spec.shape}")
    if not np.asarray(instance.mask_bev).dtype == np.bool_:
        raise ValueError("instance mask must be boolean")
    _vector(instance.xy, 2, "instance.xy")
    _vector(instance.extent, 2, "instance.extent")
    _probabilities(instance.class_probs)


def build_bev_channels(
    instances: Iterable[PredictedInstance],
    *,
    spec: BEVSpec,
    selected_track_ids: set[int] | None,
    brush_xy: np.ndarray | Iterable[float],
    brush_size: np.ndarray | Iterable[float],
    brush_yaw: float,
    tray_bounds: Sequence[float],
    forbidden_rectangles: Iterable[Sequence[float]] = (),
) -> np.ndarray:
    """Build the six float32 channels consumed by the ObjectACT policy.

    Channels are: all occupancy, selected occupancy, unselected occupancy,
    tray signed distance, current brush footprint, and free-space signed
    distance (table boundary plus explicit forbidden rectangles).
    """

    instances = list(instances)
    selected = set(selected_track_ids or set())
    brush = _vector(brush_xy, 2, "brush_xy")
    size = _vector(brush_size, 2, "brush_size")
    if np.any(size <= 0.0):
        raise ValueError("brush_size must be positive")
    all_occupancy = np.zeros(spec.shape, dtype=np.float32)
    selected_occupancy = np.zeros(spec.shape, dtype=np.float32)
    unselected_occupancy = np.zeros(spec.shape, dtype=np.float32)
    for instance in instances:
        _check_instance(instance, spec)
        mask = np.asarray(instance.mask_bev, dtype=bool)
        confidence = _bounded(instance.confidence, "confidence")
        target = selected_occupancy if int(instance.track_id) in selected else unselected_occupancy
        target[mask] = np.maximum(target[mask], confidence)
        all_occupancy[mask] = np.maximum(all_occupancy[mask], confidence)
    centres = spec.centres()
    tray_distance = rectangle_signed_distance(centres.reshape(-1, 2), tray_bounds).reshape(spec.shape)
    free_distance = rectangle_signed_distance(
        centres.reshape(-1, 2), (spec.x_min, spec.x_max, spec.y_min, spec.y_max)
    ).reshape(spec.shape)
    for forbidden in forbidden_rectangles:
        # The rectangle SDF is positive inside; negating it makes forbidden
        # cells negative and free cells positive, as required by channel 5.
        free_distance = np.minimum(
            free_distance,
            -rectangle_signed_distance(centres.reshape(-1, 2), forbidden).reshape(spec.shape),
        )
    channels = np.stack(
        [
            all_occupancy,
            selected_occupancy,
            unselected_occupancy,
            np.clip(tray_distance, -0.20, 0.20),
            _oriented_rectangle_mask(spec, brush, size, brush_yaw).astype(np.float32),
            np.clip(free_distance, -0.10, 0.10),
        ],
        axis=0,
    ).astype(np.float32)
    return channels


@dataclass
class _TrackState:
    track_id: int
    instance: PredictedInstance
    last_timestamp: float
    velocity_xy: np.ndarray
    angular_velocity: float


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


class TrackManager:
    """Small geometry-only tracker for predicted instances from both cameras."""

    def __init__(self, *, max_match_distance: float = 0.08, max_age_s: float = 0.5):
        self.max_match_distance = float(max_match_distance)
        self.max_age_s = float(max_age_s)
        if self.max_match_distance <= 0.0 or self.max_age_s <= 0.0:
            raise ValueError("tracker thresholds must be positive")
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}
        self.last_instances: dict[int, PredictedInstance] = {}

    def _assignment(self, instances: list[PredictedInstance]) -> dict[int, int]:
        pairs = []
        for det_index, instance in enumerate(instances):
            for track_id, state in self._tracks.items():
                distance = float(np.linalg.norm(np.asarray(instance.xy) - np.asarray(state.instance.xy)))
                if distance <= self.max_match_distance:
                    pairs.append((distance, int(track_id), int(det_index)))
        pairs.sort()
        assigned_det: set[int] = set()
        assigned_track: set[int] = set()
        assignment: dict[int, int] = {}
        for _, track_id, det_index in pairs:
            if det_index in assigned_det or track_id in assigned_track:
                continue
            assignment[det_index] = track_id
            assigned_det.add(det_index)
            assigned_track.add(track_id)
        return assignment

    def update(self, detections: Iterable[PredictedInstance], *, timestamp: float) -> list[TrackedObject]:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._tracks and timestamp < max(state.last_timestamp for state in self._tracks.values()):
            raise ValueError("tracker timestamps must be non-decreasing")
        instances = sorted(
            list(detections),
            key=lambda instance: (float(instance.xy[0]), float(instance.xy[1]), -float(instance.confidence)),
        )
        for instance in instances:
            if np.asarray(instance.mask_bev).ndim != 2:
                raise ValueError("tracker detections must have 2-D BEV masks")
        previous_tracks = dict(self._tracks)
        assignment = self._assignment(instances)
        updated: dict[int, _TrackState] = {}
        for det_index, instance in enumerate(instances):
            track_id = assignment.get(det_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                velocity = np.zeros(2, dtype=float)
                angular_velocity = 0.0
            else:
                previous = previous_tracks[track_id]
                dt = max(timestamp - previous.last_timestamp, 1e-9)
                velocity = (np.asarray(instance.xy, dtype=float) - np.asarray(previous.instance.xy, dtype=float)) / dt
                angular_velocity = _wrap_angle(float(instance.yaw) - float(previous.instance.yaw)) / dt
            assigned = replace(instance, track_id=int(track_id))
            updated[int(track_id)] = _TrackState(
                track_id=int(track_id),
                instance=assigned,
                last_timestamp=timestamp,
                velocity_xy=velocity,
                angular_velocity=angular_velocity,
            )
        for track_id, previous in self._tracks.items():
            if track_id in updated:
                continue
            age = timestamp - previous.last_timestamp
            if age <= self.max_age_s:
                updated[track_id] = previous
        self._tracks = updated
        self.last_instances = {track_id: state.instance for track_id, state in updated.items()}
        result: list[TrackedObject] = []
        for track_id in sorted(updated):
            state = updated[track_id]
            current = state.instance
            previous_state = previous_tracks.get(track_id)
            is_fresh = track_id in assignment.values()
            previous_instance = previous_state.instance if is_fresh and previous_state else None
            age = max(0.0, timestamp - state.last_timestamp)
            result.append(
                TrackedObject(
                    track_id=track_id,
                    class_probs=np.asarray(current.class_probs, dtype=np.float32),
                    xy=np.asarray(current.xy, dtype=float),
                    extent=np.asarray(current.extent, dtype=float),
                    yaw=float(current.yaw),
                    previous_xy=None if previous_instance is None else previous_instance.xy,
                    previous_extent=None if previous_instance is None else previous_instance.extent,
                    previous_yaw=None if previous_instance is None else previous_instance.yaw,
                    velocity_xy=np.asarray(state.velocity_xy, dtype=float),
                    angular_velocity=float(state.angular_velocity),
                    tray_overlap=float(current.tray_overlap),
                    full_inside=bool(current.full_inside),
                    confidence=float(current.confidence),
                    overhead_visibility=float(current.overhead_visibility),
                    wrist_visibility=float(current.wrist_visibility),
                    staleness=min(1.0, age / self.max_age_s),
                    valid=True,
                )
            )
        return result
