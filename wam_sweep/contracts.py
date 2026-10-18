"""Strict visual-policy boundaries; simulator labels live under ``supervision``.

No environment, goal, joint state or wrench is admitted by OnlineObservation.
Arrays are copied on construction; consumers must revalidate mutable payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch


CAMERA_VIEWS = (
    "corner_xminus_yminus", "corner_xminus_yplus", "corner_xplus_yminus",
    "corner_xplus_yplus", "wrist",
)
PHASES = ("prepare", "handoff", "sweep", "terminated")
EPISODE_ARMS = ("act", "wam", "no-prediction")
RESULT_STATUSES = (
    "success", "task_failure", "perception_invalid", "model_invalid", "safety_stop",
    "infrastructure_error", "incomplete",
)
_FP32_MAX = float(np.finfo(np.float32).max)


def _scalar(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if (not np.isfinite(number) or abs(number) > _FP32_MAX
            or (minimum is not None and number < minimum)):
        raise ValueError(f"invalid {name}")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and numeric values")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(array).all() or np.any(np.abs(array) > _FP32_MAX):
        raise ValueError(f"{name} must be finite and representable in FP32")
    return array


def _probabilities(value: Any, size: int, name: str) -> np.ndarray:
    array = _array(value, (size,), name)
    if np.any(array < 0) or np.any(array > 1) or not np.isclose(array.sum(), 1., atol=1e-6, rtol=0):
        raise ValueError(f"{name} must be a probability simplex")
    return array


def _boolean(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


def _tensor(value: Any, shape: tuple[int, ...], name: str, *, mask: bool = False) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise TypeError(f"{name} must be an FP32 tensor")
    if tuple(value.shape) != shape or not torch.isfinite(value).all():
        raise ValueError(f"invalid shape or nonfinite {name}; expected {shape}")
    if mask and not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must be binary")


def _batch_size(value: torch.Tensor) -> int:
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] < 1:
        raise ValueError("a nonempty batch dimension is required")
    return value.shape[0]


def _same_device(*values: torch.Tensor) -> None:
    if len({value.device for value in values}) != 1:
        raise ValueError("all tensors must share a device")


@dataclass(frozen=True, slots=True)
class ObjectBelief:
    center_xy: np.ndarray
    velocity_xy: np.ndarray
    half_extents_xy: np.ndarray
    yaw: float
    angular_velocity: float
    shape_probabilities: np.ndarray
    visibility: float
    state_probabilities: np.ndarray
    position_covariance: np.ndarray
    appearance: np.ndarray
    track_id: int
    valid: bool

    def __post_init__(self) -> None:
        for name, shape in (("center_xy", (2,)), ("velocity_xy", (2,)),
                            ("half_extents_xy", (2,)), ("position_covariance", (2, 2)),
                            ("appearance", (16,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.half_extents_xy <= 0):
            raise ValueError("half extents must be positive")
        cov = self.position_covariance
        if not np.allclose(cov, cov.T, atol=1e-10, rtol=0) or np.linalg.eigvalsh(cov).min() < -1e-10:
            raise ValueError("position covariance must be symmetric positive semidefinite")
        for name in ("yaw", "angular_velocity", "visibility"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if not 0 <= self.visibility <= 1:
            raise ValueError("visibility must be in [0, 1]")
        for name, size in (("shape_probabilities", 2), ("state_probabilities", 3)):
            object.__setattr__(self, name, _probabilities(getattr(self, name), size, name))
        _integer(self.track_id, "track_id", 0)
        _boolean(self.valid, "valid")


@dataclass(frozen=True, slots=True)
class RobotContext:
    tcp_xyz: np.ndarray
    yaw: float
    linear_velocity: np.ndarray
    yaw_velocity: float
    phase: str
    residual: np.ndarray
    residual_rate: np.ndarray
    episode_time: float
    contact_ready: bool
    reference_valid: bool
    visual_entry_age: float
    visual_entry_valid: bool

    def __post_init__(self) -> None:
        for name in ("tcp_xyz", "linear_velocity", "residual", "residual_rate"):
            object.__setattr__(self, name, _array(getattr(self, name), (3,), name))
        for name in ("yaw", "yaw_velocity", "episode_time", "visual_entry_age"):
            minimum = 0. if name in ("episode_time", "visual_entry_age") else None
            object.__setattr__(self, name, _scalar(getattr(self, name), name, minimum))
        if self.phase not in PHASES:
            raise ValueError("unknown robot phase")
        for name in ("contact_ready", "reference_valid", "visual_entry_valid"):
            _boolean(getattr(self, name), name)
        if not self.visual_entry_valid and self.visual_entry_age != 0:
            raise ValueError("invalid visual entry age must be zero")


@dataclass(frozen=True, slots=True)
class OnlineObservation:
    timestamp: float
    objects: tuple[ObjectBelief, ...]
    robot: RobotContext
    images: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _scalar(self.timestamp, "timestamp", 0.))
        if not isinstance(self.objects, (tuple, list)) or len(self.objects) > 6:
            raise ValueError("at most six visual objects are allowed")
        if any(type(item) is not ObjectBelief for item in self.objects):
            raise TypeError("objects must be ObjectBelief, never raw truth or backend objects")
        objects = tuple(replace(item) for item in self.objects)
        if len({item.track_id for item in objects}) != len(objects):
            raise ValueError("duplicate track IDs")
        if type(self.robot) is not RobotContext:
            raise TypeError("robot must be RobotContext")
        if not isinstance(self.images, Mapping) or set(self.images) != set(CAMERA_VIEWS):
            raise ValueError("exactly the five frozen camera views are required")
        images = {}
        for name in CAMERA_VIEWS:
            array = self.images[name]
            if (not isinstance(array, np.ndarray) or array.dtype != np.uint8
                    or array.ndim != 3 or array.shape[-1] != 3 or min(array.shape) < 1):
                raise ValueError("images must be nonempty uint8 HWC RGB arrays")
            images[name] = array.copy()
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "robot", replace(self.robot))
        object.__setattr__(self, "images", images)


class supervision:
    """Label-only namespace. Never an accepted online input."""

    @dataclass(frozen=True, slots=True)
    class RawTruth:
        fields: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class HistoryBatch:
    objects: torch.Tensor
    brush: torch.Tensor
    context: torch.Tensor
    object_valid: torch.Tensor
    history_valid: torch.Tensor

    def __post_init__(self) -> None:
        b = _batch_size(self.objects)
        for name, shape in (("objects", (b, 11, 6, 34)), ("brush", (b, 11, 9)),
                            ("context", (b, 11, 15)), ("object_valid", (b, 11, 6)),
                            ("history_valid", (b, 11))):
            _tensor(getattr(self, name), shape, name, mask=name.endswith("valid"))
        _same_device(self.objects, self.brush, self.context, self.object_valid, self.history_valid)
        if torch.any(self.object_valid > self.history_valid.unsqueeze(-1)):
            raise ValueError("an invalid history frame cannot contain valid objects")
        if (torch.any(self.objects * (1 - self.object_valid.unsqueeze(-1)) != 0)
                or torch.any(self.brush * (1 - self.history_valid.unsqueeze(-1)) != 0)
                or torch.any(self.context * (1 - self.history_valid.unsqueeze(-1)) != 0)):
            raise ValueError("masked features must be zero")


@dataclass(frozen=True, slots=True)
class CandidateProgram:
    knots: np.ndarray

    def __post_init__(self) -> None:
        knots = _array(self.knots, (5, 3), "knots")
        if np.any(np.abs(knots) > 1):
            raise ValueError("knots must be normalized to [-1, 1]")
        object.__setattr__(self, "knots", knots)


@dataclass(frozen=True, slots=True)
class ReferencePreview:
    """Absolute command targets on the 25 Hz grid, including command time t.

    Resulting state predictions use t + (j + 1) * 0.04, a separate clock.
    """
    targets: np.ndarray
    times: np.ndarray
    valid: np.ndarray
    issued_at: float
    source_hash: str

    def __post_init__(self) -> None:
        for name, shape in (("targets", (25, 4)), ("times", (25,)), ("valid", (25,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        object.__setattr__(self, "issued_at", _scalar(self.issued_at, "issued_at", 0.))
        if np.any(self.times < self.issued_at):
            raise ValueError("reference command times cannot precede issuance")
        offset_ticks = (self.times[0] - self.issued_at) * 25
        if (not np.allclose(np.diff(self.times), .04, atol=1e-7, rtol=0)
                or abs(offset_ticks - round(offset_ticks)) > 1e-6):
            raise ValueError("reference command times must follow the issued 25 Hz grid")
        if not np.isin(self.valid, (0, 1)).all():
            raise ValueError("reference valid must be binary")
        _text(self.source_hash, "source_hash")


@dataclass(frozen=True, slots=True)
class WorldPrediction:
    object_mean: torch.Tensor
    object_log_std: torch.Tensor
    state_logits: torch.Tensor
    shape_logits: torch.Tensor
    visibility_logits: torch.Tensor
    brush: torch.Tensor
    brush_log_std: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        b = _batch_size(self.object_mean)
        for name, shape in (("object_mean", (b, 25, 6, 9)), ("object_log_std", (b, 25, 6, 4)),
                            ("state_logits", (b, 25, 6, 3)), ("shape_logits", (b, 25, 6, 2)),
                            ("visibility_logits", (b, 25, 6, 1)), ("brush", (b, 25, 9)),
                            ("brush_log_std", (b, 25, 9)), ("valid", (b, 25))):
            _tensor(getattr(self, name), shape, name, mask=name == "valid")
        _same_device(self.object_mean, self.object_log_std, self.state_logits, self.shape_logits,
                     self.visibility_logits, self.brush, self.brush_log_std, self.valid)


@dataclass(frozen=True, slots=True)
class TaskScore:
    success_logit: torch.Tensor
    final_count_logits: torch.Tensor
    calibration_hash: str

    def __post_init__(self) -> None:
        b = _batch_size(self.success_logit)
        _tensor(self.success_logit, (b, 1), "success_logit")
        _tensor(self.final_count_logits, (b, 7), "final_count_logits")
        _same_device(self.success_logit, self.final_count_logits)
        _text(self.calibration_hash, "calibration_hash")


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    status: str
    native_reason: str

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError("unknown episode result category")
        _text(self.native_reason, "native_reason")

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    parent_id: str
    seed: int
    target_count: int
    split: str
    phase: int
    arm: str
    attempt_index: int

    def __post_init__(self) -> None:
        for name in ("parent_id", "split", "arm"):
            _text(getattr(self, name), name)
        if self.arm not in EPISODE_ARMS:
            raise ValueError("arm must be act, wam or no-prediction")
        _integer(self.seed, "seed", 0)
        _integer(self.target_count, "target_count", 1, 6)
        _integer(self.phase, "phase", -1, 2)
        _integer(self.attempt_index, "attempt_index", 0)


@dataclass(frozen=True, slots=True)
class InputIdentity:
    root: str
    manifest_hash: str
    model_hash: str
    native_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("root", "manifest_hash", "model_hash"):
            _text(getattr(self, name), name)
        required = {"perception", "control", "preprocessing", "config"}
        if not isinstance(self.native_hashes, Mapping) or not required <= self.native_hashes.keys():
            raise ValueError("native dependency hashes for perception/control/preprocessing/config required")
        for key, value in self.native_hashes.items():
            _text(key, "dependency name")
            _text(value, "dependency hash")
        object.__setattr__(self, "native_hashes", dict(self.native_hashes))


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    spec_hash: str
    plan_hash: str
    config_hash: str
    input_identity: InputIdentity
    split_hash: str

    def __post_init__(self) -> None:
        for name in ("run_id", "spec_hash", "plan_hash", "config_hash", "split_hash"):
            _text(getattr(self, name), name)
        if type(self.input_identity) is not InputIdentity:
            raise TypeError("input_identity must be InputIdentity")
        object.__setattr__(self, "input_identity", replace(self.input_identity))


@dataclass(frozen=True, slots=True)
class ModelVersion:
    phase: int
    kind: str
    checkpoint_path: str
    model_hash: str
    perception_hash: str
    dynamics_hash: str
    continuation_hash: str
    input_identity: InputIdentity

    def __post_init__(self) -> None:
        _integer(self.phase, "phase", -1, 2)
        if not ((self.kind == "act" and self.phase == -1)
                or (self.kind == "wam" and self.phase in (0, 1, 2))):
            raise ValueError("ACT uses phase -1; WAM uses phase 0, 1 or 2")
        for name in ("checkpoint_path", "model_hash", "perception_hash", "dynamics_hash", "continuation_hash"):
            _text(getattr(self, name), name)
        if type(self.input_identity) is not InputIdentity:
            raise TypeError("model versions require frozen input dependencies")
        object.__setattr__(self, "input_identity", replace(self.input_identity))


@runtime_checkable
class Controller(Protocol):
    def reset(self, initial: OnlineObservation) -> None: ...
    def observe(self, observation: OnlineObservation) -> None: ...
    def act(self, reference: ReferencePreview) -> Any: ...
    def snapshot(self) -> Mapping[str, Any]: ...
    def restore(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class BackendFactory(Protocol):
    def __call__(self, spec: EpisodeSpec) -> Any: ...


@runtime_checkable
class ControllerFactory(Protocol):
    def __call__(self, version: ModelVersion) -> Controller: ...
