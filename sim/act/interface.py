"""The explicit observation/action contract used by ACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ACTObservationSpec:
    image_size: tuple[int, int] = (320, 320)
    action_dim: int = 4
    # joints(6) + tcp xyz(3) + sin/cos(yaw)(2) + wrench(6) +
    # [physical_total, target_count, fully_collected](3) + contact_latched(1),
    # for current and previous frames: 21 * 2.
    state_dim: int = 42
    chunk_size: int = 25
    execute_steps: int = 5


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    width, height = size
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def action_deltas_to_absolute(start_reference: np.ndarray,
                              deltas: np.ndarray) -> np.ndarray:
    """Convert ACT's 4-D delta chunk into executor's absolute targets.

    The learned interface is fixed-frame ``[dx, dy, dz, dyaw]``.  Before
    contact all four dimensions are executed.  After the force latch, the
    executor deliberately ignores the absolute Z column produced here and
    substitutes the admittance command; retaining it in the chunk keeps the
    policy/output schema identical across the whole episode.
    """
    start = np.asarray(start_reference, dtype=np.float32).reshape(4)
    values = np.asarray(deltas, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            f"ACT action chunk must have shape (chunk, 4), got {values.shape}"
        )
    xyz = start[:3][None, :] + np.cumsum(values[:, :3], axis=0)
    yaw = start[3] + np.cumsum(values[:, 3])
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return np.concatenate((xyz, yaw[:, None]), axis=1).astype(np.float32)


class ACTObservationBuilder:
    """Build causal ACT observations from a live MuJoCo environment.

    Object positions are intentionally absent.  Physical total, requested
    target count and current fully-collected count are the only structured task
    signals in this first policy stage and are simulator truth by design.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        size = tuple(int(v) for v in cfg.act.image_size)
        self.spec = ACTObservationSpec(
            image_size=(size[0], size[1]),
            action_dim=int(cfg.act.action_dim),
            state_dim=42,
            chunk_size=int(cfg.act.chunk_size),
            execute_steps=int(cfg.act.execute_steps),
        )
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def _state(self, env, contact_latched: bool = False) -> np.ndarray:
        joints = np.asarray(env.ee.joint_state(), dtype=np.float32).ravel()
        tcp = np.asarray(env.tcp(), dtype=np.float32).ravel()
        yaw = float(env.ee.tcp_yaw())
        yaw_features = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        wrench = np.asarray(env.wrench(), dtype=np.float32).reshape(6)
        total = len(env.layout)
        target = int(np.clip(int(self.cfg.get_path("task.target_count", total)), 1, max(1, total)))
        counts = np.array([total, target, int(env.collected_mask().sum())], dtype=np.float32)
        current = np.concatenate((
            joints, tcp, yaw_features, wrench, counts,
            np.asarray([float(bool(contact_latched))], dtype=np.float32),
        ))
        previous = current if self._previous is None else self._previous
        self._previous = current.copy()
        return np.concatenate((current, previous)).astype(np.float32)

    def observe(self, env, contact_latched: bool = False) -> dict[str, Any]:
        size = self.spec.image_size
        overhead = _resize_rgb(env.render_rgb("overhead_cam", size=size), size)
        wrist = _resize_rgb(env.render_wrist_rgb(size=size), size)
        return {
            "observation.images.overhead": np.transpose(overhead, (2, 0, 1)).astype(np.float32) / 255.0,
            "observation.images.wrist": np.transpose(wrist, (2, 0, 1)).astype(np.float32) / 255.0,
            "observation.state": self._state(env, contact_latched=contact_latched),
            "observation.environment_state": np.array([
                len(env.layout),
                int(np.clip(int(self.cfg.get_path("task.target_count", len(env.layout))),
                            1, max(1, len(env.layout)))),
                int(env.collected_mask().sum()),
            ], dtype=np.float32),
            "contact_latched": bool(contact_latched),
            "t": float(env.time),
        }

    @staticmethod
    def torch_batch(observation: dict[str, Any], device: str | None = None):
        import torch

        batch = {}
        for key, value in observation.items():
            # Timestamp and the convenience scalar are executor metadata.
            # contact_latched is already encoded in the 42-D current/previous
            # state and must not become an undeclared extra LeRobot feature.
            if key in {"t", "contact_latched"}:
                continue
            # Leave batching, device placement and normalization to the
            # official LeRobot ACT preprocessor.  This method now only turns
            # simulator observations into tensors at the raw policy boundary.
            tensor = torch.as_tensor(value)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            batch[key] = tensor
        return batch
