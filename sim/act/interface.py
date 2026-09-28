"""The explicit observation/action contract used by ACT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


ACT_SCHEMA_VERSION = 5
ACT_STATE_SAMPLE_DIM = 18
ACT_STATE_DIM = ACT_STATE_SAMPLE_DIM * 2


@dataclass(frozen=True)
class ACTObservationSpec:
    image_size: tuple[int, int] = (320, 320)
    action_dim: int = 4
    # joints(6) + tcp xyz(3) + sin/cos(yaw)(2) + wrench(6) +
    # contact_latched(1), for current and 200 ms-previous samples: 18 * 2.
    # Task counts are supplied once as observation.environment_state.
    state_dim: int = ACT_STATE_DIM
    history_lag_seconds: float = 0.2
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

    Object positions are intentionally absent.  Robot state is 18-D per
    sample; current and 200 ms-old samples form ``observation.state`` (36-D).
    Physical total, target count and fully-collected count are sent once in the
    separate 3-D ``observation.environment_state`` feature.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        size = tuple(int(v) for v in cfg.act.image_size)
        self.spec = ACTObservationSpec(
            image_size=(size[0], size[1]),
            action_dim=int(cfg.act.action_dim),
            state_dim=ACT_STATE_DIM,
            history_lag_seconds=float(cfg.act.get("state_history_seconds", 0.2)),
            chunk_size=int(cfg.act.chunk_size),
            execute_steps=int(cfg.act.execute_steps),
        )
        self._history_period = 1.0 / max(float(cfg.act.action_hz), 1e-6)
        if self.spec.history_lag_seconds < 0.0:
            raise ValueError("ACT state history lag must be non-negative")
        self._history: deque[tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._history.clear()

    def _state_sample(self, env, contact_latched: bool = False) -> np.ndarray:
        joints = np.asarray(env.ee.joint_state(), dtype=np.float32).ravel()
        tcp = np.asarray(env.tcp(), dtype=np.float32).ravel()
        yaw = float(env.ee.tcp_yaw())
        yaw_features = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        wrench = np.asarray(env.wrench(), dtype=np.float32).reshape(6)
        sample = np.concatenate((
            joints, tcp, yaw_features, wrench,
            np.asarray([float(bool(contact_latched))], dtype=np.float32),
        ))
        if sample.shape != (ACT_STATE_SAMPLE_DIM,):
            raise ValueError(
                f"ACT state sample must have {ACT_STATE_SAMPLE_DIM} values, "
                f"got {sample.shape}"
            )
        return sample.astype(np.float32, copy=False)

    def advance_state_history(self, env, contact_latched: bool = False) -> None:
        """Record a causal state sample at the configured action-rate cadence.

        Inference calls this after each 25 Hz action frame while policy queries
        remain at 5 Hz.  Demo capture calls it from ``observe`` at 25 Hz.  Thus
        both paths can select a sample 200 ms before the current observation.
        """
        now = float(env.time)
        sample = self._state_sample(env, contact_latched=contact_latched)
        if self._history and now < self._history[-1][0] - 1e-9:
            self._history.clear()
        if self._history and abs(now - self._history[-1][0]) <= 1e-9:
            self._history[-1] = (now, sample.copy())
        elif (not self._history
              or now - self._history[-1][0] >= self._history_period - 1e-7):
            self._history.append((now, sample.copy()))

        oldest_needed = now - self.spec.history_lag_seconds
        while len(self._history) > 2 and self._history[1][0] < oldest_needed - 1e-7:
            self._history.popleft()

    def _state(self, env, contact_latched: bool = False) -> np.ndarray:
        current = self._state_sample(env, contact_latched=contact_latched)
        self.advance_state_history(env, contact_latched=contact_latched)
        if not bool(self.cfg.act.get("state_history", True)):
            previous = current
        elif self.spec.history_lag_seconds <= 1e-9:
            previous = current
        else:
            target_time = float(env.time) - self.spec.history_lag_seconds
            candidates = [sample for sample_time, sample in self._history
                          if sample_time <= target_time + 1e-7]
            # At episode start no 200 ms-old sample exists; repeat the initial
            # sample until the requested history interval has elapsed.
            previous = candidates[-1] if candidates else self._history[0][1]
        return np.concatenate((current, previous)).astype(np.float32)

    def observe(self, env, contact_latched: bool = False,
                include_images: bool = True) -> dict[str, Any]:
        environment_state = np.array([
            len(env.layout),
            int(np.clip(int(self.cfg.get_path("task.target_count", len(env.layout))),
                        1, max(1, len(env.layout)))),
            int(env.collected_mask().sum()),
        ], dtype=np.float32)
        observation = {
            "observation.state": self._state(env, contact_latched=contact_latched),
            "observation.environment_state": environment_state,
            "contact_latched": bool(contact_latched),
            "t": float(env.time),
        }
        if include_images:
            size = self.spec.image_size
            overhead = _resize_rgb(env.render_rgb("overhead_cam", size=size), size)
            wrist = _resize_rgb(env.render_wrist_rgb(size=size), size)
            observation.update({
                "observation.images.overhead": np.transpose(overhead, (2, 0, 1)).astype(np.float32) / 255.0,
                "observation.images.wrist": np.transpose(wrist, (2, 0, 1)).astype(np.float32) / 255.0,
            })
        return observation

    @staticmethod
    def torch_batch(observation: dict[str, Any], device: str | None = None):
        import torch

        batch = {}
        for key, value in observation.items():
            # Timestamp and the convenience scalar are executor metadata.
            # contact_latched is already encoded in the 36-D current/history
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
