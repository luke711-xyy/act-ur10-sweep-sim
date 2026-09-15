"""The explicit observation/action contract used by ACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ACTObservationSpec:
    image_size: tuple[int, int] = (320, 320)
    action_dim: int = 4
    # joints(6) + tcp xyz(3) + yaw(1) + normal force(1) + counts(2)
    # for current and previous frames: 13 * 2.
    state_dim: int = 26
    chunk_size: int = 20
    execute_steps: int = 4


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    width, height = size
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


class ACTObservationBuilder:
    """Build causal ACT observations from a live MuJoCo environment.

    Object positions are intentionally absent.  The two count values are the
    only structured task signal in this first policy stage and are simulator
    truth by design.
    """

    def __init__(self, cfg):
        size = tuple(int(v) for v in cfg.act.image_size)
        self.spec = ACTObservationSpec(
            image_size=(size[0], size[1]),
            action_dim=int(cfg.act.action_dim),
            state_dim=26,
            chunk_size=int(cfg.act.chunk_size),
            execute_steps=int(cfg.act.execute_steps),
        )
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def _state(self, env) -> np.ndarray:
        joints = np.asarray(env.ee.joint_state(), dtype=np.float32).ravel()
        tcp = np.asarray(env.tcp(), dtype=np.float32).ravel()
        yaw = np.array([float(env.ee.tcp_yaw())], dtype=np.float32)
        force = np.array([float(env.normal_force())], dtype=np.float32)
        counts = np.array([len(env.layout), int(env.collected_mask().sum())], dtype=np.float32)
        current = np.concatenate((joints, tcp, yaw, force, counts))
        previous = current if self._previous is None else self._previous
        self._previous = current.copy()
        return np.concatenate((current, previous)).astype(np.float32)

    def observe(self, env) -> dict[str, Any]:
        size = self.spec.image_size
        overhead = _resize_rgb(env.render_rgb("overhead_cam", size=size), size)
        wrist = _resize_rgb(env.render_wrist_rgb(size=size), size)
        return {
            "observation.images.overhead": np.transpose(overhead, (2, 0, 1)).astype(np.float32) / 255.0,
            "observation.images.wrist": np.transpose(wrist, (2, 0, 1)).astype(np.float32) / 255.0,
            "observation.state": self._state(env),
            "observation.environment_state": np.array(
                [len(env.layout), int(env.collected_mask().sum())], dtype=np.float32),
            "t": float(env.time),
        }

    @staticmethod
    def torch_batch(observation: dict[str, Any], device: str):
        import torch

        batch = {}
        for key, value in observation.items():
            if key == "t":
                continue
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            batch[key] = tensor.unsqueeze(0) if tensor.ndim in (1, 3) else tensor
        return batch
