"""Planner interface and the high-level sweep action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SweepStroke:
    """One planar sweeping stroke -- the high-level action of this system.

    The 5-D vector ``[x_start, y_start, x_end, y_end, yaw]`` is exactly the
    action format exported for later ACT training.  The Z axis is deliberately
    *not* part of the action: normal-force regulation stays inside the hybrid
    controller.
    """

    x_start: float
    y_start: float
    x_end: float
    y_end: float
    yaw: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> np.ndarray:
        return np.array([self.x_start, self.y_start], dtype=float)

    @property
    def end(self) -> np.ndarray:
        return np.array([self.x_end, self.y_end], dtype=float)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        vec = self.end - self.start
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else np.array([-1.0, 0.0])

    def to_action(self) -> np.ndarray:
        return np.array([self.x_start, self.y_start, self.x_end, self.y_end, self.yaw],
                        dtype=np.float32)

    @classmethod
    def from_action(cls, action: Sequence[float]) -> "SweepStroke":
        a = np.asarray(action, dtype=float).ravel()
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]),
                   float(a[4]) if a.size > 4 else 0.0)

    def clipped(self, ws) -> "SweepStroke":
        """Clamp both endpoints into the sweeping workspace."""
        xs = float(np.clip(self.x_start, ws.x_min, ws.x_max))
        ys = float(np.clip(self.y_start, ws.y_min, ws.y_max))
        xe = float(np.clip(self.x_end, ws.x_min, ws.x_max))
        ye = float(np.clip(self.y_end, ws.y_min, ws.y_max))
        return SweepStroke(xs, ys, xe, ye, self.yaw, dict(self.meta))


ACTION_DIM = 5
ACTION_LABELS = ("action_x_start", "action_y_start", "action_x_end", "action_y_end", "action_yaw")


class Planner:
    """Base class for stroke planners.

    A planner is queried once per stroke with the latest observation.  Planner A
    ignores the observation entirely (open-loop full-cover baseline); Planner B
    re-plans from it after every stroke.
    """

    name: str = "base"
    uses_perception: bool = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.stroke_count = 0

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self.stroke_count = 0

    def plan(self, observation) -> Optional[SweepStroke]:  # pragma: no cover - interface
        raise NotImplementedError

    def notify_stroke_done(self, stroke: SweepStroke, info: Dict[str, Any]) -> None:
        self.stroke_count += 1

    def is_exhausted(self) -> bool:
        return self.stroke_count >= int(self.cfg.planner.max_strokes)


def build_planner(cfg, name: Optional[str] = None) -> Planner:
    from .fixed_cover import FixedCoverPlanner
    from .global_sweep import GlobalSweepPlanner
    from .visual_greedy import VisualGreedyPlanner

    name = name or str(cfg.planner.name)
    table = {"fixed": FixedCoverPlanner, "global_sweep": GlobalSweepPlanner,
             "visual_greedy": VisualGreedyPlanner}
    if name not in table:
        raise ValueError(f"unknown planner {name!r}; expected one of {sorted(table)}")
    return table[name](cfg)


PLANNER_NAMES: List[str] = ["fixed", "global_sweep", "visual_greedy"]
