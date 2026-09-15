"""Perception interface shared by the ground-truth and conventional-vision backends.

Both backends return the *same* :class:`SceneObservation`, so a planner cannot
tell which one it is running against.  That is what makes the "does vision
change the answer?" comparison meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GridSpec:
    """Axis-aligned occupancy grid over the table plane."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    res: float = 0.01

    @property
    def nx(self) -> int:
        return max(1, int(np.ceil((self.x_max - self.x_min) / self.res)))

    @property
    def ny(self) -> int:
        return max(1, int(np.ceil((self.y_max - self.y_min) / self.res)))

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def cell_area(self) -> float:
        return float(self.res * self.res)

    def to_index(self, xy: np.ndarray):
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        ix = np.floor((xy[:, 0] - self.x_min) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y_min) / self.res).astype(int)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy, ix, valid

    def to_xy(self, iy, ix) -> np.ndarray:
        x = self.x_min + (np.asarray(ix, dtype=float) + 0.5) * self.res
        y = self.y_min + (np.asarray(iy, dtype=float) + 0.5) * self.res
        return np.stack([x, y], axis=-1)

    def empty(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=bool)

    def rasterize_disks(self, points: np.ndarray, radii) -> np.ndarray:
        """Mark every cell whose centre falls within ``radius`` of a point."""
        grid = self.empty()
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.size == 0:
            return grid
        radii = np.broadcast_to(np.asarray(radii, dtype=float).ravel(), (points.shape[0],))
        iy_all, ix_all = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        centres = self.to_xy(iy_all, ix_all)
        for point, radius in zip(points, radii):
            d2 = (centres[..., 0] - point[0]) ** 2 + (centres[..., 1] - point[1]) ** 2
            grid |= d2 <= max(radius, self.res * 0.5) ** 2
        return grid

    @classmethod
    def from_config(cls, cfg, res: float = 0.01) -> "GridSpec":
        ws = cfg.workspace
        return cls(float(ws.x_min), float(ws.x_max), float(ws.y_min), float(ws.y_max), res)


@dataclass
class SceneObservation:
    """What a planner is allowed to see before choosing the next stroke."""

    t: float
    points: np.ndarray                      # (M, 2) table-plane positions
    counts: np.ndarray                      # (M,)  estimated component count per point
    areas: np.ndarray                       # (M,)  estimated footprint area [m^2]
    occupancy: np.ndarray                   # (ny, nx) bool
    grid: GridSpec
    tcp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    backend: str = "ground_truth"
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None       # (H, W) bool image-space mask
    n_remaining_true: Optional[int] = None  # ground-truth bookkeeping, never used by planners

    @property
    def n_detected(self) -> int:
        return int(self.points.shape[0]) if self.points.size else 0

    @property
    def estimated_total(self) -> float:
        return float(np.sum(self.counts)) if self.counts.size else 0.0


class Perception:
    name: str = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.grid = GridSpec.from_config(cfg, res=float(cfg.perception.get("grid_res", 0.01)))

    def observe(self, env, rng: Optional[np.random.Generator] = None) -> SceneObservation:
        raise NotImplementedError  # pragma: no cover - interface


def build_perception(cfg, name: Optional[str] = None) -> Perception:
    from .ground_truth import GroundTruthPerception
    from .vision import ConventionalVisionPerception

    name = name or str(cfg.perception.backend)
    table = {"ground_truth": GroundTruthPerception, "vision": ConventionalVisionPerception}
    if name not in table:
        raise ValueError(f"unknown perception backend {name!r}; expected one of {sorted(table)}")
    return table[name](cfg)
