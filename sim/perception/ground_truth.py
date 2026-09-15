"""Ground-truth perception backend.

Uses the simulator's true component poses.  **Debug only** -- it exists so the
controller and the planners can be validated without perception error in the
loop.  Every reported experiment should also be run with the ``vision`` backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..environments.layout import in_target_region
from .base import Perception, SceneObservation


class GroundTruthPerception(Perception):
    name = "ground_truth"

    def observe(self, env, rng: Optional[np.random.Generator] = None) -> SceneObservation:
        positions = env.component_positions()
        collected = in_target_region(positions[:, :2], env.cfg.target) if positions.size \
            else np.zeros(0, dtype=bool)
        remaining = positions[~collected][:, :2] if positions.size else np.zeros((0, 2))

        radii = np.array([item["nominal_radius"] for item, keep
                          in zip(env.layout, ~collected) if keep], dtype=float) \
            if positions.size else np.zeros(0)
        if remaining.size == 0:
            remaining = np.zeros((0, 2))
            radii = np.zeros(0)

        occupancy = self.grid.rasterize_disks(remaining, radii) if remaining.size \
            else self.grid.empty()
        return SceneObservation(
            t=env.time,
            points=np.asarray(remaining, dtype=float).reshape(-1, 2),
            counts=np.ones(remaining.shape[0], dtype=float),
            areas=np.pi * np.asarray(radii, dtype=float) ** 2,
            occupancy=occupancy,
            grid=self.grid,
            tcp=env.tcp(),
            backend=self.name,
            n_remaining_true=int(remaining.shape[0]),
        )
