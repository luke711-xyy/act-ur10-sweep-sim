"""Planner A -- predefined directional full-cover baseline.

Properties required by the specification:

* the workspace is divided into parallel lanes
* every lane is swept from right (+X) to left (-X), i.e. towards the fixed tray
* the tool is lifted before returning to the next lane (guaranteed by the
  controller's APPROACH phase, which always lifts to ``workspace.z_travel``
  before travelling -- the return motion is therefore never in contact)
* the sequence finishes with a fixed consolidation stroke in front of the tray
* **the planner never looks at the component positions**: it is fully
  determined by the configuration, so it produces the same stroke list for
  every seed and every layout.

Lanes whose y lies outside the tray opening are aimed diagonally so that the
stroke terminates inside the tray; this is the only "funnelling" the baseline
does and it is still layout-independent.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Planner, SweepStroke
from .geometry_utils import pusher_width, stroke_yaw


class FixedCoverPlanner(Planner):
    name = "fixed"
    uses_perception = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.strokes: List[SweepStroke] = self._build_strokes()

    # ------------------------------------------------------------------ setup
    def _lane_bounds(self):
        p = self.cfg.planner
        fixed_cfg = p.get("fixed", None)
        if fixed_cfg is not None and "y_min" in fixed_cfg:
            return float(fixed_cfg.y_min), float(fixed_cfg.y_max)
        spawn = self.cfg.components.spawn
        pad = pusher_width(self.cfg) / 2.0
        ws = self.cfg.workspace
        return (max(float(spawn.y_min) - pad, float(ws.y_min)),
                min(float(spawn.y_max) + pad, float(ws.y_max)))

    def _build_strokes(self) -> List[SweepStroke]:
        cfg = self.cfg
        p = cfg.planner
        ws = cfg.workspace
        tgt = cfg.target

        width = pusher_width(cfg)
        pitch = max(width * (1.0 - float(p.lane_overlap)), 1e-3)
        y_lo, y_hi = self._lane_bounds()
        span = max(y_hi - y_lo, 0.0)
        n_lanes = max(1, int(np.ceil(span / pitch)) + 1)
        lane_ys = np.linspace(y_hi, y_lo, n_lanes)   # sweep from +y to -y

        x_start = float(ws.x_max)
        x_end = float(p.stroke_end_x)
        tray_margin = float(p.get("tray_margin", 0.03))
        y_cap_lo = float(tgt.y_min) + tray_margin
        y_cap_hi = float(tgt.y_max) - tray_margin
        align_yaw = bool(p.get("align_yaw", True))

        strokes: List[SweepStroke] = []
        for index, y in enumerate(lane_ys):
            y_target = float(np.clip(y, y_cap_lo, y_cap_hi))
            stroke = SweepStroke(x_start, float(y), x_end, y_target,
                                 meta={"kind": "lane", "lane": index})
            if align_yaw:
                stroke.yaw = stroke_yaw(stroke.end - stroke.start)
            strokes.append(stroke.clipped(ws))

        if bool(p.consolidation):
            x_mouth = float(tgt.x_max) + float(p.get("consolidation_lead", 0.12))
            consolidation = SweepStroke(
                float(np.clip(x_mouth, ws.x_min, ws.x_max)), 0.0, x_end, 0.0,
                meta={"kind": "consolidation"},
            )
            strokes.append(consolidation.clipped(ws))
        return strokes

    # ------------------------------------------------------------------- api
    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        super().reset(rng)
        self.strokes = self._build_strokes()

    def plan(self, observation) -> Optional[SweepStroke]:
        """Return the next predefined stroke; ``observation`` is ignored by design."""
        if self.stroke_count >= len(self.strokes):
            return None
        if self.is_exhausted():
            return None
        return self.strokes[self.stroke_count]

    @property
    def n_planned_strokes(self) -> int:
        return len(self.strokes)
