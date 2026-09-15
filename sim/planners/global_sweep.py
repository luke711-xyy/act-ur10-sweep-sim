"""Planner C -- global single-sweep baseline ("push-all-at-once").

This is the middle term of the three-way planner comparison, and the one that
isolates *why* segmentation helps:

* Planner A (fixed full cover) -- no perception at all
* **Planner C (this one)** -- the same perception and the same execution stack
  as Planner B, but it plans **one global stroke per iteration** aimed at every
  detected component at once, instead of short strokes aimed at one cluster
* Planner B (visual greedy) -- perception + *segmented* strokes

A and B differ in two things at once (perception *and* stroke segmentation), so
a two-planner comparison cannot attribute the gap. C changes only the
segmentation, so B − C measures segmentation and C − A measures perception.

The loop is otherwise identical to Planner B: execute one stroke, lift,
re-observe, re-plan. Each stroke is aimed at the weighted centroid of *all*
detections, starting behind the right-most one, so parts that sit far off the
centroid lane are expected to be left behind or squeezed sideways -- which is
exactly the effect being measured.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Planner, SweepStroke
from .geometry_utils import point_segment_frames, pusher_width, stroke_yaw


class GlobalSweepPlanner(Planner):
    name = "global_sweep"
    uses_perception = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.capture_halfwidth = pusher_width(cfg) / 2.0 + float(
            cfg.planner.greedy.capture_halfwidth_pad
        )
        self._no_progress = 0

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        super().reset(rng)
        self._no_progress = 0

    def plan(self, observation) -> Optional[SweepStroke]:
        if self.is_exhausted():
            return None
        points = np.atleast_2d(np.asarray(observation.points, dtype=float)).reshape(-1, 2)
        if points.shape[0] == 0:
            return None
        weights = np.asarray(observation.counts, dtype=float).ravel()
        if weights.size != points.shape[0]:
            weights = np.ones(points.shape[0])

        cfg = self.cfg
        p = cfg.planner
        ws = cfg.workspace
        tgt = cfg.target
        c = cfg.controller

        # One lane for everything: the weighted centroid of all detections.
        y_lane = float(np.average(points[:, 1], weights=np.maximum(weights, 1e-6)))
        y_lane = float(np.clip(y_lane, float(ws.y_min), float(ws.y_max)))
        tray_margin = float(p.get("tray_margin", 0.05))
        y_end = float(np.clip(y_lane, float(tgt.y_min) + tray_margin,
                              float(tgt.y_max) - tray_margin))
        x_end = float(p.stroke_end_x)

        x_min_start = float(c.x_release_line) + float(c.release_margin) + 0.06
        x_start = float(np.clip(float(np.max(points[:, 0])) + float(p.stroke_lead_in),
                                x_min_start, float(ws.x_max)))
        start = np.array([x_start, y_lane])
        end = np.array([x_end, y_end])

        along, lateral, _ = point_segment_frames(points, start, end)
        inside = (along >= -0.02) & (along <= 1.02)
        captured = inside & (np.abs(lateral) <= self.capture_halfwidth)

        stroke = SweepStroke(
            float(start[0]), float(start[1]), float(end[0]), float(end[1]),
            stroke_yaw(end - start) if bool(p.get("align_yaw", True)) else 0.0,
            meta={
                "kind": "global",
                "n_detected": int(points.shape[0]),
                "est_collected": float(np.sum(weights[captured])) if captured.any() else 0.0,
                "n_captured": int(captured.sum()),
            },
        )
        return stroke.clipped(ws)

    def notify_stroke_done(self, stroke: SweepStroke, info) -> None:
        super().notify_stroke_done(stroke, info)
        if float(info.get("collected_delta", 0.0)) <= 0.0:
            self._no_progress += 1
        else:
            self._no_progress = 0

    def is_exhausted(self) -> bool:
        if super().is_exhausted():
            return True
        return self._no_progress >= int(self.cfg.planner.greedy.get("max_no_progress", 4))
