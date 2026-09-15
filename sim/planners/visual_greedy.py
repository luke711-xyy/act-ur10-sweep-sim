"""Planner B -- conventional (non-learned) visual greedy planner.

Loop
----
``observe -> generate candidate strokes -> score -> execute one stroke ->
lift -> re-observe -> re-plan`` until the episode succeeds or the stroke /
time budget is exhausted.

Candidate generation
--------------------
Candidates are straight strokes towards the fixed tray.  Their lateral position
comes from two sources:

* one candidate centred on every detected cluster
* a uniform sweep of ``n_lane_candidates`` lane offsets across the occupied
  y-range (so a single stroke can be found that rakes several clusters at once)

Each candidate starts ``stroke_lead_in`` to the +X side of the furthest cluster
it would capture (never further right than necessary -- that is what the path
length penalty rewards) and ends inside the tray mouth.

Scoring
-------
    score(a) = estimated_collected(a) - lambda * path_length(a) - mu * risk(a)

* ``estimated_collected`` sums the estimated component counts of the clusters
  whose cross-track offset is inside the pusher's capture half-width
* ``path_length`` is the transfer from the current TCP plus the stroke itself
* ``risk`` penalises (i) clusters that would be captured near the outer edge of
  the pusher (they tend to squirt sideways), (ii) clusters that would merely be
  grazed and knocked off-lane, (iii) strokes whose lateral funnelling demand is
  large, and (iv) captured clusters that would end outside the tray opening.

The planner uses **only** the :class:`~sim.perception.base.SceneObservation`, so
it behaves identically on the ground-truth and conventional-vision backends.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Planner, SweepStroke
from .geometry_utils import point_segment_frames, pusher_width, single_link_clusters, stroke_yaw


class VisualGreedyPlanner(Planner):
    name = "visual_greedy"
    uses_perception = True

    def __init__(self, cfg):
        super().__init__(cfg)
        g = cfg.planner.greedy
        self.lambda_path = float(g.lambda_path)
        self.mu_risk = float(g.mu_risk)
        self.cluster_eps = float(g.cluster_eps)
        self.n_lane_candidates = int(g.n_lane_candidates)
        self.yaw_candidates = [float(v) for v in g.yaw_candidates]
        self.capture_halfwidth = pusher_width(cfg) / 2.0 + float(g.capture_halfwidth_pad)
        self.last_candidates: List[dict] = []
        self._no_progress = 0

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        super().reset(rng)
        self.last_candidates = []
        self._no_progress = 0

    # ------------------------------------------------------------- clustering
    def _clusters(self, observation):
        pts = np.atleast_2d(np.asarray(observation.points, dtype=float)).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros(0)
        counts = np.asarray(observation.counts, dtype=float).ravel()
        if counts.size != pts.shape[0]:
            counts = np.ones(pts.shape[0])
        labels = single_link_clusters(pts, self.cluster_eps)
        centres, weights = [], []
        for label in range(labels.max() + 1 if labels.size else 0):
            sel = labels == label
            w = counts[sel]
            centres.append(np.average(pts[sel], axis=0, weights=np.maximum(w, 1e-6)))
            weights.append(float(w.sum()))
        return np.asarray(centres).reshape(-1, 2), np.asarray(weights, dtype=float)

    # --------------------------------------------------------- candidate set
    def _candidate_lanes(self, centres: np.ndarray) -> np.ndarray:
        ws = self.cfg.workspace
        lanes = list(centres[:, 1]) if centres.size else []
        if centres.size:
            y_lo = float(np.min(centres[:, 1])) - self.capture_halfwidth
            y_hi = float(np.max(centres[:, 1])) + self.capture_halfwidth
        else:
            y_lo, y_hi = float(ws.y_min), float(ws.y_max)
        lanes.extend(np.linspace(y_lo, y_hi, max(2, self.n_lane_candidates)))
        lanes = np.clip(np.asarray(lanes, dtype=float), float(ws.y_min), float(ws.y_max))
        return np.unique(np.round(lanes, 4))

    def _make_candidate(self, y_lane: float, yaw_offset: float, centres, weights, tcp):
        cfg = self.cfg
        p = cfg.planner
        ws = cfg.workspace
        tgt = cfg.target
        tray_margin = float(p.get("tray_margin", 0.03))
        y_end = float(np.clip(y_lane, float(tgt.y_min) + tray_margin,
                              float(tgt.y_max) - tray_margin))
        x_end = float(p.stroke_end_x)

        if centres.size == 0:
            return None

        # Step 1 -- crude horizontal band, only to decide where the stroke begins.
        # (A generous band: the real capture test below runs on the actual segment.)
        band = np.abs(centres[:, 1] - float(y_lane)) <= 1.5 * self.capture_halfwidth
        if not band.any():
            return None
        x_right = float(np.max(centres[band][:, 0]))

        # Never start a stroke inside the force-release guard band: there would
        # be no room left to build up contact force before the ramp-down.
        c = cfg.controller
        x_min_start = float(c.x_release_line) + float(c.release_margin) + 0.06
        x_start = float(np.clip(x_right + float(p.stroke_lead_in),
                                x_min_start, float(ws.x_max)))
        start = np.array([x_start, float(y_lane)])
        end = np.array([x_end, y_end])

        # Step 2 -- capture test against the stroke that will actually be executed.
        along, lateral, _ = point_segment_frames(centres, start, end)
        inside = (along >= -0.02) & (along <= 1.02)
        captured = inside & (np.abs(lateral) <= self.capture_halfwidth)
        grazed = inside & (~captured) & (np.abs(lateral) <= 1.7 * self.capture_halfwidth)
        if not captured.any():
            return None

        stroke_len = float(np.linalg.norm(end - start))
        transfer = float(np.linalg.norm(start - np.asarray(tcp[:2], dtype=float)))
        path_length = stroke_len + transfer

        est_collected = float(np.sum(weights[captured]))

        # ---- risk ----
        lateral_capture = np.abs(lateral[captured])
        edge_risk = float(np.sum(weights[captured] *
                                 (lateral_capture > 0.7 * self.capture_halfwidth)))
        graze_risk = float(np.sum(weights[grazed])) if grazed.any() else 0.0
        funnel = abs(y_end - float(y_lane))
        funnel_risk = float(np.sum(weights[captured])) * min(funnel / 0.15, 1.0)
        outside_tray = 0.0
        if not (float(tgt.y_min) <= y_end <= float(tgt.y_max)):
            outside_tray = est_collected
        risk = 0.6 * edge_risk + 1.0 * graze_risk + 0.5 * funnel_risk + outside_tray

        score = est_collected - self.lambda_path * path_length - self.mu_risk * risk
        yaw = stroke_yaw(end - start) + yaw_offset if bool(p.get("align_yaw", True)) \
            else yaw_offset
        stroke = SweepStroke(
            float(start[0]), float(start[1]), float(end[0]), float(end[1]), float(yaw),
            meta={
                "kind": "greedy",
                "score": float(score),
                "est_collected": est_collected,
                "path_length": path_length,
                "risk": float(risk),
                "n_captured": int(captured.sum()),
            },
        )
        return stroke.clipped(ws)

    # -------------------------------------------------------------------- api
    def plan(self, observation) -> Optional[SweepStroke]:
        if self.is_exhausted():
            return None
        centres, weights = self._clusters(observation)
        if centres.shape[0] == 0:
            return None

        tcp = np.asarray(observation.tcp, dtype=float)
        candidates: List[SweepStroke] = []
        for y_lane in self._candidate_lanes(centres):
            for yaw_offset in self.yaw_candidates:
                cand = self._make_candidate(float(y_lane), float(yaw_offset),
                                            centres, weights, tcp)
                if cand is not None:
                    candidates.append(cand)
        if not candidates:
            return None

        self.last_candidates = [dict(c.meta, y=c.y_start) for c in candidates]
        best = max(candidates, key=lambda s: float(s.meta["score"]))
        return best

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
