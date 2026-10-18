"""Estimated-only scene for the frozen initial planner and geometric monitoring.

No simulator handle or delegation exists here. Covariances have already been
calibrated by the visual tracker. Three standard deviations enlarge the shape
envelope; this engineering envelope is not a fitted coverage guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .contracts import ObjectBelief


@dataclass(frozen=True)
class PlanningGeometry:
    """Only known static geometry and frozen geometric planner settings."""
    workspace: tuple[float, float, float, float]
    target: tuple[float, float, float, float]
    table_z: float
    brush_width: float
    brush_depth: float
    capture_margin: float
    forward_cos: float
    target_count: int

    @classmethod
    def from_config(cls, cfg):
        result = cls(tuple(float(getattr(cfg.workspace, k)) for k in ('x_min','x_max','y_min','y_max')),
                     tuple(float(getattr(cfg.target, k)) for k in ('x_min','x_max','y_min','y_max')),
                     float(cfg.table.top_z), float(cfg.end_effector.brush_width), float(cfg.end_effector.brush_depth),
                     float(cfg.get_path('planner.expert_curve_capture_margin',
                                        cfg.get_path('act.expert_curve_capture_margin', .040))),
                     max(0., float(cfg.get_path('planner.expert_capture_forward_cos',
                                               cfg.get_path('act.expert_capture_forward_cos', 0.)))),
                     int(cfg.get_path('task.target_count', 6)))
        values = (*result.workspace, *result.target, result.table_z, result.brush_width,
                  result.brush_depth, result.capture_margin, result.forward_cos)
        if (not np.isfinite(values).all() or not 1 <= result.target_count <= 6
                or result.brush_width <= 0 or result.brush_depth <= 0
                or result.workspace[0] >= result.workspace[1] or result.workspace[2] >= result.workspace[3]):
            raise ValueError('invalid known planning geometry')
        return result


@dataclass(frozen=True)
class _KnownYaw:
    yaw: float

    def tcp_yaw(self):
        return self.yaw


@dataclass(frozen=True, init=False)
class PerceivedPlanningScene:
    """Immutable estimated geometry, stable sorted track association, no GT.

    ``layout`` and ``component_positions`` return disposable copies for the
    unchanged frozen expert API. Uncertain/rectangular shapes use their outer
    circle; a certain circle uses its largest estimated half extent.
    """
    centers: tuple[tuple[float, float], ...]
    radii: tuple[float, ...]
    track_ids: tuple[int, ...]
    geometry: PlanningGeometry
    ee: _KnownYaw

    def __init__(self, beliefs, *, cfg=None, robot_yaw=0., geometry=None):
        if len(beliefs) != 6 or any(type(b) is not ObjectBelief for b in beliefs):
            raise ValueError('six visual ObjectBeliefs required')
        # Revalidate mutable numeric payloads at the consumer boundary.
        items = sorted((replace(b) for b in beliefs), key=lambda b: b.track_id)
        ids = tuple(b.track_id for b in items)
        if len(set(ids)) != 6 or not np.isfinite(robot_yaw):
            raise ValueError('invalid visual track association or known robot yaw')
        radii = []
        for b in items:
            std = float(np.sqrt(max(0., np.linalg.eigvalsh(b.position_covariance).max())))
            if not b.valid or std > .025:
                raise ValueError('invalid or uncertain visual track')
            radius = (float(b.half_extents_xy.max()) if b.shape_probabilities[0] == 1.
                      else float(np.linalg.norm(b.half_extents_xy)))
            radii.append(radius + 3.*std)
        static = PlanningGeometry.from_config(cfg) if geometry is None else geometry
        if type(static) is not PlanningGeometry:
            raise TypeError('known PlanningGeometry required')
        object.__setattr__(self, 'centers', tuple(tuple(float(v) for v in b.center_xy) for b in items))
        object.__setattr__(self, 'radii', tuple(radii))
        object.__setattr__(self, 'track_ids', ids)
        object.__setattr__(self, 'geometry', static)
        object.__setattr__(self, 'ee', _KnownYaw(float(robot_yaw)))

    @property
    def layout(self):
        return [dict(nominal_radius=r) for r in self.radii]

    def component_positions(self):
        return np.c_[np.asarray(self.centers), np.full(6, self.geometry.table_z)]

    def component_collision_x_max(self):
        return max(xy[0] + r for xy, r in zip(self.centers, self.radii))


@dataclass(frozen=True)
class PlanValidity:
    valid: bool
    reason: str = ''


def monitor_plan(scene: PerceivedPlanningScene, plan) -> PlanValidity:
    """Check the selected, frozen contact suffix against current estimates.

    Uses the frozen expert's oriented swept rectangle/circle capture equations
    (expert.py:_segment_capture_mask/_path_capture_mask), without searching or
    modifying a route. Initial airborne home->landing is excluded. Geometric
    consistency is necessary, not a prediction of physical task success.
    """
    if type(scene) is not PerceivedPlanningScene:
        raise TypeError('estimated planning scene required')
    if plan is None or not plan.feasible:
        return PlanValidity(False, 'no feasible selected visual plan')
    path = np.asarray(plan.waypoints)
    selected = np.asarray(plan.target_indices)
    g = scene.geometry
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 3 or not np.isfinite(path).all()
            or selected.ndim != 1 or selected.dtype.kind not in 'iu'
            or len(selected) != g.target_count or len(set(selected.tolist())) != len(selected)
            or np.any((selected < 0) | (selected >= 6))):
        return PlanValidity(False, 'invalid selected path or target association')
    x0, x1, y0, y1 = g.workspace
    if np.any((path[:,0] < x0) | (path[:,0] > x1) | (path[:,1] < y0) | (path[:,1] > y1)):
        return PlanValidity(False, 'selected path outside known workspace')
    region = plan.landing_region
    if region is not None:
        values = np.array([region[k] for k in ('part_x_max','clearance','brush_x_half_extent','yaw','x_min')])
        edge, clearance, half, yaw, boundary = values
        if (not np.isfinite(values).all() or scene.component_collision_x_max() + clearance + half > boundary + 1e-6
                or path[1,0] < boundary - 1e-6):
            return PlanValidity(False, 'perceived contour invalidates frozen landing boundary')
    positions, radii = np.asarray(scene.centers), np.asarray(scene.radii)
    hits = np.zeros(6, dtype=bool)
    for start, end in zip(path[1:-1], path[2:]):
        delta = end-start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        tangent = delta/length
        relative = positions-start
        along = relative @ tangent
        across = np.abs(relative @ np.array([-tangent[1], tangent[0]]))
        hit = ((along >= -(radii+g.brush_depth/2)) & (along <= length+radii+g.brush_depth/2)
               & (across <= np.maximum(g.brush_width/2+radii-g.capture_margin, 0.)))
        if hit.any() and -delta[0]/length < g.forward_cos:
            return PlanValidity(False, 'selected path pushes perceived part away from tray')
        hits |= hit
    expected = np.zeros(6, dtype=bool)
    expected[selected] = True
    if not np.array_equal(hits, expected):
        return PlanValidity(False, 'perceived contour invalidates selected path captures')
    return PlanValidity(True)
