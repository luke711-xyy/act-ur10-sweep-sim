"""Deterministic automatic demonstrations for the first ACT dataset."""

from __future__ import annotations

import numpy as np


def expert_waypoints(env, cfg) -> np.ndarray:
    """Return a single continuous polyline through the current layout.

    The expert visits object centres from right to left, then enters the tray.
    It is intentionally simple and inspectable: ACT is tested against a known
    source of demonstrations, while a later task can replace this generator.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    home = np.array([0.42, 0.0], dtype=float)
    # Stop at the tray mouth, not at its back wall.  The latter was outside
    # the official UR10e task-space bound and made the IK chase an unreachable
    # endpoint even though a component is already collected once its centre
    # crosses the target boundary.
    target_lane_x = float(np.clip(
        float(cfg.target.x_max) - max(0.02, float(cfg.end_effector.brush_depth) / 2.0),
        float(cfg.workspace.x_min), float(cfg.workspace.x_max)))
    if positions.size == 0:
        return np.array([home, [0.30, 0.0], [target_lane_x, 0.0]], dtype=float)
    order = np.argsort(-positions[:, 0])
    ordered = positions[order]
    brush_half = float(cfg.end_effector.brush_width) / 2.0
    start = np.array([min(float(cfg.workspace.x_max) - 0.04,
                          float(ordered[:, 0].max()) + brush_half + 0.06),
                      float(ordered[0, 1])], dtype=float)
    # Include the actual reset pose so the first action does not teleport the
    # arm laterally.  Contact starts only after this airborne transfer.
    points = [home, start]
    for point in ordered:
        points.append(np.asarray(point, dtype=float))
    # Keep the brush aligned with the leftmost (last visited) object while
    # crossing the tray mouth.  Only after that horizontal push is complete do
    # we move the brush centre to the tray centreline.
    points.append(np.array([target_lane_x, float(ordered[-1, 1])], dtype=float))
    target_lane = np.array([target_lane_x, 0.0], dtype=float)
    points.append(target_lane)
    # Stop at the tray mouth after the collection condition is met.  Continuing
    # with edge lanes would make the brush itself contact the fixed side walls;
    # that is a controller/planner collision, not a physically meaningful
    # recovery from a sideways-sliding component.  Such failures remain
    # represented by episodes that lose a component before reaching the mouth.
    return np.asarray(points, dtype=float)


def sample_polyline(points: np.ndarray, hz: float, speed: float,
                    yaw: float = 0.0) -> np.ndarray:
    """Sample a polyline as absolute ``[x, y, z, yaw]`` command targets."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("expert path needs at least two points")
    out = []
    dt = 1.0 / float(hz)
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(distance / max(float(speed) * dt, 1e-6))))
        for alpha in np.linspace(0.0, 1.0, count, endpoint=False):
            xy = start + alpha * (end - start)
            # The first waypoint is in the air; the brush descends over a short
            # prefix so the policy has a meaningful learned approach signal.
            out.append([xy[0], xy[1], 0.18, yaw])
    out.append([points[-1, 0], points[-1, 1], 0.18, yaw])
    return np.asarray(out, dtype=np.float32)
