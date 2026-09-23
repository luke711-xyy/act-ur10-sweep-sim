"""Deterministic automatic demonstrations for the first ACT dataset."""

from __future__ import annotations

import numpy as np


def expert_waypoints(env, cfg) -> np.ndarray:
    """Return a single continuous polyline through the current layout.

    The expert approaches the *front edge* of each object from right to left,
    rather than commanding the brush centre through an object.  That distinction
    matters for a thin free body: driving the brush centre to the object's
    centre creates an artificial ride-over/rolling impulse before the pusher can
    carry it towards the tray.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    home = np.array([0.42, 0.0], dtype=float)
    if positions.size == 0:
        return np.array([home, [0.30, 0.0], [float(cfg.target.x_min), 0.0]], dtype=float)
    order = np.argsort(-positions[:, 0])
    ordered = positions[order]
    brush_half_depth = float(cfg.end_effector.brush_depth) / 2.0
    front_clearance = 0.004
    start = np.array([min(float(cfg.workspace.x_max) - 0.04,
                          max(float(item["x"]) + brush_half_depth
                              + float(item["nominal_radius"]) + front_clearance
                              for item in env.layout) + 0.06),
                      float(ordered[0, 1])], dtype=float)
    # Include the actual reset pose so the first action does not teleport the
    # arm laterally.  Contact starts only after this airborne transfer.
    points = [home, start]
    for index in order:
        item = env.layout[int(index)]
        points.append(np.array([
            float(item["x"]) + brush_half_depth
            + float(item["nominal_radius"]) + front_clearance,
            float(item["y"]),
        ], dtype=float))
    # Keep the final carry lane aligned with the last contacted object.  A
    # diagonal turn to y=0 at the tray mouth would shear the brush away from
    # a part near either side wall and leave it halfway across the table.
    target_lane = np.array([float(cfg.target.x_min) + 0.04,
                            float(ordered[-1, 1])], dtype=float)
    points.append(target_lane)
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


def expert_sweep_lanes(env, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return monotone contact lanes covering the current loose cluster."""
    if not env.layout:
        return []
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    half_width = float(cfg.end_effector.brush_width) / 2.0
    effective_half = max(0.01, half_width - 0.008)
    y_min, y_max = float(positions[:, 1].min()), float(positions[:, 1].max())
    span = y_max - y_min
    if span <= 2.0 * effective_half:
        lane_ys = [0.5 * (y_min + y_max)]
    else:
        step = 2.0 * effective_half * 0.8
        n_lanes = int(np.ceil((span - 2.0 * effective_half) / step)) + 1
        # Leave a small overlap margin at both edges.  Exact edge-centred lanes
        # make the neighbouring part ride the brush boundary; a 3 cm inward
        # bias keeps both lanes in the broad face while retaining coverage.
        lane_margin = min(0.03, 0.25 * span)
        lane_ys = np.linspace(y_min + lane_margin, y_max - lane_margin,
                              max(2, n_lanes)).tolist()

    half_depth = float(cfg.end_effector.brush_depth) / 2.0
    rightmost_front = max(
        float(item["x"]) + half_depth + float(item["nominal_radius"]) + 0.004
        for item in env.layout
    )
    start_x = min(float(cfg.workspace.x_max) - 0.04, rightmost_front + 0.06)
    end_x = float(cfg.target.x_min) + 0.04
    return [(np.array([start_x, y], dtype=float),
             np.array([end_x, y], dtype=float)) for y in lane_ys]


def expert_path(env, cfg) -> np.ndarray:
    """Build an ACT-facing path with lifted transfers between contact lanes."""
    lanes = expert_sweep_lanes(env, cfg)
    home = np.array([0.42, 0.0], dtype=float)
    if not lanes:
        return sample_polyline(np.array([home, [float(cfg.target.x_min), 0.0]]),
                               float(cfg.act.action_hz), float(cfg.controller.sweep_speed))

    hz = float(cfg.act.action_hz)
    z_home = float(cfg.end_effector.z_home)
    z_search = float(cfg.workspace.z_search_start)
    chunks = []

    def air(points):
        part = sample_polyline(np.asarray(points, dtype=float), hz,
                               float(cfg.controller.travel_speed))
        part[:, 2] = z_home
        return part

    def descend(start):
        count = max(1, int(np.ceil((z_home - z_search)
                                   / max(float(cfg.controller.z_speed), 1e-6) * hz)))
        part = np.zeros((count, 4), dtype=np.float32)
        part[:, :2] = start
        part[:, 2] = np.linspace(z_home, z_search, count)
        return part

    chunks.append(air([home, lanes[0][0]]))
    for index, (start, end) in enumerate(lanes):
        chunks.append(descend(start))
        sweep = sample_polyline(np.array([start, end]), hz,
                                float(cfg.controller.sweep_speed))
        sweep[:, 2] = z_search
        chunks.append(sweep)
        if index + 1 < len(lanes):
            chunks.append(air([end, lanes[index + 1][0]]))
    return np.concatenate(chunks, axis=0)
