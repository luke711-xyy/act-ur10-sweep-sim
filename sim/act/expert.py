"""Deterministic automatic demonstrations for the first ACT dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import combinations

import numpy as np


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Return the signed scalar cross product for planar vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a[0] * b[1] - a[1] * b[0])



def _wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _brush_angle_delta(desired: float, current: float) -> float:
    """Shortest yaw delta for a plate whose pose is symmetric modulo pi."""
    return float((desired - current + np.pi / 2.0) % np.pi - np.pi / 2.0)


@dataclass(frozen=True)
class ExpertPlan:
    """Auditable result of the one-pass geometric expert planner.

    ``waypoints`` starts at the calibrated contact-search XY pose.  An
    infeasible plan deliberately contains only a short, straight safe motion
    so callers can record a physically meaningful failed attempt without
    silently executing a guessed route through the requested parts.
    """

    target_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    feasible: bool = False
    failure_reason: str = ""
    score: float = float("inf")
    turn_count: int = 0
    strategy: str = "infeasible"
    yaw_offset: float = 0.0


def _planner_value(cfg, name: str, default):
    """Read an expert planner knob, accepting the pre-migration ACT location."""
    value = cfg.get_path(f"planner.{name}", None)
    if value is None:
        value = cfg.get_path(f"act.{name}", default)
    return value


def expert_target_indices(env, cfg) -> np.ndarray:
    """Choose a feasible exact target set for the one-pass expert.

    The simulator deliberately keeps all six parts physical.  When fewer than
    six are requested, the remaining parts are *distractors*, not expendable
    objects.  We therefore score every target subset and keep only those for
    which a brush-width-aware A* path exists around the distractors.  The
    selected set is exposed in the episode metadata so an exact-count label
    cannot be confused with an accidental collection.
    """
    return plan_expert_sweep(env, cfg).target_indices.copy()


def _grid_astar(start: np.ndarray, goal: np.ndarray, obstacles: list[tuple[np.ndarray, float]],
                cfg) -> np.ndarray | None:
    """Find a conservative XY path around circularized brush obstacles.

    The plate is high and thin, but its transverse width is the quantity that
    can sweep a distractor sideways.  Circularizing each distractor by
    ``brush_width / 2 + object_radius + safety`` is conservative for arbitrary
    yaw and keeps the planner independent of the rendering mesh.
    """
    resolution = max(float(_planner_value(cfg, "expert_grid_resolution", 0.01)), 0.002)
    x_min, x_max = float(cfg.workspace.x_min), float(cfg.workspace.x_max)
    y_min, y_max = float(cfg.workspace.y_min), float(cfg.workspace.y_max)
    nx = int(np.floor((x_max - x_min) / resolution)) + 1
    ny = int(np.floor((y_max - y_min) / resolution)) + 1

    def to_index(point):
        point = np.asarray(point, dtype=float)
        return (int(np.clip(np.rint((point[0] - x_min) / resolution), 0, nx - 1)),
                int(np.clip(np.rint((point[1] - y_min) / resolution), 0, ny - 1)))

    def to_point(node):
        return np.array([x_min + node[0] * resolution,
                         y_min + node[1] * resolution], dtype=float)

    start_node, goal_node = to_index(start), to_index(goal)
    blocked = set()
    for ix in range(nx):
        for iy in range(ny):
            point = to_point((ix, iy))
            if any(float(np.linalg.norm(point - centre)) <= radius
                   for centre, radius in obstacles):
                blocked.add((ix, iy))
    # A segment endpoint inside an inflated target is valid when the target is
    # intentionally selected; callers omit selected parts from ``obstacles``.
    if start_node in blocked or goal_node in blocked:
        return None

    # A plain position-only A* is perfectly adequate for reachability, but it
    # treats a 90-degree turn as free.  With a wide brush that produces a
    # staircase of equally cheap grid corners, and the independent segment
    # time laws then turn those corners into visible left/right jolts.  Keep
    # the previous heading in the search state and charge a small, explicit
    # heading-change cost.  The cost is expressed in metres, just like the
    # grid step cost, so it remains interpretable when the resolution changes.
    neighbours = (
        (-1, 0, 1.0, 0), (1, 0, 1.0, 1),
        (0, -1, 1.0, 2), (0, 1, 1.0, 3),
        (-1, -1, np.sqrt(2.0), 4), (-1, 1, np.sqrt(2.0), 5),
        (1, -1, np.sqrt(2.0), 6), (1, 1, np.sqrt(2.0), 7),
    )
    no_heading = 8
    start_state = (start_node[0], start_node[1], no_heading)
    frontier = []
    heappush(frontier, (0.0, 0.0, start_state))
    came_from = {start_state: None}
    cost_so_far = {start_state: 0.0}
    turn_penalty = max(float(_planner_value(cfg, "expert_turn_penalty", 0.015)), 0.0)
    goal_state = None
    while frontier:
        _, current_cost, current = heappop(frontier)
        if current_cost > cost_so_far.get(current, float("inf")) + 1e-12:
            continue
        current_node = (current[0], current[1])
        if current_node == goal_node:
            goal_state = current
            break
        for dx, dy, step_cost, heading in neighbours:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or nxt in blocked:
                continue
            # Do not cut a diagonal corner between two inflated obstacles.
            if dx and dy and ((current[0] + dx, current[1]) in blocked
                              or (current[0], current[1] + dy) in blocked):
                continue
            heading_change = 0.0
            if current[2] != no_heading:
                previous_angle = np.arctan2(
                    neighbours[current[2]][1], neighbours[current[2]][0])
                next_angle = np.arctan2(dy, dx)
                heading_change = abs(_wrap_angle(float(next_angle - previous_angle)))
            new_state = (nxt[0], nxt[1], heading)
            new_cost = (cost_so_far[current] + step_cost * resolution
                        + turn_penalty * heading_change / np.pi)
            if new_cost < cost_so_far.get(new_state, float("inf")):
                cost_so_far[new_state] = new_cost
                target = goal_node
                heuristic = float(np.hypot(target[0] - nxt[0], target[1] - nxt[1])) * resolution
                heappush(frontier, (new_cost + heuristic, new_cost, new_state))
                came_from[new_state] = current
    if goal_state is None:
        return None

    nodes = []
    current = goal_state
    while current is not None:
        nodes.append((current[0], current[1]))
        current = came_from[current]
    nodes.reverse()
    path = np.asarray([to_point(node) for node in nodes], dtype=float)
    path[0] = np.asarray(start, dtype=float)
    path[-1] = np.asarray(goal, dtype=float)
    path = _compact_polyline(path)
    # Grid resolution should affect collision conservatism, not the visual
    # appearance of the expert trajectory.  Greedily take the farthest
    # collision-free visible point and remove the otherwise artificial
    # one-cell staircase.  The segment check is continuous and samples at
    # half a grid cell, so this cannot shortcut through an inflated obstacle.
    return _shortcut_polyline(path, obstacles, max(resolution * 0.5, 0.001))


def _segment_clear(start: np.ndarray, end: np.ndarray,
                   obstacles: list[tuple[np.ndarray, float]],
                   sample_spacing: float) -> bool:
    """Return whether a straight brush-centre segment clears all obstacles."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    distance = float(np.linalg.norm(end - start))
    samples = max(1, int(np.ceil(distance / max(float(sample_spacing), 1e-5))))
    for alpha in np.linspace(0.0, 1.0, samples + 1):
        point = start + float(alpha) * (end - start)
        if any(float(np.linalg.norm(point - centre)) <= float(radius) + 1e-9
               for centre, radius in obstacles):
            return False
    return True


def _shortcut_polyline(path: np.ndarray,
                       obstacles: list[tuple[np.ndarray, float]],
                       sample_spacing: float) -> np.ndarray:
    """Greedily retain the farthest visible waypoint at each path anchor."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) <= 2:
        return path.copy()
    kept = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        farthest = anchor + 1
        for candidate in range(anchor + 1, len(path)):
            if _segment_clear(path[anchor], path[candidate], obstacles, sample_spacing):
                farthest = candidate
        kept.append(path[farthest])
        anchor = farthest
    return np.asarray(kept, dtype=float)


def _compact_polyline(path: np.ndarray) -> np.ndarray:
    """Remove only collinear grid points; retain every clearance corner."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) <= 2:
        return path.copy()
    kept = [path[0]]
    for index, point in enumerate(path[1:-1], start=1):
        incoming = point - kept[-1]
        outgoing = path[index + 1] - point
        if np.linalg.norm(incoming) < 1e-9 or np.linalg.norm(outgoing) < 1e-9:
            continue
        if abs(_cross_2d(incoming, outgoing)) > 1e-8:
            kept.append(point)
        else:
            # Replacing the previous point with no-op is unnecessary: the
            # final append below retains the same straight segment endpoints.
            pass
    kept.append(path[-1])
    return np.asarray(kept, dtype=float)


def _rounded_polyline(path: np.ndarray, radius: float,
                      sharp_corners: set[int] | None = None) -> np.ndarray:
    """Replace polyline corners with tangent-continuous quadratic Bezier arcs."""
    path = _compact_polyline(np.asarray(path, dtype=float).reshape(-1, 2))
    radius = max(0.0, float(radius))
    sharp_corners = set() if sharp_corners is None else {
        int(index) for index in sharp_corners
    }
    if len(path) < 3 or radius <= 1e-6:
        return path.copy()

    corners = []
    for index in range(1, len(path) - 1):
        if index in sharp_corners:
            corners.append(None)
            continue
        previous, point, following = path[index - 1:index + 2]
        incoming = point - previous
        outgoing = following - point
        incoming_len = float(np.linalg.norm(incoming))
        outgoing_len = float(np.linalg.norm(outgoing))
        if incoming_len < 1e-8 or outgoing_len < 1e-8:
            corners.append(None)
            continue
        before = incoming / incoming_len
        after = outgoing / outgoing_len
        angle = abs(float(np.arctan2(_cross_2d(before, after),
                                    np.dot(before, after))))
        # A near-reversal is a cusp, not a smooth corner. Keep it explicit;
        # the time parameterizer will slow there rather than invent a loop.
        if angle < np.deg2rad(2.0) or angle > np.deg2rad(175.0):
            corners.append(None)
            continue
        tangent = float(np.tan(angle / 2.0))
        trim = min(radius * tangent,
                    0.42 * incoming_len,
                    0.42 * outgoing_len)
        if trim < 1e-5 or tangent < 1e-8:
            corners.append(None)
            continue
        entry = point - before * trim
        exit = point + after * trim
        corners.append((entry, point.copy(), exit))

    # Build from the original corner indices; adjacent fillets leave a straight
    # connector between their tangent points.
    result = [path[0].copy()]

    def append(point):
        point = np.asarray(point, dtype=float)
        if np.linalg.norm(point - result[-1]) > 1e-8:
            result.append(point.copy())

    for index, corner in enumerate(corners, start=1):
        if corner is None:
            append(path[index])
            continue
        entry, control, exit = corner
        append(entry)
        curve_length_bound = float(np.linalg.norm(entry - control)
                                   + np.linalg.norm(control - exit))
        samples = max(3, int(np.ceil(curve_length_bound / 0.002)))
        for step in range(1, samples + 1):
            t = step / samples
            point = ((1.0 - t) ** 2 * entry
                     + 2.0 * (1.0 - t) * t * control
                     + t ** 2 * exit)
            append(point)
    append(path[-1])
    return np.asarray(result, dtype=float)


def _path_turn_count(path: np.ndarray, threshold: float = np.deg2rad(8.0)) -> int:
    """Count meaningful direction changes in a planar polyline."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 3:
        return 0
    vectors = np.diff(path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    angles = []
    for index in range(len(vectors) - 1):
        if lengths[index] < 1e-9 or lengths[index + 1] < 1e-9:
            continue
        cross = _cross_2d(vectors[index], vectors[index + 1])
        dot = float(np.dot(vectors[index], vectors[index + 1]))
        angles.append(abs(float(np.arctan2(cross, dot))))
    return int(sum(angle > float(threshold) for angle in angles))


def _clearance_obstacles(env, cfg, allowed: set[int]) -> list[tuple[np.ndarray, float]]:
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    brush_half_width = float(cfg.end_effector.brush_width) / 2.0
    safety = float(_planner_value(cfg, "expert_obstacle_margin", 0.008))
    obstacles = []
    for index, position in enumerate(positions):
        if index in allowed:
            continue
        radius = float(env.layout[index]["nominal_radius"])
        obstacles.append((position, brush_half_width + radius + safety))
    return obstacles


def _append_connector(points: list[np.ndarray], connector: np.ndarray) -> None:
    if connector is None:
        return
    for point in np.asarray(connector, dtype=float):
        if np.linalg.norm(point - points[-1]) > 1e-7:
            points.append(point)


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray,
                               end: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    vector = np.asarray(end, dtype=float) - start
    denominator = float(np.dot(vector, vector))
    if denominator < 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def _polyline_point_distance(point: np.ndarray, path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) == 0:
        return float("inf")
    if len(path) == 1:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - path[0]))
    return min(_point_to_segment_distance(point, start, end)
               for start, end in zip(path[:-1], path[1:]))


def _tray_entry_point(cfg, delivery_y: float) -> np.ndarray:
    """Return a point just outside the tray with the final y already aligned."""
    margin = max(
        float(_planner_value(cfg, "expert_tray_entry_margin", 0.02)),
        float(_planner_value(cfg, "expert_grid_resolution", 0.01)),
    )
    entry_x = max(float(cfg.target.x_max) + margin,
                  float(cfg.planner.stroke_end_x))
    entry_x = float(np.clip(entry_x, float(cfg.workspace.x_min),
                            float(cfg.workspace.x_max)))
    return np.array([entry_x, float(delivery_y)], dtype=float)


def _tray_exit_x(cfg) -> float:
    """Return a wall-safe brush-centre x coordinate deep inside the tray.

    ``stroke_end_x`` is retained as an upper-level workspace limit, while the
    explicit push depth prevents a successful-looking route from stopping at
    the mouth and leaving a component only partially inside.  The lower bound
    keeps the thin brush plate clear of the closed back wall.
    """
    tgt = cfg.target
    half_depth = float(cfg.end_effector.brush_depth) / 2.0
    back_safe_x = float(tgt.x_min) + float(tgt.wall_thickness) + half_depth
    requested_depth = max(
        0.0, float(_planner_value(cfg, "expert_tray_push_depth", 0.09)))
    depth_x = float(tgt.x_max) - requested_depth
    stroke_x = float(cfg.planner.stroke_end_x)
    exit_x = max(back_safe_x, min(stroke_x, depth_x))
    return float(np.clip(exit_x, float(cfg.workspace.x_min),
                         float(cfg.workspace.x_max)))


def _delivery_y(cfg, safe_y: float) -> float:
    """Return the lane the planner believes the tray should receive parts in.

    Normal demonstrations use the calibrated tray centre.  A deliberate
    side-wall failure may provide a virtual tray lane outside that centre;
    the planner then generates the complete route to that false lane while
    the MuJoCo scene continues to contain the real tray at y=0.
    """
    virtual_y = cfg.get_path("planner.virtual_tray_y", None)
    if virtual_y is None:
        return float(np.clip(0.0, -safe_y, safe_y))
    return float(np.clip(float(virtual_y),
                         float(cfg.workspace.y_min),
                         float(cfg.workspace.y_max)))


def _delivery_y_candidates(cfg, safe_y: float) -> tuple[float, ...]:
    """Return distinct safe tray-entry lanes for global curve search.

    Searching only the tray centre can make a physically feasible route look
    impossible: the centreline may miss an edge object or force a sharp final
    turn.  Search several lanes across the usable opening; the MuJoCo rollout
    remains the authority on whether delivery through a lane is safe.
    """
    virtual_y = cfg.get_path("planner.virtual_tray_y", None)
    if virtual_y is not None:
        return (_delivery_y(cfg, safe_y),)
    fractions = _planner_value(
        cfg, "expert_tray_entry_lane_fractions", [0.0, -0.5, 0.5, -0.85, 0.85])
    lanes = []
    for fraction in fractions:
        lane = float(np.clip(float(fraction) * safe_y, -safe_y, safe_y))
        if not any(abs(lane - existing) < 1e-6 for existing in lanes):
            lanes.append(lane)
    return tuple(lanes or [0.0])


def _capture_lane_candidates(env, cfg, order: tuple[int, ...]) -> list[tuple[float, np.ndarray]]:
    """Plan one connected delivery stroke with a rotatable brush lane.

    Candidate lanes are searched in a deterministic orientation lattice.  The
    selected parts must fit inside the brush-width corridor, the continuous
    sweep segment must clear every distractor, and A* must connect both ends
    to the contact start and tray entry.  This is still an open-loop expert,
    but it is no longer restricted to a horizontal line.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    brush_half_width = float(cfg.end_effector.brush_width) / 2.0
    capture_margin = float(_planner_value(cfg, "expert_capture_margin", 0.015))
    tray_half = min(abs(float(cfg.target.y_min)), abs(float(cfg.target.y_max)))
    safe_y = max(0.0, tray_half - brush_half_width
                  - float(cfg.target.wall_thickness) - 0.005)
    delivery_y = _delivery_y(cfg, safe_y)
    tray_x = _tray_exit_x(cfg)
    obstacles = _clearance_obstacles(env, cfg, set(order))
    start = contact_home_xy(cfg).astype(float)
    resolution = max(float(_planner_value(cfg, "expert_grid_resolution", 0.01)), 0.002)
    exit_margin = max(
        float(_planner_value(cfg, "expert_capture_exit_margin", 0.02)),
        float(_planner_value(cfg, "expert_grid_resolution", 0.01)),
    )
    # The first point of the continuous stroke must be outside the selected
    # parts by the same staging clearance used by the legacy horizontal
    # planner.  Using only ``exit_margin`` makes the brush enter the lane too
    # close to the first part; A* can then reject an otherwise valid approach
    # (and, for angle 0, regresses the original planner).
    staging_extra = float(_planner_value(cfg, "expert_staging_margin", 0.018))
    depth_half = float(cfg.end_effector.brush_depth) / 2.0
    approach_margin = max(
        float(env.layout[index]["nominal_radius"]) + depth_half + staging_extra
        for index in order
    )
    x_min, x_max = float(cfg.workspace.x_min), float(cfg.workspace.x_max)
    y_min, y_max = float(cfg.workspace.y_min), float(cfg.workspace.y_max)
    tray_entry = _tray_entry_point(cfg, delivery_y)
    target_coverages = {
        index: max(0.0, brush_half_width
                   + float(env.layout[index]["nominal_radius"])
                   - capture_margin)
        for index in order
    }
    candidates = []
    allowed = set(order)
    angles = _planner_value(
        cfg, "expert_capture_angles_deg",
        [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75],
    )
    lane_fractions = _planner_value(
        cfg, "expert_lane_fractions", [0.15, 0.35, 0.50, 0.65, 0.85])
    lane_fractions = sorted({float(np.clip(value, 0.0, 1.0))
                             for value in lane_fractions})
    for angle_deg in angles:
        angle = float(np.deg2rad(float(angle_deg)))
        # Positive angles sweep toward +Y while still progressing toward the
        # tray, matching the physical brush yaw convention.
        tangent = np.array([-np.cos(angle), np.sin(angle)], dtype=float)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal_coordinates = positions[list(order)] @ normal
        lane_low = max(
            float(value - target_coverages[index])
            for value, index in zip(normal_coordinates, order))
        lane_high = min(
            float(value + target_coverages[index])
            for value, index in zip(normal_coordinates, order))
        if lane_low > lane_high:
            continue
        tangent_coordinates = positions[list(order)] @ tangent
        for lane_fraction in lane_fractions:
            lane_coordinate = (lane_low + lane_fraction * (lane_high - lane_low))
            sweep_start = (tangent * (float(np.min(tangent_coordinates))
                                      - max(exit_margin, approach_margin))
                           + normal * lane_coordinate)
            sweep_exit = (tangent * (float(np.max(tangent_coordinates)) + exit_margin)
                          + normal * lane_coordinate)
            if not (x_min <= sweep_start[0] <= x_max
                    and y_min <= sweep_start[1] <= y_max
                    and x_min <= sweep_exit[0] <= x_max
                    and y_min <= sweep_exit[1] <= y_max):
                continue
            # A* protects the connectors with a physical clearance envelope.
            # The capture segment uses a capture envelope: an edge graze can
            # be physically harmless, but a non-target inside this envelope
            # would be carried into the tray and violate the exact-count
            # label.  This distinction is layout-independent.
            capture_obstacles = []
            for index, centre in enumerate(positions):
                if index in allowed:
                    continue
                radius = float(env.layout[index]["nominal_radius"])
                capture_obstacles.append(
                    (centre, max(0.0, float(cfg.end_effector.brush_width) / 2.0
                                 + radius - capture_margin)))
            if not _segment_clear(sweep_start, sweep_exit, capture_obstacles,
                                  max(resolution * 0.5, 0.001)):
                continue
            approach = _grid_astar(start, sweep_start, obstacles, cfg)
            delivery = _grid_astar(sweep_exit, tray_entry, obstacles, cfg)
            if approach is None or delivery is None:
                continue
            points: list[np.ndarray] = [start.copy()]
            _append_connector(points, approach)
            _append_connector(points, np.asarray([sweep_start, sweep_exit], dtype=float))
            _append_connector(points, delivery)
            # Cross the tray mouth only after the brush is aligned with its
            # centre lane, preventing a lateral consolidation move inside it.
            _append_connector(points, np.asarray([tray_entry,
                                                  [tray_x, delivery_y]], dtype=float))
            path = np.asarray(points, dtype=float)
            ratios = []
            valid = True
            for index in order:
                distance = _polyline_point_distance(positions[index], path)
                coverage = target_coverages[index]
                if distance > coverage + 1e-6:
                    valid = False
                    break
                ratios.append(distance / max(coverage, 1e-6))
            if not valid:
                continue
            rounded = _rounded_polyline(
                path, float(_planner_value(cfg, "expert_corner_radius", 0.025)))
            if any(not _segment_clear(a, b, obstacles,
                                     max(resolution * 0.5, 0.001))
                   for a, b in zip(rounded[:-1], rounded[1:])):
                continue
            ratios = []
            for index in order:
                distance = _polyline_point_distance(positions[index], rounded)
                coverage = target_coverages[index]
                if distance > coverage + 1e-6:
                    valid = False
                    break
                ratios.append(distance / max(coverage, 1e-6))
            if not valid:
                continue
            length = float(np.linalg.norm(np.diff(rounded, axis=0), axis=1).sum())
            turns = _path_turn_count(rounded)
            score = (length + 0.010 * turns + 0.002 * abs(float(angle_deg))
                     + 0.20 * max(ratios, default=0.0)
                     + 0.001 * abs(lane_fraction - 0.5))
            candidates.append((score, path))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _curved_target_orders(env, cfg, order: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Generate a small deterministic set of target orders for winding passes."""
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    subset = tuple(int(index) for index in order)
    if len(subset) <= 1:
        return [subset]
    start = contact_home_xy(cfg).astype(float)
    depth_half = float(cfg.end_effector.brush_depth) / 2.0
    staging = float(_planner_value(cfg, "expert_staging_margin", 0.018))
    exits = max(float(_planner_value(cfg, "expert_capture_exit_margin", 0.02)),
                float(_planner_value(cfg, "expert_grid_resolution", 0.01)))

    def entry(index):
        radius = float(env.layout[index]["nominal_radius"])
        return positions[index] + np.array([radius + depth_half + staging, 0.0])

    def exit_point(index):
        radius = float(env.layout[index]["nominal_radius"])
        return positions[index] - np.array([radius + depth_half + exits, 0.0])

    orders = [tuple(sorted(subset,
                           key=lambda i: (positions[i, 0], positions[i, 1]),
                           reverse=True))]
    orders.append(tuple(sorted(subset,
                               key=lambda i: (positions[i, 1], positions[i, 0]),
                               reverse=True)))

    # Greedy route variants trade distance against backtracking toward +X.
    # Different first targets produce genuinely different windings while the
    # final exact-count physics check remains authoritative.
    first_targets = sorted(subset, key=lambda i: float(np.linalg.norm(entry(i) - start)))
    first_targets = first_targets[:max(1, min(3, len(first_targets)))]
    for first in first_targets:
        remaining = set(subset)
        remaining.remove(first)
        sequence = [first]
        current = exit_point(first)
        while remaining:
            next_index = min(
                remaining,
                key=lambda i: (float(np.linalg.norm(entry(i) - current))
                               + 0.35 * max(0.0, float(entry(i)[0] - current[0])),
                               int(i)),
            )
            sequence.append(next_index)
            current = exit_point(next_index)
            remaining.remove(next_index)
        orders.append(tuple(sequence))

    limit = max(1, int(_planner_value(cfg, "expert_curve_order_limit", 4)))
    unique = []
    for candidate in orders:
        if candidate not in unique:
            unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def _curved_capture_candidates(env, cfg, order: tuple[int, ...]
                               ) -> list[tuple[float, np.ndarray]]:
    """Plan continuous winding sweeps that visit selected targets then deliver.

    Unlike the straight capture-lane family, this planner connects a sequence
    of target-crossing strokes with A* detours. Unselected components are
    inflated obstacles for every connector and crossing; physical MuJoCo
    rollout still decides whether the resulting motion actually collects the
    exact requested number.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    selected = set(int(index) for index in order)
    obstacles = _clearance_obstacles(env, cfg, selected)
    resolution = max(float(_planner_value(cfg, "expert_grid_resolution", 0.01)), 0.002)
    spacing = max(resolution * 0.5, 0.001)
    depth_half = float(cfg.end_effector.brush_depth) / 2.0
    staging = float(_planner_value(cfg, "expert_staging_margin", 0.018))
    exit_margin = max(
        float(_planner_value(cfg, "expert_capture_exit_margin", 0.02)), resolution)
    x_min, x_max = float(cfg.workspace.x_min), float(cfg.workspace.x_max)
    y_min, y_max = float(cfg.workspace.y_min), float(cfg.workspace.y_max)
    tray_half = min(abs(float(cfg.target.y_min)), abs(float(cfg.target.y_max)))
    safe_y = max(0.0, tray_half - float(cfg.end_effector.brush_width) / 2.0
                  - float(cfg.target.wall_thickness) - 0.005)
    delivery_y = _delivery_y(cfg, safe_y)
    tray_entry = _tray_entry_point(cfg, delivery_y)
    tray_x = _tray_exit_x(cfg)
    brush_half = float(cfg.end_effector.brush_width) / 2.0
    capture_margin = float(_planner_value(cfg, "expert_capture_margin", 0.015))
    corner_radius = max(0.0, float(_planner_value(cfg, "expert_corner_radius", 0.025)))
    candidates = []

    for sequence in _curved_target_orders(env, cfg, tuple(order)):
        points: list[np.ndarray] = [contact_home_xy(cfg).astype(float)]
        current = points[0].copy()
        valid = True
        for index in sequence:
            target = positions[index]
            radius = float(env.layout[index]["nominal_radius"])
            stage = target + np.array([radius + depth_half + staging, 0.0])
            clear = target - np.array([radius + depth_half + exit_margin, 0.0])
            if not (x_min <= stage[0] <= x_max and y_min <= stage[1] <= y_max
                    and x_min <= clear[0] <= x_max and y_min <= clear[1] <= y_max):
                valid = False
                break
            connector = _grid_astar(current, stage, obstacles, cfg)
            if connector is None:
                valid = False
                break
            _append_connector(points, connector)
            if not _segment_clear(stage, clear, obstacles, spacing):
                valid = False
                break
            # This is an intentional target crossing, not a point-visit detour:
            # the plate pushes from the object's +X side through its centre.
            _append_connector(points, np.asarray([stage, target, clear]))
            current = clear.copy()
        if not valid:
            continue

        connector = _grid_astar(current, tray_entry, obstacles, cfg)
        if connector is None:
            continue
        _append_connector(points, connector)
        _append_connector(points, np.asarray([tray_entry, [tray_x, delivery_y]], dtype=float))
        path = np.asarray(points, dtype=float)
        rounded = _rounded_polyline(path, corner_radius)
        # The exact curve the executor will track must remain outside every
        # distractor envelope and still intersect each selected capture band.
        if any(not _segment_clear(a, b, obstacles, spacing)
               for a, b in zip(rounded[:-1], rounded[1:])):
            continue
        target_coverages = {
            index: max(0.0, brush_half + float(env.layout[index]["nominal_radius"])
                       - capture_margin)
            for index in selected
        }
        if any(_polyline_point_distance(positions[index], rounded)
               > target_coverages[index] + 1e-6 for index in selected):
            continue
        unintended = False
        for index, position in enumerate(positions):
            if index in selected:
                continue
            radius = float(env.layout[index]["nominal_radius"])
            if (_polyline_point_distance(position, rounded)
                    <= brush_half + radius - capture_margin):
                unintended = True
                break
        if unintended:
            continue
        score, _ = _score_capture_plan(env, cfg, sequence, rounded)
        # Keep equal-length straight strips ahead, but retain these winding
        # alternatives in the execution shortlist for layouts where contact
        # physics rejects the geometric optimum.
        score += 0.015
        candidates.append((score, path))

    candidates.sort(key=lambda item: item[0])
    return candidates


def _segment_capture_mask(start: np.ndarray, end: np.ndarray, positions: np.ndarray,
                          radii: np.ndarray, brush_width: float,
                          brush_depth: float, margin: float) -> int:
    """Return object bits intersected by the oriented swept brush footprint.

    The plate is a thin rectangle: its width is transverse to travel and its
    depth is along travel. A circular distance test is too permissive at the
    ends of a stroke and tends to call side grazes captures. This rectangle
    plus each object's radius is used consistently by search and post-smooth
    validation.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return 0
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    relative = np.asarray(positions, dtype=float) - start
    along = relative @ tangent
    across = np.abs(relative @ normal)
    longitudinal_reach = np.asarray(radii, dtype=float) + float(brush_depth) / 2.0
    transverse_reach = (float(brush_width) / 2.0 + np.asarray(radii, dtype=float)
                        - float(margin))
    hit = ((along >= -longitudinal_reach)
           & (along <= length + longitudinal_reach)
           & (across <= np.maximum(transverse_reach, 0.0)))
    mask = 0
    for index in np.flatnonzero(hit):
        mask |= 1 << int(index)
    return mask


def _capture_forward_cos_min(cfg, target_count: int, total_count: int) -> float:
    """Allow lateral contact for every goal while forbidding pushes away from tray.

    A positive configured threshold can make contact more trayward, but a
    negative value must never authorize pushing a contacted part away from the
    collection zone.  ``target_count`` is intentionally irrelevant: selective
    capture must not force goals 1--5 into straight lanes.
    """
    del target_count, total_count
    configured = float(_planner_value(cfg, "expert_capture_forward_cos", 0.0))
    return max(configured, 0.0)


def _path_capture_mask(path: np.ndarray, env, cfg) -> int:
    """Measure objects captured on trayward or lateral strokes.

    Returning ``-1`` marks a path that touches a part while moving away from
    the tray. Lateral contact is allowed so curved capture routes are not
    screened out before MuJoCo can evaluate their actual outcome.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    radii = np.asarray([float(item["nominal_radius"]) for item in env.layout])
    target_count = int(cfg.get_path("task.target_count", len(positions)))
    forward_cos_min = _capture_forward_cos_min(cfg, target_count, len(positions))
    mask = 0
    for start, end in zip(np.asarray(path)[:-1], np.asarray(path)[1:]):
        hit = _segment_capture_mask(
            start, end, positions, radii,
            float(cfg.end_effector.brush_width),
            float(cfg.end_effector.brush_depth),
            float(_planner_value(cfg, "expert_curve_capture_margin", 0.040)),
        )
        if not hit:
            continue
        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(delta))
        forward_cos = -float(delta[0]) / max(length, 1e-9)
        if forward_cos < forward_cos_min:
            return -1
        mask |= hit
    return mask


def _capture_preserving_rounded_path(path: np.ndarray, radius: float,
                                     env, cfg, target_count: int
                                     ) -> tuple[np.ndarray, int] | None:
    """Round every safe corner while preserving the exact geometric captures.

    A uniform fillet can cut inside a corner that was the only brush-footprint
    intersection with a part.  The grid search correctly finds that discrete
    contact, but a later all-corners smoothing pass then loses it.  Keep the
    smallest number of capture-critical corners sharp; MuJoCo remains the final
    physical acceptance test.
    """
    path = _compact_polyline(np.asarray(path, dtype=float).reshape(-1, 2))
    required = int(target_count)
    raw_mask = _path_capture_mask(path, env, cfg)
    if raw_mask < 0 or raw_mask.bit_count() != required:
        return None

    corner_indices = tuple(range(1, max(1, len(path) - 1)))
    for sharp_count in range(len(corner_indices) + 1):
        valid = []
        for sharp in combinations(corner_indices, sharp_count):
            rounded = _rounded_polyline(path, radius, set(sharp))
            mask = _path_capture_mask(rounded, env, cfg)
            if mask < 0 or mask.bit_count() != required:
                continue
            length = float(np.linalg.norm(np.diff(rounded, axis=0), axis=1).sum())
            valid.append((length, sharp, rounded, mask))
        if valid:
            _, _, rounded, mask = min(valid, key=lambda item: (item[0], item[1]))
            return rounded, mask
    return path, raw_mask


def _rounded_corner_capture_mask(previous: np.ndarray, corner: np.ndarray,
                                 following: np.ndarray, positions: np.ndarray,
                                 radii: np.ndarray, brush_width: float,
                                 brush_depth: float, margin: float,
                                 forward_cos_min: float,
                                 corner_radius: float) -> int:
    """Check the actual fillet around one grid turn, not just its edges."""
    incoming = np.asarray(corner, dtype=float) - np.asarray(previous, dtype=float)
    outgoing = np.asarray(following, dtype=float) - np.asarray(corner, dtype=float)
    incoming_len = float(np.linalg.norm(incoming))
    outgoing_len = float(np.linalg.norm(outgoing))
    if incoming_len < 1e-9 or outgoing_len < 1e-9:
        return 0
    before = incoming / incoming_len
    after = outgoing / outgoing_len
    angle = abs(float(np.arctan2(_cross_2d(before, after), np.dot(before, after))))
    if angle < np.deg2rad(2.0) or angle > np.deg2rad(175.0):
        return 0
    tangent = float(np.tan(angle / 2.0))
    trim = max(0.0, float(corner_radius)) * tangent
    if trim < 1e-6 or tangent < 1e-8:
        return 0
    # Grid neighbours are only 2 cm apart, but the final compacted polyline
    # receives the configured corner radius even when its adjacent straight
    # runs are much longer. Check that same full-radius fillet here.
    entry = np.asarray(corner, dtype=float) - before * trim
    exit = np.asarray(corner, dtype=float) + after * trim
    samples = max(8, int(np.ceil(2.0 * trim / 0.0015)))
    local = [entry]
    for step in range(1, samples + 1):
        t = step / samples
        local.append(((1.0 - t) ** 2 * entry
                      + 2.0 * (1.0 - t) * t * np.asarray(corner, dtype=float)
                      + t ** 2 * exit))
    local = np.asarray(local, dtype=float)
    mask = 0
    for start, end in zip(local[:-1], local[1:]):
        hit = _segment_capture_mask(
            start, end, positions, radii, brush_width, brush_depth, margin)
        if not hit:
            continue
        delta = end - start
        forward_cos = -float(delta[0]) / max(float(np.linalg.norm(delta)), 1e-9)
        if forward_cos < float(forward_cos_min):
            return -1
        mask |= hit
    return mask


def _coverage_astar_candidates(env, cfg, target_count: int
                               ) -> list[tuple[float, np.ndarray,
                                               tuple[int, ...], int]]:
    """Search a continuous brush route and let the route choose its targets.

    Search state is (grid XY, incoming direction, captured-object bitmask).
    Unlike the legacy target-order planner, this is one global A* over the
    swept path: it can curve, turn, or briefly backtrack, and it prunes any
    route that geometrically captures more than the requested exact count.
    Each found route is subsequently screened by the real MuJoCo rollout.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    if not len(positions) or not 1 <= int(target_count) <= len(positions):
        return []
    radii = np.asarray([float(item["nominal_radius"]) for item in env.layout])
    x_min, x_max = float(cfg.workspace.x_min), float(cfg.workspace.x_max)
    y_min, y_max = float(cfg.workspace.y_min), float(cfg.workspace.y_max)
    resolution = max(float(_planner_value(cfg, "expert_curve_grid_resolution", 0.02)),
                     0.005)
    nx = int(np.floor((x_max - x_min) / resolution)) + 1
    ny = int(np.floor((y_max - y_min) / resolution)) + 1

    def to_index(point):
        point = np.asarray(point, dtype=float)
        return (int(np.clip(np.rint((point[0] - x_min) / resolution), 0, nx - 1)),
                int(np.clip(np.rint((point[1] - y_min) / resolution), 0, ny - 1)))

    def to_point(node):
        return np.array([x_min + node[0] * resolution,
                         y_min + node[1] * resolution], dtype=float)

    tray_half = min(abs(float(cfg.target.y_min)), abs(float(cfg.target.y_max)))
    safe_y = max(0.0, tray_half - float(cfg.end_effector.brush_width) / 2.0
                 - float(cfg.target.wall_thickness) - 0.005)
    delivery_lanes = _delivery_y_candidates(cfg, safe_y)
    start = contact_home_xy(cfg).astype(float)
    goals = {}
    for lane_y in delivery_lanes:
        goal = _tray_entry_point(cfg, lane_y)
        goal_node = to_index(goal)
        goals[goal_node] = (to_point(goal_node),
                            np.array([_tray_exit_x(cfg), lane_y], dtype=float))
    goal_nodes = frozenset(goals)
    start_node = to_index(start)
    start_point = to_point(start_node)
    capture_margin = float(_planner_value(cfg, "expert_curve_capture_margin", 0.040))
    brush_width = float(cfg.end_effector.brush_width)
    brush_depth = float(cfg.end_effector.brush_depth)
    corner_radius = float(_planner_value(cfg, "expert_corner_radius", 0.025))
    forward_cos_min = _capture_forward_cos_min(cfg, int(target_count), len(positions))
    max_expansions = max(1000, int(_planner_value(
        cfg, "expert_curve_max_expansions", 180_000)))
    solutions_per_penalty = max(1, int(_planner_value(
        cfg, "expert_curve_solutions_per_penalty", 8)))
    turn_penalties = _planner_value(
        cfg, "expert_curve_turn_penalties", [0.012, 0.025, 0.045])
    heuristic_weight = float(_planner_value(cfg, "expert_curve_heuristic_weight", 1.12))
    expansion_budget = max(1000, max_expansions // max(1, len(turn_penalties)))
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1))
    no_direction = len(directions)
    direction_vectors = np.asarray(directions, dtype=float)
    turn_angles = np.zeros((len(directions), len(directions)), dtype=float)
    for previous_index, previous in enumerate(direction_vectors):
        for following_index, following in enumerate(direction_vectors):
            determinant = (previous[0] * following[1]
                           - previous[1] * following[0])
            dot = float(previous @ following)
            turn_angles[previous_index, following_index] = abs(
                float(np.arctan2(determinant, dot)))
    # Keep the search finite without imposing the old goal-dependent 135° cap.
    # 175° only excludes an immediate grid-step reversal; curved/hairpin routes
    # remain available, with their preference governed by the soft turn cost.
    max_contact_turn = np.deg2rad(175.0)
    candidates = []
    seen_paths = set()
    discouraged_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edge_mask_cache = {}
    corner_mask_cache = {}

    for turn_penalty in turn_penalties:
        turn_penalty = max(0.0, float(turn_penalty))
        start_state = (start_node[0], start_node[1], no_direction, 0)
        frontier = [(0.0, 0.0, start_state)]
        cost_so_far = {start_state: 0.0}
        came_from = {start_state: None}
        found = []
        round_candidates = []
        found_by_goal = {node: 0 for node in goal_nodes}
        solutions_per_goal = max(
            1, solutions_per_penalty // max(1, len(goal_nodes)))
        expansions = 0

        def heuristic(node, mask):
            remaining = int(target_count) - int(mask.bit_count())
            point = to_point(node)
            to_goal = min(float(np.linalg.norm(goal_point - point))
                          for goal_point, _ in goals.values())
            if remaining <= 0:
                return to_goal
            uncovered = [index for index in range(len(positions))
                         if not (mask & (1 << index))]
            reach = (brush_width / 2.0 + radii[uncovered] + brush_depth / 2.0)
            distances = np.maximum(
                0.0, np.linalg.norm(positions[uncovered] - to_point(node), axis=1) - reach)
            kth = float(np.partition(distances, remaining - 1)[remaining - 1])
            return max(to_goal, kth)

        while frontier and expansions < expansion_budget:
            _, current_cost, current = heappop(frontier)
            if current_cost > cost_so_far.get(current, float("inf")) + 1e-12:
                continue
            expansions += 1
            ix, iy, previous_direction, mask = current
            current_node = (ix, iy)
            current_point = to_point(current_node)
            if current_node in goal_nodes:
                goal_point, exit_point = goals[current_node]
                final_mask = mask | _segment_capture_mask(
                    goal_point, exit_point, positions, radii,
                    brush_width, brush_depth, capture_margin)
                if (final_mask.bit_count() == int(target_count)
                        and found_by_goal[current_node] < solutions_per_goal):
                    found.append(current)
                    found_by_goal[current_node] += 1
                    # The brush has reached the terminal delivery lane.  Keep
                    # searching for other incoming headings so the physical
                    # rollout can reject a geometrically valid but poor
                    # contact sequence and try a genuinely different curve.
                    if all(count >= solutions_per_goal
                           for count in found_by_goal.values()):
                        break
                if current_node in goal_nodes:
                    continue

            for direction_index, (dx, dy) in enumerate(directions):
                nxt = (ix + dx, iy + dy)
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                    continue
                next_point = to_point(nxt)
                corner_mask = 0
                if previous_direction != no_direction \
                        and previous_direction != direction_index:
                    corner_key = (ix, iy, previous_direction, direction_index)
                    if corner_key not in corner_mask_cache:
                        previous_step = directions[previous_direction]
                        previous_point = to_point((ix - previous_step[0],
                                                   iy - previous_step[1]))
                        corner_mask_cache[corner_key] = _rounded_corner_capture_mask(
                            previous_point, current_point, next_point,
                            positions, radii, brush_width, brush_depth,
                            capture_margin, forward_cos_min, corner_radius)
                    corner_mask = corner_mask_cache[corner_key]
                    if corner_mask < 0:
                        continue
                edge_key = (ix, iy, dx, dy)
                if edge_key not in edge_mask_cache:
                    edge_mask_cache[edge_key] = _segment_capture_mask(
                        current_point, next_point, positions, radii,
                        brush_width, brush_depth, capture_margin)
                step_mask = edge_mask_cache[edge_key]
                if step_mask:
                    delta = next_point - current_point
                    forward_cos = -float(delta[0]) / max(
                        float(np.linalg.norm(delta)), 1e-9)
                    if forward_cos < forward_cos_min:
                        continue
                else:
                    delta = next_point - current_point
                    forward_cos = -float(delta[0]) / max(
                        float(np.linalg.norm(delta)), 1e-9)
                    if forward_cos < forward_cos_min:
                        turn_clearance = float(_planner_value(
                            cfg, "expert_curve_turn_clearance", 0.025))
                        turn_mask = _segment_capture_mask(
                            current_point, next_point, positions, radii,
                            brush_width, brush_depth, -turn_clearance)
                        if turn_mask:
                            continue
                # Corner checks constrain smooth geometry but do not count as
                # reliable captures. Require each object to be crossed by a
                # full forward grid edge; the rounded path is checked again
                # below before a candidate is accepted.
                next_mask = mask | step_mask
                captured = int(next_mask.bit_count())
                if captured > int(target_count):
                    continue
                angle_cost = 0.0
                if previous_direction != no_direction:
                    angle = turn_angles[previous_direction, direction_index]
                    if angle > max_contact_turn:
                        continue
                    # All goal counts share the same near-U-turn guard; normal
                    # curves and sharp hairpins are only softly penalized.
                    angle_cost = turn_penalty * angle / np.pi
                step_length = float(np.linalg.norm(next_point - current_point))
                # A small cost discourages needless travel away from the tray,
                # without forbidding the backtracking required by a real curve.
                dx_world = float(next_point[0] - current_point[0])
                backtrack_cost = max(0.0, dx_world) * 0.08
                edge_key = tuple(sorted((current_node, nxt)))
                diversity_penalty = (float(_planner_value(
                    cfg, "expert_curve_diversity_penalty", 0.030))
                    if edge_key in discouraged_edges else 0.0)
                new_cost = (current_cost + step_length + angle_cost
                            + backtrack_cost + diversity_penalty)
                next_state = (nxt[0], nxt[1], direction_index, next_mask)
                if new_cost + 1e-12 >= cost_so_far.get(next_state, float("inf")):
                    continue
                cost_so_far[next_state] = new_cost
                came_from[next_state] = current
                estimate = heuristic(nxt, next_mask)
                heappush(frontier, (new_cost + heuristic_weight * estimate,
                                    new_cost, next_state))

        for found_state in found:
            states = []
            current = found_state
            while current is not None:
                states.append((current[0], current[1]))
                current = came_from[current]
            states.reverse()
            grid_path = np.asarray([to_point(node) for node in states], dtype=float)
            grid_path[0] = start
            goal_point, exit_point = goals[found_state[:2]]
            grid_path[-1] = goal_point
            compact = _compact_polyline(grid_path)
            raw_path = np.vstack((compact, exit_point))
            rounded_result = _capture_preserving_rounded_path(
                raw_path,
                float(_planner_value(cfg, "expert_corner_radius", 0.025)),
                env, cfg, int(target_count),
            )
            if rounded_result is None:
                continue
            rounded, mask = rounded_result
            signature = tuple(np.round(
                rounded[::max(1, len(rounded) // 30)], 3).ravel())
            if signature in seen_paths:
                continue
            seen_paths.add(signature)
            selected = tuple(index for index in range(len(positions))
                             if mask & (1 << index))
            turns = _path_turn_count(raw_path)
            # Use the same robustness objective as oriented capture lanes.
            # Length-only ranking systematically promoted short grazing paths
            # that satisfy the geometric mask but lose objects in MuJoCo.
            score, _ = _score_capture_plan(env, cfg, selected, rounded)
            candidates.append((float(score), rounded, selected, turns))
            round_candidates.append((float(score), states))

        # Each cost variant contributes one best route to the next search's
        # diversity memory.  Penalize only the object-field corridor: the
        # shared approach and tray-entry legs are unavoidable and should not
        # push later searches into pointless detours near either endpoint.
        if round_candidates:
            representative = min(round_candidates, key=lambda item: item[0])[1]
            for first, second in zip(representative[:-1], representative[1:]):
                first_point, second_point = to_point(first), to_point(second)
                midpoint = (first_point + second_point) * 0.5
                if (np.linalg.norm(midpoint - start_point) < 0.08
                        or np.linalg.norm(midpoint - goal_point) < 0.08):
                    continue
                discouraged_edges.add(tuple(sorted((first, second))))

    candidates.sort(key=lambda item: item[0])
    return candidates


def _capture_lane_path(env, cfg, order: tuple[int, ...]) -> np.ndarray | None:
    """Return the best lane candidate for the existing single-plan API."""
    candidates = _capture_lane_candidates(env, cfg, order)
    return candidates[0][1] if candidates else None


def _plan_for_order(env, cfg, order: tuple[int, ...]):
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    selected = set(int(i) for i in order)
    start = contact_home_xy(cfg).astype(float)

    # Prefer one connected capture strip.  Once the brush has contacted a
    # part, it keeps moving toward the tray instead of abandoning that part to
    # visit another centre.  This is the primary expert route.
    capture_path = _capture_lane_path(env, cfg, order)
    if capture_path is not None:
        return capture_path, "capture_lane"

    points = [start.copy()]
    current = start.copy()
    staging_extra = float(_planner_value(cfg, "expert_staging_margin", 0.018))
    depth_half = float(cfg.end_effector.brush_depth) / 2.0
    for index in order:
        target = positions[index]
        radius = float(env.layout[index]["nominal_radius"])
        stage = np.array([
            np.clip(target[0] + radius + depth_half + staging_extra,
                    float(cfg.workspace.x_min), float(cfg.workspace.x_max)),
            target[1],
        ])
        # Every selected part is an allowed contact.  Treating a future target
        # as a distractor made dense all-six layouts impossible and forced the
        # old planner into a false fallback.  Exact identity/count validation
        # still rejects any non-selected part that enters the tray.
        obstacles = _clearance_obstacles(env, cfg, selected)
        connector = _grid_astar(current, stage, obstacles, cfg)
        if connector is None:
            return None
        _append_connector(points, connector)
        connector = _grid_astar(stage, target, obstacles, cfg)
        if connector is None:
            return None
        _append_connector(points, connector)
        current = target.copy()

    tray_half = min(abs(float(cfg.target.y_min)), abs(float(cfg.target.y_max)))
    safe_y = max(0.0, tray_half - float(cfg.end_effector.brush_width) / 2.0
                  - float(cfg.target.wall_thickness) - 0.005)
    # Deposit through the centre of the tray opening.  Reusing a target's
    # outer y-coordinate can leave a small part on the side-wall boundary
    # after rigid-body sliding, especially for the 14 cm plate.  The final
    # segment is therefore a single straight -X push on the calibrated centre
    # lane.
    delivery_y = _delivery_y(cfg, safe_y)
    tray_x = _tray_exit_x(cfg)
    obstacles = _clearance_obstacles(env, cfg, selected)
    tray_entry = _tray_entry_point(cfg, delivery_y)
    connector = _grid_astar(current, tray_entry, obstacles, cfg)
    if connector is None:
        return None
    _append_connector(points, connector)
    _append_connector(points, np.asarray([tray_entry,
                                          [tray_x, delivery_y]], dtype=float))
    return np.asarray(points, dtype=float), "point_visit"


def _score_capture_plan(env, cfg, order: tuple[int, ...], path: np.ndarray) -> tuple[float, int]:
    """Score a full candidate using geometric and robustness terms.

    Length alone is a poor expert objective: it prefers grazing a target and
    taking a sharp connector, which is exactly the route most likely to lose a
    small part in MuJoCo.  The score therefore rewards centred capture,
    clearance from distractor capture envelopes, and few meaningful turns.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    edge_limit = float(_planner_value(cfg, "expert_target_edge_limit", 0.16))
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    y_span = float(np.ptp(positions[list(order), 1])) if len(order) > 1 else 0.0
    edge_risk = float(np.sum(np.maximum(
        0.0, np.abs(positions[list(order), 1]) - edge_limit)))
    turn_count = _path_turn_count(path)
    brush_half_width = float(cfg.end_effector.brush_width) / 2.0
    capture_margin = float(_planner_value(cfg, "expert_capture_margin", 0.015))
    capture_ratios = []
    for index in order:
        radius = float(env.layout[index]["nominal_radius"])
        coverage = max(1e-6, brush_half_width + radius - capture_margin)
        capture_ratios.append(_polyline_point_distance(
            positions[index], path) / coverage)
    capture_risk = (0.75 * max(capture_ratios, default=1.5)
                    + 0.25 * float(np.mean(capture_ratios or [1.5])))
    # A candidate may be outside the capture envelope yet still be only a
    # millimetre away from carrying a distractor under model perturbation.
    # Convert that small clearance into a dimensionless risk term; zero means
    # at least the configured robustness margin is available.
    robustness_margin = float(_planner_value(
        cfg, "expert_robustness_margin", 0.010))
    distractor_clearances = []
    allowed = set(order)
    for index, position in enumerate(positions):
        if index in allowed:
            continue
        radius = float(env.layout[index]["nominal_radius"])
        capture_radius = max(0.0, brush_half_width + radius - capture_margin)
        distractor_clearances.append(
            _polyline_point_distance(position, path) - capture_radius)
    min_clearance = min(distractor_clearances, default=robustness_margin)
    clearance_risk = max(0.0, (robustness_margin - min_clearance)
                         / max(robustness_margin, 1e-6))
    turn_score_weight = float(_planner_value(cfg, "expert_turn_score_weight", 0.010))
    capture_score_weight = float(
        _planner_value(cfg, "expert_capture_score_weight", 0.40))
    clearance_score_weight = float(
        _planner_value(cfg, "expert_clearance_score_weight", 0.20))
    score = (length + 0.35 * y_span + 1.5 * edge_risk
             + turn_score_weight * turn_count
             + capture_score_weight * capture_risk
             + clearance_score_weight * clearance_risk
             + 0.001 * len(path))
    return float(score), int(turn_count)


def expert_plan_candidates(env, cfg, max_candidates: int | None = None) -> list[ExpertPlan]:
    """Return deterministic exact-count capture candidates, best first.

    This is the common search used by preview generation and dataset
    collection.  It enumerates target subsets, brush orientations, lane
    offsets and A* connectors; it is deliberately independent of any one
    seed or target count.  A caller may physically validate the returned
    shortlist and retry the next candidate without changing the task label.
    Every geometric route is returned in both equivalent brush yaw branches:
    the thin rectangular plate has the same XY footprint after a 180-degree
    rotation, while the corresponding UR10 joint configurations can differ
    substantially in reachability and tracking quality.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    if positions.size == 0:
        return []
    target_count = int(np.clip(int(cfg.get_path("task.target_count", len(positions))),
                               1, len(positions)))
    lane_candidates: list[ExpertPlan] = []
    curved_candidates: list[ExpertPlan] = []
    for candidate in combinations(range(len(positions)), int(target_count)):
        # Right-to-left gives each selected object a chance to be captured
        # before final -X delivery, while A* handles lateral detours.
        order = tuple(sorted(candidate, key=lambda index: positions[index, 0], reverse=True))
        for _, path in _capture_lane_candidates(env, cfg, order):
            score, turn_count = _score_capture_plan(env, cfg, order, path)
            lane_candidates.append(ExpertPlan(
                target_indices=np.asarray(order, dtype=int),
                waypoints=np.asarray(path, dtype=float),
                feasible=True, score=score, turn_count=turn_count,
                strategy="capture_lane"))
    # One global coverage search lets geometry choose which exact N objects to
    # sweep and where to turn. It is deliberately not seeded with a fixed list
    # of object-centre visits; real MuJoCo rollout is the final authority.
    for score, path, selected, turns in _coverage_astar_candidates(
            env, cfg, int(target_count)):
        curved_candidates.append(ExpertPlan(
            target_indices=np.asarray(selected, dtype=int),
            waypoints=np.asarray(path, dtype=float),
            feasible=True, score=float(score),
            turn_count=int(turns),
            strategy="coverage_astar"))
    if not lane_candidates and not curved_candidates:
        return []

    limit = max_candidates
    if limit is None:
        limit = int(_planner_value(cfg, "expert_candidate_limit", 12))
    limit = max(1, int(limit))
    key = lambda item: (item.score, tuple(item.target_indices.tolist()),
                        item.strategy, tuple(np.round(item.waypoints[-1], 4)))
    lane_candidates.sort(key=key)
    curved_candidates.sort(key=key)
    if limit == 1:
        selected = sorted(lane_candidates + curved_candidates, key=key)[:1]
        return _add_symmetric_yaw_branches(selected)

    # Reserve shortlist capacity for both straight capture strips and global
    # curved coverage routes so a short strip cannot crowd out all turns.
    lane_budget = min(len(lane_candidates), max(1, (limit + 1) // 2))
    curved_budget = min(len(curved_candidates), max(1, limit // 2))
    selected = lane_candidates[:lane_budget] + curved_candidates[:curved_budget]
    if len(selected) < limit:
        already = {id(item) for item in selected}
        remaining = [item for item in sorted(lane_candidates + curved_candidates, key=key)
                     if id(item) not in already]
        selected.extend(remaining[:limit - len(selected)])
    return _add_symmetric_yaw_branches(sorted(selected, key=key))


def _add_symmetric_yaw_branches(plans: list[ExpertPlan]) -> list[ExpertPlan]:
    """Duplicate geometric paths at yaw 0 and pi for the symmetric brush."""
    branched = []
    for plan in plans:
        for yaw_offset in (0.0, float(np.pi)):
            branched.append(ExpertPlan(
                target_indices=plan.target_indices.copy(),
                waypoints=plan.waypoints.copy(),
                feasible=plan.feasible,
                failure_reason=plan.failure_reason,
                score=plan.score,
                turn_count=plan.turn_count,
                strategy=plan.strategy,
                yaw_offset=yaw_offset,
            ))
    return branched


def _best_target_sweep_plan(env, cfg, target_count: int):
    """Compatibility wrapper returning the best tuple-shaped plan."""
    candidates = expert_plan_candidates(env, cfg, max_candidates=1)
    if not candidates:
        return None
    return candidates[0]


def _safe_contact_failure_path(cfg) -> np.ndarray:
    """Return a short contact-only path for an infeasible expert layout."""
    start = contact_home_xy(cfg).astype(float)
    # A bounded 2 cm probe preserves approach/contact observations while
    # guaranteeing that a planning failure does not become a guessed sweep.
    return np.asarray([start, start + np.array([-0.02, 0.0])], dtype=float)


def plan_expert_sweep(env, cfg) -> ExpertPlan:
    """Build an auditable exact-count one-pass plan for the current layout."""
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    if positions.size == 0:
        return ExpertPlan(failure_reason="empty_layout",
                          waypoints=_safe_contact_failure_path(cfg))
    target_count = int(np.clip(int(cfg.get_path("task.target_count", len(positions))),
                               1, len(positions)))
    plan = _best_target_sweep_plan(env, cfg, target_count)
    if plan is not None:
        return plan

    # Keep a deterministic attempted set for audit metadata, but never use it
    # to synthesize target-centre waypoints.  The caller can now distinguish a
    # genuine A* infeasibility from a physics failure after execution.
    attempted = np.argsort(-positions[:, 0])[:target_count].astype(int)
    return ExpertPlan(target_indices=attempted,
                      waypoints=_safe_contact_failure_path(cfg),
                      feasible=False, failure_reason="astar_no_feasible_path",
                      strategy="infeasible")


def contact_home_xy(cfg) -> np.ndarray:
    """Return the airborne reset XY used by both expert and ACT supervisor."""
    return np.asarray(cfg.get_path("task.contact_start_xy", [0.42, 0.0]),
                      dtype=np.float32).reshape(2)


def expert_staging_xy(env, cfg, plan: ExpertPlan | None = None) -> np.ndarray:
    """Return the first layout-specific contact staging point.

    The expert plan starts at ``contact_home_xy`` and its second waypoint is
    the first A* connector point.  The demonstration executor descends there
    before beginning the contact pass.  ACT inference must use this same
    supervisor-owned point; otherwise its first observation is out of the
    training distribution whenever the component cluster changes position.
    """
    home = contact_home_xy(cfg)
    plan = plan if plan is not None else plan_expert_sweep(env, cfg)
    waypoints = np.asarray(plan.waypoints, dtype=np.float32).reshape(-1, 2)
    if len(waypoints) >= 2:
        return waypoints[1].copy()
    return home.copy()


def expert_waypoints(env, cfg) -> np.ndarray:
    """Return one continuous, collision-aware contact polyline.

    The first point after the fixed contact start is found by A* around all
    distractors.  Selected objects are then approached from their +X side and
    the route is connected to the tray with the same clearance model.  Every
    corner becomes a brush yaw change in :func:`sample_polyline`.
    """
    home = contact_home_xy(cfg).astype(float)
    # The ACT policy always receives control from the same calibrated contact
    # start pose.  The expert may use object truth only for the sweep that
    # follows; approach/descent must not leak a truth-derived starting lane.
    plan = plan_expert_sweep(env, cfg)
    # ``home`` is airborne; all remaining points are the one contact pass.
    # For an infeasible layout ``plan.waypoints`` is intentionally only the
    # short safe probe.  It is not a target-centre fallback.
    waypoints = np.asarray(plan.waypoints, dtype=float)
    return (waypoints if len(waypoints) and np.allclose(waypoints[0], home)
            else np.vstack((home, waypoints)))


def expert_recovery_waypoints(env, cfg) -> np.ndarray:
    """Plan one contact-preserving chase for a currently missed part.

    The primary expert path is intentionally cheap and open loop.  Object
    contacts can move a part away from its initial centre, so after that pass
    the expert re-observes simulator truth, returns outside the tray mouth,
    approaches the right side of the right-most missed part, and pushes it
    through a wall-safe tray lane.  No waypoint changes Z; the lower controller
    continues to own normal contact throughout the recovery.
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    collected = np.asarray(env.collected_mask(), dtype=bool)
    missed = np.flatnonzero(~collected)
    if not len(missed):
        return np.zeros((0, 2), dtype=float)

    index = int(missed[np.argmax(positions[missed, 0])])
    target = positions[index]
    radius = float(env.layout[index]["nominal_radius"])
    brush_depth = float(cfg.end_effector.brush_depth)
    current = np.asarray(env.tcp(), dtype=float)[:2]
    mouth_out_x = float(cfg.target.x_max) + brush_depth / 2.0 + 0.04
    stage_x = float(np.clip(
        max(mouth_out_x + 0.02, target[0] + radius + brush_depth / 2.0 + 0.02),
        float(cfg.workspace.x_min), float(cfg.workspace.x_max)))
    chase_y = float(np.clip(target[1], float(cfg.workspace.y_min),
                            float(cfg.workspace.y_max)))
    tray_half = min(abs(float(cfg.target.y_min)), abs(float(cfg.target.y_max)))
    safe_y_limit = max(0.0, tray_half - float(cfg.end_effector.brush_width) / 2.0
                       - float(cfg.target.wall_thickness))
    delivery_y = float(np.clip(chase_y, -safe_y_limit, safe_y_limit))
    tray_x = _tray_exit_x(cfg)

    points = [current]
    # Leave the three-sided tray through its open +X mouth before moving in Y.
    if current[0] < mouth_out_x:
        points.append(np.array([mouth_out_x, current[1]], dtype=float))
    points.extend((
        np.array([stage_x, current[1]], dtype=float),
        np.array([stage_x, chase_y], dtype=float),
        np.array([mouth_out_x, chase_y], dtype=float),
        np.array([mouth_out_x, delivery_y], dtype=float),
        np.array([tray_x, delivery_y], dtype=float),
        np.array([tray_x, 0.0], dtype=float),
    ))
    # Consecutive identical waypoints are harmless but waste high-level frames.
    compact = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - compact[-1]) > 1e-6:
            compact.append(point)
    return np.asarray(compact, dtype=float)


def sample_polyline(points: np.ndarray, hz: float, speed: float,
                    yaw: float | None = None,
                    yaw_rate: float = np.deg2rad(45.0),
                    initial_yaw: float | None = None,
                    accel: float = 0.8,
                    curve_radius: float = 0.025) -> np.ndarray:
    """Time-parameterize a rounded path without letting XY outrun brush yaw.

    Curvature limits the local translational speed to ``yaw_rate / |curvature|``;
    forward/backward acceleration passes then produce a continuous speed law.
    Thus a tight turn slows the Cartesian path instead of independently clipping
    yaw while the brush continues past the corner.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("expert path needs at least one point")
    if len(points) == 1:
        # A geometrically safe probe can legitimately have no lateral segment
        # after the approach point.  Treat it as a one-sample hold instead of
        # letting a batch generation request abort with a 500 error.
        return np.asarray([[points[0, 0], points[0, 1], 0.18,
                            float(initial_yaw if initial_yaw is not None else yaw or 0.0)]],
                          dtype=np.float32)
    points = _rounded_polyline(points, curve_radius)
    dt = 1.0 / float(hz)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])
    if total_length < 1e-9:
        return np.asarray([[points[0, 0], points[0, 1], 0.18,
                            float(initial_yaw if initial_yaw is not None else yaw or 0.0)]],
                          dtype=np.float32)

    # Resolve the rounded curve into a fine arc-length table so curvature can
    # constrain speed even when the original waypoints are far apart.
    spacing = min(0.002, max(total_length / 2000.0, 0.0005))
    dense_count = max(2, int(np.ceil(total_length / spacing)))
    arc = np.linspace(0.0, total_length, dense_count + 1)
    segment_index = np.clip(np.searchsorted(cumulative, arc, side="right") - 1,
                            0, len(lengths) - 1)
    local = arc - cumulative[segment_index]
    fractions = local / np.maximum(lengths[segment_index], 1e-9)
    curve = points[segment_index] + fractions[:, None] * (
        points[segment_index + 1] - points[segment_index])
    tangent = np.gradient(curve, arc, axis=0, edge_order=2)
    raw_yaw = np.arctan2(tangent[:, 1], tangent[:, 0]) - np.pi
    curve_yaw = 0.5 * np.unwrap(2.0 * raw_yaw)
    if yaw is not None:
        curve_yaw[:] = float(yaw)
    elif initial_yaw is not None:
        curve_yaw += np.pi * round((float(initial_yaw) - curve_yaw[0]) / np.pi)

    curvature = np.gradient(curve_yaw, arc)
    speed_limit = np.full_like(arc, max(float(speed), 1e-4))
    if yaw is None:
        angular_limit = max(float(yaw_rate), 1e-4)
        turning = np.abs(curvature) > 1e-6
        speed_limit[turning] = np.minimum(
            speed_limit[turning], angular_limit / np.abs(curvature[turning]))

    acceleration = max(float(accel), 1e-4)
    velocity = speed_limit.copy()
    velocity[0] = 0.0
    for index in range(1, len(velocity)):
        ds = float(arc[index] - arc[index - 1])
        velocity[index] = min(
            velocity[index],
            np.sqrt(max(0.0, velocity[index - 1] ** 2 + 2.0 * acceleration * ds)),
        )
    velocity[-1] = 0.0
    for index in range(len(velocity) - 2, -1, -1):
        ds = float(arc[index + 1] - arc[index])
        velocity[index] = min(
            velocity[index],
            np.sqrt(max(0.0, velocity[index + 1] ** 2 + 2.0 * acceleration * ds)),
        )

    interval_time = (2.0 * np.diff(arc)
                     / np.maximum(velocity[:-1] + velocity[1:], 1e-8))
    time_at_arc = np.concatenate(([0.0], np.cumsum(interval_time)))
    duration = float(time_at_arc[-1])
    times = np.arange(0.0, duration, dt)
    if not len(times) or duration - float(times[-1]) > 1e-9:
        times = np.append(times, duration)
    sampled_arc = np.interp(times, time_at_arc, arc)
    sampled_xy = np.column_stack((
        np.interp(sampled_arc, arc, curve[:, 0]),
        np.interp(sampled_arc, arc, curve[:, 1]),
    ))
    sampled_yaw = np.interp(sampled_arc, arc, curve_yaw)
    if initial_yaw is not None and yaw is None:
        sampled_yaw[0] = float(initial_yaw)
    max_step = max(1e-6, float(yaw_rate) * dt)
    for index in range(1, len(sampled_yaw)):
        delta = float(sampled_yaw[index] - sampled_yaw[index - 1])
        sampled_yaw[index] = sampled_yaw[index - 1] + float(
            np.clip(delta, -max_step, max_step))
    return np.column_stack((sampled_xy, np.full(len(times), 0.18),
                            sampled_yaw)).astype(np.float32)
