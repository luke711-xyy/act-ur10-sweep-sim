"""Deterministic automatic demonstrations for the first ACT dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import combinations

import numpy as np

from ..planners.trajectory import make_segment


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
        if abs(float(np.cross(incoming, outgoing))) > 1e-8:
            kept.append(point)
        else:
            # Replacing the previous point with no-op is unnecessary: the
            # final append below retains the same straight segment endpoints.
            pass
    kept.append(path[-1])
    return np.asarray(kept, dtype=float)


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
        cross = float(np.cross(vectors[index], vectors[index + 1]))
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
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            turns = _path_turn_count(path)
            score = (length + 0.010 * turns + 0.002 * abs(float(angle_deg))
                     + 0.20 * max(ratios, default=0.0)
                     + 0.001 * abs(lane_fraction - 0.5))
            candidates.append((score, path))
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
    """
    positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
    if positions.size == 0:
        return []
    target_count = int(np.clip(int(cfg.get_path("task.target_count", len(positions))),
                               1, len(positions)))
    candidates: list[ExpertPlan] = []
    for candidate in combinations(range(len(positions)), int(target_count)):
        # Right-to-left gives each selected object a chance to be captured
        # before final -X delivery, while A* handles lateral detours.
        order = tuple(sorted(candidate, key=lambda index: positions[index, 0], reverse=True))
        for _, path in _capture_lane_candidates(env, cfg, order):
            score, turn_count = _score_capture_plan(env, cfg, order, path)
            candidates.append(ExpertPlan(
                target_indices=np.asarray(order, dtype=int),
                waypoints=np.asarray(path, dtype=float),
                feasible=True, score=score, turn_count=turn_count,
                strategy="capture_lane"))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item.score, tuple(item.target_indices.tolist())))
    limit = max_candidates
    if limit is None:
        limit = int(_planner_value(cfg, "expert_candidate_limit", 12))
    return candidates[:max(1, int(limit))]


def _best_target_sweep_plan(env, cfg, target_count: int):
    """Compatibility wrapper returning the best tuple-shaped plan."""
    candidates = expert_plan_candidates(env, cfg, max_candidates=1)
    if not candidates:
        return None
    best = candidates[0]
    return (best.target_indices.copy(), best.waypoints.copy(),
            int(best.turn_count), best.strategy, float(best.score))


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
        indices, waypoints, turn_count, strategy, score = plan
        if strategy != "capture_lane":
            # A point-visit route can be collision-free in a static XY model
            # and still abandon the first object as soon as it turns toward a
            # later centre.  It is useful for diagnostics, but it is not an
            # admissible success demonstration under the one-pass contract.
            return ExpertPlan(
                target_indices=np.asarray(indices, dtype=int),
                waypoints=_safe_contact_failure_path(cfg),
                feasible=False,
                failure_reason="no_single_capture_lane",
                score=score,
                turn_count=turn_count,
                strategy="no_capture_lane",
            )
        return ExpertPlan(target_indices=np.asarray(indices, dtype=int),
                          waypoints=np.asarray(waypoints, dtype=float),
                          feasible=True, score=score, turn_count=turn_count,
                          strategy=strategy)

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
                    accel: float = 0.8) -> np.ndarray:
    """Sample an absolute execution path with tangent-following brush yaw.

    A yaw of zero keeps the brush face transverse to a sweep towards ``-X``.
    For a general segment the desired tool yaw is therefore the segment heading
    minus pi.  The result is unwrapped locally and rate-limited at every 25 Hz
    reference point so corners do not produce a wrist-angle jump.
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
    dt = 1.0 / float(hz)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])
    if total_length < 1e-9:
        return np.asarray([[points[0, 0], points[0, 1], 0.18,
                            float(initial_yaw if initial_yaw is not None else yaw or 0.0)]],
                          dtype=np.float32)

    # Use one clock for the whole polyline.  The old implementation created a
    # fresh stop-start trapezoid for every A* corner; at each internal corner
    # its sampled displacement collapsed almost to zero, then accelerated in
    # the opposite direction.  A global arc-length profile keeps moving
    # through the corner while retaining the same speed and acceleration
    # limits for the complete contact pass.
    profile = make_segment(np.array([0.0]), np.array([total_length]),
                           float(speed), kind="trapezoidal",
                           accel=float(accel), min_duration=dt)
    times = list(np.arange(0.0, profile.duration, dt))
    if not times or abs(times[-1] - profile.duration) > 1e-9:
        times.append(profile.duration)
    xy_samples = []
    yaw_targets = []
    for time in times:
        travelled = float(np.clip(profile.point(float(time))[0], 0.0, total_length))
        segment_index = int(np.clip(np.searchsorted(cumulative, travelled, side="right") - 1,
                                    0, len(lengths) - 1))
        local_length = travelled - float(cumulative[segment_index])
        fraction = (local_length / float(lengths[segment_index])
                    if lengths[segment_index] > 1e-9 else 1.0)
        xy = points[segment_index] + fraction * (
            points[segment_index + 1] - points[segment_index])
        direction = points[segment_index + 1] - points[segment_index]
        segment_yaw = (float(yaw) if yaw is not None else
                       _wrap_angle(float(np.arctan2(direction[1], direction[0]) - np.pi)))
        xy_samples.append(xy)
        yaw_targets.append(segment_yaw)

    current_yaw = float(yaw_targets[0] if initial_yaw is None else initial_yaw)
    max_step = max(1e-6, float(yaw_rate) * dt)
    out = []
    for xy, desired in zip(xy_samples, yaw_targets):
        delta = _brush_angle_delta(float(desired), current_yaw)
        current_yaw = _wrap_angle(current_yaw + float(np.clip(delta, -max_step, max_step)))
        out.append([xy[0], xy[1], 0.18, current_yaw])
    return np.asarray(out, dtype=np.float32)
