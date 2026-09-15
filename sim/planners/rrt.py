"""RRT-Connect for the contact-free transfer motions.

Where this is used -- and where it deliberately is not
------------------------------------------------------
RRT plans the **transfer** motions only: lifting off, travelling to the next
stroke's start, and moving to the observation pose.  The sweeping stroke itself
stays a straight Cartesian segment with a trapezoidal/quintic time law.

That split is not an omission, it is the design:

1. **A sweep is supposed to collide.** The stroke's entire purpose is to make
   contact with the components. A sampling-based planner is a collision
   *avoidance* algorithm; asking it to plan a path whose goal is contact is a
   category error, and any obstacle model that let it through would also let it
   through everything else.
2. **Randomised geometry would fight the force loop.** RRT paths are jagged and
   need shortcutting and smoothing; sparse macro-waypoints fed to a hybrid
   force/position controller inject acceleration transients exactly where the
   normal force is being regulated.
3. **Reproducibility.** The experiment protocol is paired seeds and identical
   layouts. A stochastic planner inside the measured motion would add variance
   that has nothing to do with the question being asked.
4. **The action space.** A stroke is exported as 5 numbers (or 16 waypoints).
   That only works because a stroke is a simple geometric primitive.

The transfer phase has none of those constraints: it is contact-free by
construction (the controller always lifts to ``workspace.z_travel`` first), the
force target is zero throughout, and nothing about it enters the action. So a
planner that can route around the tray walls, around parts already on the table,
and later around fixtures and a real arm's self-collisions belongs exactly here.

Model
-----
The prototype end-effector is a Cartesian stage, so task space *is*
configuration space and planning happens in ``(x, y, z)``.  The tool is reduced
to its TCP point by inflating every obstacle:

* horizontally by the tool's circumradius (yaw-independent, hence conservative);
* downwards by the tool height, because the TCP is the *bottom* of the tips, so
  the tool sweeps the volume above it.

For a UR10e the same routines run in joint space instead: swap the
:class:`CollisionModel` for one that checks the arm, and nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Box:
    """Axis-aligned obstacle in configuration space."""

    lo: np.ndarray
    hi: np.ndarray
    label: str = ""

    def contains(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lo) and np.all(q <= self.hi))

    def inflated(self, xy: float, down: float) -> "Box":
        lo = self.lo - np.array([xy, xy, down])
        hi = self.hi + np.array([xy, xy, 0.0])
        return Box(lo, hi, self.label)


class CollisionModel:
    """Point-in-boxes collision checking with segment discretisation."""

    def __init__(self, boxes: Sequence[Box], bounds_lo, bounds_hi, resolution: float = 0.005):
        self.boxes: List[Box] = list(boxes)
        self.lo = np.asarray(bounds_lo, dtype=float)
        self.hi = np.asarray(bounds_hi, dtype=float)
        self.resolution = float(resolution)

    # -- queries ------------------------------------------------------------
    def in_bounds(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lo - 1e-9) and np.all(q <= self.hi + 1e-9))

    def free(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)
        if not self.in_bounds(q):
            return False
        return not any(box.contains(q) for box in self.boxes)

    def segment_free(self, a, b) -> bool:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        distance = float(np.linalg.norm(b - a))
        steps = max(2, int(np.ceil(distance / max(self.resolution, 1e-6))) + 1)
        for s in np.linspace(0.0, 1.0, steps):
            if not self.free(a + s * (b - a)):
                return False
        return True

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lo, self.hi)

    # -- construction -------------------------------------------------------
    @classmethod
    def for_transfer(cls, cfg, start, goal, component_points=None) -> "CollisionModel":
        """Build the transfer-phase model from the scene configuration.

        ``component_points`` should come from the *perception* observation, not
        from simulator ground truth, so the transfer planner knows exactly what
        the stroke planner knows.
        """
        from .geometry_utils import pusher_width

        tip_half = np.asarray(cfg.end_effector.tip_half, dtype=float)
        tool_radius = float(np.hypot(tip_half[0], pusher_width(cfg) / 2.0))
        tool_height = 2.0 * float(tip_half[2])
        clearance = float(cfg.get_path("planner.transfer.clearance", 0.004))
        inflate = tool_radius + clearance

        top_z = float(cfg.table.top_z)
        tgt = cfg.target
        t = float(tgt.wall_thickness)
        h = float(tgt.wall_height)
        boxes: List[Box] = []

        # tray walls (same three boxes the scene builder emits)
        x_min, x_max = float(tgt.x_min), float(tgt.x_max)
        y_min, y_max = float(tgt.y_min), float(tgt.y_max)
        boxes.append(Box(np.array([x_min - t, y_min - t, top_z]),
                         np.array([x_min, y_max + t, top_z + h]), "tray_back"))
        boxes.append(Box(np.array([x_min - t, y_max, top_z]),
                         np.array([x_max, y_max + t, top_z + h]), "tray_yp"))
        boxes.append(Box(np.array([x_min - t, y_min - t, top_z]),
                         np.array([x_max, y_min, top_z + h]), "tray_yn"))
        boxes = [box.inflated(inflate, tool_height) for box in boxes]

        if component_points is not None and bool(
            cfg.get_path("planner.transfer.avoid_components", True)
        ):
            points = np.atleast_2d(np.asarray(component_points, dtype=float)).reshape(-1, 2)
            pad = inflate + float(cfg.get_path("planner.transfer.component_clearance", 0.010))
            part_h = top_z + float(cfg.get_path("planner.transfer.component_height", 0.012))
            for point in points:
                boxes.append(Box(
                    np.array([point[0] - pad, point[1] - pad, top_z - tool_height]),
                    np.array([point[0] + pad, point[1] + pad, part_h]),
                    "component",
                ))

        ws = cfg.workspace
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        z_lo = min(float(start[2]), float(goal[2]))
        z_hi = max(float(start[2]), float(goal[2]),
                   float(ws.z_travel), float(cfg.end_effector.z_home)) + 0.02
        bounds_lo = np.array([float(ws.safe_x_min), float(ws.safe_y_min), z_lo])
        bounds_hi = np.array([float(ws.safe_x_max), float(ws.safe_y_max), z_hi])
        resolution = float(cfg.get_path("planner.transfer.resolution", 0.005))
        return cls(boxes, bounds_lo, bounds_hi, resolution)


# ---------------------------------------------------------------------------
# RRT-Connect
# ---------------------------------------------------------------------------
class _Tree:
    def __init__(self, root: np.ndarray):
        self.nodes = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, q: np.ndarray) -> int:
        data = np.asarray(self.nodes)
        return int(np.argmin(np.linalg.norm(data - q[None, :], axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(q, dtype=float))
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def path_to_root(self, index: int) -> List[np.ndarray]:
        out = []
        while index != -1:
            out.append(self.nodes[index])
            index = self.parents[index]
        return out[::-1]


def _steer(frm: np.ndarray, to: np.ndarray, step: float) -> np.ndarray:
    delta = to - frm
    distance = float(np.linalg.norm(delta))
    if distance <= step:
        return to.copy()
    return frm + delta / distance * step


def rrt_connect(
    start,
    goal,
    model: CollisionModel,
    rng: np.random.Generator,
    step_size: float = 0.05,
    max_iters: int = 3000,
    goal_bias: float = 0.15,
) -> Optional[List[np.ndarray]]:
    """Bidirectional RRT.  Returns a waypoint list, or ``None`` if it fails.

    Deterministic for a given ``rng`` state, which is what keeps paired-seed
    experiments reproducible even with the planner enabled.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if not (model.free(start) and model.free(goal)):
        return None
    if model.segment_free(start, goal):
        return [start, goal]          # no sampling needed, and no randomness used

    tree_a, tree_b = _Tree(start), _Tree(goal)
    swapped = False
    for _ in range(int(max_iters)):
        target = goal if rng.random() < goal_bias else model.sample(rng)
        near = tree_a.nearest(target)
        new = _steer(tree_a.nodes[near], target, step_size)
        if not model.segment_free(tree_a.nodes[near], new):
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped
            continue
        index_a = tree_a.add(new, near)

        # greedily connect the other tree towards the new node
        near_b = tree_b.nearest(new)
        current = tree_b.nodes[near_b]
        parent = near_b
        while True:
            step = _steer(current, new, step_size)
            if not model.segment_free(current, step):
                break
            parent = tree_b.add(step, parent)
            current = step
            if np.linalg.norm(current - new) < 1e-9:
                path_a = tree_a.path_to_root(index_a)
                path_b = tree_b.path_to_root(parent)[::-1]
                path = path_a + path_b[1:]
                return path[::-1] if swapped else path
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped
    return None


def shortcut_path(path: Sequence[np.ndarray], model: CollisionModel,
                  rng: np.random.Generator, iterations: int = 100) -> List[np.ndarray]:
    """Random pairwise shortcutting; keeps the endpoints fixed."""
    out = [np.asarray(p, dtype=float) for p in path]
    for _ in range(int(iterations)):
        if len(out) <= 2:
            break
        i = int(rng.integers(0, len(out) - 2))
        j = int(rng.integers(i + 2, len(out)))
        if model.segment_free(out[i], out[j]):
            out = out[: i + 1] + out[j:]
    return out


def path_length(path: Sequence[np.ndarray]) -> float:
    pts = np.asarray(path, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def plan_transfer(cfg, start, goal, rng: np.random.Generator,
                  component_points=None) -> List[np.ndarray]:
    """Plan one contact-free transfer; falls back to a straight line.

    A failure here is never fatal: the caller gets the direct path, which is what
    the prototype used before RRT existed.  The fallback is reported through the
    returned path only (two points == direct), so it cannot silently change the
    force behaviour.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    mode = str(cfg.get_path("planner.transfer.mode", "direct"))
    if mode != "rrt":
        return [start, goal]

    model = CollisionModel.for_transfer(cfg, start, goal, component_points)
    if not (model.free(start) and model.free(goal)):
        # An endpoint already touching an inflated obstacle (e.g. the TCP parked
        # right beside the tray wall) is not a reason to refuse to move.
        return [start, goal]

    path = rrt_connect(
        start, goal, model, rng,
        step_size=float(cfg.get_path("planner.transfer.step_size", 0.05)),
        max_iters=int(cfg.get_path("planner.transfer.max_iters", 3000)),
        goal_bias=float(cfg.get_path("planner.transfer.goal_bias", 0.15)),
    )
    if path is None:
        return [start, goal]
    path = shortcut_path(path, model, rng,
                         int(cfg.get_path("planner.transfer.shortcut_iters", 100)))
    max_waypoints = int(cfg.get_path("planner.transfer.max_waypoints", 8))
    if len(path) > max_waypoints:
        path = shortcut_path(path, model, rng, 200)
    return [np.asarray(p, dtype=float) for p in path]
