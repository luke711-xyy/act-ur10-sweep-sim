"""OFFLINE ONLY: versioned, causal carry supervision from a bounded truth window.

Requires consecutive collision/contact truth ticks (<=40ms gaps), independent
of sparse RGB visibility evidence. A positive requires a contemporaneous brush
contact path at every tick, front-of-brush geometry, common center translation
and bounded relative displacement. No union of historical contact identities.
Unknown evidence/transition/conflicts/threshold equality produce invalid labels.

Front bounds: brush-frame leading half-plane and brush lateral support expanded
by object half-scales. Forward reach expands by the current contact path's full
intermediate envelopes and the object's half-scale. No nominal-radius input.
Brush translation must exceed the allowed relative-displacement tolerance to
distinguish transport from a stationary touching object. The 1e-9 numerical edge
tolerance is not reviewed label quality. Task13 must review representative clips.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .truth import (GEOMETRY_VERSION, ObjectTargets, TruthFrame, _angular_secant, _bounds,
                    _project_body, _rotation, project_envelopes, stack_targets)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class LabelConfig:
    version: str = "wam-carry-v1"
    window_seconds: float = .200
    join_seconds: float = .120
    leave_seconds: float = .120
    speed_difference_mps: float = .025
    relative_displacement_m: float = .008
    max_evidence_gap_seconds: float = .040
    edge_tolerance: float = 1e-9

    def __post_init__(self):
        if self.version != "wam-carry-v1":
            raise ValueError("unsupported label derivation version")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
               for k, v in asdict(self).items() if k != "version"):
            raise ValueError("positive finite label thresholds required")
        if self.max_evidence_gap_seconds > self.window_seconds:
            raise ValueError("evidence gap cannot exceed window")

    @property
    def config_hash(self):
        return _hash(dict(config=asdict(self), geometry_version=GEOMETRY_VERSION))


@dataclass(frozen=True)
class LabelFrame:
    timestamp: float
    targets: ObjectTargets
    carried: np.ndarray
    carried_valid: np.ndarray
    state: np.ndarray  # 0 free, 1 carried, 2 native-collected; invalid is not free
    state_valid: np.ndarray
    valid_steps: np.ndarray
    metadata: dict

    def native_count_matches(self, target_count: int) -> bool | None:
        """Instantaneous spatial predicate, NOT an episode/terminal success label.

        The frozen backend still owns first-entry age, two-second collection
        window, native safety/termination priority and the final EpisodeResult.
        """
        if type(target_count) is not int or not 1 <= target_count <= 6:
            raise ValueError("target count must be 1..6")
        if not self.targets.native_valid.all():
            return None
        return bool(self.targets.native_collected.sum() == target_count
                    and not self.targets.native_partial.any())


def future_targets(frames, *, steps=25):
    """Consecutive 25 Hz future prefix only; actual future states, not t0.

    The first supplied frame has no velocity unless the caller separately provides
    its preceding capture to project_envelopes. Missing/terminated future steps are
    zero-filled with valid_steps=False, never synthesized as stationary negatives.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("positive horizon required")
    frames = tuple(frames)
    if not frames:
        raise ValueError("at least one actual future frame required")
    if any(abs(b.timestamp - a.timestamp - .04) > 1e-9 for a, b in zip(frames, frames[1:])):
        raise ValueError("future prefix requires consecutive 25 Hz captures")
    if any(f.raw.fields["object_ids"] != frames[0].raw.fields["object_ids"] for f in frames):
        raise ValueError("future object IDs must stay stable")
    targets = [project_envelopes(f, previous=frames[i - 1] if i else None)
               for i, f in enumerate(frames)]
    return stack_targets(targets, steps)


def _summary(frame, targets):
    f = frame.raw.fields
    brush = f["brush"]
    bc = _project_body(brush)[0]
    low, high = _bounds(brush, brush.yaw)
    axes = _rotation(brush.yaw)
    ids = f["object_ids"]
    # One BFS per contemporaneous graph, never a cumulative ever-contacted set.
    graph = {i: set() for i in (*ids, -1)}
    for a, b in f["contacts"]:
        graph[a].add(b)
        graph[b].add(a)
    half = []
    for obj in f["objects"]:
        lo, hi = _bounds(obj, brush.yaw)
        half.append((hi - lo) / 2)
    half = np.asarray(half)
    index = {identity: i for i, identity in enumerate(ids)}
    paths, queue = {-1: ()}, [-1]
    for node in queue:
        for other in sorted(graph[node]):
            if other not in paths:
                paths[other] = (*paths[node], other)
                queue.append(other)
    front, linked = [], []
    for i, identity in enumerate(ids):
        c = targets.center_xy[i] @ axes
        path = paths.get(identity, ())
        extra = sum(2 * half[index[j], 0] for j in path[:-1])
        # Signed distances also keep spatial equality explicitly unknown.
        margins = (c[0] - bc @ axes[:, 0], high[0] + half[i, 0] + extra - c[0],
                   c[1] - (low[1] - half[i, 1]), high[1] + half[i, 1] - c[1])
        front.append(float(min(margins)))
        linked.append(identity in paths)
    return dict(time=frame.timestamp, centers=targets.center_xy.tolist(), brush=bc.tolist(),
                yaw_double=targets.yaw_double.tolist(), yaw_valid=targets.yaw_valid.tolist(),
                front=front, linked=linked, contacts_valid=f["contacts_valid"])


class CarriedLabeler:
    def __init__(self, config: LabelConfig | None = None):
        self.config = config or LabelConfig()
        self._history = []
        self._ids = None
        self._pending = []
        self._since = []

    def _classify(self, i):
        c = self.config
        h = self._history
        if h[-1]["time"] - h[0]["time"] < c.window_seconds - c.edge_tolerance:
            return None
        if any(not s["contacts_valid"] for s in h):
            return None
        times = np.array([s["time"] for s in h])
        brush = np.array([s["brush"] for s in h])
        centers = np.array([s["centers"][i] for s in h])
        # Clip the oldest secant to the exact 200ms boundary if ticks are irregular.
        cutoff = times[-1] - c.window_seconds
        if times[0] < cutoff:
            alpha = (cutoff - times[0]) / (times[1] - times[0])
            centers[0] += alpha * (centers[1] - centers[0])
            brush[0] += alpha * (brush[1] - brush[0])
            times[0] = cutoff
        if np.linalg.norm(brush[-1] - brush[0]) <= c.relative_displacement_m + c.edge_tolerance:
            return None
        relative = centers - brush
        speed = np.linalg.norm(np.diff(relative, axis=0) / np.diff(times)[:, None], axis=1).max()
        drift = np.linalg.norm(relative[:, None] - relative[None, :], axis=2).max()
        if (abs(speed - c.speed_difference_mps) <= c.edge_tolerance
                or abs(drift - c.relative_displacement_m) <= c.edge_tolerance):
            return None
        front = np.array([s["front"][i] for s in h])
        linked = np.array([s["linked"][i] for s in h])
        if np.any(np.abs(front) <= c.edge_tolerance) or np.any(linked & (front < 0)):
            return None
        comoving = speed < c.speed_difference_mps and drift < c.relative_displacement_m
        if linked.all() and np.all(front > 0) and comoving:
            return True
        if not linked.any() and not comoving:
            return False
        return None

    def update(self, frame: TruthFrame) -> LabelFrame:
        c, now = self.config, frame.timestamp
        f = frame.raw.fields
        ids = f["object_ids"]
        if self._history and now <= self._history[-1]["time"]:
            raise ValueError("truth timestamps must be strictly increasing")
        if self._ids is not None and ids != self._ids:
            raise ValueError("object identities changed within episode")
        n = len(ids)
        if self._ids is None:
            self._ids, self._pending, self._since = ids, [None] * n, [None] * n
        targets = project_envelopes(frame)
        # Center secants use bounded stored projected evidence, not raw qvel.
        if self._history:
            last = self._history[-1]
            dt = now - last["time"]
            if dt <= c.max_evidence_gap_seconds + c.edge_tolerance:
                targets.velocity_xy[:] = (targets.center_xy - np.asarray(last["centers"])) / dt
                targets.velocity_valid[:] = True
                angular, angular_valid = _angular_secant(
                    targets.yaw_double, targets.yaw_valid, np.asarray(last["yaw_double"]),
                    np.asarray(last["yaw_valid"]), dt)
                targets.angular_velocity[:] = angular
                targets.angular_velocity_valid[:] = angular_valid
            else:
                self._history, self._pending, self._since = [], [None] * n, [None] * n
        self._history.append(_summary(frame, targets))
        cutoff = now - c.window_seconds
        while len(self._history) > 1 and self._history[1]["time"] <= cutoff + c.edge_tolerance:
            self._history.pop(0)
        carried, valid = np.zeros(n, bool), np.zeros(n, bool)
        for i in range(n):
            candidate = self._classify(i)
            if candidate is None:
                self._pending[i], self._since[i] = None, None
            else:
                if self._pending[i] != candidate:
                    self._pending[i], self._since[i] = candidate, now
                duration = c.join_seconds if candidate else c.leave_seconds
                if now - self._since[i] >= duration - c.edge_tolerance:
                    carried[i], valid[i] = candidate, True
        collected = targets.native_collected & targets.native_valid
        carried[collected], valid[collected] = False, True
        state = np.where(collected, 2, np.where(carried, 1, 0))
        state_valid = collected | (valid & targets.native_valid)
        return LabelFrame(now, targets, carried, valid, state, state_valid, targets.valid_steps,
                          dict(config=asdict(c), config_hash=c.config_hash,
                               geometry_version=GEOMETRY_VERSION, quality_reviewed=False))

    def snapshot(self):
        """JSON-safe bounded evidence, pending timers and exact config provenance."""
        return json.loads(_canonical(dict(schema=1, config=asdict(self.config),
            config_hash=self.config.config_hash, geometry_version=GEOMETRY_VERSION,
            object_ids=self._ids, history=self._history, pending=self._pending, since=self._since)))

    @property
    def state_hash(self):
        return _hash(self.snapshot())

    def restore(self, state):
        state = json.loads(_canonical(state))
        if (state.get("schema") != 1 or state.get("config") != asdict(self.config)
                or state.get("config_hash") != self.config.config_hash
                or state.get("geometry_version") != GEOMETRY_VERSION):
            raise ValueError("snapshot config/version mismatch")
        ids = state["object_ids"]
        n = len(ids) if ids is not None else 0
        if (ids is not None and (not 1 <= n <= 6 or any(type(i) is not int or i < 0 for i in ids))
                or len(set(ids or [])) != n or len(state["pending"]) != n or len(state["since"]) != n
                or any(p is not None and type(p) is not bool for p in state["pending"])):
            raise ValueError("invalid label snapshot")
        history = state["history"]
        if bool(history) != (ids is not None):
            raise ValueError("invalid label snapshot identity/history")
        for j, s in enumerate(history):
            if (len(s["centers"]) != n or np.asarray(s["centers"]).shape != (n, 2)
                    or np.asarray(s["brush"]).shape != (2,)
                    or np.asarray(s["yaw_double"]).shape != (n, 2)
                    or len(s["yaw_valid"]) != n or any(type(v) is not bool for v in s["yaw_valid"])
                    or len(s["front"]) != n or len(s["linked"]) != n
                    or any(type(v) is not bool for v in s["linked"])
                    or type(s["contacts_valid"]) is not bool
                    or type(s["time"]) not in (int, float) or s["time"] < 0
                    or (j and not 0 < s["time"] - history[j - 1]["time"]
                        <= self.config.max_evidence_gap_seconds + self.config.edge_tolerance)):
                raise ValueError("invalid label snapshot evidence")
            if not all(np.isfinite(np.asarray(s[name], dtype=float)).all()
                       for name in ("centers", "brush", "yaw_double", "front")):
                raise ValueError("invalid label snapshot numbers")
        if history and history[-1]["time"] - history[0]["time"] > (
                self.config.window_seconds + self.config.max_evidence_gap_seconds + self.config.edge_tolerance):
            raise ValueError("invalid label snapshot window")
        for pending, since in zip(state["pending"], state["since"]):
            if ((pending is None) != (since is None) or (since is not None
                    and (type(since) not in (int, float) or not 0 <= since <= history[-1]["time"]))):
                raise ValueError("invalid label snapshot timer")
        self._ids = tuple(ids) if ids is not None else None
        self._history, self._pending, self._since = history, state["pending"], state["since"]
