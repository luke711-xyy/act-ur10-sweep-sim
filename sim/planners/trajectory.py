"""Smooth Cartesian trajectory generation.

Planar sweeping does not need a sampling-based motion planner: the strokes are
straight-line Cartesian segments.  What matters is that the *time scaling* is
smooth, so that the position loop does not inject step changes into the contact
force.  Two scalar profiles are provided:

* ``quintic``      -- C2 continuous, zero velocity and acceleration at both ends
* ``trapezoidal``  -- constant-acceleration / cruise / constant-deceleration

Both are normalised to ``s(0) = 0``, ``s(T) = 1`` and are combined with a
straight line in Cartesian space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


class ScalarProfile:
    duration: float

    def s(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def sd(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class QuinticProfile(ScalarProfile):
    """s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 with tau = t / T."""

    #: peak of ds/dtau, used to convert a speed limit into a duration
    PEAK_RATE = 1.875

    def __init__(self, duration: float):
        self.duration = max(float(duration), 1e-6)

    def s(self, t: float) -> float:
        tau = float(np.clip(t / self.duration, 0.0, 1.0))
        return 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5

    def sd(self, t: float) -> float:
        if t <= 0.0 or t >= self.duration:
            return 0.0
        tau = t / self.duration
        return (30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4) / self.duration

    @classmethod
    def for_distance(cls, distance: float, speed: float, min_duration: float = 0.05):
        distance = abs(float(distance))
        speed = max(float(speed), 1e-6)
        return cls(max(cls.PEAK_RATE * distance / speed, min_duration))


class TrapezoidalProfile(ScalarProfile):
    """Constant-acceleration profile, degenerating to triangular when needed."""

    def __init__(self, distance: float, speed: float, accel: float, min_duration: float = 0.05):
        distance = abs(float(distance))
        self.distance = distance
        v = max(float(speed), 1e-6)
        a = max(float(accel), 1e-6)
        if distance < 1e-9:
            self.t_acc = self.t_flat = 0.0
            self.v_peak = 0.0
            self.duration = min_duration
            return
        if v * v / a >= distance:           # triangular
            self.v_peak = float(np.sqrt(distance * a))
            self.t_acc = self.v_peak / a
            self.t_flat = 0.0
        else:                                # trapezoidal
            self.v_peak = v
            self.t_acc = v / a
            self.t_flat = (distance - v * v / a) / v
        self.duration = max(2.0 * self.t_acc + self.t_flat, min_duration)

    def _distance_at(self, t: float) -> float:
        a = self.v_peak / self.t_acc if self.t_acc > 0 else 0.0
        if t <= 0.0:
            return 0.0
        if t < self.t_acc:
            return 0.5 * a * t * t
        if t < self.t_acc + self.t_flat:
            return 0.5 * a * self.t_acc ** 2 + self.v_peak * (t - self.t_acc)
        if t < 2 * self.t_acc + self.t_flat:
            td = t - self.t_acc - self.t_flat
            return (0.5 * a * self.t_acc ** 2 + self.v_peak * self.t_flat
                    + self.v_peak * td - 0.5 * a * td * td)
        return self.distance

    def s(self, t: float) -> float:
        if self.distance < 1e-9:
            return 1.0 if t >= self.duration else float(np.clip(t / self.duration, 0.0, 1.0))
        return float(np.clip(self._distance_at(float(t)) / self.distance, 0.0, 1.0))

    def sd(self, t: float) -> float:
        if self.distance < 1e-9:
            return 0.0
        a = self.v_peak / self.t_acc if self.t_acc > 0 else 0.0
        t = float(t)
        if t <= 0.0 or t >= 2 * self.t_acc + self.t_flat:
            return 0.0
        if t < self.t_acc:
            return a * t / self.distance
        if t < self.t_acc + self.t_flat:
            return self.v_peak / self.distance
        td = t - self.t_acc - self.t_flat
        return (self.v_peak - a * td) / self.distance


@dataclass
class LinearSegment:
    """Straight Cartesian segment with a scalar time scaling."""

    start: np.ndarray
    end: np.ndarray
    profile: ScalarProfile

    @property
    def duration(self) -> float:
        return self.profile.duration

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.end) - np.asarray(self.start)))

    def point(self, t: float) -> np.ndarray:
        s = self.profile.s(t)
        return np.asarray(self.start) + s * (np.asarray(self.end) - np.asarray(self.start))

    def velocity(self, t: float) -> np.ndarray:
        return self.profile.sd(t) * (np.asarray(self.end) - np.asarray(self.start))


class Trajectory:
    """A sequence of :class:`LinearSegment` evaluated on a common clock."""

    def __init__(self, segments: Sequence[LinearSegment]):
        self.segments: List[LinearSegment] = list(segments)
        self._starts: List[float] = []
        t = 0.0
        for seg in self.segments:
            self._starts.append(t)
            t += seg.duration
        self.duration = t

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def length(self) -> float:
        return float(sum(seg.length for seg in self.segments))

    def point(self, t: float) -> np.ndarray:
        if not self.segments:
            raise ValueError("empty trajectory")
        t = float(t)
        if t <= 0.0:
            return np.asarray(self.segments[0].start, dtype=float).copy()
        if t >= self.duration:
            return np.asarray(self.segments[-1].end, dtype=float).copy()
        idx = int(np.searchsorted(self._starts, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self.segments) - 1))
        return self.segments[idx].point(t - self._starts[idx])

    def velocity(self, t: float) -> np.ndarray:
        if not self.segments:
            raise ValueError("empty trajectory")
        t = float(t)
        if t <= 0.0 or t >= self.duration:
            return np.zeros_like(np.asarray(self.segments[0].start, dtype=float))
        idx = int(np.searchsorted(self._starts, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self.segments) - 1))
        return self.segments[idx].velocity(t - self._starts[idx])

    def is_done(self, t: float, tol: float = 1e-9) -> bool:
        return float(t) >= self.duration - tol


def make_segment(start, end, speed: float, kind: str = "quintic",
                 accel: float = 1.5, min_duration: float = 0.05) -> LinearSegment:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    distance = float(np.linalg.norm(end - start))
    if kind == "quintic":
        profile: ScalarProfile = QuinticProfile.for_distance(distance, speed, min_duration)
    elif kind == "trapezoidal":
        profile = TrapezoidalProfile(distance, speed, accel, min_duration)
    else:
        raise ValueError(f"unknown interpolation {kind!r}")
    return LinearSegment(start=start, end=end, profile=profile)


def make_linear_trajectory(waypoints: Sequence[Sequence[float]], speed: float,
                           kind: str = "quintic", accel: float = 1.5) -> Trajectory:
    pts = [np.asarray(w, dtype=float) for w in waypoints]
    if len(pts) < 2:
        raise ValueError("need at least two waypoints")
    segments = [make_segment(pts[i], pts[i + 1], speed, kind, accel)
                for i in range(len(pts) - 1)]
    return Trajectory(segments)
