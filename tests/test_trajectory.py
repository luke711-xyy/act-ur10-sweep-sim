"""Unit tests for the Cartesian trajectory generation."""

import numpy as np
import pytest

from sim.planners.trajectory import (QuinticProfile, TrapezoidalProfile, Trajectory,
                                     make_linear_trajectory, make_segment)


@pytest.mark.parametrize("kind", ["quintic", "trapezoidal"])
def test_endpoints_and_monotonicity(kind):
    start, end = np.array([0.4, 0.1]), np.array([-0.4, -0.05])
    traj = make_linear_trajectory([start, end], speed=0.15, kind=kind)
    assert np.allclose(traj.point(0.0), start)
    assert np.allclose(traj.point(traj.duration), end)
    assert np.allclose(traj.point(traj.duration * 10), end)      # clamped after the end
    ts = np.linspace(0.0, traj.duration, 400)
    progress = [np.dot(traj.point(t) - start, end - start) for t in ts]
    assert np.all(np.diff(progress) >= -1e-12)


@pytest.mark.parametrize("kind", ["quintic", "trapezoidal"])
def test_speed_limit_is_respected(kind):
    speed = 0.15
    traj = make_linear_trajectory([[0.4, 0.1], [-0.4, 0.1]], speed=speed, kind=kind)
    ts = np.linspace(0.0, traj.duration, 2000)
    peak = max(float(np.linalg.norm(traj.velocity(t))) for t in ts)
    assert peak <= speed * 1.001


def test_zero_and_negative_endpoint_velocity():
    traj = make_linear_trajectory([[0.4, 0.0], [-0.4, 0.0]], speed=0.15, kind="quintic")
    assert np.allclose(traj.velocity(0.0), 0.0)
    assert np.allclose(traj.velocity(traj.duration), 0.0)


def test_multi_segment_is_continuous_and_hits_waypoints():
    waypoints = [[0.4, 0.0, 0.2], [0.4, 0.0, 0.06], [-0.2, 0.1, 0.06], [-0.2, 0.1, 0.008]]
    traj = make_linear_trajectory(waypoints, speed=0.3, kind="trapezoidal")
    assert len(traj) == 3
    assert np.allclose(traj.point(0.0), waypoints[0])
    assert np.allclose(traj.point(traj.duration), waypoints[-1])
    ts = np.linspace(0.0, traj.duration, 3000)
    pts = np.array([traj.point(t) for t in ts])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() < 0.01          # no discontinuous jumps between segments
    assert traj.length == pytest.approx(
        sum(np.linalg.norm(np.subtract(waypoints[i + 1], waypoints[i])) for i in range(3))
    )


def test_trapezoidal_degenerates_to_triangular_for_short_moves():
    profile = TrapezoidalProfile(distance=0.001, speed=1.0, accel=0.8)
    assert profile.t_flat == 0.0
    assert profile.s(profile.duration) == pytest.approx(1.0)


def test_quintic_peak_rate_constant():
    profile = QuinticProfile(1.0)
    rates = [profile.sd(t) for t in np.linspace(0, 1, 1001)]
    assert max(rates) == pytest.approx(QuinticProfile.PEAK_RATE, rel=1e-3)


def test_is_done_and_empty_trajectory():
    seg = make_segment([0, 0], [1, 0], speed=1.0, kind="trapezoidal")
    traj = Trajectory([seg])
    assert not traj.is_done(0.0)
    assert traj.is_done(traj.duration)
    with pytest.raises(ValueError):
        Trajectory([]).point(0.0)
