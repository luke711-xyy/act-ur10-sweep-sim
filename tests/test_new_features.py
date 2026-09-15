"""Tests for the global-sweep baseline, clustered layouts, failure-mode metrics
and the ACT-facing dataset export."""

import numpy as np
import pytest

from sim.config import load_config
from sim.controllers.hybrid import ControlRecord, HybridForcePositionController
from sim.environments.layout import in_target_region, sample_layout
from sim.export_dataset import (N_WAYPOINTS, _resample_xy_yaw, _wrench_history,
                                goal_region_grid, stroke_waypoints)
from sim.metrics import compute_episode_metrics, count_jam_events, touchdown_overshoot
from sim.perception.base import GridSpec, SceneObservation
from sim.planners.base import PLANNER_NAMES, SweepStroke, build_planner
from sim.planners.geometry_utils import pusher_width
from sim.planners.global_sweep import GlobalSweepPlanner


@pytest.fixture
def cfg():
    return load_config()


def make_observation(cfg, points, tcp=(0.44, -0.34, 0.26)):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    grid = GridSpec.from_config(cfg, res=0.008)
    return SceneObservation(
        t=0.0, points=points, counts=np.ones(len(points)),
        areas=np.full(len(points), 8e-5),
        occupancy=grid.empty(), grid=grid,
        tcp=np.asarray(tcp, dtype=float), backend="test",
    )


# ------------------------------------------------- Planner C (global sweep)
def test_global_sweep_is_registered(cfg):
    assert "global_sweep" in PLANNER_NAMES
    assert isinstance(build_planner(cfg, "global_sweep"), GlobalSweepPlanner)


def test_global_sweep_aims_at_the_centroid_of_everything(cfg):
    planner = GlobalSweepPlanner(cfg)
    stroke = planner.plan(make_observation(cfg, [[0.30, 0.20], [0.25, -0.20], [0.10, 0.00]]))
    assert stroke is not None
    assert stroke.y_start == pytest.approx(0.0, abs=0.01)     # centroid lane
    assert stroke.x_start > 0.30                              # behind the right-most part
    assert stroke.meta["kind"] == "global"
    assert stroke.meta["n_detected"] == 3


def test_global_sweep_leaves_off_lane_parts_behind(cfg):
    """The whole point of the baseline: one lane cannot capture a spread-out scene."""
    planner = GlobalSweepPlanner(cfg)
    spread = planner.plan(make_observation(cfg, [[0.30, 0.22], [0.25, -0.22], [0.10, 0.0]]))
    tight = planner.plan(make_observation(cfg, [[0.30, 0.01], [0.25, -0.01], [0.10, 0.0]]))
    assert spread.meta["n_captured"] < 3
    assert tight.meta["n_captured"] == 3


def test_global_sweep_emits_one_stroke_per_observation(cfg):
    planner = GlobalSweepPlanner(cfg)
    obs = make_observation(cfg, [[0.30, 0.10], [0.20, -0.10]])
    first = planner.plan(obs)
    planner.notify_stroke_done(first, {"collected_delta": 1})
    second = planner.plan(make_observation(cfg, [[0.20, -0.10]]))
    assert second.y_start == pytest.approx(-0.10, abs=0.01)   # re-planned on what is left


def test_global_sweep_handles_an_empty_scene_and_gives_up(cfg):
    planner = GlobalSweepPlanner(cfg)
    assert planner.plan(make_observation(cfg, [])) is None
    stroke = planner.plan(make_observation(cfg, [[0.3, 0.0]]))
    for _ in range(int(cfg.planner.greedy.max_no_progress)):
        planner.notify_stroke_done(stroke, {"collected_delta": 0})
    assert planner.is_exhausted()


def test_global_sweep_never_starts_inside_the_release_band(cfg):
    planner = GlobalSweepPlanner(cfg)
    guard = float(cfg.controller.x_release_line) + float(cfg.controller.release_margin)
    stroke = planner.plan(make_observation(cfg, [[-0.10, 0.0]]))
    assert stroke.x_start > guard


# ------------------------------------------------------- clustered layouts
def test_cluster_mode_puts_everything_in_one_group(cfg):
    cfg.set_path("components.spawn_mode", "cluster")
    cfg.set_path("components.count", 5)
    radius = float(cfg.components.cluster.max_radius)
    for seed in range(15):
        layout = sample_layout(cfg, np.random.default_rng(seed))
        xy = np.array([[i["x"], i["y"]] for i in layout])
        centroid = xy.mean(axis=0)
        assert np.max(np.linalg.norm(xy - centroid, axis=1)) <= 2 * radius
        assert not in_target_region(xy, cfg.target).any()


def test_cluster_mode_is_tighter_than_uniform(cfg):
    cfg.set_path("components.count", 5)

    def spread(mode):
        cfg.set_path("components.spawn_mode", mode)
        out = []
        for seed in range(10):
            xy = np.array([[i["x"], i["y"]]
                           for i in sample_layout(cfg, np.random.default_rng(seed))])
            out.append(np.linalg.norm(xy - xy.mean(axis=0), axis=1).max())
        return float(np.mean(out))

    assert spread("cluster") < 0.5 * spread("uniform")


def test_cluster_mode_is_reproducible_and_randomises_the_centre(cfg):
    cfg.set_path("components.spawn_mode", "cluster")
    cfg.set_path("components.count", 4)
    a = sample_layout(cfg, np.random.default_rng(2))
    b = sample_layout(cfg, np.random.default_rng(2))
    c = sample_layout(cfg, np.random.default_rng(3))
    assert [i["x"] for i in a] == [i["x"] for i in b]
    assert np.mean([i["x"] for i in a]) != np.mean([i["x"] for i in c])


def test_unknown_spawn_mode_is_rejected(cfg):
    cfg.set_path("components.spawn_mode", "nope")
    with pytest.raises(ValueError):
        sample_layout(cfg, np.random.default_rng(0))


# --------------------------------------------------- failure-mode detection
def record(t, phase, fx=0.0, fy=0.0, force=3.0):
    return ControlRecord(
        t=t, stroke=0, phase=phase, tcp_x=0.0, tcp_y=0.0, tcp_z=0.0,
        cmd_x=0.0, cmd_y=0.0, cmd_z=0.0, cmd_yaw=0.0, z_nominal=0.0, delta_z=0.0,
        force_desired=3.0, force_raw=force, force_filtered=force, in_contact=True,
        fx=fx, fy=fy,
    )


def test_tangential_force_is_the_in_plane_magnitude():
    assert record(0.0, "SWEEP", fx=3.0, fy=4.0).tangential_force == pytest.approx(5.0)


def test_jam_detection_needs_a_sustained_load(cfg):
    dt = 1.0 / float(cfg.sim.control_hz)
    threshold = float(cfg.metrics.jam_force_threshold)
    samples = max(1, int(round(float(cfg.metrics.jam_min_duration) / dt)))

    quiet = [record(i * dt, "SWEEP", fx=threshold * 0.4) for i in range(200)]
    assert count_jam_events(quiet, cfg) == 0

    spike = list(quiet)
    for i in range(samples - 1):                       # one sample too short
        spike[50 + i] = record((50 + i) * dt, "SWEEP", fx=threshold + 1.0)
    assert count_jam_events(spike, cfg) == 0

    jam = list(quiet)
    for i in range(samples + 5):
        jam[50 + i] = record((50 + i) * dt, "SWEEP", fx=threshold + 1.0)
    assert count_jam_events(jam, cfg) == 1


def test_jam_detection_counts_separate_events(cfg):
    dt = 1.0 / float(cfg.sim.control_hz)
    threshold = float(cfg.metrics.jam_force_threshold)
    samples = max(1, int(round(float(cfg.metrics.jam_min_duration) / dt)))
    trace = [record(i * dt, "SWEEP", fx=0.0) for i in range(300)]
    for start in (40, 180):
        for i in range(samples + 3):
            trace[start + i] = record((start + i) * dt, "SWEEP", fx=threshold + 2.0)
    assert count_jam_events(trace, cfg) == 2


def test_jam_detection_ignores_non_contact_phases(cfg):
    dt = 1.0 / float(cfg.sim.control_hz)
    trace = [record(i * dt, "APPROACH", fx=50.0) for i in range(300)]
    assert count_jam_events(trace, cfg) == 0


def test_touchdown_overshoot():
    trace = ([record(i * 0.01, "FORCE_RAMP", force=1.0 + i * 0.5) for i in range(8)]
             + [record(0.08 + i * 0.01, "SWEEP", force=3.1) for i in range(200)])
    assert touchdown_overshoot(trace, 3.0) == pytest.approx(4.5 - 3.0, abs=1e-6)
    flat = [record(i * 0.01, "SWEEP", force=2.5) for i in range(50)]
    assert touchdown_overshoot(flat, 3.0) == 0.0
    assert touchdown_overshoot([], 3.0) == 0.0


def test_failure_metrics_reach_the_episode_record():
    m = compute_episode_metrics(
        trace=[record(0.0, "SWEEP")], n_components=3, n_collected=2, n_pushed_out=0,
        n_strokes=2, sim_time=10.0, success=False, seed=0, planner="visual_greedy",
        perception="ground_truth", geometry="washer",
        n_ride_over=2, n_jam_events=1, touchdown_overshoot_max=1.4,
        touchdown_overshoot_mean=0.9,
    )
    assert m.n_ride_over == 2 and m.n_jam_events == 1
    assert m.touchdown_overshoot_max == pytest.approx(1.4)
    assert "n_ride_over" in m.to_dict()


# ------------------------------------------------------------ ACT export
def test_resample_keeps_endpoints_and_spaces_by_arc_length():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    yaws = np.array([0.0, 0.0, 0.5])
    out = _resample_xy_yaw(path, yaws, 9)
    assert out.shape == (9, 3)
    assert np.allclose(out[0, :2], path[0])
    assert np.allclose(out[-1, :2], path[-1])
    steps = np.linalg.norm(np.diff(out[:, :2], axis=0), axis=1)
    assert np.allclose(steps, steps[0], atol=1e-9)


def test_resample_handles_degenerate_paths():
    assert _resample_xy_yaw(np.zeros((0, 2)), np.zeros(0), 4).shape == (4, 3)
    single = _resample_xy_yaw(np.array([[1.0, 2.0]]), np.array([0.3]), 4)
    assert np.allclose(single, [[1.0, 2.0, 0.3]] * 4)


def test_stroke_waypoints_are_relative_to_the_capture_pose():
    capture = np.array([0.44, -0.34, 0.0])
    trace = [
        ControlRecord(t=i * 0.01, stroke=0, phase="SWEEP",
                      tcp_x=0.30 - i * 0.01, tcp_y=0.05, tcp_z=0.0,
                      cmd_x=0.0, cmd_y=0.0, cmd_z=0.0, cmd_yaw=0.2,
                      z_nominal=0.0, delta_z=0.0, force_desired=3.0, force_raw=3.0,
                      force_filtered=3.0, in_contact=True)
        for i in range(40)
    ]
    wp = stroke_waypoints(trace, 0, capture, fallback=((0.3, 0.05), (-0.1, 0.05), 0.2))
    assert wp.shape == (N_WAYPOINTS, 3)
    assert wp[0, 0] == pytest.approx(0.30 - 0.44, abs=1e-5)
    assert wp[0, 1] == pytest.approx(0.05 + 0.34, abs=1e-5)
    assert np.allclose(wp[:, 2], 0.2)
    assert wp[-1, 0] < wp[0, 0]            # the stroke moves towards -X


def test_stroke_waypoints_fall_back_to_the_commanded_stroke():
    wp = stroke_waypoints([], 0, np.zeros(3), fallback=((0.3, 0.1), (-0.4, 0.1), 0.0))
    assert wp.shape == (N_WAYPOINTS, 3)
    assert wp[0, 0] == pytest.approx(0.3)
    assert wp[-1, 0] == pytest.approx(-0.4)


def test_goal_region_grid_matches_the_tray(cfg):
    grid = goal_region_grid(cfg)
    spec = GridSpec.from_config(cfg, res=float(cfg.perception.grid_res))
    assert grid.shape == spec.shape
    assert grid.any() and not grid.all()
    iy, ix = np.nonzero(grid)
    centres = spec.to_xy(iy, ix)
    assert centres[:, 0].max() <= float(cfg.target.x_max) + spec.res
    assert abs(centres[:, 1]).max() <= float(cfg.target.y_max) + spec.res


def test_wrench_history_pads_at_the_start():
    arrays = {"t": np.arange(100) * 0.01,
              "wrench": np.tile(np.arange(100).reshape(-1, 1), (1, 6)).astype(np.float32)}
    early = _wrench_history(arrays, 0.02, 10)
    assert early.shape == (10, 6)
    assert np.allclose(early[:7], 0.0)                 # zero-padded
    late = _wrench_history(arrays, 0.50, 10)
    assert np.allclose(late[-1], 50.0)
    assert _wrench_history({}, 0.0, 10).shape == (10, 6)


# ------------------------------------------- full state reaches the trace
def test_controller_logs_the_wrench_and_velocity(cfg):
    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    tcp = np.array([0.42, 0.0, 0.15])
    controller.reset(tcp)
    controller.start_stroke(SweepStroke(0.35, 0.0, -0.40, 0.0), tcp, 0.0)
    controller.step(0.0, tcp, 0.0,
                    wrench=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
                    tcp_velocity=np.array([0.1, 0.2, 0.3]), tcp_yaw=0.4)
    r = controller.trace[-1]
    assert (r.fx, r.fy, r.fz) == (1.0, 2.0, 3.0)
    assert (r.tx, r.ty, r.tz) == (0.1, 0.2, 0.3)
    assert (r.tcp_vx, r.tcp_vy, r.tcp_vz) == (0.1, 0.2, 0.3)
    assert r.tcp_yaw == pytest.approx(0.4)
    assert r.tangential_force == pytest.approx(np.hypot(1.0, 2.0))


def test_controller_still_works_without_the_optional_state(cfg):
    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    tcp = np.array([0.42, 0.0, 0.15])
    controller.reset(tcp)
    controller.start_stroke(SweepStroke(0.35, 0.0, -0.40, 0.0), tcp, 0.0)
    controller.step(0.0, tcp, 0.0)
    r = controller.trace[-1]
    assert r.fx == 0.0 and r.tcp_vx == 0.0


# ------------------------------------------- the wrench is not optional
def test_controller_refuses_to_run_blind_when_full_state_is_required(cfg):
    """A missing wrench must stop the run, not quietly log a row of zeros."""
    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    controller.require_full_state = True
    tcp = np.array([0.42, 0.0, 0.15])
    controller.reset(tcp)
    controller.start_stroke(SweepStroke(0.35, 0.0, -0.40, 0.0), tcp, 0.0)
    with pytest.raises(RuntimeError):
        controller.step(0.0, tcp, 0.0)
    # ... and runs normally once the wrench is supplied
    controller.step(0.0, tcp, 0.0, wrench=np.zeros(6))


def test_episode_runner_rejects_an_environment_without_a_wrench():
    from sim.environments.episode import require_wrench

    class NoSensor:
        def tcp(self):
            return np.zeros(3)

    with pytest.raises(RuntimeError, match="wrench"):
        require_wrench(NoSensor())

    class WithSensor(NoSensor):
        def wrench(self):
            return np.zeros(6)

    require_wrench(WithSensor())      # must not raise


def test_part_contact_share_is_logged_separately(cfg):
    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    tcp = np.array([0.42, 0.0, 0.15])
    controller.reset(tcp)
    controller.start_stroke(SweepStroke(0.35, 0.0, -0.40, 0.0), tcp, 0.0)
    controller.step(0.0, tcp, 3.0,
                    wrench=np.array([1.5, 0.0, 3.0, 0.0, 0.0, 0.0]),
                    wrench_parts=np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    n_part_contacts=2)
    r = controller.trace[-1]
    assert r.fx_p == pytest.approx(0.05)
    assert r.n_part_contacts == 2
    assert r.tangential_force == pytest.approx(1.5)
    assert r.tangential_force_parts == pytest.approx(0.05)


def test_wrench_diagnostics_reach_the_metrics():
    def sample(ft, ft_parts, n_parts):
        return ControlRecord(
            t=0.0, stroke=0, phase="SWEEP", tcp_x=0.0, tcp_y=0.0, tcp_z=0.0,
            cmd_x=0.0, cmd_y=0.0, cmd_z=0.0, cmd_yaw=0.0, z_nominal=0.0, delta_z=0.0,
            force_desired=3.0, force_raw=3.0, force_filtered=3.0, in_contact=True,
            fx=ft, fx_p=ft_parts, n_part_contacts=n_parts,
        )

    trace = [sample(2.0, 0.1, 1) for _ in range(8)] + [sample(2.0, 0.0, 0) for _ in range(2)]
    m = compute_episode_metrics(
        trace=trace, n_components=1, n_collected=1, n_pushed_out=0, n_strokes=1,
        sim_time=1.0, success=True, seed=0, planner="visual_greedy",
        perception="ground_truth", geometry="hex_nut",
    )
    assert m.part_contact_ratio == pytest.approx(0.8)
    assert m.part_force_snr == pytest.approx(0.05)
    assert m.mean_tangential_force == pytest.approx(2.0)


def test_export_rejects_a_dead_wrench_channel(tmp_path):
    from sim.export_dataset import WrenchMissingError, validate_wrench

    class Result:
        pass

    def sample(fx):
        return ControlRecord(
            t=0.0, stroke=0, phase="SWEEP", tcp_x=0.0, tcp_y=0.0, tcp_z=0.0,
            cmd_x=0.0, cmd_y=0.0, cmd_z=0.0, cmd_yaw=0.0, z_nominal=0.0, delta_z=0.0,
            force_desired=3.0, force_raw=3.0, force_filtered=3.0, in_contact=True, fx=fx,
        )

    result = Result()
    result.metrics = type("M", (), {"seed": 0, "planner": "fixed"})()

    result.trace = [sample(0.0) for _ in range(10)]
    with pytest.raises(WrenchMissingError):
        validate_wrench(result)

    result.trace = [sample(1.2) for _ in range(10)]
    validate_wrench(result)                       # alive -> passes

    result.trace = []                             # nothing swept -> nothing to check
    validate_wrench(result)
