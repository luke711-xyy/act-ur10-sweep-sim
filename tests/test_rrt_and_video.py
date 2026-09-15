"""Tests for transfer-phase RRT planning and the multi-camera recorder."""

import numpy as np
import pytest

from sim.config import load_config
from sim.planners.rrt import (Box, CollisionModel, path_length, plan_transfer,
                              rrt_connect, shortcut_path)
from sim.video import HUD_HEIGHT, EpisodeRecorder


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg.set_path("planner.transfer.mode", "rrt")
    return cfg


def planar_model(obstacle, z=0.008, resolution=0.004):
    """A model with a single wall and no room to climb over it."""
    return CollisionModel([obstacle], [-0.5, -0.4, z], [0.5, 0.4, z], resolution)


WALL = Box(np.array([0.18, -0.20, -0.1]), np.array([0.22, 0.20, 0.05]), "wall")


# ------------------------------------------------------------------- geometry
def test_box_containment_and_inflation():
    box = Box(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    assert box.contains(np.array([0.05, 0.05, 0.05]))
    assert not box.contains(np.array([0.15, 0.05, 0.05]))
    grown = box.inflated(xy=0.02, down=0.06)
    assert grown.contains(np.array([0.12, 0.05, 0.05]))       # widened sideways
    assert grown.contains(np.array([0.05, 0.05, -0.05]))      # extended downwards
    assert not grown.contains(np.array([0.05, 0.05, 0.15]))   # never upwards


def test_collision_model_bounds_and_segments():
    model = planar_model(WALL)
    assert model.free(np.array([0.40, 0.0, 0.008]))
    assert not model.free(np.array([0.20, 0.0, 0.008]))
    assert not model.free(np.array([9.0, 0.0, 0.008]))        # out of bounds
    assert not model.segment_free([0.40, 0.0, 0.008], [-0.05, 0.0, 0.008])
    assert model.segment_free([0.40, 0.30, 0.008], [0.30, 0.30, 0.008])


def test_tray_walls_become_obstacles_only_below_their_rim(cfg):
    model = CollisionModel.for_transfer(cfg, [0.0, 0.0, 0.0], [-0.45, 0.0, 0.0])
    rim = float(cfg.target.top_z) if "top_z" in cfg.target else float(cfg.table.top_z)
    back_wall_x = float(cfg.target.x_min) - float(cfg.target.wall_thickness) / 2
    assert not model.free(np.array([back_wall_x, 0.0, rim + 0.01]))    # inside the wall
    assert model.free(np.array([back_wall_x, 0.0, rim + 0.09]))        # clear above it


def test_detected_components_become_obstacles_only_when_low(cfg):
    points = np.array([[0.20, 0.0]])
    model = CollisionModel.for_transfer(cfg, [0.42, 0.0, 0.0], [-0.05, 0.0, 0.0], points)
    assert not model.free(np.array([0.20, 0.0, 0.004]))   # tip would sweep through it
    assert model.free(np.array([0.20, 0.0, 0.060]))       # the default travel height clears it


# ----------------------------------------------------------------- rrt-connect
def test_direct_line_is_used_when_it_is_free():
    model = planar_model(WALL)
    path = rrt_connect([0.40, 0.30, 0.008], [0.30, 0.30, 0.008], model,
                       np.random.default_rng(0))
    assert len(path) == 2                      # no sampling, therefore no randomness


def test_rrt_routes_around_an_obstacle_and_the_path_is_collision_free():
    model = planar_model(WALL)
    start, goal = [0.40, 0.0, 0.008], [-0.05, 0.0, 0.008]
    path = rrt_connect(start, goal, model, np.random.default_rng(3),
                       step_size=0.05, max_iters=6000)
    assert path is not None, "RRT failed to find an obvious detour"
    assert np.allclose(path[0], start) and np.allclose(path[-1], goal)
    assert all(model.segment_free(path[i], path[i + 1]) for i in range(len(path) - 1))
    assert path_length(path) > np.linalg.norm(np.subtract(goal, start))


def test_rrt_is_deterministic_for_a_seed():
    model = planar_model(WALL)
    args = ([0.40, 0.0, 0.008], [-0.05, 0.0, 0.008], model)
    a = rrt_connect(*args, np.random.default_rng(11), step_size=0.05, max_iters=6000)
    b = rrt_connect(*args, np.random.default_rng(11), step_size=0.05, max_iters=6000)
    assert a is not None
    assert len(a) == len(b) and np.allclose(np.asarray(a), np.asarray(b))


def test_rrt_returns_none_when_sealed_in():
    sealed = CollisionModel([Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))],
                            [-1, -1, -1], [1, 1, 1], 0.01)
    assert rrt_connect([0.4, 0.0, 0.0], [-0.05, 0.0, 0.0], sealed,
                       np.random.default_rng(0)) is None


def test_shortcutting_shortens_and_stays_free():
    model = planar_model(WALL)
    raw = rrt_connect([0.40, 0.0, 0.008], [-0.05, 0.0, 0.008], model,
                      np.random.default_rng(3), step_size=0.04, max_iters=6000)
    assert raw is not None and len(raw) > 2
    short = shortcut_path(raw, model, np.random.default_rng(3), 200)
    assert len(short) <= len(raw)
    assert path_length(short) <= path_length(raw) + 1e-9
    assert all(model.segment_free(short[i], short[i + 1]) for i in range(len(short) - 1))
    assert np.allclose(short[0], raw[0]) and np.allclose(short[-1], raw[-1])


# ------------------------------------------------------------- plan_transfer
def test_transfer_defaults_to_a_straight_line(cfg):
    cfg.set_path("planner.transfer.mode", "direct")
    wall = np.stack([np.full(9, 0.20), np.linspace(-0.1, 0.1, 9)], axis=1)
    assert len(plan_transfer(cfg, [0.42, 0.0, 0.0], [-0.05, 0.0, 0.0],
                             np.random.default_rng(0), wall)) == 2


def test_default_travel_height_already_clears_the_parts(cfg):
    """At the default z_travel the transfer is free by construction, so RRT is a no-op."""
    wall = np.stack([np.full(11, 0.20), np.linspace(-0.12, 0.12, 11)], axis=1)
    z = float(cfg.workspace.z_travel)
    assert len(plan_transfer(cfg, [0.42, 0.0, z], [-0.05, 0.0, z],
                             np.random.default_rng(0), wall)) == 2


def test_transfer_falls_back_instead_of_refusing_to_move(cfg):
    """A start point inside an inflated obstacle must not deadlock the episode."""
    points = np.array([[0.20, 0.0]])
    path = plan_transfer(cfg, [0.20, 0.0, 0.0], [-0.05, 0.0, 0.0],
                         np.random.default_rng(0), points)
    assert len(path) == 2


def test_transfer_is_reproducible(cfg):
    cfg.set_path("workspace.z_travel", 0.008)
    wall = np.stack([np.full(13, 0.20), np.linspace(-0.14, 0.14, 13)], axis=1)
    args = (cfg, [0.42, 0.0, 0.008], [-0.05, 0.0, 0.008])
    a = plan_transfer(*args, np.random.default_rng(5), wall)
    b = plan_transfer(*args, np.random.default_rng(5), wall)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_controller_approach_uses_the_planned_transfer(cfg):
    """The APPROACH trajectory must contain the planned waypoints, in order."""
    from sim.controllers.hybrid import HybridForcePositionController
    from sim.planners.base import SweepStroke

    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    tcp = np.array([0.44, -0.34, 0.26])
    controller.reset(tcp)
    controller.start_stroke(SweepStroke(0.35, 0.10, -0.40, 0.10), tcp, 0.0,
                            obstacles=np.array([[0.30, 0.10]]))
    assert len(controller.last_transfer_waypoints) >= 2
    traj = controller._approach_traj
    # lift, then one segment per transfer leg, then the descent
    assert len(traj) == 1 + (len(controller.last_transfer_waypoints) - 1) + 1
    assert np.allclose(traj.point(0.0), tcp)
    assert np.allclose(traj.point(traj.duration),
                       [0.35, 0.10, float(cfg.workspace.z_search_start)])


# --------------------------------------------------------------------- video
def test_recorder_reads_its_configuration(cfg):
    rec = EpisodeRecorder(cfg, "/tmp/does-not-exist.mp4")
    assert rec.cameras == list(cfg.video.cameras)
    assert rec.fps == int(cfg.video.fps)
    assert rec.force_gauge_max == pytest.approx(2.5 * float(cfg.controller.desired_force))


def test_recorder_composes_panes_and_hud(cfg):
    rec = EpisodeRecorder(cfg, "/tmp/does-not-exist.mp4",
                          cameras=["scene_cam", "overhead_cam"])
    pane = np.zeros((rec.height, rec.width, 3), dtype=np.uint8)
    labelled = rec._label_pane(pane, "scene_cam")
    assert labelled.shape == pane.shape
    assert labelled[:HUD_HEIGHT].any()                 # the label strip drew something
    frame = np.concatenate([labelled, labelled], axis=1)
    hud = rec._hud(frame.shape[1], {"t": 1.0, "phase": "SWEEP", "stroke": 0,
                                    "force_desired": 3.0, "force_measured": 3.2,
                                    "in_contact": True, "collected": 1, "total": 3,
                                    "planner": "visual_greedy"})
    assert hud.shape == (HUD_HEIGHT, frame.shape[1], 3)
    full = np.concatenate([frame, hud], axis=0)
    assert full.shape == (rec.height + HUD_HEIGHT, 2 * rec.width, 3)


def test_recorder_pads_to_a_macroblock_multiple(cfg):
    rec = EpisodeRecorder(cfg, "/tmp/does-not-exist.mp4")
    padded = rec._pad(np.zeros((101, 203, 3), dtype=np.uint8))
    assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0


def test_hud_flags_an_over_range_force(cfg):
    rec = EpisodeRecorder(cfg, "/tmp/does-not-exist.mp4")
    normal = rec._hud(800, {"force_desired": 3.0, "force_measured": 3.0})
    over = rec._hud(800, {"force_desired": 3.0, "force_measured": 99.0})
    assert not np.array_equal(normal, over)


def test_extra_cameras_are_render_only():
    """Perception must never be able to pick up a video camera."""
    import inspect

    from sim.environments import sweep_env
    from sim.perception import camera as camera_module

    source = inspect.getsource(sweep_env) + inspect.getsource(camera_module)
    for name in ("overhead_cam", "side_cam", "follow_cam"):
        assert name not in source, f"{name} leaked into the perception path"
    assert source.count('"scene_cam"') >= 1


# ------------------------------------------------------------- 2-D preview
def _synthetic_result(cfg, n_steps=60, n_parts=3):
    """A minimal EpisodeResult-shaped object, enough to drive the animator."""
    from sim.controllers.hybrid import ControlRecord
    from sim.environments.episode import EpisodeResult
    from sim.metrics import compute_episode_metrics

    trace = []
    for i in range(n_steps):
        sweeping = 20 <= i < 50
        trace.append(ControlRecord(
            t=i * 0.01, stroke=0, phase="SWEEP" if sweeping else "APPROACH",
            tcp_x=0.30 - 0.01 * i, tcp_y=0.02, tcp_z=0.0,
            cmd_x=0.0, cmd_y=0.0, cmd_z=0.0, cmd_yaw=0.0,
            z_nominal=0.0, delta_z=0.0, force_desired=3.0 if sweeping else 0.0,
            force_raw=3.0 if sweeping else 0.0, force_filtered=3.0 if sweeping else 0.0,
            in_contact=sweeping,
        ))
    start = np.stack([np.linspace(0.05, 0.25, n_parts), np.zeros(n_parts)], axis=1)
    positions = np.concatenate([start, np.zeros((n_parts, 1))], axis=1)
    metrics = compute_episode_metrics(
        trace=trace, n_components=n_parts, n_collected=n_parts, n_pushed_out=0,
        n_strokes=1, sim_time=n_steps * 0.01, success=True, seed=0,
        planner="visual_greedy", perception="ground_truth", geometry="hex_nut",
    )
    return EpisodeResult(
        metrics=metrics, trace=trace, strokes=[], observations=[], events=[],
        layout=[{"index": i} for i in range(n_parts)], config=cfg.to_dict(),
        initial_positions=positions, final_positions=positions,
        component_tracks=np.repeat(start[None, :, :], n_steps // 2, axis=0),
        track_times=np.arange(n_steps // 2) * 0.02,
    )


def test_preview_writes_a_playable_file(cfg, tmp_path):
    import imageio.v2 as imageio

    from sim.preview import animate_episode

    path = str(tmp_path / "preview.mp4")
    written = animate_episode(_synthetic_result(cfg), path, fps=10, speed=2.0)
    assert written == path
    reader = imageio.get_reader(path)
    assert reader.count_frames() > 2
    frame = reader.get_data(1)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_preview_handles_a_missing_track(cfg, tmp_path):
    from sim.preview import animate_episode

    result = _synthetic_result(cfg)
    result.component_tracks = None
    result.track_times = None
    assert animate_episode(result, str(tmp_path / "p.mp4"), fps=10, speed=4.0) is not None


def test_preview_returns_none_on_an_empty_trace(cfg, tmp_path):
    from sim.preview import animate_episode

    result = _synthetic_result(cfg)
    result.trace = []
    assert animate_episode(result, str(tmp_path / "p.mp4")) is None


def test_comparison_preview_runs_on_several_results(cfg, tmp_path):
    import imageio.v2 as imageio

    from sim.preview import animate_comparison

    # the two episodes have different lengths: the shorter pane must freeze while
    # the longer one keeps running, which is the whole point of the figure
    results = [_synthetic_result(cfg, n_steps=40), _synthetic_result(cfg, n_steps=120)]
    path = str(tmp_path / "compare.mp4")
    written = animate_comparison(results, path, fps=10, speed=1.0, labels=["A", "B"])
    assert written == path
    expected = int(np.ceil(results[1].trace[-1].t / 0.1))
    frames = imageio.get_reader(path).count_frames()
    assert frames >= expected - 1, (frames, expected)
    assert animate_comparison([], str(tmp_path / "empty.mp4")) is None


def test_tool_polygon_rotates_about_the_tcp():
    from sim.preview import _tool_polygon

    square = _tool_polygon(0.0, 0.0, 0.0, 0.01, 0.03)
    turned = _tool_polygon(0.0, 0.0, np.pi / 2, 0.01, 0.03)
    assert square.shape == (4, 2)
    assert np.allclose(np.mean(square, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.mean(turned, axis=0), 0.0, atol=1e-12)
    # a 90 deg turn swaps the extents
    assert np.ptp(square[:, 0]) == pytest.approx(np.ptp(turned[:, 1]))


def test_ffmpeg_is_found_without_it_being_on_path():
    """A clean `pip install` puts ffmpeg in site-packages, not on PATH."""
    import matplotlib
    import matplotlib.animation as animation

    from sim.preview import ensure_ffmpeg

    original = matplotlib.rcParams["animation.ffmpeg_path"]
    try:
        matplotlib.rcParams["animation.ffmpeg_path"] = "no-such-ffmpeg-binary"
        assert not animation.FFMpegWriter.isAvailable()
        assert ensure_ffmpeg(), "imageio-ffmpeg's bundled binary was not picked up"
        assert animation.FFMpegWriter.isAvailable()
    finally:
        matplotlib.rcParams["animation.ffmpeg_path"] = original
