"""Unit tests for both stroke planners."""

import numpy as np
import pytest

from sim.config import load_config
from sim.perception.base import GridSpec, SceneObservation
from sim.planners.base import SweepStroke, build_planner
from sim.planners.fixed_cover import FixedCoverPlanner
from sim.planners.geometry_utils import (point_segment_frames, pusher_width,
                                         single_link_clusters, stroke_yaw)
from sim.planners.visual_greedy import VisualGreedyPlanner


@pytest.fixture
def cfg():
    return load_config()


def make_observation(cfg, points, tcp=(0.44, -0.34, 0.26)):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    grid = GridSpec.from_config(cfg, res=0.008)
    return SceneObservation(
        t=0.0, points=points, counts=np.ones(len(points)),
        areas=np.full(len(points), 8e-5),
        occupancy=grid.rasterize_disks(points, 0.008) if len(points) else grid.empty(),
        grid=grid, tcp=np.asarray(tcp, dtype=float), backend="test",
    )


# --------------------------------------------------------------- helpers
def test_stroke_yaw_points_the_pusher_face_along_the_stroke():
    assert stroke_yaw(np.array([-1.0, 0.0])) == pytest.approx(0.0)
    assert stroke_yaw(np.array([-1.0, -1.0])) == pytest.approx(np.pi / 4)
    assert stroke_yaw(np.array([-1.0, 1.0])) == pytest.approx(-np.pi / 4)
    assert abs(stroke_yaw(np.array([0.0, 0.0]))) < 1e-12


def test_point_segment_frames():
    along, lateral, length = point_segment_frames(
        [[0.5, 0.0], [0.0, 0.1], [-0.5, 0.0]], [1.0, 0.0], [-1.0, 0.0]
    )
    assert length == pytest.approx(2.0)
    assert np.allclose(along, [0.25, 0.5, 0.75])
    assert np.allclose(lateral, [0.0, -0.1, 0.0])


def test_single_link_clustering():
    pts = np.array([[0.0, 0.0], [0.01, 0.0], [0.3, 0.3]])
    labels = single_link_clusters(pts, 0.05)
    assert labels[0] == labels[1] != labels[2]
    assert single_link_clusters(np.zeros((0, 2)), 0.05).size == 0


def test_action_round_trip():
    stroke = SweepStroke(0.4, 0.1, -0.4, 0.05, 0.2)
    assert np.allclose(SweepStroke.from_action(stroke.to_action()).to_action(),
                       stroke.to_action())
    assert stroke.to_action().shape == (5,)


# --------------------------------------------------------- Planner A (fixed)
def test_fixed_planner_ignores_the_observation(cfg):
    planner = FixedCoverPlanner(cfg)
    a = planner.plan(make_observation(cfg, [[0.3, 0.2]]))
    b = planner.plan(make_observation(cfg, [[-0.05, -0.21], [0.35, 0.0]]))
    c = planner.plan(make_observation(cfg, []))
    assert a.to_action().tolist() == b.to_action().tolist() == c.to_action().tolist()


def test_fixed_planner_strokes_all_move_towards_the_target(cfg):
    for stroke in FixedCoverPlanner(cfg).strokes:
        assert stroke.x_end < stroke.x_start          # right to left, towards -X
        assert stroke.x_end == pytest.approx(float(cfg.planner.stroke_end_x))
        assert float(cfg.target.y_min) < stroke.y_end < float(cfg.target.y_max)


def test_fixed_planner_lanes_cover_the_spawn_region(cfg):
    planner = FixedCoverPlanner(cfg)
    half = pusher_width(cfg) / 2.0
    lanes = sorted(s.y_start for s in planner.strokes if s.meta["kind"] == "lane")
    assert lanes[0] <= float(cfg.components.spawn.y_min)
    assert lanes[-1] >= float(cfg.components.spawn.y_max)
    gaps = np.diff(lanes)
    assert gaps.max() <= 2 * half + 1e-9          # no uncovered strip between lanes


def test_fixed_planner_ends_with_a_consolidation_stroke(cfg):
    strokes = FixedCoverPlanner(cfg).strokes
    assert strokes[-1].meta["kind"] == "consolidation"
    assert all(s.meta["kind"] == "lane" for s in strokes[:-1])


def test_fixed_planner_is_exhausted_after_its_stroke_list(cfg):
    cfg.set_path("planner.max_strokes", 100)
    planner = FixedCoverPlanner(cfg)
    for _ in range(planner.n_planned_strokes):
        assert planner.plan(make_observation(cfg, [])) is not None
        planner.notify_stroke_done(SweepStroke(0, 0, -0.4, 0), {})
    assert planner.plan(make_observation(cfg, [])) is None


def test_fixed_planner_respects_the_stroke_budget(cfg):
    cfg.set_path("planner.max_strokes", 3)
    planner = FixedCoverPlanner(cfg)
    for _ in range(3):
        assert planner.plan(make_observation(cfg, [])) is not None
        planner.notify_stroke_done(SweepStroke(0, 0, -0.4, 0), {})
    assert planner.plan(make_observation(cfg, [])) is None


# -------------------------------------------------- Planner B (visual greedy)
def test_greedy_returns_none_when_nothing_is_detected(cfg):
    assert VisualGreedyPlanner(cfg).plan(make_observation(cfg, [])) is None


def test_greedy_targets_the_detected_component(cfg):
    planner = VisualGreedyPlanner(cfg)
    stroke = planner.plan(make_observation(cfg, [[0.20, 0.13]]))
    assert stroke is not None
    assert stroke.y_start == pytest.approx(0.13, abs=pusher_width(cfg) / 2)
    assert stroke.x_start > 0.20                     # starts behind the component
    assert stroke.x_end == pytest.approx(float(cfg.planner.stroke_end_x))


def test_greedy_prefers_a_single_stroke_that_rakes_a_cluster(cfg):
    planner = VisualGreedyPlanner(cfg)
    cluster = [[0.20, 0.005], [0.24, -0.004], [0.28, 0.010]]
    stroke = planner.plan(make_observation(cfg, cluster))
    assert stroke.meta["n_captured"] == 1            # the three merge into one cluster
    assert stroke.meta["est_collected"] == pytest.approx(3.0)


def test_greedy_captures_multiple_separate_components_in_one_lane(cfg):
    planner = VisualGreedyPlanner(cfg)
    spread = [[0.10, 0.0], [0.25, 0.012], [0.35, -0.012]]
    stroke = planner.plan(make_observation(cfg, spread))
    assert stroke.meta["est_collected"] >= 2.0


def test_greedy_never_starts_inside_the_force_release_band(cfg):
    planner = VisualGreedyPlanner(cfg)
    guard = float(cfg.controller.x_release_line) + float(cfg.controller.release_margin)
    for point in ([[-0.10, 0.0]], [[-0.05, 0.2]], [[0.39, -0.23]]):
        stroke = planner.plan(make_observation(cfg, point))
        assert stroke.x_start > guard


def test_greedy_shortens_the_stroke_for_a_near_component(cfg):
    planner = VisualGreedyPlanner(cfg)
    near = planner.plan(make_observation(cfg, [[0.05, 0.0]]))
    far = planner.plan(make_observation(cfg, [[0.38, 0.0]]))
    assert near.length < far.length


def test_greedy_replans_when_the_scene_changes(cfg):
    planner = VisualGreedyPlanner(cfg)
    first = planner.plan(make_observation(cfg, [[0.30, 0.20], [0.30, -0.20]]))
    planner.notify_stroke_done(first, {"collected_delta": 1})
    second = planner.plan(make_observation(cfg, [[0.30, -0.20]]))
    assert abs(second.y_start - (-0.20)) < pusher_width(cfg)
    assert second.to_action().tolist() != first.to_action().tolist()


def test_greedy_gives_up_after_repeated_no_progress(cfg):
    planner = VisualGreedyPlanner(cfg)
    stroke = planner.plan(make_observation(cfg, [[0.3, 0.0]]))
    for _ in range(int(cfg.planner.greedy.max_no_progress)):
        planner.notify_stroke_done(stroke, {"collected_delta": 0})
    assert planner.is_exhausted()
    assert planner.plan(make_observation(cfg, [[0.3, 0.0]])) is None


def test_greedy_end_points_stay_inside_the_tray(cfg):
    planner = VisualGreedyPlanner(cfg)
    for y in np.linspace(float(cfg.workspace.y_min), float(cfg.workspace.y_max), 15):
        stroke = planner.plan(make_observation(cfg, [[0.3, float(y)]]))
        assert float(cfg.target.y_min) < stroke.y_end < float(cfg.target.y_max)


def test_build_planner_dispatch(cfg):
    assert isinstance(build_planner(cfg, "fixed"), FixedCoverPlanner)
    assert isinstance(build_planner(cfg, "visual_greedy"), VisualGreedyPlanner)
    with pytest.raises(ValueError):
        build_planner(cfg, "nope")
