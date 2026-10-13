"""A* boundary-intersection landings, distinct from frozen minimum-only profiles."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from sim.config import Config, load_config
from sim.act import expert
from sim.act.rollout import _expert_execution_path
from sim.act.collect_dataset import _validate_existing_generation_protocol


def intersection_configuration():
    cfg = load_config()
    cfg.set_path('task.expert_landing_clearance', .07)
    cfg.set_path('task.expert_landing_selection', 'path_intersection')
    env = SimpleNamespace(component_collision_x_max=lambda: .11)
    resolved, region = expert.expert_landing_configuration(env, cfg)
    return cfg, resolved, region


def test_landing_skips_far_right_waypoints_and_uses_first_actual_crossing():
    # Retaining the former second waypoint would descend 17.6 cm too far right.
    _, cfg, _ = intersection_configuration()
    original = np.array([[.42, 0], [.32, .10], [.20, .10],
                         [.05, -.05], [-.30, 0]])
    saved = original.copy()
    route = expert._landing_boundary_path(original, cfg)
    np.testing.assert_allclose(route, [[.42, 0], [.184, .084],
                                      [.05, -.05], [-.30, 0]], atol=1e-12)
    np.testing.assert_array_equal(original, saved)


def test_boundary_vertex_is_not_duplicated_and_later_recrossings_are_retained():
    _, cfg, _ = intersection_configuration()
    original = np.array([[.42, 0], [.32, .10], [.184, -.03],
                         [.10, -.02], [.22, .08], [-.30, 0]])
    route = expert._landing_boundary_path(original, cfg)
    np.testing.assert_allclose(route, [[.42, 0], [.184, -.03],
                                      [.10, -.02], [.22, .08], [-.30, 0]], atol=1e-12)


@pytest.mark.parametrize('path', [
    [[.42, 0], [.32, .1], [.22, 0]],
    [[.10, 0], [.32, .1], [.10, 0]],
])
def test_missing_outside_to_inside_crossing_cannot_create_a_landing(path):
    _, cfg, _ = intersection_configuration()
    assert expert._landing_boundary_path(np.asarray(path), cfg) is None


def test_execution_descends_at_intersection_then_keeps_the_astar_suffix():
    original_cfg, cfg, region = intersection_configuration()
    route = expert._landing_boundary_path(np.array(
        [[.42, 0], [.32, .1], [.20, .1], [.05, -.05], [-.30, 0]]), cfg)
    plan = expert.ExpertPlan(waypoints=route, landing_region=region,
                            strategy='coverage_astar')
    path, phases = _expert_execution_path(original_cfg, plan)
    descent = path[np.asarray(phases) == 'descent']
    np.testing.assert_allclose(descent[:, :2],
                               np.tile([.184, .084], (len(descent), 1)), atol=1e-7)
    assert path[np.asarray(phases) == 'sweep', 0].min() < -.25


def test_external_plan_cannot_keep_an_arbitrary_far_right_landing():
    cfg, _, region = intersection_configuration()
    plan = expert.ExpertPlan(waypoints=np.array([[.42, 0], [.32, .10], [-.30, 0]]),
                            landing_region=region, strategy='coverage_astar')
    with pytest.raises(ValueError, match='intersection'):
        _expert_execution_path(cfg, plan)


def test_new_protocol_cannot_resume_or_relabel_old_minimum_only_successes():
    cfg, _, _ = intersection_configuration()
    new = expert.expert_generation_protocol(cfg)
    previous_cfg = cfg.copy()
    previous_cfg.set_path('task.expert_landing_selection', 'waypoint_minimum')
    old = expert.expert_generation_protocol(previous_cfg)
    assert new != old  # Same gap alone does not identify the changed selection rule.
    assert new['landing_selection'] == 'first_path_boundary_intersection'
    record = {'episode_id': 'old', 'episode_kind': 'expert', 'split': 'train',
              'success': True, 'expert_generation_protocol': old}
    before = copy.deepcopy(record)
    with pytest.raises(ValueError, match='generation protocol'):
        _validate_existing_generation_protocol([record], [new])
    assert record == before


def test_missing_selection_in_frozen_config_preserves_minimum_only_waypoint():
    raw = load_config().to_dict()
    raw['task']['expert_landing_clearance'] = .04
    raw['task'].pop('expert_landing_selection', None)
    cfg = Config(raw)
    resolved, _ = expert.expert_landing_configuration(
        SimpleNamespace(component_collision_x_max=lambda: .11), cfg)
    original = np.array([[.42, 0], [.32, .1], [.10, 0], [-.30, 0]])
    np.testing.assert_array_equal(expert._landing_boundary_path(original, resolved), original)
    assert 'landing_selection' not in expert.expert_generation_protocol(cfg)


def test_new_default_contact_behavior_uses_seven_centimetre_intersection():
    cfg = load_config()
    resolved, _ = expert.expert_landing_configuration(
        SimpleNamespace(component_collision_x_max=lambda: .11), cfg)
    route = expert._landing_boundary_path(np.array(
        [[.42, 0], [.32, .1], [.20, .1], [.05, -.05], [-.30, 0]]), resolved)
    np.testing.assert_allclose(route[1], [.184, .084], atol=1e-12)


def test_invalid_selection_is_rejected_before_planning():
    cfg = load_config()
    cfg.set_path('task.expert_landing_selection', 'unknown')
    with pytest.raises(ValueError, match='selection'):
        expert.expert_landing_configuration(
            SimpleNamespace(component_collision_x_max=lambda: .11), cfg)


def test_infeasible_attempt_probe_uses_valid_intersection_landing_without_a_sweep():
    _, cfg, _ = intersection_configuration()
    route = expert._safe_contact_failure_path(cfg)
    np.testing.assert_allclose(route[1], [.184, 0], atol=1e-12)
    expert.expert_validate_landing_pose(cfg, route[1], 0.0)


def test_invalid_candidate_intersection_is_rejected_without_aborting_alternatives():
    _, cfg, _ = intersection_configuration()
    cfg.set_path('table.half_size', [.5, .35, .025])
    assert expert._landing_boundary_path(np.array([[.42,.30],[.10,.30]]), cfg) is None
    assert expert._landing_boundary_path(np.array([[.42,0],[.10,0]]), cfg) is not None
    cfg.set_path('task.expert_landing_x_min', 99.)
    with pytest.raises(ValueError, match='workspace'):
        expert._landing_boundary_path(np.array([[.42,0],[.10,0]]), cfg)


def test_boundary_origin_is_the_first_intersection_even_if_path_first_moves_right():
    _, cfg, _ = intersection_configuration()
    original = np.array([[.184,.02],[.30,.1],[.10,.1]])
    route = expert._landing_boundary_path(original, cfg)
    np.testing.assert_allclose(route, [[.184,.02],[.184,.02],[.30,.1],[.10,.1]], atol=1e-12)


def test_brush_aligns_at_intersection_before_xy_moves_into_first_astar_leg():
    original_cfg, cfg, region = intersection_configuration()
    route = expert._landing_boundary_path(np.array(
        [[.42,0],[.30,.1],[.11,-.09],[-.30,0]]), cfg)
    plan = expert.ExpertPlan(waypoints=route, landing_region=region,
                            strategy='coverage_astar')
    path, phases = _expert_execution_path(original_cfg, plan)
    sweep = path[np.asarray(phases) == 'sweep']
    first_moving = np.flatnonzero(np.linalg.norm(np.diff(sweep[:,:2],axis=0),axis=1)>1e-7)[0]+1
    # First southwest segment needs +45 degrees modulo brush symmetry.
    assert sweep[first_moving,3] == pytest.approx(np.pi/4, abs=1e-4)
    assert first_moving >= 25
    np.testing.assert_allclose(sweep[:first_moving,:2],
        np.tile([.184,-.016],(first_moving,1)),atol=1e-7)
    assert np.max(np.abs(np.diff(sweep[:,3]))) <= float(cfg.controller.yaw_speed_limit)/25+1e-6


@pytest.mark.parametrize('half_x', [.43, .47005])
def test_alignment_rejects_full_swept_brush_footprint_not_only_endpoint_yaws(half_x):
    cfg = load_config()
    cfg.set_path('table.half_size', [half_x,.4,.025])
    resolved, region = expert.expert_landing_configuration(
        SimpleNamespace(component_collision_x_max=lambda:.326), cfg)
    original = np.array([[.42,0],[.40,0],[.40,.10],[-.30,.10]])
    # Both the landing and (for .47005) final yaw fit, but intermediate yaw
    # reaches X=.47011419257183. A bad route must not abort alternatives.
    assert expert._landing_boundary_path(original, resolved) is None
    horizontal = np.array([[.42,0],[.40,0],[.10,0],[-.30,0]])
    assert expert._landing_boundary_path(horizontal, resolved) is not None
    plan = expert.ExpertPlan(waypoints=original, landing_region=region,
                            strategy='coverage_astar')
    with pytest.raises(ValueError, match='rotation.*table'):
        _expert_execution_path(cfg, plan)


def test_rotation_guard_uses_sampler_entry_yaw_after_a_tiny_intersection_fragment():
    cfg = load_config()
    cfg.set_path('table.half_size', [.43,.4,.025])
    resolved, region = expert.expert_landing_configuration(
        SimpleNamespace(component_collision_x_max=lambda:.326), cfg)
    original = np.array([[.42,0],[.40,0],[.39999,0],[.39999,-.10],[-.30,-.10]])
    assert expert._landing_boundary_path(original, resolved) is None
    plan = expert.ExpertPlan(waypoints=original, landing_region=region,
                            strategy='coverage_astar')
    with pytest.raises(ValueError, match='rotation.*table'):
        _expert_execution_path(cfg, plan)
