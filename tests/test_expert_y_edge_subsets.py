"""Partial demonstrations must use one contiguous extreme of initial Y ranks."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

import sim.act.expert as expert
import sim.act.collect_dataset as collect
import sim.act.rollout as rollout
from sim.config import Config, load_config


@pytest.mark.parametrize('count,want', [
    (1, ((1,), (2,))),
    (2, ((1, 3), (2, 4))),
    (3, ((1, 3, 5), (0, 2, 4))),
    (4, ((0, 1, 3, 5), (0, 2, 4, 5))),
    (5, ((0, 1, 3, 4, 5), (0, 2, 3, 4, 5))),
    (6, ((0, 1, 2, 3, 4, 5),)),
])
def test_y_edge_subsets_are_only_lowest_or_highest_initial_y(count, want):
    # Hand-ranked Y order is 1,3,5,0,4,2; allowing middle/skipped ranks breaks this.
    positions = np.array([[.15, .03], [.15, -.12], [.15, .19],
                          [.15, -.07], [.15, .11], [.15, -.02]])
    assert expert.expert_target_subsets(positions, count, 'y_edge_contiguous') == want


def test_y_edge_ties_use_stable_component_identity():
    positions = np.array([[.1, -.1], [.2, -.1], [.1, 0],
                          [.2, 0], [.1, .1], [.2, .1]])
    assert expert.expert_target_subsets(positions, 2, 'y_edge_contiguous') == ((0, 1), (4, 5))


def test_legacy_subset_mode_keeps_all_combinations():
    subsets = expert.expert_target_subsets(np.zeros((6, 2)), 3, 'any')
    assert len(subsets) == 20 and subsets[0] == (0, 1, 2) and subsets[-1] == (3, 4, 5)


def test_unknown_subset_mode_is_rejected_not_silently_ignored():
    with pytest.raises(ValueError, match='subset'):
        expert.expert_target_subsets(np.zeros((6, 2)), 4, 'typo')


@pytest.mark.parametrize('selected,actual', [
    ([1, 2, 3, 4], [0, 1, 2, 3]),
    ([0, 1, 2, 3], [1, 2, 3, 4]),
])
def test_demo_audit_rejects_middle_planned_or_physically_collected_group(selected, actual):
    cfg = load_config()
    cfg.set_path('planner.expert_target_subset_mode', 'y_edge_contiguous')
    positions = np.array([[.2, -.175, 0], [.2, -.105, 0], [.2, -.035, 0],
                          [.2, .035, 0], [.2, .105, 0], [.2, .175, 0]])
    result = SimpleNamespace(initial_object_positions=positions, target_indices=selected,
                             final_collected_mask=np.isin(np.arange(6), actual))
    with pytest.raises(RuntimeError, match='Y-edge'):
        collect._expert_selection_audit(result, cfg, 4)


def test_demo_audit_retains_initial_ranks_without_exposing_policy_target_ids():
    cfg = load_config()
    cfg.set_path('planner.expert_target_subset_mode', 'y_edge_contiguous')
    positions = np.array([[.2, -.175, 0], [.2, -.105, 0], [.2, -.035, 0],
                          [.2, .035, 0], [.2, .105, 0], [.2, .175, 0]])
    result = SimpleNamespace(initial_object_positions=positions, target_indices=[3, 2, 1, 0],
                             final_collected_mask=np.array([1, 1, 1, 1, 0, 0], dtype=bool))
    audit = collect._expert_selection_audit(result, cfg, 4)
    assert audit['mode'] == 'y_edge_contiguous'
    assert audit['y_order'] == [0, 1, 2, 3, 4, 5]
    assert audit['planned_component_ids'] == [0, 1, 2, 3]
    assert audit['collected_component_ids'] == [0, 1, 2, 3]
    assert 'target_indices' not in audit


def test_legacy_act_mode_location_cannot_bypass_the_physical_selection_audit():
    cfg = Config({'act': {'expert_target_subset_mode': 'y_edge_contiguous'}})
    positions = np.array([[.2, -.175, 0], [.2, -.105, 0], [.2, -.035, 0],
                          [.2, .035, 0], [.2, .105, 0], [.2, .175, 0]])
    result = SimpleNamespace(initial_object_positions=positions, target_indices=[0, 1, 2, 3],
                             final_collected_mask=np.array([0, 1, 1, 1, 1, 0], dtype=bool))
    with pytest.raises(RuntimeError, match='Y-edge'):
        collect._expert_selection_audit(result, cfg, 4)


@pytest.mark.parametrize('selection', [None, {
    'mode': 'y_edge_contiguous',
    'initial_positions': [[.2, -.175], [.2, -.105], [.2, -.035], [.2, .035], [.2, .105], [.2, .175]],
    'y_order': [0, 1, 2, 3, 4, 5],
    'planned_component_ids': [0, 1, 2, 3],
    'collected_component_ids': [1, 2, 3, 4],
}])
def test_resuming_y_edge_data_rejects_missing_or_forbidden_selection_provenance(selection):
    record = {'episode_id': 'old_expert', 'target_count': 4}
    if selection is not None:
        record['target_selection'] = selection
    with pytest.raises(ValueError, match='selection'):
        collect._validate_existing_target_selection([record], 'y_edge_contiguous')


def test_expert_continues_same_layout_after_count_correct_middle_capture(monkeypatch):
    cfg = load_config(overrides=['task.target_count=3', 'planner.expert_validation_workers=1'])
    positions = np.array([[.2, -.175, 0], [.2, -.105, 0], [.2, -.035, 0],
                          [.2, .035, 0], [.2, .105, 0], [.2, .175, 0]])
    plans = [expert.ExpertPlan(target_indices=np.array([0, 1, 2]),
                              waypoints=np.array([[x, 0], [-.36, 0]]), feasible=True)
             for x in (.25, .30)]

    class Probe:
        layout = [{}] * 6
        table_top_z = 0.
        def __init__(self, *_a, **_k): pass
        def reset(self, **_k): pass
        def close(self): pass

    executed = []
    def execute(_env, _cfg, path, *, target_indices, **kwargs):
        executed.append(float(path[0, 0]))
        mask = ([0, 1, 1, 1, 0, 0] if path[0, 0] == .25 else [1, 1, 1, 0, 0, 0])
        return rollout.ActRolloutResult(
            success=True, failure_reason='', total=6, target_count=3, collected=3,
            target_indices=target_indices.tolist(), initial_object_positions=positions,
            final_collected_mask=np.asarray(mask, dtype=bool))

    # Keep real candidate acceptance; replace only simulator/execution costs.
    monkeypatch.setattr(rollout, 'SweepEnv', Probe)
    monkeypatch.setattr(rollout, 'expert_plan_candidates', lambda *_: plans)
    monkeypatch.setattr(rollout, '_expert_execution_path', lambda _cfg, plan, *_a, **_k: (plan.waypoints, []))
    monkeypatch.setattr(rollout, 'run_action_path', execute)
    result = rollout.run_expert_episode(cfg, seed=3200000, collect_observations=False)
    assert result.success
    assert executed == [.25, .30]
    assert result.final_collected_mask.tolist() == [True, True, True, False, False, False]


def test_six_part_demo_rule_does_not_reject_legacy_single_part_physics_probe(monkeypatch):
    cfg = load_config(overrides=['components.count=1', 'task.target_count=1',
                                'planner.expert_validation_workers=1'])
    plan = expert.ExpertPlan(target_indices=np.array([0]),
                            waypoints=np.array([[.25, 0], [-.36, 0]]), feasible=True)

    class Probe:
        layout = [{}]
        table_top_z = 0.
        def __init__(self, *_a, **_k): pass
        def reset(self, **_k): pass
        def close(self): pass

    def execute(_env, _cfg, _path, **kwargs):
        return rollout.ActRolloutResult(
            success=True, failure_reason='', total=1, target_count=1, collected=1,
            target_indices=[0], initial_object_positions=np.array([[.2, 0, .02]]),
            final_collected_mask=np.array([True]))

    monkeypatch.setattr(rollout, 'SweepEnv', Probe)
    monkeypatch.setattr(rollout, 'expert_plan_candidates', lambda *_: [plan])
    monkeypatch.setattr(rollout, '_expert_execution_path', lambda *_a, **_k: (plan.waypoints, []))
    monkeypatch.setattr(rollout, 'run_action_path', execute)
    result = rollout.run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success


def test_initial_pose_provenance_retains_sub_float32_y_rank_differences(monkeypatch):
    from sim.environments.sweep_env import SweepEnv
    cfg = load_config(overrides=['sim.real_time=false'])
    env = SweepEnv(cfg, seed=3200000)
    env.reset(seed=3200000)
    positions = np.array([[.2, .100000001, .02], [.2, .100000000, .02],
                          [.2, -.17, .02], [.2, -.10, .02], [.2, -.03, .02], [.2, .17, .02]])
    monkeypatch.setattr(env, 'component_positions', lambda: positions.copy())
    try:
        result = rollout.run_action_path(env, cfg, np.empty((0, 4)), collect_observations=False)
        assert result.initial_object_positions[0, 1] > result.initial_object_positions[1, 1]
        np.testing.assert_array_equal(result.initial_object_positions, positions)
    finally:
        env.close()


def test_public_shortlist_excludes_middle_group_from_both_search_paths(monkeypatch):
    cfg = load_config()
    # Explicit historical two-search profile; the new default is A*-only.
    cfg.set_path('planner.expert_planning_mode','mixed')
    cfg.set_path('task.expert_landing_x_min',None)
    cfg.set_path('task.expert_landing_clearance',None)
    cfg.set_path('task.target_count', 4)
    cfg.set_path('planner.expert_target_subset_mode', 'y_edge_contiguous')
    positions = np.array([[.18, -.175], [.15, -.105], [.20, -.035],
                          [.16, .035], [.19, .105], [.14, .175]])
    env = SimpleNamespace(component_positions=lambda: positions,
                          layout=[{'nominal_radius': .02} for _ in range(6)])
    path = np.array([[.30, 0], [-.36, 0]])
    # Replace expensive route geometry below the real subset-selection boundary.
    monkeypatch.setattr(expert, '_capture_lane_candidates', lambda *_: [(0., path)])
    monkeypatch.setattr(expert, '_coverage_astar_candidates', lambda *_: [
        (-100., path, (1, 2, 3, 4), 0), (0., path, (0, 1, 2, 3), 0)])
    plans = expert.expert_plan_candidates(env, cfg, max_candidates=100)
    assert plans
    assert {tuple(sorted(p.target_indices)) for p in plans} == {
        (0, 1, 2, 3), (2, 3, 4, 5)}
    assert all(tuple(sorted(p.target_indices)) != (1, 2, 3, 4) for p in plans)


def test_real_coverage_astar_search_avoids_a_cheaper_middle_four_group():
    # A real search (no route mocks): the old branch selects the interior four.
    cfg = load_config()
    cfg.set_path('task.expert_landing_clearance',None)
    cfg.set_path('task.expert_landing_x_min',.36)
    cfg.set_path('planner.expert_curve_max_expansions', 30000)
    cfg.set_path('planner.expert_curve_turn_penalties', [.025])
    positions = np.array([[.10, -.23], [.18, -.09], [.17, -.03],
                          [.15, .03], [.16, .09], [.10, .23]])
    env = SimpleNamespace(component_positions=lambda: positions,
                          layout=[{'nominal_radius': .02} for _ in range(6)])
    cfg.set_path('planner.expert_target_subset_mode', 'any')
    legacy = expert._coverage_astar_candidates(env, cfg, 4)
    assert legacy and {tuple(sorted(item[2])) for item in legacy} == {(1, 2, 3, 4)}
    cfg.set_path('planner.expert_target_subset_mode', 'y_edge_contiguous')
    current = expert._coverage_astar_candidates(env, cfg, 4)
    assert current
    assert {tuple(sorted(item[2])) for item in current} == {(0, 1, 2, 3), (2, 3, 4, 5)}


def test_rounding_keeps_a_legal_extreme_instead_of_an_equal_count_middle_part():
    cfg = load_config(overrides=[
        'end_effector.brush_width=0.14', 'end_effector.brush_depth=0.008',
        'planner.expert_curve_capture_margin=0.055',
        'planner.expert_capture_forward_cos=0.0',
        'planner.expert_target_subset_mode=y_edge_contiguous',
    ])
    positions = np.array([
        [-.035, .055], [.250, -.080], [.320, -.150], [.200, .060],
        [.08929895223794562, -.034321591418987774],
        [.11107222467504319, .12149926789246054],
    ])
    path = np.array([[.30, 0], [.08, 0], [.08, .10], [-.26, .10]])
    env = SimpleNamespace(component_positions=lambda: positions,
                          layout=[{'nominal_radius': .018} for _ in range(6)])
    # Literal identities: lowest Y is 2, highest Y is 5, middle part is 4.
    assert expert._path_capture_mask(path, env, cfg) == 32
    assert expert._path_capture_mask(expert._rounded_polyline(path, .025), env, cfg) == 16
    result = expert._capture_preserving_rounded_path(path, .025, env, cfg, 1)
    assert result is not None
    assert result[1] == 32


def test_unusable_rounded_routes_do_not_consume_the_search_goal_quota(monkeypatch):
    cfg = load_config(overrides=[
        'planner.expert_curve_max_expansions=30000',
        'planner.expert_curve_turn_penalties=[0.025]',
        'planner.expert_curve_solutions_per_penalty=3',
        'planner.expert_target_subset_mode=y_edge_contiguous',
        'task.expert_landing_clearance=null', 'task.expert_landing_x_min=.36',
    ])
    positions = np.array([[.10, -.23], [.18, -.09], [.17, -.03],
                          [.15, .03], [.16, .09], [.10, .23]])
    env = SimpleNamespace(component_positions=lambda: positions,
                          layout=[{'nominal_radius': .02} for _ in range(6)])
    original_round = expert._capture_preserving_rounded_path
    seen_exits = set()

    def first_route_unusable(path, *args):
        # Inject one rejected geometry per goal; keep subsequent geometry real.
        exit_key = tuple(np.round(path[-1], 6))
        if exit_key not in seen_exits:
            seen_exits.add(exit_key)
            return None
        return original_round(path, *args)

    monkeypatch.setattr(expert, '_capture_preserving_rounded_path', first_route_unusable)
    candidates = expert._coverage_astar_candidates(env, cfg, 4)
    assert candidates, 'an unusable first route must not exhaust a delivery goal'
    assert all(tuple(sorted(item[2])) in ((0, 1, 2, 3), (2, 3, 4, 5))
               for item in candidates)


def test_parallel_generation_finishes_each_target_before_starting_next(tmp_path, monkeypatch):
    cfg = load_config()
    live_targets = set()
    completed_targets = []
    started = []
    lock = threading.Lock()

    def capture(_cfg, _writer, _root, spec, _requested_seed, _attempt,
                episode_id=None, **_kwargs):
        with lock:
            live_targets.add(spec.target_count)
            started.append((spec.target_count, len(live_targets), tuple(completed_targets)))
        time.sleep(.04)
        with lock:
            completed_targets.append(spec.target_count)
            if completed_targets.count(spec.target_count) == 2:
                live_targets.remove(spec.target_count)
        return {'episode_id': episode_id, 'seed': spec.seed, 'success': True}

    class Pool(ThreadPoolExecutor):
        def __init__(self, max_workers, mp_context=None):
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(collect, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(collect, '_capture_training_episode', capture)
    monkeypatch.setattr(collect, 'validate_training_manifest', lambda *a, **k: {'total': 12})
    monkeypatch.setattr(collect, 'write_split_manifests', lambda *_: {})
    result = collect.generate_approved_training_dataset(
        cfg, tmp_path, seed_base=3200000, episodes_per_target=2, workers=4)
    assert result['generated'] == 12
    assert [target for target, _, _ in started] == [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    assert all(live_count == 1 for _, live_count, _ in started)
    assert all(all(done.count(higher) == 2 for higher in range(target + 1, 7))
               for target, _, done in started)
