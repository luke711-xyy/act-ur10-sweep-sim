"""Layout-specific landing clearance, derived independently from physical shapes."""
import numpy as np
import pytest
import mujoco

from sim.config import load_config
from sim.environments.sweep_env import SweepEnv
from sim.act import expert
from sim.act.rollout import _expert_execution_path, _annotate_expert_result, ActRolloutResult


def scene(geoms):
    cfg = load_config()
    cfg.set_path('task.expert_landing_clearance', .04)
    cfg.set_path('task.expert_landing_selection', 'waypoint_minimum')
    env = SweepEnv(cfg)
    bodies = ''.join(f'<body name="comp_{i}" pos="{x} {y} 0">{xml}</body>'
                     for i,(x,y,xml) in enumerate(geoms))
    env.model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{bodies}</worldbody></mujoco>')
    env.data = mujoco.MjData(env.model)
    env.component_body_ids = list(range(1,len(geoms)+1))
    env.component_geom_ids = [[g for g in range(env.model.ngeom)
                              if env.model.geom_bodyid[g] == b] for b in env.component_body_ids]
    env.layout = [{'nominal_radius':.006} for _ in geoms]
    mujoco.mj_forward(env.model,env.data)
    return env,cfg


@pytest.mark.parametrize('xml,want',[
    ('<geom type="sphere" size=".01"/>',.11),
    ('<geom type="box" size=".03 .02 .01" euler="0 0 90"/>',.12),
    ('<geom type="cylinder" size=".02 .03"/>',.12),
    ('<geom type="cylinder" size=".01 .03" euler="0 90 0"/>',.13),
    ('<geom type="capsule" size=".01 .03" euler="0 90 0"/>',.14),
])
def test_surface_uses_current_transformed_collision_geometry_not_body_center(xml,want):
    # Ignoring radius, yaw or half-length fails one of the literal support values.
    env,_ = scene([(.1,0,xml)])
    assert env.component_collision_x_max() == pytest.approx(want,abs=1e-12)


def test_surface_includes_compound_head_and_all_parts_but_ignores_visual_geoms():
    env,_ = scene([(.1,0,'<geom type="sphere" size=".01"/><geom type="box" pos=".03 0 0" size=".02 .01 .01"/>'),
                   (.12,.2,'<geom type="sphere" size=".01"/><geom type="sphere" pos=".7 0 0" size=".1" contype="0" conaffinity="0"/>')])
    assert env.component_collision_x_max() == pytest.approx(.15)


@pytest.mark.parametrize('x,want',[(.10,.154),(.20,.254),(-.10,-.046)])
def test_dynamic_boundary_moves_with_this_layout_and_does_not_keep_old_floor(x,want):
    env,cfg = scene([(x,0,'<geom type="sphere" size=".01"/>')])
    original = cfg.to_dict()
    resolved,region = expert.expert_landing_configuration(env,cfg)
    assert expert.expert_landing_x_min(resolved) == pytest.approx(want)
    assert region['part_x_max'] == pytest.approx(x+.01)
    assert region['clearance'] == .04
    assert cfg.to_dict() == original  # resolver never mutates input


def test_boundary_insertion_keeps_planned_y_and_freezes_only_landing_not_sweep():
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    resolved,region = expert.expert_landing_configuration(env,cfg)
    old = np.array([[.42,0],[.10,-.12],[-.30,0]])
    route = expert._landing_boundary_path(old,resolved)
    assert route[1,0] == pytest.approx(.154)
    assert route[1,1] == pytest.approx(-.09975)
    np.testing.assert_array_equal(route[2:],old[1:])
    plan = expert.ExpertPlan(waypoints=route,landing_region=region,strategy='coverage_astar')
    # Moving the parts later must not move the stored episode landing boundary.
    env.data.xpos[1,0] = .4
    path,phases = _expert_execution_path(cfg,plan)
    descent = path[np.isin(phases,['descent','contact_build'])]
    assert descent[:,0].min() == pytest.approx(.154,abs=1e-7)
    assert path[np.asarray(phases)=='sweep',0].min() < -.2
    branched = expert._add_symmetric_yaw_branches([plan],fixed_yaw=0)
    assert branched[0].landing_region == region


def test_nested_planning_reuses_resolved_initial_boundary_after_parts_move():
    # A nested resolver must not turn the initial-layout line into a moving one.
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    resolved,initial = expert.expert_landing_configuration(env,cfg)
    env.data.geom_xpos[0,0] = .3
    nested,region = expert.expert_landing_configuration(env,resolved)
    assert region == initial
    assert expert.expert_landing_x_min(nested) == pytest.approx(.154)


def test_execution_rejects_missing_or_inconsistent_dynamic_boundary():
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    _,region = expert.expert_landing_configuration(env,cfg)
    route = np.array([[.42,0],[.16,0],[-.30,0]])
    with pytest.raises(ValueError,match='landing'):
        _expert_execution_path(cfg,expert.ExpertPlan(waypoints=route))
    bad = dict(region,x_min=.13)
    with pytest.raises(ValueError,match='landing'):
        _expert_execution_path(cfg,expert.ExpertPlan(waypoints=route,landing_region=bad))
    with pytest.raises(ValueError,match='landing'):
        _expert_execution_path(cfg,expert.ExpertPlan(waypoints=np.array([[.42,0],[.15,0],[-.3,0]]),landing_region=region))


@pytest.mark.parametrize('gap',[-.01,float('nan'),float('inf')])
def test_invalid_clearance_is_rejected_without_silent_reduction(gap):
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    cfg.set_path('task.expert_landing_clearance',gap)
    with pytest.raises(ValueError,match='landing'):
        expert.expert_landing_configuration(env,cfg)


def test_clearance_outside_workspace_is_reported_not_clamped():
    env,cfg = scene([(.43,0,'<geom type="sphere" size=".01"/>')])
    with pytest.raises(ValueError,match='landing'):
        expert.expert_landing_configuration(env,cfg)


def test_episode_audit_and_new_protocol_separate_dynamic_from_fixed_data():
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    _,region = expert.expert_landing_configuration(env,cfg)
    p = expert.ExpertPlan(waypoints=np.array([[.42,0],[.154,0],[-.30,0]]),landing_region=region)
    result = _annotate_expert_result(ActRolloutResult(success=False,failure_reason=''),p,1)
    assert result.expert_landing_region == region
    dynamic = expert.expert_generation_protocol(cfg)
    fixed_cfg = cfg.copy()
    fixed_cfg.set_path('task.expert_landing_clearance',None)
    fixed_cfg.set_path('task.expert_landing_x_min',.36)
    assert dynamic != expert.expert_generation_protocol(fixed_cfg)
    assert dynamic['landing_clearance'] == .04


def test_frozen_fixed_boundary_paths_keep_existing_behavior():
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    cfg.set_path('task.expert_landing_clearance',None)
    cfg.set_path('task.expert_landing_x_min',.36)
    resolved,region = expert.expert_landing_configuration(env,cfg)
    assert resolved is cfg and region is None
    path = expert._landing_boundary_path(np.array([[.42,0],[.1,.1],[-.3,0]]),cfg)
    assert path[1,0] == pytest.approx(.36)


@pytest.mark.parametrize('target',range(1,7))
def test_real_coverage_plan_carries_dynamic_region_for_all_parts_not_target_subset(target):
    geoms = [(.12,y,'<geom type="sphere" size=".006"/>')
             for y in [-.12,-.07,-.02,.03,.08,.13]]
    env,cfg = scene(geoms)
    cfg.set_path('task.target_count',target)
    cfg.set_path('planner.expert_curve_max_expansions',480000 if target == 6 else 25000)
    cfg.set_path('planner.expert_curve_turn_penalties',[.025])
    cfg.set_path('planner.expert_curve_solutions_per_penalty',2)
    plans = expert.expert_plan_candidates(env,cfg,max_candidates=2)
    assert plans
    for p in plans:
        assert p.strategy == 'coverage_astar'
        assert p.landing_region['x_min'] == pytest.approx(.17)
        assert p.waypoints[1,0] >= .17-1e-9
        _expert_execution_path(cfg,p)


def test_fixed_protocol_records_cannot_be_counted_or_rewritten_by_new_dynamic_collection(tmp_path):
    import json
    from sim.act.collect_dataset import generate_approved_training_dataset
    cfg = load_config()
    frozen = cfg.copy()
    frozen.set_path('task.expert_landing_clearance',None)
    frozen.set_path('task.expert_landing_x_min',.36)
    record = {'episode_id':'fixed-rule-evidence',
              'schema_version':5,'state_dim':36,'episode_kind':'expert',
              'split':'train','layout_kind':'independent','layout_fingerprint':'fixed-layout',
              'expert_generation_protocol':expert.expert_generation_protocol(frozen)}
    original = json.dumps(record)+'\n'
    (tmp_path/'manifest.jsonl').write_text(original)
    with pytest.raises(ValueError,match='generation protocol'):
        generate_approved_training_dataset(cfg,tmp_path,episodes_per_target=1)
    assert {p.name:p.read_text() for p in tmp_path.iterdir()} == {'manifest.jsonl':original}


def test_staging_validates_double_precision_boundary_before_float32_output():
    env,cfg = scene([(.12,0,'<geom type="sphere" size=".01"/>')])
    resolved,region = expert.expert_landing_configuration(env,cfg)
    route = expert._landing_boundary_path(np.array([[.42,0],[.1,.12],[-.3,0]]),resolved)
    plan = expert.ExpertPlan(waypoints=route,landing_region=region)
    np.testing.assert_allclose(expert.expert_staging_xy(env,cfg,plan),route[1],atol=1e-7)
    bad = route.copy(); bad[1,0] -= .001
    with pytest.raises(ValueError,match='landing'):
        expert.expert_staging_xy(env,cfg,expert.ExpertPlan(waypoints=bad,landing_region=region))


def test_dynamic_landing_explicitly_rejects_legacy_capture_lane_override():
    env,cfg = scene([(.1,0,'<geom type="sphere" size=".01"/>')])
    cfg.set_path('planner.expert_planning_mode','mixed')
    with pytest.raises(ValueError,match='coverage_astar_only'):
        expert.expert_plan_candidates(env,cfg)


def test_infeasible_probe_lands_safely_even_when_boundary_is_right_of_reset(monkeypatch):
    env,cfg = scene([(.38,0,'<geom type="sphere" size=".01"/>')])
    monkeypatch.setattr(expert,'_best_target_sweep_plan',lambda *_:None)
    p = expert.plan_expert_sweep(env,cfg)
    assert not p.feasible
    assert p.waypoints[1,0] >= .434-1e-9
    path,phases = _expert_execution_path(cfg,p)
    assert min(path[np.asarray(phases)=='descent',0]) >= .434-1e-7


def test_workbench_new_preview_rejects_legacy_root_before_physics_or_writes(tmp_path,monkeypatch):
    import json
    import sim.web.workbench as web
    root = tmp_path/'previews'; root.mkdir()
    old = json.dumps({'episode_id':'old','preview':True})+'\n'
    (root/'manifest.jsonl').write_text(old)
    state = web.WorkbenchState(load_config(),dataset_root=tmp_path/'data',preview_root=root)
    monkeypatch.setattr(web,'run_expert_episode',lambda *_a,**_k:pytest.fail('incompatible preview reached physics'))
    with pytest.raises(ValueError,match='generation protocol'):
        state.build_preview(1,target_count=6)
    assert {p.name:p.read_text() for p in root.iterdir()} == {'manifest.jsonl':old}


def test_workbench_wrong_count_source_cannot_clone_fixed_generation_rules(tmp_path):
    import json
    from sim.web.workbench import WorkbenchState
    root = tmp_path/'previews'; root.mkdir()
    old = json.dumps({'episode_id':'old','preview':True,'success':True,'target_count':3})+'\n'
    (root/'manifest.jsonl').write_text(old)
    state = WorkbenchState(load_config(),dataset_root=tmp_path/'data',preview_root=root)
    with pytest.raises(ValueError,match='generation protocol'):
        state._clone_success_as_wrong_count(1,2,'wrong_count_over')
    assert {p.name:p.read_text() for p in root.iterdir()} == {'manifest.jsonl':old}


@pytest.mark.parametrize('raw_count,partial_spawn,source_spawn',[
    (1,'cluster','cluster'),
    (6,'uniform','uniform'),
])
def test_workbench_wrong_count_clone_uses_normalized_preview_configuration(
        tmp_path,raw_count,partial_spawn,source_spawn):
    # Rechecking the unnormalized count/spawn mode rejects a compatible source.
    from sim.web.workbench import WorkbenchState
    from sim.act.dataset import ActDatasetWriter
    cfg = load_config()
    cfg.set_path('components.count',raw_count)
    cfg.set_path('components.spawn_mode','cluster')
    cfg.set_path('components.exact_partial_spawn_mode',partial_spawn)
    source_cfg = cfg.copy()
    source_cfg.set_path('components.count',6)
    source_cfg.set_path('components.spawn_mode',source_spawn)
    root = tmp_path/'previews'
    image = np.zeros((12,12,3),dtype=np.uint8)
    observation = {name:image for name in ['overhead','wrist','inspection']}
    observation.update(state=np.zeros(36,dtype=np.float32),
                       environment_state=np.array([6,3,0],dtype=np.float32),phase='sweep')
    source_id = 'preview_goal_3_0001'
    ActDatasetWriter(str(root)).add_episode(
        source_id,[observation],np.zeros((1,4),dtype=np.float32),True,
        {'preview':True,'seed':3,'target_count':3,'total_count':6,'collected':3,
         'fps':25.0,'episode_kind':'expert_preview','split':'pilot',
         'expert_generation_protocol':expert.expert_generation_protocol(source_cfg)})
    original_arrays = (root/source_id/'arrays.npz').read_bytes()
    state = WorkbenchState(cfg,dataset_root=tmp_path/'data',preview_root=root)
    result = state.build_preview(7,target_count=2,outcome='failed',failure_mode='wrong_count_over')
    assert result['derived_from'] == source_id
    assert result['target_count'] == 2
    assert result['collected'] == 3
    assert result['expert_generation_protocol']['components']['count'] == 6
    assert result['expert_generation_protocol']['components']['spawn_mode'] == source_spawn
    assert (root/source_id/'arrays.npz').read_bytes() == original_arrays
    with np.load(root/result['episode_id']/'arrays.npz') as arrays:
        assert arrays['environment_state'][0,1] == 2


def test_rightmost_middle_part_sets_boundary_even_when_never_an_eligible_one_part_target():
    geoms = [(x,y,'<geom type="sphere" size=".006"/>')
             for x,y in [(.11,-.12),(.11,-.07),(.20,-.02),(.11,.03),(.11,.08),(.11,.13)]]
    env,cfg = scene(geoms)
    cfg.set_path('task.target_count',1)
    cfg.set_path('planner.expert_curve_max_expansions',25000)
    cfg.set_path('planner.expert_curve_turn_penalties',[.025])
    cfg.set_path('planner.expert_curve_solutions_per_penalty',2)
    plans = expert.expert_plan_candidates(env,cfg,max_candidates=2)
    assert plans
    for p in plans:
        assert tuple(p.target_indices) in [(0,),(5,)]
        assert p.landing_region['x_min'] == pytest.approx(.25)
        assert p.waypoints[1,0] >= .25-1e-9
