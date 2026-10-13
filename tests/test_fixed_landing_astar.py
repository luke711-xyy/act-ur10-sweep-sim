"""Frozen 36cm landing profile: compatibility after the dynamic rule update."""
from types import SimpleNamespace
import numpy as np
import pytest

from sim.config import load_config as load_current_config
from sim.environments.layout import sample_layout
from sim.act.expert import ExpertPlan, expert_plan_candidates, expert_staging_xy
from sim.act.rollout import _expert_execution_path


def load_config():
    """Explicit preceding profile, not mutable future defaults."""
    return load_current_config(overrides=['task.expert_landing_clearance=null',
                                          'task.expert_landing_x_min=0.36',
                                          'episode.expert_height_guard_until_exact_count=false'])


def test_default_cloud_reaches_left_of_old_spawn_limit_without_changing_center_disk():
    # A retained -8cm rejection boundary prevents this consumer-visible result.
    cfg=load_config()
    layouts=[sample_layout(cfg,np.random.default_rng(seed)) for seed in range(300)]
    points=np.array([[p['x'],p['y']] for layout in layouts for p in layout])
    assert points[:,0].min() < -.14
    assert points[:,0].min() >= -.19-1e-12
    assert points[:,0].max() <= .29+1e-12
    assert np.max(np.linalg.norm(points-[.05,0],axis=1)) <= .24+1e-12


def test_default_offset_cap_is_fourteen_centimetres_for_every_part():
    # A 16cm cap or only changing one ellipse axis violates the new support.
    cfg=load_config()
    from sim.model.geometries import resolve_geometry_list
    from sim.environments.layout import _sample_cluster
    for seed in range(120):
        rng=np.random.default_rng(seed)
        resolve_geometry_list('mixed',6,rng)
        centre,_=_sample_cluster(cfg,rng)
        layout=sample_layout(cfg,np.random.default_rng(seed))
        assert max(np.linalg.norm([p['x']-centre[0],p['y']-centre[1]]) for p in layout) <= .14+1e-12


@pytest.mark.parametrize('target',range(1,7))
def test_every_default_target_uses_real_coverage_astar_and_landing_half_plane(target):
    # Lane candidates, a left-side landing, or locking every landing to Y=0 breaks it.
    cfg=load_config()
    cfg.set_path('task.target_count',target)
    cfg.set_path('planner.expert_curve_max_expansions',480000 if target==6 else 25000)
    cfg.set_path('planner.expert_curve_turn_penalties',[.025])
    cfg.set_path('planner.expert_curve_solutions_per_penalty',2)
    positions=np.array([[.12,-.12],[.12,-.07],[.12,-.02],
                        [.12,.03],[.12,.08],[.12,.13]])
    env=SimpleNamespace(component_positions=lambda:positions,
                        layout=[{'nominal_radius':.006} for _ in range(6)])
    plans=expert_plan_candidates(env,cfg,max_candidates=4)
    assert plans
    assert {p.strategy for p in plans} == {'coverage_astar'}
    for p in plans:
        np.testing.assert_allclose(p.waypoints[0],[.42,0],rtol=0,atol=1e-7)
        assert p.waypoints[1,0] >= .36-1e-8
        assert p.yaw_offset == pytest.approx(0)
        np.testing.assert_allclose(expert_staging_xy(env,cfg,p),p.waypoints[1],atol=1e-7)
        path,phases=_expert_execution_path(cfg,p)
        indices=np.isin(phases,['descent','contact_build'])
        np.testing.assert_allclose(path[indices,:2],np.tile(p.waypoints[1],(indices.sum(),1)),atol=1e-7)
        np.testing.assert_allclose(path[indices,3],0,atol=1e-7)
        assert path[np.asarray(phases)=='sweep',0].min() < .30
    if target==1:
        assert any(abs(p.waypoints[1,1])>.001 for p in plans)


def test_execution_rejects_a_left_side_landing_instead_of_silently_rebasing():
    cfg=load_config()
    p=ExpertPlan(target_indices=np.arange(6),feasible=True,strategy='coverage_astar',
                 waypoints=np.array([[.42,0],[.20,0],[-.30,0],[-.37,0]]))
    with pytest.raises(ValueError,match='landing'):
        _expert_execution_path(cfg,p)


@pytest.mark.parametrize('landing',[[.36,-.12],[.40,.12],[.45,0]])
def test_execution_accepts_planned_landing_anywhere_in_right_half_plane(landing):
    cfg=load_config()
    p=ExpertPlan(target_indices=np.arange(6),feasible=True,strategy='coverage_astar',
                 waypoints=np.array([[.42,0],landing,[.1,0],[-.30,0],[-.37,0]]))
    path,phases=_expert_execution_path(cfg,p)
    descent=path[np.asarray(phases)=='descent']
    np.testing.assert_allclose(descent[:,:2],np.tile(landing,(len(descent),1)),atol=1e-7)


def test_misroute_failure_keeps_planned_descent_and_changes_only_later_contact_route():
    cfg=load_config()
    p=ExpertPlan(target_indices=np.arange(6),feasible=True,strategy='coverage_astar',
                 waypoints=np.array([[.42,0],[.36,0],[.1,0],[-.30,0],[-.37,0]]))
    path,phases=_expert_execution_path(cfg,p,failure_mode='misroute',failure_seed=0)
    descent=path[np.asarray(phases)=='descent']
    np.testing.assert_allclose(descent[:,:2],np.tile([.36,0],(len(descent),1)),atol=1e-7)
    assert np.max(path[np.asarray(phases)=='sweep',1]) > .10


@pytest.mark.parametrize('boundary',[.6,-.5,float('nan')])
def test_invalid_landing_boundary_is_explicitly_rejected(boundary):
    cfg=load_config()
    cfg.set_path('task.expert_landing_x_min',boundary)
    p=ExpertPlan(waypoints=np.array([[.42,0],[.2,0],[-.37,0]]))
    with pytest.raises(ValueError,match='landing'):
        expert_staging_xy(None,cfg,p)


def test_formal_resume_rejects_old_generation_rules_before_any_attempt(tmp_path,monkeypatch):
    # Missing protocol must not silently make old slots count toward the new batch.
    import json
    from sim.act.collect_dataset import generate_approved_training_dataset
    import sim.act.collect_dataset as collect
    old={'episode_id':'old','schema_version':5,'state_dim':36,'episode_kind':'expert',
         'split':'train','layout_kind':'independent','layout_fingerprint':'existing',
         'target_count':6,'success':True,'layout_id':'train_n6_layout_000',
         'target_selection':{'mode':'y_edge_contiguous',
          'initial_positions':[[.1,-.15,0],[.1,-.09,0],[.1,-.03,0],
                               [.1,.03,0],[.1,.09,0],[.1,.15,0]],
          'y_order':[0,1,2,3,4,5],'planned_component_ids':[0,1,2,3,4,5],
          'collected_component_ids':[0,1,2,3,4,5]}}
    monkeypatch.setattr(collect,'_capture_training_episode',
                        lambda *_a,**_k:pytest.fail('old protocol reached a physical attempt'))
    manifest=tmp_path/'manifest.jsonl'
    original=json.dumps(old)+'\n'
    manifest.write_text(original)
    with pytest.raises(ValueError,match='generation protocol'):
        generate_approved_training_dataset(load_config(),tmp_path,episodes_per_target=1)
    assert manifest.read_text()==original
    assert list(tmp_path.iterdir())==[manifest]


def _resume_record(protocol=None):
    return {'episode_id':'old','schema_version':5,'state_dim':36,'episode_kind':'expert',
            'split':'train','layout_kind':'independent','layout_fingerprint':'existing',
            'target_count':6,'success':True,'layout_id':'train_n6_layout_000',
            **({'expert_generation_protocol':protocol} if protocol is not None else {}),
            'target_selection':{'mode':'y_edge_contiguous',
             'initial_positions':[[.1,-.15,0],[.1,-.09,0],[.1,-.03,0],
                                  [.1,.03,0],[.1,.09,0],[.1,.15,0]],
             'y_order':list(range(6)),'planned_component_ids':list(range(6)),
             'collected_component_ids':list(range(6))}}


@pytest.mark.parametrize('approved',[True,False])
def test_cli_protocol_rejection_keeps_config_and_split_files_unchanged(tmp_path,monkeypatch,approved):
    import json
    from sim.act import collect_dataset as collect
    originals={'manifest.jsonl':json.dumps(_resume_record())+'\n',
               'train_config.yaml':'original provenance\n',
               'val_layouts.jsonl':'original validation plan\n'}
    for name,value in originals.items():
        (tmp_path/name).write_text(value)
    monkeypatch.setattr(collect,'run_expert_episode',
                        lambda *_a,**_k:pytest.fail('incompatible CLI reached physics'))
    args=['--out',str(tmp_path),'--episodes','1','--composition','mixed']
    if approved:
        args+=['--approved-plan','--write-split-manifests-only']
    with pytest.raises(ValueError,match='generation protocol'):
        collect.main(args)
    assert {p.name:p.read_text() for p in tmp_path.iterdir()}==originals


def test_frozen_legacy_resume_cannot_append_to_new_protocol(tmp_path,monkeypatch):
    import json
    from sim.act import collect_dataset as collect
    from sim.act.expert import expert_generation_protocol
    (tmp_path/'manifest.jsonl').write_text(json.dumps(_resume_record(expert_generation_protocol(load_config())))+'\n')
    legacy=load_config()
    legacy.set_path('task.expert_landing_x_min',None)
    legacy.set_path('planner.expert_planning_mode','mixed')
    monkeypatch.setattr(collect,'_capture_training_episode',
                        lambda *_a,**_k:pytest.fail('new records reached legacy physics'))
    with pytest.raises(ValueError,match='generation protocol'):
        collect.generate_approved_training_dataset(legacy,tmp_path,episodes_per_target=1)


def test_approved_resume_protocol_uses_actual_normalized_expert_configuration(tmp_path,monkeypatch):
    import json
    from sim.act import collect_dataset as collect
    from sim.act.expert import expert_generation_protocol
    cfg=load_config()
    cfg.set_path('components.geometry','screw')
    cfg.set_path('components.count',1)
    # Formal generation still uses six mixed parts, as it did before this task.
    normalized=collect._expert_config(cfg,6)
    record=_resume_record(expert_generation_protocol(normalized))
    (tmp_path/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    monkeypatch.setattr(collect,'training_episode_plan',lambda *_a:[])
    monkeypatch.setattr(collect,'validate_training_manifest',lambda *_a,**_k:{'total':1})
    monkeypatch.setattr(collect,'write_split_manifests',lambda *_a:{})
    report=collect.generate_approved_training_dataset(cfg,tmp_path,episodes_per_target=1)
    assert report['generated']==0


def test_normal_cli_writes_actual_protocol_metadata(tmp_path,monkeypatch):
    from sim.act import collect_dataset as collect
    from sim.act.expert import expert_generation_protocol
    writes=[]
    cfgs=[]
    class Writer:
        def __init__(self,*_a,**_k): pass
        def add_episode(self,*args,**_kwargs): writes.append(args[4])
    def rollout(cfg,**_kwargs):
        cfgs.append(cfg)
        return SimpleNamespace(success=True,observations=[],actions=[],trace=[],
            planner_status='feasible',planner_failure_reason='',planner_strategy='coverage_astar',
            planner_turn_count=1,planner_attempts=1,planner_score=1.,target_indices=list(range(6)),
            collected=6,failure_reason='',total=6)
    monkeypatch.setattr(collect,'ActDatasetWriter',Writer)
    monkeypatch.setattr(collect,'run_expert_episode',rollout)
    assert collect.main(['--out',str(tmp_path),'--episodes','1','--composition','mixed'])==0
    assert writes[0]['expert_generation_protocol']==expert_generation_protocol(cfgs[0])


def test_landing_insertion_preserves_the_original_astar_segment_and_suffix():
    from sim.act.expert import _landing_boundary_path
    cfg=load_config()
    original=np.array([[.42,.03],[.20,-.14],[.10,.16],[-.37,0]])
    revised=_landing_boundary_path(original,cfg)
    fraction=(.42-.36)/(.42-.20)
    expected=original[0]+fraction*(original[1]-original[0])
    np.testing.assert_allclose(revised[1],expected,atol=1e-12)
    np.testing.assert_array_equal(revised[2:],original[1:])
    assert np.linalg.norm(np.diff(revised,axis=0),axis=1).sum()==pytest.approx(
        np.linalg.norm(np.diff(original,axis=0),axis=1).sum())
    # Sweeping proceeds on the original left-hand side; only descent changes.
    assert revised[2,0]<.36


def test_already_allowed_landing_and_old_frozen_paths_are_not_projected():
    from sim.act.expert import _landing_boundary_path
    cfg=load_config()
    path=np.array([[.42,0],[.40,.12],[.20,-.1],[-.37,0]])
    np.testing.assert_array_equal(_landing_boundary_path(path,cfg),path)
    cfg.set_path('task.expert_landing_x_min',None)
    old=np.array([[.42,0],[.12,-.1],[-.37,0]])
    assert _landing_boundary_path(old,cfg) is old


@pytest.mark.parametrize('landing,yaw',[([.47,0],0),([.4,.33],0),([.4,0],np.pi),([np.nan,0],0)])
def test_external_landing_cannot_violate_workspace_or_descent_orientation(landing,yaw):
    cfg=load_config()
    plan=ExpertPlan(waypoints=np.array([[.42,0],landing,[-.37,0]]),yaw_offset=yaw)
    with pytest.raises(ValueError,match='landing'):
        _expert_execution_path(cfg,plan)


def test_frozen_legacy_to_legacy_resume_still_skips_completed_slots(tmp_path,monkeypatch):
    import json
    from sim.act import collect_dataset as collect
    cfg=load_config()
    cfg.set_path('task.expert_landing_x_min',None)
    cfg.set_path('planner.expert_planning_mode','mixed')
    (tmp_path/'manifest.jsonl').write_text(json.dumps(_resume_record())+'\n')
    monkeypatch.setattr(collect,'training_episode_plan',lambda *_a:[])
    monkeypatch.setattr(collect,'validate_training_manifest',lambda *_a,**_k:{'total':1})
    monkeypatch.setattr(collect,'write_split_manifests',lambda *_a:{})
    assert collect.generate_approved_training_dataset(cfg,tmp_path,episodes_per_target=1)['generated']==0
