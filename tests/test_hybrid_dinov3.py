"""Portable Hybrid contracts; never start a physical episode or training."""
import ast
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def subject():
    try:
        return importlib.import_module('sim.act.hybrid_trial4')
    except ModuleNotFoundError as error:
        pytest.fail('portable Hybrid/DINO runtime is missing: ' + str(error))


def test_runtime_loads_without_original_machine_files_or_global_import_alias_leaks(monkeypatch, tmp_path):
    # A loader which accidentally imports the old generic phases/bench alias
    # or requires the historical exports cannot supply this real controller.
    marker = SimpleNamespace(must_not_be_used=True)
    monkeypatch.setitem(sys.modules, 'phases', marker)
    monkeypatch.setitem(sys.modules, 'bench', marker)
    monkeypatch.setitem(sys.modules, 'bench_v2', marker)
    monkeypatch.chdir(tmp_path)
    before = sys.path.copy()
    runtime = subject().runtime()
    loop = runtime.TwoStageAdmittance(.01, runtime.TRIAL_SPEC)
    assert (loop.params.m, loop.params.b, loop.params.k) == (10., 400., 10.)
    assert loop.spec.force_N == 1.
    assert sys.path == before
    assert sys.modules['phases'] is marker
    assert sys.modules['bench'] is marker
    assert sys.modules['bench_v2'] is marker
    assert runtime.PROJECT == ROOT


def test_ported_control_functions_are_AST_identical_to_the_independent_hybrid_snapshot():
    subject().runtime()
    snapshot = ROOT / 'hybrid_trial4_snapshot/legacy_exports'
    portable = ROOT / 'sim/act/hybrid_trial4/legacy'
    def definitions(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for original in snapshot.rglob('*.py'):
        if original.name.startswith('test_'):
            continue
        port = portable / original.relative_to(snapshot)
        assert port.is_file(), original
        assert definitions(port) == definitions(original), original
        tree = ast.parse(port.read_text())
        assert not any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                       and n.value.startswith('/Users/') for n in ast.walk(tree))


def test_contact_gain_switch_and_stability_gate_preserve_state_and_delayed_ACT_ownership():
    r = subject().runtime()
    loop = r.TwoStageAdmittance(.01, r.TRIAL_SPEC)
    for _ in range(10):
        loop.step(1., 0.)
    before = (loop.delta_z, loop.delta_z_rate)
    for i in range(12):
        loop.observe(i*.01, .5)
    assert loop.phase == 'settle'
    assert (loop.params.m, loop.params.b, loop.params.k) == (10., 800., 10.)
    assert (loop.delta_z, loop.delta_z_rate) == before
    assert not loop.ready
    loop.observe(.12, 1.)
    loop.observe(.31, 1.)
    assert not loop.ready
    loop.observe(.32, 1.)
    assert loop.ready


def test_two_second_handoff_keeps_Z_and_restores_exact_native_ACT_after_expiry():
    from sim.act.evaluate import _policy_reference_substep
    r = subject().runtime()
    g = r.handoff.TwoSecondHandoff([.2, .1, .001, 0.], .01, 0.,
                                   bounds=[[-.8,.8],[-.8,.8]])
    g.chunk_ack(True, 0.)
    g.action_ready(0.)
    for i in range(200):
        target = np.array([.6,.5,-.9,1.])
        native = _policy_reference_substep(g.last_reference,target,1,1,.01,.08,.12,.35,True,.0008)
        command = g.propose(g.last_time,native,target,.0008)
        assert command[2] == .0008
        row = {'t':(i+1)*.01,'command':command.copy()}
        g.prepare_record(row); g.acknowledge(row)
    target = np.array([-.6,-.5,-.9,-1.])
    native = _policy_reference_substep(g.last_reference,target,1,1,.01,.08,.12,.35,True,.0009)
    assert np.array_equal(g.propose(2.,native,target,.0009),native)


def test_total_Z_correction_including_follow_compensation_never_exceeds_two_mm():
    r = subject().runtime()
    loop = r.TwoStageAdmittance(.01,r.TRIAL_SPEC)
    loop.reset(delta_z=-.00195)
    limiter = r.ZCommandLimiter(-.00095,.01)
    z = r.compose_z(.001,.00005,1.,-.012,loop,limiter,None)
    assert z == pytest.approx(-.001,abs=1e-12)


def test_force_or_prediction_exception_is_not_replaced_by_a_hold_or_new_query():
    from sim.act import evaluate
    formal = subject().formal
    strict = formal.strict_exception_function(evaluate.run_act_episode)
    # This is the real strict AST adapter; a deliberate execution failure must
    # escape it, not take the original pure-ACT swallowed-exception path.
    assert strict.__name__ == 'run_act_episode'
    source = ''.join(importlib.import_module('linecache').getlines(strict.__code__.co_filename))
    handlers = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ExceptHandler)]
    assert handlers
    assert all(isinstance(n.body[0], ast.Raise) for n in handlers)


@pytest.mark.parametrize('configured', [False, True])
def test_dino_checkpoint_uses_hybrid_even_with_old_workbench_config(monkeypatch, tmp_path, configured):
    from sim.act import evaluate
    from sim.config import load_config
    r = subject()
    cfg = load_config('configs/default.yaml')
    if configured:
        cfg.set_path('act.inference_mode', 'hybrid_trial4')
    model = tmp_path / 'checkpoint'; model.mkdir()
    (model / 'config.json').write_text(json.dumps({'type': 'act_dinov3', 'vision_backbone': 'dinov3_vits16'}))
    (model / 'model.safetensors').touch()
    expected = object(); calls = []
    monkeypatch.setattr(r, 'run_hybrid_episode', lambda *a, **kw: calls.append(kw) or expected)
    monkeypatch.setattr(evaluate, 'run_act_episode', lambda *a, **kw: pytest.fail('pure ACT selected'))
    dispatch = getattr(evaluate, 'run_inference_episode', None)
    assert callable(dispatch), 'public inference dispatcher is missing'
    assert dispatch(cfg, seed=42, model_path=str(model), preview=True, capture_replay=True) is expected
    assert calls == [dict(seed=42, model_path=str(model), preview=True, capture_replay=True)]


def test_resnet_keeps_original_inference_and_missing_hybrid_checkpoint_fails_before_environment(monkeypatch):
    from sim.act import evaluate
    from sim.config import load_config
    cfg = load_config('configs/default.yaml')
    expected = object()
    monkeypatch.setattr(evaluate, 'run_act_episode', lambda *a, **kw: expected)
    dispatch = getattr(evaluate, 'run_inference_episode', None)
    assert callable(dispatch), 'public inference dispatcher is missing'
    assert dispatch(cfg) is expected
    cfg.set_path('act.inference_mode', 'hybrid_trial4')
    cfg.set_path('act.model_dir', 'missing-checkpoint-for-contract-test')
    with pytest.raises(FileNotFoundError):
        dispatch(cfg)


@pytest.mark.parametrize('path', ['configs/five-view-dinov3-frozen-300.yaml',
                                 'configs/five-view-dinov3-frozen-300-cuda.yaml'])
def test_dino_configs_advertise_trial4_and_keep_safety(path):
    from sim.config import load_config
    cfg = load_config(path)
    assert cfg.get_path('act.inference_mode') == 'hybrid_trial4'
    assert cfg.controller.desired_force == 1.
    assert cfg.controller.delta_z_limit == .002
    assert (cfg.controller.admittance.m, cfg.controller.admittance.b, cfg.controller.admittance.k) == (10.,400.,10.)
    assert cfg.controller.safe_max_force == 20.
    assert cfg.controller.safe_force_dwell_steps == 3


def test_formal_configuration_and_adapters_equal_snapshot_and_only_controller_overlay():
    from sim.config import load_config
    r = subject()
    original = ast.parse((ROOT/'hybrid_trial4_snapshot/evaluate_hybrid_paired_batch.py').read_text())
    ported = ast.parse((ROOT/'sim/act/hybrid_trial4/formal.py').read_text())
    lookup = {n.name: ast.dump(n, include_attributes=False) for n in original.body if isinstance(n, ast.FunctionDef)}
    for n in ported.body:
        if isinstance(n, ast.FunctionDef):
            assert ast.dump(n, include_attributes=False) == lookup[n.name]
    assert r.PROFILE == json.loads((ROOT/'hybrid_trial4_snapshot/control_profile.json').read_text())
    base = load_config('configs/five-view-dinov3-frozen-300.yaml')
    cfg = r.formal.candidate_config(base, 6)
    baseline = load_config('hybrid_trial4_snapshot/evaluation_config.yaml')
    reference = r.formal.candidate_config(baseline,6)
    for field in ('sim','table','target','end_effector','workspace','controller','components','planner','task'):
        assert cfg.get_path(field).to_dict() == reference.get_path(field).to_dict(), field


def test_live_adapter_carries_state_and_hooks_into_original_ACT_without_running_physics():
    from sim.config import load_config
    from sim.act import evaluate
    r = subject(); module = r.runtime()
    cfg = r.formal.candidate_config(load_config('configs/five-view-dinov3-frozen-300.yaml'),1)
    reference = np.array([.2,.1,.001,0.])
    env = SimpleNamespace(time=0.,control_dt=.01,tcp=lambda:np.array([.2,.1,.001]))
    loop = module.TwoStageAdmittance(.01,module.TRIAL_SPEC)
    loop.reset(-.0002,.0001)
    context = dict(env=env,admittance=loop,relief=object(),z_nominal=.001,
        last_reference=reference,trace=[{'t':-.01,'command':reference.copy()},
        {'t':0.,'command':reference.copy()}],force_gate=object(),collection_timer=object(),
        z_limiter=module.ZCommandLimiter(.001,.01))
    original = evaluate.run_act_episode
    driver, governor = r.formal.build_formal_act_driver(module,cfg,context)
    assert evaluate.run_act_episode is original
    assert driver is not original
    assert context['admittance'] is loop
    assert (loop.delta_z,loop.delta_z_rate) == (-.0002,.0001)
    driver.__globals__['_trial_chunk_ack'](True)
    governor.action_ready(0.)
    out = driver.__globals__['_policy_reference_substep'](reference,[.5,.1,-.8,1.],1,4,.01,.08,.12,.35,True,.0009)
    assert out[0] == pytest.approx(.200005)
    assert out[2] == .0009
    assert context['accepted_chunks'] == 1


def test_public_driver_binds_actual_Z_follow_after_the_strict_globals_copy():
    from sim.config import load_config
    r = subject(); module = r.runtime()
    cfg = r.formal.candidate_config(load_config('configs/five-view-dinov3-frozen-300.yaml'),1)
    reference = np.array([.2,.1,-.00095,0.])
    env = SimpleNamespace(time=0.,control_dt=.01,tcp=lambda:np.array([.2,.1,.00005]))
    loop = module.TwoStageAdmittance(.01,module.TRIAL_SPEC); loop.reset(-.00195)
    context = dict(env=env,admittance=loop,relief=object(),z_nominal=.001,
        last_reference=reference,trace=[{'t':-.01,'command':reference.copy()},
        {'t':0.,'command':reference.copy()}],force_gate=object(),collection_timer=object(),
        z_limiter=module.ZCommandLimiter(-.00095,.01))
    factory = getattr(r, 'build_driver', r.formal.build_formal_act_driver)
    driver, _ = factory(module,cfg,context)
    z = driver.__globals__['_contact_z_command'](cfg,loop,.001,1.,relief=None)
    assert z == pytest.approx(-.001,abs=1e-12)


def test_public_driver_rejects_nonfinite_chunk_instead_of_silent_hold():
    from sim.config import load_config
    r = subject(); module = r.runtime()
    cfg = load_config('configs/five-view-dinov3-frozen-300.yaml')
    reference = np.array([.2,.1,.001,0.])
    context = dict(env=SimpleNamespace(time=0.,control_dt=.01),
        admittance=module.TwoStageAdmittance(.01,module.TRIAL_SPEC),relief=object(),z_nominal=.001,
        last_reference=reference,trace=[{'t':-.01,'command':reference.copy()}, {'t':0.,'command':reference.copy()}],
        force_gate=object(),collection_timer=object(),z_limiter=module.ZCommandLimiter(.001,.01))
    factory = getattr(r, 'build_driver', r.formal.build_formal_act_driver)
    driver, _ = factory(module,cfg,context)
    scheduler = driver.__globals__['ActionChunkScheduler'](cfg,allow_late=True)
    with pytest.raises(ValueError,match='nonfinite|invalid'):
        scheduler.accept(0.,np.full((25,4),np.nan),0.)


def test_workbench_DINO_discovery_uses_configured_portable_output_not_Resnet(tmp_path):
    from sim.web.app import latest_inference_model
    model_root = tmp_path/'runs'/'new_dino'
    checkpoint = model_root/'checkpoints'/'step_050000'; checkpoint.mkdir(parents=True)
    files = ['config.json','model.safetensors','policy_preprocessor.json',
        'policy_preprocessor_step_3_normalizer_processor.safetensors','policy_postprocessor.json',
        'policy_postprocessor_step_0_unnormalizer_processor.safetensors','training_state.pt']
    for name in files:
        (checkpoint/name).write_text('data')
    (checkpoint/'config.json').write_text(json.dumps({'type':'act_dinov3'}))
    (model_root/'latest_checkpoint.txt').write_text('checkpoints/step_050000')
    assert latest_inference_model(tmp_path,model_dir='runs/new_dino') == 'runs/new_dino/checkpoints/step_050000'
    (checkpoint/'config.json').write_text(json.dumps({'type':'act'}))
    assert latest_inference_model(tmp_path,model_dir='runs/new_dino') == ''


@pytest.mark.parametrize('outcome', ['ready','no_plan','unknown_acquisition_error'])
def test_public_pipeline_single_plan_single_environment_and_causal_context(monkeypatch, outcome):
    from contextlib import nullcontext
    from sim.config import load_config
    from sim.act.rollout import ActRolloutResult, CollectionTerminationTracker
    from sim.act import expert
    from sim.environments import sweep_env
    r=subject(); module=r.runtime(); calls=[]
    class BoundaryEnv:
        time=2.; control_dt=.01
        def __init__(self,*args,**kwargs): calls.append('create')
        def reset(self,**kwargs): calls.append('reset')
        def close(self): calls.append('close')
        def component_positions(self): return np.zeros((6,3))
        def component_quats(self): return np.tile([1.,0.,0.,0.],(6,1))
        def collected_mask(self): return np.array([1,0,0,0,0,0],dtype=bool)
    monkeypatch.setattr(sweep_env,'SweepEnv',BoundaryEnv)
    monkeypatch.setattr(r,'require_dinov3_checkpoint',lambda path:'test-checkpoint')
    monkeypatch.setattr(r,'runtime',lambda:module)
    monkeypatch.setattr(module.old.precision,'temporary_solver',lambda *args:nullcontext())
    def plan(env,cfg,max_candidates):
        calls.append(('plan',max_candidates))
        return [] if outcome=='no_plan' else [SimpleNamespace(feasible=True)]
    monkeypatch.setattr(expert,'expert_plan_candidates',plan)
    timer=CollectionTerminationTracker(1); timer.first_entry_time=.7
    context=dict(ready=outcome=='ready',step_failures=[],invalid_samples=[],
        trace=[{'t':.1}],actions=[[0,0,0,0]],collection_timer=timer,
        error='unknown cause',peak_force=1.,prefix_frames=1)
    monkeypatch.setattr(module,'prepare_contact',lambda *args:context)
    def factory(m,cfg,c):
        assert m is module and c is context and c['collection_timer'] is timer
        def drive(*args,**kw):
            assert kw['capture_replay'] is True
            calls.append('ACT_boundary')
            return ActRolloutResult(False,'count mismatch',trace=[{'t':2.}],
                actions=np.zeros((1,4)),sampled_frames=2,final_collected=2,collected=2)
        return drive,None
    monkeypatch.setattr(r,'build_driver',factory)
    cfg=load_config('configs/five-view-dinov3-frozen-300.yaml')
    if outcome=='unknown_acquisition_error':
        with pytest.raises(RuntimeError,match='unexpected acquisition') as raised:
            r.run_hybrid_episode(cfg,model_path='test',capture_replay=False)
        assert raised.value.hybrid_context is context
        assert 'ACT_boundary' not in calls
    else:
        result=r.run_hybrid_episode(cfg,model_path='test',capture_replay=False)
        assert result.physical_attempts == (0 if outcome=='no_plan' else 1)
        if outcome=='ready':
            assert len(result.actions)==2 and result.sampled_frames==2
            assert timer.first_entry_time==.7
            assert result.inference_config.controller.delta_z_limit==.002
            assert calls.count('ACT_boundary')==1
    assert calls.count('create')==calls.count('reset')==calls.count('close')==1
    assert calls.count(('plan',1))==1


def test_cli_publishes_unexecuted_no_plan_without_trying_to_render(monkeypatch,capsys):
    from sim.act import evaluate,replay
    from sim.act.rollout import ActRolloutResult
    result=ActRolloutResult(False,'no feasible single geometric plan',target_count=1,total=6,
        final_collected=0,collected=0,termination_reason='no_plan',planner_attempts=1)
    result.physical_attempts=0
    result.planner_strategy='hybrid_astar_act_trial4_dinov3_frozen'
    result.initial_object_poses=np.tile([0,0,0,1,0,0,0],(6,1))
    monkeypatch.setattr(evaluate,'run_inference_episode',lambda *a,**kw:result)
    monkeypatch.setattr(replay,'persist_replay_input',lambda *a,**kw:pytest.fail('no physical trace to replay'))
    assert evaluate._run_main(['--record-replay','--target-count','1','--seed','42'])==1
    row=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert row.get('media_not_applicable_no_plan') is True
    assert row['physical_attempts']==0 and row['planning_calls']==1 and row['seed']==42
    assert row['replay_error']==row['video_error']==''


def test_batch_recovers_unexecuted_hybrid_failure_without_video_or_physical_retry(tmp_path,monkeypatch):
    import hashlib
    from sim.act import batch_evaluate as b
    poses=np.tile([0,0,0,1,0,0,0],(6,1)).astype('<f4')
    fingerprint=hashlib.sha256(poses.tobytes()).hexdigest()
    layouts=[dict(layout_index=i,seed=42+i,layout_fingerprint=fingerprint if i==0 else f'layout-{i}',
                  initial_object_poses=poses.tolist()) for i in range(120)]
    plan=b.make_episode_plan(layouts,[dict(label='hybrid_dino_100k',path='checkpoint')])
    (tmp_path/'episode_plan.json').write_text(json.dumps(plan))
    video=tmp_path/'prior.mp4'; video.write_text('prior verified media')
    prior=[{**s,'video_path':str(video),'success':False,'peak_collected':0,'final_collected':0} for s in plan[1:]]
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prior))
    (tmp_path/'logs').mkdir(); config=tmp_path/'evaluation_config.yaml';config.write_text('{}')
    physical=dict(success=False,target_count=1,peak_collected=0,final_collected=0,seed=42,
        model_checkpoint='checkpoint',inference_strategy='hybrid_astar_act_trial4_dinov3_frozen',
        termination_reason='no_plan',physical_attempts=0,planning_calls=1,media_not_applicable_no_plan=True,
        initial_object_poses=poses.tolist(),replay_episode_id=None,video_path=None,replay_error='',video_error='')
    (tmp_path/'logs'/'hybrid_dino_100k_n1_l00.log').write_text(json.dumps(physical)+'\n')
    metadata=dict(project=str(tmp_path),models=[],source_fingerprints={},config_sha256=b._sha256(config),
        episode_plan_sha256=b._sha256(tmp_path/'episode_plan.json'),layout_seed_mode='independent_per_episode')
    monkeypatch.setattr(b.subprocess,'check_output',lambda *a,**kw:'')
    monkeypatch.setattr(b.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not retry measured no-plan'))
    assert b.parse_episode_output(json.dumps(physical),1)['physical_attempts']==0
    b.run_batch(tmp_path,metadata)
    rows=[json.loads(s) for s in (tmp_path/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==120 and rows[-1]['success'] is False
    assert rows[-1]['layout_identity_verification']['measured_pose_fingerprint']==fingerprint
    assert json.loads((tmp_path/'summary.json').read_text())['complete'] is True


def test_replay_recovery_uses_the_same_effective_trial4_config_without_mutating_raw_config():
    from sim.config import load_config
    from sim.act import batch_evaluate as b
    r=subject(); raw=load_config('configs/default.yaml'); before=raw.to_dict()
    physical=dict(inference_strategy='hybrid_astar_act_trial4_dinov3_frozen',hybrid_control_profile=r.PROFILE)
    helper=getattr(b,'effective_replay_config',None)
    assert callable(helper),'recovery overlay helper missing'
    cfg=helper(raw,physical)
    assert cfg.to_dict()==r.formal.candidate_config(raw,int(raw.task.target_count)).to_dict()
    assert raw.to_dict()==before


def test_batch_DINO_identity_is_not_mislabeled_as_pure_five_view_ACT(tmp_path):
    from sim.act import batch_evaluate as b
    checkpoint=tmp_path/'checkpoints'/'step_100000';checkpoint.mkdir(parents=True)
    for name in ('model.safetensors','training_state.pt'):
        (checkpoint/name).write_text('fixture')
    (checkpoint/'config.json').write_text(json.dumps({'type':'act_dinov3'}))
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):
        (checkpoint/name).write_text(json.dumps({'steps':[]}))
    (tmp_path/'latest_checkpoint.txt').write_text('checkpoints/step_100000')
    assert b.select_models(tmp_path)[0]['label']=='hybrid_dinov3_100k'
