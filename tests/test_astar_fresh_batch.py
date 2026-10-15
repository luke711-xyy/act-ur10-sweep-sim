"""Fresh single-opportunity A* must retain live counts even on execution errors."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_astar_fresh_batch.py'
    assert path.exists(), 'fresh A* evaluator with live exception counts is missing'
    spec=importlib.util.spec_from_file_location('astar_fresh',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def scene(runner,monkeypatch):
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    cfg=load_config();cfg.set_path('planner.expert_planning_mode','coverage_astar_only')
    class Env:
        collected_mask=SweepEnv.collected_mask
        partial_collection_overlap_mask=SweepEnv.partial_collection_overlap_mask
        def __init__(self):
            self.cfg=cfg;self.positions=np.zeros((6,3));self.quats=np.tile([1.,0,0,0],(6,1))
            self.layout=[{'nominal_radius':.001} for _ in range(6)]
            self.table_top_z=cfg.table.top_z;self.closed=False;self.model=None
            self.data=SimpleNamespace(time=0.,qpos=np.zeros(7),qvel=np.zeros(7),ctrl=np.zeros(6))
        @property
        def time(self):return self.data.time
        def component_positions(self):return self.positions.copy()
        def component_quats(self):return self.quats.copy()
        def reset(self,seed):pass
        def step_control(self,command):self.data.time+=.01
        def close(self):self.closed=True
        def collect(self,count):
            self.positions[:count,0]=(cfg.target.x_min+cfg.target.x_max)/2
            self.positions[:count,1]=(cfg.target.y_min+cfg.target.y_max)/2
    env=Env();spec={'key':'k0','seed':2120000,'target_count':3,
                  'layout_fingerprint':runner.layout_fingerprint(env)}
    monkeypatch.setattr(runner.base,'SweepEnv',lambda local,seed:env)
    return cfg,env,spec


@pytest.mark.parametrize('count',[0,1,2,3,6])
def test_execution_error_uses_actual_quantity_without_retry(runner,scene,monkeypatch,tmp_path,count):
    cfg,env,spec=scene;events=[]
    plan=SimpleNamespace(feasible=True,target_indices=np.array([0,1,2]))
    monkeypatch.setattr(runner.base,'expert_plan_candidates',lambda *a,**k:[plan])
    monkeypatch.setattr(runner.base,'_expert_execution_path',lambda *a,**k:(np.zeros((2,4)),['approach','sweep']))
    def failing_control(command):
        env.collect(count)
        raise RuntimeError('UR10 IK target is unreachable: position residual=0.001941 m, orientation residual=0.021053 rad')
    env.step_control=failing_control
    def execute(actual,local,path,**kwargs):
        from sim.controllers.hybrid import Command
        events.append('physical')
        assert local.episode.max_frames==2000
        assert local.episode.max_time==2000/local.act.action_hz
        assert kwargs['recovery_planner'] is None and kwargs['max_recovery_rounds']==0
        actual.step_control(Command(0.,0.,0.,0.))
    monkeypatch.setattr(runner.base,'run_action_path',execute)
    row=runner.run_trial(cfg,spec,tmp_path/'episode')
    assert row['success'] is False and row['final_collected']==count
    assert row['outcome']=='execution_exception'
    assert row['opportunities']==row['planning_calls']==row['physical_attempts']==1
    assert events==['physical'] and env.closed
    with np.load(tmp_path/'episode'/'final_state.npz',allow_pickle=False) as state:
        assert int(state['fully_collected_mask'].sum())==count
        assert np.array_equal(state['component_positions'],env.positions)


def test_no_plan_does_not_screen_or_execute_another_candidate(runner,scene,monkeypatch,tmp_path):
    cfg,env,spec=scene
    monkeypatch.setattr(runner.base,'expert_plan_candidates',lambda *a,**k:[SimpleNamespace(feasible=False),SimpleNamespace(feasible=True)])
    monkeypatch.setattr(runner.base,'run_action_path',lambda *a,**k:pytest.fail('no physical fallback'))
    row=runner.run_trial(cfg,spec,tmp_path/'episode')
    assert row['success'] is False and row['no_plan'] is True
    assert row['physical_attempts']==0 and row['opportunities']==1
    assert row['final_collected']==0 and env.closed


def test_success_measured_from_live_environment_not_result_counter(runner,scene,monkeypatch,tmp_path):
    cfg,env,spec=scene;plan=SimpleNamespace(feasible=True,target_indices=np.array([0,1,2]))
    calls=[]
    def plans(*a,**kw):calls.append(kw['max_candidates']);return [plan,SimpleNamespace(feasible=True)]
    monkeypatch.setattr(runner.base,'expert_plan_candidates',plans)
    monkeypatch.setattr(runner.base,'_expert_execution_path',lambda *a,**k:(np.zeros((2,4)),['approach','sweep']))
    def execute(*a,**k):env.collect(3);return SimpleNamespace(success=True,failure_reason='',elapsed=1.,planner_strategy='coverage_astar')
    monkeypatch.setattr(runner.base,'run_action_path',execute)
    monkeypatch.setattr(runner.base,'_annotate_expert_result',lambda r,*a:r)
    row=runner.run_trial(cfg,spec,tmp_path/'episode')
    assert row['success'] is True and row['final_collected']==3 and calls==[1]


def test_unreadable_terminal_state_stops_instead_of_inventing_zero(runner,scene,monkeypatch,tmp_path):
    cfg,env,spec=scene
    monkeypatch.setattr(runner.base,'expert_plan_candidates',lambda *a,**k:[])
    env.collected_mask=lambda:np.array([True,False])
    with pytest.raises(ValueError,match='measurement'):
        runner.run_trial(cfg,spec,tmp_path/'episode')
    assert not (tmp_path/'episode'/'result.json').exists() and env.closed


def test_layout_mismatch_never_plans(runner,scene,monkeypatch,tmp_path):
    cfg,env,spec=scene;spec['layout_fingerprint']='wrong'
    monkeypatch.setattr(runner.base,'expert_plan_candidates',lambda *a,**k:pytest.fail('mismatched layout planned'))
    with pytest.raises(ValueError,match='layout'):runner.run_trial(cfg,spec,tmp_path/'episode')


@pytest.mark.parametrize('kind',['schema','io_control','unidentified_control','ik_outside_control'])
def test_infrastructure_errors_preserve_evidence_but_do_not_score_or_continue(runner,scene,monkeypatch,tmp_path,kind):
    cfg,env,spec=scene;env.collect(2)
    # Preserve the paired initial identity after constructing this test scene.
    spec['layout_fingerprint']=runner.layout_fingerprint(env)
    plan=SimpleNamespace(feasible=True,target_indices=np.array([0,1,2]))
    monkeypatch.setattr(runner.base,'expert_plan_candidates',lambda *a,**k:[plan])
    monkeypatch.setattr(runner.base,'_expert_execution_path',lambda *a,**k:(np.zeros((2,4)),['sweep']))
    error={'schema':ValueError('invalid observation schema'),'io_control':OSError('disk failed'),
           'unidentified_control':RuntimeError('unexpected internal fault'),
           'ik_outside_control':RuntimeError('UR10 IK target is unreachable: position residual=0.001941 m, orientation residual=0.021053 rad')}[kind]
    def failing_control(command):raise error
    if kind in ('io_control','unidentified_control'):env.step_control=failing_control
    def execute(actual,*args,**kwargs):
        from sim.controllers.hybrid import Command
        if kind in ('io_control','unidentified_control'):actual.step_control(Command(0.,0.,0.,0.))
        raise error
    monkeypatch.setattr(runner.base,'run_action_path',execute)
    with pytest.raises(type(error),match=str(error)):
        runner.run_trial(cfg,spec,tmp_path/'episode')
    assert not (tmp_path/'episode'/'result.json').exists()
    with np.load(tmp_path/'episode'/'final_state.npz') as state:
        assert int(state['fully_collected_mask'].sum())==2
    assert (tmp_path/'episode'/'execution_exception.json').exists() and env.closed


def test_attempt_marker_is_exclusive_even_without_result(runner,tmp_path):
    spec={'key':'k0','seed':2120000,'target_count':1,'layout_fingerprint':'f0'}
    runner.claim_attempt(tmp_path,spec)
    with pytest.raises(FileExistsError):runner.claim_attempt(tmp_path,spec)


def test_summary_includes_failure_quantity_in_error_statistics(runner):
    rows=[{'target_count':3,'success':False,'final_collected':2,'physical_attempts':1,'no_plan':False,'outcome':'execution_exception'},
          {'target_count':3,'success':True,'final_collected':3,'physical_attempts':1,'no_plan':False,'outcome':'native_result'}]
    summary=runner.summarize(rows)
    assert summary['successes']==1 and summary['execution_exceptions']==1
    assert summary['by_target']['3']['mean_absolute_error']==.5
    assert summary['by_target']['3']['exact_quantity']==1


def test_summary_rejects_unknown_count_instead_of_coercion(runner):
    with pytest.raises(ValueError):runner.summarize([{'target_count':1,'success':False,'final_collected':None,'physical_attempts':1,'no_plan':False,'outcome':'execution_exception'}])


@pytest.fixture
def prepared(runner,monkeypatch,tmp_path):
    source=tmp_path/'act';source.mkdir()
    plan=[{'key':f'k{i}','seed':2120000+i,'target_count':i%6+1,'layout_fingerprint':f'f{i}'} for i in range(120)]
    (source/'episode_plan.json').write_text(json.dumps(plan))
    (source/'evaluation_config.yaml').write_text('planner: {expert_planning_mode: coverage_astar_only}\n')
    (source/'metadata.json').write_text(json.dumps({'source_fingerprints':{},
        'episode_plan_sha256':runner.sha(source/'episode_plan.json'),
        'config_sha256':runner.sha(source/'evaluation_config.yaml')}))
    (source/'episodes.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in plan))
    (source/'status.json').write_text(json.dumps({'state':'complete','completed':120}))
    (source/'summary.json').write_text(json.dumps({'complete':True,'completed':120}))
    monkeypatch.setattr(runner,'EXPORTS',tmp_path)
    monkeypatch.setattr(runner.base,'act_complete',lambda root:True)
    monkeypatch.setattr(runner.base,'_source_fingerprints',lambda project:{})
    root=tmp_path/'fresh';runner.prepare(source,root)
    return root,source,plan


def test_preparation_copies_exact_plan_and_has_no_attempts(runner,prepared):
    root,source,plan=prepared
    assert (root/'episode_plan.json').read_bytes()==(source/'episode_plan.json').read_bytes()
    assert not (root/'attempts').exists() and not (root/'started.json').exists()
    assert runner.verify(root)['ACT_loaded'] is False


@pytest.mark.parametrize('changed',['episode_plan.json','evaluation_config.yaml'])
def test_frozen_input_change_is_rejected_before_any_opportunity(runner,prepared,changed):
    root,source,plan=prepared;(root/changed).write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen input'):runner.run_batch(root)
    assert not (root/'attempts').exists() and not (root/'started.json').exists()


def test_metadata_change_is_rejected(runner,prepared):
    root,_,_=prepared;meta=json.loads((root/'metadata.json').read_text());meta['planned']=119
    runner.write_json(root/'metadata.json',meta)
    with pytest.raises(ValueError,match='seal'):runner.verify(root)


@pytest.mark.parametrize('existing',['started.json','episodes.jsonl','attempts'])
def test_started_batch_is_never_resumed_or_partially_skipped(runner,prepared,existing):
    root,_,_=prepared
    if existing=='attempts':(root/existing).mkdir()
    else:(root/existing).write_text('{}')
    with pytest.raises(ValueError,match='already started'):runner.run_batch(root)


def test_competing_parent_cannot_overwrite_status(runner,prepared):
    import fcntl
    root,_,_=prepared;before=(root/'status.json').read_bytes()
    with (root/'batch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):runner.run_batch(root)
    assert (root/'status.json').read_bytes()==before


def test_standalone_worker_cannot_consume_a_plan(runner,prepared):
    root,_,_=prepared;runner.write_json(root/'started.json',{'pid':-1})
    with pytest.raises(ValueError,match='owning parent'):runner.authorize_worker(root,0)
    with pytest.raises(ValueError,match='outside plan'):runner.authorize_worker(root,-1)


def test_parent_commits_120_unique_opportunities_and_does_not_restart(runner,prepared,monkeypatch):
    from contextlib import nullcontext
    root,source,plan=prepared;calls=[]
    monkeypatch.setattr(runner.base,'inference_lease',lambda **kwargs:nullcontext())
    def worker(command,**kwargs):
        index=int(command[-1]);spec=plan[index];calls.append(spec['seed'])
        destination=root/'episodes'/f"seed_{spec['seed']}";destination.mkdir(parents=True)
        count=2 if index==0 else spec['target_count']
        mask=np.arange(6)<count
        np.savez(destination/'final_state.npz',fully_collected_mask=mask)
        row={**spec,'success':index!=0,'final_collected':count,'opportunities':1,'planning_calls':1,
            'physical_attempts':1,'no_plan':False,'outcome':'execution_exception' if index==0 else 'native_result',
            'final_state':str(destination/'final_state.npz'),'final_state_sha256':runner.sha(destination/'final_state.npz')}
        runner.write_json(destination/'result.json',row)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',worker)
    monkeypatch.setattr(runner,'render_artifacts',lambda out,rows:runner.write_json(out/'summary.json',runner.summarize(rows)))
    runner.run_batch(root)
    rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==len(calls)==len(set(calls))==120
    assert len(list((root/'attempts').iterdir()))==120
    assert rows[0]['success'] is False and rows[0]['final_collected']==2
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    assert json.loads((root/'summary.json').read_text())['execution_exceptions']==1
    with pytest.raises(ValueError,match='already started'):runner.run_batch(root)
    assert len(calls)==120


def test_failed_worker_halts_without_fabricated_row_or_retry(runner,prepared,monkeypatch):
    from contextlib import nullcontext
    root,_,_=prepared;calls=[]
    monkeypatch.setattr(runner.base,'inference_lease',lambda **kwargs:nullcontext())
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:(calls.append(1) or SimpleNamespace(returncode=9)))
    with pytest.raises(RuntimeError,match='no retry'):runner.run_batch(root)
    assert calls==[1] and len(list((root/'attempts').iterdir()))==1
    assert not (root/'episodes.jsonl').exists()
    assert json.loads((root/'status.json').read_text())['state']=='needs_attention'


def test_fresh_output_must_not_reuse_an_existing_directory(runner,prepared):
    root,source,_=prepared;before=(source/'metadata.json').read_bytes()
    with pytest.raises(ValueError,match='new independent'):runner.prepare(source,source)
    assert (source/'metadata.json').read_bytes()==before


def test_heatmap_artifact_uses_failures_and_final_counts(runner,tmp_path):
    rows=[{'target_count':1,'final_collected':2,'success':False,'physical_attempts':1,
           'no_plan':False,'outcome':'execution_exception','key':'a','seed':1,'reason':'IK','elapsed':1.}]
    runner.render_artifacts(tmp_path,rows)
    assert (tmp_path/'final_count_density.png').exists() and (tmp_path/'final_count_density.svg').exists()
    assert json.loads((tmp_path/'summary.json').read_text())['by_target']['1']['mean_absolute_error']==1
