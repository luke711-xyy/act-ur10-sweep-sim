"""Pilot lifecycle contracts without launching a pilot or training workload."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def call(name,*args,**kwargs):
    try: module=importlib.import_module('residual_rl.pilot')
    except ModuleNotFoundError: module=SimpleNamespace()
    function=getattr(module,name,None)
    assert callable(function),name+' missing'
    return function(*args,**kwargs)


def test_fixed_seed_schedule_is_balanced_and_disjoint():
    schedule=call('seed_schedule')
    train=schedule['training']; valid=schedule['validation']; probe=schedule['preflight']
    assert {v['target_count'] for v in valid}==set(range(1,7)) and len(valid)==6
    assert [r['target_count'] for r in train[:12]]==[1,2,3,4,5,6]*2
    assert not ({r['seed'] for r in train}&{r['seed'] for r in valid})
    assert not ({r['seed'] for r in train}&{r['seed'] for r in probe})
    assert not ({r['seed'] for r in train}|{r['seed'] for r in valid})&set(range(2120000,2120120))


def test_preflight_requires_actual_act_not_merely_matching_empty_failures():
    empty=dict(command_sha256='a',Z_sha256='a',success=False,final_collected=0,
               control_steps=0,accepted_ACT_chunks=0,residual_frames=0,reason='no_plan')
    with pytest.raises(RuntimeError): call('require_equivalent',empty,empty)
    native=dict(empty,control_steps=400,accepted_ACT_chunks=10,residual_frames=20)
    call('require_equivalent',native,native)
    with pytest.raises(RuntimeError): call('require_equivalent',native,dict(native,Z_sha256='b'))


def test_update_schedule_does_not_train_before_all_goals_or_repeat_old_steps():
    schedule=call('UpdateSchedule')
    schedule.begin_episode(100,ready=False)
    assert not schedule.due()
    schedule.begin_episode(100,ready=True)
    assert not schedule.due()  # readiness transition never backpays warmup
    schedule.begin_episode(100,ready=True)
    assert schedule.due()
    for _ in range(4): schedule.completed()
    assert not schedule.due()
    schedule.begin_episode(24,ready=True); assert not schedule.due()
    schedule.begin_episode(1,ready=True); assert schedule.due()


def test_summary_keeps_prefix_failures_in_denominator_and_count_error_separate():
    rows=[dict(phase='train',target_count=1,success=False,final_collected=1,residual_frames=0,
               outcome='failure',opportunities=1),
          dict(phase='train',target_count=6,success=True,final_collected=6,residual_frames=10,
               outcome='success',opportunities=1)]
    report=call('summarize',rows)
    assert report['train']['episodes']==2 and report['train']['strict_successes']==1
    assert report['train']['quantity_matches']==2 and report['train']['prefix_only_episodes']==1


def test_budget_truncated_attempts_are_reported_separately_from_task_failures():
    report=call('summarize',[dict(phase='train',target_count=1,success=None,
        outcome='budget_truncated',final_collected=2,residual_frames=5,opportunities=1)])
    assert report['train']['episodes']==0 and report['train']['administrative_truncations']==1
    assert report['train']['strict_success_rate'] is None


def test_cli_help_and_default_inspect_never_create_run_or_load_act(tmp_path):
    entry=Path(__file__).parents[1]/'scripts/train_residual_sac.py'
    assert entry.is_file(),'training entry missing'
    help_result=subprocess.run([sys.executable,'-B',str(entry),'--help'],capture_output=True,text=True)
    assert help_result.returncode==0 and 'acknowledge-pilot-training' in help_result.stdout
    output=tmp_path/'must_not_exist'
    result=subprocess.run([sys.executable,'-B',str(entry),'--out',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not output.exists()
    assert json.loads(result.stdout)['execution_started'] is False


def test_cli_refuses_training_without_explicit_execution_acknowledgement(tmp_path):
    entry=Path(__file__).parents[1]/'scripts/train_residual_sac.py'
    assert entry.is_file(),'training entry missing'
    output=tmp_path/'must_not_exist'
    result=subprocess.run([sys.executable,'-B',str(entry),'--mode','pilot','--out',str(output)],
                           capture_output=True,text=True)
    assert result.returncode!=0 and not output.exists()


def test_prepare_refuses_source_drift_before_creating_output(monkeypatch,tmp_path):
    import residual_rl.runtime as runtime
    project=tmp_path/'project'; (project/'sim').mkdir(parents=True)
    (project/'sim'/'changed.py').write_text('current source')
    batch=tmp_path/'batch'; batch.mkdir()
    (batch/'metadata.json').write_text(json.dumps({'source_fingerprints':{'sim/changed.py':'old hash'}}))
    monkeypatch.setattr(runtime,'PROJECT',project); monkeypatch.setattr(runtime,'HYBRID',batch)
    output=tmp_path/'never_created'
    with pytest.raises(RuntimeError,match='source drift'): call('prepare',output)
    assert not output.exists()


def test_output_cannot_be_nested_in_old_results_or_checkpoints(tmp_path):
    import residual_rl.runtime as runtime
    with pytest.raises(ValueError): call('validate_output_path',runtime.HYBRID/'new_child')
    with pytest.raises(ValueError): call('validate_output_path',runtime.PROJECT/'runs'/'new_child')
    assert call('validate_output_path',tmp_path/'new_run')==(tmp_path/'new_run').resolve()


def constructed_probe(tmp_path,*,zero=False,jump=False,clipped=False):
    """Independent hand-integrated pose offset, held then returned; no reversal."""
    from residual_rl.storage import TraceWriter
    path=tmp_path/'probe.jsonl.gz'; trace=TraceWriter(path)
    applied=np.zeros(4); offset=np.zeros(3); previous_native=applied.copy()
    nominal=np.array([.079,0.,.3]) if clipped else np.array([.02,0.,0.])
    for i in range(240):
        stage='small_offset' if i<40 else 'hold_offset' if i<100 else 'return_offset'
        prior=offset.copy()
        if not zero:
            offset[1]=(i+1)*.00001 if i<40 else .0004 if i<100 else max(0.,.0004-(i-99)*.00001)
        desired=np.zeros(3) if zero or i>=100 else np.array([0.,.001,0.])
        before=applied.copy(); native=previous_native+np.r_[nominal[:2]*.01,0.,nominal[2]*.01]
        applied=native+np.r_[offset[:2],0.,offset[2]]
        if jump: applied[0]=before[0]+1.
        trace.append(dict(t=(i+1)*.01,command=applied.copy(),tcp=applied[:3].copy(),active=True,
            nominal_velocity=nominal.copy(),probe_stage=stage,control_index=i+1,
            command_before=before.copy(),command_started_at=i*.01,
            tcp_before=before[:3].copy(),tcp_yaw_before=before[3],tcp_yaw=applied[3],
            control_audit=dict(raw_action=desired/np.array([.005,.005,np.pi/60]),desired_offset=desired,
                applied_offset=offset.copy(),offset_before=prior,action_contract='bounded_pose_xy5mm_yaw3deg_v1',
                filtered_residual=(offset-prior)/.01,native_before=previous_native.copy(),native=native,applied=applied.copy(),
                Z_unchanged=True,active=True,start=before,dt=.01,
                bounds=[[-10.,10.],[-10.,10.]])))
        previous_native=native.copy()
    trace.close()
    return SimpleNamespace(trace_path=path,row=dict(outcome='failure',accepted_ACT_chunks=2,
        probe_modes=['small_offset','hold_offset','return_offset'],control_steps=240))


@pytest.mark.parametrize('bad', ['zero','jump'])
def test_probe_rejects_all_zero_labels_and_100_m_s_command_jumps(tmp_path,bad):
    rollout=constructed_probe(tmp_path,zero=bad=='zero',jump=bad=='jump')
    with pytest.raises(RuntimeError): call('audit_probe',rollout)


def test_probe_accepts_independently_constructed_small_hold_return_offsets(tmp_path):
    result=call('audit_probe',constructed_probe(tmp_path))
    assert result['passed'] is True
    assert result['nonzero_offset_steps']>0
    assert result['max_xy_offset_m']==pytest.approx(.0004)
    assert result['max_command_xy_m_s']==pytest.approx(np.hypot(.02,.001))
    assert result['continuous_contact_verified'] is False


def test_pose_probe_requires_actual_return_not_just_return_stage_labels(tmp_path):
    import gzip
    rollout=constructed_probe(tmp_path)
    with gzip.open(rollout.trace_path,'rt') as stream: rows=[json.loads(line) for line in stream]
    # A perfectly valid bounded trajectory but the offset never returns.
    for i,row in enumerate(rows):
        if i<100: continue
        previous=rows[i-1]
        row['command'][1]=row['tcp'][1]=.0004
        row['command_before']=previous['command'].copy(); row['tcp_before']=previous['tcp'].copy()
        audit=row['control_audit']; audit['start']=row['command_before'].copy()
        audit['applied']=row['command'].copy(); audit['applied_offset']=[0.,.0004,0.]
        audit['offset_before']=[0.,.0004,0.]; audit['filtered_residual']=[0.,0.,0.]
    with gzip.open(rollout.trace_path,'wt') as stream:
        stream.write('\n'.join(json.dumps(row) for row in rows)+'\n')
    with pytest.raises(RuntimeError): call('audit_probe',rollout)


def test_pose_probe_rejects_cancelling_native_yaw_even_when_xy_keeps_moving(tmp_path):
    import gzip
    rollout=constructed_probe(tmp_path)
    with gzip.open(rollout.trace_path,'rt') as stream: rows=[json.loads(line) for line in stream]
    for i,row in enumerate(rows):
        audit=row['control_audit']
        audit['native_before'][3]=i*.0001; audit['native'][3]=(i+1)*.0001
        audit['offset_before'][2]=-i*.0001; audit['applied_offset'][2]=-(i+1)*.0001
        audit['filtered_residual'][2]=-.01
    with gzip.open(rollout.trace_path,'wt') as stream:
        stream.write('\n'.join(json.dumps(row) for row in rows)+'\n')
    with pytest.raises(RuntimeError): call('audit_probe',rollout)


def test_completed_probe_with_ik_failure_is_not_a_passing_preflight(tmp_path):
    rollout=constructed_probe(tmp_path)
    rollout.row['error']="RuntimeError('UR10 IK target is unreachable')"
    with pytest.raises(RuntimeError): call('audit_probe',rollout)


@pytest.mark.parametrize('problem',[{'outcome':'safety'},{'error':'UR10 IK target is unreachable'}])
def test_zero_preflight_stops_on_physical_anomaly_before_next_layout(monkeypatch,problem):
    import residual_rl.pilot as pilot
    calls=[]
    def run(spec,mode):
        calls.append(spec)
        if len(calls)>1: raise AssertionError('second layout started after failed preflight')
        return SimpleNamespace(row=problem)
    monkeypatch.setattr(pilot,'require_shadow_equivalent',lambda row:{'zero_equivalent':True})
    plan=[dict(episode_id='first',seed=1),dict(episode_id='second',seed=2)]
    with pytest.raises(RuntimeError):
        pilot.zero_residual_preflight(SimpleNamespace(run=run),plan)
    assert len(calls)==1


@pytest.mark.parametrize('bad',['missing_audit','missing_record','clock','nan','fake_filter',
                              'missing_hold','applied_lie','yaw_jump','missing_stage','missing_tcp'])
def test_probe_rejects_missing_malformed_and_nonmonotonic_records(tmp_path,bad):
    import gzip
    rollout=constructed_probe(tmp_path)
    with gzip.open(rollout.trace_path,'rt') as stream: rows=[json.loads(line) for line in stream]
    if bad=='missing_audit': rows[120]['control_audit']=None
    elif bad=='missing_record': rows.pop(120)
    elif bad=='clock': rows[120]['t']=rows[119]['t']
    elif bad=='nan': rows[120]['command'][0]=float('nan')
    elif bad=='fake_filter': rows[120]['control_audit']['filtered_residual'][1]=0.
    elif bad=='missing_hold':
        for row in rows: row['probe_stage']='small_offset'
    elif bad=='applied_lie': rows[120]['control_audit']['applied'][0]+=1.
    elif bad=='yaw_jump': rows[120]['command'][3]+=1.
    elif bad=='missing_stage': rows[120].pop('probe_stage')
    elif bad=='missing_tcp': rows[120].pop('tcp')
    with gzip.open(rollout.trace_path,'wt') as stream:
        stream.write('\n'.join(json.dumps(row) for row in rows)+'\n')
    with pytest.raises(RuntimeError): call('audit_probe',rollout)


def test_probe_derives_executed_xy_and_yaw_clipping(tmp_path):
    result=call('audit_probe',constructed_probe(tmp_path,clipped=True))
    assert result['max_command_xy_m_s']==pytest.approx(np.hypot(.079,.001))
    assert result['max_command_yaw_rad_s']==pytest.approx(.3)


def test_probe_rejects_falsified_nominal_anchor(tmp_path):
    import gzip
    rollout=constructed_probe(tmp_path)
    with gzip.open(rollout.trace_path,'rt') as stream: rows=[json.loads(line) for line in stream]
    offset=np.array([.01,0.,0.,.02])
    for row in rows:
        audit=row['control_audit']; audit['composition_native']=audit['native']
        audit['native']=(np.asarray(audit['native'])-offset).tolist()
        audit['native_before']=(np.asarray(audit['start'])-offset).tolist()
    with gzip.open(rollout.trace_path,'wt') as stream:
        stream.write('\n'.join(json.dumps(row) for row in rows)+'\n')
    with pytest.raises(RuntimeError): call('audit_probe',rollout)


def numeric_checkpoint_context(tmp_path,monkeypatch):
    import torch
    from residual_rl import safety,storage
    store=storage.RunStore(tmp_path/'run',create=True)
    monkeypatch.setattr(safety.time,'monotonic',lambda:21545.)
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=8*2**30+48*2**20,
        output_bytes=4*2**30-48*2**20,rss=0,dangerous_memory=False))
    guard=safety.LiveGuard(store.root,0.)
    learner=SimpleNamespace(updates=3,checkpoint=lambda:dict(actor={'w':torch.ones(2)},
        actor_optimizer={},critics=[],target_critics=[],critic_optimizer={},rng_state=torch.get_rng_state()))
    replay=SimpleNamespace(state_dict=lambda: (_ for _ in ()).throw(AssertionError('huge replay accessed')))
    return store,learner,replay,guard


def test_resource_stop_can_save_bounded_compact_without_replay(tmp_path,monkeypatch):
    import torch
    from residual_rl.safety import BudgetStop
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    with pytest.raises(BudgetStop): guard.poll()
    path=call('checkpoint',store,learner,replay,guard,'interrupted',compact=True)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['replay_included'] is False and 'replay' not in saved
    assert {'learner','python_rng','numpy_rng','torch_rng'}<=saved.keys()
    evidence=json.loads(path.with_suffix('.json').read_text())
    assert evidence['kind']=='compact' and evidence['bytes']<32*2**20


def test_shutdown_hard_risk_reuses_verified_state_and_records_failure(tmp_path,monkeypatch):
    from residual_rl import safety
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    previous=call('checkpoint',store,learner,replay,guard,'previous',compact=True)
    previous_bytes=previous.read_bytes()
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=7*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    result=call('shutdown_save',store,learner,replay,guard,previous)
    assert result['status']=='previous_reused' and 'stop_disk' in result['error']
    assert previous.read_bytes()==previous_bytes
    assert json.loads((store.root/'save_result.json').read_text())==result
    previous.write_bytes(b'corrupt')
    result=call('shutdown_save',store,learner,replay,guard,previous)
    assert result['status']=='failed' and result['checkpoint'] is None


def test_shutdown_without_learner_explicitly_not_applicable(tmp_path,monkeypatch):
    store,_,_,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    result=call('shutdown_save',store,None,None,guard,None)
    assert result['status']=='not_applicable'


def test_failed_checkpoint_write_preserves_last_complete_full_checkpoint(tmp_path,monkeypatch):
    import torch
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    from residual_rl import safety
    monkeypatch.setattr(safety.time,'monotonic',lambda:1.)
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=20*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    replay=SimpleNamespace(state_dict=lambda:dict(episodes=[],rng={'seed':7}))
    previous=call('checkpoint',store,learner,replay,guard,'previous')
    from residual_rl.storage import unpack_replay
    assert unpack_replay(torch.load(previous,weights_only=False)['replay'])['rng']=={'seed':7}
    before=previous.read_bytes()
    def fail(payload,stream):
        stream.write(b'partial'); raise OSError('synthetic write failure')
    monkeypatch.setattr(torch,'save',fail)
    result=call('shutdown_save',store,learner,replay,guard,previous)
    assert result['status']=='previous_reused' and 'synthetic write failure' in result['error']
    assert previous.read_bytes()==before
    assert not list(previous.parent.glob('*.tmp'))


def load_entry():
    import importlib.util
    spec=importlib.util.spec_from_file_location('residual_entry',Path(__file__).parents[1]/'scripts/train_residual_sac.py')
    entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    return entry


def test_supervisor_soft_stop_waits_for_worker_save_handshake(tmp_path,monkeypatch):
    from residual_rl import safety,storage
    entry=load_entry(); now=[21540.]; signals=[]
    sample=dict(disk_free=20*2**30,output_bytes=0,rss=0,dangerous_memory=False)
    monkeypatch.setattr(safety,'resources',lambda root,pid=None:sample)
    monkeypatch.setattr(entry.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(entry.time,'sleep',lambda dt:now.__setitem__(0,now[0]+dt))
    monkeypatch.setattr(entry.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(entry.os,'killpg',lambda *a:pytest.fail('healthy cleanup must finish without killing group'))
    storage.write_json(tmp_path/'checkpoint_state.json',dict(initialized=False))
    class Worker:
        pid=123456; returncode=None
        def poll(self):
            if signals:
                assert (tmp_path/'stop_request.json').exists()
                storage.write_json(tmp_path/'save_result.json',dict(status='not_applicable',checkpoint=None,
                    evidence=None,learner_initialized=False,error=None))
                self.returncode=0
            return self.returncode
    result=entry.supervise(Worker(),tmp_path,0.)
    assert len(signals)==1 and signals[0]==(123456,entry.signal.SIGTERM)
    assert result['save']['status']=='not_applicable' and result['worker_acknowledged']
    assert now[0]<21600.


def test_supervisor_hard_caps_kill_only_owned_group_and_retain_verified_save(tmp_path,monkeypatch):
    from residual_rl import safety,storage
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    previous=call('checkpoint',store,learner,replay,guard,'before_risk',compact=True)
    storage.write_json(store.root/'checkpoint_state.json',dict(initialized=True))
    entry=load_entry(); groups=[]
    monkeypatch.setattr(safety,'resources',lambda root,pid=None:dict(disk_free=7*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    class Worker:
        pid=123456; returncode=None
        def poll(self): return self.returncode
        def wait(self,timeout): self.returncode=-9
    worker=Worker()
    monkeypatch.setattr(entry.os,'kill',lambda *a:pytest.fail('hard cap must not wait for new serialization'))
    monkeypatch.setattr(entry.os,'killpg',lambda pid,sig:groups.append((pid,sig)))
    result=entry.supervise(worker,store.root,0.)
    assert groups==[(worker.pid,entry.signal.SIGKILL)]
    assert not result['worker_acknowledged']
    assert result['save']['status']=='previous_reused'
    assert result['save']['checkpoint']==str(previous)


@pytest.mark.parametrize('risk',['time','output','disk','memory'])
def test_compact_checkpoint_refuses_hard_risk_without_serializing(tmp_path,monkeypatch,risk):
    import torch
    from residual_rl import safety
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    sample=dict(disk_free=20*2**30,output_bytes=0,rss=0,dangerous_memory=False)
    if risk=='time': monkeypatch.setattr(safety.time,'monotonic',lambda:21600.)
    if risk=='output': sample['output_bytes']=4*2**30
    if risk=='disk': sample['disk_free']=7*2**30
    if risk=='memory': sample['dangerous_memory']=True
    monkeypatch.setattr(safety,'resources',lambda root:sample)
    monkeypatch.setattr(torch,'save',lambda *a:pytest.fail('hard risk serialized checkpoint'))
    result=call('shutdown_save',store,learner,replay,guard,None)
    assert result['status']=='failed' and result['error']


def test_compact_checkpoint_write_is_bounded_and_cleans_partial(tmp_path,monkeypatch):
    import torch
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    def oversized(payload,stream): stream.write(bytes(32*2**20+1))
    monkeypatch.setattr(torch,'save',oversized)
    result=call('shutdown_save',store,learner,replay,guard,None)
    assert result['status']=='failed' and 'byte allowance' in result['error']
    assert not list((store.root/'checkpoints').iterdir())


def test_derived_report_failure_cannot_obscure_stop_and_save_evidence(tmp_path,monkeypatch):
    from residual_rl import storage
    store=storage.RunStore(tmp_path/'run',create=True)
    storage.write_json(store.root/'summary.json',dict(primary=True))
    storage.write_json(store.root/'status.json',dict(state='budget_stopped',save={'status':'failed'}))
    original=(store.root/'status.json').read_bytes()
    def report(root,rows):
        assert (root/'summary.json').is_file() and json.loads((root/'status.json').read_text())['save']['status']=='failed'
        raise OSError('derived report unavailable')
    monkeypatch.setitem(sys.modules,'residual_rl.reporting',SimpleNamespace(write_report=report))
    call('write_derived_report',store)
    assert (store.root/'status.json').read_bytes()==original
    assert 'derived report unavailable' in json.loads((store.root/'report_error.json').read_text())['error']


def test_supervisor_escalates_unresponsive_worker_before_six_hours(tmp_path,monkeypatch):
    from residual_rl import safety,storage
    entry=load_entry(); now=[21540.]; groups=[]; requests=[]
    monkeypatch.setattr(safety,'resources',lambda root,pid=None:dict(disk_free=20*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    monkeypatch.setattr(entry.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(entry.time,'sleep',lambda dt:now.__setitem__(0,now[0]+dt))
    monkeypatch.setattr(entry.os,'kill',lambda pid,sig:requests.append((pid,sig)))
    monkeypatch.setattr(entry.os,'killpg',lambda pid,sig:groups.append((pid,sig)))
    storage.write_json(tmp_path/'checkpoint_state.json',dict(initialized=False))
    class Worker:
        pid=123456; returncode=None
        def poll(self): return self.returncode
        def wait(self,timeout): self.returncode=-9
    worker=Worker(); result=entry.supervise(worker,tmp_path,0.)
    assert requests==[(worker.pid,entry.signal.SIGTERM)] and groups==[(worker.pid,entry.signal.SIGKILL)]
    assert result['forced'] and result['save']['status']=='not_applicable'
    assert now[0]==21595.


def test_checkpoint_does_not_delete_preexisting_orphan_temporary(tmp_path,monkeypatch):
    store,learner,replay,guard=numeric_checkpoint_context(tmp_path,monkeypatch)
    folder=store.root/'checkpoints'; folder.mkdir()
    orphan=folder/'interrupted.tmp'; orphan.write_bytes(b'old incomplete attempt')
    result=call('shutdown_save',store,learner,replay,guard,None)
    assert result['status']=='failed' and 'FileExistsError' in result['error']
    assert orphan.read_bytes()==b'old incomplete attempt'


def test_execute_startup_failure_records_save_before_calling_derived_report(tmp_path,monkeypatch):
    from residual_rl import storage
    store=storage.RunStore(tmp_path/'run',create=True)
    storage.write_json(store.root/'seed_schedule.json',{})
    # No seal exists: real verification stops before inference or runtime setup.
    reports=[]
    def report(root,rows):
        assert json.loads((root/'status.json').read_text())['save']['status']=='not_applicable'
        assert (root/'summary.json').is_file()
        reports.append(rows)
    monkeypatch.setitem(sys.modules,'residual_rl.reporting',SimpleNamespace(write_report=report))
    with pytest.raises(FileNotFoundError): call('execute',store)
    assert reports==[[]]
    assert json.loads((store.root/'save_result.json').read_text())['status']=='not_applicable'


def test_supervisor_request_write_failure_still_stops_only_owned_group(tmp_path,monkeypatch):
    from residual_rl import safety,storage
    entry=load_entry(); groups=[]
    monkeypatch.setattr(entry.time,'monotonic',lambda:21540.)
    monkeypatch.setattr(safety,'resources',lambda root,pid=None:dict(disk_free=20*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    native=storage.write_json
    def fail_request(path,value):
        if path.name=='stop_request.json': raise OSError('stop request write failed')
        return native(path,value)
    monkeypatch.setattr(storage,'write_json',fail_request)
    monkeypatch.setattr(entry.os,'killpg',lambda pid,sig:groups.append((pid,sig)))
    class Worker:
        pid=123456; returncode=None
        def poll(self): return self.returncode
        def wait(self,timeout): self.returncode=-9
    worker=Worker()
    with pytest.raises(OSError,match='stop request write failed'): entry.supervise(worker,tmp_path,0.)
    assert groups==[(worker.pid,entry.signal.SIGKILL)]
    evidence=json.loads((tmp_path/'supervisor_shutdown.json').read_text())
    assert evidence['save']['status']=='failed' and 'stop request write failed' in evidence['monitor_error']


def test_supervisor_sigterm_itself_requests_worker_cleanup(tmp_path,monkeypatch):
    from residual_rl import safety,storage
    entry=load_entry(); signals=[]; first=[True]
    monkeypatch.setattr(entry.time,'monotonic',lambda:10.)
    monkeypatch.setattr(entry.time,'sleep',lambda dt:None)
    def sample(root,pid=None):
        if first[0]:
            first[0]=False
            handler=entry.signal.getsignal(entry.signal.SIGTERM)
            assert callable(handler),'supervisor SIGTERM must request bounded cleanup'
            handler(entry.signal.SIGTERM,None)
        return dict(disk_free=20*2**30,output_bytes=0,rss=0,dangerous_memory=False)
    monkeypatch.setattr(safety,'resources',sample)
    monkeypatch.setattr(entry.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(entry.os,'killpg',lambda *a:pytest.fail('cooperative worker must acknowledge'))
    storage.write_json(tmp_path/'checkpoint_state.json',dict(initialized=False))
    class Worker:
        pid=123456; returncode=None
        def poll(self):
            if signals:
                storage.write_json(tmp_path/'save_result.json',dict(status='not_applicable',checkpoint=None,
                    evidence=None,learner_initialized=False,error=None))
                self.returncode=0
            return self.returncode
    result=entry.supervise(Worker(),tmp_path,0.)
    assert result['reason']=='signal_SIGTERM' and result['worker_acknowledged']


def test_wall_watchdog_does_not_signal_a_reaped_worker_group(tmp_path,monkeypatch):
    entry=load_entry()
    monkeypatch.setattr(entry.os,'killpg',lambda *a:pytest.fail('reaped PID/group may no longer be owned'))
    worker=SimpleNamespace(pid=123456,returncode=0,poll=lambda:0)
    def finish_then_alarm(*args,**kwargs):
        entry.signal.getsignal(entry.signal.SIGALRM)(entry.signal.SIGALRM,None)
    monkeypatch.setattr(entry,'_supervise',finish_then_alarm)
    with pytest.raises(TimeoutError,match='already exited'): entry.supervise(worker,tmp_path,entry.time.monotonic())
