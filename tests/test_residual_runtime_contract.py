"""Bridge, resource and persistence contracts; no ACT rollout required."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def call(module,name,*args,**kwargs):
    try: api=importlib.import_module('residual_rl.'+module)
    except ModuleNotFoundError: api=SimpleNamespace()
    value=getattr(api,name,None)
    assert callable(value), f'{module}.{name} missing'
    return value(*args,**kwargs)


def test_features_are_unavailable_until_exact_chunk_is_accepted():
    box=call('features','FeatureMailbox')
    tokens=np.arange(25*512,dtype=np.float32).reshape(25,512)
    box.publish(tokens)
    assert box.current(0.)['available'] is False
    box.accept(issued_at=0.,accepted=False,length=25)
    assert box.current(.2)['available'] is False
    box.publish(tokens); box.accept(issued_at=.2,accepted=True,length=25)
    box.select(.2,5,.4)
    view=box.current(.4)
    assert view['index']==5 and view['age']==pytest.approx(.2)
    assert np.array_equal(view['feature'],tokens[5])
    with pytest.raises(RuntimeError): box.select(.1,0,.4)


def test_missing_or_bad_feature_binding_fails_closed():
    box=call('features','FeatureMailbox')
    with pytest.raises(RuntimeError): box.accept(issued_at=0,accepted=True,length=25)
    with pytest.raises(ValueError): box.publish(np.zeros((25,511)))
    box.publish(np.zeros((25,512),np.float32))
    with pytest.raises(RuntimeError): box.accept(issued_at=0,accepted=True,length=24)


def test_actor_boundary_rejects_privileged_reward_and_has_explicit_missing_features():
    kwargs=dict(robot_state=np.zeros(36),feature=np.ones(512),target_count=3,
        elapsed=4.,feature_age=0.,chunk_index=0,feature_available=False,active=False,
        nominal_velocity=np.zeros(3),previous_action=np.zeros(3),residual_velocity=np.zeros(3),residual_offset=np.zeros(3))
    obs=call('observations','actor_observation',**kwargs)
    assert obs.shape==(572,) and not obs[36:548].any()
    for forbidden in ('reward','collected','object_positions','seen'):
        with pytest.raises(TypeError): call('observations','actor_observation',**kwargs,**{forbidden:1})


def test_budget_limits_cover_startup_and_keep_final_validation_inside_total():
    guard=call('safety','BudgetGuard',started_at=100.)
    assert guard.check(now=100.,residual_steps=0,disk_free=20*2**30,output_bytes=0,rss=0)=='continue'
    assert guard.check(now=18100.,residual_steps=0,disk_free=20*2**30,output_bytes=0,rss=0)=='stop_collection'
    assert guard.check(now=21700.,residual_steps=0,disk_free=20*2**30,output_bytes=0,rss=0)=='stop_total'
    guard=call('safety','BudgetGuard',started_at=0.)
    assert guard.check(now=1.,residual_steps=50000,disk_free=20*2**30,output_bytes=0,rss=0)=='stop_collection'


def test_resource_limits_are_sustained_for_memory_but_immediate_for_disk():
    guard=call('safety','BudgetGuard',started_at=0.)
    sample=dict(residual_steps=0,disk_free=20*2**30,output_bytes=0,rss=11*2**30)
    assert guard.check(now=0.,**sample)=='continue'
    assert guard.check(now=19.,**sample)=='continue'
    assert guard.check(now=20.,**sample)=='stop_reserve_memory'
    assert guard.check(now=30.,**sample)=='stop_memory'
    sample.update(rss=0,disk_free=7*2**30)
    assert guard.check(now=31.,**sample)=='stop_disk'
    sample.update(disk_free=20*2**30,output_bytes=4*2**30)
    assert guard.check(now=32.,**sample)=='stop_output'


def test_ledger_retains_real_failure_and_refuses_repeat_or_orphan_attempt(tmp_path):
    ledger=call('storage','RunStore',tmp_path/'new_run',create=True)
    spec=dict(episode_id='train_0',seed=3130000,target_count=1,phase='train')
    ledger.claim(spec)
    with pytest.raises(FileExistsError): ledger.claim(spec)
    with pytest.raises(RuntimeError): ledger.require_clean()
    ledger.commit(dict(**spec,success=False,final_collected=2,opportunities=1,physical_attempts=1))
    assert ledger.rows()[0]['final_collected']==2 and ledger.rows()[0]['success'] is False
    ledger.require_clean()
    with pytest.raises(RuntimeError): ledger.commit(dict(**spec,success=True,final_collected=1))
    with pytest.raises(FileExistsError): call('storage','RunStore',tmp_path/'new_run',create=True)


def test_unknown_quantities_are_preserved_as_null_not_zero(tmp_path):
    ledger=call('storage','RunStore',tmp_path/'run',create=True)
    spec=dict(episode_id='x',seed=1,target_count=6,phase='preflight')
    ledger.claim(spec)
    ledger.commit(dict(**spec,success=False,final_collected=None,opportunities=1,physical_attempts=1))
    assert ledger.rows()[0]['final_collected'] is None


def test_budget_truncation_is_not_a_fabricated_policy_failure(tmp_path):
    ledger=call('storage','RunStore',tmp_path/'run',create=True)
    spec=dict(episode_id='budget',seed=1,target_count=1,phase='train')
    ledger.claim(spec)
    ledger.commit(dict(**spec,success=None,outcome='budget_truncated',final_collected=2,
                       opportunities=1,physical_attempts=1))
    assert ledger.rows()[0]['success'] is None


def test_contact_probe_checks_signed_distance_and_component_identity():
    env=SimpleNamespace(data=SimpleNamespace(ncon=3,contact=[
        SimpleNamespace(geom1=1,geom2=10,dist=.0001),
        SimpleNamespace(geom1=2,geom2=11,dist=-.0001),
        SimpleNamespace(geom1=4,geom2=12,dist=-.0001)]),
        ee=SimpleNamespace(tip_geom_ids=[1,2]),geom_to_component={10:0,11:1,12:2})
    mask=call('bridge','contact_mask',env)
    assert mask.tolist()==[False,True,False,False,False,False]


def test_substep_instrumentation_observes_each_physics_step_without_global_patch():
    import mujoco
    class TinyEnv:
        def step_control(self,command):
            import mujoco
            for _ in range(3):
                mujoco.mj_step(self.model,self.data)
    model=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.001"/><worldbody/></mujoco>')
    env=TinyEnv(); env.model=model; env.data=mujoco.MjData(model)
    substeps=[]; controls=[]; before=mujoco.mj_step
    call('bridge','instrument_physics',env,
         lambda e:substeps.append(float(e.data.time)),
         lambda e,c:controls.append(float(e.data.time)))
    env.step_control(None)
    assert substeps==pytest.approx([.001,.002,.003]) and controls==pytest.approx([.003])
    assert mujoco.mj_step is before


def test_streaming_adapter_does_not_retain_training_rgb():
    sink=call('bridge','DiscardObservations')
    for _ in range(10): sink.append({'image':np.ones((128,128,3))})
    assert len(sink)==0


def test_layout_registry_rejects_old_seeds_and_pose_hashes_without_retry():
    registry=call('runtime','LayoutRegistry',[dict(seed=2120000,layout_fingerprint='old')])
    with pytest.raises(ValueError): registry.register(2120000,'new','train')
    with pytest.raises(ValueError): registry.register(3130000,'old','train')
    registry.register(3140000,'val','validation_base')
    with pytest.raises(ValueError): registry.register(3130001,'val','train')
    registry.register(3140000,'val','validation_final')
    with pytest.raises(ValueError): registry.register(3140000,'val','validation_final')
    registry.register(3150000,'probe','preflight_native')
    registry.register(3150000,'probe','preflight_zero')


@pytest.mark.parametrize('first,second',[
    ('validation_base','validation_final'),('preflight_native','preflight_zero'),
    ('train','train'),('probe','validation_base')])
def test_layout_seed_cannot_change_initial_pose_hash(first,second):
    registry=call('runtime','LayoutRegistry',[])
    registry.register(123,'pose_a',first)
    with pytest.raises(ValueError,match='seed'):
        registry.register(123,'pose_b',second)
    # A rejected registration must not consume the legitimate second arm.
    if first in {'validation_base','preflight_native'}:
        registry.register(123,'pose_a',second)


def test_resource_stop_leaves_checkpoint_reserves_before_hard_caps():
    guard=call('safety','BudgetGuard',started_at=0.)
    sample=dict(residual_steps=0,disk_free=20*2**30,output_bytes=0,rss=0)
    assert guard.check(now=21540.,**sample)=='stop_reserve_total'
    assert guard.check(now=10.,**dict(sample,disk_free=8*2**30+64*2**20))=='stop_reserve_disk'
    assert guard.check(now=10.,**dict(sample,output_bytes=4*2**30-64*2**20))=='stop_reserve_output'


def test_worker_sigterm_requests_cooperative_stop_and_restores_handler(tmp_path):
    import signal
    from residual_rl import safety
    previous=signal.getsignal(signal.SIGTERM)
    with safety.worker_stop_signals() as request:
        signal.raise_signal(signal.SIGTERM)
        assert request.reason=='signal_SIGTERM'
        guard=safety.LiveGuard(tmp_path,0.,stop_request=request)
        with pytest.raises(safety.BudgetStop,match='signal_SIGTERM'): guard.poll()
    assert signal.getsignal(signal.SIGTERM)==previous


def test_actor_time_budget_excludes_post_reset_settling_time(monkeypatch,tmp_path):
    from residual_rl import runtime
    from sim.act.interface import ACTObservationBuilder
    from sim.config import load_config
    monkeypatch.setattr(ACTObservationBuilder,'observe',lambda *a,**k:{'observation.state':np.zeros(36)})
    cfg=load_config()
    env=SimpleNamespace(time=1.25,data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1)),
        collected_mask=lambda:np.zeros(6))
    collector=runtime.Collector(env,cfg,dict(target_count=1,episode_id='time'),
        'zero',None,SimpleNamespace(),tmp_path,call('features','FeatureMailbox'))
    try:
        first,_=collector.observations()
        assert first[558]==pytest.approx(1.)
        env.time=21.25
        later,_=collector.observations()
        assert later[558]==pytest.approx(.75)
        assert env.time==21.25
    finally: collector.trace.close()


def test_training_decision_stop_cannot_hide_resource_stop(monkeypatch,tmp_path):
    from residual_rl import safety
    monkeypatch.setattr(safety.time,'monotonic',lambda:10.)
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=7*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    guard=safety.LiveGuard(tmp_path,0.); guard.phase='train'; guard.training_decision_stop=0
    with pytest.raises(safety.BudgetStop,match='stop_disk'): guard.poll()


@pytest.mark.parametrize('administrative',[False,True])
def test_episode_numeric_experience_is_atomic_and_durable_before_ledger(tmp_path,monkeypatch,administrative):
    from residual_rl import safety,storage
    from residual_rl.learning import Episode
    monkeypatch.setattr(safety.time,'monotonic',lambda:10.)
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=20*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    store=storage.RunStore(tmp_path/'run',create=True); guard=safety.LiveGuard(store.root,0.)
    spec=dict(episode_id='numeric',target_count=1,seed=91,phase='train'); store.claim(spec)
    episode=Episode('numeric',1,np.arange(12,dtype=np.float32).reshape(3,4),
        np.ones((3,5),np.float32),np.array([[.2,-.3,.1],[-.4,.5,0.]],np.float32),
        np.array([0.,-20.],np.float32),np.array([False,not administrative]),np.array([False,True]))
    row=dict(spec,success=None if administrative else False,
        outcome='budget_truncated' if administrative else 'failure',final_collected=2,
        opportunities=1,physical_attempts=1)
    native_commit=store.commit
    def commit_after_durable(value):
        evidence=value['experience']; path=store.root/'episodes'/'numeric'/'experience.npz'
        assert path.is_file() and storage.sha256(path)==evidence['sha256']
        assert evidence['complete'] is True and evidence['administrative_truncation'] is administrative
        with np.load(path,allow_pickle=False) as saved:
            assert set(saved.files)=={'actor_obs','critic_obs','actions','rewards','terminated','learn_mask'}
            for name in saved.files:
                assert np.array_equal(saved[name],getattr(episode,name))
                assert saved[name].dtype==(np.bool_ if name in {'terminated','learn_mask'} else np.float32)
        native_commit(value)
    monkeypatch.setattr(store,'commit',commit_after_durable)
    store.commit_episode(row,episode,guard)
    assert len(store.rows())==1
    assert not list((store.root/'episodes'/'numeric').glob('*.tmp'))


def test_experience_write_failure_never_commits_episode_or_publishes_partial(tmp_path,monkeypatch):
    from residual_rl import safety,storage
    from residual_rl.learning import Episode
    monkeypatch.setattr(safety.time,'monotonic',lambda:10.)
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=20*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    store=storage.RunStore(tmp_path/'run',create=True); guard=safety.LiveGuard(store.root,0.)
    spec=dict(episode_id='numeric',target_count=1,seed=91,phase='train'); store.claim(spec)
    episode=Episode('numeric',1,np.zeros((2,4),np.float32),np.zeros((2,5),np.float32),
        np.zeros((1,3),np.float32),np.zeros(1,np.float32),np.ones(1,bool),np.ones(1,bool))
    def fail(stream,**arrays): stream.write(b'partial'); raise OSError('npz write failed')
    monkeypatch.setattr(np,'savez_compressed',fail)
    with pytest.raises(OSError,match='npz write failed'):
        store.commit_episode(dict(spec,success=False,outcome='failure',final_collected=0,
            opportunities=1,physical_attempts=1),episode,guard)
    assert store.rows()==[] and not list((store.root/'episodes'/'numeric').iterdir())


def test_interrupted_collector_retains_only_available_incomplete_numeric_context(tmp_path,monkeypatch):
    from residual_rl import runtime,safety,storage
    collector=object.__new__(runtime.Collector)
    collector.spec=dict(episode_id='partial',target_count=2)
    collector.actor_rows=[np.arange(4,dtype=np.float32)]
    collector.critic_rows=[np.arange(5,dtype=np.float32)]
    collector.actions=[np.array([.2,.3,-.1],np.float32)]
    collector.rewards=[0.]; collector.masks=[True]
    collector.trace=SimpleNamespace(path=tmp_path/'physical_trace.jsonl.gz')
    collector.guard=safety.LiveGuard(tmp_path,0.)
    monkeypatch.setattr(safety.time,'monotonic',lambda:10.)
    monkeypatch.setattr(safety,'resources',lambda root:dict(disk_free=20*2**30,
        output_bytes=0,rss=0,dangerous_memory=False))
    evidence=collector.preserve_partial('unknown infrastructure failure')
    assert evidence['complete'] is False and evidence['valid_episode'] is False
    with np.load(tmp_path/'partial_experience.npz',allow_pickle=False) as saved:
        assert saved['actor_obs'].shape==(1,4) and saved['actions'][0,0]==pytest.approx(.2)
        assert 'terminated' not in saved.files
    assert not (tmp_path/'result.json').exists() and not (tmp_path/'episodes.jsonl').exists()
    meta=json.loads((tmp_path/'partial_experience.json').read_text())
    assert meta['reason']=='unknown infrastructure failure'
