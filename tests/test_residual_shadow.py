"""Exact same-stream counterfactual checks; no extra policy/physical rollout."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def api(module, name, *args, **kwargs):
    try: target=importlib.import_module('residual_rl.'+module)
    except ModuleNotFoundError: target=None
    function=getattr(target,name,None)
    assert callable(function), f'{module}.{name} missing'
    return function(*args,**kwargs)


def test_production_reference_composition_retains_zero_and_nonzero_route_behavior():
    from residual_rl.control import ResidualGovernor
    composer=api('control','ResidualReference',ResidualGovernor([[-1,1],[-1,1]]))
    start=np.array([0.,0.,.001,.1],np.float32)
    composer.begin_frame(start)
    raw=start+np.array([.0008,0.,.000001,.0035],np.float32)
    output=composer.step(raw,np.zeros(3),dt=.01,active=True)
    assert output.dtype==raw.dtype and output.tobytes()==raw.tobytes()
    native_before=raw.copy()
    raw2=raw+np.array([.0008,0.,.000001,.0035],np.float32)
    output2=composer.step(raw2,np.array([-1.,0.,-1.]),dt=.01,active=True)
    assert output2[0]==pytest.approx(float(output[0])+.00078,abs=1e-9)
    assert output2[3]==pytest.approx(float(output[3])+.00345,abs=2e-8)
    assert output2[2]==raw2[2] and np.array_equal(composer.control.last_audit['native_before'],native_before)


def test_shadow_is_exact_and_uses_only_isolated_filter_state():
    shadow=api('shadow','ZeroResidualShadow',[[-1,1],[-1,1]])
    start=np.array([0.,0.,.001,.1],np.float32)
    shadow.begin_frame(start)
    for i in range(4):
        raw=start+np.array([.0002*(i+1),0.,-.00001*(i+1),.0005*(i+1)],np.float32)
        before=raw.copy()
        checked=shadow.check(raw,raw,dt=.01,active=True)
        assert np.array_equal(raw,before)
        assert checked['exact'] and checked['active'] and checked['max_abs_error']==0.
    report=shadow.summary()
    assert report['checked_steps']==report['active_checked_steps']==4
    assert report['native_sha256']==report['zero_sha256']
    assert report['extra_policy_queries']==report['extra_physics_steps']==0


@pytest.mark.parametrize('axis',[0,1,2,3])
def test_shadow_stops_on_even_one_ulp_difference_including_z(axis):
    shadow=api('shadow','ZeroResidualShadow',[[-1,1],[-1,1]])
    raw=np.array([.1,.2,.001,.3],np.float32)
    shadow.begin_frame(raw)
    actual=raw.copy(); actual[axis]=np.nextafter(actual[axis],np.float32(1.))
    with pytest.raises(RuntimeError,match='same-stream zero residual mismatch'):
        shadow.check(actual,raw,dt=.01,active=True)
    assert shadow.summary()['mismatches']==1


def test_shadow_handoff_bypass_preserves_exact_reference_and_requires_no_physics():
    shadow=api('shadow','ZeroResidualShadow',[[-1,1],[-1,1]])
    start=np.zeros(4,np.float32); shadow.begin_frame(start)
    actual=np.array([.0002,0.,.001,-.001],np.float32)
    audit=shadow.check(actual,actual,dt=.01,active=False)
    assert audit['exact'] and shadow.summary()['active_checked_steps']==0
    assert not shadow.composer.control.velocity.any()


def valid_row():
    import hashlib
    digest=hashlib.sha256(b'actual same-stream reference fixture').hexdigest()
    return dict(accepted_ACT_chunks=4,residual_frames=10,control_steps=200,ACT_control_steps=100,
        physical_attempts=1,zero_residual_shadow=dict(single_actual_ACT_stream=True,
            checked_steps=100,active_checked_steps=40,mismatches=0,max_abs_error=0.,
            native_sha256=digest,zero_sha256=digest,extra_policy_queries=0,extra_physics_steps=0))


def test_same_stream_gate_does_not_require_a_second_asynchronous_rollout():
    row=valid_row()
    result=api('pilot','require_shadow_equivalent',row)
    assert result['zero_equivalent'] and result['single_actual_ACT_stream']


@pytest.mark.parametrize('change', ['empty','inactive','mismatch','hash','queries','physics',
                                  'coverage','missing_coverage','too_many','digest_format','different_stream'])
def test_same_stream_gate_rejects_missing_weak_or_incomplete_proof(change):
    row=valid_row(); proof=row['zero_residual_shadow']
    if change=='empty':row['accepted_ACT_chunks']=0
    elif change=='inactive':proof['active_checked_steps']=0
    elif change=='mismatch':proof['mismatches']=1
    elif change=='hash':proof['zero_sha256']='f'*64
    elif change=='queries':proof['extra_policy_queries']=1
    elif change=='physics':proof['extra_physics_steps']=1
    elif change=='coverage':row['ACT_control_steps']=101
    elif change=='missing_coverage':del row['ACT_control_steps']
    elif change=='too_many':row['control_steps']=99
    elif change=='digest_format':proof['native_sha256']=proof['zero_sha256']='not-a-digest'
    else:proof['single_actual_ACT_stream']=False
    with pytest.raises(RuntimeError):api('pilot','require_shadow_equivalent',row)


def test_orchestrator_executes_each_new_preflight_layout_once():
    calls=[]
    class Engine:
        def run(self,item,mode):
            calls.append((item,mode));return SimpleNamespace(row=valid_row())
    plan=[dict(episode_id='one',seed=3152000,target_count=1),dict(episode_id='six',seed=3152001,target_count=6)]
    checks=api('pilot','zero_residual_preflight',Engine(),plan)
    assert len(checks)==2 and len(calls)==2
    assert [mode for _,mode in calls]==['native','native']
    assert all(item['phase']=='preflight_shadow' for item,_ in calls)


def test_new_schedule_avoids_both_prior_invocations():
    from residual_rl.pilot import seed_schedule
    seeds={s['seed'] for s in seed_schedule()['preflight']}
    assert not seeds & {3150000,3150001,3150100,3150101,3151000,3151001,3151100,3151101}
