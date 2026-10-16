"""Numerical contracts for the new residual layer; no ACT/model/physics."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def api(name):
    try:
        return importlib.import_module('residual_rl.' + name)
    except ModuleNotFoundError:
        return SimpleNamespace()


def construct(module, name, *args, **kwargs):
    constructor = getattr(api(module), name, None)
    assert callable(constructor), f'{module}.{name} is not implemented'
    return constructor(*args, **kwargs)


@pytest.mark.parametrize('target,positive,negative', [(1,60,-120),(2,30,-60),
    (3,20,-40),(4,15,-30),(5,12,-24),(6,10,-20)])
def test_first_contacts_earn_quota_then_double_penalty(target, positive, negative):
    reward = construct('rewards','ContactReward',target)
    total = 0.
    for component in range(6):
        mask = np.zeros(6,dtype=bool); mask[component] = True
        event = reward.observe(mask,.020,active=True)
        assert event['reward'] == positive if component < target else event['reward'] == negative
        total += event['reward']
        reward.observe(np.zeros(6,dtype=bool),.001,active=True)
    assert total == pytest.approx(60 + max(0,6-target)*negative)


def test_short_contacts_reset_dwell_and_never_earn_duplicate_credit():
    reward = construct('rewards','ContactReward',1)
    mask=np.array([True,False,False,False,False,False])
    assert reward.observe(mask,.019,active=True)['reward'] == 0
    reward.observe(np.zeros(6,dtype=bool),.001,active=True)
    assert reward.observe(mask,.001,active=True)['reward'] == 0
    assert reward.observe(mask,.019,active=True)['reward'] == 60
    reward.observe(np.zeros(6,dtype=bool),.001,active=True)
    assert reward.observe(mask,.1,active=True)['reward'] == 0


def test_prefix_consumes_quota_without_retrospective_rl_credit():
    reward=construct('rewards','ContactReward',1)
    event=reward.observe(np.array([True,False,False,False,False,False]),.02,active=False)
    assert event['event_reward'] == 60 and event['reward'] == 0
    assert reward.observe(np.ones(6,dtype=bool),.02,active=True)['reward'] == -600
    assert reward.unique_count == 6


def test_simultaneous_contacts_are_identity_order_independent():
    for mask in (np.array([1,1,1,0,0,0],bool),np.array([0,0,0,1,1,1],bool)):
        reward=construct('rewards','ContactReward',2)
        event=reward.observe(mask,.02,active=True)
        assert event['reward'] == 0 and event['quota_events'] == 2 and event['excess_events'] == 1


@pytest.mark.parametrize('outcome,expected',[('success',0),('failure',0),('safety',-60)])
def test_terminal_is_once_and_mutually_exclusive(outcome,expected):
    reward=construct('rewards','ContactReward',1)
    assert reward.finish(outcome) == expected
    with pytest.raises(RuntimeError): reward.finish('failure')
    with pytest.raises(RuntimeError): reward.observe(np.zeros(6,bool),.01,active=True)


@pytest.mark.parametrize('outcome', ['success','failure'])
def test_same_contacts_have_same_return_regardless_of_task_success(outcome):
    reward=construct('rewards','ContactReward',2)
    event=reward.observe(np.array([True,False,False,False,False,False]),.020,active=True)
    assert event['reward']+reward.finish(outcome)==30.


def test_safety_penalty_is_still_added_to_contact_return():
    reward=construct('rewards','ContactReward',1)
    event=reward.observe(np.array([True,False,False,False,False,False]),.020,active=True)
    assert event['reward']+reward.finish('safety')==0.


def test_infrastructure_is_not_scored_as_policy_failure():
    classify=getattr(api('rewards'),'classify_failure',None)
    assert callable(classify)
    assert classify(ValueError('UR10 IK position residual=0.01 orientation residual=0.1')) == 'failure'
    assert classify(RuntimeError('unsafe robot collision')) == 'safety'
    assert classify(TypeError('ndarray is not JSON serializable')) is None
    assert classify(RuntimeError('prediction failed')) is None


def governor():
    return construct('control','ResidualGovernor',bounds=np.array([[-1.,1.],[-1.,1.]]))


def test_zero_residual_is_bit_exact_and_z_never_changes():
    control=governor(); start=np.array([0.,0.,.001,.2],np.float32)
    native=np.array([.0008,0.,-.00021,.2035],np.float32)
    out=control.step(start,native,np.zeros(3),dt=.01,active=True)
    assert out.dtype == native.dtype and np.array_equal(out,native)
    out=control.step(native,native,np.ones(3),dt=.01,active=False)
    assert np.array_equal(out,native)


def test_pose_action_opposing_act_never_brakes_to_zero_or_reverses():
    control=governor(); current=np.array([0.,0.,.001,0.],np.float32)
    velocities=[]; residuals=[]; previous_native=current.copy()
    for _ in range(100):
        native=previous_native+np.array([.0008,0.,.000001,.0035],np.float32)
        out=control.step(current,native,np.array([-1.,0.,-1.]),dt=.01,active=True,native_before=previous_native)
        velocities.append((out[0]-current[0])/.01); residuals.append(control.velocity.copy())
        assert out[2] == native[2]
        assert np.linalg.norm((out[:2]-current[:2])/.01) <= .080001
        current=out; previous_native=native
        assert np.linalg.norm(control.offset[:2])<=.0050001
    assert min(velocities)>.0749
    assert np.max(np.linalg.norm(residuals,axis=1))<.053


def test_workspace_safety_never_clamps_z_or_permits_outside_xy():
    control=governor(); start=np.array([1.,0.,.001,0.],np.float32)
    native=start.copy(); native[2]=-.0001
    out=control.step(start,native,np.array([1.,0.,0.]),dt=.01,active=True)
    assert out[0] <= 1 and out[2] == native[2]
    with pytest.raises(ValueError): control.step(start,native,np.array([np.nan,0.,0.]),dt=.01,active=True)
