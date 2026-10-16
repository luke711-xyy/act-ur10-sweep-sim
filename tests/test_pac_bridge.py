import importlib
import numpy as np
import pytest


def test_portable_hybrid_loader_has_no_historical_export_dependency():
    from pac_act.hybrid import load_trial4
    formal,module=load_trial4()
    assert module.TRIAL_SPEC.force_N==1.
    assert module.TRIAL_SPEC.seek==(10.,400.,10.)
    assert module.TRIAL_SPEC.maintain==(10.,800.,10.)
    assert 'hybrid_trial4_snapshot' in str(module.ROOT)
    from sim.config import load_config
    from pac_act.identity import PROJECT
    cfg=formal.candidate_config(load_config(str(PROJECT/'hybrid_trial4_snapshot/evaluation_config.yaml')),6)
    assert cfg.controller.desired_force==1.
    assert cfg.task.target_count==6


def test_contact_quota_crosses_warmup_without_ppo_credit():
    from pac_act.rewards import ContactReward
    tracker=ContactReward(2)
    mask=np.array([1,0,0,0,0,0],bool)
    assert tracker.observe(mask,.010,active=False)['reward']==0.
    tracker.observe(mask,.010,active=False)
    assert tracker.unique_count==1
    mask[:3]=True
    assert tracker.observe(mask,.02,active=True)['reward']==-30.
    assert tracker.observe(mask,.1,active=True)['reward']==0.
    assert tracker.finish('failure')==0.
    with pytest.raises(RuntimeError): tracker.finish('safety')


def test_short_contacts_reset_and_simultaneous_allocation_is_unbiased():
    from pac_act.rewards import ContactReward
    for n in range(1,7):
        r=ContactReward(n)
        r.observe(np.ones(6,bool),.01,active=True)
        r.observe(np.zeros(6,bool),.001,active=True)
        assert r.observe(np.ones(6,bool),.019,active=True)['reward']==0
        got=r.observe(np.ones(6,bool),.001,active=True)
        assert got['reward']==pytest.approx(60-(6-n)*120/n)
        assert got['quota_events']==n and got['excess_events']==6-n


def test_z_predictions_never_affect_integrated_targets():
    from pac_act.hybrid import integrate_xy_yaw
    base=np.array([.1,.2,.001,.4]); a=np.zeros((5,4),np.float32); a[:,0]=.001
    b=a.copy(); b[:,2]=999.
    one=integrate_xy_yaw(base,a); two=integrate_xy_yaw(base,b)
    np.testing.assert_array_equal(one,two)
    np.testing.assert_allclose(one[:,0],[.101,.102,.103,.104,.105],atol=1e-7)
    np.testing.assert_array_equal(one[:,2],np.full(5,.001,dtype=np.float32))


def test_sync_clock_queries_at_first_frame_and_every_five_without_delay():
    from pac_act.hybrid import BlockClock
    clock=BlockClock()
    assert clock.query_due
    for i in range(1,51):
        clock.executed()
        assert clock.query_due==(i%5==0)
    assert clock.learned_active
    assert clock.active_actions==0
    clock.executed()
    assert clock.active_actions==1


def test_guard_refuses_existing_training_without_stopping_it(monkeypatch):
    from pac_act.lifecycle import require_local_slot
    import pac_act.safety as safety
    monkeypatch.setattr(safety,'local_inference_available',lambda:(False,{'pid':123,'command':['train_residual_long.py']}))
    with pytest.raises(RuntimeError,match='123'): require_local_slot()
