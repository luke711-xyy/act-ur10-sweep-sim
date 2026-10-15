"""Parameter search contracts; no policy load, simulation reset, or training."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'hybrid_astar_act_gate1mm_acc50_2s_goal1_20261014'
PROJECT=Path('/Users/huazhi_luke/.codex/worktrees/act-five-view')
sys.path.insert(0,str(PROJECT))

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def phases():
    path=ROOT/'phases.py' if (ROOT/'phases.py').exists() else BASE/'phases.py'
    return load(path,'actual_tuning_phases')

def spec_for(module,force=1.,seek=(25.,1200.,1.),maintain=(25.,2000.,1.)):
    cls=getattr(module,'AdmittanceSpec',None)
    return cls(force,seek,maintain) if cls else SimpleNamespace(force_N=force,seek=seek,maintain=maintain)

def controller(module,force=1.,seek=(25.,1200.,1.),maintain=(25.,2000.,1.)):
    spec=spec_for(module,force,seek,maintain)
    if hasattr(module,'AdmittanceSpec'):return module.TwoStageAdmittance(.01,spec)
    return module.TwoStageAdmittance(.01)

def runner():
    sys.modules['phases']=phases()
    path=ROOT/'runner.py' if (ROOT/'runner.py').exists() else BASE/'runner.py'
    return load(path,'actual_tuning_runner')

def test_no_contact_cannot_integrate_beyond_two_mm():
    module=phases();loop=controller(module)
    for _ in range(5000):value=loop.step(1.,0.)
    assert value==pytest.approx(-.002,abs=1e-12)

@pytest.mark.parametrize('force',[.2,.3])
def test_lower_target_force_is_executed_not_silently_one_newton(force):
    module=phases();loop=controller(module,force)
    try:correction=loop.step(force,0.)
    except ValueError as error:pytest.fail('approved lower setpoint was rejected: '+str(error))
    assert correction==pytest.approx(-force*.000004,abs=1e-12)

@pytest.mark.parametrize('force,threshold,low,high',[(.2,.1,.16,.24),(.3,.15,.24,.36),(1.,.5,.8,1.2)])
def test_wrist_dwell_and_ready_band_follow_the_target(force,threshold,low,high):
    module=phases();loop=controller(module,force)
    for i in range(12):loop.observe(i*.01,threshold-.0001)
    assert loop.phase=='seek'
    for i in range(12,23):loop.observe(i*.01,threshold)
    assert loop.phase=='settle'
    for i in range(23,43):loop.observe(i*.01,force)
    assert not loop.ready
    loop.observe(.43,force)
    assert loop.ready

def test_phase_switch_keeps_correction_and_velocity_with_different_gains():
    module=phases();loop=controller(module,.3,(10.,400.,10.),(10.,800.,10.))
    try:loop.step(.3,0.)
    except ValueError as error:pytest.fail(str(error))
    before=(loop.delta_z,loop.delta_z_rate)
    for i in range(11):loop.observe(i*.01,.15)
    assert loop.phase=='settle'
    assert (loop.params.m,loop.params.b,loop.params.k)==(10.,800.,10.)
    assert (loop.delta_z,loop.delta_z_rate)==before

def test_total_two_mm_bound_includes_actual_z_compensation():
    module=phases();loop=controller(module);loop.reset(delta_z=-.00195)
    limiter=module.ZCommandLimiter(.001-.00195,.01)
    command=module.compose_z(.001,limiter.last_z+.001,1.,-.005,loop,limiter,None)
    assert command>=-.001-1e-12
    assert command==pytest.approx(-.001,abs=1e-12)

def test_native_emergency_unloading_rate_is_unchanged():
    module=phases();limiter=module.ZCommandLimiter(0.,.01)
    assert limiter.apply(.1,10.)==pytest.approx(.000201,abs=1e-12)
    assert limiter.apply(.1,1.)==pytest.approx(.0002,abs=1e-12)

@pytest.mark.parametrize('gains,want', [((10.,800.,10.),-.001),((10.,800.,20.),-.0005)])
def test_spring_gain_changes_equilibrium_not_the_hard_bound(gains,want):
    module=phases();loop=controller(module,.3,gains,gains)
    assert loop.loop.steady_state_delta(.3,.29)==pytest.approx(want,abs=1e-12)

@pytest.mark.parametrize('force',[1.,.3,.2])
def test_real_config_receives_force_gains_and_two_mm_without_safety_drift(force):
    module=runner();phase_module=sys.modules['phases']
    module.TRIAL_SPEC=spec_for(phase_module,force)
    from sim.config import load_config
    frozen=load_config(str(module.BATCH/'evaluation_config.yaml'))
    actual=module.make_config(frozen,1)
    assert actual.controller.desired_force==force
    assert actual.controller.delta_z_limit==.002
    assert actual.controller.safe_max_force==20.
    assert actual.controller.safe_force_dwell_steps==3
    assert actual.controller.force_sensor=='wrist_ft'
    assert actual.controller.delta_z_upward_rate_limit==.02
    assert actual.controller.delta_z_downward_rate_limit==.1

def test_six_unique_trial_claims_prevent_retry_or_seventh_attempt(tmp_path):
    module=runner();claim=getattr(module,'claim_trial',None)
    assert callable(claim),'new runner must enforce the approved six-trial budget'
    for i in range(1,7):claim(tmp_path,i)
    with pytest.raises((ValueError,FileExistsError)):claim(tmp_path,1)
    with pytest.raises(ValueError):claim(tmp_path,7)

def test_wrong_setpoint_or_nonfinite_wrist_is_rejected():
    module=phases();loop=controller(module,.3)
    with pytest.raises(ValueError):loop.step(.2,1.)
    with pytest.raises(ValueError):loop.observe(0.,float('nan'))

def test_unstable_candidate_is_rejected_before_claim_or_expensive_execution(tmp_path,monkeypatch):
    from contextlib import nullcontext
    module=runner();phase=sys.modules['phases']
    module.TRIAL_SPEC=phase.AdmittanceSpec(.3,(1.,4000.,10.),(10.,800.,10.))
    module.TRIAL_NUMBER=1
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'formal_guard',nullcontext)
    monkeypatch.setattr(module,'_run_one_excluded',lambda target:pytest.fail('unstable candidate reached physical execution'))
    with pytest.raises(ValueError,match='unstable'):module.run_one(1)
    assert not list(tmp_path.iterdir())

def test_total_upper_two_mm_bound_includes_emergency_relief():
    from sim.controllers.admittance import OverforceReliefController1D,OverforceReliefParams
    module=phases();loop=controller(module);loop.reset(delta_z=.00199)
    limiter=module.ZCommandLimiter(.001+.00199,.01)
    relief=OverforceReliefController1D(OverforceReliefParams(),.01)
    command=module.compose_z(.001,limiter.last_z,10.,-.005,loop,limiter,relief)
    assert command==pytest.approx(.003,abs=1e-12)
    assert relief.offset==pytest.approx(.0002)

def test_lower_target_reaches_the_actual_act_contact_z_hook_without_loading_policy():
    module=runner();phase=sys.modules['phases'];module.TRIAL_SPEC=phase.AdmittanceSpec(.3)
    from sim.config import load_config
    from sim.act.rollout import CollectionTerminationTracker
    from sim.controllers.admittance import OverforceReliefController1D,OverforceReliefParams
    cfg=module.make_config(load_config(str(module.BATCH/'evaluation_config.yaml')),1)
    command=np.array([.2,.1,.001,0.])
    env=SimpleNamespace(control_dt=.01,time=.02,tcp=lambda:command[:3].copy())
    loop=phase.TwoStageAdmittance(.01,module.TRIAL_SPEC)
    context=dict(env=env,admittance=loop,z_limiter=phase.ZCommandLimiter(.001,.01,initial_time=.02),
        last_reference=command.copy(),trace=[dict(t=.01,command=command.copy()),dict(t=.02,command=command.copy())],
        force_gate=module.old.geometry.ForceGate(20.,3),collection_timer=CollectionTerminationTracker(target_count=1),
        actions=[],act_trace=[],invalid_samples=[],step_failures=[],peak_force=0.,ready=True,
        last_guard_time=None,safety_reason='',collection_reason='',act_outer_frames=0,act_queries=0,accepted_chunks=0,
        z_nominal=.001,relief=OverforceReliefController1D(OverforceReliefParams(),.01),history=[],
        prefix_frames=0,handoff_time=.02,handoff_force=.3,handoff_tcp=command[:3].copy(),
        contact_time=.02,first_contact_position=command[:3].copy())
    adapted,governor=module.build_act_driver(cfg,context)
    hook=adapted.__globals__['_contact_z_command']
    assert hook(cfg,loop,.001,0.,None)==pytest.approx(.0009988,abs=1e-12)
