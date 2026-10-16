"""Bounded pose micro-corrections; no model loading or physical rollout."""
from concurrent.futures import ThreadPoolExecutor
import math
import threading

import numpy as np
import pytest

from residual_rl.control import ResidualGovernor,ResidualReference


def control():
    return ResidualGovernor([[-100.,100.],[-100.,100.]])


def test_pose_requests_hold_bounded_offsets_over_many_control_frames():
    c=control(); reference=ResidualReference(c); executed=np.zeros(4,np.float32)
    for frame in range(400):
        nominal=reference.begin_frame(executed)
        # A fresh query uses the same uncorrected anchor, not the executed pose.
        assert np.allclose(c.remove_offset(executed),nominal,atol=1e-7)
        for i in range(4):
            raw=nominal+np.array([.0004*(i+1),0.,-.0001,.001*(i+1)],np.float32)
            executed=reference.step(raw,np.ones(3),dt=.01,active=True)
            assert np.linalg.norm(executed[:2]-raw[:2])<=.0050001
            assert abs(executed[3]-raw[3])<=math.pi/60+1e-7
            assert executed[2]==raw[2]
    assert .004<np.linalg.norm(c.offset[:2])<=.0050001
    assert c.offset[2]==pytest.approx(math.pi/60,abs=2e-6)
    assert executed[0]<.65  # ~0.64 m ACT motion, never +5 mm every frame.


def test_pose_limits_and_continuity_do_not_stop_or_reverse_native_motion():
    c=control(); executed=np.zeros(4,np.float32); native_before=executed.copy()
    for i in range(120):
        native=native_before+np.array([.0008,0.,0.,.0035],np.float32)
        out=c.step(executed,native,np.array([-1.,1.,-1.]),dt=.01,active=True,native_before=native_before)
        velocity=(out[:2]-executed[:2])/.01
        assert np.linalg.norm(velocity)<=.08002
        assert velocity[0]>=.064-2e-5
        assert abs(math.atan2(velocity[1],velocity[0]))<=math.radians(10)+1e-3
        assert abs(out[3]-native[3])<=math.pi/60+1e-7
        assert (out[3]-executed[3])/.01>=.28-2e-5
        assert np.linalg.norm(c.velocity[:2])<=.00502
        assert abs(c.velocity[2])<=math.pi/60+2e-5
        native_before=native; executed=out


def test_native_idle_freezes_existing_offset_without_snap_or_extra_motion():
    c=control(); executed=np.zeros(4,np.float32); native_before=executed.copy()
    for _ in range(80):
        native=native_before+np.array([.0005,0.,0.,0.],np.float32)
        executed=c.step(executed,native,np.array([0.,1.,1.]),dt=.01,active=True,native_before=native_before)
        native_before=native
    before=executed.copy()
    for _ in range(40):
        executed=c.step(executed,native_before,-np.ones(3),dt=.01,active=True,native_before=native_before)
        assert executed.tobytes()==before.tobytes()
    assert np.array_equal(c.velocity,np.zeros(3))


def test_full_speed_lateral_request_steers_without_exceeding_original_speed():
    c=control(); before=np.zeros(4,np.float32)
    native=np.array([.0008,0.,.001,0.],np.float32)
    out=c.step(before,native,np.array([0.,1.,0.]),dt=.01,active=True)
    assert out[1]>0 and out[0]>0
    assert np.linalg.norm(out[:2]/.01)<=.080001


def test_zero_pose_path_is_exact_for_all_dtypes_and_native_yaw_wrap():
    for dtype in (np.float32,np.float64):
        c=control(); before=np.array([0.,0.,.001,math.pi-.001],dtype=dtype)
        raw=np.array([.0002,0.,-.0002,-math.pi+.001],dtype=dtype)
        out=c.step(before,raw,np.zeros(3),dt=.01,active=True)
        assert out.dtype==raw.dtype and out.tobytes()==raw.tobytes()


def test_incompatible_workspace_rate_constraints_fail_without_mutating_state():
    c=ResidualGovernor([[-1.,1.],[-1.,1.]])
    c.offset[:]=[.005,0.,0.]
    offset=c.offset.copy(); velocity=c.velocity.copy()
    before=np.array([.999,0.,0.,0.],np.float32)
    native=np.array([1.,0.,0.,0.],np.float32)
    with pytest.raises(RuntimeError,match='infeasible'):
        c.step(before,native,np.zeros(3),dt=.01,active=True,
               native_before=np.array([.994,0.,0.,0.],np.float32))
    assert np.array_equal(c.offset,offset) and np.array_equal(c.velocity,velocity)


def test_query_reference_is_captured_at_submission_not_worker_completion():
    from residual_rl.control import nominal_reference_executor
    c=control(); c.offset[:]=[.004,-.002,.03]
    ready=threading.Event()
    def delayed(obs,start):
        ready.wait(timeout=2)
        return start
    with nominal_reference_executor(ThreadPoolExecutor,c)(max_workers=1) as pool:
        future=pool.submit(delayed,None,np.array([.104,.198,.001,.23],np.float32))
        c.offset[:]=[-.003,.001,-.02]
        ready.set(); result=future.result(timeout=2)
    assert np.allclose(result,[.1,.2,.001,.2],atol=1e-8)


@pytest.mark.parametrize('bad',[np.array([np.nan,0.,0.]),np.array([2.,0.,0.])])
def test_bad_pose_action_is_rejected_without_state_change(bad):
    c=control(); zero=np.zeros(4,np.float32)
    with pytest.raises(ValueError): c.step(zero,zero,bad,dt=.01,active=True)
    assert not c.offset.any() and not c.velocity.any()


def test_changing_native_directions_and_yaw_wrap_keep_pose_bounded():
    c=control(); before=np.array([0.,0.,0.,np.pi-.001],np.float32); nominal=before.copy()
    for i in range(500):
        angle=(i//100)*np.pi/2
        raw=nominal+np.array([.0004*np.cos(angle),.0004*np.sin(angle),0.,.0005],np.float32)
        raw[3]=np.arctan2(np.sin(raw[3]),np.cos(raw[3]))
        out=c.step(before,raw,np.array([np.sin(i/20),np.cos(i/20),-.4]),dt=.01,active=True,native_before=nominal)
        assert np.linalg.norm(out[:2]-raw[:2])<=.0050001
        assert abs(np.arctan2(np.sin(out[3]-raw[3]),np.cos(out[3]-raw[3])))<=np.pi/60+1e-7
        assert out[2]==raw[2]
        before=out; nominal=raw


def test_repeated_query_and_frame_reanchoring_never_adds_offset_again():
    from residual_rl.control import nominal_reference_executor
    c=control(); composer=ResidualReference(c); executed=np.zeros(4,np.float32)
    with nominal_reference_executor(ThreadPoolExecutor,c)(max_workers=1) as pool:
        for frame in range(200):
            anchor=pool.submit(lambda obs,start:start,None,executed.copy()).result(timeout=2)
            native_start=composer.begin_frame(executed)
            assert anchor.tobytes()==native_start.tobytes()
            for i in range(4):
                raw=anchor+np.array([.0004*(i+1),0.,0.,0.],np.float32)
                executed=composer.step(raw,np.array([0.,.5,.5]),dt=.01,active=True)
            assert np.linalg.norm(executed[:2]-raw[:2])<=.0050001
    assert executed[1]<.002501


def test_installed_relative_adapter_does_not_touch_z_or_native_handoff():
    import subprocess,sys
    from pathlib import Path
    script='''
from types import SimpleNamespace
import numpy as np
from residual_rl.runtime import PROJECT,identity,load_formal,install_driver,PolicyCache
from residual_rl.baseline import pre_dino_sources
from residual_rl.features import FeatureMailbox
from residual_rl.relative_action import RelativeActionGenerator as ResidualGovernor
f=identity(); pre_dino_sources(PROJECT,f['source_fingerprints']).activate()
from sim.config import load_config
formal=load_formal(); module=formal.runtime()
cfg=formal.candidate_config(load_config(str(formal.OLD/'evaluation_config.yaml')),1)
env=SimpleNamespace(control_dt=.01,time=5.,tcp=lambda:np.r_[last[:2],0.],ee=SimpleNamespace(tcp_yaw=lambda:float(last[3])))
controller=object(); force_gate=object()
c=dict(env=env,admittance=controller,force_gate=force_gate,collection_timer=object(),
       trace=[dict(t=4.99,command=np.zeros(4)),dict(t=5.,command=np.zeros(4))],
       last_reference=np.zeros(4),act_trace=[])
collector=SimpleNamespace(mailbox=FeatureMailbox(),env=env,mode='evaluate',shadow=None,
    control=ResidualGovernor([[-.42,.46],[-.32,.32]]),raw_action=np.array([0.,.2,.2]))
collector.is_active=lambda: (collector.governor is not None and
    collector.governor.started_at is not None and env.time-collector.governor.started_at>=2.-1e-9)
collector.begin=lambda nominal:None
cache=PolicyCache(); driver=install_driver(formal,module,cfg,c,collector,cache)
collector.governor.chunk_ack(True,env.time)
collector.governor.action_ready(env.time)
reference=driver.__globals__['_policy_reference_substep']; last=np.zeros(4,np.float32)
for i in range(600):
    if i%4==0:
        start=last.copy(); target=start+np.array([.0016,0.,0.,.0004],np.float32)
    z=.001+i*1e-8
    active=collector.is_active()
    command=reference(start,target,i%4+1,4,.01,.08,.1,.35,True,z)
    if active:
        audit=collector.control.last_audit
        assert command[2]==np.float32(z)
        assert np.linalg.norm(audit['xy_correction'])<=.8*audit['original_xy_length']+1e-12
        assert audit['action_contract']=='relative_xy80_yaw45_v1'
    row=dict(t=env.time+.01,command=np.asarray(command,dtype=np.float32),control_step_advanced=True)
    collector.governor.prepare_record(row); collector.governor.acknowledge(row)
    last=np.asarray(command,dtype=np.float32); env.time+=.01
assert collector.control.offset[1]>.0002  # Prove the relative correction really activated.
assert c['admittance'] is controller and c['force_gate'] is force_gate
assert cache.bundle is None
# Missing ACT chunks hold the native anchor, not the already shifted pose.
collector.scheduler=SimpleNamespace(observation_chunk=None)
held=last.copy()
for i in range(160):
    if i%4==0: start=last.copy(); target=start.copy()
    command=reference(start,target,i%4+1,4,.01,.08,.1,.35,True,.001)
    row=dict(t=env.time+.01,command=np.asarray(command,dtype=np.float32),control_step_advanced=True)
    collector.governor.prepare_record(row); collector.governor.acknowledge(row)
    last=np.asarray(command,dtype=np.float32); env.time+=.01
assert np.linalg.norm(last[:2]-held[:2])<2e-7, (last,held)
print('actual installed reference adapter, no ACT/model/physical execution')
'''
    r=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr


def test_feasible_tangent_disks_on_full_speed_turn_do_not_false_reject():
    c=ResidualGovernor([[-1.,1.],[-1.,1.]])
    c.offset[:]=[.005,0.,0.]
    out=c.step(np.array([.105,0.,0.,0.]),np.array([.0992,0.,0.,0.]),
        np.array([0.,1.,0.]),dt=.01,active=True,native_before=np.array([.1,0.,0.,0.]))
    assert np.allclose(out,[.1042,0.,0.,0.],atol=1e-10)
    assert np.allclose(c.offset,[.005,0.,0.],atol=1e-10)


def test_float32_yaw_cap_chooses_adjacent_feasible_representable_value():
    c=control(); c.offset[2]=.05235981941223144
    before=np.array([0.,0.,0.,4.001321315765381],np.float32)
    native=np.array([0.,0.,0.,3.9499616622924805],np.float32)
    previous=np.array([0.,0.,0.,3.9489614963531494],np.float32)
    out=c.step(before,native,[0.,0.,1.],dt=.01,active=True,native_before=previous)
    assert out[3]==np.float32(4.002321243286133)
    assert abs(float(out[3])-float(native[3]))<=np.pi/60
