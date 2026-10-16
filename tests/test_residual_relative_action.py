"""Approved B/80% geometry: independently calculated expectations, no physics."""
import math
import numpy as np
import pytest


def generate(p,q,a,**kwargs):
    import residual_rl.relative_action as api
    return api.generate_relative_action(p,np.asarray(q),a,**kwargs)


def test_relative_action_uses_actual_xy_and_limits_correction_not_final_length():
    out,audit=generate([.01,.02],np.array([.012,.02,.001,.2]),[1.,0.,0.])
    assert out==pytest.approx([.0136,.02,.001,.2])
    assert audit['original_xy_length']==pytest.approx(.002)
    assert audit['correction_radius']==pytest.approx(.0016)


def test_relative_action_uses_act_direction_as_forward_basis():
    out,_=generate([0.,0.],np.array([0.,.002,.001,0.]),[0.,1.,0.])
    assert out[:2]==pytest.approx([-.0016,.002])


def test_generated_square_corner_lies_on_80_percent_disk_without_radial_clipping():
    out,_=generate([0.,0.],np.array([.002,0.,.001,0.]),[1.,1.,1.])
    assert out[:2]==pytest.approx([.002+.0016/math.sqrt(2),.0016/math.sqrt(2)])
    assert out[3]==pytest.approx(math.pi/4)


def test_zero_relative_action_preserves_original_target_bytes_and_dtype():
    for dtype in (np.float32,np.float64):
        q=np.array([.1,-.2,.001,-3.14],dtype=dtype)
        out,_=generate([-.3,.2],q,[0.,0.,0.])
        assert out.dtype==q.dtype and out.tobytes()==q.tobytes()


def test_zero_at_original_float32_workspace_edge_retains_native_quantization():
    q=np.array([.46,0.,.001,0.],np.float32)
    out,_=generate([.4,0.],q,[0.,0.,0.],bounds=[[-.42,.46],[-.32,.32]])
    assert out.tobytes()==q.tobytes()


def test_sub_ulp_displacement_never_rounds_to_a_full_length_correction():
    q=np.array([.25,0.,.001,0.])
    p=np.array([np.nextafter(.25,-np.inf),0.])
    out,_=generate(p,q,[-1.,0.,0.])
    assert np.linalg.norm(out[:2]-q[:2])<=.8*np.linalg.norm(q[:2]-p)
    assert (out[:2]-p)@(q[:2]-p)>0.


def test_stationary_act_has_no_xy_correction_even_at_full_policy_action():
    out,audit=generate([.25,.5],np.array([.25,.5,.003,.2]),[-1.,1.,-.5])
    assert out[:2].tolist()==[.25,.5]
    assert audit['correction_radius']==0.
    assert out[3]==pytest.approx(.2-math.pi/8)


def test_workspace_radius_is_conditioned_before_generation_not_clipped_afterward():
    out,audit=generate([.99,0.],np.array([.999,0.,.003,0.]),[.5,0.,0.],
        bounds=np.array([[-1.,1.],[-1.,1.]]))
    assert out[0]==pytest.approx(.9995)  # half of the available 1mm ray, not hard-clamped to 1
    assert audit['feasible_radius']==pytest.approx(.001)


def test_yaw_wrap_preserves_bounded_relative_angle_and_z():
    out,_=generate([0.,0.],np.array([.002,0.,-.002,3.13]),[0.,0.,1.])
    delta=math.atan2(math.sin(out[3]-3.13),math.cos(out[3]-3.13))
    assert delta==pytest.approx(math.pi/4) and out[2]==-.002


@pytest.mark.parametrize('bad',[[float('nan'),0,0],[1.1,0,0],[0,0,float('inf')]])
def test_invalid_relative_action_is_rejected(bad):
    with pytest.raises(ValueError):generate([0.,0.],np.array([.002,0.,.001,0.]),bad)


def test_random_finite_inputs_obey_relative_disk_and_forward_half_plane():
    rng=np.random.default_rng(76)
    for _ in range(500):
        p=rng.uniform(-.5,.5,2);q=np.r_[p+rng.uniform(-.01,.01,2),.001,rng.uniform(-math.pi,math.pi)]
        out,_=generate(p,q,rng.uniform(-1.,1.,3))
        original=q[:2]-p;correction=out[:2]-q[:2];new=out[:2]-p
        assert np.linalg.norm(correction)<=.8*np.linalg.norm(original)+1e-12
        assert new@original>=.2*(original@original)-1e-12
        cosine=(new@original)/(np.linalg.norm(new)*np.linalg.norm(original))
        assert cosine>=.6-1e-10  # cos(asin(.8))=.6


def test_relative_filter_updates_targets_not_accumulating_old_world_offset():
    import residual_rl.relative_action as api
    c=api.RelativeActionGenerator([[-1.,1.],[-1.,1.]])
    p=np.zeros(2)
    for _ in range(250):
        native=np.r_[p+[.002,0.],.001,0.]
        out=c.generate(p,native,[0.,.5,.1],dt=.04,active=True)
        assert np.linalg.norm(out[:2]-native[:2])<=.0016+1e-12
        p=out[:2]
    assert abs(c.offset[1])<=.0016


def test_failed_generator_does_not_commit_filter_or_offsets():
    import residual_rl.relative_action as api
    c=api.RelativeActionGenerator([[-1.,1.],[-1.,1.]])
    with pytest.raises(ValueError):c.generate([0.,0.],np.array([.002,0.,.001,0.]),[2.,0.,0.],dt=.04,active=True)
    assert not c.offset.any() and not c.filtered_action.any()


def test_continuous_probe_varies_every_25hz_frame_and_is_not_a_single_pulse():
    from residual_rl.relative_action import continuous_probe_action
    sequence=np.array([continuous_probe_action(i*.04) for i in range(500)])
    assert not sequence[0].any()
    assert np.max(np.abs(sequence[:,:2]))<=.30+1e-12
    assert np.max(np.abs(sequence[:,2]))<=.10+1e-12
    assert np.count_nonzero(np.linalg.norm(np.diff(sequence,axis=0),axis=1)>1e-6)>490
    assert np.linalg.norm(sequence[200])>.1 and np.linalg.norm(sequence[400])>.1


def test_policy_prior_uses_the_same_direct_square_to_disk_map():
    import torch
    from residual_rl.policy_guard import requested_residual
    result=requested_residual(torch.tensor([[.5,1.,.25]],dtype=torch.float64))
    assert result.detach().numpy()[0]==pytest.approx([.3535533905932738,.9354143466934853,.25])
    result.sum().backward() if result.requires_grad else None


def test_actor_receives_accepted_displacement_not_object_truth():
    from residual_rl.observations import actor_observation
    obs=actor_observation(robot_state=np.zeros(36),feature=np.zeros(512),target_count=1,
        elapsed=0.,feature_age=0.,chunk_index=0,feature_available=False,active=True,
        nominal_velocity=np.zeros(3),previous_action=np.zeros(3),residual_velocity=np.zeros(3),
        residual_offset=np.zeros(3),act_displacement=np.array([.002,-.001,.2]),filtered_action=[.1,.2,.3])
    assert obs.shape==(572,) and obs[-6:-3]==pytest.approx([.002,-.001,.2/math.pi])
    assert obs[-3:]==pytest.approx([.1,.2,.3])


def test_relative_action_contract_refuses_old_pose_snapshot():
    import copy
    from residual_rl.learning import RecurrentSAC,EpisodicReplay
    model=RecurrentSAC(4,6);state=model.checkpoint()
    assert state['config']['action_contract']=='relative_xy80_yaw45_v1'
    old=copy.deepcopy(state);old['config']['action_contract']='bounded_pose_xy5mm_yaw3deg_v1'
    with pytest.raises(ValueError):model.load_checkpoint(old)
    replay=EpisodicReplay();old=replay.state_dict();old['action_contract']='bounded_pose_xy5mm_yaw3deg_v1'
    with pytest.raises(ValueError):replay.load_state_dict(old)


def test_installed_relative_adapter_uses_actual_basis_and_retains_native_z_and_hold():
    import subprocess,sys
    from pathlib import Path
    script=r'''
from types import SimpleNamespace
import numpy as np
from residual_rl.runtime import PROJECT,identity,load_formal,install_driver,PolicyCache
from residual_rl.baseline import pre_dino_sources
from residual_rl.features import FeatureMailbox
from residual_rl.relative_action import RelativeActionGenerator
f=identity();pre_dino_sources(PROJECT,f['source_fingerprints']).activate()
from sim.config import load_config
formal=load_formal();module=formal.runtime()
cfg=formal.candidate_config(load_config(str(formal.OLD/'evaluation_config.yaml')),1)
position=np.array([.1,0.,0.]);env=SimpleNamespace(control_dt=.01,time=5.,tcp=lambda:position.copy(),ee=SimpleNamespace(tcp_yaw=lambda:0.))
controller=object();force_gate=object()
c=dict(env=env,admittance=controller,force_gate=force_gate,collection_timer=object(),
    trace=[dict(t=4.99,command=np.zeros(4)),dict(t=5.,command=np.zeros(4))],
    last_reference=np.zeros(4),act_trace=[])
collector=SimpleNamespace(mailbox=FeatureMailbox(),env=env,mode='evaluate',shadow=None,
    control=RelativeActionGenerator([[-.42,.46],[-.32,.32]]),raw_action=np.array([0.,.2,.2]))
collector.is_active=lambda:(collector.governor is not None and collector.governor.started_at is not None and env.time-collector.governor.started_at>=2.-1e-9)
collector.begin=lambda nominal:None
driver=install_driver(formal,module,cfg,c,collector,PolicyCache())
collector.governor.chunk_ack(True,env.time);collector.governor.action_ready(env.time)
reference=driver.__globals__['_policy_reference_substep'];last=np.array([.1,0.,0.,0.],np.float32)
for i in range(320):
    if i%4==0:
        position[:2]=last[:2]-[.001,0.]
        start=last.copy();target=start+np.array([.0016,0.,0.,.0004],np.float32)
    active=collector.is_active();z=.001+i*1e-8
    command=reference(start,target,i%4+1,4,.01,.08,.1,.35,True,z)
    if active:
        audit=collector.control.last_audit
        assert command[2]==np.float32(z)
        assert np.allclose(audit['actual_xy'],position[:2])
        assert audit['original_xy_length']>.0025  # actual lag included, not .0016 reference-only step
        assert np.linalg.norm(audit['xy_correction'])<=.8*audit['original_xy_length']+1e-12
        assert audit['action_contract']=='relative_xy80_yaw45_v1'
    row=dict(t=env.time+.01,command=np.asarray(command,dtype=np.float32),control_step_advanced=True)
    collector.governor.prepare_record(row);collector.governor.acknowledge(row)
    last=command.copy();env.time+=.01
assert collector.control.offset[1]>0
assert c['admittance'] is controller and c['force_gate'] is force_gate
collector.scheduler=SimpleNamespace(observation_chunk=None)
held=last.copy()
for i in range(80):
    if i%4==0:start=last.copy();target=start.copy()
    command=reference(start,target,i%4+1,4,.01,.08,.1,.35,True,.001)
    row=dict(t=env.time+.01,command=np.asarray(command,dtype=np.float32),control_step_advanced=True)
    collector.governor.prepare_record(row);collector.governor.acknowledge(row)
    last=command.copy();env.time+=.01
assert np.allclose(last[:2],held[:2],atol=1e-7)
assert abs(last[3]-held[3])<1e-7
print('installed relative adapter exercised without models or physical rollout')
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
