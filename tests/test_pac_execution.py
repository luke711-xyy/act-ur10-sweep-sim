"""Run the real trial4 reference/ack/Z chain with deterministic sensor input.

Only the MuJoCo integration boundary is replaced. This is not a physical
rollout clearance; it catches control-chain wiring and prefix accounting.
"""
import json
import shutil
from types import SimpleNamespace
import numpy as np
import pytest


def wired_episode(monkeypatch,tmp_path,*,fail_after=None,force=1.,learned=False):
    from pac_act.hybrid import HybridEpisode,load_trial4,BlockClock
    from pac_act.lifecycle import cfg_for
    from pac_act.identity import PROJECT
    from pac_act.rewards import ContactReward
    from sim.act.rollout import CollectionTerminationTracker,_overforce_relief_loop
    formal,module=load_trial4()
    bundle=tmp_path/'legacy_bundle'; bundle.mkdir(exist_ok=True)
    shutil.copyfile(PROJECT/'hybrid_trial4_snapshot/evaluation_config.yaml',bundle/'evaluation_config.yaml')
    (bundle/'manifest.json').write_text(json.dumps({}))
    cfg=cfg_for(bundle,1)
    class Env:
        time=.02; control_dt=.01
        ee=SimpleNamespace(tcp_yaw=lambda:0.,wrist_ft_world=lambda:np.zeros(6),compensated_wrench_world=lambda:np.zeros(6))
        data=SimpleNamespace(qpos=np.zeros(6),qvel=np.zeros(6),sensordata=np.zeros(6))
        def tcp(self): return np.array([.42,0.,.001])
        def normal_force(self): return force
        def raw_normal_force(self): return force
        def compensated_normal_force_raw(self): return force
        def collected_mask(self): return np.zeros(6,bool)
        def partial_collection_overlap_mask(self): return np.zeros(6,bool)
        def unsafe_robot_collision(self): return False
        def step_control(self,command):
            if fail_after is not None and self.time>=fail_after:
                raise ValueError('UR10 IK target is unreachable: position residual=0.003')
            self.time+=.01
    env=Env(); reference=np.array([.42,0.,.001,0.],np.float32)
    context=dict(env=env,ready=True,history=[],last_reference=reference,
        admittance=module.TwoStageAdmittance(.01,module.TRIAL_SPEC),
        z_limiter=module.ZCommandLimiter(.001,.01,initial_time=.02),z_nominal=.001,
        relief=_overforce_relief_loop(cfg),force_gate=module.old.geometry.ForceGate(20.,3),
        collection_timer=CollectionTerminationTracker(1),last_guard_time=None,
        safety_reason='',collection_reason='',act_trace=[],actions=[],act_outer_frames=0,act_queries=0,
        trace=[dict(t=.01,command=reference.copy()),dict(t=.02,command=reference.copy())])
    driver,governor=formal.build_formal_act_driver(module,cfg,context)
    world=HybridEpisode.__new__(HybridEpisode)
    world.module=module; world.cfg=cfg; world.env=env; world.context=context; world.governor=governor
    world.reference=reference; world.clock=BlockClock(); world.active=False; world.active_actions=0
    world.done=False; world.current_reward=0.; world.reward=ContactReward(1); world.terminal_reward=0.
    world.physical_error=None; world.ik_error=None; world.saved_actions=[]; world.path=tmp_path
    world.measured_count=0; world.outcome='failure'
    world.on_action=None; world.builder=SimpleNamespace(advance_state_history=lambda *a,**kw:None)
    world.reference_substep=driver.__globals__['_policy_reference_substep']
    world.record=driver.__globals__['_trial_record']; world.chunk_ack=driver.__globals__['_trial_chunk_ack']
    world.z_command=driver.__globals__['_contact_z_command']; world._save_row=lambda row:None
    if learned:
        world.clock.frames=50; governor.started_at=env.time-2.
    def row(env,command,*args):
        return dict(t=env.time,command=np.asarray(command).copy(),tcp=env.tcp(),normal_force=force,
                    object_collected=np.zeros(6,bool),z_owner='admittance')
    monkeypatch.setattr(module,'physical_row',row)
    return world


def test_real_Z_controller_state_same_with_different_ACT_Z(monkeypatch,tmp_path):
    one=wired_episode(monkeypatch,tmp_path)
    a=np.zeros((5,4)); a[:,0]=.0003
    assert one.step_block(a,learned=False)[0]==5
    z1=one.context['z_limiter'].last_z
    two=wired_episode(monkeypatch,tmp_path)
    a[:,2]=999
    assert two.step_block(a,learned=False)[0]==5
    assert two.context['z_limiter'].last_z==z1
    assert two.context['admittance'].delta_z==one.context['admittance'].delta_z
    assert two.governor.last_time==pytest.approx(.22)


def test_IK_failure_counts_only_physically_executed_prefix(monkeypatch,tmp_path):
    world=wired_episode(monkeypatch,tmp_path,fail_after=.06,learned=True)
    k,reward,done=world.step_block(np.zeros((5,4)),learned=True)
    assert k==1 and done and reward==0.
    assert world.active_actions==1 and world.outcome=='failure'
    assert world.terminal_reward==0. and world.ik_error


def test_safety_penalty_once_for_actual_prefix(monkeypatch,tmp_path):
    world=wired_episode(monkeypatch,tmp_path,force=21.,learned=True)
    k,reward,done=world.step_block(np.zeros((5,4)),learned=True)
    assert k==1 and done and reward==-60.
    world._terminal('safety','again')
    assert world.current_reward==-60.
