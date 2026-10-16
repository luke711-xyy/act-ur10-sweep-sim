"""Pose action units and persistence cannot silently reuse velocity semantics."""
import copy
import numpy as np
import pytest


def test_actor_sees_bounded_pose_and_rate_as_distinct_control_states():
    from residual_rl.observations import actor_observation
    obs=actor_observation(robot_state=np.zeros(36),feature=np.zeros(512),target_count=1,
        elapsed=0.,feature_age=0.,chunk_index=0,feature_available=False,active=True,
        nominal_velocity=np.zeros(3),previous_action=np.zeros(3),
        residual_velocity=[.005,0.,0.],residual_offset=[0.,.0025,0.],
        act_displacement=[.01,0.,0.],filtered_action=[.1,.2,.3])
    assert obs.shape==(572,)
    assert obs[554]==pytest.approx(.0625)
    assert obs[563:566].tolist()==[0.,.3125,0.]
    assert obs[-3:]==pytest.approx([.1,.2,.3])


def test_pose_learner_and_replay_reject_old_same_shape_velocity_snapshots():
    from residual_rl.learning import RecurrentSAC,EpisodicReplay
    learner=RecurrentSAC(4,6); snapshot=learner.checkpoint()
    assert snapshot['schema_version']==3
    assert snapshot['config']['action_contract']=='relative_xy80_yaw45_v1'
    old=copy.deepcopy(snapshot); old['schema_version']=2
    with pytest.raises(ValueError): learner.load_checkpoint(old)
    replay=EpisodicReplay(); state=replay.state_dict()
    assert state['schema_version']==2
    old=copy.deepcopy(state); old['action_contract']='velocity_residual_v1'
    with pytest.raises(ValueError): replay.load_state_dict(old)


def test_policy_guard_reports_pose_change_in_metres_and_radians():
    from residual_rl.learning import RecurrentSAC
    from residual_rl.policy_guard import PolicyUpdateGuard,ProbeHistory
    learner=RecurrentSAC(4,6)
    history=ProbeHistory('pose',1,np.zeros((4,4),np.float32),np.array([0,1,2]))
    guard=PolicyUpdateGuard(learner.actor,learner.actor_optimizer,[history])
    report=guard.check()
    assert report['max_xy_change_m'] is None and report['max_xy_change_fraction']==0.
    assert report['max_yaw_change_rad']==0.
    assert 'max_xy_change_m_s' not in report
