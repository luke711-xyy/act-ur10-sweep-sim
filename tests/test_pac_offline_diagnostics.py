import math

import numpy as np
import pytest


def test_independent_gaussian_math_excludes_unexecuted_suffix():
    from pac_act.offline_diagnostics import gaussian_kl
    old=np.zeros((1,5,3));new=old.copy();new[:,0,:]=.1;new[:,1:,:]=100
    logs=np.zeros_like(old)
    assert gaussian_kl(old,logs,new,logs,[1])==pytest.approx(.015)


def test_physical_mean_shift_sensitivity_uses_joint_15_dimensional_kl():
    from pac_act.offline_diagnostics import shift_sensitivity
    table=shift_sensitivity(np.full((32,5,3),.1),[5]*32,.02)
    assert table['equal_component_shift_at_hard_limit']['xy_mm']==pytest.approx(.00516397779)
    assert table['equal_component_shift_at_hard_limit']['yaw_deg']==pytest.approx(.00516397779)
    assert table['xy_only_shift_0_01mm_KL']==pytest.approx(.05)
    assert table['yaw_only_shift_0_01deg_KL']==pytest.approx(.025)


def test_legacy_missing_proposed_state_is_not_fabricated(tmp_path):
    import json
    from pac_act.offline_diagnostics import numeric_diagnosis
    root=tmp_path/'run';folder=root/'episodes/preflight_8000006_actor';folder.mkdir(parents=True)
    record=dict(observation=None,action=np.zeros((5,3)).tolist(),mean=np.zeros((5,3)).tolist(),
        log_std=np.full((5,3),math.log(.1)).tolist(),length=5,reward=10.,environment_reward=10.,
        prior_cost=0.,value=0.,next_value=0.,terminated=True)
    (folder/'learning_blocks.jsonl').write_text(json.dumps(record)+'\n')
    result=numeric_diagnosis(root,np.array([.001,.001,math.pi/180]))
    assert result['historical_proposed_means_available'] is False
    assert result['historical_rollback_verified'] is None
    assert result['logged_fields_advantage']['min']==10.
    assert result['physical_std']['dx_mm']['mean']==pytest.approx(.1)


def test_logged_gae_is_not_claimed_exact_without_durable_terminal_corrections(tmp_path):
    import json
    from pac_act.offline_diagnostics import numeric_diagnosis
    folder=tmp_path/'episodes/preflight_8000006_actor';folder.mkdir(parents=True)
    record=dict(action=np.zeros((5,3)).tolist(),mean=np.zeros((5,3)).tolist(),
        log_std=np.full((5,3),math.log(.1)).tolist(),length=5,reward=1.,environment_reward=1.,
        prior_cost=0.,value=0.,next_value=99.,terminated=False)
    (folder/'learning_blocks.jsonl').write_text(json.dumps(record)+'\n')
    # A later zero-action terminal can seal this transition only in collector memory.
    result=numeric_diagnosis(tmp_path,np.array([.001,.001,math.pi/180]))
    assert result['logged_fields_advantage']['mean']==100.
    assert result['recomputed_advantage'] is None
    assert result['historical_GAE_exact_reconstruction_available'] is False
    assert any('terminal' in x for x in result['limitations'])
