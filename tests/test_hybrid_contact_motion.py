"""A maintained-contact sweep must not release into abrupt ACT rotation."""
import numpy as np
import pytest

from sim.act import hybrid_trial4
from sim.config import load_config


def configured():
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_sweep_smoothing', {
        'xy_acceleration_m_s2': .1, 'yaw_acceleration_rad_s2': .5})
    module = hybrid_trial4.runtime()
    profile = hybrid_trial4.configure_admittance(module, cfg)
    governor = module.handoff.TwoSecondHandoff([0., 0., 0., 0.], .01, 0.,
                                             [[-.4, .4], [-.3, .3]])
    governor.chunk_ack(True, 0.)
    governor.action_ready(0.)
    return cfg, module, profile, governor


def test_contact_sweep_keeps_acceleration_bounds_after_two_seconds():
    _, _, profile, g = configured()
    for i in range(400):
        now = i * .01
        target = np.array([.2 if i < 250 else -.2, .1, 0., 1. if i < 250 else -1.])
        command = g.propose(now, target, target, -.0003)
        p = g.pending
        assert np.linalg.norm(p['acceleration_xy_m_s2']) <= (.05 if i < 200 else .1) + 1e-8
        assert abs(p['yaw_acceleration_rad_s2']) <= .5 + 1e-8
        assert np.linalg.norm(p['velocity_xy_m_s']) <= .08 + 1e-8
        assert command[2] == -.0003
        row = {'t': now + .01, 'command': command.copy()}
        g.prepare_record(row)
        g.acknowledge(row)
    assert g.pending['mode'] == 'contact_sweep'
    assert profile['sweep_smoothing']['xy_acceleration_m_s2'] == .1
    assert 'exact_native' not in profile['after_handoff_window']


def test_rejected_physics_does_not_advance_contact_motion_state():
    _, _, _, g = configured()
    command = g.propose(0., [.1, .1, 0., .2], [.1, .1, 0., .2], 0.)
    before = g.last_reference.copy()
    row = {'t': 0., 'command': command, 'control_step_advanced': False}
    g.prepare_record(row)
    g.acknowledge(row)
    assert g.last_time == 0.
    assert np.array_equal(g.last_reference, before)


def test_replay_configuration_cannot_drop_continuous_sweep_smoothing():
    from sim.act.batch_evaluate import effective_replay_config
    cfg, _, profile, _ = configured()
    physical = {'inference_strategy': 'hybrid_astar_act_trial4_dinov3_frozen',
                'hybrid_control_profile': profile}
    assert effective_replay_config(cfg, physical).get_path('act.hybrid_sweep_smoothing')
    cfg.set_path('act.hybrid_sweep_smoothing', None)
    with pytest.raises(RuntimeError, match='profile differs'):
        effective_replay_config(cfg, physical)


def test_omitting_sweep_tuning_restores_the_archived_governor_on_reconfiguration():
    cfg, module, _, _ = configured()
    cfg.set_path('act.hybrid_sweep_smoothing', None)
    profile = hybrid_trial4.configure_admittance(module, cfg)
    g = module.handoff.TwoSecondHandoff([0., 0., 0., 0.], .01, 3., [[-.4, .4], [-.3, .3]])
    g.started_at = 0.
    target = np.array([.1, .1, 0., .2])
    assert np.array_equal(g.propose(3., target, target, 0.), target)
    assert 'sweep_smoothing' not in profile


def test_continuous_yaw_accepts_exact_float32_trace_rounding_at_large_angles():
    _, _, _, g = configured()
    for i in range(1600):
        target = np.array([.01, .01, 0., 3.])
        command = g.propose(i*.01, target, target, -.0003)
        row = {'t': (i+1)*.01, 'command': command.astype(np.float32)}
        g.prepare_record(row)
        g.acknowledge(row)
    assert g.last_reference[3] > 2.9


def test_continuous_yaw_rejects_even_one_extra_float32_unit_of_command_change():
    _, _, _, g = configured()
    g.last_time = 3.
    g.last_reference[3] = 2.5
    command = g.propose(3., [.01, .01, 0., 3.], [.01, .01, 0., 3.], -.0003)
    actual = command.astype(np.float32)
    actual[3] = np.nextafter(actual[3], np.float32(np.inf))
    row = {'t': 3.01, 'command': actual}
    g.prepare_record(row)
    with pytest.raises(RuntimeError, match='accepted command differs'):
        g.acknowledge(row)


@pytest.mark.parametrize('corruption', ['nan_velocity', 'changed_limit', 'missing_mode',
                                       'missing_payload', 'missing_start', 'false_advanced', 'stale_elapsed'])
def test_contact_sweep_audit_uses_configured_limits_and_requires_finite_telemetry(corruption):
    from types import SimpleNamespace
    _, module, _, g = configured()
    g.last_time = 3.
    command = g.propose(3., [.1, .1, 0., .2], [.1, .1, 0., .2], 0.)
    row = {'t': 3.01, 'command': command.copy(), 'tcp': command[:3].copy(),
           'z_owner': 'admittance', 'normal_force': 1.}
    g.prepare_record(row)
    p = row['handoff_smoothing']
    if corruption == 'nan_velocity':
        p['velocity_xy_m_s'][0] = float('nan')
    elif corruption == 'changed_limit':
        p['xy_acceleration_limit_m_s2'] = 10.
    elif corruption == 'missing_mode':
        p.update(mode='native_restored', native_reference=p['command'].copy())
    elif corruption == 'missing_payload':
        row.pop('handoff_smoothing')
    elif corruption == 'missing_start':
        p['started_at'] = None
    elif corruption == 'stale_elapsed':
        p.update(elapsed_s=1., mode='native_restored', native_reference=p['command'].copy())
    else:
        p['advanced'] = False
    with pytest.raises(RuntimeError, match='contact sweep'):
        hybrid_trial4._audit_result(module, {'step_failures': [], 'invalid_samples': [],
                                           'handoff_governor': g},
                                   SimpleNamespace(trace=[row]))
