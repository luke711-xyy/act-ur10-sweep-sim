"""Configured contact tuning must reach the actual two-stage Z controller."""
import pytest

from sim.config import load_config
from sim.act import hybrid_trial4


def configured_runtime(gains):
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.maintain_MBK', gains)
    runtime = hybrid_trial4.runtime()
    configure = getattr(hybrid_trial4, 'configure_admittance', None)
    assert callable(configure), 'Hybrid ignores configured maintenance admittance gains'
    profile = configure(runtime, cfg)
    return runtime, profile


def test_configured_contact_loop_unloads_force_excess_without_resetting_at_gain_switch():
    runtime, _ = configured_runtime([5., 400., 10.])
    loop = runtime.TwoStageAdmittance(.01, runtime.TRIAL_SPEC)
    loop.reset(delta_z=-.0005, velocity=.0001)
    for i in range(12):
        loop.observe(i * .01, .5)
    assert loop.phase == 'settle'
    assert loop.delta_z == -.0005 and loop.delta_z_rate == .0001
    previous = loop.delta_z
    for _ in range(10):
        z = loop.step(1., 2.)
        assert abs(z) <= .002
    # Measured table stiffness is about 1.6 N/mm. Prompt unloading must
    # remove at least 0.2 mm within 100 ms of a one-newton force excess.
    assert loop.delta_z - previous > .0002


def test_unstable_configured_gains_are_rejected_before_physical_execution():
    with pytest.raises(ValueError, match='unstable'):
        configured_runtime([1., 4000., 10.])


def test_reported_profile_describes_gains_used_after_the_contact_gate():
    runtime, profile = configured_runtime([5., 400., 10.])
    loop = runtime.TwoStageAdmittance(.01, runtime.TRIAL_SPEC)
    assert profile['seek_MBK'] == [10., 400., 10.]
    for i in range(12):
        loop.observe(i * .01, .5)
    # The profile is evidence for actual control, not merely the YAML input.
    assert [loop.params.m, loop.params.b, loop.params.k] == profile['maintain_MBK']
    assert profile['maintain_MBK'] == [5., 400., 10.]


def test_media_recovery_preserves_configured_contact_gains_and_rejects_mismatched_measurement():
    from sim.act.batch_evaluate import effective_replay_config
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.maintain_MBK', [5., 400., 10.])
    measured_profile = {**hybrid_trial4.PROFILE, 'maintain_MBK': [5., 400., 10.],
                        'admittance_configuration': 'act.hybrid_admittance',
                        'overforce_relief': {'activation_force': 10., 'release_force': 7.,
                            'upward_rate': .02, 'return_rate': .1, 'max_offset': .00025}}
    physical = {'inference_strategy': 'hybrid_astar_act_trial4_dinov3_frozen',
                'hybrid_control_profile': measured_profile}
    recovered = effective_replay_config(cfg, physical)
    assert recovered.get_path('act.hybrid_admittance.maintain_MBK') == [5., 400., 10.]
    physical['hybrid_control_profile'] = {**measured_profile, 'maintain_MBK': [5., 200., 10.]}
    with pytest.raises(RuntimeError, match='profile differs'):
        effective_replay_config(cfg, physical)


def test_soft_relief_ramps_and_returns_gently_and_reports_its_actual_parameters():
    from sim.act.evaluate import _overforce_relief_loop
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.maintain_MBK', [10., 800., 10.])
    parameters = {'activation_force': 1.5, 'release_force': 1.1,
                  'upward_rate': .0005, 'return_rate': .0005, 'max_offset': .00025}
    cfg.set_path('controller.overforce_relief', parameters)
    profile = hybrid_trial4.configure_admittance(hybrid_trial4.runtime(), cfg)
    assert profile['overforce_relief'] == parameters
    relief = _overforce_relief_loop(cfg)
    for _ in range(20):
        relief.step(1.5)
    assert relief.offset == pytest.approx(.0001)
    for _ in range(5):
        relief.step(1.)
    assert relief.offset == pytest.approx(.000075)


def test_maintained_contact_responds_promptly_without_resetting_the_integral_state():
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.force_compliance_m_per_N', .00015)
    tuned = hybrid_trial4.runtime()
    profile = hybrid_trial4.configure_admittance(tuned, cfg)
    original = hybrid_trial4.runtime()
    baseline = original.TwoStageAdmittance(.01, original.TRIAL_SPEC)
    loop = tuned.TwoStageAdmittance(.01, tuned.TRIAL_SPEC)
    for controller in (loop, baseline):
        controller.phase = 'maintain'
        controller.loop = controller._loop(4000.)
        controller.reset(delta_z=-.0005, velocity=0.)
    before = loop.delta_z
    relief = loop.step(1., 2.)
    assert relief - before > .0001  # unload before a delayed force integral can
    assert loop.delta_z == baseline.step(1., 2.)  # retain the same integral state
    assert profile['maintain_force_compliance']['gain_m_per_N'] == .00015
    # A falling load restores contact instead of retaining an upward offset.
    assert loop.step(1., 0.) < loop.delta_z - .0001


def test_force_compliance_preserves_acquisition_and_the_total_two_mm_envelope():
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.force_compliance_m_per_N', .00015)
    tuned = hybrid_trial4.runtime()
    hybrid_trial4.configure_admittance(tuned, cfg)
    original = hybrid_trial4.runtime()
    a = tuned.TwoStageAdmittance(.01, tuned.TRIAL_SPEC)
    b = original.TwoStageAdmittance(.01, original.TRIAL_SPEC)
    for phase in ('seek', 'settle'):
        a.phase = b.phase = phase
        for force in (0., .5, 1., 2.):
            assert a.step(1., force) == b.step(1., force)
    a.phase = 'maintain'
    for force in ([20.] * 300 + [0.] * 300):
        assert abs(a.step(1., force)) <= .002


@pytest.mark.parametrize('value', [-.0001, float('nan'), .001, 'invalid'])
def test_invalid_force_compliance_is_rejected_before_controller_creation(value):
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.force_compliance_m_per_N', value)
    with pytest.raises(ValueError, match='force compliance'):
        hybrid_trial4.configure_admittance(hybrid_trial4.runtime(), cfg)


def test_force_compliance_is_not_stacked_and_media_recovery_checks_it():
    from sim.act.batch_evaluate import effective_replay_config
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance.force_compliance_m_per_N', .00015)
    runtime = hybrid_trial4.runtime()
    profile = hybrid_trial4.configure_admittance(runtime, cfg)
    first_type = runtime.TwoStageAdmittance
    hybrid_trial4.configure_admittance(runtime, cfg)
    assert not issubclass(runtime.TwoStageAdmittance, first_type)
    physical = {'inference_strategy': 'hybrid_astar_act_trial4_dinov3_frozen',
                'hybrid_control_profile': profile}
    recovered = effective_replay_config(cfg, physical)
    assert recovered.get_path('act.hybrid_admittance.force_compliance_m_per_N') == .00015
    physical['hybrid_control_profile'] = {**profile, 'maintain_force_compliance': {
        **profile['maintain_force_compliance'], 'gain_m_per_N': .0003}}
    with pytest.raises(RuntimeError, match='profile differs'):
        effective_replay_config(cfg, physical)


def test_omission_on_reconfiguration_restores_archived_admittance_and_profile():
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_admittance', {'maintain_MBK': [5., 400., 10.],
                                         'force_compliance_m_per_N': .00015})
    runtime = hybrid_trial4.runtime()
    hybrid_trial4.configure_admittance(runtime, cfg)
    cfg.set_path('act.hybrid_admittance', None)
    profile = hybrid_trial4.configure_admittance(runtime, cfg)
    assert list(runtime.TRIAL_SPEC.maintain) == profile['maintain_MBK'] == [10., 800., 10.]
    loop = runtime.TwoStageAdmittance(.01, runtime.TRIAL_SPEC)
    loop.phase = 'maintain'
    loop.loop = loop._loop(4000.)
    assert loop.step(1., 2.) < .0001
    assert 'maintain_force_compliance' not in profile


def test_motion_only_configuration_still_reports_actual_custom_relief():
    cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
    cfg.set_path('act.hybrid_sweep_smoothing', {'xy_acceleration_m_s2': .1,
                                               'yaw_acceleration_rad_s2': .5})
    cfg.set_path('controller.overforce_relief.upward_rate', .0005)
    profile = hybrid_trial4.configure_admittance(hybrid_trial4.runtime(), cfg)
    assert profile['overforce_relief']['upward_rate'] == .0005
