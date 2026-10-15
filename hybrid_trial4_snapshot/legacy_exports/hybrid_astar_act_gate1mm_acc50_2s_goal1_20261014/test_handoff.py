"""Behavioral checks for the disposable two-second, real-ACT output hook."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

PROJECT = Path('/Users/huazhi_luke/.codex/worktrees/act-five-view')
sys.path.insert(0, str(PROJECT))
from sim.act.evaluate import _policy_reference_substep as native_reference
from sim.act.realtime import ActionChunkScheduler


def subject():
    path = Path(__file__).with_name('handoff.py')
    if not path.exists():
        pytest.fail('two-second live ACT handoff module has not been connected')
    spec = importlib.util.spec_from_file_location('temporary_two_second_handoff', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def governor():
    return subject().TwoSecondHandoff([.2, .1, .003, 0.], .01, 0.,
                                     bounds=[[-.8, .8], [-.8, .8]])


def activate(g, now=0.):
    g.chunk_ack(True,now)
    if hasattr(g,'action_ready'): g.action_ready(now)


def apply(g, target, z=.002):
    native = native_reference(g.last_reference, target, 1, 1, .01,
                              .08, .12, .35, True, z)
    command = g.propose(g.last_time, native, target, z)
    row = {'t': g.last_time + .01, 'command': command.copy()}
    g.prepare_record(row)
    g.acknowledge(row)
    return row


def test_first_live_act_step_starts_with_bounded_velocity_not_the_native_speed_jump():
    g = governor(); activate(g)
    row = apply(g, [.5, .1, -.7, 1.])
    assert row['command'][0] == pytest.approx(.200005, abs=1e-12)
    assert row['command'][3] == pytest.approx(.00005, abs=1e-12)
    assert row['command'][2] == .002  # Learned Z must never enter the hook.
    assert row['handoff_smoothing']['mode'] == 'handoff_2s'


def test_diagonal_acceleration_is_a_vector_limit_not_two_independent_100_limits():
    g = governor(); activate(g)
    apply(g, [.6, .5, .4, 0.])
    assert np.linalg.norm(g.velocity_xy) == pytest.approx(.0005)


def test_direction_changes_during_the_handoff_do_not_reset_velocity():
    g = governor(); activate(g)
    for _ in range(25): apply(g, [.7, .1, .7, 1.])
    previous = g.velocity_xy.copy(); previous_yaw = g.yaw_velocity
    apply(g, [-.7, -.6, -.4, -1.])
    assert np.linalg.norm(g.velocity_xy - previous) <= .0005 + 1e-12
    assert abs(g.yaw_velocity - previous_yaw) <= .005 + 1e-12


def test_a_policy_hold_inside_the_window_brakes_without_an_instant_stop():
    g = governor(); activate(g)
    for _ in range(20): apply(g, [.7, .1, .003, 1.])
    previous = g.velocity_xy.copy()
    apply(g, g.last_reference.copy())
    assert g.velocity_xy[0] == pytest.approx(previous[0] - .0005)
    assert g.velocity_xy[0] > 0.


def test_waiting_for_the_first_accepted_chunk_does_not_consume_the_window():
    g = governor()
    for _ in range(30): apply(g, g.last_reference.copy())
    g.chunk_ack(False, .3)
    assert g.started_at is None
    activate(g,.3)
    row = apply(g, [.5, .1, .003, 1.])
    assert row['command'][0] == pytest.approx(.200005)
    assert row['handoff_smoothing']['elapsed_s'] == pytest.approx(0.)


def test_two_second_expiry_restores_the_exact_native_interpolator_even_on_reversal():
    g = governor(); activate(g)
    for _ in range(200): apply(g, [.7, .1, .003, 1.])
    start = g.last_reference.copy(); target = [-.7, -.6, -.5, -1.]
    expected = native_reference(start, target, 1, 1, .01, .08, .12, .35, True, .001)
    command = g.propose(2., expected, target, .001)
    assert np.array_equal(command, expected)
    row = {'t': 2.01, 'command': command.copy()}
    g.prepare_record(row); g.acknowledge(row)
    assert row['handoff_smoothing']['mode'] == 'native_restored'
    g.chunk_ack(True, 2.01)
    assert g.started_at == 0.  # Every later chunk must not restart the limiter.


def test_proposals_and_rejected_nonadvanced_steps_never_advance_velocity_or_clock():
    g = governor(); activate(g)
    native = native_reference(g.last_reference, [.5,.1,.003,0.],1,1,.01,.08,.12,.35,True,.002)
    first = g.propose(0., native, [.5,.1,.003,0.], .002)
    row = {'t': 0., 'command': first, 'control_step_advanced': False}
    g.prepare_record(row); g.acknowledge(row)
    assert g.last_time == 0. and np.array_equal(g.velocity_xy, [0., 0.])
    second = g.propose(0., native, [.5,.1,.003,0.], .002)
    assert np.array_equal(first, second)


def test_an_advanced_fallback_hold_is_identified_before_failing_closed():
    g = governor(); activate(g)
    g.propose(0., [.2008,.1,.002,.0035], [.5,.1,.003,1.], .002)
    row = {'t': .01, 'command': [.2,.1,.002,0.]}
    g.prepare_record(row)
    assert row['handoff_smoothing']['accepted'] is False
    with pytest.raises(RuntimeError, match='accepted command differs'):
        g.acknowledge(row)
    assert g.last_time == 0.
    row.update(normal_force=1.,tcp=np.array([.2,.1,.002,0.]))
    audit=subject().metrics([row])
    if 'accepted_ACT_control_steps' not in audit:
        pytest.fail('fallback hold incorrectly counts as ACT execution')
    assert audit['accepted_ACT_control_steps']==0
    assert audit['rejected_advanced_rows']==1


@pytest.mark.parametrize('now,target,z', [(.02,[.5,.1,.3,0.],.002),
    (0.,[float('nan'),.1,.3,0.],.002), (0.,[.5,.1,.3,0.],float('nan'))])
def test_bad_inputs_fail_before_mutating_the_accepted_command(now,target,z):
    g = governor(); activate(g)
    with pytest.raises((ValueError, RuntimeError)):
        g.propose(now, [.2008,.1,.002,0.], target, z)
    assert g.last_time == 0. and g.last_reference[0] == .2


def test_shortest_yaw_path_is_used_across_the_pi_wrap():
    m = subject(); g = m.TwoSecondHandoff([.2,.1,.003,3.13],.01,0.,bounds=[[-.8,.8],[-.8,.8]])
    activate(g); apply(g,[.2,.1,.003,-3.13])
    assert g.yaw_velocity == pytest.approx(.005)


def test_the_live_namespace_hook_leaves_the_native_module_and_z_ownership_unchanged():
    m = subject(); g = governor(); env = SimpleNamespace(time=0.)
    recorded = []
    driver = SimpleNamespace(__globals__={
        '_policy_reference_substep': native_reference,
        'ActionChunkScheduler': ActionChunkScheduler,
        '_trial_record': lambda row,current_env: recorded.append(dict(row)),
        '_trial_chunk_ack': lambda accepted: None})
    m.install(driver, env, g)
    driver.__globals__['_trial_chunk_ack'](True)
    if hasattr(g,'action_ready'): g.action_ready(0.)
    reference = driver.__globals__['_policy_reference_substep'](
        [.2,.1,.003,0.],[.5,.1,-.7,1.],1,4,.01,.08,.12,.35,True,.00123)
    assert reference[0] == pytest.approx(.200005)
    assert reference[2] == .00123
    row = {'t': .01, 'command': reference.copy()}
    env.time = .01; driver.__globals__['_trial_record'](row,env)
    assert recorded[0]['handoff_smoothing']['accepted'] is True
    assert g.last_time == .01
    from sim.act import evaluate
    assert evaluate._policy_reference_substep is native_reference


def test_actual_runner_builds_the_instrumented_live_entry_not_an_unused_helper():
    root = Path(__file__).parent
    path = root/'runner.py'
    if not path.exists(): pytest.fail('actual ACT runner has not been connected')
    spec = importlib.util.spec_from_file_location('phases', root/'phases.py')
    phase = importlib.util.module_from_spec(spec); sys.modules['phases'] = phase; spec.loader.exec_module(phase)
    spec = importlib.util.spec_from_file_location('temporary_handoff_driver', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    if not hasattr(runner,'build_act_driver'): pytest.fail('actual ACT output hook missing')
    from sim.config import load_config
    cfg = runner.make_config(load_config(str(runner.BATCH/'evaluation_config.yaml')),1)
    reference = np.array([.2,.1,.003,0.])
    env = SimpleNamespace(time=0.,control_dt=.01)
    loop = phase.TwoStageAdmittance(.01)
    context = dict(env=env,admittance=loop,relief=object(),z_nominal=.003,
        last_reference=reference,trace=[{'t':-.01,'command':reference.copy()},
        {'t':0.,'command':reference.copy()}],force_gate=object(),collection_timer=object())
    adapted,g = runner.build_act_driver(cfg,context)
    adapted.__globals__['_trial_chunk_ack'](True)
    if hasattr(g,'action_ready'): g.action_ready(0.)
    out = adapted.__globals__['_policy_reference_substep'](
        reference,[.5,.1,-.8,1.],1,4,.01,.08,.12,.35,True,.001)
    assert out[0] == pytest.approx(.200005)
    assert out[2] == .001
    assert context['admittance'] is loop
    assert context['accepted_chunks'] == 1
    assert g.started_at == 0.


def test_all_four_native_substeps_use_one_continuous_handoff_velocity_state():
    m = subject(); g = governor(); env = SimpleNamespace(time=0.)
    driver = SimpleNamespace(__globals__={
        '_policy_reference_substep': native_reference,
        'ActionChunkScheduler': ActionChunkScheduler,
        '_trial_record': lambda row,current_env: None,
        '_trial_chunk_ack': lambda accepted: None})
    m.install(driver,env,g);driver.__globals__['_trial_chunk_ack'](True)
    if hasattr(g,'action_ready'): g.action_ready(0.)
    frame_start=g.last_reference.copy();commands=[]
    for index in range(1,5):
        command=driver.__globals__['_policy_reference_substep'](
            frame_start,[.7,.1,.003,1.],index,4,.01,.08,.12,.35,True,.002)
        row={'t':env.time+.01,'command':command.copy()};env.time=row['t']
        driver.__globals__['_trial_record'](row,env);commands.append(command[0])
    assert commands == pytest.approx([.200005,.200015,.20003,.20005],abs=1e-12)


def test_small_nearby_target_brakes_to_rest_without_endpoint_snapping():
    g=governor();activate(g);positions=[]
    for _ in range(100):
        positions.append(apply(g,[.2005,.1,.003,0.])['command'][0])
    assert min(positions)>=.2 and max(positions)<=.2005+1e-12
    assert positions[-1]==pytest.approx(.2005,abs=1e-12)
    assert np.linalg.norm(g.velocity_xy)==0.


def test_scope_audit_separates_native_release_changes_from_the_limited_first_two_seconds():
    m=subject();g=governor();activate(g);rows=[]
    for _ in range(200):
        row=apply(g,[.7,.1,.003,1.]);row['normal_force']=2.
        row['tcp']=row['command'].copy();row['tcp'][2]+=.0001;rows.append(row)
    row=apply(g,[-.7,-.6,.003,-1.]);row['normal_force']=4.
    row['tcp']=row['command'].copy();row['tcp'][2]+=.0002;rows.append(row)
    result=m.metrics(rows)
    assert result['active_steps']==200 and result['restored_native_steps']==1
    assert result['active_limit_violations']==[]
    assert result['active_peak_wrist_N']==2. and result['restored_peak_wrist_N']==4.
    assert result['first_native_time']==pytest.approx(2.)
    assert result['posthandoff_actual_Z_error_max_mm']==pytest.approx(.2)


def test_workspace_boundary_failure_does_not_clamp_or_commit_an_unsafe_proposal():
    m=subject();g=m.TwoSecondHandoff([.8,.1,.003,0.],.01,0.,[[-.8,.8],[-.8,.8]],
        velocity_xy=[.04,0.]);activate(g)
    with pytest.raises(RuntimeError,match='inside workspace'):
        g.propose(0.,[.8,.1,.002,0.],[.8,.1,.003,0.],.002)
    assert g.last_time==0. and g.last_reference[0]==.8


def test_early_chunk_acceptance_waits_for_the_real_scheduler_executable_time():
    from sim.config import load_config
    cfg=load_config(str(Path(__file__).parent.parent/'batch_fine_teached_100k_120_unique_seeds_20261014/evaluation_config.yaml'))
    m=subject();g=governor();env=SimpleNamespace(time=0.)
    driver=SimpleNamespace(__globals__={'ActionChunkScheduler':ActionChunkScheduler,
        '_policy_reference_substep':native_reference,
        '_trial_record':lambda row,current_env:None,
        '_trial_chunk_ack':lambda accepted:None})
    m.install(driver,env,g)
    scheduler=driver.__globals__['ActionChunkScheduler'](cfg,allow_late=True)
    scheduler.reset(0.)
    assert scheduler.accept(0.,np.tile([.5,.1,.003,0.],(25,1)),0.)
    driver.__globals__['_trial_chunk_ack'](True)
    assert g.started_at is None, 'acceptance before valid_from must not start the2seconds'
    for _ in range(20):
        assert scheduler.action_for(env.time) is None
        apply(g,g.last_reference.copy());env.time=g.last_time
    assert g.started_at is None
    assert scheduler.action_for(.2) is not None
    assert g.started_at==pytest.approx(.2)
    for _ in range(200): apply(g,[.5,.1,.003,0.])
    row=apply(g,[-.5,.1,.003,0.])
    assert row['handoff_smoothing']['mode']=='native_restored'
    assert row['handoff_smoothing']['proposal_time']==pytest.approx(2.2)


def test_partial_actions_keep_selected_act_frames_even_if_the_episode_raises():
    m=subject()
    if not hasattr(m,'partial_actions'):pytest.fail('selected ACT frames are missing from failure backup')
    prefix=[[.2,.1,.003,0.]]
    g=governor();g.frame_targets=[np.array([.5,.1,-.7,1.]),np.array([.6,.1,-.8,1.])]
    actions=m.partial_actions(prefix,g)
    assert np.array_equal(actions,np.array([[.2,.1,.003,0.],[.5,.1,-.7,1.],[.6,.1,-.8,1.]],dtype=np.float32))
