"""Batch integrity checks; no model loading, physics, replay or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/evaluate_hybrid_paired_batch.py'


@pytest.fixture
def api():
    if not SCRIPT.exists():
        return SimpleNamespace()
    spec = importlib.util.spec_from_file_location('hybrid_batch_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(api, name, *args):
    function = getattr(api, name, None)
    assert callable(function), f'{name} batch behavior not implemented'
    return function(*args)


def plan():
    return [dict(key=f'five_view_100k:n{i % 6 + 1}:l{i:02}', seed=2120000+i,
                 target_count=i % 6+1, layout_fingerprint=f'{i:064x}',
                 model_path='/frozen/step_100000', checkpoint_step=100000,
                 initial_object_poses=[[0, 0, 0, 1, 0, 0, 0]]*6)
            for i in range(120)]


def row(spec, success=False, count=0):
    return dict(key=spec['key'], seed=spec['seed'], target_count=spec['target_count'],
                layout_fingerprint=spec['layout_fingerprint'],
                model_path=spec['model_path'], checkpoint_step=100000,
                model_label='hybrid_astar_act_trial4_100k', success=success,
                final_collected=count, opportunities=1, planning_calls=1,
                physical_attempts=1, no_plan=False, frame_budget=2000,
                sampled_frames=30, media_complete=True)


def test_same_120_plan_is_accepted_without_changing_order(api):
    data = plan()
    call(api, 'validate_plan', data)
    assert data[5]['seed'] == 2120005
    assert data[5]['target_count'] == 6


@pytest.mark.parametrize('mutation', ['duplicate_seed', 'duplicate_layout', 'wrong_goal',
                                      'short', 'wrong_checkpoint'])
def test_nonpaired_or_unbalanced_plan_is_rejected(api, mutation):
    data = plan()
    if mutation == 'duplicate_seed': data[1]['seed'] = data[0]['seed']
    if mutation == 'duplicate_layout': data[1]['layout_fingerprint'] = data[0]['layout_fingerprint']
    if mutation == 'wrong_goal': data[0]['target_count'] = 6
    if mutation == 'short': data.pop()
    if mutation == 'wrong_checkpoint': data[0]['checkpoint_step'] = 103243
    with pytest.raises(ValueError): call(api, 'validate_plan', data)


@pytest.mark.parametrize('target', range(1, 7))
def test_all_goals_get_trial4_configuration_with_unchanged_safety(api, target):
    from sim.config import load_config
    frozen = load_config('configs/default.yaml')
    before = frozen.to_dict()
    cfg = call(api, 'candidate_config', frozen, target)
    assert cfg.task.target_count == target
    assert cfg.controller.desired_force == 1
    assert cfg.controller.delta_z_limit == .002
    assert (cfg.controller.admittance.m, cfg.controller.admittance.b,
            cfg.controller.admittance.k) == (10, 400, 10)
    assert cfg.controller.delta_z_upward_rate_limit == .02
    assert cfg.controller.delta_z_downward_rate_limit == .1
    assert cfg.controller.safe_max_force == frozen.controller.safe_max_force
    assert cfg.controller.safe_force_dwell_steps == frozen.controller.safe_force_dwell_steps
    assert cfg.controller.force_sensor == frozen.controller.force_sensor
    assert frozen.to_dict() == before


def test_marker_consumes_the_opportunity_even_if_result_is_missing(api, tmp_path):
    spec = plan()[0]
    call(api, 'claim_attempt', tmp_path, spec)
    with pytest.raises(FileExistsError): call(api, 'claim_attempt', tmp_path, spec)
    with pytest.raises(RuntimeError, match='attempt'): call(api, 'require_clean_start', tmp_path)
    assert not (tmp_path / 'episodes.jsonl').exists()


def test_failed_episode_is_durably_committed_and_cannot_be_replaced(api, tmp_path):
    spec = plan()[0]
    call(api, 'claim_attempt', tmp_path, spec)
    call(api, 'commit_result', tmp_path, spec, row(spec, False, 2))
    rows = [json.loads(s) for s in (tmp_path / 'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['success'] is False and rows[0]['final_collected'] == 2
    with pytest.raises(RuntimeError): call(api, 'commit_result', tmp_path, spec, row(spec, True, 1))
    assert (tmp_path / 'episodes.jsonl').read_text().count('\n') == 1


@pytest.mark.parametrize('field,value', [('seed', 99), ('layout_fingerprint', 'different'),
                                       ('physical_attempts', 2), ('opportunities', 2),
                                       ('sampled_frames', 2001), ('final_collected', None),
                                       ('media_complete', False)])
def test_invalid_or_unknown_result_never_enters_ledger(api, tmp_path, field, value):
    spec = plan()[0]
    call(api, 'claim_attempt', tmp_path, spec)
    result = row(spec); result[field] = value
    with pytest.raises(ValueError): call(api, 'commit_result', tmp_path, spec, result)
    assert not (tmp_path / 'episodes.jsonl').exists()


def test_no_plan_is_one_failed_opportunity_not_a_skipped_seed(api):
    specs = plan()
    result = row(specs[0]); result.update(no_plan=True, physical_attempts=0, sampled_frames=0)
    summary = call(api, 'summarize', [result], specs)
    assert summary['completed'] == 1 and summary['opportunities'] == 1
    assert summary['no_plan_failures'] == 1 and summary['successes'] == 0
    assert summary['complete'] is False


def test_full_complete_requires_exact_pairing_not_only_120_rows(api):
    specs = plan(); rows = [row(s) for s in specs]
    assert call(api, 'summarize', rows, specs)['complete'] is True
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError): call(api, 'summarize', rows, specs)


def test_cleanup_deletes_only_this_batch_verified_video_and_keeps_states(api, tmp_path):
    video = tmp_path / 'episodes/seed_2120000/videos/evaluation.mp4'
    video.parent.mkdir(parents=True); video.write_bytes(b'verified-media')
    state = video.parent.parent / 'physical_trace.json.gz'; state.write_bytes(b'keep')
    verification = dict(bytes=14, sha256=hashlib.sha256(b'verified-media').hexdigest())
    report = call(api, 'remove_verified_video', tmp_path, video, verification)
    assert not video.exists() and state.read_bytes() == b'keep'
    assert report['deleted_files'] == 1 and report['freed_bytes'] == 14


def test_cleanup_refuses_unverified_changed_media(api, tmp_path):
    video = tmp_path / 'episodes/seed_2120000/videos/evaluation.mp4'
    video.parent.mkdir(parents=True); video.write_bytes(b'changed')
    with pytest.raises(ValueError):
        call(api, 'remove_verified_video', tmp_path, video, dict(bytes=7, sha256='bad'))
    assert video.exists()


def test_cleanup_refuses_media_outside_its_exact_episode_scope(api, tmp_path):
    video = tmp_path / 'manual.mp4'; video.write_bytes(b'keep')
    with pytest.raises(ValueError):
        call(api, 'remove_verified_video', tmp_path, video,
             dict(bytes=4, sha256=hashlib.sha256(b'keep').hexdigest()))
    assert video.exists()


def test_direct_worker_without_parent_ownership_cannot_consume_a_layout(api, tmp_path):
    with pytest.raises(RuntimeError, match='parent'):
        call(api, 'authorize_worker', tmp_path)
    assert not (tmp_path / 'attempts').exists()


def test_worker_checks_parent_pid_and_exclusive_batch_lock(api, tmp_path):
    import fcntl
    import os
    (tmp_path / 'started.json').write_text(json.dumps(dict(pid=os.getppid())))
    with pytest.raises(RuntimeError, match='lock'):
        call(api, 'authorize_worker', tmp_path)
    with (tmp_path / 'batch.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        call(api, 'authorize_worker', tmp_path)
    (tmp_path / 'started.json').write_text(json.dumps(dict(pid=-1)))
    with pytest.raises(RuntimeError, match='parent'):
        call(api, 'authorize_worker', tmp_path)


def test_input_hash_changed_after_preparation_is_rejected(api, tmp_path):
    sealed = tmp_path / 'metadata.json'; sealed.write_text('{"profile":"before"}')
    (tmp_path / 'seal.json').write_text(json.dumps(dict(metadata_sha256=hashlib.sha256(sealed.read_bytes()).hexdigest())))
    call(api, 'validate_metadata_seal', tmp_path)
    sealed.write_text('{"profile":"after"}')
    with pytest.raises(RuntimeError, match='metadata'):
        call(api, 'validate_metadata_seal', tmp_path)


@pytest.mark.parametrize('target', range(1, 7))
def test_real_trial4_act_hook_preserves_Z_on_each_formal_target(api, target):
    from sim.config import load_config
    from sim.act.rollout import CollectionTerminationTracker
    from sim.controllers.admittance import OverforceReliefController1D, OverforceReliefParams
    module = call(api, 'runtime')
    cfg = call(api, 'candidate_config', load_config('configs/default.yaml'), target)
    command = __import__('numpy').array([.2, .1, .001, 0.])
    env = SimpleNamespace(control_dt=.01, time=.02, tcp=lambda: command[:3].copy())
    loop = module.TwoStageAdmittance(.01, module.TRIAL_SPEC)
    context = dict(env=env, admittance=loop, z_limiter=module.ZCommandLimiter(.001, .01, initial_time=.02),
                   last_reference=command.copy(), trace=[dict(t=.01, command=command.copy()), dict(t=.02, command=command.copy())],
                   force_gate=module.old.geometry.ForceGate(20., 3), collection_timer=CollectionTerminationTracker(target_count=target),
                   actions=[], act_trace=[], invalid_samples=[], step_failures=[], peak_force=0., ready=True,
                   last_guard_time=None, safety_reason='', collection_reason='', act_outer_frames=0, act_queries=0, accepted_chunks=0,
                   z_nominal=.001, relief=OverforceReliefController1D(OverforceReliefParams(), .01), history=[],
                   prefix_frames=0, handoff_time=.02, handoff_force=1., handoff_tcp=command[:3].copy(),
                   contact_time=.02, first_contact_position=command[:3].copy())
    adapted, governor = call(api, 'build_formal_act_driver', module, cfg, context)
    z = adapted.__globals__['_contact_z_command'](cfg, loop, .001, 0., None)
    assert z == pytest.approx(.00099, abs=1e-12)
    assert adapted.__globals__['_handoff']['admittance'] is loop
    assert (loop.params.m, loop.params.b, loop.params.k) == (10., 400., 10.)
    for i in range(11): loop.observe(i*.01, .5)
    assert (loop.params.m, loop.params.b, loop.params.k) == (10., 800., 10.)
    assert governor.started_at is None


def test_frozen_dependencies_include_imported_stage_and_follow_controllers(api):
    module = call(api, 'runtime')
    paths = call(api, 'dependency_hashes', module)
    exports = api.RUNTIME_ROOT / 'vendor'
    required = [
        'admittance_2mm_target_gain_search_goal1_20261014/phases.py',
        'hybrid_astar_act_gate1mm_acc50_2s_goal1_20261014/phases.py',
        'admittance_wrist1n_small_motion_20261014/control.py',
        'admittance_wrist1n_small_motion_20261014/follow.py',
        'hybrid_astar_act_wrist1n_twostage_20261014/phases.py',
    ]
    assert all(str(exports / p) in paths for p in required)


def _native_swallow_prediction():
    try:
        raise ValueError('prediction failed')
    except Exception:
        return 'silently held'


def _native_hold_fallback(step):
    try:
        raise ValueError('command rejected')
    except Exception:
        step()
        return 'fallback executed'


def test_prediction_exception_cannot_be_swallowed_or_scored_as_timeout(api):
    strict = call(api, 'strict_exception_function', _native_swallow_prediction)
    with pytest.raises(ValueError, match='prediction failed'): strict()


def test_rejected_command_cannot_execute_a_hold_fallback(api):
    executions = []
    strict = call(api, 'strict_exception_function', _native_hold_fallback)
    with pytest.raises(ValueError, match='command rejected'): strict(lambda: executions.append('extra execution'))
    assert executions == []


@pytest.mark.parametrize('error', ["ValueError('observer failed')", "RuntimeError('history advance failed')"])
def test_uninstrumented_acquisition_exception_cannot_be_committed_as_normal_failure(api, error):
    context = dict(ready=False, step_failures=[], invalid_samples=[], safety_reason='', collection_reason='', error=error)
    with pytest.raises(RuntimeError, match='unexpected'): call(api, 'validate_acquisition', context)


@pytest.mark.parametrize('reason', ['stable wrist-force band not established within 30 s; ACT sweep not started',
                                   'shared frame budget exhausted before stable wrist contact'])
def test_expected_contact_timeout_or_budget_remains_a_real_failure_outcome(api, reason):
    context = dict(ready=False, step_failures=[], invalid_samples=[], safety_reason='', collection_reason='',
                   error=repr(RuntimeError(reason)), terminal_reason=reason)
    call(api, 'validate_acquisition', context)


def test_replay_requires_all_saved_states_not_command_resimulation(api):
    call(api, 'require_saved_replay_states', [dict(replay_state=dict(spec=1, values=[0.]))])
    with pytest.raises(RuntimeError, match='saved'):
        call(api, 'require_saved_replay_states', [dict(replay_state=dict(spec=1, values=[0.])), dict(command=[0, 0, 0, 0])])


@pytest.mark.parametrize('flag', ['safety_reason', 'collection_reason'])
def test_terminal_flag_cannot_hide_unexpected_observer_failure(api, flag):
    context = dict(ready=False, step_failures=[], invalid_samples=[], safety_reason='', collection_reason='',
                   error="ValueError('observer failed')", terminal_reason='observed terminal')
    context[flag] = 'observed terminal'
    with pytest.raises(RuntimeError, match='unexpected'): call(api, 'validate_acquisition', context)


@pytest.mark.parametrize('phase', ['prefix terminated before ACT handoff: ', 'acquisition terminated: '])
def test_only_the_exact_known_terminal_exception_is_a_scored_failure(api, phase):
    reason = 'unexpected component entered collection region'
    context = dict(ready=False, step_failures=[], invalid_samples=[], safety_reason='', collection_reason=reason,
                   error=repr(RuntimeError(phase+reason)), terminal_reason=reason)
    call(api, 'validate_acquisition', context)
    context['error'] = "RuntimeError('history failed after terminal observation')"
    with pytest.raises(RuntimeError, match='unexpected'): call(api, 'validate_acquisition', context)


def test_numpy_handoff_audit_can_be_saved_without_losing_nanosecond_integers(api, tmp_path):
    import numpy as np
    value = dict(handoff_audit=dict(release_step=dict(command=np.array([.2, .1, .001, 0.]),
                accepted=np.bool_(True))), count=np.int64(2), force=np.float64(6.4),
                mtime=np.int64(1791906334792456398))
    call(api, 'write_json', tmp_path/'result.json', value)
    actual = json.loads((tmp_path/'result.json').read_text())
    assert actual['handoff_audit']['release_step']['command'] == [.2, .1, .001, 0.]
    assert actual['handoff_audit']['release_step']['accepted'] is True
    assert actual['count'] == 2 and actual['mtime'] == 1791906334792456398


def test_nonfinite_numpy_is_rejected_without_overwriting_previous_evidence(api, tmp_path):
    import numpy as np
    path = tmp_path/'result.json'; path.write_text('{"retain":true}')
    with pytest.raises(ValueError): call(api, 'write_json', path, dict(force=np.array([float('nan')])))
    assert path.read_text() == '{"retain":true}'


def test_truncated_native_result_recovers_only_fully_written_fields(api):
    value = call(api, 'complete_json_prefix', '{"success":false,"final_collected":2,"handoff_audit":{"command":')
    assert value == {'success': False, 'final_collected': 2}


def test_truncated_success_or_count_is_not_invented(api):
    value = call(api, 'complete_json_prefix', '{"key":"k","final_collected":')
    assert value == {'key': 'k'}


def test_completed_first_record_selects_exactly_the_remaining_119(api, tmp_path):
    specs = plan(); first = row(specs[0], False, 2)
    call(api, 'claim_attempt', tmp_path, specs[0]); call(api, 'commit_result', tmp_path, specs[0], first)
    todo = call(api, 'select_unattempted', tmp_path, specs)
    assert len(todo) == 119 and todo[0]['seed'] == 2120001
    assert not any(s['seed'] == 2120000 for s in todo)


def test_orphan_attempt_still_prevents_continuation(api, tmp_path):
    specs = plan()
    call(api, 'claim_attempt', tmp_path, specs[0])
    with pytest.raises(RuntimeError, match='unfinished'): call(api, 'select_unattempted', tmp_path, specs)


def test_completed_record_without_a_consumed_opportunity_is_not_accepted(api, tmp_path):
    specs = plan(); (tmp_path/'episodes.jsonl').write_text(json.dumps(row(specs[0]))+'\n')
    with pytest.raises(RuntimeError, match='marker'): call(api, 'select_unattempted', tmp_path, specs)


def recovery_fixture():
    import gzip
    import os
    supplied = os.environ.get('SWEEPSIM_HISTORICAL_HYBRID_BATCH')
    if not supplied:
        pytest.skip('optional historical recovery fixture was not transferred; set SWEEPSIM_HISTORICAL_HYBRID_BATCH')
    root = Path(supplied)
    episode = root/'episodes/seed_2120000'
    spec = json.loads((root/'episode_plan.json').read_text())[0]
    backup = json.load(gzip.open(episode/'exception_trace.json.gz', 'rt'))
    initial = json.loads((episode/'initial_layout.json').read_text())['poses']
    return root, episode, spec, backup, initial


def test_real_first_failure_recovers_from_states_without_any_policy_or_execution(api, monkeypatch):
    from sim.environments.sweep_env import SweepEnv
    from sim.act import evaluate, expert
    monkeypatch.setattr(SweepEnv, '__init__', lambda *a, **k: pytest.fail('recovery started physics'))
    monkeypatch.setattr(evaluate, 'run_act_episode', lambda *a, **k: pytest.fail('recovery ran ACT'))
    monkeypatch.setattr(expert, 'expert_plan_candidates', lambda *a, **k: pytest.fail('recovery replanned'))
    root, episode, spec, backup, initial = recovery_fixture()
    header = call(api, 'complete_json_prefix', (episode/'.physical_result.json.tmp').read_text())
    recovered, result = call(api, 'recover_measurements', spec, header, backup['trace'], initial)
    assert recovered['success'] is False and recovered['final_collected'] == 2
    assert recovered['sampled_frames'] == 577
    assert recovered['handoff_audit']['active_steps'] == 200
    assert recovered['peak_filtered_wrist_N'] == pytest.approx(6.409381526276512)
    assert len(result.trace) == 2307 and result.final_collected == 2
    assert recovered['scheduler_queries'] is None
    assert recovered['physical_attempts'] == 1


@pytest.mark.parametrize('mutation', ['missing_count', 'wrong_count', 'missing_state'])
def test_incomplete_or_conflicting_recovery_evidence_is_rejected(api, mutation):
    root, episode, spec, backup, initial = recovery_fixture()
    header = call(api, 'complete_json_prefix', (episode/'.physical_result.json.tmp').read_text())
    if mutation == 'missing_count': header.pop('final_collected')
    if mutation == 'wrong_count': header['final_collected'] = 1
    if mutation == 'missing_state': backup['trace'][-1].pop('replay_state')
    with pytest.raises((RuntimeError, ValueError)):
        call(api, 'recover_measurements', spec, header, backup['trace'], initial)


def test_continuation_requires_authorization_and_can_only_claim_once(api, tmp_path):
    specs = plan(); first = row(specs[0], False, 2)
    call(api, 'claim_attempt', tmp_path, specs[0]); call(api, 'commit_result', tmp_path, specs[0], first)
    with pytest.raises(RuntimeError, match='authorization'): call(api, 'claim_continuation', tmp_path, specs)
    (tmp_path/'recovery_authorization.json').write_text(json.dumps(dict(recovered_seed=2120000, continue_unattempted=119)))
    call(api, 'claim_continuation', tmp_path, specs)
    with pytest.raises(RuntimeError, match='already'): call(api, 'claim_continuation', tmp_path, specs)
    assert len(list((tmp_path/'attempts').glob('seed_*.json'))) == 1
