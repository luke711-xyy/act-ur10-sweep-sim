import json

import pytest


def test_training_requires_matching_cpu_acceptance_by_default(tmp_path):
    from pac_act.training import cpu_acceptance_authorization

    identity = 'bundle-123'
    with pytest.raises(ValueError, match='acceptance report'):
        cpu_acceptance_authorization(identity, None)

    report = tmp_path / 'acceptance.json'
    report.write_text(json.dumps({'engineering_passed': True, 'bundle_identity': 'other'}))
    with pytest.raises(RuntimeError, match='matching'):
        cpu_acceptance_authorization(identity, report)

    report.write_text(json.dumps({'engineering_passed': True, 'bundle_identity': identity}))
    authorization = cpu_acceptance_authorization(identity, report)
    assert authorization['status'] == 'passed'
    assert authorization['engineering_passed'] is True
    assert authorization['bundle_identity'] == identity


def test_explicit_cpu_acceptance_waiver_is_scoped_and_never_claims_pass(tmp_path):
    from pac_act.training import cpu_acceptance_authorization, write_cpu_acceptance_waiver

    authorization = cpu_acceptance_authorization(
        'bundle-456', None, waive_cpu_acceptance=True,
    )
    assert authorization['status'] == 'waived'
    assert authorization['engineering_passed'] is None
    assert authorization['bundle_identity'] == 'bundle-456'
    assert authorization['scope'] == 'this formal training run only'

    out = tmp_path / 'training'
    out.mkdir()
    write_cpu_acceptance_waiver(out, authorization)
    saved = json.loads((out / 'cpu_acceptance_waiver.json').read_text())
    assert saved == authorization
    with pytest.raises(FileExistsError):
        write_cpu_acceptance_waiver(out, authorization)


def test_cpu_acceptance_waiver_cannot_be_combined_with_report(tmp_path):
    from pac_act.training import cpu_acceptance_authorization

    report = tmp_path / 'acceptance.json'
    report.write_text('{}')
    with pytest.raises(ValueError, match='cannot be combined'):
        cpu_acceptance_authorization('bundle-789', report, waive_cpu_acceptance=True)


def test_cli_allows_only_explicit_train_waiver(monkeypatch, tmp_path):
    import pac_act.lifecycle
    import pac_act.training
    from pac_act.cli import main

    calls = []
    monkeypatch.setattr(pac_act.lifecycle, 'require_runtime', lambda: None)
    monkeypatch.setattr(pac_act.training, 'train', lambda *a, **kw: calls.append((a, kw)))
    args = ['--mode', 'train', '--bundle', str(tmp_path / 'bundle'), '--out', str(tmp_path / 'run'),
            '--device', 'cuda', '--acknowledge-gpu-training', '--waive-cpu-acceptance',
            '--skip-baseline-validation']
    assert main(args) == 0
    assert calls[0][0] == (tmp_path / 'bundle', tmp_path / 'run', None)
    assert calls[0][1]['waive_cpu_acceptance'] is True
    assert calls[0][1]['skip_baseline_validation'] is True

    with pytest.raises(SystemExit):
        main(['--mode', 'train', '--bundle', str(tmp_path / 'bundle'), '--out', str(tmp_path / 'run'),
              '--device', 'cuda', '--acknowledge-gpu-training'])
    with pytest.raises(SystemExit):
        main(['--mode', 'resume', '--bundle', str(tmp_path / 'bundle'), '--out', str(tmp_path / 'run'),
              '--device', 'cuda', '--acknowledge-gpu-training', '--waive-cpu-acceptance'])
    with pytest.raises(SystemExit):
        main(['--mode', 'resume', '--bundle', str(tmp_path / 'bundle'), '--out', str(tmp_path / 'run'),
              '--device', 'cuda', '--acknowledge-gpu-training', '--acceptance-report',
              str(tmp_path / 'acceptance.json'), '--skip-baseline-validation'])


def test_skipping_baseline_never_calls_evaluator_and_is_explicit():
    from pac_act.training import run_baseline_validation

    calls = []
    record = run_baseline_validation(lambda *a, **kw: calls.append((a, kw)), skip=True)
    assert calls == []
    assert record['status'] == 'skipped'
    assert record['scope'] == 'this formal training run only'
    assert record['operator_direction'] == 'User directed formal training to skip the baseline evaluation.'

    run_baseline_validation(lambda *a, **kw: calls.append((a, kw)), skip=False)
    assert calls == [(('baseline',), {'media': 'all', 'baseline': True})]


def test_dashboard_exposes_waiver_without_calling_it_a_pass(tmp_path):
    from pac_act.dashboard import snapshot
    from pac_act.storage import write_json

    write_json(tmp_path / 'run.json', {'cpu_acceptance': {
        'status': 'waived', 'engineering_passed': None, 'bundle_identity': 'bundle-abc',
    }})
    write_json(tmp_path / 'status.json', {'state': 'training', 'training_started': True})
    value = snapshot(tmp_path)
    assert value['run']['cpu_acceptance']['status'] == 'waived'
    assert value['run']['cpu_acceptance']['engineering_passed'] is None
