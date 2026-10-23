import json
from pathlib import Path

import pytest


def test_six_demo_entry_rejects_failed_contact_gate(tmp_path):
    assert Path('scripts/collect_rudra_six_demos.py').exists()
    from scripts.collect_rudra_six_demos import checked_configuration
    path=tmp_path/'report.json'
    path.write_text(json.dumps({'reports':{'native':{'passed':False}}}))
    with pytest.raises(ValueError,match='contact test'):
        checked_configuration(path)


def test_six_demo_plan_has_one_independent_layout_per_target():
    assert Path('scripts/collect_rudra_six_demos.py').exists()
    from scripts.collect_rudra_six_demos import six_demo_plan
    plan=six_demo_plan()
    assert [s.target_count for s in plan]==list(range(1,7))
    assert len({s.seed for s in plan})==6


def test_known_no_plan_failure_can_use_the_original_seed_window(monkeypatch):
    from scripts import collect_rudra_six_demos as entry
    def no_plan(*args, **kwargs):
        raise RuntimeError('expert candidate search returned no executable plan')
    monkeypatch.setattr(entry, 'run_expert_episode', no_plan)
    result, failure = entry.run_candidate(object(), 410000)
    assert result is None
    assert failure['collected'] is None
    assert failure['frames'] is None
    assert 'no executable plan' in failure['reason']


def test_unknown_execution_exception_is_not_retried(monkeypatch, tmp_path):
    from scripts import collect_rudra_six_demos as entry
    def bad_prediction(*args, **kwargs):
        raise RuntimeError('unrecognized numerical failure')
    monkeypatch.setattr(entry, 'run_expert_episode', bad_prediction)
    (tmp_path/'progress.json').write_text(json.dumps({'state':'running','completed':1}))
    with pytest.raises(RuntimeError, match='unrecognized numerical'):
        with entry.terminal_failure_status(tmp_path):
            entry.run_candidate(object(), 410000)
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['state']=='failed'
    assert progress['completed']==1
    assert progress['error_type']=='RuntimeError'


def test_recording_exception_is_terminal_and_preserves_saved_count(tmp_path):
    from scripts import collect_rudra_six_demos as entry
    (tmp_path/'progress.json').write_text(json.dumps({'state':'running','completed':2}))
    with pytest.raises(OSError):
        with entry.terminal_failure_status(tmp_path):
            raise OSError('recording failed')
    assert json.loads((tmp_path/'progress.json').read_text())['state']=='failed'
