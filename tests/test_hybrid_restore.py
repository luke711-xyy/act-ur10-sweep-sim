"""The published pre-DINO Hybrid is the execution baseline, not a pure ACT default."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT / 'hybrid_trial4_snapshot'


def test_disk_sim_is_the_complete_pre_dino_source_set():
    from residual_rl.runtime import HYBRID
    expected = json.loads((HYBRID / 'metadata.json').read_text())['source_fingerprints']
    actual = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (PROJECT / 'sim').rglob('*.py')}
    assert len(actual) == 67
    assert actual == expected


def test_published_hybrid_sources_are_present_without_rewriting_the_originals():
    manifest = SNAPSHOT / 'source_manifest.json'
    assert manifest.is_file(), 'published independent Hybrid snapshot is absent'
    data = json.loads(manifest.read_text())
    assert data['branch'] == 'feat/hybrid-astar-act-trial4'
    for relative, entry in data['files'].items():
        assert hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest() == entry['sha256']
    from residual_rl.runtime import load_formal
    assert load_formal().PROFILE == data['profile']


def test_native_restoration_inspection_does_not_import_rl_or_start_physics():
    result = subprocess.run([sys.executable, '-B', '-c', '''
import sys
from scripts.verify_restored_hybrid import inspect_baseline
info=inspect_baseline()
assert info['reference_commit']=='eb887843c9849d4a72be116bb87c44519dc7ef74'
assert info['sim_source_count']==67 and info['execution_started'] is False
assert not any(n=='residual_rl' or n.startswith('residual_rl.') for n in sys.modules)
assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
'''], cwd=PROJECT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rl_reconnection_gate_rejects_prefix_only_safety_or_missing_z_owner():
    import importlib
    import pytest
    try:
        module = importlib.import_module('scripts.verify_restored_hybrid')
    except ModuleNotFoundError:
        module = None
    validate = getattr(module, 'validate_restoration', None)
    assert callable(validate), 'restoration validation gate is absent'
    row=dict(ACT_executed=True, native_restored_steps=5, invalid_samples=0,
             step_failures=0, safety_reason=None, Z_policy_rows=0,
             handoff_limit_violations=[], rejected_advanced_rows=[], handoff_ready=True)
    assert validate(row) is True
    for change in (dict(ACT_executed=False), dict(native_restored_steps=0),
                   dict(safety_reason='force ceiling'), dict(Z_policy_rows=1),
                   dict(invalid_samples=1), dict(handoff_limit_violations=[{'t': 1.}]),
                   dict(rejected_advanced_rows=[{'t': 1.}])):
        assert validate(dict(row, **change)) is False
    with pytest.raises(KeyError): validate({})


def test_media_error_preserves_measured_outcome_and_records_needs_attention(tmp_path, monkeypatch):
    import pytest
    from scripts import verify_restored_hybrid as script
    from sim.act import replay
    finish = getattr(script, 'finish_media', None)
    assert callable(finish), 'media completion must record failures without rerunning physics'
    original = b'{"success":false,"final_collected":2}'
    (tmp_path / 'result.json').write_bytes(original)
    def failed_saved_state_export(*args, **kwargs):
        raise RuntimeError('renderer unavailable')
    monkeypatch.setattr(replay, 'save_inference_replay', failed_saved_state_export)
    with pytest.raises(RuntimeError, match='renderer unavailable'):
        finish(tmp_path, None, 3152200, None, 'step_100000')
    assert (tmp_path / 'result.json').read_bytes() == original
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['state'] == 'needs_attention' and status['stage'] == 'media'
    assert status['physical_execution_complete'] is True
    assert status['automatic_retry_forbidden'] is True
