"""Pinned pre-DINO imports: no model loading, physics rollout or learning."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def api(name, *args, **kwargs):
    try:
        module = importlib.import_module('residual_rl.baseline')
    except ModuleNotFoundError:
        module = None
    function = getattr(module, name, None)
    assert callable(function), f'baseline.{name} missing'
    return function(*args, **kwargs)


def test_selected_sources_match_all_67_historical_fingerprints_without_disk_rollback():
    from residual_rl.runtime import HYBRID
    expected = json.loads((HYBRID/'metadata.json').read_text())['source_fingerprints']
    before = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (PROJECT/'sim').rglob('*.py')}
    snapshot = api('pre_dino_sources', PROJECT, expected)
    assert len(snapshot.sources) == 67
    assert {p: hashlib.sha256(s).hexdigest() for p, s in snapshot.sources.items()} == expected
    assert snapshot.manifest()['git_fallback_files'] == ['sim/act/policy.py', 'sim/act/train.py']
    assert snapshot.manifest()['commit'] == '9102680d8e018b225ab34d36422572cd4b547166'
    after = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (PROJECT/'sim').rglob('*.py')}
    assert after == before
    assert 'sim.act.dinov3' not in sys.modules


def test_wrong_historical_fingerprint_is_rejected():
    with pytest.raises(RuntimeError, match='source drift'):
        api('pre_dino_sources', PROJECT, {'sim/act/policy.py': '0'*64})


@pytest.mark.parametrize('path', ['../outside.py', '/tmp/outside.py', 'sim/../../outside.py'])
def test_manifest_cannot_read_outside_sim_namespace(path):
    with pytest.raises(ValueError, match='source path'):
        api('pre_dino_sources', PROJECT, {path: '0'*64})


def run_fresh(source):
    return subprocess.run([sys.executable, '-B', '-c', source], cwd=PROJECT,
                          capture_output=True, text=True, timeout=30)


def test_fresh_process_imports_reviewed_native_functions_not_dino_and_supports_ast_adapters():
    result = run_fresh('''
import inspect, json, sys
from residual_rl.runtime import HYBRID, PROJECT, identity, load_formal
from residual_rl.baseline import pre_dino_sources
meta=json.loads((HYBRID/'metadata.json').read_text())
snap=pre_dino_sources(PROJECT,meta['source_fingerprints'])
snap.activate()
from sim.act import evaluate, policy
assert policy.build_act_policy is evaluate.build_act_policy
assert 'dinov3' not in inspect.getsource(policy.build_act_policy).lower()
assert policy.__loader__.get_source(policy.__name__).encode()==snap.sources['sim/act/policy.py']
assert inspect.getsource(policy).encode()==snap.sources['sim/act/policy.py']
from pathlib import Path
assert Path(policy.__file__).parent==PROJECT/'sim/act'
assert not Path(policy.__file__).exists()
from sim.model import scene_builder
assert scene_builder.MENAGERIE_ROOT==PROJECT/'sim/model/assets/universal_robots_ur10e'
try: import sim.act.dinov3
except ModuleNotFoundError: pass
else: raise AssertionError('unapproved DINO module imported')
formal=load_formal()
strict=formal.strict_exception_function(evaluate.run_act_episode)
assert strict is not evaluate.run_act_episode
driver=formal.runtime()
assert driver.NATIVE_CONTACT_Z is evaluate._contact_z_command
assert identity()['baseline']['commit']==snap.manifest()['commit']
assert 'sim.act.dinov3' not in sys.modules
print('legacy imports verified; no model loaded or execution attempted')
''')
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'legacy imports verified' in result.stdout


def test_foreign_preloaded_sim_is_rejected_not_replaced():
    result = run_fresh('''
import json, sys
from residual_rl.runtime import HYBRID, PROJECT
from residual_rl.baseline import pre_dino_sources
import sim.act.policy as current
snap=pre_dino_sources(PROJECT,json.loads((HYBRID/'metadata.json').read_text())['source_fingerprints'])
try: snap.activate()
except RuntimeError as e: assert 'already imported' in str(e)
else: raise AssertionError('mixed legacy/current namespace was accepted')
assert sys.modules['sim.act.policy'] is current
''')
    assert result.returncode == 0, result.stdout+result.stderr


def test_identity_and_default_inspect_use_selected_baseline_without_loading_sim():
    result = run_fresh('''
import sys
from residual_rl.runtime import identity
frozen=identity()
assert len(frozen['source_fingerprints'])==67
assert frozen['baseline']['commit']=='9102680d8e018b225ab34d36422572cd4b547166'
assert frozen['baseline']['disk_source_audit']['actual_files']==69
assert not any(n=='sim' or n.startswith('sim.') for n in sys.modules)
assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
''')
    assert result.returncode == 0, result.stdout+result.stderr
    inspected = subprocess.run([sys.executable, '-B', 'scripts/train_residual_sac.py'],
                              cwd=PROJECT, capture_output=True, text=True, timeout=30)
    assert inspected.returncode == 0, inspected.stderr
    payload = json.loads(inspected.stdout)
    assert payload['execution_started'] is False
    assert payload['baseline_source_audit']['effective_matches'] is True
    assert payload['baseline_source_audit']['disk_source_audit']['matches'] is False


@pytest.fixture
def isolated_metadata(monkeypatch, tmp_path):
    """Small evidence fixture only; never mutate historical files/checkpoints."""
    from residual_rl import runtime
    metadata = json.loads((runtime.HYBRID/'metadata.json').read_text())
    dependency = tmp_path/'dependency.py'
    dependency.write_text('# frozen dependency\n')
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    weight = checkpoint/'model.dat'
    weight.write_bytes(b'fixture, not a loaded model')
    stat = weight.stat()
    metadata['checkpoint'] = dict(path=str(checkpoint),
        checkpoint_files={'model.dat': [stat.st_size, stat.st_mtime_ns]})
    metadata['frozen_sha256'] = {str(dependency): hashlib.sha256(dependency.read_bytes()).hexdigest()}
    batch = tmp_path/'batch'
    batch.mkdir()
    def seal():
        path = batch/'metadata.json'
        path.write_text(json.dumps(metadata))
        (batch/'seal.json').write_text(json.dumps({'metadata_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    seal()
    monkeypatch.setattr(runtime, 'HYBRID', batch)
    return metadata, dependency, weight, seal


@pytest.mark.parametrize('change,expected_message', [
    ('seal', 'metadata changed'), ('profile', 'approved trial4'),
    ('dependency', 'frozen input changed'), ('checkpoint', 'frozen checkpoint changed')])
def test_legacy_selection_keeps_original_identity_guards(isolated_metadata, change, expected_message):
    from residual_rl import runtime
    metadata, dependency, weight, seal = isolated_metadata
    assert runtime.identity()['baseline']['effective_files'] == 67
    if change == 'seal':
        (runtime.HYBRID/'metadata.json').write_text(json.dumps(metadata)+'\n')
    elif change == 'profile':
        metadata['profile'] = {'unapproved': True}
        seal()
    elif change == 'dependency':
        dependency.write_text('# changed dependency\n')
    else:
        weight.write_bytes(b'changed size')
    with pytest.raises(RuntimeError, match=expected_message):
        runtime.identity()
    with pytest.raises(RuntimeError, match=expected_message):
        runtime.audit_frozen_sources()
