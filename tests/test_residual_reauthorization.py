"""Fresh authorization excludes every consumed preflight and native layout."""
import hashlib
import json

import numpy as np
import pytest


def test_new_invocation_cannot_reuse_earlier_preflight_or_restore_layouts():
    from residual_rl.pilot import seed_schedule
    prior = {3150000, 3150001, 3150100, 3150101, 3151000, 3151001, 3151100,
             3151101, 3152000, 3152001, 3152100, 3152101, 3152200}
    assert not {r['seed'] for r in seed_schedule()['preflight']} & prior


def test_prepare_seals_latest_pilot_and_native_consumed_layouts(tmp_path):
    from residual_rl.pilot import prepare
    store = prepare(tmp_path / 'prepared_only')
    schedule = json.loads((store.root / 'seed_schedule.json').read_text())
    excluded = schedule['prior_consumed_layouts']
    assert {3150000, 3151000, 3152000, 3152001, 3152100, 3152200} <= {r['seed'] for r in excluded}
    for entry in excluded:
        assert len(entry['layout_fingerprint']) == 64 and entry['source_sha256']
    assert json.loads((store.root / 'status.json').read_text())['training_started'] is False
    assert not (store.root / 'started.json').exists()


def test_native_failure_exclusion_uses_measured_float32_pose_hash(tmp_path):
    from residual_rl import pilot
    read = getattr(pilot, 'native_check_consumed_layout', None)
    assert callable(read), 'native restoration layout must also be excluded'
    poses = np.zeros((6, 7), dtype='<f4')
    digest = hashlib.sha256(poses.tobytes()).hexdigest()
    marker = tmp_path / 'attempt.json'
    marker.write_text(json.dumps(dict(seed=3152200, target_count=1, opportunities=1)))
    layout = tmp_path / 'initial_layout.json'
    layout.write_text(json.dumps(dict(poses=poses.tolist(), sha256=digest)))
    rows = read(tmp_path)
    assert rows[0]['seed'] == 3152200 and rows[0]['layout_fingerprint'] == digest
    assert rows[0]['source_sha256'][str(marker)] == hashlib.sha256(marker.read_bytes()).hexdigest()
    poses[0, 0] = .1
    layout.write_text(json.dumps(dict(poses=poses.tolist(), sha256=digest)))
    with pytest.raises(RuntimeError, match='fingerprint'):
        read(tmp_path)
