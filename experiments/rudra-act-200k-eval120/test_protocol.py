import importlib.util
import json
from pathlib import Path

import pytest


def adapter():
    path = Path(__file__).with_name('evaluate_batch.py')
    assert path.is_file(), 'independent Rudra evaluation adapter is missing'
    spec = importlib.util.spec_from_file_location('rudra_eval200k', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_preserves_original_seed_goal_pairs():
    m = adapter()
    rows = [{'seed': 2120000+i, 'target_count': i % 6+1} for i in range(120)]
    assert m.seed_target_pairs(rows) == [(r['seed'], r['target_count']) for r in rows]
    rows[40]['seed'] = rows[39]['seed']
    with pytest.raises(ValueError):
        m.seed_target_pairs(rows)


def test_final_count_metrics_distinguish_strict_success_and_unknown():
    m = adapter()
    rows = [
        {'key': 'a', 'target_count': 2, 'final_collected': 2, 'peak_collected': 3,
         'success': False, 'reason': 'partial overlap', 'termination_reason': '2-second collection window elapsed'},
        {'key': 'b', 'target_count': 6, 'final_collected': 4, 'peak_collected': 4,
         'success': False, 'reason': '2000-frame timeout before task completion'},
        {'key': 'c', 'target_count': 1, 'final_collected': None, 'peak_collected': None,
         'success': False, 'reason': 'IK failed'},
        {'key': 'd', 'target_count': 3, 'final_collected': 3, 'peak_collected': 3,
         'success': True, 'reason': ''},
    ]
    s = m.summarize(rows, 120)
    assert s['strict_successes'] == 1
    assert s['exact_final_matches'] == 2
    assert s['unknown_counts'] == 1
    assert s['strict_success_rate'] == .25
    assert s['count_match_rate'] == .5
    assert s['mean_absolute_count_error'] == pytest.approx(2/3)
    assert s['mean_signed_count_error'] == pytest.approx(-2/3)
    assert sum(map(sum, s['final_count_matrix'])) == 3
    assert s['final_count_matrix'][0][0] == 0
    assert s['final_count_matrix'][2][1] == 1
    assert s['failure_types'] == {'collection_window': 1, 'timeout': 1, 'ik_execution': 1}


def test_chart_and_ledger_outputs_are_labeled_for_two_view_200k(tmp_path):
    m = adapter()
    row = {'key':'x', 'seed':2120000, 'target_count':1, 'success':True,
           'final_collected':1, 'peak_collected':1, 'model_label':m.LABEL,
           'reason':'', 'video_path':''}
    m.write_artifacts(tmp_path, [row], 120)
    s = json.loads((tmp_path/'summary.json').read_text())
    assert s['label'] == 'Rudra two-view DINOv3 ACT | step 200K | current native schedule'
    assert s['completed'] == 1 and s['planned'] == 120
    assert s['by_target']['1']['tested'] == 1
    assert 'step 200K' in (tmp_path/'success_density.svg').read_text()
    assert 'Five-view' not in (tmp_path/'success_density.svg').read_text()
    assert (tmp_path/'episodes.csv').is_file()
    assert (tmp_path/'density_matrix.csv').is_file()
    assert (tmp_path/'success_density.png').is_file()


def test_execution_uses_preserved_checkpoint_and_current_native_entry(tmp_path):
    m = adapter()
    command = m.episode_command(tmp_path, {'key':'test', 'seed':2120000, 'target_count':1,
                                          'model_path':'/preserved/step_200000'})
    assert command[2].endswith('episode_entrypoint.py')
    assert command[command.index('--model')+1] == '/preserved/step_200000'
    assert '--spec' in command
    assert '--record-replay' not in command and '--defer-video' not in command
    assert not any('pac_act' in value or 'hybrid' in value for value in command)


def test_saved_measurement_can_be_loaded_without_rendering_images(tmp_path):
    import numpy as np
    from sim.config import Config
    from sim.act.rollout import ActRolloutResult
    from sim.act.replay import load_replay_input
    m = adapter()
    cfg = Config({'task': {'target_count': 1}})
    result = ActRolloutResult(success=False, failure_reason='timeout', target_count=1,
                             total=6, final_collected=2, peak_collected=2)
    result.initial_object_poses = np.zeros((6, 7))
    result.trace = [{'replay_state': {'spec': 1, 'values': np.arange(8.)}, 't': .4,
                     'object_pose': np.zeros((6, 7)), 'command': np.zeros(4)}]
    record = m.save_measurement(tmp_path, cfg, 2120000, '/preserved/step_200000', result)
    loaded = load_replay_input(record['replay_input_path'], cfg, seed=2120000,
                               model='/preserved/step_200000')
    assert loaded.final_collected == 2
    assert len(loaded.trace) == 1
    assert loaded.trace[0]['replay_state']['values'] == list(np.arange(8.))
    assert record['video_path'] is None and record['media_policy'] == 'on_demand'
    assert not list(tmp_path.rglob('*.mp4'))
    assert not list(tmp_path.rglob('*.webp'))
    assert not list(tmp_path.rglob('*.png'))


def test_failed_physical_attempt_is_not_retried_after_resume(tmp_path, monkeypatch):
    m = adapter()
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir()
    (checkpoint/'weights').write_text('frozen')
    monkeypatch.setattr(m, 'CHECKPOINT', checkpoint)
    for name in ('evaluation_config.yaml','evaluation_scene.yaml'):
        (tmp_path/name).write_text('{}')
    plan=[{'key':str(i),'seed':2120000+i,'target_count':i%6+1,'model_path':str(checkpoint),
           'layout_fingerprint':str(i)} for i in range(120)]
    (tmp_path/'episode_plan.json').write_text(json.dumps(plan))
    m.atomic_json(tmp_path/'metadata.json',{
        'episode_plan_sha256':m.sha256(tmp_path/'episode_plan.json'),
        'source_hashes':m.source_hashes(), 'config_sha256':m.sha256(tmp_path/'evaluation_config.yaml'),
        'scene_sha256':m.sha256(tmp_path/'evaluation_scene.yaml'),
        'checkpoint_sha256':{'weights':m.sha256(checkpoint/'weights')},
    })
    count=tmp_path/'physical_calls'
    code=f"from pathlib import Path; p=Path({str(count)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x'); raise SystemExit(2)"
    monkeypatch.setattr(m,'episode_command',lambda root,spec:[m.sys.executable,'-c',code])
    with pytest.raises(RuntimeError,match='no retry'):
        m.run()
    with pytest.raises(RuntimeError,match='no retry'):
        m.run()
    assert count.read_text() == 'x'
    assert len((tmp_path/'attempts.jsonl').read_text().splitlines()) == 1
    assert not (tmp_path/'episodes.jsonl').exists()
