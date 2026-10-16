from pac_act.storage import append_json, write_json


def test_monitor_separates_training_results_from_validation_and_keeps_curves(tmp_path):
    from pac_act.dashboard import snapshot
    rows = [dict(role='train', target_count=2, final_collected=1, success=False, environment_return=30.),
            dict(role='validation', target_count=1, final_collected=1, success=True, environment_return=60.),
            dict(role='train', target_count=1, final_collected=1, success=True, environment_return=60.)]
    for row in rows:
        append_json(tmp_path/'episodes.jsonl', row)
    write_json(tmp_path/'progress.json', dict(sampled_actions=3))
    append_json(tmp_path/'action_journal.jsonl', dict(sampled_actions=8))
    append_json(tmp_path/'block_journal.jsonl', dict(sampled_blocks=2))
    result = snapshot(tmp_path)
    assert result['progress']['sampled_actions'] == 8
    assert result['progress']['sampled_blocks'] == 2
    assert result['training_metrics'][-1]['strict_success_percent'] == 50.
    assert result['training_metrics'][-1]['mean_absolute_count_error'] == .5
    assert result['training_metrics'][-1]['episode_return'] == 60.


def test_last_journal_row_ignores_incomplete_append_but_keeps_previous_record(tmp_path):
    from pac_act.storage import last_row
    path = tmp_path/'journal'
    path.write_bytes(b'{"n":1}\n{"n":2}\n{"n":')
    assert last_row(path) == {'n':2}
    assert last_row(tmp_path/'absent') is None
