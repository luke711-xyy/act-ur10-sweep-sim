from pac_act.storage import append_json, write_json
import json
from pathlib import Path

import pytest


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


def test_preflight_uses_only_actor_learning_prefixes_and_accepted_rounds(tmp_path):
    from pac_act.dashboard import snapshot
    write_json(tmp_path/'status.json', dict(state='preflight', training_started=False))
    write_json(tmp_path/'preflight_design.json', dict(budget=dict(started_at=10, deadline=7210), comparisons=[]))
    for seed, arm, revision in [(8000120, 'new_zero', 0), (8000126, 'actor', 0), (8000132, 'actor', 1)]:
        folder=tmp_path/'episodes'/f'preflight_{seed}_{arm}'
        folder.mkdir(parents=True)
        append_json(folder/'learning_blocks.jsonl', dict(length=5, policy_revision=revision))
    append_json(tmp_path/'updates.jsonl', dict(update=1, accepted=True, acceptance_round=1, exact_kl=.021))
    result=snapshot(tmp_path)
    assert result['run']['phase']=='preflight'
    assert result['run']['training_started'] is False
    assert result['progress']['sampled_actions']==10
    assert result['progress']['sampled_blocks']==2
    assert result['progress']['used_actions']==5
    assert result['progress']['pending_actions']==5
    assert result['progress']['updates']==1
    assert result['preflight']['completed_rounds']==1
    assert result['training_metrics']==[]


def test_phase_curves_never_mix_arms_or_administrative_cuts(tmp_path):
    from pac_act.dashboard import snapshot
    for arm, success, count, outcome in [('old_hybrid', True, 1, 'success'),
            ('new_zero', False, 3, 'failure'), ('new_zero', None, 0, 'interrupted')]:
        append_json(tmp_path/'episodes.jsonl', dict(role='preflight', arm=arm, seed=8000120,
            target_count=1, success=success, final_collected=count, outcome=outcome, environment_return=60))
    result=snapshot(tmp_path)
    zero=result['effect_groups']['preflight:new_zero']
    assert zero['summary']['strict_success_rate']==0
    assert zero['summary']['mean_absolute_count_error']==2
    assert zero['summary']['task_episodes']==1
    assert zero['summary']['interrupted']==1
    assert len(zero['metrics'])==1
    assert result['effect_groups']['preflight:old_hybrid']['summary']['strict_success_rate']==1


def test_filtered_pagination_covers_every_episode_without_replacing_unknowns(tmp_path):
    from pac_act.dashboard import snapshot
    for i in range(63):
        append_json(tmp_path/'episodes.jsonl', dict(role='train' if i%2 else 'preflight', arm='actor',
            seed=8000000+i, target_count=2, final_collected=None, success=False, outcome='failure'))
    first=snapshot(tmp_path, page=1, page_size=10, role='train')
    last=snapshot(tmp_path, page=999, page_size=10, role='train')
    assert first['total_episodes']==63 and first['filtered_episodes']==31
    assert first['pages']==4 and last['page']==4
    assert first['episodes'][0]['seed']==8000001
    assert last['episodes'][0]['seed']==8000061
    assert last['episodes'][0]['final_collected'] is None
    assert first['overview']['mean_absolute_count_error'] is None


def test_media_links_and_checkpoint_status_do_not_load_weights(tmp_path):
    from pac_act.dashboard import snapshot
    row=dict(role='preflight', arm='new_zero', seed=8000120, target_count=1, final_collected=2, success=False)
    append_json(tmp_path/'episodes.jsonl', row)
    folder=tmp_path/'episodes/preflight_8000120_new_zero'; folder.mkdir(parents=True)
    (folder/'replay.mp4').write_bytes(b'video')
    (folder/'force_Z.png').write_bytes(b'image')
    write_json(folder/'media.json', dict(file='replay.mp4', bytes=5, sha256='a'*64))
    saved=tmp_path/'checkpoints'; saved.mkdir()
    (saved/'acceptance.pt').write_bytes(b'not a torch checkpoint')
    write_json(saved/'index.json', dict(slots=[], milestones=[dict(file='acceptance.pt', bytes=22,
        sha256='b'*64, actions=160, kind='acceptance')]))
    result=snapshot(tmp_path)
    assert result['episodes'][0]['video_url']=='/episodes/preflight_8000120_new_zero/replay.mp4'
    assert result['episodes'][0]['force_url'].endswith('/force_Z.png')
    assert result['checkpoint_records'][0]['verification']=='metadata_and_file_size_only'
    (saved/'acceptance.pt').write_bytes(b'bad')
    assert snapshot(tmp_path)['checkpoint_records'][0]['verification']=='incomplete'


def test_read_only_snapshot_handles_partial_append_and_missing_metrics(tmp_path):
    from pac_act.dashboard import snapshot
    (tmp_path/'episodes.jsonl').write_bytes(b'{"role":"train","target_count":1,"success":false}\n{"role":')
    append_json(tmp_path/'updates.jsonl', dict(update=1, accepted=True, actor_loss=0.))
    before={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=snapshot(tmp_path)
    assert result['updates'][0]['actor_loss']==0.
    assert result['overview']['mean_absolute_count_error'] is None
    assert result['run']['training_started'] is None
    assert {p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
    json.dumps(result, allow_nan=False)


def test_rejected_updates_not_plotted_as_accepted_and_nonfinite_is_missing(tmp_path):
    from pac_act.dashboard import snapshot
    append_json(tmp_path/'updates.jsonl', dict(update=1, accepted=True, actor_loss=0.))
    append_json(tmp_path/'updates.jsonl', dict(update=2, accepted=False, actor_loss=99.))
    (tmp_path/'updates.jsonl').write_text((tmp_path/'updates.jsonl').read_text()+'{"update":3,"accepted":true,"value_loss":NaN}\n')
    result=snapshot(tmp_path)
    assert len(result['updates'])==2
    assert result['updates'][-1]['value_loss'] is None
    assert result['warnings']
    json.dumps(result, allow_nan=False)


def test_dashboard_server_streams_ranges_and_refuses_external_paths(tmp_path):
    import threading
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from pac_act.dashboard import create_server
    folder=tmp_path/'episodes/one'; folder.mkdir(parents=True)
    (folder/'replay.mp4').write_bytes(b'0123456789')
    (tmp_path/'secret.pt').write_bytes(b'private')
    server=create_server(tmp_path, port=0)
    thread=threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(Request(base+'/episodes/one/replay.mp4', headers={'Range':'bytes=2-5'})) as response:
            assert response.status==206 and response.read()==b'2345'
            assert response.headers['Content-Range']=='bytes 2-5/10'
        with pytest.raises(HTTPError) as error: urlopen(base+'/secret.pt')
        assert error.value.code==404
        with pytest.raises(HTTPError) as error: urlopen(base+'/%2e%2e/secret.mp4')
        assert error.value.code==404
        with pytest.raises(HTTPError) as error: urlopen(Request(base+'/api', data=b'{}', method='POST'))
        assert error.value.code==501
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_completed_acceptance_uses_reported_duration_not_an_ever_running_clock(tmp_path):
    from pac_act.dashboard import snapshot
    write_json(tmp_path/'status.json', dict(state='preflight_complete', training_started=False))
    write_json(tmp_path/'preflight_design.json', dict(budget=dict(started_at=10, deadline=7210)))
    write_json(tmp_path/'acceptance.json', dict(elapsed_s=61, engineering_passed=True, checks={}))
    assert snapshot(tmp_path)['run']['elapsed_seconds']==61


def test_corrupt_complete_lines_preserve_valid_records_but_not_fake_complete_totals(tmp_path):
    from pac_act.dashboard import snapshot
    (tmp_path/'episodes.jsonl').write_text('{"role":"train","success":false,"target_count":1}\nBAD\n')
    result=snapshot(tmp_path)
    assert len(result['episodes'])==1
    assert result['parsed_episodes']==1 and result['total_episodes'] is None
    assert result['data_complete'] is False and result['warnings']


def test_unreadable_preflight_block_counts_are_unknown_not_zero(tmp_path):
    from pac_act.dashboard import snapshot
    write_json(tmp_path/'preflight_design.json', dict(budget={}))
    folder=tmp_path/'episodes/preflight_8000126_actor'; folder.mkdir(parents=True)
    (folder/'learning_blocks.jsonl').write_text('BAD\n')
    result=snapshot(tmp_path)
    assert result['progress']['sampled_actions'] is None
    assert result['progress']['used_actions'] is None
    assert result['progress']['pending_actions'] is None


@pytest.mark.parametrize('bad',[None, [None], 'not-a-list'])
def test_malformed_checkpoint_index_does_not_hide_other_data(tmp_path,bad):
    from pac_act.dashboard import snapshot
    (tmp_path/'checkpoints').mkdir()
    write_json(tmp_path/'checkpoints/index.json', dict(milestones=bad, slots=[]))
    result=snapshot(tmp_path)
    assert result['checkpoint_records']==[]
    assert result['warnings']
