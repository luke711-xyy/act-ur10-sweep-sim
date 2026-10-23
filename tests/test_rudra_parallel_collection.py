import fcntl
import json
import os

import pytest


@pytest.fixture(autouse=True)
def available_test_storage(monkeypatch):
    from types import SimpleNamespace
    from scripts import collect_rudra_parallel as entry
    monkeypatch.setattr(entry.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))


def fake_success(cfg,slot,out,source_sha,camera_sha,connection):
    from pathlib import Path
    out=Path(out)
    row=dict(episode_id=slot['episode_id'],target_count=slot['target_count'],seed=slot['seed'],
             schema_version=5,success=True,frame_count=1,camera_views=['top','wrist'],
             source_manifest_sha256=source_sha,scene_profile_sha256=camera_sha)
    if 'source_seed' in slot:
        row.update(source_seed=slot['source_seed'],requested_seed=slot['source_seed'],
                   generation_attempt=slot['generation_attempt'])
    with (out/'.manifest.jsonl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (out/'manifest.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    connection.send(dict(result=dict(episode_id=slot['episode_id'],target_count=slot['target_count'],
                                    seed=slot['seed'],state='saved'),elapsed_seconds=.1,peak_rss_kib=100))
    connection.close()


def fake_crash(*args):
    os._exit(7)


def fake_fail_original(cfg,slot,out,source_sha,camera_sha,connection):
    if slot['seed']==slot['source_seed']:
        connection.send(dict(result=dict(episode_id=slot['episode_id'],target_count=slot['target_count'],
                                         seed=slot['seed'],state='failed',collected=None,frames=None),
                             elapsed_seconds=.1,peak_rss_kib=100))
        connection.close()
    else:fake_success(cfg,slot,out,source_sha,camera_sha,connection)


def fake_two_commits_before_result(cfg,slot,out,source_sha,camera_sha,connection):
    import time
    from pathlib import Path
    class Held:
        def send(self,message):self.message=message
        def close(self):pass
    held=Held();fake_success(cfg,slot,out,source_sha,camera_sha,held)
    deadline=time.monotonic()+5
    while len((Path(out)/'manifest.jsonl').read_text().splitlines())<2:
        if time.monotonic()>deadline:raise RuntimeError('test barrier timed out')
        time.sleep(.01)
    connection.send(held.message);connection.close()


def inputs(tmp_path):
    from sim.config import Config
    slots=[dict(episode_id=f'preview_goal_1_{i:04d}',target_count=1,seed=4200000+i) for i in range(4)]
    cfg=Config(dict(act=dict(image_size=[640,480]),episode=dict(max_frames=2)))
    pinned=dict(files={},source_manifest_sha256='source',scene_profile_sha256='camera')
    return cfg,dict(entries=slots),pinned


def test_parallel_slots_are_unique_and_committed_resume_is_skipped(monkeypatch,tmp_path):
    from scripts import collect_rudra_parallel as entry
    cfg,plan,pinned=inputs(tmp_path)
    a,b=__import__('multiprocessing').Pipe()
    fake_success(cfg.to_dict(),plan['entries'][0],tmp_path,'source','camera',b);a.recv();a.close()
    monkeypatch.setattr(entry,'validate_training_manifest',lambda *a,**k:None)
    assert entry.collect_parallel(cfg,plan,pinned,tmp_path,2,worker_target=fake_success)==0
    rows=[json.loads(s) for s in (tmp_path/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==4 and len({r['episode_id'] for r in rows})==4
    events=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text().splitlines()]
    started=[r for r in events if r['event']=='episode_started']
    assert {r['seed'] for r in started}=={r['seed'] for r in plan['entries'][1:]}
    assert json.loads((tmp_path/'progress.json').read_text())['saved']==4


def test_abrupt_worker_exit_stops_without_retry_or_invented_count(tmp_path):
    from scripts import collect_rudra_parallel as entry
    cfg,plan,pinned=inputs(tmp_path)
    with pytest.raises(RuntimeError,match='worker'):
        entry.collect_parallel(cfg,plan,pinned,tmp_path,1,worker_target=fake_crash)
    events=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert len([r for r in events if r['event']=='episode_started'])==1
    assert not (tmp_path/'manifest.jsonl').exists()


def test_parallel_worker_count_is_bounded(tmp_path):
    from scripts.collect_rudra_parallel import worker_count
    assert worker_count(tmp_path,2)==2
    (tmp_path/'worker-count.json').write_text('{"workers": 3}')
    assert worker_count(tmp_path,2)==3
    (tmp_path/'worker-count.json').write_text('{"workers": 100}')
    with pytest.raises(ValueError,match='workers'):worker_count(tmp_path,2)


def test_scratch_cleanup_removes_only_dead_worker_owned_directories(tmp_path):
    from scripts.collect_rudra_parallel import cleanup_spools
    parent=tmp_path/'.image-spool';parent.mkdir()
    dead=parent/'preview_goal_6_0001-pid999999999-test';dead.mkdir()
    (dead/'frames.rgb').write_bytes(b'cache')
    cleanup_spools(tmp_path,pid=999999999)
    assert not dead.exists()
    live=parent/f'preview_goal_6_0002-pid{os.getpid()}-test';live.mkdir()
    (live/'frames.rgb').write_bytes(b'keep')
    with pytest.raises(RuntimeError,match='still alive'):cleanup_spools(tmp_path)
    assert (live/'frames.rgb').read_bytes()==b'keep'


def test_fill_mode_replaces_failed_seeds_and_only_finishes_after_all_slots_succeed(monkeypatch,tmp_path):
    from scripts import collect_rudra_parallel as entry
    cfg,plan,pinned=inputs(tmp_path)
    first=plan['entries'][0]
    (tmp_path/'events.jsonl').write_text(json.dumps(dict(first,event='episode_failed',state='failed',collected=0))+'\n')
    validations=[]
    monkeypatch.setattr(entry,'validate_training_manifest',lambda *a,**k:validations.append(True))
    assert entry.collect_parallel(cfg,plan,pinned,tmp_path,2,worker_target=fake_fail_original,fill_missing=True)==0
    rows=[json.loads(s) for s in (tmp_path/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==4 and len({r['seed'] for r in rows})==4
    assert not ({r['seed'] for r in rows}&{r['seed'] for r in plan['entries']})
    assert all(r['requested_seed']!=r['seed'] and r['generation_attempt']==2 for r in rows)
    status=json.loads((tmp_path/'progress.json').read_text())
    assert status['state']=='complete' and status['saved']==4 and status['failed']==0
    assert status['failed_attempts']==4 and validations==[True]
    before=(tmp_path/'events.jsonl').read_text()
    assert entry.collect_parallel(cfg,plan,pinned,tmp_path,2,worker_target=fake_crash,fill_missing=True)==0
    # A completed resume validates but never starts another physical attempt.
    later=(tmp_path/'events.jsonl').read_text()[len(before):]
    assert 'episode_started' not in later


def test_low_space_finishes_active_worker_before_dispatching_more(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from scripts import collect_rudra_parallel as entry
    cfg,plan,pinned=inputs(tmp_path);calls=[]
    def space(path):
        calls.append(True)
        free=100 if len(calls)==1 or (tmp_path/'manifest.jsonl').exists() else 11
        return SimpleNamespace(free=free*1024**3)
    monkeypatch.setattr(entry.shutil,'disk_usage',space)
    monkeypatch.setattr(entry,'validate_training_manifest',lambda *a,**k:None)
    assert entry.collect_parallel(cfg,plan,pinned,tmp_path,2,worker_target=fake_success)==0
    assert json.loads((tmp_path/'progress.json').read_text())['saved']==4


def test_progress_does_not_double_count_committed_replacements_awaiting_results(monkeypatch,tmp_path):
    from scripts import collect_rudra_parallel as entry
    cfg,plan,pinned=inputs(tmp_path);plan['entries']=plan['entries'][:2]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(dict(s,event='episode_failed'))+'\n' for s in plan['entries']))
    seen=[];original=entry.write_json
    def observed(path,value):
        if path.name=='progress.json':seen.append(dict(value))
        original(path,value)
    monkeypatch.setattr(entry,'write_json',observed)
    monkeypatch.setattr(entry,'validate_training_manifest',lambda *a,**k:None)
    assert entry.collect_parallel(cfg,plan,pinned,tmp_path,2,worker_target=fake_two_commits_before_result,fill_missing=True)==0
    assert all(r['processed']<=2 for r in seen)
