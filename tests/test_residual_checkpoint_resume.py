"""Persistence/recovery tests: no ACT, renderer or physical training is run."""
import copy
import importlib
import io
import time

import numpy as np
import pytest
import torch


def api(module, name):
    try:
        value = getattr(importlib.import_module('residual_rl.' + module), name, None)
    except ModuleNotFoundError:
        value = None
    assert callable(value), f'{module}.{name} is not implemented'
    return value


def test_replay_arrays_are_streamed_as_tensor_records_and_roundtrip_exactly():
    from residual_rl.learning import Episode, EpisodicReplay
    replay = EpisodicReplay(seed=19)
    episode = Episode('train_0000000', 1, np.arange(15, dtype=np.float32).reshape(3, 5),
                      np.zeros((3, 7), np.float32), np.zeros((2, 3), np.float32),
                      np.array([60., 0.], np.float32), np.array([False, True]),
                      np.array([True, True]))
    replay.add(episode)
    packed = api('storage', 'pack_replay')(replay.state_dict())
    assert packed['encoding'] == 'tensor_records_v1'
    assert isinstance(packed['state']['episodes'][0]['actor_obs'], torch.Tensor)
    stream = io.BytesIO()
    torch.save(packed, stream)
    stream.seek(0)
    restored = api('storage', 'unpack_replay')(torch.load(stream, weights_only=False))
    target = EpisodicReplay()
    target.load_state_dict(restored)
    assert np.array_equal(target.episodes[0].actor_obs, episode.actor_obs)
    assert np.array_equal(target.episodes[0].rewards, [60., 0.])
    assert target.state_dict()['rng_state'] == replay.state_dict()['rng_state']
    # Loading the original NumPy checkpoint encoding remains supported.
    assert api('storage', 'unpack_replay')(replay.state_dict())['schema_version'] == 2


def test_checkpoint_reports_first_deadline_failure_not_zip_finalizer_error(monkeypatch):
    from residual_rl.storage import BoundedCheckpointWriter
    import residual_rl.storage as storage
    clock = [10.]
    monkeypatch.setattr(storage.time, 'monotonic', lambda: clock[0])

    class SlowPayload:
        def __reduce__(self):
            clock[0] = 20.
            return int, (7,)

    sink = BoundedCheckpointWriter(io.BytesIO(), max_bytes=2**20, deadline=15.,
                                   check=lambda: (2**20, 15.))
    save = api('storage', 'save_checkpoint_payload')
    with pytest.raises(RuntimeError, match='checkpoint deadline exhausted'):
        save({'value': SlowPayload()}, sink)


def test_reconstructed_sampling_and_schedule_match_real_sampler_with_eviction():
    from residual_rl.learning import Episode, EpisodicReplay
    from residual_rl.pilot import UpdateSchedule
    replay = EpisodicReplay(capacity=8000, seed=20261015)
    schedule = UpdateSchedule()
    records = []
    for i in range(40):
        mask = np.arange(600) >= 100
        episode = Episode(f'train_{i:07d}', i % 6 + 1,
                          np.zeros((601, 4), np.float32), np.zeros((601, 6), np.float32),
                          np.zeros((600, 3), np.float32), np.zeros(600, np.float32),
                          np.arange(600) == 599, mask)
        replay.add(episode)
        schedule.begin_episode(500, ready=replay.training_ready())
        while schedule.due():
            replay.sample(batch_size=4, length=32)
            schedule.completed()
        records.append(dict(episode_id=episode.episode_id, target_count=episode.target_count,
                            mask=mask, terminated=True))
    result = api('resume', 'reconstruct_sampling')(records, capacity=8000, seed=20261015)
    assert result['rng_state'] == replay.state_dict()['rng_state']
    assert result['goal_cursor'] == replay.state_dict()['goal_cursor']
    assert result['schedule'] == vars(schedule)
    assert result['retained_ids'] == [e.episode_id for e in replay.episodes]
    assert result['seen_ids'] == sorted(replay.state_dict()['seen_ids'])


def test_trace_append_keeps_old_gzip_members_and_adds_new_records(tmp_path):
    from residual_rl.storage import TraceWriter
    import gzip
    path = tmp_path / 'metrics.gz'
    writer = TraceWriter(path)
    writer.append({'updates': 1225})
    writer.close()
    before = path.read_bytes()
    writer = TraceWriter(path, append=True)
    writer.append({'updates': 1226})
    writer.close()
    assert path.read_bytes().startswith(before)
    with gzip.open(path, 'rt') as stream:
        assert len(stream.readlines()) == 2


def test_iteration_can_start_at_next_unattempted_layout_without_reset():
    from residual_rl.long_training import iter_training_specs, training_schedule
    result = next(iter_training_specs(training_schedule(), start_index=142))
    assert result == dict(episode_id='train_0000142', seed=5000142, target_count=5, phase='train')
    with pytest.raises(ValueError):
        next(iter_training_specs(training_schedule(), start_index=-1))


def test_reference_snapshot_uses_durable_blocks_and_detects_corruption(tmp_path):
    import json
    from residual_rl.storage import RunStore
    from residual_rl.learning import Episode, EpisodicReplay
    from types import SimpleNamespace
    store=RunStore(tmp_path/'run',create=True)
    spec=dict(episode_id='train_0000000',seed=5000000,target_count=1,phase='train')
    store.claim(spec)
    episode=Episode(spec['episode_id'],1,np.ones((3,4),np.float32),np.zeros((3,6),np.float32),
                    np.zeros((2,3),np.float32),np.array([60.,0.],np.float32),
                    np.array([False,True]),np.array([True,True]))
    guard=SimpleNamespace(checkpoint_allowance=lambda **kw:(2**20,time.monotonic()+55))
    store.commit_episode(dict(**spec,success=False,final_collected=0,
                              opportunities=1,physical_attempts=1),episode,guard)
    replay=EpisodicReplay(seed=33);replay.add(episode)
    snapshot=api('storage','reference_replay')(store.root,replay)
    assert snapshot['encoding']=='durable_records_v1'
    assert 'episodes' not in snapshot['state']
    restored=api('storage','unpack_replay')(snapshot,root=store.root)
    assert np.array_equal(restored['episodes'][0]['rewards'],[60.,0.])
    assert restored['rng_state']==replay.state_dict()['rng_state']
    path=store.root/'episodes/train_0000000/experience.npz'
    with path.open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='evidence'):
        api('storage','unpack_replay')(snapshot,root=store.root)


def test_reference_checkpoint_reuse_rejects_missing_numeric_dependencies(tmp_path):
    from residual_rl.storage import checkpoint_evidence,sha256,write_json
    root=tmp_path/'run';folder=root/'checkpoints';folder.mkdir(parents=True)
    path=folder/'state.pt';path.write_bytes(b'checkpoint')
    write_json(path.with_suffix('.json'),dict(bytes=10,sha256=sha256(path),kind='reference_full',
        replay_root=str(root.resolve()),replay_dependencies=[dict(path='episodes/lost.npz',
            bytes=10,sha256='0'*64)]))
    with pytest.raises((OSError,ValueError)):
        checkpoint_evidence(path)


def test_resume_supervisor_rejects_previous_attempt_shutdown_ack(tmp_path):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from residual_rl.storage import write_json
    spec=importlib.util.spec_from_file_location('_resume_supervisor_test',
          Path(__file__).parents[1]/'scripts/train_residual_sac.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    write_json(tmp_path/'save_result.json',dict(status='not_applicable',attempt_token='old'))
    write_json(tmp_path/'checkpoint_state.json',dict(initialized=False))
    process=SimpleNamespace(poll=lambda:1,returncode=1)
    result=module._supervise(process,tmp_path,time.monotonic(),attempt_token='new')
    assert result['worker_acknowledged'] is False
    assert result['attempt_token']=='new'


def test_resumed_lifecycle_appends_and_starts_after_old_opportunities(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    code = r'''
import json,sys,time,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from residual_rl.config import activate_long_training_profile
activate_long_training_profile()
from residual_rl.config import DEFAULT
from residual_rl import runtime,safety,resume
from residual_rl.storage import RunStore,write_json,sha256,TraceWriter
from residual_rl.pilot import own_code_hashes,UpdateSchedule
from residual_rl.learning import Episode,RecurrentSAC,EpisodicReplay
from residual_rl.long_training import execute,training_schedule
root=Path(sys.argv[1]);store=RunStore(root,create=True)
write_json(root/'configuration.json',DEFAULT.to_dict())
write_json(root/'seed_schedule.json',training_schedule())
write_json(root/'frozen_identity.json',runtime.identity())
write_json(root/'code_hashes.json',own_code_hashes())
write_json(root/'preflight_reference.json',dict(actor_dim=572,critic_dim=791))
write_json(root/'seal.json',{name:sha256(root/name) for name in
 ['configuration.json','seed_schedule.json','frozen_identity.json','code_hashes.json','preflight_reference.json']})
safety.resources=lambda *args,**kwargs:dict(disk_free=30*2**30,output_bytes=0,rss=0,dangerous_memory=False)
def episode(i):
 return Episode(f'train_{i:07d}',i%6+1,np.zeros((2,572),np.float32),np.zeros((2,791),np.float32),
  np.zeros((1,3),np.float32),np.zeros(1,np.float32),np.array([True]),np.array([True]))
def commit(i,g):
 ep=episode(i);spec=dict(episode_id=ep.episode_id,seed=5000000+i,target_count=i%6+1,phase='train')
 store.claim(spec)
 row=dict(**spec,success=False,outcome='failure',reason='synthetic lifecycle',final_collected=0,
  opportunities=1,planning_calls=1,physical_attempts=1,residual_frames=1,
  terminal_reward=0.,credited_contact_return=0.,rl_return=0.)
 return SimpleNamespace(row=store.commit_episode(row,ep,g),episode=ep)
g=SimpleNamespace(checkpoint_allowance=lambda **kw:(32*2**20,time.monotonic()+55))
replay=EpisodicReplay()
for i in range(2): replay.add(commit(i,g).episode)
before=(root/'episodes.jsonl').read_bytes()
writer=TraceWriter(root/'learning_metrics.jsonl.gz');writer.close()
learner=RecurrentSAC(572,791);schedule=UpdateSchedule();schedule.steps=2
payload=dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),mps_rng=None)
resume.load_recovery=lambda *a,**kw:dict(learner=learner,replay=replay,schedule=schedule,
 payload=payload,next_index=2,training_steps=2,proof={})
calls=[]
class Engine:
 def __init__(self,store,guard):
  self.guard=guard
  self.registry=SimpleNamespace(register=lambda *a:None)
 def run(self,spec,mode,policy):
  if calls:raise safety.BudgetStop('test_stop')
  calls.append(spec['seed']);assert spec['seed']==5000002
  self.guard.residual_steps+=1
  return commit(2,self.guard)
runtime.HybridRuntime=Engine
execute(store,started_at=time.monotonic(),resume=True)
progress=json.loads((root/'training_progress.json').read_text())
assert calls==[5000002] and progress['training_steps']==3
assert (root/'episodes.jsonl').read_bytes().startswith(before)
assert len(store.rows())==3 and not (root/'checkpoints/initial.pt').exists()
assert progress['updates']==0 and progress['actor_updates']==0
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path/'run')],
                            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
