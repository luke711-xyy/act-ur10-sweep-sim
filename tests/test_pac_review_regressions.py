from types import SimpleNamespace
import json
import pytest
import numpy as np

from pac_act.storage import RunStore, append_json, read_rows, write_json


def test_native_measurement_collision_is_safety_terminal_not_interruption(monkeypatch,tmp_path):
    from test_pac_execution import wired_episode
    world=wired_episode(monkeypatch,tmp_path,learned=True)
    def collision(row,env):
        raise RuntimeError('unsafe robot collision')
    world.record=collision
    k,reward,done=world.step_block(np.zeros((5,4)),learned=True)
    assert k==1 and done and reward==-60.
    assert world.outcome=='safety' and world.terminal_reward==-60.
    assert world.final_count()==0


def test_append_after_interrupted_journal_preserves_tail_and_record_boundary(tmp_path):
    path=tmp_path/'action_journal.jsonl'
    fragment=b'{"sampled_actions":104'
    path.write_bytes(b'{"sampled_actions":103}\n'+fragment)
    assert read_rows(path)==[{'sampled_actions':103}]
    append_json(path,dict(sampled_actions=105))
    assert read_rows(path)==[{'sampled_actions':103},{'sampled_actions':105}]
    preserved=list(tmp_path.glob('*.incomplete.*'))
    assert len(preserved)==1 and preserved[0].read_bytes()==fragment


def test_orphan_counts_durable_actions_beyond_last_block_and_keeps_final_unknown(tmp_path):
    from pac_act.training import recover_orphans
    store=RunStore(tmp_path)
    spec=dict(role='train',seed=6000000,target_count=1,arm='actor')
    folder=store.claim(spec)
    append_json(folder/'blocks.jsonl',dict(executed=5,learned=True,final_collected=0))
    for i in range(7):
        append_json(tmp_path/'action_journal.jsonl',dict(seed=6000000,sampled_actions=i+1))
    recover_orphans(store)
    row=store.rows()[0]
    assert row['active_actions']==7
    assert row['final_collected'] is None
    assert row['last_known_collected']==0
    assert row['outcome']=='interrupted' and row['success'] is None
    recover_orphans(store)
    assert len(store.rows())==1


def test_fallback_lineage_does_not_suppress_or_display_stale_update(tmp_path):
    from pac_act.training import reconcile_update_audit
    from pac_act.dashboard import snapshot
    append_json(tmp_path/'updates.jsonl',dict(update=1,accepted=True,commit_id='one',parent_commit_id=None,value_loss=11.))
    append_json(tmp_path/'updates.jsonl',dict(update=2,accepted=True,commit_id='lost',parent_commit_id='one',value_loss=22.))
    reconcile_update_audit(tmp_path,dict(last_update_audit=dict(update=1,accepted=True,commit_id='one',parent_commit_id=None,value_loss=11.)))
    assert [r['value_loss'] for r in snapshot(tmp_path)['updates']]==[11.]
    new=dict(update=2,accepted=True,commit_id='new',parent_commit_id='one',value_loss=33.)
    reconcile_update_audit(tmp_path,dict(last_update_audit=new))
    assert len(read_rows(tmp_path/'updates.jsonl'))==3  # All historical evidence retained.
    assert [r['value_loss'] for r in snapshot(tmp_path)['updates']]==[11.,33.]


def test_cleanup_failure_still_visits_every_world_and_preserves_primary_error(tmp_path):
    from pac_act.lifecycle import Collection
    visited=[]
    class World:
        def __init__(self,n):self.n=n
        def interrupt(self,error):
            visited.append(self.n)
            if self.n==1:raise OSError('cannot write first result')
    collection=Collection(None,None,None,None,None,{})
    collection.worlds={1:World(1),6:World(6)}
    primary=RuntimeError('primary prediction failure')
    errors=collection.interrupt(primary)
    assert visited==[1,6] and not collection.worlds
    assert len(errors)==1 and 'cannot write' in errors[0]
    assert 'cannot write' in ' '.join(primary.__notes__)


def test_live_progress_separates_pending_from_previously_discarded(tmp_path):
    from pac_act.dashboard import snapshot
    write_json(tmp_path/'progress.json',dict(sampled_actions=100,used_actions=80,discarded_actions=20))
    append_json(tmp_path/'action_journal.jsonl',dict(sampled_actions=112))
    p=snapshot(tmp_path)['progress']
    assert p['pending_actions']==12
    assert p['sampled_actions']==p['used_actions']+p['discarded_actions']+p['pending_actions']


def test_physical_cleanup_releases_solver_even_if_trace_close_fails():
    from pac_act.hybrid import HybridEpisode
    world=HybridEpisode.__new__(HybridEpisode)
    visited=[]
    def trace_close():
        visited.append('trace')
        raise OSError('trace write error')
    world.closed=False;world.solver_entered=True
    world.trace=SimpleNamespace(close=trace_close)
    world.env=SimpleNamespace(close=lambda:visited.append('env'))
    world.solver=SimpleNamespace(__exit__=lambda *args:visited.append('solver'))
    with pytest.raises(BaseExceptionGroup):world.close()
    assert visited==['trace','env','solver'] and world.closed


def test_complete_invalid_journal_record_is_not_silently_discarded(tmp_path):
    path=tmp_path/'log.jsonl';path.write_bytes(b'{bad json}\n')
    with pytest.raises(json.JSONDecodeError):read_rows(path)


def test_real_checkpoint_commits_link_new_audits_by_unique_identity(tmp_path):
    from pac_act.training import commit_update
    from pac_act.storage import Checkpoints,active_updates
    cp=Checkpoints(tmp_path/'checkpoints','id')
    counts=dict(sampled_actions=10,used_actions=0,updates=0)
    one=dict(update=1,accepted=True,value_loss=1.)
    commit_update(cp,dict(counters=dict(counts,used_actions=10,updates=1)),counts,one,tmp_path)
    two=dict(update=2,accepted=True,value_loss=2.)
    commit_update(cp,dict(counters=dict(counts,sampled_actions=20,used_actions=20,updates=2),
                         last_update_audit=one),counts,two,tmp_path)
    assert two['parent_commit_id']==one['commit_id']
    assert len(active_updates(tmp_path))==2
    assert cp.latest()['state']['last_update_audit']['commit_id']==two['commit_id']


@pytest.mark.parametrize('context,wanted',[
    ({'collection_reason':'exact target count reached','safety_reason':''},('success','exact target count reached')),
    ({'collection_reason':'2-second collection window elapsed','safety_reason':''},('failure','2-second collection window elapsed')),
    ({'collection_reason':'exact target count reached','safety_reason':'unsafe robot collision'},('safety','unsafe robot collision')),
])
def test_prefix_native_termination_keeps_success_or_safety(context,wanted):
    from pac_act.hybrid import acquisition_outcome
    assert acquisition_outcome(context)==wanted
