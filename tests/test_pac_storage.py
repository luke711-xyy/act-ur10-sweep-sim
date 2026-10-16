import json
from pathlib import Path

import pytest
import torch


def test_failed_checkpoint_keeps_last_two_valid_points(tmp_path):
    from pac_act.storage import Checkpoints
    cp=Checkpoints(tmp_path,'identity')
    cp.save({'value':torch.tensor([1.])},1)
    cp.save({'value':torch.tensor([2.])},2)
    def fail(value,stream):
        stream.write(b'broken'); raise OSError('injected write error')
    with pytest.raises(OSError,match='injected'): cp.save({'value':torch.ones(1)},3,writer=fail)
    assert cp.latest()['state']['value'].item()==2
    assert len(cp.index()['slots'])==2
    assert json.loads((tmp_path/'checkpoint_error.json').read_text())['error'].endswith("injected write error')")


def test_corrupt_latest_uses_previous_verified_slot_and_identity_must_match(tmp_path):
    from pac_act.storage import Checkpoints
    cp=Checkpoints(tmp_path,'identity'); cp.save({'n':1},1); cp.save({'n':2},2)
    latest=cp.index()['slots'][-1]
    (tmp_path/latest['file']).write_bytes(b'bad')
    assert cp.latest()['state']['n']==1
    with pytest.raises(ValueError,match='identity'): Checkpoints(tmp_path,'different').latest()


def test_attempts_never_retry_or_fill_unknown_count(tmp_path):
    from pac_act.storage import RunStore
    store=RunStore(tmp_path)
    spec=dict(seed=6000000,target_count=1,role='train',arm='actor')
    store.claim(spec)
    with pytest.raises(FileExistsError): store.claim(spec)
    store.commit(dict(**spec,outcome='interrupted',final_collected=None,active_actions=3))
    assert store.rows()[0]['final_collected'] is None
    assert store.next_seed('train')==6000001
    with pytest.raises(ValueError,match='already'): store.commit(dict(**spec,final_collected=0))


def test_resource_dwell_and_deadline_survive_restore(tmp_path):
    from pac_act.safety import Guard,StopExperiment
    current=[100.]
    guard=Guard(tmp_path,started_at=0.,deadline=200.,collect_deadline=150.,
        free_gib=8,output_gib=5,clock=lambda:current[0])
    guard.poll(free_bytes=9*2**30,rss_bytes=11*2**30,output_bytes=0)
    current[0]=129.
    guard.poll(free_bytes=9*2**30,rss_bytes=11*2**30,output_bytes=0)
    current[0]=130.
    with pytest.raises(StopExperiment,match='memory'): guard.poll(free_bytes=9*2**30,rss_bytes=11*2**30,output_bytes=0)
    guard2=Guard(tmp_path,**guard.identity(),clock=lambda:201.)
    with pytest.raises(StopExperiment,match='deadline'): guard2.poll(free_bytes=99*2**30,rss_bytes=0,output_bytes=0)


def test_seed_schedule_reserves_roles_and_rejects_fingerprint_collisions():
    from pac_act.identity import LayoutRegistry,seed_spec
    r=LayoutRegistry({'seeds':[8000005],'fingerprints':['old']})
    r.register(seed_spec('train',0),'new')
    with pytest.raises(ValueError): r.register(seed_spec('train',1),'old')
    with pytest.raises(ValueError): r.register(seed_spec('preflight',5),'other')
    with pytest.raises(ValueError): r.register(seed_spec('validation',0),'new')
    assert [seed_spec('validation',i)['target_count'] for i in range(6)]==[1,2,3,4,5,6]
