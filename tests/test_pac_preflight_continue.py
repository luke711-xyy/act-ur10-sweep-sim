"""Catch repeated physical opportunities and reset budgets after a media repair."""
import json

import pytest

from pac_act.identity import seed_spec
from pac_act.storage import RunStore, sha256, write_json


def interrupted_media_run(root):
    store=RunStore(root)
    spec=seed_spec('preflight',0,'old_hybrid')
    folder=store.claim(spec)
    store.commit(dict(**spec,outcome='success',success=True,final_collected=1,
                      layout_fingerprint='initial-layout',active_actions=0))
    for name in ('configuration.json','layout.json','physical_trace.jsonl.gz'):
        (folder/name).write_bytes(b'original evidence')
    (folder/'replay.mp4').write_bytes(b'verified replay')
    write_json(folder/'media.json',dict(file='replay.mp4',bytes=15,
        sha256=sha256(folder/'replay.mp4'),verification=dict(frames=2)))
    write_json(root/'acceptance.json',dict(bundle_identity='old',incomplete=True,
        engineering_passed=False,checks={},budget=dict(started_at=10.,deadline=7210.,
        collect_deadline=7210.,free_gib=8,output_gib=5,rss_gib=10.,rss_dwell=30.)))
    write_json(root/'status.json',dict(state='needs_attention',error='original media error'))
    return store,folder


def test_continuation_excludes_only_completed_arm_and_preserves_original_deadline(tmp_path):
    from pac_act.preflight_continuation import continuation_state, pending_comparisons
    store,folder=interrupted_media_run(tmp_path)
    before={p.name:sha256(p) for p in folder.iterdir()}
    state=continuation_state(tmp_path,'old')
    specs=pending_comparisons(state['rows'])
    assert [(s['seed'],s['arm']) for s in specs]==[
        (8000005,'old_hybrid'),(8000000,'new_zero'),(8000005,'new_zero'),
        (8000000,'new_exploration'),(8000005,'new_exploration')]
    assert state['budget']['started_at']==10. and state['budget']['deadline']==7210.
    assert {p.name:sha256(p) for p in folder.iterdir()}==before
    assert len(store.rows())==1


@pytest.mark.parametrize('hazard',['orphan','updated','stochastic','tampered_result','tampered_video'])
def test_continuation_refuses_ambiguous_or_changed_history(tmp_path,hazard):
    from pac_act.preflight_continuation import continuation_state
    store,folder=interrupted_media_run(tmp_path)
    if hazard=='orphan': store.claim(seed_spec('preflight',5,'old_hybrid'))
    elif hazard=='updated': (tmp_path/'updates.jsonl').write_text('{"accepted":true}\n')
    elif hazard=='stochastic':
        spec=seed_spec('preflight',5,'new_exploration'); path=store.claim(spec)
        store.commit(dict(**spec,outcome='failure',success=False,final_collected=0))
    elif hazard=='tampered_result': write_json(folder/'result.json',{'success':False})
    else: (folder/'replay.mp4').write_bytes(b'altered')
    with pytest.raises((RuntimeError,ValueError)):
        continuation_state(tmp_path,'old')


def test_continuation_authorization_is_single_use_and_archives_failure(tmp_path):
    from pac_act.preflight_continuation import continuation_state, claim_continuation
    interrupted_media_run(tmp_path)
    previous=(tmp_path/'acceptance.json').read_bytes()
    state=continuation_state(tmp_path,'old')
    claim_continuation(tmp_path,state,'replacement')
    assert (tmp_path/'acceptance_before_authorized_continue.json').read_bytes()==previous
    evidence=json.loads((tmp_path/'authorized_preflight_continuation.json').read_text())
    assert evidence['budget']['deadline']==7210.
    assert evidence['completed_keys']==['preflight_8000000_old_hybrid']
    with pytest.raises(FileExistsError): claim_continuation(tmp_path,state,'replacement')
