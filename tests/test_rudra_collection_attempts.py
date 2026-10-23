import json

import pytest


def slots():
    return [dict(episode_id=f'preview_goal_6_{i+1:04d}',target_count=6,seed=seed,
                 layout_id=f'train_n6_layout_{i:03d}') for i,seed in enumerate([100,101,102,104])]


def failed(slot):
    return dict(event='episode_failed',episode_id=slot['episode_id'],target_count=slot['target_count'],seed=slot['seed'])


def test_replacement_skips_all_original_and_previously_allocated_seeds():
    from scripts.rudra_collection_attempts import CollectionAttempts
    entries=slots();history=CollectionAttempts(entries,[failed(entries[0])])
    first=history.next_attempt(entries[0]['episode_id'])
    assert (first['seed'],first['source_seed'],first['generation_attempt'])==(103,100,2)
    history.register(first);history.record_failure(failed(first))
    second=history.next_attempt(entries[0]['episode_id'])
    assert second['seed']==105 and second['generation_attempt']==3
    history.register(second)
    history.record_failure(failed(entries[1]))
    assert history.next_attempt(entries[1]['episode_id'])['seed']==106


def test_allocated_but_uncommitted_attempt_is_recovered_without_silent_seed_change():
    from scripts.rudra_collection_attempts import CollectionAttempts
    entries=slots();allocation=dict(entries[0],seed=103,source_seed=100,generation_attempt=2)
    history=CollectionAttempts(entries,[failed(entries[0]),dict(allocation,event='attempt_allocated')])
    assert history.next_attempt(entries[0]['episode_id'])==allocation


def test_history_rejects_unallocated_failure_or_cross_slot_seed_collision():
    from scripts.rudra_collection_attempts import CollectionAttempts
    entries=slots();history=CollectionAttempts(entries,[])
    with pytest.raises(ValueError):history.record_failure(dict(failed(entries[0]),seed=999))
    with pytest.raises(ValueError):history.register(dict(entries[0],seed=101,source_seed=100,generation_attempt=2))


def test_replacement_success_requires_recorded_allocation_and_original_provenance(tmp_path):
    from scripts.rudra_collection_attempts import CollectionAttempts
    from scripts.collect_rudra_seed_matched import recovered_successes
    entries=slots();history=CollectionAttempts(entries,[failed(entries[0])])
    allocation=history.next_attempt(entries[0]['episode_id']);history.register(allocation)
    record=dict(allocation,requested_seed=100,success=True,schema_version=5,frame_count=1,
                camera_views=['top','wrist'],source_manifest_sha256='source',scene_profile_sha256='camera')
    (tmp_path/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError):recovered_successes(tmp_path,entries,'source','camera')
    done=recovered_successes(tmp_path,entries,'source','camera',allowed_attempts=history.assignments)
    assert set(done)=={entries[0]['episode_id']}
    record['source_seed']=999;(tmp_path/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError):
        recovered_successes(tmp_path,entries,'source','camera',allowed_attempts=history.assignments)
