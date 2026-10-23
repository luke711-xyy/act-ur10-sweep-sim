import hashlib
import json
from types import SimpleNamespace

import pytest


def source_records():
    return [dict(episode_id=f'preview_goal_{n}_{i+1:04d}', target_count=n,
                 seed=4200000+n*100000+i*256, requested_seed=100+i,
                 layout_id=f'train_n{n}_layout_{i:03d}', layout_kind='independent',
                 episode_kind='expert', split='train', success=True,
                 schema_version=5, frame_count=800, action_dim=4, state_dim=36)
            for n in range(1,7) for i in range(50)]


def test_source_plan_uses_all_300_actual_seeds_without_changing_source(tmp_path):
    from scripts.collect_rudra_seed_matched import load_source_plan
    source=tmp_path/'manifest.jsonl'
    records=source_records()
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    before=source.read_bytes()
    plan=load_source_plan(source)
    assert len(plan['entries'])==300
    assert [(r['target_count'],r['seed']) for r in plan['entries']]==[
        (r['target_count'],r['seed']) for r in records]
    assert plan['entries'][0]['source_requested_seed']==records[0]['requested_seed']
    assert plan['source_manifest_sha256']==hashlib.sha256(before).hexdigest()
    assert source.read_bytes()==before


@pytest.mark.parametrize('fault',['duplicate_seed','wrong_balance','unsuccessful'])
def test_source_plan_rejects_invalid_300_layout_identity(tmp_path,fault):
    from scripts.collect_rudra_seed_matched import load_source_plan
    records=source_records()
    if fault=='duplicate_seed':records[1]['seed']=records[0]['seed']
    elif fault=='wrong_balance':records[-1]['target_count']=1
    else:records[0]['success']=False
    source=tmp_path/'manifest.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError):load_source_plan(source)


def test_known_physical_failure_keeps_fixed_seed_and_measured_count(monkeypatch,tmp_path):
    from scripts import collect_rudra_seed_matched as entry
    from sim.controllers.rudra_contact import demonstration_config
    from sim.environments.rudra_sweep_env import collection_config
    calls=[]
    def failed(cfg,seed):
        calls.append((cfg.task.target_count,seed))
        return SimpleNamespace(success=False,collected=5,sampled_frames=906,
                               failure_reason='exact count not reached'),None
    monkeypatch.setattr(entry,'run_candidate',failed)
    slot=dict(episode_id='preview_goal_6_0001',target_count=6,seed=4200000,layout_id='train_n6_layout_000')
    outcome=entry.capture_one(demonstration_config(collection_config()),slot,tmp_path,None,'source','camera')
    assert calls==[(6,4200000)]
    assert outcome['state']=='failed' and outcome['seed']==4200000
    assert outcome['collected']==5
    assert not (tmp_path/'manifest.jsonl').exists()


def test_no_executable_plan_preserves_unknown_count(monkeypatch,tmp_path):
    from scripts import collect_rudra_seed_matched as entry
    from sim.controllers.rudra_contact import demonstration_config
    from sim.environments.rudra_sweep_env import collection_config
    monkeypatch.setattr(entry,'run_candidate',lambda cfg,seed:(None,dict(reason='no executable plan',collected=None,frames=None)))
    slot=dict(episode_id='preview_goal_1_0001',target_count=1,seed=4200000,layout_id='layout')
    outcome=entry.capture_one(demonstration_config(collection_config()),slot,tmp_path,None,'source','camera')
    assert outcome['collected'] is None and outcome['frames'] is None


def test_committed_manifest_prevents_duplicate_physical_execution_on_resume(tmp_path):
    from scripts.collect_rudra_seed_matched import recovered_successes
    slot=dict(episode_id='preview_goal_1_0001',target_count=1,seed=4200000)
    record=dict(slot,success=True,frame_count=50,schema_version=5,camera_views=['top','wrist'],
                source_manifest_sha256='source',scene_profile_sha256='camera')
    (tmp_path/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    done=recovered_successes(tmp_path,[slot],'source','camera')
    assert set(done)=={slot['episode_id']}
    record['seed']+=1;(tmp_path/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='identity'):
        recovered_successes(tmp_path,[slot],'source','camera')


def test_frozen_profile_tampering_is_rejected_before_resume(tmp_path):
    from scripts.collect_rudra_seed_matched import verify_frozen_files
    path=tmp_path/'scene-profile.yaml';path.write_text('camera: original\n')
    pinned={'scene-profile.yaml':hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_frozen_files(tmp_path,pinned)
    path.write_text('camera: changed\n')
    with pytest.raises(ValueError,match='frozen'):
        verify_frozen_files(tmp_path,pinned)


def test_prepare_freezes_task_config_reloaded_by_scene_reset(monkeypatch,tmp_path):
    import yaml
    from scripts import collect_rudra_seed_matched as entry
    from sim.controllers.rudra_contact import demonstration_config
    from sim.environments.rudra_sweep_env import collection_config
    cfg=demonstration_config(collection_config())
    task=tmp_path/'live-task.yaml';task.write_text('objects:\n  count: 6\n')
    profile=tmp_path/'live-profile.yaml'
    profile.write_text(yaml.safe_dump(dict(sweep_task=dict(source_config=str(task)),
                                          render=dict(width=640,height=480))))
    cfg.set_path('scene.profile',str(profile))
    monkeypatch.setattr(entry,'checked_configuration',lambda gate:cfg.copy())
    records=source_records()
    for record in records:record['expert_generation_protocol']=entry.expert_generation_protocol(cfg)
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    gate=tmp_path/'gate.json';gate.write_text('{}')
    out=tmp_path/'capture';out.mkdir()
    frozen,_,pinned=entry.prepare(out,source,gate)
    frozen_profile=yaml.safe_load((out/'scene-profile.yaml').read_text())
    frozen_task=__import__('pathlib').Path(frozen_profile['sweep_task']['source_config'])
    assert frozen_task==out/'task-source.yaml'
    assert frozen_task.read_bytes()==task.read_bytes()
    assert 'task-source.yaml' in pinned['files']
    task.write_text('objects:\n  count: 99\n')
    assert yaml.safe_load(frozen_task.read_text())['objects']['count']==6
    assert frozen.scene.profile==str(out/'scene-profile.yaml')
    assert list(frozen.act.image_size)==[640,480]
    assert frozen.controller.to_dict()==cfg.controller.to_dict()
    assert frozen.planner.to_dict()==cfg.planner.to_dict()
    entry.verify_frozen_files(out,pinned['files'])


@pytest.mark.parametrize('replacement',[False,True])
def test_success_uses_native_writer_and_remains_a_train_expert(monkeypatch,tmp_path,replacement):
    import numpy as np
    from scripts import collect_rudra_seed_matched as entry
    from sim.act.dataset import ActDatasetWriter
    from sim.act.rollout import ActRolloutResult
    from sim.controllers.rudra_contact import demonstration_config
    from sim.environments.rudra_sweep_env import collection_config
    positions=np.array([[.1,-.2+i*.07,.004] for i in range(6)])
    mask=np.array([True,False,False,False,False,False])
    poses=np.c_[positions,np.tile([1.,0.,0.,0.],(6,1))]
    observations=[dict(images={v:np.zeros((8,8,3),dtype=np.uint8) for v in ['top','wrist']},
                       state=np.zeros(36),environment_state=np.array([1.]),
                       supervisor_state=np.array([6.,1.,1.]),t=i*.04,phase='sweep',
                       contact_latched=True,object_pose=poses,object_collected=mask,fully_collected=1)
                  for i in range(2)]
    result=ActRolloutResult(True,'',observations=observations,actions=np.zeros((2,4)),
                            collected=1,total=6,target_count=1,target_indices=[0],
                            initial_object_positions=positions,final_collected_mask=mask,
                            first_contact_position=np.array([.2,0,0]),sampled_frames=2,
                            trace=[dict(t=.01,tcp=np.zeros(3),phase='sweep')])
    monkeypatch.setattr(entry,'run_candidate',lambda cfg,seed:(result,None))
    slot=dict(episode_id='preview_goal_1_0001',target_count=1,seed=4276545,
              layout_id='train_n1_layout_049',source_requested_seed=4276544)
    if replacement:slot.update(seed=4276546,source_seed=4276545,generation_attempt=2)
    outcome=entry.capture_one(demonstration_config(collection_config()),slot,tmp_path,
                              ActDatasetWriter(str(tmp_path),image_format='webp-lossless'),'source','camera')
    record=json.loads((tmp_path/'manifest.jsonl').read_text())
    assert outcome['state']=='saved'
    assert (record['episode_kind'],record['split'],record['seed'])==('expert','train',slot['seed'])
    assert record['source_requested_seed']==4276544
    assert record['frame_count']==2 and record['camera_views']==['top','wrist']
    assert 'target_indices' not in record and record['target_selection']['collected_component_ids']==[0]
    if replacement:
        assert record['requested_seed']==record['source_seed']==4276545
        assert record['generation_attempt']==2 and record['seed_replaced'] is True
        original=dict(slot,seed=4276545)
        assert 'preview_goal_1_0001' in entry.recovered_successes(tmp_path,[original],'source','camera',
                                  allowed_attempts={slot['episode_id']:{4276545:1,4276546:2}})
    else:
        assert 'preview_goal_1_0001' in entry.recovered_successes(tmp_path,[slot],'source','camera')
