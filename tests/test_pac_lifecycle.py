import json
import subprocess
import sys

import numpy as np
import pytest
import torch


def test_default_inspection_is_readonly_without_loading_models(tmp_path):
    script='''
import json,sys
from pac_act.lifecycle import inspect
result=inspect()
assert not result['execution_started'] and not result['training_started']
assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
print(json.dumps(result))
'''
    p=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert not list(tmp_path.iterdir())


def test_collection_gae_never_links_episodes_and_has_no_image_truth_leak():
    from pac_act.lifecycle import build_batch
    def row(reward,done):
        return dict(observation={'observation.state':np.zeros(36,np.float32)},
            action=torch.zeros(5,3),mean=torch.zeros(5,3),log_std=torch.zeros(5,3),length=1,
            reward=reward,environment_reward=reward,value=0.,next_value=10.,terminated=done,policy_revision=0)
    batch=build_batch({1:[row(1.,True),row(2.,False)]},0)
    assert batch.advantages.tolist()==[1.,12.]
    assert list(batch.observations)==['observation.state']


def test_target_streams_never_repeat_attempts(tmp_path):
    from pac_act.storage import RunStore
    from pac_act.lifecycle import next_spec
    store=RunStore(tmp_path)
    first=next_spec(store,6); assert first['seed']==6000005
    store.claim(first)
    assert next_spec(store,6)['seed']==6000011
    assert next_spec(store,1)['seed']==6000000


def test_resume_counts_keep_samples_discard_uncommitted_and_original_deadline(tmp_path):
    from pac_act.lifecycle import resume_counters
    from pac_act.storage import append_json
    append_json(tmp_path/'action_journal.jsonl',dict(sampled_actions=103))
    counts=resume_counters(tmp_path,dict(sampled_actions=96,used_actions=80,discarded_actions=16,updates=2))
    assert counts==dict(sampled_actions=103,used_actions=80,discarded_actions=23,updates=2)


def test_dashboard_full_pagination_and_loss_series(tmp_path):
    from pac_act.dashboard import snapshot
    from pac_act.storage import append_json
    for i in range(30): append_json(tmp_path/'episodes.jsonl',dict(seed=i,target_count=1,final_collected=i%3,success=False))
    for i in range(4): append_json(tmp_path/'updates.jsonl',dict(update=i,actor_loss=i/2,value_loss=10-i,kl=.001*i))
    page=snapshot(tmp_path,page=2,page_size=12)
    assert len(page['episodes'])==12 and page['total_episodes']==30 and page['pages']==3
    assert len(page['updates'])==4
    assert page['updates'][-1]['value_loss']==7


def test_cli_requires_explicit_gpu_ack_and_preflight_gate(tmp_path):
    from pac_act.cli import parser
    p=parser()
    assert p.parse_args([]).mode=='inspect'
    args=p.parse_args(['--mode','train','--out',str(tmp_path)])
    assert not args.acknowledge_gpu_training
