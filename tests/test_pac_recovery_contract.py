import copy
import json
import random

import numpy as np
import pytest
import torch

from test_pac_ppo import TinyPolicy,batch


def test_successful_update_above_old_early_limit_is_accepted_once():
    from pac_act.ppo import ChunkPPO
    p=TinyPolicy(); t=ChunkPPO(p)
    t.exact_kl=lambda b:.016
    result=t.train(batch(p))
    assert result['accepted'] is True and result['update']==1 and result['epochs']==8
    assert [g['lr'] for g in t.optimizer.param_groups]==[1e-6,1e-5]


def test_rollback_restores_populated_adam_state_and_noise():
    from pac_act.ppo import ChunkPPO,UpdateRejected
    p=TinyPolicy(); t=ChunkPPO(p)
    t.update_minibatch(batch(p),torch.arange(32))
    before=t.snapshot(); b=batch(p)
    step=t.update_minibatch
    def failing_step(*args):
        step(*args)
        raise UpdateRejected('injected optimizer exception after populated Adam changed')
    t.update_minibatch=failing_step
    with pytest.raises(UpdateRejected): t.train(b)
    after=t.snapshot()
    def same(a,b):
        if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,atol=0,rtol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a: same(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b): same(x,y)
        else: assert a==b
    same(before['optimizer'],after['optimizer']); same(before['model'],after['model'])


def test_final_state_uses_committed_rng_not_discarded_sampling_rng():
    from pac_act.training import final_accepted_state
    old=dict(learner={'rng':'committed'},counters={'sampled_actions':80},registry={})
    cp=type('CP',(),{'latest':lambda self:{'state':old}})()
    counts=dict(sampled_actions=103,used_actions=80)
    value=final_accepted_state(cp,counts,{'new':'layout'})
    assert value['learner']['rng']=='committed'
    assert value['counters']['sampled_actions']==103
    assert value['registry']=={'new':'layout'}
    assert old['counters']['sampled_actions']==80


def test_resume_recovers_fingerprints_after_last_update(tmp_path):
    from pac_act.training import restore_registry
    from pac_act.identity import LayoutRegistry
    from pac_act.storage import RunStore,write_json
    store=RunStore(tmp_path); spec=dict(seed=6000000,target_count=1,role='train',arm='actor')
    folder=store.claim(spec)
    write_json(folder/'layout.json',dict(layout_fingerprint='later'))
    registry=LayoutRegistry({})
    restore_registry(store,registry)
    assert registry.used['later']['seed']==6000000
    with pytest.raises(ValueError): registry.register(dict(spec,seed=6000006),'later')


def test_preflight_cap_applies_to_all_arms(tmp_path):
    from pac_act.storage import RunStore
    from pac_act.safety import StopExperiment
    store=RunStore(tmp_path)
    for i in range(24): store.claim(dict(seed=8000000+i,target_count=i%6+1,role='preflight',arm='zero'))
    with pytest.raises(StopExperiment): store.claim(dict(seed=8000000,target_count=1,role='preflight',arm='post_update'))
