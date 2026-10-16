import copy
import random

import numpy as np
import pytest
import torch
from torch import nn


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        from pac_act.distribution import PhysicalNoise
        self.actor=nn.Linear(2,15); self.critic=nn.Linear(2,1)
        self.noise=PhysicalNoise(torch.tensor([.001,.001,.01]))
    @property
    def device(self): return torch.device('cpu')
    def means(self,obs): return self.actor(obs['x']).reshape(-1,5,3)
    def values(self,obs): return self.critic(obs['x']).flatten()
    def learned_state(self): return self.state_dict()
    def load_learned(self,state): self.load_state_dict(state)


def batch(policy,count=32):
    from pac_act.ppo import Batch
    x=torch.linspace(-1,1,count*2).reshape(count,2)
    with torch.no_grad():
        mean=policy.means({'x':x}); logs=policy.noise.log_std.detach().expand(count,5,3).clone()
        action=mean+torch.linspace(-.1,.1,count)[:,None,None]
    return Batch({'x':x},action,mean,logs,torch.full((count,),5,dtype=torch.long),
        torch.linspace(-1,1,count),torch.zeros(count),torch.ones(count),torch.zeros(count))


def test_gae_bootstraps_cuts_but_not_task_termination():
    from pac_act.ppo import gae
    adv,ret=gae([1.,2.],[3.,4.],[4.,10.],[False,True])
    np.testing.assert_allclose(adv,[.1,-2.],atol=1e-7)
    np.testing.assert_allclose(ret,[3.1,2.])
    adv,_=gae([1.],[3.],[10.],[False]); assert adv[0]==8.


def test_update_starts_at_ratio_one_and_has_differentiable_kl():
    from pac_act.ppo import ChunkPPO
    p=TinyPolicy(); trainer=ChunkPPO(p); b=batch(p)
    metrics=trainer.objective(b,torch.arange(32))
    assert metrics['ratio'].detach().tolist()==[1.]*32
    assert metrics['kl'].item()==0.
    with torch.no_grad(): p.actor.bias.add_(.01)
    loss=trainer.objective(b,torch.arange(32))['kl']
    loss.backward()
    assert p.actor.bias.grad.abs().sum()>0


def test_microbatch_matches_logical_batch_and_preserves_learning_rates():
    from dataclasses import replace
    from pac_act.config import DEFAULT
    from pac_act.ppo import ChunkPPO
    p=TinyPolicy(); q=copy.deepcopy(p); b=batch(p)
    a=ChunkPPO(p,replace(DEFAULT,microbatch=4,epochs=1))
    c=ChunkPPO(q,replace(DEFAULT,microbatch=1,epochs=1))
    a.update_minibatch(b,torch.arange(32)); c.update_minibatch(b,torch.arange(32))
    for x,y in zip(p.parameters(),q.parameters()): torch.testing.assert_close(x,y,atol=2e-7,rtol=2e-6)
    assert [g['lr'] for g in a.optimizer.param_groups]==[1e-6,1e-5]
    a.restore(a.snapshot())
    assert [g['lr'] for g in a.optimizer.param_groups]==[1e-6,1e-5]


def test_hard_kl_rolls_back_model_optimizer_noise_and_rng():
    from pac_act.ppo import ChunkPPO,UpdateRejected
    p=TinyPolicy(); trainer=ChunkPPO(p); b=batch(p); saved=trainer.snapshot()
    def corrupt(batch,indices):
        with torch.no_grad(): p.actor.bias.add_(10.); p.noise.log_std.add_(1.)
        random.random(); np.random.rand(); torch.rand(2)
        return {}
    trainer.update_minibatch=corrupt
    with pytest.raises(UpdateRejected): trainer.train(b)
    for key,value in saved['model'].items(): torch.testing.assert_close(p.state_dict()[key],value)
    assert trainer.optimizer.state_dict()==saved['optimizer']
    torch.testing.assert_close(torch.get_rng_state(),saved['rng']['torch'])
    assert random.getstate()==saved['rng']['python']
    assert trainer._n_updates==0


def test_nonfinite_update_is_rejected_and_rollout_data_not_reused():
    from pac_act.ppo import ChunkPPO,UpdateRejected
    p=TinyPolicy(); t=ChunkPPO(p); b=batch(p); b.returns[0]=float('nan')
    with pytest.raises(UpdateRejected): t.train(b)
    with pytest.raises(RuntimeError,match='consumed'): t.train(b)
