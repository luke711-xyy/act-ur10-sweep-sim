import copy
import random
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def _stable_baselines_test_shim_when_optional_extra_is_missing(monkeypatch):
    """Keep optimizer tests runnable without installing the PAC training extra."""
    try:
        from stable_baselines3 import PPO  # noqa: F401
        return
    except ImportError:
        pass

    package = types.ModuleType("stable_baselines3")
    package.__path__ = []
    common = types.ModuleType("stable_baselines3.common")
    common.__path__ = []
    policies = types.ModuleType("stable_baselines3.common.policies")

    class BasePolicy(nn.Module):
        def __init__(self, observation_space, action_space, normalize_images=False):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
            self.normalize_images = normalize_images

        @property
        def device(self):
            parameter = next(self.parameters(), None)
            return parameter.device if parameter is not None else torch.device("cpu")

    class PPO:
        def __init__(self, *args, **kwargs):
            self._test_init = (args, kwargs)

    package.PPO = PPO
    policies.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "stable_baselines3", package)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.policies", policies)


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


def test_update_exception_rolls_back_model_optimizer_noise_and_rng():
    from pac_act.ppo import ChunkPPO,UpdateRejected
    p=TinyPolicy(); trainer=ChunkPPO(p); b=batch(p); saved=trainer.snapshot()
    def corrupt(batch,indices):
        with torch.no_grad(): p.actor.bias.add_(10.); p.noise.log_std.add_(1.)
        random.random(); np.random.rand(); torch.rand(2)
        raise UpdateRejected('injected optimizer exception')
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


def test_ppo_updates_act_modules_without_changing_any_dino_encoder(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_pac_dinov3 import (
        _TinyPPOACTModel,
        _encoder_digest,
        _fake_local_checkpoint,
        _observation,
    )
    from pac_act.config import DEFAULT
    from pac_act.ppo import Batch, ChunkPPO

    checkpoint = tmp_path / "ppo-dinov3-checkpoint"
    checkpoint.mkdir()
    config, pac_policy, _loads = _fake_local_checkpoint(
        checkpoint, monkeypatch, model_factory=_TinyPPOACTModel)
    policy = pac_policy.ChunkPolicy(checkpoint, device="cpu", build_critic=True)
    trainer = ChunkPPO(policy, replace(DEFAULT, microbatch=1, epochs=1))

    one = _observation(config)
    observations = {key: value.unsqueeze(0).repeat(2, *([1] * value.ndim))
                    for key, value in one.items()}
    observations["observation.state"][0, 0] = -1.0
    observations["observation.state"][1, 0] = 1.0
    with torch.no_grad():
        old_means = policy.means(observations)
        old_log_std = policy.noise.log_std.detach().expand(2, 5, 3).clone()
        old_values = policy.values(observations)
    actions = old_means.detach().clone()
    actions[0] += 0.03
    actions[1] -= 0.01
    rollout = Batch(
        observations=observations,
        actions=actions,
        old_means=old_means.detach(),
        old_log_std=old_log_std,
        lengths=torch.full((2,), 5, dtype=torch.long),
        advantages=torch.tensor([-1.0, 2.0]),
        old_values=old_values.detach(),
        returns=old_values.detach() + torch.tensor([-0.5, 1.5]),
        env_rewards=torch.zeros(2),
    )

    encoder_modules = (
        policy.actor.backbone.encoder,
        policy.reference.backbone.encoder,
        policy.critic.model.backbone.encoder,
    )
    before_encoder = tuple(_encoder_digest(module) for module in encoder_modules)
    before_actor = policy.actor.action_head.weight.detach().clone()
    before_critic = policy.critic.head[-1].bias.detach().clone()

    trainer.update_minibatch(rollout, torch.arange(2))

    assert not torch.equal(policy.actor.action_head.weight, before_actor)

    # A zero actor-mean residual isolates the value gradient while retaining
    # the production PPO clipping and learning-rate settings.
    with torch.no_grad():
        critic_old_means = policy.means(observations)
        critic_old_values = policy.values(observations)
    critic_only_rollout = Batch(
        observations=observations,
        actions=critic_old_means.detach().clone(),
        old_means=critic_old_means.detach(),
        old_log_std=policy.noise.log_std.detach().expand(2, 5, 3).clone(),
        lengths=torch.full((2,), 5, dtype=torch.long),
        advantages=torch.zeros(2),
        old_values=critic_old_values.detach(),
        returns=critic_old_values.detach() + torch.tensor([-0.5, 1.5]),
        env_rewards=torch.zeros(2),
    )
    before_critic = policy.critic.head[-1].bias.detach().clone()
    trainer.update_minibatch(critic_only_rollout, torch.arange(2))
    assert not torch.equal(policy.critic.head[-1].bias, before_critic)
    assert tuple(_encoder_digest(module) for module in encoder_modules) == before_encoder
