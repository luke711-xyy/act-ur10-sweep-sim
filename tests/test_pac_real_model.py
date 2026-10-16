"""Optional full checkpoint CPU contract test; not a physical clearance."""
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get('PAC_ACT_BASE'),reason='supply PAC_ACT_BASE to verify the actual frozen100K model')
def test_actual_five_view_checkpoint_mean_and_gradients():
    from pac_act.policy import ChunkPolicy
    policy=ChunkPolicy(Path(os.environ['PAC_ACT_BASE']),'cpu')
    obs={k:torch.zeros((1,*feature.shape),dtype=torch.uint8 if k in policy.image_keys else torch.float32)
         for k,feature in policy.observation_space.spaces.items()}
    obs['observation.environment_state'][:]=1
    with torch.no_grad():
        old=policy.means(obs,reference=True); current=policy.means(obs)
    torch.testing.assert_close(current,old,atol=0,rtol=0)
    assert current.shape==(1,5,3)
    physical=policy.physical_actions(current)
    assert torch.count_nonzero(physical[...,2])==0
    mean=policy.means(obs); value=policy.values(obs)
    (mean.square().mean()+value.square().mean()).backward()
    for module in (policy.actor.backbone,policy.actor.encoder,policy.actor.decoder,policy.actor.action_head,
                   policy.critic.model.backbone,policy.critic.model.encoder,policy.critic.head):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad!=0) for p in module.parameters())
    assert all(p.grad is None for p in policy.reference.parameters())
