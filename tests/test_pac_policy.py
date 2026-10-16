"""Numerical contract tests: an unmasked suffix or wrong unit must fail."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_prefix_probabilities_ignore_z_and_unexecuted_suffix():
    from pac_act.distribution import PrefixGaussian
    mean=torch.zeros(2,5,3); std=torch.ones(5,3)
    g=PrefixGaussian(mean,std.log(),torch.tensor([1,3]))
    action=mean.clone(); action[0,1:]=900; action[1,3:]=-900
    expected=torch.tensor([-3*math.log(2*math.pi)/2,-9*math.log(2*math.pi)/2])
    torch.testing.assert_close(g.log_prob(action),expected)
    torch.testing.assert_close(g.kl(mean,std.log()),torch.zeros(2))
    changed=mean.clone(); changed[0,0,0]=2
    torch.testing.assert_close(g.kl(changed,std.log()),torch.tensor([2.,0.]))
    with pytest.raises(ValueError): PrefixGaussian(mean,std.log(),torch.tensor([0,3]))


def test_noise_physical_units_and_clamps_have_independent_components():
    from pac_act.distribution import PhysicalNoise
    scales=torch.tensor([.002,.001,.01])
    noise=PhysicalNoise(scales)
    wanted=torch.tensor([.0001,.0001,math.pi/1800]).expand(5,3)
    torch.testing.assert_close(noise.physical_std(),wanted)
    with torch.no_grad(): noise.log_std[0,0]=100; noise.log_std[0,1]=-100
    noise.project_()
    assert noise.physical_std()[0,0].item()==pytest.approx(.0003)
    assert noise.physical_std()[0,1].item()==pytest.approx(.00001)
    assert noise.physical_std()[1,0].item()==pytest.approx(.0001)


def small_act():
    from lerobot.configs.types import FeatureType,PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACT
    cfg=ACTConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(36,)),
        'observation.environment_state':PolicyFeature(FeatureType.ENV,(1,))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(4,))},
        dim_model=32,n_heads=4,dim_feedforward=64,n_encoder_layers=1,n_decoder_layers=1,
        chunk_size=25,n_action_steps=5,latent_dim=4,n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,dropout=.4)
    return ACT(cfg).eval()


def test_stripped_actor_equivalence_and_gradients_without_decoder_in_critic():
    from pac_act.policy import strip_vae,EncoderCritic,actor_chunk
    source=small_act(); frozen=copy.deepcopy(source).requires_grad_(False)
    actor=copy.deepcopy(source); strip_vae(actor)
    obs={'observation.state':torch.randn(2,36),'observation.environment_state':torch.ones(2,1)}
    original=frozen(obs)[0]
    torch.testing.assert_close(actor_chunk(actor,obs),original)
    actor.train(); out=actor_chunk(actor,obs)
    torch.testing.assert_close(out,original)
    out[:,:5,[0,1,3]].square().sum().backward()
    for name in ('encoder','decoder','action_head','encoder_robot_state_input_proj'):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in getattr(actor,name).parameters())
    assert not any(n.startswith('vae_') for n,_ in actor.named_parameters())
    critic=EncoderCritic(actor)
    assert not hasattr(critic.model,'decoder')
    value=critic(obs); assert value.shape==(2,)
    value.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in critic.model.encoder.parameters())
    assert all(p.grad is None for p in frozen.parameters())


def test_prior_uses_only_executed_normalized_means():
    from pac_act.distribution import prior_cost
    a=torch.zeros(2,5,3); b=a.clone(); a[0,0]=1; a[0,1:]=100
    assert prior_cost(a,b,torch.tensor([1,5])).tolist()==[2.,0.]
