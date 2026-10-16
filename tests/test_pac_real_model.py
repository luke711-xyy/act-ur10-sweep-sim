"""Optional full checkpoint CPU contract test; not a physical clearance."""
import os
import hashlib
import json
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


@pytest.mark.skipif(not os.environ.get('PAC_ACT_BASE'),reason='set PAC_ACT_BASE to the local DINOv3 ACT step_200000 directory')
def test_actual_dinov3_200k_reference_equivalence(monkeypatch):
    from safetensors.torch import load_file
    from sim.act.configuration_dinov3 import DINOv3ACTConfig
    from sim.act.dinov3 import DINOv3ACTPolicy
    from sim.act.policy import build_act_processors
    from pac_act.policy import ChunkPolicy, DINO_CAMERA_ORDER

    base=Path(os.environ['PAC_ACT_BASE']).resolve(strict=True)
    config_path=base/'config.json'
    saved_config=json.loads(config_path.read_text())
    assert saved_config['type']=='act_dinov3'
    assert saved_config['vision_backbone']=='dinov3_vits16'
    checkpoint_path=base/'model.safetensors'
    def digest(path):
        value=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):
                value.update(block)
        return value.hexdigest()
    base_hash=digest(checkpoint_path)
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    # A local checkpoint must never fall back to fetching another DINO revision.
    import sim.act.dinov3 as dino_module
    monkeypatch.setattr(dino_module,'hf_hub_download',lambda *a,**k:pytest.fail('DINO download attempted'))

    policy=ChunkPolicy(base,'cpu',build_critic=True)
    policy.eval()
    config=DINOv3ACTConfig.from_pretrained(str(base),local_files_only=True)
    config.device='cpu'
    native=DINOv3ACTPolicy.from_pretrained(str(base),config=config,local_files_only=True)
    native.to('cpu').eval()
    native_preprocessor,_=build_act_processors(native.config,pretrained_path=str(base))

    expected_keys=tuple(f'observation.images.{name}' for name in DINO_CAMERA_ORDER)
    assert tuple(policy.image_keys)==expected_keys
    assert tuple(native.config.image_features)==expected_keys
    pixels=torch.arange(3*320*320,dtype=torch.int32).reshape(1,3,320,320)
    observations={key:((pixels+camera*37)%256).to(torch.uint8)
        for camera,key in enumerate(expected_keys)}
    observations['observation.state']=torch.linspace(-0.75,0.75,36).reshape(1,36)
    observations['observation.environment_state']=torch.tensor([[1.0]])
    pac_batch=policy.prepare(observations)
    native_input={key:(value.float()/255.0 if key in expected_keys else value.clone())
        for key,value in observations.items()}
    native_batch=native_preprocessor(native_input)
    native_batch['observation.images']=[native_batch[key] for key in expected_keys]
    for key in (*expected_keys,'observation.state','observation.environment_state'):
        torch.testing.assert_close(pac_batch[key],native_batch[key],atol=0,rtol=0)

    action_stats=load_file(str(next(base.glob('*unnormalizer*.safetensors'))))
    torch.testing.assert_close(policy.action_mean.cpu(),action_stats['action.mean'],atol=0,rtol=0)
    torch.testing.assert_close(policy.action_scale.cpu(),action_stats['action.std'],atol=0,rtol=0)
    with torch.inference_mode():
        native_chunk=native.model(native_batch)[0]
        reference_chunk=policy.reference(pac_batch)[0]
    torch.testing.assert_close(reference_chunk,native_chunk,atol=1e-6,rtol=1e-6)

    means=policy.means(observations)
    assert means.shape==(1,5,3) and torch.isfinite(means).all()
    physical=policy.physical_actions(means)
    assert physical.shape==(1,5,4) and torch.count_nonzero(physical[...,2])==0

    policy.train()
    assert not policy.actor.backbone.encoder.training
    loss=policy.means(observations).square().mean()+policy.values(observations).square().mean()
    loss.backward()
    for module in (policy.actor.encoder,policy.actor.decoder,policy.actor.action_head,
                   policy.actor.encoder_img_feat_input_proj,policy.critic.model.encoder,policy.critic.head):
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   and torch.any(parameter.grad!=0) for parameter in module.parameters())
    for _name,_backbone,encoder in policy._dino_encoder_modules():
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    policy._assert_dino_encoders_unchanged()
    assert digest(checkpoint_path)==base_hash
