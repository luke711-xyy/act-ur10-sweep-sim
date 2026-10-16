"""ACT parameter policy and an independent encoder-only state-value model.

Uses the LeRobot 0.6.1 ACT modules directly, not an upstream PAC-ACT source copy.
"""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces

from .distribution import PhysicalNoise,PrefixGaussian

VAE=('vae_encoder','vae_encoder_cls_embed','vae_encoder_robot_state_input_proj',
     'vae_encoder_action_input_proj','vae_encoder_latent_output_proj','vae_encoder_pos_enc')
ACTIVE=(0,1,3)


def strip_vae(model):
    model.config=copy.deepcopy(model.config)
    model.config.use_vae=False
    for name in VAE:
        if hasattr(model,name): delattr(model,name)
    return model


def actor_chunk(model,batch):
    # eval disables dropout/BN mutation, NOT autograd.
    model.eval()
    output=model(batch)[0]
    if output.ndim!=3 or output.shape[1:]!=(25,4) or not torch.isfinite(output).all():
        raise ValueError('nonfinite or incompatible ACT chunk')
    return output


def encoder_features(model,batch):
    model.eval()
    state=batch['observation.state']
    latent=torch.zeros((state.shape[0],model.config.latent_dim),device=state.device,dtype=state.dtype)
    tokens=[model.encoder_latent_input_proj(latent)]
    positions=list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    if model.config.robot_state_feature:
        tokens.append(model.encoder_robot_state_input_proj(state))
    if model.config.env_state_feature:
        tokens.append(model.encoder_env_state_input_proj(batch['observation.environment_state']))
    for image in batch.get('observation.images',[]):
        features=model.backbone(image)['feature_map']
        pos=model.encoder_cam_feat_pos_embed(features).to(features.dtype)
        projected=model.encoder_img_feat_input_proj(features)
        tokens.extend(projected.flatten(2).permute(2,0,1).unbind(0))
        positions.extend(pos.flatten(2).permute(2,0,1).unbind(0))
    return model.encoder(torch.stack(tokens),pos_embed=torch.stack(positions)).mean(0)


class EncoderCritic(nn.Module):
    def __init__(self,actor):
        super().__init__()
        self.model=strip_vae(copy.deepcopy(actor))
        for attr in ('decoder','decoder_pos_embed','action_head'):
            if hasattr(self.model,attr): delattr(self.model,attr)
        self.model.requires_grad_(True)
        self.head=nn.Sequential(nn.Linear(actor.config.dim_model,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))

    def forward(self,batch):
        return self.head(encoder_features(self.model,batch)).squeeze(-1)


class ChunkPolicy(BasePolicy):
    def __init__(self,base_path,device='cpu',build_critic=True):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from sim.act.policy import build_act_processors
        from safetensors.torch import load_file
        base_path=Path(base_path)
        frozen=ACTPolicy.from_pretrained(str(base_path),local_files_only=True)
        cfg=frozen.config
        image_keys=tuple(cfg.image_features)
        if len(image_keys)!=5 or cfg.robot_state_feature.shape!=(36,) or cfg.env_state_feature.shape!=(1,):
            raise ValueError('requires five-view ResNet, state36 and target1')
        if 'resnet' not in cfg.vision_backbone or cfg.chunk_size!=25:
            raise ValueError('pre-DINO ResNet ACT step100000 required')
        obs_space=spaces.Dict({**{k:spaces.Box(0,255,tuple(cfg.input_features[k].shape),np.uint8) for k in image_keys},
            'observation.state':spaces.Box(-np.inf,np.inf,(36,),np.float32),
            'observation.environment_state':spaces.Box(-np.inf,np.inf,(1,),np.float32)})
        super().__init__(obs_space,spaces.Box(-np.inf,np.inf,(5,3),np.float32),normalize_images=False)
        self.image_keys=image_keys; self.obs_keys=tuple(obs_space.spaces)
        frozen.config.device=device
        self.preprocessor,self.postprocessor=build_act_processors(frozen.config,pretrained_path=str(base_path))
        self.actor=strip_vae(copy.deepcopy(frozen.model)).requires_grad_(True)
        self.reference=strip_vae(frozen.model).requires_grad_(False).eval()
        self.critic=EncoderCritic(self.actor) if build_critic else None
        stats=load_file(str(next(base_path.glob('*unnormalizer*.safetensors'))))
        means=stats['action.mean']; scales=stats['action.std']
        self.register_buffer('action_mean',means.clone())
        self.register_buffer('action_scale',scales.clone())
        self.noise=PhysicalNoise(scales[list(ACTIVE)])
        self.to(device)

    def prepare(self,observations):
        if set(observations)!=set(self.obs_keys):
            raise ValueError('actor observation keys differ; privileged inputs forbidden')
        batch={k:torch.as_tensor(v,device=self.device) for k,v in observations.items()}
        for k in self.image_keys:
            if batch[k].dtype!=torch.uint8: raise ValueError('rollout camera buffers must be uint8')
            batch[k]=batch[k].float()/255.
        batch=self.preprocessor(batch)
        batch['observation.images']=[batch[k] for k in self.image_keys]
        return batch

    def means(self,observations,reference=False):
        batch=self.prepare(observations)
        model=self.reference if reference else self.actor
        chunk=actor_chunk(model,batch)
        return chunk[:,:5,list(ACTIVE)]

    def values(self,observations):
        if self.critic is None: raise RuntimeError('inference policy has no critic')
        return self.critic(self.prepare(observations))

    def distribution(self,observations,lengths):
        return PrefixGaussian(self.means(observations),self.noise.log_std,lengths)

    def physical_actions(self,normalized):
        # Predicted Z is deliberately absent. The placeholder is never used.
        full=torch.zeros((*normalized.shape[:-1],4),device=normalized.device)
        full[...,list(ACTIVE)]=normalized*self.action_scale[list(ACTIVE)]+self.action_mean[list(ACTIVE)]
        return full

    def _predict(self,observation,deterministic=False):
        lengths=torch.full((len(next(iter(observation.values()))),),5,device=self.device,dtype=torch.long)
        return self.distribution(observation,lengths).sample(deterministic)

    def learned_state(self):
        # Never duplicate the 100K reference in checkpoints.
        return {k:v for k,v in self.state_dict().items() if not k.startswith('reference.')}

    def load_learned(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('reference.') for k in missing):
            raise ValueError('checkpoint has incompatible learned modules')
