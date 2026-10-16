"""ACT parameter policy and an independent encoder-only state-value model.

Uses the LeRobot 0.6.1 ACT modules directly, not an upstream PAC-ACT source copy.
"""
import copy
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces

from .distribution import PhysicalNoise,PrefixGaussian

VAE=('vae_encoder','vae_encoder_cls_embed','vae_encoder_robot_state_input_proj',
     'vae_encoder_action_input_proj','vae_encoder_latent_output_proj','vae_encoder_pos_enc')
ACTIVE=(0,1,3)
DINO_TYPE='act_dinov3'
DINO_BACKBONE='dinov3_vits16'
DINO_CAMERA_ORDER=('corner_xminus_yminus','corner_xminus_yplus',
                   'corner_xplus_yminus','corner_xplus_yplus','wrist')
DINO_BASE_IDENTITY_KEY='__dino_base_identity__'


def _module_sha256(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _normalization_name(value):
    return getattr(value,'value',value)


def _validate_dinov3_config(cfg):
    from sim.act.configuration_dinov3 import (
        DINO_MODEL_ID,DINO_REVISION,DINO_WEIGHT_SHA256,
    )
    images=tuple(cfg.image_features)
    cameras=tuple(key.removeprefix('observation.images.') for key in images)
    if cameras!=DINO_CAMERA_ORDER:
        raise ValueError('DINOv3 requires the serialized five-camera order')
    if any(tuple(cfg.input_features[key].shape)!=(3,320,320) for key in images):
        raise ValueError('DINOv3 requires five 320x320 RGB camera features')
    if (tuple(cfg.input_features['observation.state'].shape)!=(36,)
            or tuple(cfg.input_features['observation.environment_state'].shape)!=(1,)
            or tuple(cfg.output_features['action'].shape)!=(4,)):
        raise ValueError('DINOv3 requires state36, target-count1 and action4')
    if (cfg.n_obs_steps!=1 or cfg.chunk_size!=25 or cfg.n_action_steps!=5
            or cfg.temporal_ensemble_coeff is not None):
        raise ValueError('DINOv3 requires one observation, chunk25, execute5 and no temporal ensemble')
    if (cfg.type!=DINO_TYPE or cfg.vision_backbone!=DINO_BACKBONE
            or cfg.dino_model_id!=DINO_MODEL_ID or cfg.dino_revision!=DINO_REVISION
            or cfg.dino_weight_sha256!=DINO_WEIGHT_SHA256 or cfg.dino_pool_stride!=2):
        raise ValueError('DINOv3 checkpoint identity differs from the pinned contract')
    if _normalization_name(cfg.normalization_mapping.get('VISUAL'))!='IDENTITY':
        raise ValueError('DINOv3 saved processor must leave RGB values unnormalized')


def _validate_dinov3_processors(base_path,cfg):
    """Reject stale processor graphs before ACT sees any observation."""
    root=Path(base_path).resolve()
    expected_images=tuple(cfg.image_features)
    expected_map={'VISUAL':'IDENTITY','STATE':'MEAN_STD','ACTION':'MEAN_STD'}
    for filename,registry_name in (
        ('policy_preprocessor.json','normalizer_processor'),
        ('policy_postprocessor.json','unnormalizer_processor'),
    ):
        path=root/filename
        if not path.is_file(): raise FileNotFoundError(f'missing saved DINO processor: {filename}')
        saved=json.loads(path.read_text())
        matches=[step for step in saved.get('steps',[])
                 if step.get('registry_name')==registry_name]
        if len(matches)!=1: raise ValueError(f'invalid saved DINO processor: {filename}')
        settings=matches[0].get('config',{})
        if settings.get('norm_map')!=expected_map:
            raise ValueError('DINOv3 saved processor normalization differs')
        features=settings.get('features',{})
        image_features=tuple(key for key in features if key.startswith('observation.images.'))
        if filename=='policy_preprocessor.json':
            required_features=set(expected_images)|{
                'observation.state','observation.environment_state','action'}
            if set(features)!=required_features:
                raise ValueError('DINOv3 preprocessor feature set differs')
            if image_features!=expected_images:
                raise ValueError('DINOv3 processor camera order differs from serialized ACT config')
            if any(tuple(features[key].get('shape',()))!=(3,320,320) for key in expected_images):
                raise ValueError('DINOv3 processor camera dimensions differ')
            if (tuple(features.get('observation.state',{}).get('shape',()))!=(36,)
                    or tuple(features.get('observation.environment_state',{}).get('shape',()))!=(1,)
                    or tuple(features.get('action',{}).get('shape',()))!=(4,)):
                raise ValueError('DINOv3 processor state/environment contract differs')
        elif (image_features or tuple(features.get('action',{}).get('shape',()))!=(4,)):
            raise ValueError('DINOv3 action unnormalizer contract differs')
        state_file=matches[0].get('state_file')
        if not isinstance(state_file,str): raise ValueError(f'missing processor state: {filename}')
        state_path=(root/state_file).resolve()
        if not state_path.is_relative_to(root) or not state_path.is_file():
            raise FileNotFoundError(f'missing or unsafe saved DINO processor state: {filename}')


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
        from lerobot.policies.act.configuration_act import ACTConfig
        from sim.act.policy import build_act_processors
        from safetensors.torch import load_file
        base_path=Path(base_path)
        saved=json.loads((base_path/'config.json').read_text())
        saved_type=saved.get('type','act')
        saved_backbone=saved.get('vision_backbone','')
        is_dino=saved_type==DINO_TYPE and saved_backbone==DINO_BACKBONE
        if saved_type==DINO_TYPE or saved_backbone==DINO_BACKBONE:
            if not is_dino:
                raise ValueError('DINOv3 checkpoint requires the exact act_dinov3 / dinov3_vits16 identity pair')
            from sim.act.configuration_dinov3 import DINOv3ACTConfig
            from sim.act.dinov3 import DINOv3ACTPolicy
            configuration=DINOv3ACTConfig.from_pretrained(str(base_path),local_files_only=True)
            configuration.device=device
            _validate_dinov3_config(configuration)
            _validate_dinov3_processors(base_path,configuration)
            frozen=DINOv3ACTPolicy.from_pretrained(
                str(base_path),config=configuration,local_files_only=True)
        else:
            configuration=ACTConfig.from_pretrained(str(base_path),local_files_only=True)
            # Complete safetensors already contain the ResNet. Do not download
            # ImageNet initialization on a new/offline GPU machine.
            configuration.pretrained_backbone_weights=None
            configuration.device=device
            frozen=ACTPolicy.from_pretrained(str(base_path),config=configuration,local_files_only=True)
        cfg=frozen.config
        image_keys=tuple(cfg.image_features)
        if is_dino:
            _validate_dinov3_config(cfg)
        else:
            if len(image_keys)!=5 or cfg.robot_state_feature.shape!=(36,) or cfg.env_state_feature.shape!=(1,):
                raise ValueError('requires five-view ResNet, state36 and target1')
            if 'resnet' not in cfg.vision_backbone or cfg.chunk_size!=25:
                raise ValueError('pre-DINO ResNet ACT step100000 required')
        obs_space=spaces.Dict({**{k:spaces.Box(0,255,tuple(cfg.input_features[k].shape),np.uint8) for k in image_keys},
            'observation.state':spaces.Box(-np.inf,np.inf,(36,),np.float32),
            'observation.environment_state':spaces.Box(-np.inf,np.inf,(1,),np.float32)})
        super().__init__(obs_space,spaces.Box(-np.inf,np.inf,(5,3),np.float32),normalize_images=False)
        self.image_keys=image_keys; self.obs_keys=tuple(obs_space.spaces)
        self.is_dinov3=is_dino
        frozen.config.device=device
        self.preprocessor,self.postprocessor=build_act_processors(frozen.config,pretrained_path=str(base_path))
        self.actor=strip_vae(copy.deepcopy(frozen.model)).requires_grad_(True)
        self.reference=strip_vae(frozen.model).requires_grad_(False).eval()
        self.critic=EncoderCritic(self.actor) if build_critic else None
        if is_dino:
            stats_paths=list(base_path.glob('*unnormalizer*.safetensors'))
            if len(stats_paths)!=1: raise ValueError('DINOv3 requires exactly one saved action unnormalizer state')
            stats_path=stats_paths[0]
        else:
            stats_path=next(base_path.glob('*unnormalizer*.safetensors'))
        stats=load_file(str(stats_path))
        means=stats['action.mean']; scales=stats['action.std']
        self.register_buffer('action_mean',means.clone())
        self.register_buffer('action_scale',scales.clone())
        self.noise=PhysicalNoise(scales[list(ACTIVE)])
        self.to(device)
        if self.is_dinov3:
            self._seal_dino_encoders()

    def _dino_encoder_modules(self):
        modules=[]
        for name,owner in (
            ('actor',self.actor),
            ('reference',self.reference),
            ('critic',self.critic.model if self.critic is not None else None),
        ):
            if owner is None:
                continue
            backbone=getattr(owner,'backbone',None)
            encoder=getattr(backbone,'encoder',None)
            if encoder is None:
                raise ValueError(f'DINOv3 {name} is missing its frozen encoder')
            modules.append((name,backbone,encoder))
        if not modules or modules[0][0]!='actor' or not any(name=='reference' for name,_,_ in modules):
            raise ValueError('DINOv3 requires actor and reference encoder copies')
        return modules

    def _seal_dino_encoders(self):
        baselines={}
        for name,backbone,encoder in self._dino_encoder_modules():
            encoder.requires_grad_(False)
            encoder.eval()
            backbone.eval()
            baselines[name]=_module_sha256(encoder)
        if len(set(baselines.values()))!=1:
            raise ValueError('DINOv3 actor, reference and critic encoders differ from the base')
        self._dino_encoder_baselines=baselines
        self._dino_base_identity=baselines['actor']

    def _assert_dino_encoders_unchanged(self):
        current={name:_module_sha256(encoder)
                 for name,_backbone,encoder in self._dino_encoder_modules()}
        if current!=self._dino_encoder_baselines:
            raise ValueError('a frozen DINOv3 encoder changed from its base identity')

    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'is_dinov3',False):
            for _name,backbone,encoder in self._dino_encoder_modules():
                encoder.eval()
                backbone.eval()
        return self

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
        # Never duplicate the frozen reference or a frozen DINO encoder.
        state={key:value for key,value in self.state_dict().items()
               if not key.startswith('reference.')
               and not (self.is_dinov3 and '.backbone.encoder.' in key)}
        if self.is_dinov3:
            identity=bytes.fromhex(self._dino_base_identity)
            state[DINO_BASE_IDENTITY_KEY]=torch.tensor(list(identity),dtype=torch.uint8)
        return state

    def load_learned(self,state:Mapping[str,torch.Tensor]):
        if self.is_dinov3:
            if not isinstance(state,Mapping):
                raise ValueError('DINOv3 learner checkpoint must be a tensor mapping')
            payload=dict(state)
            identity=payload.pop(DINO_BASE_IDENTITY_KEY,None)
            expected_identity=torch.tensor(list(bytes.fromhex(self._dino_base_identity)),dtype=torch.uint8)
            if (not isinstance(identity,torch.Tensor)
                    or identity.dtype!=torch.uint8
                    or not torch.equal(identity.detach().cpu(),expected_identity)):
                raise ValueError('learner checkpoint DINOv3 base identity differs')
            self._assert_dino_encoders_unchanged()

            current=self.state_dict()
            excluded={key for key in current
                      if key.startswith('reference.') or '.backbone.encoder.' in key}
            expected=set(current)-excluded
            missing=expected-set(payload)
            unexpected=set(payload)-expected
            if missing or unexpected:
                raise ValueError(
                    'DINOv3 learner checkpoint has incompatible learned modules '
                    f'(missing={sorted(missing)}, unexpected={sorted(unexpected)})')
            for key,value in payload.items():
                if not isinstance(value,torch.Tensor):
                    raise ValueError(f'DINOv3 learner state {key} is not a tensor')
                if value.shape!=current[key].shape or value.dtype!=current[key].dtype:
                    raise ValueError(f'DINOv3 learner state {key} has an incompatible tensor contract')
            missing_keys,unexpected_keys=self.load_state_dict(payload,strict=False)
            if unexpected_keys or set(missing_keys)!=excluded:
                raise ValueError('DINOv3 learner checkpoint has incompatible frozen modules')
            self._assert_dino_encoders_unchanged()
            return
        if self.critic is None:
            state={k:v for k,v in state.items() if not k.startswith('critic.')}
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('reference.') for k in missing):
            raise ValueError('checkpoint has incompatible learned modules')
