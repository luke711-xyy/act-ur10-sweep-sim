"""Task adapter around the pinned official DP image encoder and 1D U-Net."""
from __future__ import annotations
import torch
from torch import nn
from diffusers import DDIMScheduler
from .dataset import VIEWS
from .vendor.model.vision.model_getter import get_resnet
from .vendor.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from .vendor.model.diffusion.conditional_unet1d import ConditionalUnet1D


def executable_actions(sequence):
    if sequence.ndim != 3 or sequence.shape[1:] != (28,4):
        raise ValueError("DP internal actions must be Bx28x4")
    return sequence[:,1:26]


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("invalid diffusion loss mask")
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("diffusion window has no valid actions")
    error = (prediction-target).square()
    return error.masked_fill(~mask[...,None],0).sum()/(count*prediction.shape[-1])


class DiffusionPolicy(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.normalization = normalization
        shape_meta = {"obs":{v:{"shape":[3,320,320],"type":"rgb"} for v in VIEWS}}
        shape_meta["obs"].update(state={"shape":[36]},environment_state={"shape":[1]})
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta, get_resnet("resnet18",weights=None),
            use_group_norm=True,share_rgb_model=False,imagenet_norm=True,
            crop_shape=None,resize_shape=None)
        self.unet = ConditionalUnet1D(
            input_dim=4,global_cond_dim=2*(5*512+36+1),
            diffusion_step_embed_dim=128,down_dims=[256,512,1024],
            kernel_size=5,n_groups=8,cond_predict_scale=True)
        for name in ("state","action"):
            for kind in ("scale","offset"):
                self.register_buffer(f"{name}_{kind}",
                    torch.tensor(normalization[name][kind],dtype=torch.float32))
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,beta_start=.0001,beta_end=.02,
            beta_schedule="squaredcos_cap_v2",clip_sample=True,
            set_alpha_to_one=True,steps_offset=0,prediction_type="epsilon")

    def encode(self, obs):
        # Explicit allowlist: truth fields can never flow through the encoder.
        batch = obs["state"].shape[0]
        inputs = {v:obs[v].reshape(batch*2,3,320,320) for v in VIEWS}
        state = obs["state"]*self.state_scale+self.state_offset
        inputs["state"] = state.reshape(batch*2,36)
        inputs["environment_state"] = obs["environment_state"].reshape(batch*2,1)
        return self.obs_encoder(inputs).reshape(batch,-1)

    def loss(self, batch, noise=None, timesteps=None):
        actions = batch["action"]*self.action_scale+self.action_offset
        if noise is None:
            noise = torch.randn_like(actions)
        if timesteps is None:
            timesteps = torch.randint(0,100,(actions.shape[0],),device=actions.device)
        noisy = self.noise_scheduler.add_noise(actions,noise,timesteps)
        pred = self.unet(noisy,timesteps,global_cond=self.encode(batch["obs"]))
        return masked_mse(pred,noise,batch["valid_mask"])

    @torch.no_grad()
    def predict(self, obs, generator=None):
        condition = self.encode(obs)
        device = condition.device
        trajectory = torch.randn((condition.shape[0],28,4),
                                 device=device,dtype=torch.float32,generator=generator)
        self.noise_scheduler.set_timesteps(8,device=device)
        for t in self.noise_scheduler.timesteps:
            noise = self.unet(trajectory,t,global_cond=condition)
            trajectory = self.noise_scheduler.step(
                noise,t,trajectory,eta=0).prev_sample
        trajectory = (trajectory-self.action_offset)/self.action_scale
        return executable_actions(trajectory)
