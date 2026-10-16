"""Raw normalized Gaussian actions, with an executed-prefix likelihood."""
import math
import torch
from torch import nn


def prefix_mask(lengths, steps=5):
    lengths=torch.as_tensor(lengths)
    if lengths.ndim!=1 or not torch.all((lengths>=1)&(lengths<=steps)) or lengths.is_floating_point():
        raise ValueError('learning blocks need integer executed prefixes in 1..5')
    return (torch.arange(steps,device=lengths.device)[None,:]<lengths[:,None])[...,None]


class PrefixGaussian:
    def __init__(self,mean,log_std,lengths):
        if mean.ndim!=3 or mean.shape[1:]!=(5,3): raise ValueError('Gaussian must be Bx5x3, no Z')
        self.mean=mean
        self.log_std=log_std.expand_as(mean)
        self.mask=prefix_mask(lengths).to(mean.device)
        if self.mask.shape[0]!=mean.shape[0]: raise ValueError('prefix batch differs')
        self.dist=torch.distributions.Normal(mean,self.log_std.exp(),validate_args=True)

    def sample(self,deterministic=False):
        return self.mean.clone() if deterministic else self.dist.sample()

    def log_prob(self,actions):
        if actions.shape!=self.mean.shape: raise ValueError('raw actions must be Bx5x3')
        return torch.where(self.mask,self.dist.log_prob(actions),0.).sum((1,2))

    def entropy(self):
        return torch.where(self.mask,self.dist.entropy(),0.).sum((1,2))

    def kl(self,new_mean,new_log_std):
        # KL(old || new), differentiable in both new mean and learned std.
        new_std=new_log_std.exp()
        log_ratio=self.log_std-new_log_std
        # expm1 keeps identical policies exactly zero, avoiding cancellation
        # between separately rounded exp(log_std*2) and exp(log_std)**2.
        parts=.5*(torch.expm1(2*log_ratio)-2*log_ratio)+.5*((self.mean-new_mean)/new_std).square()
        return torch.where(self.mask,parts,0.).sum((1,2))


class PhysicalNoise(nn.Module):
    def __init__(self,action_scales):
        super().__init__()
        scales=torch.as_tensor(action_scales,dtype=torch.float32)
        if scales.shape!=(3,) or not torch.isfinite(scales).all() or torch.any(scales<=0):
            raise ValueError('positive frozen dx,dy,dyaw normalization scales required')
        self.register_buffer('scales',scales)
        self.register_buffer('lower',torch.tensor([1e-5,1e-5,math.radians(.01)]))
        self.register_buffer('upper',torch.tensor([3e-4,3e-4,math.radians(.30)]))
        initial=torch.tensor([1e-4,1e-4,math.radians(.10)])
        self.log_std=nn.Parameter((initial/scales).log().expand(5,3).clone())

    def physical_std(self):
        return self.log_std.exp()*self.scales

    @torch.no_grad()
    def project_(self):
        self.log_std.clamp_((self.lower/self.scales).log(),(self.upper/self.scales).log())


def prior_cost(mean,reference,lengths,coefficient=2.):
    mask=prefix_mask(lengths).to(mean.device)
    if mean.shape!=reference.shape or mean.shape[1:]!=(5,3): raise ValueError('prior must be Bx5x3')
    return coefficient*torch.where(mask,(mean-reference).square(),0.).sum((1,2))/(lengths.to(mean.device)*3)
