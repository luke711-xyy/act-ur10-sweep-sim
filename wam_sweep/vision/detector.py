"""Six unordered BEV queries, two cross-attention layers and offline losses."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from .bev import BEVEncoder, BEVOutput, BEVSpec


@dataclass(frozen=True)
class DetectionOutput:
    timestamp: float
    center_xy: torch.Tensor
    velocity_xy: torch.Tensor
    half_extents_xy: torch.Tensor
    yaw_sincos: torch.Tensor  # sin(2*yaw), cos(2*yaw)
    angular_velocity: torch.Tensor
    shape_logits: torch.Tensor
    visibility_logits: torch.Tensor
    state_logits: torch.Tensor
    position_covariance: torch.Tensor
    appearance: torch.Tensor


class _CrossLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.)
        self.norm1, self.norm2 = nn.LayerNorm(128), nn.LayerNorm(128)
        self.ff = nn.Sequential(nn.Linear(128,256), nn.GELU(), nn.Linear(256,128))

    def forward(self, queries, memory, mask):
        attended = self.cross_attention(queries, memory, memory, key_padding_mask=mask, need_weights=False)[0]
        queries = self.norm1(queries + attended)
        return self.norm2(queries + self.ff(queries))


class SixQueryDetector(nn.Module):
    """One six-query model. Identity never appears among its inputs/outputs."""
    def __init__(self, spec: BEVSpec | None = None):
        super().__init__()
        self.spec = spec or BEVSpec()
        self.queries = nn.Parameter(torch.randn(6,128)*.02)
        self.position = nn.Linear(4,128)
        self.layers = nn.ModuleList([_CrossLayer(), _CrossLayer()])
        # center2, velocity2, extent2, double-angle2, angular1, shape2,
        # visibility1, state3, covariance3 =18. Appearance uses a local raster
        # patch at the predicted object, rather than the global query token.
        self.head = nn.Linear(128,18)
        self.appearance_head = nn.Linear(128,16)

    def forward(self, bev: BEVOutput) -> DetectionOutput:
        if (type(bev) is not BEVOutput or bev.spec != self.spec or bev.features.dtype != torch.float32
                or bev.features.shape[1:] != (128,*self.spec.shape)
                or bev.valid.shape != (bev.features.shape[0],1,*self.spec.shape)):
            raise ValueError('detector requires matching calibrated FP32 BEV')
        memory = bev.features.flatten(2).transpose(1,2)
        b, _, _ = memory.shape
        h, w = self.spec.shape
        ys = (torch.arange(h, device=memory.device,dtype=torch.float32)+.5)/h
        xs = (torch.arange(w, device=memory.device,dtype=torch.float32)+.5)/w
        y, x = torch.meshgrid(ys,xs,indexing='ij')
        positional = torch.stack((torch.sin(2*math.pi*x),torch.cos(2*math.pi*x),
                                  torch.sin(2*math.pi*y),torch.cos(2*math.pi*y)),-1).reshape(-1,4)
        memory = memory + self.position(positional)[None]
        mask = ~bev.valid.flatten(1)
        unseen = mask.all(1)
        # Fully masked attention otherwise produces NaNs. The dummy token
        # carries no evidence and cannot generate confident detections below.
        mask = mask.clone()
        mask[:,0] &= ~unseen
        memory = torch.where(unseen[:,None,None],torch.zeros_like(memory),memory)
        queries = self.queries[None].expand(b,-1,-1)
        for layer in self.layers:
            queries = layer(queries,memory,mask)
        raw = self.head(queries)
        lower = raw.new_tensor([self.spec.x_bounds[0],self.spec.y_bounds[0]])
        upper = raw.new_tensor([self.spec.x_bounds[1],self.spec.y_bounds[1]])
        center = lower+(upper-lower)*raw[...,:2].sigmoid()
        velocity = .5*raw[...,2:4].tanh()
        extents = .001+.119*raw[...,4:6].sigmoid()
        angle = F.normalize(raw[...,6:8]+raw.new_tensor([0.,1e-6]),dim=-1,eps=1e-8)
        angular = math.pi*raw[...,8:9].tanh()
        std = .001+.02*F.softplus(raw[...,15:17])
        rho = .95*raw[...,17].tanh()
        xx, yy, xy = std[...,0].square(),std[...,1].square(),rho*std[...,0]*std[...,1]
        covariance = torch.stack((xx,xy,xy,yy),-1).reshape(b,6,2,2)
        visibility = torch.where(unseen[:,None,None],torch.full_like(raw[...,11:12],-20.),raw[...,11:12])
        covariance = torch.where(unseen[:,None,None,None],torch.eye(2,device=raw.device)[None,None]*.05**2,covariance)
        # Nine samples of the predicted planar footprint, no segmentation/GT
        # height. Center/size remain differentiable on MPS through the sampler.
        offsets = raw.new_tensor([[-1.,-1.],[0.,-1.],[1.,-1.],[-1.,0.],[0.,0.],[1.,0.],[-1.,1.],[0.,1.],[1.,1.]])
        points = center[:,:,None,:]+extents[:,:,None,:]*offsets[None,None]
        grid = 2*(points-lower)/(upper-lower)-1
        local = BEVEncoder.sample_plane(bev.features,grid)
        local_valid = BEVEncoder.sample_plane(bev.valid.to(torch.float32),grid)
        pooled = (local*local_valid).sum(-1)/local_valid.sum(-1).clamp_min(1e-6)
        appearance = F.normalize(self.appearance_head(pooled.transpose(1,2)),dim=-1,eps=1e-8)
        appearance = torch.where((local_valid.sum(-1).transpose(1,2) > 0),appearance,torch.zeros_like(appearance))
        return DetectionOutput(bev.timestamp,center,velocity,extents,angle,angular,raw[...,9:11],visibility,
                               raw[...,12:15],covariance,appearance)


@dataclass(frozen=True)
class DetectionTargets:
    """OFFLINE supervision only; labels have no persistent/fixed true ID.

    shape:0=circle,1=rectangle; state ordering follows the label producer.
    Unknown class/visibility=-1 is masked, never treated as a negative.
    Optional velocity/angular/appearance labels are offline targets, not inputs.
    """
    center_xy: torch.Tensor
    half_extents_xy: torch.Tensor
    yaw: torch.Tensor
    shape_labels: torch.Tensor
    visibility: torch.Tensor
    state_labels: torch.Tensor
    velocity_xy: torch.Tensor | None = None
    angular_velocity: torch.Tensor | None = None
    appearance: torch.Tensor | None = None

    def __post_init__(self):
        n = len(self.center_xy)
        if n > 6:
            raise ValueError('at most six unordered targets')
        for name, shape in [('center_xy',(n,2)),('half_extents_xy',(n,2)),('yaw',(n,)),
                            ('shape_labels',(n,)),('visibility',(n,)),('state_labels',(n,)),
                            ('velocity_xy',(n,2)),('angular_velocity',(n,)),('appearance',(n,16))]:
            t = getattr(self,name)
            if t is None and name in ('velocity_xy','angular_velocity','appearance'):
                continue
            if not isinstance(t,torch.Tensor) or tuple(t.shape) != shape or not torch.isfinite(t).all():
                raise ValueError('invalid offline target '+name)
        if ((self.half_extents_xy <= 0).any() or not torch.isin(self.shape_labels,self.shape_labels.new_tensor([-1,0,1])).all()
                or not torch.isin(self.state_labels,self.state_labels.new_tensor([-1,0,1,2])).all()
                or ((self.visibility != -1) & ((self.visibility < 0) | (self.visibility > 1))).any()):
            raise ValueError('invalid target dimensions/classes/probabilities')


def hungarian_match(prediction: DetectionOutput, targets: list[DetectionTargets]):
    """Per-frame geometry/shape matching, independent of target list order/ID."""
    if len(targets) != prediction.center_xy.shape[0]:
        raise ValueError('one unordered target set per batch')
    matches = []
    for batch, target in enumerate(targets):
        device = prediction.center_xy.device
        center, size = target.center_xy.to(device),target.half_extents_xy.to(device)
        cost = (prediction.center_xy[batch,:,None]-center[None]).abs().sum(-1)/.01
        cost = cost+(prediction.half_extents_xy[batch,:,None]-size[None]).abs().sum(-1)/.02
        labels = target.shape_labels.to(device).long()
        if len(labels):
            cost = cost - prediction.shape_logits[batch].softmax(-1)[:,labels.clamp_min(0)]*(labels >= 0)[None]
        rows, cols = linear_sum_assignment(cost.detach().cpu().double().numpy())
        matches.append((torch.as_tensor(rows,device=device),torch.as_tensor(cols,device=device)))
    return matches


def perception_loss(prediction: DetectionOutput, targets: list[DetectionTargets]):
    """Hungarian geometry, Gaussian location NLL, shape/state/visibility losses.

    Rectangle angle is pi periodic; circles carry no yaw/omega supervision.
    Appearance can use local offline descriptors; online tracking only compares
    the predicted16-vector, never truth IDs. No optimizer/training is run here.
    """
    zero = prediction.center_xy.sum()*0.
    names = ('center','extent','yaw','shape','visibility','state','position_nll','velocity','angular','appearance')
    losses = dict.fromkeys(names,zero)
    matches = hungarian_match(prediction,targets)
    for b, (rows,cols) in enumerate(matches):
        target = targets[b]
        device = prediction.center_xy.device
        # Unmatched queries are unseen; matched unknown visibility stays unknown.
        visibility = prediction.visibility_logits.new_zeros(6)
        visibility[rows] = target.visibility.to(device)[cols]
        known = visibility >= 0
        if known.any():
            losses['visibility'] = losses['visibility']+F.binary_cross_entropy_with_logits(prediction.visibility_logits[b,:,0][known],visibility[known])
        if not len(rows):
            continue
        centers = target.center_xy.to(device)[cols]
        sizes = target.half_extents_xy.to(device)[cols]
        delta = prediction.center_xy[b,rows]-centers
        losses['center'] = losses['center']+F.smooth_l1_loss(prediction.center_xy[b,rows]/.01,centers/.01)
        losses['extent'] = losses['extent']+F.smooth_l1_loss(prediction.half_extents_xy[b,rows]/.02,sizes/.02)
        cov = prediction.position_covariance[b,rows]
        a,c,d = cov[:,0,0],cov[:,0,1],cov[:,1,1]
        det = (a*d-c.square()).clamp_min(1e-16)
        maha = (d*delta[:,0].square()-2*c*delta[:,0]*delta[:,1]+a*delta[:,1].square())/det
        losses['position_nll'] = losses['position_nll']+.5*(maha+det.log()+2*math.log(2*math.pi)).mean()
        shapes = target.shape_labels.to(device)[cols].long()
        valid = shapes >= 0
        if valid.any():
            losses['shape'] = losses['shape']+F.cross_entropy(prediction.shape_logits[b,rows][valid],shapes[valid])
        rectangular = shapes == 1
        if rectangular.any():
            yaw = target.yaw.to(device)[cols][rectangular]
            angle = torch.stack((torch.sin(2*yaw),torch.cos(2*yaw)),-1)
            losses['yaw'] = losses['yaw']+(1-(prediction.yaw_sincos[b,rows][rectangular]*angle).sum(-1)).mean()
        states = target.state_labels.to(device)[cols].long()
        valid = states >= 0
        if valid.any():
            losses['state'] = losses['state']+F.cross_entropy(prediction.state_logits[b,rows][valid],states[valid])
        if target.velocity_xy is not None:
            losses['velocity'] = losses['velocity']+F.smooth_l1_loss(prediction.velocity_xy[b,rows],target.velocity_xy.to(device)[cols])
        if target.angular_velocity is not None and rectangular.any():
            losses['angular'] = losses['angular']+F.smooth_l1_loss(prediction.angular_velocity[b,rows,0][rectangular],target.angular_velocity.to(device)[cols][rectangular])
        if target.appearance is not None:
            local = F.normalize(target.appearance.to(device)[cols],dim=-1,eps=1e-8)
            losses['appearance'] = losses['appearance']+(1-(prediction.appearance[b,rows]*local).sum(-1)).mean()
    losses = {name:value/len(targets) for name,value in losses.items()}
    losses['total'] = sum(value for key,value in losses.items() if key != 'position_nll')+.01*losses['position_nll']
    return losses
