"""Physical likelihood, one-to-one successful-set imitation and terminal loss.

    Unknown entries are excluded BEFORE arithmetic, including NaN padding.
    All-fail proposal loss is graph-connected zero; Task10 must also skip actor
    optimizer.step (AdamW weight decay would otherwise move unsupervised modes).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from ..data.datasets import AlignedTargets
from .graph import rows


def tensor(value, shape, dtype, device, name):
    result = torch.as_tensor(value, device=device)
    if result.dtype != dtype or tuple(result.shape) != shape:
        raise ValueError(f'{name} must have shape {shape} and dtype {dtype}')
    return result


def masked(value, mask):
    selected = value[mask]
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite supervised loss')
    return selected.sum() / mask.sum().clamp_min(1)


def safe_target(value, mask):
    selected = value[mask]
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite supervised target')
    return torch.where(mask, value, torch.zeros_like(value))


def nll(mean, log_std, target, mask):
    target = safe_target(target, mask)
    return masked(.5*((mean-target)*(-log_std).exp()).square()+log_std, mask)


def prediction_loss(prediction, targets, brush, brush_valid):
    prediction.__post_init__()
    b = prediction.object_mean.shape[0]
    device = prediction.object_mean.device
    target_rows = rows(targets, b, AlignedTargets, 'targets')
    fields = {}
    for name in AlignedTargets.__dataclass_fields__:
        shape = (25,6,9) if name == 'object_mean' else ((25,) if name == 'valid_steps' else ((6,) if name == 'mapping_valid' else (25,6)))
        dtype = torch.int64 if name == 'state' else (torch.float32 if name in ('object_mean','visibility') else torch.bool)
        fields[name] = torch.stack([tensor(getattr(t,name),shape,dtype,device,name) for t in target_rows])
    brush = torch.as_tensor(brush, device=device)
    brush_valid = torch.as_tensor(brush_valid, device=device)
    if b == 1 and brush.shape == (25,9) and brush_valid.shape == (25,):
        brush, brush_valid = brush[None], brush_valid[None]
    brush = tensor(brush,(b,25,9),torch.float32,device,'brush')
    brush_valid = tensor(brush_valid,(b,25),torch.bool,device,'brush_valid')
    step = fields['valid_steps'] & prediction.valid.bool()
    base = fields['object_valid'] & fields['mapping_valid'][:,None] & step[:,:,None]
    means, truth = prediction.object_mean, fields['object_mean']
    pmask = base[...,None].expand(-1,-1,-1,2)
    vmask = (base & fields['velocity_valid'])[...,None].expand(-1,-1,-1,2)
    loss = nll(means[...,:2],prediction.object_log_std[...,:2],truth[...,:2],pmask)
    loss = loss + nll(means[...,2:4],prediction.object_log_std[...,2:4],truth[...,2:4],vmask)
    extent = safe_target(truth[...,4:6],pmask)
    loss = loss + masked(F.smooth_l1_loss(means[...,4:6],extent,reduction='none'),pmask)
    ymask = base & fields['rectangle'] & fields['yaw_valid']
    yaw = safe_target(truth[...,6:8],ymask[...,None].expand(-1,-1,-1,2))
    yaw = F.normalize(yaw, dim=-1)
    loss = loss + masked(1-(means[...,6:8]*yaw).sum(-1),ymask)
    amask = base & fields['rectangle'] & fields['angular_valid']
    angular = safe_target(truth[...,8],amask)
    loss = loss + masked(F.smooth_l1_loss(means[...,8],angular,reduction='none'),amask)
    smask = base & fields['state_valid']
    states = torch.where(smask,fields['state'],0)
    loss = loss + masked(F.cross_entropy(prediction.state_logits.movedim(-1,1),states,reduction='none'),smask)
    loss = loss + masked(F.cross_entropy(prediction.shape_logits.movedim(-1,1),fields['rectangle'].long(),reduction='none'),base)
    vmask = base & fields['visibility_valid']
    visibility = safe_target(fields['visibility'],vmask)
    loss = loss + masked(F.binary_cross_entropy_with_logits(prediction.visibility_logits[...,0],visibility,reduction='none'),vmask)
    bmask = (step & brush_valid)[...,None].expand(-1,-1,9)
    # Brush sin/cos uses the full2pi embedding. Its Gaussian chord error is
    # periodic and distinguishes yaw+pi, unlike the object's double-angle pair.
    return loss + nll(prediction.brush,prediction.brush_log_std,brush,bmask)


def proposal_loss(programs, successful_sets, successful_mask):
    if (not isinstance(programs,torch.Tensor) or programs.dtype != torch.float32 or programs.ndim != 4
            or programs.shape[1:] != (4,5,3) or not torch.isfinite(programs).all() or (programs.abs()>1).any()):
        raise ValueError('programs must be normalized finite FP32 [B,4,5,3]')
    targets = torch.as_tensor(successful_sets,device=programs.device)
    if targets.ndim != 4 or targets.shape[0] != len(programs) or targets.shape[2:] != (5,3) or targets.dtype != torch.float32:
        raise ValueError('explicitly grouped successful_sets FP32 [B,S,5,3] required')
    mask = tensor(successful_mask,targets.shape[:2],torch.bool,programs.device,'successful_mask')
    positive = targets[mask]
    if not torch.isfinite(positive).all() or (positive.abs()>1).any():
        raise ValueError('successful programs must be normalized and finite')
    matched = []
    for row in range(len(programs)):
        target = targets[row,mask[row]]
        if not len(target):
            continue
        cost = (programs[row,:,None]-target[None]).square().mean((-1,-2))
        mode, success = linear_sum_assignment(cost.detach().cpu().numpy())
        matched.append(cost[torch.as_tensor(mode,device=cost.device),torch.as_tensor(success,device=cost.device)])
    return torch.cat(matched).mean() if matched else programs.sum()*0


def terminal_loss(score, success, final_count, count_valid):
    score.__post_init__()
    b, device = len(score.success_logit), score.success_logit.device
    success = tensor(success,(b,1),torch.float32,device,'success')
    if not torch.isfinite(success).all() or not ((success==0)|(success==1)).all():
        raise ValueError('terminal success must be known binary labels')
    count = tensor(final_count,(b,),torch.int64,device,'final_count')
    mask = tensor(count_valid,(b,),torch.bool,device,'count_valid')
    if ((count[mask]<0)|(count[mask]>6)).any():
        raise ValueError('known counts must lie in0..6')
    count = torch.where(mask,count,0)
    return F.binary_cross_entropy_with_logits(score.success_logit,success) + masked(
        F.cross_entropy(score.final_count_logits,count,reduction='none'),mask)
