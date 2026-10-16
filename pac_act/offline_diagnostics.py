"""Numeric-only diagnosis. No environment, policy network or optimizer imports."""
import json
import re
from pathlib import Path

import numpy as np


def gaussian_kl(old_mean,old_log_std,new_mean,new_log_std,lengths):
    """Independent float64 closed form, not the training distribution helper."""
    a,b,c,d=[np.asarray(x,np.float64) for x in (old_mean,old_log_std,new_mean,new_log_std)]
    if a.ndim!=3 or a.shape[1:]!=(5,3) or c.shape!=a.shape:raise ValueError('requires Bx5x3')
    mask=np.arange(5)[None,:,None]<np.asarray(lengths)[:,None,None]
    term=d-b+(np.exp(2*b)+(a-c)**2)/(2*np.exp(2*d))-.5
    return float(np.where(mask,term,0.).sum((1,2)).mean())


def shift_sensitivity(physical_std,lengths,hard_limit):
    std=np.asarray(physical_std,np.float64);zero=np.zeros_like(std);logs=np.log(std)
    xy=zero.copy();xy[:,:,:2]=.01
    yaw=zero.copy();yaw[:,:,2]=.01
    coefficient=gaussian_kl(zero,logs,np.ones_like(std),logs,lengths)
    shift=float(np.sqrt(hard_limit/coefficient))
    return dict(equal_component_shift_at_hard_limit=dict(xy_mm=shift,yaw_deg=shift),
        xy_only_shift_0_01mm_KL=gaussian_kl(zero,logs,xy,logs,lengths),
        yaw_only_shift_0_01deg_KL=gaussian_kl(zero,logs,yaw,logs,lengths),
        assumption='Counterfactual fixed variance and equal physical mean shifts; not recovered historical movement.')


def _stats(values):
    a=np.asarray(values,np.float64)
    return dict(min=float(a.min()),mean=float(a.mean()),max=float(a.max()),std=float(a.std()))


def numeric_diagnosis(root,physical_scales):
    root=Path(root);groups=[];files=[]
    for path in sorted((root/'episodes').glob('preflight_*_actor/learning_blocks.jsonl')):
        records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if records:groups.append(records);files.append(str(path))
    records=[r for group in groups for r in group]
    if not records:raise ValueError('no saved numeric learning blocks')
    lengths=np.array([r['length'] for r in records]);means=np.array([r['mean'] for r in records])
    logs=np.array([r['log_std'] for r in records]);actions=np.array([r['action'] for r in records])
    mask=np.broadcast_to(np.arange(5)[None,:,None]<lengths[:,None,None],means.shape)
    std=np.exp(logs);scales=np.asarray(physical_scales,np.float64)
    if scales.shape!=(3,) or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('invalid frozen normalization scales')
    units=np.array([1000,1000,180/np.pi]);physical_std=std*scales*units
    advantages=[]
    for group in groups:
        carry=0.;result=[]
        for r in reversed(group):
            delta=r['reward']+(0. if r['terminated'] else r['next_value'])-r['value']
            carry=delta+(0. if r['terminated'] else .95*carry);result.append(carry)
        advantages.extend(reversed(result))
    env=np.array([r['environment_reward'] for r in records]);prior=np.array([r['prior_cost'] for r in records])
    raw=(actions-means)/std
    result=dict(blocks=len(records),effective_actions=int(lengths.sum()),source_numeric_files=files,
        historical_proposed_means_available=False,historical_rollback_verified=None,
        historical_exact_KL_recomputable=False,environment_steps_executed=0,optimizer_steps_executed=0,
        environment_reward=dict(nonzero_blocks=int((env!=0).sum()),total=float(env.sum()),**_stats(env)),
        critic_value=_stats([r['value'] for r in records]),logged_fields_advantage=_stats(advantages),
        recomputed_advantage=None,historical_GAE_exact_reconstruction_available=False,
        max_abs_prior_cost=float(np.max(np.abs(prior))),standardized_sampling_noise=_stats(raw[mask]),
        physical_std={key:_stats(physical_std[:,:,i][mask[:,:,i]]) for i,key in enumerate(('dx_mm','dy_mm','dyaw_deg'))},
        initial_identity_KL=gaussian_kl(means,logs,means,logs,lengths),
        sensitivity=shift_sensitivity(physical_std,lengths,.02),
        limitations=['Historical proposed means/std/optimizer-step metrics were not saved.',
            'Later zero-action terminal corrections can change collector GAE only in memory; exact historical GAE is unavailable from these logs.',
            'RGB/state observation batch and rejected weights were not saved; an exact historical optimizer replay cannot be claimed.',
            'Counterfactual sensitivity is not a measurement of the historical physical command change.'])
    acceptance=root/'acceptance.json'
    if acceptance.exists():
        report=json.loads(acceptance.read_text());match=re.search(r'exact KL ([0-9.eE+-]+)',report.get('error',''))
        result['historical_reported_exact_kl']=float(match.group(1)) if match else None
    return result
