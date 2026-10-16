"""SB3 PPO adaptation for prefix-masked ACT chunks and transactional updates.

Collection is deliberately outside SB3's step-level VecEnv runner: a native
task can end inside a chunk and its unused Gaussian suffix must not be learned.
"""
import copy
from dataclasses import dataclass,field
import random
import time
import traceback
import uuid

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.nn import functional as F

from .config import DEFAULT
from .distribution import PrefixGaussian


class UpdateRejected(RuntimeError):
    pass


def capture_rng(device='cpu'):
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if str(device).startswith('cuda'): state['cuda']=torch.cuda.get_rng_state_all()
    if str(device).startswith('mps'): state['mps']=torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if 'cuda' in state: torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state: torch.mps.set_rng_state(state['mps'])


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def gae(rewards,values,next_values,terminated,lam=.95):
    rewards,values,next_values=np.asarray(rewards),np.asarray(values),np.asarray(next_values)
    done=np.asarray(terminated,dtype=bool)
    advantage=np.empty_like(rewards,dtype=np.float64); carry=0.
    for i in range(len(rewards)-1,-1,-1):
        delta=rewards[i]+(0. if done[i] else next_values[i])-values[i]
        carry=delta+(0. if done[i] else lam*carry)
        advantage[i]=carry
    return advantage,advantage+values


@dataclass
class Batch:
    observations: dict
    actions: torch.Tensor
    old_means: torch.Tensor
    old_log_std: torch.Tensor
    lengths: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    env_rewards: torch.Tensor
    consumed: bool = field(default=False,init=False)

    def __len__(self): return len(self.lengths)

    def take(self,index,device):
        names=('actions','old_means','old_log_std','lengths','advantages','old_values','returns','env_rewards')
        index=torch.as_tensor(index,dtype=torch.long)
        data={name:getattr(self,name)[index].to(device) for name in names}
        data['observations']={k:v[index].to(device) for k,v in self.observations.items()}
        return data


class ChunkPPO(PPO):
    """Keep SB3's public algorithm identity without its unmasked rollout buffer.

    Never use inherited learn/collect_rollouts/save: the experiment lifecycle
    owns chunk collection, atomic checkpoints and resource limits explicitly.
    """
    def __init__(self,policy,config=DEFAULT,*,poll=None,audit_sink=None):
        super().__init__(type(policy),None,learning_rate=config.actor_lr,n_steps=64,
            batch_size=config.batch,n_epochs=config.epochs,gamma=1.,gae_lambda=config.gae_lambda,
            clip_range=config.clip,ent_coef=config.entropy_coef,vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,target_kl=config.target_kl,
            device=str(policy.device),_init_setup_model=False)
        self.policy=policy; self.config=config
        self.poll=poll or (lambda:None)
        self.optimizer=torch.optim.Adam([
            {'params':list(policy.actor.parameters())+list(policy.noise.parameters()),'lr':config.actor_lr,'role':'actor'},
            {'params':list(policy.critic.parameters()),'lr':config.critic_lr,'role':'critic'}],eps=1e-5)
        self._n_updates=0; self.last_audit=None;self.audit_sink=audit_sink
        self.last_distribution_audit=None
        self.last_accepted_audit=None;self._rejected_exception=None
        self._audit_step_calls=0;self._audit_step_returns=0
        self._step_audits=[]

    def _update_learning_rate(self,optimizers):
        # SB3's scalar schedule would erase the independently approved LR.
        for group in self.optimizer.param_groups:
            group['lr']=self.config.actor_lr if group['role']=='actor' else self.config.critic_lr

    def snapshot(self):
        return {'model':{k:v.detach().cpu().clone() for k,v in self.policy.learned_state().items()},
            'optimizer':cpu_tree(self.optimizer.state_dict()),
            'rng':capture_rng(self.policy.device),'updates':self._n_updates}

    def restore(self,state):
        self.policy.load_learned(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self._n_updates=int(state['updates']); restore_rng(state['rng'])
        self._update_learning_rate([self.optimizer])

    def objective(self,batch,index,normalized_advantage=None):
        data=batch.take(index,self.policy.device)
        means=self.policy.means(data['observations'])
        current=PrefixGaussian(means,self.policy.noise.log_std,data['lengths'])
        old=PrefixGaussian(data['old_means'],data['old_log_std'],data['lengths'])
        ratio=(current.log_prob(data['actions'])-old.log_prob(data['actions'])).exp()
        adv=data['advantages'] if normalized_advantage is None else normalized_advantage.to(self.policy.device)
        actor=-torch.minimum(adv*ratio,adv*ratio.clamp(1-self.config.clip,1+self.config.clip)).mean()
        values=self.policy.values(data['observations'])
        value=F.mse_loss(values,data['returns'])
        entropy=current.entropy().mean()
        kl=old.kl(means,self.policy.noise.log_std).mean()
        loss=actor+self.config.value_coef*value-self.config.entropy_coef*entropy+self.config.kl_coef*kl
        return dict(loss=loss,actor_loss=actor,value_loss=value,entropy=entropy,kl=kl,ratio=ratio)

    def update_minibatch(self,batch,index):
        step_audit=dict(status='backward',learning_rates=[dict(role=g['role'],lr=g['lr']) for g in self.optimizer.param_groups])
        self._step_audits.append(step_audit)
        # Normalize ONCE over logical32, not separately for each microbatch.
        adv=batch.advantages[index].float()
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        self.optimizer.zero_grad(set_to_none=True); result={}
        for start in range(0,len(index),self.config.microbatch):
            self.poll()
            end=min(start+self.config.microbatch,len(index)); weight=(end-start)/len(index)
            parts=self.objective(batch,index[start:end],adv[start:end])
            if not all(torch.isfinite(v).all() for v in parts.values()):
                raise UpdateRejected('nonfinite PPO objective')
            (parts['loss']*weight).backward()
            for key in ('loss','actor_loss','value_loss','entropy','kl'):
                result[key]=result.get(key,0.)+float(parts[key].detach())*weight
        parameters=[p for g in self.optimizer.param_groups for p in g['params']]
        self.poll()
        norm=torch.nn.utils.clip_grad_norm_(parameters,self.config.max_grad_norm,error_if_nonfinite=True)
        step_audit.update(pre_clip_gradient_norm=float(norm),
            post_clip_gradient_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in parameters if p.grad is not None))),
            metrics_before_optimizer=dict(result),status='optimizer_entered')
        self._audit_step_calls+=1
        self.optimizer.step();self._audit_step_returns+=1
        step_audit['status']='optimizer_returned'
        self.policy.noise.project_()
        if any(not torch.isfinite(p).all() for p in parameters): raise UpdateRejected('nonfinite PPO parameters')
        result['gradient_norm']=float(norm)
        step_audit['status']='complete'
        return result

    @torch.no_grad()
    def exact_kl(self,batch):
        self.last_distribution_audit=None
        total=0.;old_means=[];new_means=[];old_logs=[]
        for start in range(0,len(batch),self.config.microbatch):
            self.poll()
            data=batch.take(torch.arange(start,min(start+self.config.microbatch,len(batch))),self.policy.device)
            old=PrefixGaussian(data['old_means'],data['old_log_std'],data['lengths'])
            proposed=self.policy.means(data['observations'])
            value=old.kl(proposed,self.policy.noise.log_std)
            old_means.append(data['old_means'].cpu());new_means.append(proposed.cpu())
            old_logs.append(old.log_std.cpu())
            finite=bool(torch.isfinite(value).all())
            if finite:total+=float(value.sum())
            # Capture the offending microbatch BEFORE rejecting it. This is
            # partial evidence, not a successful whole-rollout KL pass.
            before=torch.cat(old_means);after=torch.cat(new_means);count=len(after)
            lengths=batch.lengths.cpu()[:count]
            mask=torch.arange(5)[None,:]<lengths[:,None]
            scaled=(after-before)*self.policy.noise.scales.detach().cpu()
            units=(1000.,1000.,180./np.pi)
            shifts={key:float((scaled[:,:,i][mask].square().mean()).sqrt())*units[i]
                for i,key in enumerate(('dx_mm','dy_mm','dyaw_deg'))}
            self.last_distribution_audit=dict(exact_kl=total/count if finite else None,
                complete=finite and count==len(batch),blocks_evaluated=count,expected_blocks=len(batch),nonfinite_KL=not finite,
                kl_definition='KL(old || proposed), sum executed steps and XY/yaw, mean blocks',
                active_scalar_dimensions=int(lengths.sum())*3,
                physical_mean_shift_rms=shifts,lengths=lengths.tolist(),
                old_means=before.tolist(),proposed_means=after.tolist(),old_log_std=torch.cat(old_logs).tolist(),
                proposed_log_std=self.policy.noise.log_std.detach().cpu().tolist(),
                sampled_actions=batch.actions.cpu()[:count].tolist(),physical_scales=self.policy.noise.scales.detach().cpu().tolist())
            if not finite:raise UpdateRejected('nonfinite full rollout KL')
        return total/len(batch)

    def train(self,batch):
        if batch.consumed: raise RuntimeError('on-policy rollout already consumed; never replay a PPO batch')
        if len(batch)<32 or len(batch)%32: raise ValueError('rollout must contain whole logical minibatches32')
        from .update_audit import state_hashes,json_safe
        batch.consumed=True; saved=self.snapshot(); records=[];before_hash=state_hashes(saved)
        attempt_id=uuid.uuid4().hex;completed_epochs=0;before_dist=None
        self._rejected_exception=None
        self._audit_step_calls=0;self._audit_step_returns=0
        self._step_audits=[]
        self.last_distribution_audit=None;stage='initial_distribution'
        try:
            self.poll()
            self.exact_kl(batch);before_dist=copy.deepcopy(self.last_distribution_audit)
            for epoch in range(self.config.epochs):
                indices=torch.randperm(len(batch))
                for start in range(0,len(batch),self.config.batch):
                    stage=f'epoch_{epoch+1}_minibatch_{start//self.config.batch+1}'
                    records.append(self.update_minibatch(batch,indices[start:start+self.config.batch]))
                completed_epochs=epoch+1;stage=f'epoch_{epoch+1}_whole_rollout_KL'
                kl=self.exact_kl(batch)
                if kl>self.config.hard_kl: raise UpdateRejected(f'full rollout exact KL {kl:.8g} exceeds .02')
                if kl>self.config.early_kl: break
            self._n_updates+=1
            self.last_audit=dict(accepted=True,update=self._n_updates,epochs=epoch+1,exact_kl=kl,
                **{k:float(np.mean([r[k] for r in records if k in r]))
                   for k in records[0]} if records else {})
            self.last_accepted_audit=copy.deepcopy(self.last_audit)
            return self.last_audit
        except BaseException as error:
            self._rejected_exception=error
            audit_errors=[];proposed_hash=None;after_hash=None;restored=False
            self.last_audit=dict(accepted=False,attempt_id=attempt_id,audit_errors=audit_errors)
            def diagnostic(label,operation,default=None):
                try:return operation()
                except BaseException as audit_error:
                    audit_errors.append(label+': '+repr(audit_error));return default
            try:
                raw_traceback=diagnostic('traceback formatting',traceback.format_exc)
                proposed_hash=diagnostic('proposed state hash',lambda:state_hashes(dict(model=self.policy.learned_state(),
                    optimizer=self.optimizer.state_dict(),rng=capture_rng(self.policy.device),updates=self._n_updates)))
            finally:
                # No traceback/hash/diagnostic failure may skip restoration.
                try:
                    self.restore(saved);restored=True
                except BaseException as restoration:
                    error.add_note(f'rollback could not be completed: {restoration!r}; last verified disk checkpoint remains authoritative')
            if restored:
                after_hash=diagnostic('restored state hash',lambda:state_hashes(dict(model=self.policy.learned_state(),
                    optimizer=self.optimizer.state_dict(),rng=capture_rng(self.policy.device),updates=self._n_updates)))
            verified=before_hash==after_hash if after_hash is not None else None
            if restored and verified is False:error.add_note('rollback hash verification failed')
            # A minimal record remains usable even if richer diagnostics fail.
            self.last_audit.update(rolled_back=restored,rollback_verified=verified,
                error_type=type(error).__name__,stage=stage,
                error=diagnostic('exception representation',lambda:repr(error)),traceback=raw_traceback)
            rich=diagnostic('rejection record construction',lambda:json_safe(dict(accepted=False,attempt_id=attempt_id,time=time.time(),
                provenance='live_instrumented_update',rolled_back=restored,rollback_verified=verified,
                error=repr(error),error_type=type(error).__name__,traceback=raw_traceback,stage=stage,
                updates=self._n_updates,optimizer_steps=self._audit_step_returns,
                optimizer_step_calls=self._audit_step_calls,
                optimizer_step_may_have_partially_applied=self._audit_step_calls!=self._audit_step_returns,
                completed_epochs=completed_epochs,
                config=self.config.asdict(),metrics_before_optimizer_steps=records,
                optimizer_step_audits=self._step_audits,
                distribution_before=before_dist,
                distribution_after=self.last_distribution_audit if stage.endswith('whole_rollout_KL') else None,
                state_hashes=dict(before=before_hash,proposed=proposed_hash,restored=after_hash),
                batch=dict(blocks=len(batch),effective_actions=int(batch.lengths.sum()),
                    nonfinite_returns=int((~torch.isfinite(batch.returns)).sum()),
                    env_reward_nonzero_blocks=int((batch.env_rewards!=0).sum())),audit_errors=audit_errors)))
            if rich is not None:self.last_audit=rich
            if self.audit_sink:
                try:self.last_audit.update(self.audit_sink(self.last_audit) or {})
                except BaseException as audit_error:
                    self.last_audit['audit_errors'].append('audit persistence: '+repr(audit_error))
            for failure in self.last_audit['audit_errors']:error.add_note('rejected-update diagnostic: '+failure)
            from .safety import StopExperiment
            if isinstance(error,(KeyboardInterrupt,SystemExit,UpdateRejected,StopExperiment)): raise
            raise UpdateRejected(f'PPO update rolled back: {type(error).__name__}: {error}') from error

    def learn(self,*args,**kwargs):
        raise RuntimeError('use train_pac_act lifecycle; SB3 step-level learn is intentionally disabled')


def audited_trainer(policy,poll,root,identity,config=DEFAULT):
    from .update_audit import persist_rejection
    return ChunkPPO(policy,config,poll=poll,
        audit_sink=lambda audit:persist_rejection(root,audit,identity))
