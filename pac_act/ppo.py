"""SB3 PPO adaptation for prefix-masked ACT chunks and transactional updates.

Collection is deliberately outside SB3's step-level VecEnv runner: a native
task can end inside a chunk and its unused Gaussian suffix must not be learned.
"""
import copy
from dataclasses import dataclass,field
import random

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
    def __init__(self,policy,config=DEFAULT):
        super().__init__(type(policy),None,learning_rate=config.actor_lr,n_steps=64,
            batch_size=config.batch,n_epochs=config.epochs,gamma=1.,gae_lambda=config.gae_lambda,
            clip_range=config.clip,ent_coef=config.entropy_coef,vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,target_kl=config.target_kl,
            device=str(policy.device),_init_setup_model=False)
        self.policy=policy; self.config=config
        self.optimizer=torch.optim.Adam([
            {'params':list(policy.actor.parameters())+list(policy.noise.parameters()),'lr':config.actor_lr,'role':'actor'},
            {'params':list(policy.critic.parameters()),'lr':config.critic_lr,'role':'critic'}],eps=1e-5)
        self._n_updates=0; self.last_audit=None

    def _update_learning_rate(self,optimizers):
        # SB3's scalar schedule would erase the independently approved LR.
        for group in self.optimizer.param_groups:
            group['lr']=self.config.actor_lr if group['role']=='actor' else self.config.critic_lr

    def snapshot(self):
        return {'model':{k:v.detach().cpu().clone() for k,v in self.policy.learned_state().items()},
            'optimizer':copy.deepcopy(self.optimizer.state_dict()),
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
        # Normalize ONCE over logical32, not separately for each microbatch.
        adv=batch.advantages[index].float()
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        self.optimizer.zero_grad(set_to_none=True); result={}
        for start in range(0,len(index),self.config.microbatch):
            end=min(start+self.config.microbatch,len(index)); weight=(end-start)/len(index)
            parts=self.objective(batch,index[start:end],adv[start:end])
            if not all(torch.isfinite(v).all() for v in parts.values()):
                raise UpdateRejected('nonfinite PPO objective')
            (parts['loss']*weight).backward()
            for key in ('loss','actor_loss','value_loss','entropy','kl'):
                result[key]=result.get(key,0.)+float(parts[key].detach())*weight
        parameters=[p for g in self.optimizer.param_groups for p in g['params']]
        norm=torch.nn.utils.clip_grad_norm_(parameters,self.config.max_grad_norm,error_if_nonfinite=True)
        self.optimizer.step(); self.policy.noise.project_()
        if any(not torch.isfinite(p).all() for p in parameters): raise UpdateRejected('nonfinite PPO parameters')
        result['gradient_norm']=float(norm)
        return result

    @torch.no_grad()
    def exact_kl(self,batch):
        total=0.
        for start in range(0,len(batch),self.config.microbatch):
            data=batch.take(torch.arange(start,min(start+self.config.microbatch,len(batch))),self.policy.device)
            old=PrefixGaussian(data['old_means'],data['old_log_std'],data['lengths'])
            value=old.kl(self.policy.means(data['observations']),self.policy.noise.log_std)
            if not torch.isfinite(value).all(): raise UpdateRejected('nonfinite full rollout KL')
            total+=float(value.sum())
        return total/len(batch)

    def train(self,batch):
        if batch.consumed: raise RuntimeError('on-policy rollout already consumed; never replay a PPO batch')
        if len(batch)<32 or len(batch)%32: raise ValueError('rollout must contain whole logical minibatches32')
        batch.consumed=True; saved=self.snapshot(); records=[]
        try:
            for epoch in range(self.config.epochs):
                indices=torch.randperm(len(batch))
                for start in range(0,len(batch),self.config.batch):
                    records.append(self.update_minibatch(batch,indices[start:start+self.config.batch]))
                kl=self.exact_kl(batch)
                if kl>self.config.hard_kl: raise UpdateRejected(f'full rollout exact KL {kl:.8g} exceeds .02')
                if kl>self.config.early_kl: break
            self._n_updates+=1
            self.last_audit=dict(accepted=True,update=self._n_updates,epochs=epoch+1,exact_kl=kl,
                **{k:float(np.mean([r[k] for r in records if k in r]))
                   for k in records[0]} if records else {})
            return self.last_audit
        except BaseException as error:
            self.restore(saved)
            self.last_audit=dict(accepted=False,rolled_back=True,error=repr(error),updates=self._n_updates)
            if isinstance(error,(KeyboardInterrupt,SystemExit,UpdateRejected)): raise
            raise UpdateRejected(f'PPO update rolled back: {type(error).__name__}: {error}') from error

    def learn(self,*args,**kwargs):
        raise RuntimeError('use train_pac_act lifecycle; SB3 step-level learn is intentionally disabled')
