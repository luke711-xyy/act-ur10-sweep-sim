"""Experiment lifecycle. Inspect never imports torch/MuJoCo or writes files."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from .config import DEFAULT
from .storage import Checkpoints,RunStore,append_json,read_rows,write_json,last_row


def seal_terminal(rows,seed):
    # An IK failure before the first substep contributes no fake action. It
    # still terminates the preceding real transition of this same episode.
    if rows and rows[-1]['seed']==seed:
        rows[-1].update(terminated=True,next_value=0.)


def build_batch(records,version):
    from .ppo import Batch,gae
    import numpy as np
    import torch
    flat=[]; advantages=[]; returns=[]
    for rows in records.values():
        adv,ret=gae([r['reward'] for r in rows],[r['value'] for r in rows],
            [r['next_value'] for r in rows],[r['terminated'] for r in rows])
        flat.extend(rows); advantages.extend(adv); returns.extend(ret)
    if not flat or any(r['policy_revision']!=version for r in flat): raise RuntimeError('empty or mixed-policy rollout')
    return Batch({k:torch.from_numpy(np.stack([r['observation'][k] for r in flat])) for k in flat[0]['observation']},
        torch.stack([r['action'] for r in flat]),torch.stack([r['mean'] for r in flat]),
        torch.stack([r['log_std'] for r in flat]),torch.tensor([r['length'] for r in flat]),
        torch.tensor(advantages,dtype=torch.float32),torch.tensor([r['value'] for r in flat]),
        torch.tensor(returns,dtype=torch.float32),torch.tensor([r['environment_reward'] for r in flat]))


def resume_counters(root,saved):
    journal=last_row(Path(root)/'action_journal.jsonl')
    sampled=max(int(saved['sampled_actions']),journal['sampled_actions'] if journal else 0)
    used=int(saved['used_actions'])
    result=dict(saved,sampled_actions=sampled,used_actions=used,discarded_actions=sampled-used,updates=int(saved['updates']))
    blocks=last_row(Path(root)/'block_journal.jsonl')
    if blocks: result['sampled_blocks']=max(result.get('sampled_blocks',0),blocks['sampled_blocks'])
    return result


def require_local_slot():
    from .safety import local_inference_available
    ready,owner=local_inference_available()
    if not ready: raise RuntimeError(f"local inference busy: pid {owner['pid']}; no existing task was stopped")


@contextmanager
def experiment_lock(root,local=False):
    import fcntl
    from contextlib import nullcontext
    from .safety import local_inference_lease
    if local: require_local_slot()
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/'experiment.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with local_inference_lease() if local else nullcontext(): yield


def inspect(bundle=None):
    import importlib.metadata
    from .identity import verify_bundle
    from .safety import local_inference_available
    versions={}
    expected={'torch':'2.11.0','lerobot':'0.6.1','mujoco':'3.14.0','stable-baselines3':'2.9.0'}
    for name in expected:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    import sys
    errors=[name+': expected '+wanted+', found '+str(versions[name]) for name,wanted in expected.items()
            if not versions[name] or versions[name].split('+')[0]!=wanted]
    if sys.version_info[:2]!=(3,12): errors.append('Python 3.12 required')
    result=dict(execution_started=False,training_started=False,config=DEFAULT.asdict(),
        versions=versions,runtime_compatible=not errors,runtime_errors=errors)
    if bundle:
        info=verify_bundle(bundle)
        result.update(bundle_identity=info['identity'],model=str(info['model']))
    result['local_inference_available'],result['local_owner']=local_inference_available()
    return result


def require_runtime():
    result=inspect()
    if not result['runtime_compatible']: raise RuntimeError('runtime mismatch: '+'; '.join(result['runtime_errors']))


def make_guard(root,profile,original=None,started_at=None):
    from .safety import Guard
    if original: return Guard(root,**original)
    started=time.time() if started_at is None else started_at; local=profile=='mac'
    return Guard(root,started_at=started,deadline=started+(2 if local else 72)*3600,
        collect_deadline=started+(2 if local else 66)*3600,free_gib=8 if local else 20,
        output_gib=5 if local else 50)


def cfg_for(bundle,target):
    from sim.config import load_config
    from .hybrid import load_trial4
    from .identity import controller_profile_id
    bundle=Path(bundle).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    profile_id=controller_profile_id(manifest)
    if profile_id=='dinov3_sync5':
        from .dinov3_control import load_sync5_profile
        load_sync5_profile(bundle)
    formal,_=load_trial4()
    cfg=formal.candidate_config(load_config(str(bundle/'evaluation_config.yaml')),target)
    # Profile selection comes only from the bundle manifest, never from an
    # ambient/default evaluation config. An omitted id preserves legacy runs.
    cfg.set_path('act.hybrid_control_profile_id',profile_id)
    if profile_id is not None:
        cfg.set_path('act.hybrid_control_profile_root',str(bundle))
    return cfg


def next_spec(store,target,role='train',arm='actor',*,start_index=0):
    from .identity import RANGES,seed_spec
    start,end=RANGES[role]
    if type(start_index) is not int or start_index<0 or start_index%6 or start+start_index+5>=end:
        raise ValueError('preflight offset must be a nonnegative six-aligned registered index')
    if start_index and role!='preflight':raise ValueError('offset is preflight-only')
    attempts=[json.loads(p.read_text()) for p in (store.root/'attempts').glob(f'{role}_*.json')]
    used=[a['seed'] for a in attempts if a['target_count']==target]
    seed=max([start+start_index+target-1-6,*used])+6
    if seed>=end: raise RuntimeError('registered layout stream exhausted')
    return dict(seed=seed,target_count=target,role=role,arm=arm)


class Collection:
    """Six synchronous time-sliced worlds, one fixed target per environment.

    Physics pauses during all policy/critic computation and optimizer updates.
    The shared Actor is never updated until every target has its valid blocks.
    """
    def __init__(self,bundle,policy,store,registry,guard,counters,on_status=None,*,preflight_start_index=0):
        self.bundle,self.policy,self.store,self.registry,self.guard=bundle,policy,store,registry,guard
        self.counters=counters; self.worlds={}; self.on_status=on_status
        self.version=0; self.commit_id=None
        self.preflight_start_index=preflight_start_index

    def count_action(self,spec):
        from .safety import StopExperiment
        if spec['role']=='train':
            self.counters['sampled_actions']+=1
            # Journal every physically executed action, independent of learning
            # checkpoints; a killed partial rollout can never lower this count.
            append_json(self.store.root/'action_journal.jsonl',dict(
                sampled_actions=self.counters['sampled_actions'],seed=spec['seed']))
            if self.counters['sampled_actions']>=DEFAULT.max_actions:
                raise StopExperiment('one million collected effective actions reached')

    def _new_world(self,target,role='train',arm='actor'):
        from .hybrid import HybridEpisode
        spec=next_spec(self.store,target,role,arm,start_index=self.preflight_start_index if role=='preflight' else 0)
        if role=='preflight' and len(list((self.store.root/'attempts').glob('preflight_*.json')))>=24:
            from .safety import StopExperiment
            raise StopExperiment('24 preflight opportunity cap reached')
        return HybridEpisode(cfg_for(self.bundle,target),self.policy,spec,self.store,self.registry,self.guard,on_action=self.count_action)

    @staticmethod
    def tensor_obs(obs):
        import torch
        return {k:torch.as_tensor(v)[None] for k,v in obs.items()}

    def block(self,world,*,deterministic=False,baseline=False,learning=True):
        import torch
        from .distribution import PrefixGaussian,prior_cost
        obs=world.observe(); batched=self.tensor_obs(obs)
        learned=world.clock.learned_active
        with torch.no_grad():
            mean=self.policy.means(batched,reference=baseline or not learned)
            log_std=self.policy.noise.log_std.detach().cpu().clone()
            sampled=PrefixGaussian(mean,self.policy.noise.log_std,
                torch.tensor([5],device=self.policy.device)).sample(deterministic or not learned or baseline)
            base=self.policy.means(batched,reference=True) if learned and learning else mean
            value=float(self.policy.values(batched).item()) if learned and learning else 0.
            physical=self.policy.physical_actions(sampled)[0].cpu().numpy()
        k,reward,done=world.step_block(physical,learned=learned)
        if not learned or k==0: return None
        with torch.no_grad():
            lengths=torch.tensor([k],device=self.policy.device)
            cost=float(prior_cost(mean,base,lengths).item())
            next_value=0. if done or not learning else float(self.policy.values(self.tensor_obs(world.observe())).item())
        record=dict(observation=obs,action=sampled[0].cpu(),mean=mean[0].cpu(),log_std=log_std,
            length=k,environment_reward=reward,prior_cost=cost,reward=reward-cost,
            value=value,next_value=next_value,terminated=done,policy_revision=self.version,actor_commit_id=self.commit_id,
            seed=world.spec['seed'],target_count=world.spec['target_count'])
        append_json(world.path/'learning_blocks.jsonl',{k:v for k,v in record.items() if k!='observation'})
        if world.spec['role']=='train':
            self.counters['sampled_blocks']=self.counters.get('sampled_blocks',0)+1
            append_json(self.store.root/'block_journal.jsonl',dict(sampled_blocks=self.counters['sampled_blocks'],
                sampled_actions=self.counters['sampled_actions']))
        return record

    def collect(self,targets=None,role='train'):
        from .ppo import Batch,gae
        import numpy as np
        import torch
        targets=targets or {i:64 for i in range(1,7)}
        records={i:[] for i in targets}
        while any(len(records[i])<targets[i] for i in targets):
            for goal in targets:
                if len(records[goal])>=targets[goal]: continue
                self.guard.poll(collecting=role=='train')
                world=self.worlds.get(goal)
                if world is None:
                    world=self._new_world(goal,role); self.worlds[goal]=world
                if not world.done:
                    record=self.block(world)
                    if record is not None: records[goal].append(record)
                if world.done:
                    seal_terminal(records[goal],world.spec['seed'])
                    row=world.finish(); self.worlds.pop(goal)
                    if self.on_status: self.on_status(row)
        flat=[r for rows in records.values() for r in rows]
        return build_batch(records,self.version),flat

    def interrupt(self,error):
        errors=[]
        for world in list(self.worlds.values()):
            try: world.interrupt(error)
            except BaseException as failure:
                errors.append(repr(failure)); error.add_note('world cleanup: '+repr(failure))
                if hasattr(world,'close'):
                    try: world.close()
                    except BaseException as failure:
                        errors.append(repr(failure)); error.add_note('resource cleanup: '+repr(failure))
        self.worlds.clear()
        self.cleanup_errors=getattr(self,'cleanup_errors',[])+errors
        return errors
