"""Actual frozen Hybrid bridge. All changes are process-local, never sim edits."""
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

from .bridge import (RecentRows, closure_value, contact_mask, instrument_physics,
                     streaming_native)
from .config import DEFAULT
from .relative_action import RelativeActionGenerator,continuous_probe_action
from .features import FeatureMailbox,observed_scheduler
from .observations import actor_observation,critic_observation
from .rewards import ContactReward,classify_failure
from .safety import BudgetStop
from .storage import TraceWriter,sha256,write_experience,write_json

PROJECT=Path(__file__).resolve().parents[1]
EXPORTS=Path('/Users/huazhi_luke/Documents/Codex/2026-09-08/zhe/exports')
HYBRID=EXPORTS/'batch_hybrid_trial4_100k_120_paired_20261014'


class LayoutRegistry:
    def __init__(self,old_plan,prior_consumed=()):
        self.old_seeds={p['seed'] for p in old_plan}
        self.old_hashes={p['layout_fingerprint'] for p in old_plan}
        self.prior_seeds={p['seed'] for p in prior_consumed}
        self.prior_hashes={p['layout_fingerprint'] for p in prior_consumed}
        self.used={}
        self.seeds={}

    def register(self,seed,fingerprint,role):
        if seed in self.prior_seeds or fingerprint in self.prior_hashes:
            raise ValueError('previously consumed layout cannot enter a replacement pilot')
        if seed in self.old_seeds or fingerprint in self.old_hashes:
            raise ValueError('old120 layout cannot enter residual training or tuning')
        if seed in self.seeds and self.seeds[seed]['fingerprint']!=fingerprint:
            raise ValueError('seed changed initial-pose fingerprint before planning/control')
        prior=self.used.get(fingerprint,[])
        pairs=({'validation_base','validation_final'},{'preflight_native','preflight_zero'})
        if prior and not (len(prior)==1 and prior[0][0]==seed
                          and {prior[0][1],role} in pairs):
            raise ValueError('layout reuse outside the two validation arms')
        self.used.setdefault(fingerprint,[]).append((seed,role))
        self.seeds.setdefault(seed,dict(fingerprint=fingerprint,roles=[]))['roles'].append(role)


def load_formal():
    name='_residual_frozen_hybrid_batch'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PROJECT/'scripts/evaluate_hybrid_paired_batch.py')
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        try: spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    return sys.modules[name]


def identity():
    from .baseline import pre_dino_sources
    # Equivalent historical input/model guards, with the EFFECTIVE pinned sim
    # bytes verified instead of demanding a destructive rollback of current DINO.
    # The old formal verifier/script/seal/fingerprints are never modified.
    metadata=json.loads((HYBRID/'metadata.json').read_text())
    snapshot=pre_dino_sources(PROJECT,metadata['source_fingerprints'])
    formal=load_formal()
    formal.validate_metadata_seal(HYBRID)
    if metadata['kind']!=formal.LABEL or metadata['profile']!=formal.PROFILE:
        raise RuntimeError('not the approved trial4 formal profile')
    for path,expected in metadata['frozen_sha256'].items():
        if sha256(Path(path))!=expected: raise RuntimeError('frozen input changed: '+path)
    model=metadata['checkpoint']
    for name,expected in model['checkpoint_files'].items():
        stat=(Path(model['path'])/name).stat()
        if [stat.st_size,stat.st_mtime_ns]!=expected:
            raise RuntimeError('frozen checkpoint changed: '+name)
    return dict(metadata_sha256=sha256(HYBRID/'metadata.json'),
        checkpoint=metadata['checkpoint'],source_fingerprints=metadata['source_fingerprints'],
        baseline=snapshot.manifest(),profile=formal.PROFILE,
        formal_script_sha256=sha256(Path(formal.__file__)))


def audit_frozen_sources():
    """Read-only audit; a green display also requires all original input guards."""
    return identity()['baseline']


@contextmanager
def exclusive_inference():
    """Never wait behind or stop the user's manual task."""
    import fcntl
    path=PROJECT/'runs/.act_inference.lock'
    with path.open('a+') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('inference is already in use; no user task was stopped') from error
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)


class PolicyCache:
    def __init__(self):
        self.bundle=None; self.mailbox=None

    def build(self,native,cfg,pretrained_path):
        if self.bundle is None:
            self.bundle=native(cfg,pretrained_path=pretrained_path)
            policy,policy_cfg=self.bundle
            if str(policy_cfg.device)!='mps':
                raise RuntimeError('approved local pilot requires frozen ACT on MPS')
            policy.requires_grad_(False); policy.eval()
            def capture(_module,inputs):
                self.mailbox.publish(inputs[0].detach().cpu().numpy())
            self.hook=policy.model.action_head.register_forward_pre_hook(capture)
        return self.bundle


@dataclass
class Rollout:
    row: dict
    episode: object
    configuration: object
    trace_path: Path


class Collector:
    def __init__(self,env,cfg,spec,mode,learner,guard,destination,mailbox):
        from sim.act.interface import ACTObservationBuilder
        self.env,self.cfg,self.spec,self.mode=env,cfg,spec,mode
        self.episode_started_at=float(env.time)  # reset has already settled physics
        self.learner,self.guard,self.mailbox=learner,guard,mailbox
        self.builder=ACTObservationBuilder(cfg,environment_state_dim=1)
        self.reward=ContactReward(spec['target_count']); self.context=None
        self.governor=None; self.scheduler=None
        self.control=RelativeActionGenerator([[cfg.workspace.x_min,cfg.workspace.x_max],
                                      [cfg.workspace.y_min,cfg.workspace.y_max]])
        from .shadow import ZeroResidualShadow
        self.shadow=(ZeroResidualShadow(self.control.bounds)
                     if spec.get('phase')=='preflight_shadow' else None)
        self.ACT_control_steps=0
        self.hidden=None; self.previous_action=np.zeros(3)
        self.raw_action=np.zeros(3); self.nominal_velocity=np.zeros(3)
        self.actor_rows=[]; self.critic_rows=[]; self.actions=[]; self.rewards=[]; self.masks=[]
        self.open=False; self.control_steps=0; self.active=False
        self.steps_in_frame=0; self.physical_error=None; self.current_phase='Astar_prefix'
        self.trace=TraceWriter(destination/'physical_trace.jsonl.gz')
        self.command_digest=hashlib.sha256(); self.z_digest=hashlib.sha256()
        self.peak=0.; self.force_sum=0.; self.force_samples=0; self.low_force=0
        self.z_error_sum=0.; self.z_error_max=0.; self.active_frames=0
        self.probe_modes=set(); self.limit_errors=[]; self.first_physics_time=None
        self.accepted_velocity=np.zeros(3)
        self.accepted_offset=np.zeros(3)
        self.accepted_filter=np.zeros(3)
        self.residual_norm_sum=0.; self.residual_samples=0; self.nonzero_residual_samples=0
        self.probe_stage=None; self.previous_command=None; self.command_before=None
        self.command_started_at=None; self.tcp_before=None; self.tcp_yaw_before=None
        self.act_displacement=np.zeros(3)

    def is_active(self):
        return (self.governor is not None and self.governor.started_at is not None
                and self.env.time-self.governor.started_at>=2.-1e-9)

    def observations(self):
        view=self.mailbox.current(self.env.time)
        robot=self.builder.observe(self.env,contact_latched=bool(self.context and self.context['ready']),
                                   include_images=False)['observation.state']
        actor=actor_observation(robot_state=robot,feature=view['feature'],
            target_count=self.spec['target_count'],elapsed=float(self.env.time)-self.episode_started_at,
            feature_age=view['age'],chunk_index=view['index'],feature_available=view['available'],
            active=self.is_active(),nominal_velocity=self.nominal_velocity,
            previous_action=self.previous_action,residual_velocity=self.control.velocity,
            residual_offset=self.control.offset,act_displacement=self.act_displacement,
            filtered_action=self.control.filtered_action)
        c=self.context or {}; timer=c.get('collection_timer'); loop=c.get('admittance')
        limiter=c.get('z_limiter'); gate=c.get('force_gate')
        first=getattr(timer,'first_entry_time',None)
        queue=np.zeros((25,4),np.float32)
        chunk=getattr(self.scheduler,'observation_chunk',None)
        if chunk is not None: queue[:len(chunk.actions)]=chunk.actions
        critic=critic_observation(actor,qpos=self.env.data.qpos,qvel=self.env.data.qvel,
            collected=self.env.collected_mask(),seen=self.reward.seen,dwell=self.reward.durations,
            collection_clock=[float(first is not None),0. if first is None else self.env.time-first,
                              getattr(timer,'peak_collected',0)],
            controller_state=[getattr(loop,'delta_z',0.),getattr(loop,'delta_z_rate',0.),
                getattr(limiter,'last_z',0.),c.get('z_nominal',0.),
                getattr(gate,'count',0),float(chunk.issued_at) if chunk is not None else 0.,
                float(chunk.valid_from) if chunk is not None else 0.,
                float(chunk.valid_until) if chunk is not None else 0.],scheduled_actions=queue)
        return actor,critic

    def begin(self,nominal=None):
        if nominal is not None: self.nominal_velocity=np.asarray(nominal,dtype=float)
        self.guard.poll()
        self.active=self.is_active()
        if (self.active and self.mode in {'train','evaluate','probe'}
                and self.guard.residual_steps>=DEFAULT.max_residual_steps):
            raise BudgetStop('stop_step_budget')
        actor,critic=self.observations()
        self.actor_rows.append(actor); self.critic_rows.append(critic)
        action=np.zeros(3)
        if self.learner is not None:
            action,self.hidden=self.learner.act(actor,self.hidden,deterministic=self.mode!='train')
        if not self.active: action=np.zeros(3)
        if self.active and self.mode=='probe':
            elapsed=self.env.time-self.governor.started_at-2.
            action=continuous_probe_action(max(0.,elapsed));label='continuous_micro_adjust'
            self.probe_modes.add(label)
            self.probe_stage=label
        elif self.mode in {'native','zero'}: action=np.zeros(3)
        self.raw_action=action.copy(); self.actions.append(action.copy()); self.rewards.append(0.)
        self.masks.append(self.active and self.mode in {'train','evaluate','probe'})
        self.previous_action=action.copy(); self.open=True; self.steps_in_frame=0
        if self.active:
            self.active_frames+=1
            if self.mode in {'train','evaluate','probe'}:
                self.guard.residual_steps+=1
                if self.guard.residual_steps>DEFAULT.max_residual_steps:
                    raise BudgetStop('stop_step_budget')

    def before_control(self,env,command):
        self.guard.poll()
        if not self.open: self.begin()
        self.command_started_at=float(env.time)
        self.tcp_before=np.asarray(env.tcp()).copy(); self.tcp_yaw_before=float(env.ee.tcp_yaw())
        self.command_before=(self.previous_command.copy() if self.previous_command is not None else
                             np.r_[self.tcp_before,self.tcp_yaw_before])

    def substep(self,env):
        dt=float(env.model.opt.timestep)
        event=self.reward.observe(contact_mask(env),dt,active=self.active)
        if self.rewards: self.rewards[-1]+=event['reward']

    def after_control(self,env,command):
        import mujoco
        self.control_steps+=1; self.steps_in_frame+=1
        if self.governor is not None: self.ACT_control_steps+=1
        self.accepted_velocity=self.control.velocity.copy()
        self.accepted_offset=self.control.offset.copy()
        self.accepted_filter=self.control.filtered_action.copy()
        if self.active:
            norm=float(np.linalg.norm(self.control.velocity[:2]))
            self.residual_norm_sum+=norm; self.residual_samples+=1
            self.nonzero_residual_samples+=int(norm>1e-8 or abs(self.control.velocity[2])>1e-8)
        values=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(env.model,env.data,values,mujoco.mjtState.mjSTATE_INTEGRATION)
        applied=np.array([command.x,command.y,command.z,command.yaw],dtype=np.float32)
        audit=self.control.last_audit.copy() if self.active and self.mode!='native' and self.control.last_audit else None
        if audit is not None:
            audit['applied']=applied.copy()
            audit['filter_accepted']=True
            audit['filter_after_accepted']=self.accepted_velocity.copy()
        self.command_digest.update(applied.tobytes()); self.z_digest.update(applied[2:3].tobytes())
        force=float(env.normal_force()); error=abs(float(env.tcp()[2])-float(command.z))
        self.peak=max(self.peak,force); self.force_sum+=force; self.force_samples+=1
        self.low_force+=int(force<.1); self.z_error_sum+=error; self.z_error_max=max(self.z_error_max,error)
        self.trace.append(dict(t=float(env.time),command=applied,tcp=env.tcp(),
            control_index=self.control_steps,command_before=self.command_before,
            command_started_at=self.command_started_at,tcp_before=self.tcp_before,
            tcp_yaw_before=self.tcp_yaw_before,tcp_yaw=float(env.ee.tcp_yaw()),
            nominal_velocity=self.nominal_velocity.copy(),probe_stage=self.probe_stage,
            normal_force=force,raw_normal_force=float(env.raw_normal_force()),
            compensated_normal_force_raw=float(env.compensated_normal_force_raw()),
            final_collected=int(env.collected_mask().sum()),active=self.active,
            phase=self.current_phase,raw_action=self.raw_action,
            ACT_feature_sources=getattr(self.scheduler,'feature_components',[]),
            zero_shadow_audit=self.shadow.last_audit if self.shadow is not None else None,
            residual_velocity=self.control.velocity.copy(),
            residual_offset=self.control.offset.copy(),action_contract=DEFAULT.action_contract,
            replay_state=dict(spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),values=values),
            control_audit=audit))
        self.previous_command=applied.copy()
        if self.steps_in_frame>=4: self.open=False

    def finish(self,outcome):
        from .learning import Episode
        if outcome is None:
            terminal=0.
            if self.actions and self.steps_in_frame==0:
                self.actions.pop(); self.rewards.pop(); self.masks.pop()
                self.actor_rows.pop(); self.critic_rows.pop()
                self.previous_action=self.actions[-1].copy() if self.actions else np.zeros(3)
                self.control.velocity=self.accepted_velocity.copy()
                self.control.offset=self.accepted_offset.copy()
                self.control.filtered_action=self.accepted_filter.copy()
        else:
            terminal=self.reward.finish(outcome)
        if not self.actions: return None,terminal
        self.rewards[-1]+=terminal
        actor,critic=self.observations()
        self.actor_rows.append(actor); self.critic_rows.append(critic)
        terminated=np.zeros(len(self.actions),bool); terminated[-1]=outcome is not None
        episode=Episode(self.spec['episode_id'],self.spec['target_count'],
            np.stack(self.actor_rows),np.stack(self.critic_rows),np.stack(self.actions),
            np.asarray(self.rewards,np.float32),terminated,np.asarray(self.masks,bool))
        return episode,terminal

    def preserve_partial(self,reason):
        """Available observations/actions only; never synthesize a terminal or Episode."""
        path=self.trace.path.parent/'partial_experience.npz'
        try:
            self.guard.checkpoint_allowance(compact=True)  # before stacking numeric context
            arrays={}
            for name,rows in (('actor_obs',self.actor_rows),('critic_obs',self.critic_rows),
                              ('actions',self.actions),('rewards',self.rewards),('learn_mask',self.masks)):
                if len(rows): arrays[name]=np.asarray(rows,dtype=bool if name=='learn_mask' else np.float32)
            return write_experience(path,arrays,self.guard,episode_id=self.spec['episode_id'],
                target_count=self.spec['target_count'],complete=False,reason=str(reason),
                transitions_may_be_unexecuted=True)
        except Exception as failure:
            evidence=dict(complete=False,valid_episode=False,saved=False,reason=str(reason),
                error=f'{type(failure).__name__}: {failure}')
            try: write_json(path.with_suffix('.json'),evidence)
            except Exception as evidence_failure:
                warnings.warn(f'partial experience not saved: {evidence}; evidence error: {evidence_failure}',RuntimeWarning)
            return evidence


def install_driver(formal,module,cfg,context,collector,cache):
    from sim.act import evaluate
    original=evaluate.run_act_episode
    try:
        evaluate.run_act_episode=streaming_native(original)
        adapted,governor=formal.build_formal_act_driver(module,cfg,context)
    finally: evaluate.run_act_episode=original
    collector.context=context; collector.governor=governor; collector.open=False
    context['trace']=RecentRows(context['trace'][-2:]); context['act_trace']=RecentRows()
    ns=adapted.__globals__
    old_reference=ns['_policy_reference_substep']; old_record=ns['_trial_record']
    native=closure_value(old_reference,'native'); base_record=closure_value(old_record,'record')
    base_builder=ns['build_act_policy']; base_scheduler=ns['ActionChunkScheduler']
    cache.mailbox=collector.mailbox
    ns['build_act_policy']=lambda cfg,pretrained_path:cache.build(base_builder,cfg,pretrained_path)
    ns['ActionChunkScheduler']=observed_scheduler(base_scheduler,collector.mailbox,
        on_created=lambda scheduler:setattr(collector,'scheduler',scheduler))
    generated_target=None
    def reference(start,target,substep,substeps,dt,max_xy_speed,max_z_speed,max_yaw_rate,
                  contact_latched,admittance_z=None):
        nonlocal generated_target
        active=collector.is_active()
        if int(substep)==1:
            scheduler=getattr(collector,'scheduler',None)
            holding=(scheduler is not None and scheduler.observation_chunk is None)
            actual_xy=np.asarray(collector.env.tcp(),dtype=float)[:2].copy()
            actual_yaw=float(collector.env.ee.tcp_yaw())
            collector.act_displacement=np.r_[np.asarray(target)[:2]-actual_xy,
                np.arctan2(np.sin(float(target[3])-actual_yaw),np.cos(float(target[3])-actual_yaw))]
            endpoint=native(start,target,substeps,substeps,dt,max_xy_speed,max_z_speed,
                max_yaw_rate,contact_latched,admittance_z)
            delta=endpoint-np.asarray(start)
            collector.begin(np.r_[delta[:2],np.arctan2(np.sin(delta[3]),np.cos(delta[3]))]/(dt*substeps))
            generated_target=collector.control.generate(actual_xy,target,
                np.zeros(3) if collector.mode=='native' else collector.raw_action,
                dt=dt*substeps,active=active,allow_motion=not holding)
            collector.control.last_audit['frame_started_at']=float(collector.env.time)
            if collector.mode=='native' and generated_target.tobytes()!=np.asarray(target).tobytes():
                raise RuntimeError('zero relative target changed native ACT target')
            if getattr(collector,'shadow',None) is not None: collector.shadow.begin_frame(start)
        collector.current_phase='ACT_residual' if active else 'ACT_handoff'
        if not active or collector.mode=='native':
            actual=old_reference(start,target,substep,substeps,dt,max_xy_speed,max_z_speed,
                                 max_yaw_rate,contact_latched,admittance_z)
            if getattr(collector,'shadow',None) is not None:
                raw=(native(start,target,substep,substeps,dt,max_xy_speed,max_z_speed,max_yaw_rate,
                            contact_latched,admittance_z) if active else actual)
                collector.shadow.check(actual,raw,dt=dt,active=active)
            return actual
        result=old_reference(start,generated_target,substep,substeps,dt,max_xy_speed,max_z_speed,max_yaw_rate,
                             contact_latched,admittance_z)
        collector.control.last_audit.update(execution_start=np.asarray(start).copy(),
            substep=int(substep),substeps=int(substeps),control_dt=float(dt),
            admittance_z=float(admittance_z),contact_latched=bool(contact_latched),
            control_reference=np.asarray(result).copy(),
            residual_elapsed=max(0.,float(collector.env.time)-float(governor.started_at)-2.-(int(substep)-1)*float(dt)))
        collector.current_phase='ACT_residual' if active else 'ACT_handoff'
        return result
    def record(row,env):
        # The unmodified native proposer also runs after the 2s handoff. Keep
        # its genuine physical clock/acknowledgement chain for generated targets.
        old_record(row,env)
    ns.update(_policy_reference_substep=reference,_trial_record=record)
    return adapted


class HybridRuntime:
    def __init__(self,store,guard):
        self.store,self.guard=store,guard
        from .baseline import pre_dino_sources
        self.frozen=identity()
        pre_dino_sources(PROJECT,self.frozen['source_fingerprints']).activate()
        self.formal=load_formal()
        self.module=self.formal.runtime(); self.cache=PolicyCache()
        self.plan=json.loads((HYBRID/'episode_plan.json').read_text())
        schedule=json.loads((store.root/'seed_schedule.json').read_text())
        self.registry=LayoutRegistry(self.plan,schedule.get('prior_consumed_layouts',[]))

    def run(self,spec,mode,learner=None):
        import mujoco
        from sim.config import load_config
        from sim.environments.sweep_env import SweepEnv
        from sim.environments.ur10_interface import UR10CB3EndEffector
        from sim.act.expert import expert_plan_candidates
        if identity()!=self.frozen: raise RuntimeError('frozen Hybrid identity drift')
        cfg=self.formal.candidate_config(load_config(str(self.formal.OLD/'evaluation_config.yaml')),
                                         spec['target_count'])
        model=self.frozen['checkpoint']['path']; self.guard.phase=spec['phase']
        destination=self.store.root/'episodes'/spec['episode_id']
        self.store.claim(spec); destination.mkdir(parents=True,exist_ok=False)
        env=None; collector=None; result=None; error=None; outcome='failure'; physical_attempts=0
        try:
            self.guard.poll()
            with self.module.old.precision.temporary_solver(UR10CB3EndEffector,1e-6):
                env=SweepEnv(cfg,seed=spec['seed']); env.reset(seed=spec['seed'])
                poses=np.c_[env.component_positions(),env.component_quats()].astype('<f4')
                fingerprint=hashlib.sha256(poses.tobytes()).hexdigest()
                self.registry.register(spec['seed'],fingerprint,spec['phase'])
                write_json(destination/'layout.json',dict(poses=poses,layout_fingerprint=fingerprint))
                write_json(destination/'configuration.json',cfg.to_dict())
                collector=Collector(env,cfg,spec,mode,learner,self.guard,destination,FeatureMailbox())
                # Planning is geometry only, exactly once; no physical screening.
                plans=expert_plan_candidates(env,cfg,max_candidates=1)
                write_json(destination/'planning.json',dict(planning_calls=1,
                    feasible=bool(plans and plans[0].feasible),physical_candidate_screening=False))
                if plans and plans[0].feasible:
                    physical_attempts=1
                    instrument_physics(env,collector.substep,collector.after_control,collector.before_control)
                    native_step=env.step_control
                    def physical_step(command):
                        try: return native_step(command)
                        except Exception as failure:
                            collector.physical_error=failure
                            raise
                    env.step_control=physical_step
                    context=self.module.prepare_contact(env,cfg,plans[0])
                    collector.context=context
                    self.guard.poll()
                    if collector.physical_error is not None: raise collector.physical_error
                    self.formal.validate_acquisition(context)
                    if context['ready']:
                        driver=install_driver(self.formal,self.module,cfg,context,collector,self.cache)
                        result=driver(cfg,seed=spec['seed'],model_path=model,preview=True,capture_replay=True)
                        outcome='success' if result.success else ('safety' if context['safety_reason'] else 'failure')
                    elif context.get('safety_reason'): outcome='safety'
                    reason=(result.termination_reason if result is not None
                            else context.get('terminal_reason','no_handoff'))
                else: reason='no_plan'
        except BudgetStop as stop:
            if str(stop) in {'stop_collection','stop_step_budget'} and spec['phase']=='train' and collector is not None:
                outcome='budget_truncated'; reason=str(stop)
            else: raise
        except Exception as failure:
            known=classify_failure(failure)
            if known=='failure' and (collector is None or collector.physical_error is not failure):
                known=None
            if known is None:
                write_json(destination/'exception.json',dict(type=type(failure).__name__,error=str(failure),
                    final_collected=int(env.collected_mask().sum()) if env is not None else None,
                    zero_shadow_audit=(collector.shadow.last_audit
                        if collector is not None and collector.shadow is not None else None)))
                raise
            outcome=known; reason=str(failure); error=repr(failure)
        finally:
            pending=sys.exc_info()[1]
            if pending is not None and collector is not None:
                collector.preserve_partial(f'{type(pending).__name__}: {pending}')
            try:
                if collector is not None: collector.trace.close()
            finally:
                if env is not None: env.close()
        count=int(env.collected_mask().sum()) if env is not None else None
        try: episode,terminal=collector.finish(None if outcome=='budget_truncated' else outcome)
        except BaseException as failure:
            collector.preserve_partial(f'episode finalization: {type(failure).__name__}: {failure}')
            raise
        n=max(1,collector.force_samples)
        row=dict(**spec,success=None if outcome=='budget_truncated' else outcome=='success',outcome=outcome,reason=reason,
            final_collected=count,opportunities=1,planning_calls=1,physical_attempts=physical_attempts,
            no_plan=physical_attempts==0,error=error,terminal_reward=terminal,
            contact_event_return=collector.reward.event_return,
            credited_contact_return=collector.reward.rl_contact_return,
            rl_return=collector.reward.rl_contact_return+terminal,
            unique_contacts=collector.reward.unique_count,short_contact_breaks=collector.reward.short_contact_breaks,
            residual_frames=collector.active_frames,control_steps=collector.control_steps,
            ACT_control_steps=collector.ACT_control_steps,
            zero_residual_shadow=collector.shadow.summary() if collector.shadow is not None else None,
            command_sha256=collector.command_digest.hexdigest(),Z_sha256=collector.z_digest.hexdigest(),
            peak_wrist_N=collector.peak if collector.force_samples else None,
            mean_wrist_N=collector.force_sum/n if collector.force_samples else None,
            wrist_below_0_1N_fraction=collector.low_force/n if collector.force_samples else None,
            mean_Z_error_m=collector.z_error_sum/n if collector.force_samples else None,
            max_Z_error_m=collector.z_error_max if collector.force_samples else None,
            mean_XY_residual_m_s=(collector.residual_norm_sum/collector.residual_samples
                                 if collector.residual_samples else None),
            nonzero_residual_fraction=(collector.nonzero_residual_samples/collector.residual_samples
                                      if collector.residual_samples else None),
            probe_modes=sorted(collector.probe_modes),
            wrist_force_is_not_table_only_force=True,metrics_rate_hz=100,
            actor_dim=episode.actor_obs.shape[1] if episode is not None else None,
            critic_dim=episode.critic_obs.shape[1] if episode is not None else None,
            layout_fingerprint=fingerprint,configuration_frozen=True,action_contract=DEFAULT.action_contract,
            accepted_ACT_chunks=(collector.context or {}).get('accepted_chunks',0))
        row=self.store.commit_episode(row,episode,self.guard)
        if identity()!=self.frozen: raise RuntimeError('frozen inputs drifted during rollout')
        return Rollout(row,episode,cfg,collector.trace.path)
