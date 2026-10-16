"""Portable trial4 control bridge and synchronous first-five chunk execution.

All legacy source edits are in-memory path substitutions. The 67 sim source
files, control functions, parameters, states and saved model remain unchanged.
"""
import copy
from functools import lru_cache
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import time

import numpy as np

from .identity import PROJECT
from .rewards import ContactReward,classify_failure
from .storage import append_json,write_json

_solver_owner=None
_solver_users=0


class SolverLease:
    """Six live worlds share one process-local, pinned precision change."""
    def __init__(self,module,cls): self.module,self.cls=module,cls
    def __enter__(self):
        global _solver_owner,_solver_users
        if not _solver_users:
            _solver_owner=self.module.old.precision.temporary_solver(self.cls,1e-6)
            _solver_owner.__enter__()
        _solver_users+=1
    def __exit__(self,*args):
        global _solver_owner,_solver_users
        _solver_users-=1
        if not _solver_users:
            _solver_owner.__exit__(*args); _solver_owner=None


@lru_cache(maxsize=1)
def load_trial4():
    """Load pinned snapshot without importing anything from historical exports."""
    legacy=PROJECT/'hybrid_trial4_snapshot/legacy_exports'
    manifest=json.loads((PROJECT/'hybrid_trial4_snapshot/source_manifest.json').read_text())
    from .storage import sha256
    for name,expected in manifest['files'].items():
        if sha256(PROJECT/name)!=expected['sha256']: raise RuntimeError('frozen control source drift: '+name)
    for name,expected in manifest['source_fingerprints'].items():
        if sha256(PROJECT/name)!=expected: raise RuntimeError('frozen sim drift: '+name)
    original_spec=importlib.util.spec_from_file_location
    original_path=sys.path.copy()

    class PortableLoader(importlib.machinery.SourceFileLoader):
        def get_code(self,fullname):
            source=self.get_source(fullname)
            source=source.replace("Path('/Users/huazhi_luke/.codex/worktrees/act-five-view')",f'Path({str(PROJECT)!r})')
            # inspect used by native control adapters must see the same code.
            linecache.cache[self.path]=(len(source),None,source.splitlines(True),self.path)
            return compile(source,self.path,'exec')

    def portable_spec(name,location,*args,**kwargs):
        path=Path(location).resolve()
        if path.is_relative_to(legacy): kwargs['loader']=PortableLoader(name,str(path))
        return original_spec(name,location,*args,**kwargs)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            found=importlib.machinery.PathFinder.find_spec(fullname,path)
            if found and found.origin and Path(found.origin).is_relative_to(legacy):
                return portable_spec(fullname,found.origin)
            return None

    name='_pac_frozen_hybrid_formal'
    spec=original_spec(name,PROJECT/'scripts/evaluate_hybrid_paired_batch.py')
    formal=importlib.util.module_from_spec(spec); sys.modules[name]=formal
    finder=Finder(); sys.meta_path.insert(0,finder)
    try:
        importlib.util.spec_from_file_location=portable_spec
        spec.loader.exec_module(formal)
        formal.CANDIDATE=legacy/'admittance_2mm_target_gain_search_goal1_20261014'
        module=formal.runtime()
    finally:
        importlib.util.spec_from_file_location=original_spec
        sys.meta_path.remove(finder); sys.path[:]=original_path
    if module.TRIAL_SPEC.force_N!=1. or formal.PROFILE!=manifest['profile']:
        raise RuntimeError('effective trial4 profile differs')
    return formal,module


def integrate_xy_yaw(start,actions):
    from sim.act.interface import action_deltas_to_absolute
    actions=np.asarray(actions,dtype=np.float32).copy()
    if actions.ndim!=2 or actions.shape[1]!=4 or not np.isfinite(actions).all():
        raise ValueError('invalid physical action deltas')
    actions[:,2]=0.  # ACT Z can never enter integration or execution.
    return action_deltas_to_absolute(start,actions)


class BlockClock:
    def __init__(self): self.frames=0; self.active_actions=0
    @property
    def query_due(self): return self.frames%5==0
    @property
    def learned_active(self): return self.frames>=50
    def executed(self):
        if self.learned_active: self.active_actions+=1
        self.frames+=1


def acquisition_outcome(context):
    safety=context.get('safety_reason')
    reason=safety or context.get('collection_reason') or context.get('terminal_reason') or 'no_handoff'
    return ('safety' if safety else 'success' if reason=='exact target count reached' else 'failure'),reason


class HybridEpisode:
    def __init__(self,cfg,model,spec,store,registry,guard,*,on_action=None):
        from residual_rl.bridge import contact_mask,instrument_physics
        from residual_rl.storage import TraceWriter
        from sim.environments.sweep_env import SweepEnv
        from sim.environments.ur10_interface import UR10CB3EndEffector
        from sim.act.expert import expert_plan_candidates
        from sim.act.interface import ACTObservationBuilder
        self.spec,self.store,self.registry,self.guard,self.model=spec,store,registry,guard,model
        self.cfg=cfg; self.formal,self.module=load_trial4()
        from .dinov3_control import (SYNC5_PROFILE_ID, configure_sync5_admittance,
                                     load_sync5_profile, reset_trial4_admittance)
        profile_id=cfg.get_path('act.hybrid_control_profile_id')
        reset_trial4_admittance(self.module)
        if profile_id==SYNC5_PROFILE_ID:
            profile_root=cfg.get_path('act.hybrid_control_profile_root')
            if not profile_root:
                raise ValueError('DINO sync5 profile root missing from verified bundle config')
            profile=load_sync5_profile(profile_root)
            configure_sync5_admittance(self.module,cfg,profile)
        elif profile_id is not None:
            raise ValueError('unknown controller profile')
        self.path=store.claim(spec)
        self.trace=TraceWriter(self.path/'physical_trace.jsonl.gz')
        self.reward=ContactReward(spec['target_count']); self.clock=BlockClock()
        self.active=False; self.done=False; self.closed=False; self.context=None
        self.current_reward=0.; self.terminal_reward=0.; self.physical_error=None
        self.reason=''; self.outcome='failure'; self.on_action=on_action
        self.active_actions=0; self.planning_calls=0; self.physical_attempts=0
        self.peak=None; self.forces=[]; self.z_errors=[]; self.measured_count=None
        self.ik_error=None; self.saved_actions=[]; self.last_trace=None
        self.env=None; self.solver=None; self.solver_entered=False
        self.started_wall=time.time()
        try:
            self.guard.poll(collecting=spec['role']=='train')
            self.solver=SolverLease(self.module,UR10CB3EndEffector)
            self.solver.__enter__()
            self.solver_entered=True
            self.env=SweepEnv(cfg,seed=spec['seed']); self.env.reset(seed=spec['seed'])
            self.episode_start=float(self.env.time)
            initial=np.c_[self.env.component_positions(),self.env.component_quats()].astype('<f4')
            self.fingerprint=hashlib.sha256(initial.tobytes()).hexdigest()
            registry.register(spec,self.fingerprint)
            write_json(self.path/'layout.json',dict(poses=initial,layout_fingerprint=self.fingerprint))
            write_json(self.path/'configuration.json',cfg.to_dict())
            self.initial=initial
            self.builder=ACTObservationBuilder(cfg,camera_views=[k.removeprefix('observation.images.') for k in model.image_keys],environment_state_dim=1)
            instrument_physics(self.env,self._substep,self._after_control,self._before_control)
            native_step=self.env.step_control
            def physical_step(command):
                try: return native_step(command)
                except BaseException as error:
                    self.physical_error=error; self.capture_failure(error); raise
            self.env.step_control=physical_step
            self.planning_calls=1
            plans=expert_plan_candidates(self.env,cfg,max_candidates=1)
            write_json(self.path/'planning.json',dict(planning_calls=1,physical_candidate_screening=False,
                feasible=bool(plans and plans[0].feasible)))
            if not plans or not plans[0].feasible:
                self._terminal('failure','no_plan'); return
            self.physical_attempts=1
            # Retain native record/safety/termination ordering, but stream rows.
            record=self.module.old.record_row
            def stream_record(context,row,env,measurement,*,prefix=False):
                record(context,row,env,measurement,prefix=prefix)
                self._save_row(row)
                rows=context['trace' if prefix else 'act_trace']
                if len(rows)>2: del rows[:-2]
            measure=self.module.measure
            def measured(env,controller=None):
                try: return measure(env,controller)
                except BaseException as error:
                    if classify_failure(error)=='safety':
                        self.physical_error=error; self.capture_failure(error)
                    raise
            self.module.old.record_row=stream_record; self.module.measure=measured
            try: self.context=self.module.prepare_contact(self.env,cfg,plans[0])
            finally: self.module.old.record_row=record; self.module.measure=measure
            if self.physical_error is not None: raise self.physical_error
            self.formal.validate_acquisition(self.context)
            if not self.context['ready']:
                self._terminal(*acquisition_outcome(self.context))
                return
            # Obtain the exact existing Z/reference/ack chain; do not invoke its
            # asynchronous ACT prediction loop for the new chunk scheduler.
            driver,self.governor=self.formal.build_formal_act_driver(self.module,cfg,self.context)
            namespace=driver.__globals__
            self.reference_substep=namespace['_policy_reference_substep']
            self.record=namespace['_trial_record']
            self.chunk_ack=namespace['_trial_chunk_ack']
            self.z_command=namespace['_contact_z_command']
            self.reference=np.asarray(self.context['last_reference'],np.float32).copy()
            self.builder._history.extend(self.context['history'])
        except BaseException as error:
            if self._known_physical(error): self._terminal(classify_failure(error),str(error))
            else:
                self.interrupt(error); raise

    def _known_physical(self,error):
        return error is self.physical_error and classify_failure(error) is not None

    def capture_failure(self,error):
        """Measure before a legacy transaction can restore the physical state."""
        evidence=dict(error=repr(error),final_collected=None)
        try: evidence['final_collected']=int(self.env.collected_mask().sum())
        except Exception as failure: evidence['measurement_error']=repr(failure)
        self.failure_count=evidence['final_collected']
        if getattr(self.env,'model',None) is not None:
            try:
                import mujoco
                spec=mujoco.mjtState.mjSTATE_INTEGRATION
                values=np.empty(mujoco.mj_stateSize(self.env.model,spec),np.float64)
                mujoco.mj_getState(self.env.model,self.env.data,values,spec)
                evidence.update(t=float(self.env.time),replay_state=dict(spec=int(spec),values=values))
            except Exception as failure: error.add_note('failure-state capture: '+repr(failure))
        if getattr(self,'path',None) is not None:
            try: write_json(self.path/'physical_exception_state.json',evidence)
            except Exception as failure: error.add_note('failure evidence write: '+repr(failure))

    def final_count(self):
        if hasattr(self,'failure_count'): return self.failure_count
        return int(self.env.collected_mask().sum()) if self.env is not None else None

    def _before_control(self,env,command):
        self.guard.poll(collecting=self.spec['role']=='train')

    def _substep(self,env):
        from residual_rl.bridge import contact_mask
        event=self.reward.observe(contact_mask(env),float(env.model.opt.timestep),active=self.active)
        self.current_reward+=event['reward']

    def _after_control(self,env,command):
        self.measured_count=int(env.collected_mask().sum())

    def _save_row(self,row):
        if row.get('phase') in ('PAC_ACT','ACT_warmup'):
            row['pac_nominal_z_m']=float(self.context['z_nominal'])
        self.last_trace=copy.deepcopy(row)
        self.trace.append(row)
        force=float(row['normal_force']); error=abs(float(row['tcp'][2])-float(row['command'][2]))
        self.forces.append(force); self.z_errors.append(error)
        self.peak=force if self.peak is None else max(self.peak,force)

    def record_measurement(self,row,record=None):
        try: (self.record if record is None else record)(row,self.env)
        except BaseException as error:
            if classify_failure(error)=='safety':
                self.physical_error=error; self.capture_failure(error)
            raise

    def observe(self):
        obs=self.builder.observe(self.env,contact_latched=True)
        result={k:obs[k] for k in self.model.obs_keys}
        for key in self.model.image_keys:
            result[key]=np.rint(np.clip(result[key],0,1)*255).astype(np.uint8)
        return result

    def step_block(self,deltas,*,learned):
        self.block_executed=0; self.block_targets=None
        try: return self._execute_block(deltas,learned=learned)
        finally:
            if self.block_targets is not None:
                append_json(self.path/'blocks.jsonl',dict(learned=learned,executed=self.block_executed,
                    active_actions=self.active_actions,frames=self.clock.frames,reward=self.current_reward,
                    done=self.done,outcome=self.outcome if self.done else None,final_collected=self.measured_count,
                    deltas=np.asarray(deltas),targets=self.block_targets))

    def _execute_block(self,deltas,*,learned):
        from sim.controllers.hybrid import Command
        if self.done: raise RuntimeError('cannot step a terminated layout')
        self.active=bool(learned)
        if self.active!=self.clock.learned_active: raise RuntimeError('learned Actor used outside the exact warmup boundary')
        if not self.clock.query_due: raise RuntimeError('chunk boundary changed')
        targets=integrate_xy_yaw(self.reference,deltas)
        if targets.shape!=(5,4): raise ValueError('execute exactly first five actions')
        self.block_targets=targets
        self.chunk_ack(True)
        self.current_reward=0.; executed=0
        for action in targets:
            if len(self.context['actions'])+self.clock.frames>=2000:
                self._terminal('failure','2000-frame timeout before task completion'); break
            self.governor.action_ready(float(self.env.time))
            action=action.copy()
            action[0]=np.clip(action[0],self.cfg.workspace.x_min,self.cfg.workspace.x_max)
            action[1]=np.clip(action[1],self.cfg.workspace.y_min,self.cfg.workspace.y_max)
            start=self.reference.copy(); before=float(self.env.time)
            try:
                for substep in range(1,5):
                    z=self.z_command(self.cfg,self.context['admittance'],self.context['z_nominal'],
                        float(self.env.normal_force()),relief=self.context['relief'])
                    command=self.reference_substep(start,action,substep,4,.01,
                        float(self.cfg.controller.sweep_speed),float(self.cfg.controller.z_speed),
                        float(self.cfg.controller.yaw_speed_limit),True,admittance_z=z)
                    try: self.env.step_control(Command(*map(float,command)))
                    except BaseException as error: self.physical_error=error; raise
                    row=self.module.physical_row(self.env,command,self.builder,'PAC_ACT' if learned else 'ACT_warmup',True)
                    self.record_measurement(row)
                    self._save_row(row)
                    if len(self.context['act_trace'])>2: del self.context['act_trace'][:-2]
                    self.reference=np.asarray(command,np.float32).copy()
                    if self.context['safety_reason'] or self.context['collection_reason']:
                        reason=self.context['safety_reason'] or self.context['collection_reason']
                        outcome='safety' if self.context['safety_reason'] else 'success' if reason=='exact target count reached' else 'failure'
                        self._terminal(outcome,reason); break
            except BaseException as error:
                if self._known_physical(error): self._terminal(classify_failure(error),str(error))
                else: raise
            finally:
                # Partial physical action counts once, even when its last
                # substep ended in known IK/safety failure. Never count k=0.
                if float(self.env.time)>before+1e-10:
                    executed+=1; self.clock.executed()
                    self.block_executed=executed
                    if learned:
                        self.active_actions+=1
                    self.saved_actions.append(action.copy())
                    self.builder.advance_state_history(self.env,contact_latched=True)
                    if learned and self.on_action: self.on_action(self.spec)
            if self.done: break
        return executed,self.current_reward,self.done

    def _terminal(self,outcome,reason):
        if self.done: return
        self.done=True; self.outcome=outcome; self.reason=reason
        self.terminal_reward=self.reward.finish(outcome)
        if self.active: self.current_reward+=self.terminal_reward
        if 'ik' in reason.lower(): self.ik_error=reason

    def finish(self,*,interrupted=False):
        if self.closed: return
        if not self.done and not interrupted: raise RuntimeError('unfinished physical task cannot be committed as failure')
        count=self.final_count()
        row=dict(**self.spec,opportunities=1,planning_calls=self.planning_calls,physical_attempts=self.physical_attempts,
            outcome='interrupted' if interrupted else self.outcome,
            success=None if interrupted else self.outcome=='success',final_collected=count,
            reason=self.reason,active_actions=self.active_actions,frames=self.clock.frames,
            environment_return=self.reward.event_return+(0. if interrupted else self.terminal_reward),
            credited_contact_return=self.reward.rl_contact_return,terminal_reward=None if interrupted else self.terminal_reward,
            unique_contacts=self.reward.unique_count,peak_wrist_N=self.peak,
            mean_wrist_N=float(np.mean(self.forces)) if self.forces else None,
            wrist_below_0_1N_fraction=float(np.mean(np.asarray(self.forces)<.1)) if self.forces else None,
            mean_Z_error_m=float(np.mean(self.z_errors)) if self.z_errors else None,
            max_Z_error_m=max(self.z_errors) if self.z_errors else None,IK_error=self.ik_error,
            layout_fingerprint=getattr(self,'fingerprint',None),wall_seconds=time.time()-self.started_wall,
            metrics_rate_hz=100,wrist_force_is_not_table_only_force=True)
        try:
            self.trace.close(); self.store.commit(row)
        except BaseException as error:
            try: self.close()
            except BaseException as failure: error.add_note('resource cleanup: '+repr(failure))
            raise
        else: self.close()
        return row

    def close(self):
        if self.closed: return
        errors=[]
        for function in (self.trace.close,self.env.close if self.env is not None else lambda:None,
                         (lambda:self.solver.__exit__(None,None,None)) if self.solver_entered else lambda:None):
            try: function()
            except BaseException as error: errors.append(error)
        self.closed=True
        if errors: raise BaseExceptionGroup('physical resource cleanup failed',errors)

    def interrupt(self,error):
        if self.closed: return
        if self.done: return self.finish()
        self.reason=f'{type(error).__name__}: {error}'
        try:
            write_json(self.path/'exception.json',dict(error=repr(error),final_collected=self.measured_count,
                active_actions=self.active_actions,last_row=self.last_trace))
        finally: self.finish(interrupted=True)
