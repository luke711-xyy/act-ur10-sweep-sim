"""Explicit, non-resumable pilot lifecycle. Importing never executes it."""
import json
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np

from .config import DEFAULT
from .safety import BudgetStop,LiveGuard
from .storage import (BoundedCheckpointWriter,RunStore,TraceWriter,checkpoint_evidence,
                      experience_reference,previous_checkpoint,sha256,write_json)


def seed_schedule():
    validation=[dict(episode_id=f'validation_{i+1}',target_count=i+1,
                     seed=DEFAULT.validation_seed_base+i) for i in range(6)]
    preflight=[dict(episode_id=f'preflight_{i+1}',target_count=target,
                    seed=DEFAULT.preflight_seed_base+i) for i,target in enumerate((1,6))]
    preflight += [dict(episode_id=f'probe_{i+1}',target_count=target,
                      seed=DEFAULT.preflight_seed_base+100+i) for i,target in enumerate((1,6))]
    training=[dict(episode_id=f'train_{i:05}',target_count=i%6+1,
                   seed=DEFAULT.training_seed_base+i,phase='train')
              for i in range(DEFAULT.max_residual_steps)]
    return dict(training=training,validation=validation,preflight=preflight,
        layout_fingerprints='verified after deterministic reset, before any commanded execution')


def require_equivalent(native,zero):
    if any(r.get('accepted_ACT_chunks',0)<1 or r.get('residual_frames',0)<1 for r in (native,zero)):
        raise RuntimeError('preflight did not exercise real ACT after handoff')
    fields=('command_sha256','Z_sha256','control_steps','accepted_ACT_chunks',
            'success','final_collected','reason','layout_fingerprint')
    if any(native.get(key)!=zero.get(key) for key in fields):
        raise RuntimeError('zero residual changed actual commands, Z, scheduler or termination')


class UpdateSchedule:
    def __init__(self):
        self.steps=0; self.updates=0
    def observe(self,new_steps):
        if type(new_steps) is not int or new_steps<0: raise ValueError('invalid new step count')
        self.steps+=new_steps
    def due(self,*,ready):
        return bool(ready and self.updates<self.steps//25)
    def completed(self):
        if self.updates>=self.steps//25: raise RuntimeError('update has no newly collected step budget')
        self.updates+=1


def summarize(rows):
    report={}
    for phase in sorted({row['phase'] for row in rows}):
        attempts=[row for row in rows if row['phase']==phase]
        selected=[row for row in attempts if type(row.get('success')) is bool]
        known=[row for row in selected if row.get('final_collected') is not None]
        report[phase]=dict(episodes=len(selected),attempts=len(attempts),
            administrative_truncations=len(attempts)-len(selected),
            strict_successes=sum(row['success'] for row in selected),
            strict_success_rate=sum(row['success'] for row in selected)/len(selected) if selected else None,
            quantity_matches=sum(row['final_collected']==row['target_count'] for row in known),
            mean_absolute_count_error=(sum(abs(row['final_collected']-row['target_count']) for row in known)/len(known)
                                       if known else None),
            unknown_count_episodes=len(selected)-len(known),
            prefix_only_episodes=sum(row.get('residual_frames',0)==0 for row in selected),
            native_quantity_match_is_not_strict_success=True)
    return report


def own_code_hashes():
    project=Path(__file__).resolve().parents[1]
    paths=list((project/'residual_rl').glob('*.py'))+[project/'scripts/train_residual_sac.py']
    return {str(p.relative_to(project)):sha256(p) for p in sorted(paths)}


def prepare(root):
    from .runtime import identity
    root=validate_output_path(root)
    baseline=identity()  # Must fail BEFORE writing a new run on frozen-source drift.
    store=RunStore(root,create=True)
    write_json(store.root/'configuration.json',DEFAULT.to_dict())
    write_json(store.root/'seed_schedule.json',seed_schedule())
    write_json(store.root/'frozen_identity.json',baseline)
    write_json(store.root/'code_hashes.json',own_code_hashes())
    seal={name:sha256(store.root/name) for name in
          ('configuration.json','seed_schedule.json','frozen_identity.json','code_hashes.json')}
    write_json(store.root/'seal.json',seal)
    return store


def validate_output_path(root):
    from .runtime import PROJECT,HYBRID,EXPORTS
    root=Path(root).resolve()
    protected=(HYBRID,EXPORTS/'batch_fine_teached_100k_120_unique_seeds_20261014',
        EXPORTS/'batch_astar_single_attempt_120_paired_20261014',
        EXPORTS/'batch_astar_single_attempt_120_paired_fresh_20261015',
        PROJECT/'runs',PROJECT/'sim')
    if any(root==path.resolve() or root.is_relative_to(path.resolve()) for path in protected):
        raise ValueError('pilot output must not be nested in frozen results, data, models or source')
    return root


def verify_run(store):
    from .runtime import identity
    seal=json.loads((store.root/'seal.json').read_text())
    if any(sha256(store.root/name)!=value for name,value in seal.items()):
        raise RuntimeError('pilot frozen manifest changed')
    if json.loads((store.root/'configuration.json').read_text())!=DEFAULT.to_dict():
        raise RuntimeError('pilot configuration differs from approved constants')
    if own_code_hashes()!=json.loads((store.root/'code_hashes.json').read_text()):
        raise RuntimeError('pilot source changed; no fingerprint rewrite/resume allowed')
    if identity()!=json.loads((store.root/'frozen_identity.json').read_text()):
        raise RuntimeError('baseline source/config/checkpoint identity changed')
    store.require_clean()


def checkpoint(store,learner,replay,guard,label,*,compact=False,replace=False):
    import torch
    if not compact: guard.poll()
    max_bytes,deadline=guard.checkpoint_allowance(compact=compact)
    path=store.root/'checkpoints'/f'{label}.pt'; path.parent.mkdir(exist_ok=True)
    if path.exists() and not (compact and replace): raise FileExistsError(path)
    payload=dict(learner=learner.checkpoint(),replay_included=not compact,
        residual_decisions=guard.residual_steps,python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        run_config=DEFAULT.to_dict(),code_hashes=own_code_hashes(),
        durable_experience=experience_reference(store.root))
    if not compact: payload['replay']=replay.state_dict()
    temporary=path.with_suffix('.tmp'); created=False
    try:
        with temporary.open('xb') as stream:
            created=True
            sink=BoundedCheckpointWriter(stream,max_bytes=max_bytes,deadline=deadline,
                check=lambda:guard.checkpoint_allowance(compact=compact))
            torch.save(payload,sink); stream.flush(); os.fsync(stream.fileno())
        guard.checkpoint_allowance(compact=compact)
        evidence=dict(sha256=sha256(temporary),bytes=temporary.stat().st_size,
            residual_decisions=guard.residual_steps,updates=learner.updates,
            kind='compact' if compact else 'full',replay_included=not compact,
            scope='residual actor/twin critics only; frozen ACT untouched')
        temporary.replace(path)
        write_json(path.with_suffix('.json'),evidence)
        checkpoint_evidence(path)
    finally:
        if created: temporary.unlink(missing_ok=True)  # Only this attempt's incomplete file.
    return path


def shutdown_save(store,learner,replay,guard,last_checkpoint,*,complete=False):
    """Every exit records successful save, verified reuse, failure, or no learner."""
    error=None; evidence=None; rejected=[]
    if learner is None:
        status='not_applicable'
    else:
        try:
            path=(last_checkpoint if complete else
                  checkpoint(store,learner,replay,guard,'interrupted',compact=True))
            evidence=checkpoint_evidence(path); status='succeeded'
        except Exception as failure:
            error=f'{type(failure).__name__}: {failure}'
            evidence,rejected=previous_checkpoint(store.root,last_checkpoint)
            status='previous_reused' if evidence else 'failed'
    result=dict(status=status,checkpoint=evidence['checkpoint'] if evidence else None,
        evidence=evidence,error=error,rejected_checkpoints=rejected,
        durable_experience=experience_reference(store.root),
        learner_initialized=learner is not None,no_automatic_resume=True)
    write_json(store.root/'save_result.json',result)
    return result


def write_derived_report(store):
    """Derived output runs last; its failure cannot replace primary stop/save facts."""
    try:
        from .reporting import write_report
        write_report(store.root,store.rows())
    except Exception as failure:
        error=f'{type(failure).__name__}: {failure}'
        try: write_json(store.root/'report_error.json',dict(error=error,primary_evidence_preserved=True))
        except Exception as evidence_failure:
            warnings.warn(f'derived report failed: {error}; error evidence failed: {evidence_failure}',RuntimeWarning)


def audit_probe(rollout):
    import gzip
    if (rollout.row['outcome']=='safety' or rollout.row.get('accepted_ACT_chunks',0)<1
        or not {'small_start','brake','reverse'}<=set(rollout.row['probe_modes'])):
        raise RuntimeError('real ACT probe did not safely exercise start/brake/reverse stages')
    active=0; total=0; previous=None; previous_filter=np.zeros(3)
    stages=[]; counts=dict(small_start=0,brake=0,reverse=0); tcp_reverse=0
    maxima=np.zeros(4)
    def vector(value,shape):
        array=np.asarray(value,dtype=float)
        if array.shape!=shape or not np.isfinite(array).all(): raise ValueError('invalid vector')
        return array
    def close(left,right,tolerance=1e-7):
        if not np.allclose(left,right,atol=tolerance,rtol=0.): raise ValueError('inconsistent integration/filter')
    def wrap(value): return np.arctan2(np.sin(value),np.cos(value))
    def bound(value,maximum):
        norm=np.linalg.norm(value)
        return value*min(1.,maximum/norm) if norm else value
    try:
        with gzip.open(rollout.trace_path,'rt') as stream:
            for line in stream:
                row=json.loads(line); total+=1
                if row['control_index']!=total: raise ValueError('missing control record')
                t=float(row['t']); started=float(row['command_started_at'])
                if not np.isfinite([t,started]).all() or abs(t-started-.01)>1e-7:
                    raise ValueError('nonmonotonic or missing control timestamp')
                command=vector(row['command'],(4,)); before=vector(row['command_before'],(4,))
                tcp=vector(row['tcp'],(3,)); tcp_before=vector(row['tcp_before'],(3,))
                if previous is not None:
                    close(started,previous['t']); close(before,previous['command'])
                    close(tcp_before,previous['tcp']); close(row['tcp_yaw_before'],previous['tcp_yaw'])
                if row['active']:
                    active+=1; audit=row['control_audit']; dt=t-started
                    if not audit or audit.get('active') is not True: raise ValueError('missing active audit')
                    close(float(audit['dt']),dt); close(audit['start'],before)
                    close(audit['applied'],command)
                    if audit.get('filter_accepted',True) is not True: raise ValueError('unaccepted filter state')
                    nominal=vector(row['nominal_velocity'],(3,))
                    native=vector(audit['native'],(4,)); raw=vector(audit['raw_action'],(3,))
                    composition=vector(audit.get('composition_native',native),(4,))
                    native_before=vector(audit.get('native_before',before),(4,))
                    residual=vector(audit['filtered_residual'],(3,))
                    prior=vector(audit['filter_before'],(3,)); close(prior,previous_filter)
                    if np.any(np.abs(raw)>1.+1e-7): raise ValueError('invalid raw action')
                    desired=np.r_[bound(raw[:2],1.)*.16,raw[2]*.7]
                    expected=prior+np.r_[bound(desired[:2]-prior[:2],.2*dt),
                        np.clip(desired[2]-prior[2],-.5*dt,.5*dt)]
                    close(audit['desired_residual'],desired); close(residual,expected)
                    close(audit['residual_acceleration'],(residual-prior)/dt)
                    if 'filter_after_accepted' in audit: close(audit['filter_after_accepted'],residual)
                    native_velocity=np.r_[(native[:2]-native_before[:2])/dt,
                                           wrap(native[3]-native_before[3])/dt]
                    close(native_velocity,nominal,2e-5)
                    close((composition[:2]-before[:2])/dt,native_velocity[:2],2e-5)
                    close(wrap(composition[3]-before[3])/dt,native_velocity[2],2e-5)
                    expected_command=composition.copy()
                    if np.any(residual):
                        bounds=vector(audit['bounds'],(2,2))
                        if np.any(bounds[:,0]>=bounds[:,1]): raise ValueError('invalid bounds')
                        expected_command[:2]=np.clip(before[:2]+bound(native_velocity[:2]+residual[:2],.08)*dt,
                                                    bounds[:,0],bounds[:,1])
                        expected_command[3]=before[3]+np.clip(native_velocity[2]+residual[2],-.35,.35)*dt
                    close(command,expected_command)
                    if command[2]!=native[2] or audit['Z_unchanged'] is not True:
                        raise ValueError('Z changed')
                    velocity=np.r_[(command[:2]-before[:2])/dt,wrap(command[3]-before[3])/dt]
                    tcp_velocity=(tcp[:2]-tcp_before[:2])/dt
                    tcp_yaw=wrap(float(row['tcp_yaw'])-float(row['tcp_yaw_before']))/dt
                    speeds=np.array([np.linalg.norm(velocity[:2]),abs(velocity[2]),
                                     np.linalg.norm(tcp_velocity),abs(tcp_yaw)])
                    if not np.isfinite(speeds).all() or speeds[0]>.08+2e-5 or speeds[1]>.35+2e-5:
                        raise ValueError('executed command speed exceeds limit')
                    maxima=np.maximum(maxima,speeds)
                    stage=row['probe_stage']
                    if stage not in ('small_start','brake','reverse','release'): raise ValueError('invalid probe stage')
                    if not stages or stage!=stages[-1]: stages.append(stage)
                    if stage=='small_start' and np.linalg.norm(residual)>1e-4: counts[stage]+=1
                    moving=np.linalg.norm(nominal[:2])>1e-3
                    if stage=='brake' and moving and np.dot(residual[:2],nominal[:2])<0.:
                        if speeds[0]<np.linalg.norm(nominal[:2])-.001: counts[stage]+=1
                    if stage=='reverse' and moving and np.dot(velocity[:2],nominal[:2])<-1e-6:
                        counts[stage]+=1
                        if np.dot(tcp_velocity,nominal[:2])<-1e-6: tcp_reverse+=1
                    previous_filter=residual.copy()
                elif active: raise ValueError('active probe gate closed')
                previous=row
        if total!=rollout.row['control_steps']: raise ValueError('trace length differs from physical ledger')
        if stages not in (['small_start','brake','reverse'],['small_start','brake','reverse','release']):
            raise ValueError('missing or reordered probe stages')
        if active<50 or min(counts.values())<5 or tcp_reverse<3:
            raise ValueError('probe did not demonstrate nonzero start, braking and physical reversal')
    except (KeyError,TypeError,ValueError,OSError,EOFError) as failure:
        raise RuntimeError(f'residual physical command audit failed: {failure}') from failure
    return dict(passed=True,audited_control_steps=active,braking_steps=counts['brake'],
        reverse_steps=counts['reverse'],tcp_reverse_steps=tcp_reverse,
        max_command_xy_m_s=float(maxima[0]),max_command_yaw_rad_s=float(maxima[1]),
        max_tcp_xy_m_s=float(maxima[2]),max_tcp_yaw_rad_s=float(maxima[3]),
        continuous_contact_verified=False)


def execute(store,*,preflight_only=False,started_at=None,stop_request=None):
    """Called only after CLI authorization and exclusive supervisor ownership."""
    from .runtime import HybridRuntime,exclusive_inference
    from .learning import EpisodicReplay,RecurrentSAC
    from .media import export_validation_video
    import torch
    started_at=time.monotonic() if started_at is None else started_at
    guard=LiveGuard(store.root,started_at,stop_request=stop_request); learner=None; replay=None
    initial_config=json.loads((store.root/'seed_schedule.json').read_text())
    outcome='needs_attention'; training_steps=0; updates=UpdateSchedule(); milestones=set()
    metrics=TraceWriter(store.root/'learning_metrics.jsonl.gz')
    last_checkpoint=None; recovery_slot=0
    write_json(store.root/'checkpoint_state.json',dict(initialized=False))
    try:
        verify_run(store)
        with exclusive_inference():
            engine=HybridRuntime(store,guard)
            write_json(store.root/'status.json',dict(state='preflight',training_started=False))
            preflight=[]
            for spec in initial_config['preflight'][:2]:
                paired=[]
                for mode in ('native','zero'):
                    item=dict(spec,episode_id=spec['episode_id']+'_'+mode,phase='preflight_'+mode)
                    paired.append(engine.run(item,mode).row)
                require_equivalent(*paired); preflight.append(dict(seed=spec['seed'],zero_equivalent=True))
            for spec in initial_config['preflight'][2:]:
                probe=engine.run(dict(spec,phase='probe'),'probe')
                preflight.append(dict(seed=spec['seed'],**audit_probe(probe)))
            write_json(store.root/'preflight.json',dict(complete=True,checks=preflight,
                actual_ACT=True,frozen_identity_verified=True))
            if preflight_only:
                outcome='preflight_complete'; return
            baseline=[]
            for spec in initial_config['validation']:
                item=dict(spec,episode_id=spec['episode_id']+'_base',phase='validation_base')
                result=engine.run(item,'zero'); baseline.append(result)
                if result.row['physical_attempts']:
                    export_validation_video(result.configuration,spec['seed'],result.trace_path,
                        store.root/'videos'/f'{item["episode_id"]}.mp4',guard)
            sample=next((r.episode for r in baseline if r.episode is not None),None)
            if sample is None: raise RuntimeError('validation produced no physical observation dimensions')
            learner=RecurrentSAC(sample.actor_obs.shape[1],sample.critic_obs.shape[1],seed=DEFAULT.random_seed)
            write_json(store.root/'checkpoint_state.json',dict(initialized=True))
            replay=EpisodicReplay(seed=DEFAULT.random_seed)
            last_checkpoint=checkpoint(store,learner,replay,guard,'initial')
            last_checkpoint=checkpoint(store,learner,replay,guard,'recovery_0',compact=True,replace=True)
            # Reserve at most 6*2000 residual decisions for final validation.
            # The 50k cap covers probe/training/final decisions, not just replay.
            train_stop=DEFAULT.max_residual_steps-6*DEFAULT.max_frames
            guard.training_decision_stop=train_stop
            write_json(store.root/'status.json',dict(state='training',training_started=True))
            for spec in initial_config['training']:
                guard.phase='train'
                try: guard.poll()
                except BudgetStop as stop:
                    if str(stop)=='stop_collection': break
                    raise
                if guard.residual_steps>=train_stop: break
                verify_run(store)
                before=guard.residual_steps
                try: result=engine.run(spec,'train',learner)
                except BudgetStop as stop:
                    # A partially consumed opportunity is never scored/re-run.
                    # Preserve checkpoint, mark interrupted, no final evaluation
                    # on an orphan physical episode.
                    raise
                added=guard.residual_steps-before; training_steps+=added; updates.observe(added)
                if result.episode is not None: replay.add(result.episode)
                if result.row['outcome']=='budget_truncated': break
                while updates.due(ready=replay.ready()):
                    try: guard.poll()
                    except BudgetStop as stop:
                        if str(stop)=='stop_collection': break
                        raise
                    values=learner.update(replay); metrics.append(values); updates.completed()
                recovery_slot=1-recovery_slot
                last_checkpoint=checkpoint(store,learner,replay,guard,f'recovery_{recovery_slot}',
                                           compact=True,replace=True)
                if guard.status=='stop_collection': break
                for milestone in (10000,20000,30000,40000,50000):
                    if guard.residual_steps>=milestone and milestone not in milestones:
                        milestones.add(milestone)
                        last_checkpoint=checkpoint(store,learner,replay,guard,
                            f'step_{guard.residual_steps}_milestone_{milestone}')
            guard.phase='validation_final'
            last_checkpoint=checkpoint(store,learner,replay,guard,'final')
            saved=torch.load(last_checkpoint,map_location='cpu',weights_only=False)
            final=RecurrentSAC(learner.actor_dim,learner.critic_dim)
            final.load_checkpoint(saved['learner'])
            for spec in initial_config['validation']:
                item=dict(spec,episode_id=spec['episode_id']+'_final',phase='validation_final')
                result=engine.run(item,'evaluate',final)
                if result.row['physical_attempts']:
                    export_validation_video(result.configuration,spec['seed'],result.trace_path,
                        store.root/'videos'/f'{item["episode_id"]}.mp4',guard)
            verify_run(store)
            outcome='complete'
    except BudgetStop as stop:
        outcome='budget_stopped'
        write_json(store.root/'stop.json',dict(reason=str(stop),training_steps=training_steps,
            residual_decisions=guard.residual_steps,no_automatic_resume=True))
    except BaseException as error:
        write_json(store.root/'exception.json',dict(type=type(error).__name__,error=str(error),
            no_automatic_resume=True,training_steps=training_steps))
        raise
    finally:
        cleanup_error=None
        try: metrics.close()
        except Exception as failure: cleanup_error=f'{type(failure).__name__}: {failure}'
        save=shutdown_save(store,learner,replay,guard,last_checkpoint,complete=outcome=='complete')
        last_checkpoint=save['checkpoint']
        report=summarize(store.rows())
        write_json(store.root/'summary.json',dict(complete=outcome=='complete',by_phase=report,
            training_steps=training_steps,residual_decisions=guard.residual_steps,
            updates=learner.updates if learner else 0,last_checkpoint=str(last_checkpoint) if last_checkpoint else None,
            save=save,cleanup_error=cleanup_error,
            quantity_error_is_not_strict_success=True,small_validation_is_not_significance_proof=True))
        write_json(store.root/'status.json',dict(state=outcome,training_started=learner is not None,
            complete=outcome=='complete',elapsed_wall_seconds=time.monotonic()-started_at,
            save=save,cleanup_error=cleanup_error,
            no_automatic_resume=True))
        write_derived_report(store)
