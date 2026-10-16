"""Explicit, non-resumable pilot lifecycle. Importing never executes it."""
import json
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np

from .config import DEFAULT, STABILITY
from .safety import BudgetStop,LiveGuard
from .storage import (BoundedCheckpointWriter,RunStore,TraceWriter,checkpoint_evidence,
                      experience_reference,previous_checkpoint,sha256,write_json)


def seed_schedule(*,preflight_seed_base=None):
    base=DEFAULT.preflight_seed_base if preflight_seed_base is None else preflight_seed_base
    if type(base) is not int or base<0 or base+101>=2**32:
        raise ValueError('preflight seed base must yield valid uint32 seeds')
    validation=[dict(episode_id=f'validation_{i+1}',target_count=i+1,
                     seed=DEFAULT.validation_seed_base+i) for i in range(6)]
    preflight=[dict(episode_id=f'preflight_{i+1}',target_count=target,
                    seed=base+i) for i,target in enumerate((1,6))]
    preflight += [dict(episode_id=f'probe_{i+1}',target_count=target,
                      seed=base+100+i) for i,target in enumerate((1,6))]
    training=[dict(episode_id=f'train_{i:05}',target_count=i%6+1,
                   seed=DEFAULT.training_seed_base+i,phase='train')
              for i in range(DEFAULT.max_residual_steps)]
    other_seeds={r['seed'] for r in training+validation}
    if any(r['seed'] in other_seeds for r in preflight):
        raise ValueError('preflight seeds overlap training or validation')
    return dict(training=training,validation=validation,preflight=preflight,
        layout_fingerprints='verified after deterministic reset, before any commanded execution')


def require_equivalent(native,zero):
    """Legacy dual-rollout diagnostic, NOT the same-stream pilot gate."""
    if any(r.get('accepted_ACT_chunks',0)<1 or r.get('residual_frames',0)<1 for r in (native,zero)):
        raise RuntimeError('preflight did not exercise real ACT after handoff')
    fields=('command_sha256','Z_sha256','control_steps','accepted_ACT_chunks',
            'success','final_collected','reason','layout_fingerprint')
    if any(native.get(key)!=zero.get(key) for key in fields):
        raise RuntimeError('zero residual changed actual commands, Z, scheduler or termination')


def require_shadow_equivalent(row):
    proof=row.get('zero_residual_shadow') or {}
    checked=proof.get('checked_steps',0); active=proof.get('active_checked_steps',0)
    native=proof.get('native_sha256'); zero=proof.get('zero_sha256')
    valid_digest=(isinstance(native,str) and len(native)==64
                  and all(c in '0123456789abcdef' for c in native))
    if (row.get('physical_attempts')!=1 or row.get('accepted_ACT_chunks',0)<1
            or row.get('residual_frames',0)<1 or type(checked) is not int or type(active) is not int
            or not 0<active<=checked or checked!=row.get('ACT_control_steps')
            or checked>row.get('control_steps',0) or proof.get('single_actual_ACT_stream') is not True
            or proof.get('mismatches')!=0 or proof.get('max_abs_error')!=0.
            or not valid_digest or native!=zero or proof.get('extra_policy_queries')!=0
            or proof.get('extra_physics_steps')!=0):
        raise RuntimeError('same-stream zero residual preflight lacks exact complete active ACT proof')
    return dict(zero_equivalent=True,**proof)


def zero_residual_preflight(engine,plan):
    checks=[]
    for spec in plan:
        item=dict(spec,episode_id=spec['episode_id']+'_shadow',phase='preflight_shadow')
        row=engine.run(item,'native').row
        if row.get('outcome')=='safety' or row.get('error'):
            raise RuntimeError('zero residual preflight stopped on physical failure: '+str(row.get('reason',row)))
        checks.append(dict(seed=spec['seed'],**require_shadow_equivalent(row)))
    return checks


class UpdateSchedule:
    def __init__(self):
        self.steps=0; self.updates=0; self.pending=0; self.remainder=0
        self.ready_seen=False; self.discarded=0

    def begin_episode(self,new_steps,*,ready):
        if type(new_steps) is not int or new_steps<0: raise ValueError('invalid new step count')
        if type(ready) is not bool: raise ValueError('readiness must be boolean')
        self.steps+=new_steps
        self.discarded+=self.pending; self.pending=0
        if not ready or not self.ready_seen:
            self.ready_seen=ready; self.remainder=0
            return
        permissions,self.remainder=divmod(self.remainder+new_steps,STABILITY.steps_per_critic)
        self.pending=min(permissions,STABILITY.critics_per_episode)
        self.discarded+=permissions-self.pending

    def due(self):
        return self.pending>0

    def completed(self):
        if not self.pending: raise RuntimeError('update has no fresh-data episode permission')
        self.pending-=1; self.updates+=1


def learn_after_episode(learner,replay,schedule,episode,poll):
    """Only complete, actually executed masked transitions earn new permissions.

    Metrics are published only after the entire Actor transaction commits, so a
    rejected second Actor step cannot leave the first step looking deployed.
    """
    if episode is not None:
        replay.add(episode)
        if not episode.terminated[-1]:
            # Administrative partial data may be retained for bootstrap, but
            # it is not a completed episode and never earns update permissions.
            schedule.begin_episode(0,ready=False)
            return []
    valid_steps=int(episode.learn_mask.sum()) if episode is not None else 0
    schedule.begin_episode(valid_steps,ready=replay.training_ready())
    if not schedule.due():
        return []
    metrics=[]
    with learner.update_round(replay,latest_episode=episode) as policy_guard:
        while schedule.due():
            poll()
            values=learner.update(replay)
            if values['skipped']:
                raise RuntimeError('fresh-data learning permission lacks eligible replay')
            schedule.completed()
            metrics.append(values)
    for values in metrics:
        values['round_policy_guard']=policy_guard.last_report
    return metrics


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
    long_entry=project/'scripts/train_residual_long.py'
    if long_entry.exists():paths.append(long_entry)
    return {str(p.relative_to(project)):sha256(p) for p in sorted(paths)}


def prior_consumed_layouts(previous):
    """Read only the explicitly superseded run's small marker/layout evidence."""
    import hashlib
    previous=Path(previous)
    excluded=[]
    for marker in sorted((previous/'attempts').glob('*.json')):
        item=json.loads(marker.read_text()); episode_id=item['episode_id']
        if not isinstance(episode_id,str) or Path(episode_id).name!=episode_id:
            raise ValueError('invalid previous episode identifier')
        layout=previous/'episodes'/episode_id/'layout.json'
        source=json.loads(layout.read_text())
        poses=np.asarray(source['poses'],dtype='<f4')
        if poses.shape!=(6,7) or not np.isfinite(poses).all():
            raise RuntimeError('previous consumed layout fingerprint has invalid poses')
        fingerprint=hashlib.sha256(poses.tobytes()).hexdigest()
        if fingerprint!=source['layout_fingerprint']:
            raise RuntimeError('previous consumed layout fingerprint mismatch')
        excluded.append(dict(seed=item['seed'],layout_fingerprint=fingerprint,
            source_sha256={str(marker):sha256(marker),str(layout):sha256(layout)}))
    if not excluded: raise RuntimeError('previous consumed attempt evidence is missing')
    return excluded


def native_check_consumed_layout(previous):
    """Exclude the separately executed original-Hybrid check without replaying it."""
    import hashlib
    previous = Path(previous)
    marker = previous/'attempt.json'
    layout = previous/'initial_layout.json'
    item = json.loads(marker.read_text())
    source = json.loads(layout.read_text())
    poses = np.asarray(source['poses'], dtype='<f4')
    if (type(item['seed']) is not int or item.get('opportunities') != 1
            or type(item['target_count']) is not int or not 1 <= item['target_count'] <= 6
            or poses.shape != (6, 7) or not np.isfinite(poses).all()):
        raise RuntimeError('invalid prior native-check layout evidence')
    fingerprint = hashlib.sha256(poses.tobytes()).hexdigest()
    if fingerprint != source['sha256']:
        raise RuntimeError('prior native-check layout fingerprint mismatch')
    return [dict(seed=item['seed'], layout_fingerprint=fingerprint,
                 source_sha256={str(marker): sha256(marker), str(layout): sha256(layout)})]


def prepare(root,*,preflight_seed_base=None):
    from .runtime import EXPORTS,identity
    root=validate_output_path(root)
    baseline=identity()  # Must fail BEFORE writing a new run on frozen-source drift.
    schedule=seed_schedule(preflight_seed_base=preflight_seed_base)
    schedule['prior_consumed_layouts']=sum((prior_consumed_layouts(EXPORTS/name) for name in (
        'residual_sac_pilot_pre_dino_20261015_2143',
        'residual_sac_pilot_ensemble_fixed_20261015_2200',
        'residual_sac_pilot_same_stream_20261015_2219')),[])
    schedule['prior_consumed_layouts'] += native_check_consumed_layout(
        EXPORTS/'hybrid_restore_goal1_20261015')
    store=RunStore(root,create=True)
    write_json(store.root/'configuration.json',DEFAULT.to_dict())
    write_json(store.root/'seed_schedule.json',schedule)
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
        from .resume import verify_repair
        try: verify_repair(store.root)
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise RuntimeError('pilot source changed without a valid persistence repair authorization') from error
    if identity()!=json.loads((store.root/'frozen_identity.json').read_text()):
        raise RuntimeError('baseline source/config/checkpoint identity changed')
    store.require_clean()


def checkpoint(store,learner,replay,guard,label,*,compact=False,replace=False,resume_state=None):
    import torch
    from .storage import pack_replay,save_checkpoint_payload,reference_replay,replay_sampling_state
    if not compact: guard.poll()
    max_bytes,deadline=guard.checkpoint_allowance(compact=compact)
    path=store.root/'checkpoints'/f'{label}.pt'; path.parent.mkdir(exist_ok=True)
    if path.exists() and not (compact and replace): raise FileExistsError(path)
    payload=dict(learner=learner.checkpoint(),replay_included=not compact,
        residual_decisions=guard.residual_steps,python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        run_config=DEFAULT.to_dict(),code_hashes=own_code_hashes(),
        durable_experience=experience_reference(store.root),
        resume_state=(resume_state if resume_state is not None else getattr(guard,'resume_state',None)))
    if hasattr(replay,'_rng'):
        payload['replay_sampling']=replay_sampling_state(replay)
    referenced=not compact and hasattr(replay,'_episodes') and bool(replay._episodes)
    if not compact:
        payload['replay']=(reference_replay(store.root,replay) if referenced else
                           pack_replay(replay.state_dict()))
    payload['replay_included']=not compact and not referenced
    payload['replay_referenced']=referenced
    temporary=path.with_suffix('.tmp'); created=False
    try:
        with temporary.open('xb') as stream:
            created=True
            sink=BoundedCheckpointWriter(stream,max_bytes=max_bytes,deadline=deadline,
                check=lambda:guard.checkpoint_allowance(compact=compact))
            save_checkpoint_payload(payload,sink); stream.flush(); os.fsync(stream.fileno())
        guard.checkpoint_allowance(compact=compact)
        evidence=dict(sha256=sha256(temporary),bytes=temporary.stat().st_size,
            residual_decisions=guard.residual_steps,updates=learner.updates,
            actor_updates=getattr(learner,'actor_updates',0),
            kind='compact' if compact else 'reference_full' if referenced else 'full',
            replay_included=payload['replay_included'],replay_referenced=referenced,
            scope='residual actor/twin critics only; frozen ACT untouched')
        if referenced:
            evidence.update(replay_root=str(store.root.resolve()),
                            replay_dependencies=payload['replay']['records'])
        temporary.replace(path)
        write_json(path.with_suffix('.json'),evidence)
        checkpoint_evidence(path)
    finally:
        if created: temporary.unlink(missing_ok=True)  # Only this attempt's incomplete file.
    return path


def shutdown_save(store,learner,replay,guard,last_checkpoint,*,complete=False,interrupted_label='interrupted'):
    """Every exit records successful save, verified reuse, failure, or no learner."""
    error=None; evidence=None; rejected=[]
    if learner is None:
        status='not_applicable'
    else:
        try:
            path=(last_checkpoint if complete else
                  checkpoint(store,learner,replay,guard,interrupted_label,compact=True))
            evidence=checkpoint_evidence(path); status='succeeded'
        except Exception as failure:
            error=f'{type(failure).__name__}: {failure}'
            evidence,rejected=previous_checkpoint(store.root,last_checkpoint)
            status='previous_reused' if evidence else 'failed'
    result=dict(status=status,checkpoint=evidence['checkpoint'] if evidence else None,
        evidence=evidence,error=error,rejected_checkpoints=rejected,
        durable_experience=experience_reference(store.root),
        learner_initialized=learner is not None,no_automatic_resume=True)
    if getattr(guard,'attempt_token',None) is not None:
        result['attempt_token']=guard.attempt_token
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
    """Validate saved executed micro-offsets; never run another physical attempt."""
    if rollout.row.get('action_contract')=='relative_xy80_yaw45_v1':
        from .relative_audit import audit_relative_rollout
        return audit_relative_rollout(rollout)
    import gzip
    # Historical saved-pose fixtures keep their original, explicitly legacy audit.
    from .config import LEGACY_POSE
    cfg=LEGACY_POSE
    required={'small_offset','hold_offset','return_offset'}
    if (rollout.row['outcome']=='safety' or rollout.row.get('error')
            or rollout.row.get('accepted_ACT_chunks',0)<1
            or not required<=set(rollout.row['probe_modes'])):
        raise RuntimeError('real ACT pose probe did not safely exercise small/hold/return offsets')
    def vector(value,shape):
        array=np.asarray(value,dtype=float)
        if array.shape!=shape or not np.isfinite(array).all(): raise ValueError('invalid vector')
        return array
    def close(a,b,tol=1e-7):
        if not np.allclose(a,b,atol=tol,rtol=0.): raise ValueError('inconsistent pose audit')
    def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
    total=active=nonzero=0; previous=None; offset=np.zeros(3); stages=[]
    maxima=np.zeros(6); counts={s:0 for s in required}; return_start=None; return_end=None
    try:
        with gzip.open(rollout.trace_path,'rt') as stream:
            for line in stream:
                row=json.loads(line); total+=1
                if row['control_index']!=total: raise ValueError('missing control record')
                t=float(row['t']); started=float(row['command_started_at']); dt=t-started
                if not np.isfinite([t,started]).all() or abs(dt-.01)>1e-7: raise ValueError('invalid clock')
                command=vector(row['command'],(4,)); before=vector(row['command_before'],(4,))
                tcp=vector(row['tcp'],(3,)); tcp_before=vector(row['tcp_before'],(3,))
                if previous is not None:
                    close(started,previous['t']); close(before,previous['command'])
                    close(tcp_before,previous['tcp']); close(row['tcp_yaw_before'],previous['tcp_yaw'])
                if row['active']:
                    active+=1; audit=row['control_audit']
                    if not audit or audit['active'] is not True or audit['action_contract']!=cfg.action_contract:
                        raise ValueError('missing pose action contract/audit')
                    if audit.get('filter_accepted',True) is not True: raise ValueError('unaccepted command')
                    close(audit['dt'],dt); close(audit['start'],before); close(audit['applied'],command)
                    raw=vector(audit['raw_action'],(3,)); native=vector(audit['native'],(4,))
                    native_before=vector(audit['native_before'],(4,)); prior=vector(audit['offset_before'],(3,))
                    close(prior,offset); close(native_before[:2]+prior[:2],before[:2])
                    close(wrap(native_before[3]+prior[2]-before[3]),0.)
                    if np.any(np.abs(raw)>1.+1e-7): raise ValueError('invalid raw pose action')
                    wanted=np.r_[raw[:2]/max(1.,np.linalg.norm(raw[:2]))*DEFAULT.residual_xy_offset,
                                 raw[2]*DEFAULT.residual_yaw_offset]
                    close(audit['desired_offset'],wanted)
                    offset=np.r_[command[:2]-native[:2],wrap(command[3]-native[3])]
                    close(audit['applied_offset'],offset); close(audit['filtered_residual'],(offset-prior)/dt)
                    if command[2]!=native[2] or audit['Z_unchanged'] is not True: raise ValueError('Z changed')
                    bounds=vector(audit['bounds'],(2,2))
                    if np.any(bounds[:,0]>=bounds[:,1]) or np.any(command[:2]<bounds[:,0]-1e-7) or np.any(command[:2]>bounds[:,1]+1e-7):
                        raise ValueError('workspace violation')
                    speed=(command[:2]-before[:2])/dt
                    yaw=wrap(command[3]-before[3])/dt
                    native_speed=(native[:2]-native_before[:2])/dt
                    rate=(offset-prior)/dt
                    if (np.linalg.norm(offset[:2])>DEFAULT.residual_xy_offset+1e-7
                            or abs(offset[2])>DEFAULT.residual_yaw_offset+1e-7
                            or np.linalg.norm(rate[:2])>DEFAULT.residual_xy_speed+2e-5
                            or abs(rate[2])>DEFAULT.residual_yaw_speed+2e-5
                            or np.linalg.norm(speed)>DEFAULT.total_xy_speed+2e-5
                            or abs(yaw)>DEFAULT.total_yaw_speed+2e-5):
                        raise ValueError('pose/rate/total limit exceeded')
                    length=np.linalg.norm(native_speed)
                    native_yaw=wrap(native[3]-native_before[3])/dt
                    if abs(native_yaw)>1e-8:
                        if np.sign(native_yaw)*yaw<DEFAULT.min_forward_fraction*abs(native_yaw)-2e-5:
                            raise ValueError('residual stopped/reversed native yaw')
                    elif length<=1e-8 and abs(yaw)>2e-5:
                        raise ValueError('residual yaw moves during complete native idle')
                    if length>1e-8:
                        direction=native_speed/length; forward=speed@direction
                        side=speed@np.array([-direction[1],direction[0]])
                        if forward<DEFAULT.min_forward_fraction*length-2e-5 or abs(side)>np.tan(DEFAULT.direction_cone_radians)*forward+2e-5:
                            raise ValueError('residual stopped/reversed/redirected native motion')
                    elif np.linalg.norm(speed)>2e-5: raise ValueError('residual moves during native idle')
                    stage=row['probe_stage']
                    if stage not in required: raise ValueError('invalid pose probe stage')
                    if not stages or stages[-1]!=stage: stages.append(stage)
                    counts[stage]+=1; nonzero+=int(np.linalg.norm(offset)>1e-6)
                    if stage=='return_offset':
                        scale=np.array([DEFAULT.residual_xy_offset]*2+[DEFAULT.residual_yaw_offset])
                        if return_start is None: return_start=float(np.linalg.norm(prior/scale))
                        return_end=float(np.linalg.norm(offset/scale))
                    maxima=np.maximum(maxima,[np.linalg.norm(offset[:2]),abs(offset[2]),
                        np.linalg.norm(speed),abs(yaw),np.linalg.norm((tcp[:2]-tcp_before[:2])/dt),
                        abs(wrap(row['tcp_yaw']-row['tcp_yaw_before'])/dt)])
                elif active: raise ValueError('active pose gate closed')
                previous=row
        if total!=rollout.row['control_steps']: raise ValueError('trace/ledger length mismatch')
        if stages!=['small_offset','hold_offset','return_offset'] or min(counts.values())<5 or nonzero<5:
            raise ValueError('missing actual small/hold/return pose motion')
        if return_start is None or return_end>=return_start-1e-5:
            raise ValueError('pose return not actually exercised; probe incomplete')
    except (KeyError,TypeError,ValueError,OSError,EOFError) as failure:
        raise RuntimeError(f'residual physical pose audit failed: {failure}') from failure
    return dict(passed=True,audited_control_steps=active,nonzero_offset_steps=nonzero,
        max_xy_offset_m=float(maxima[0]),max_yaw_offset_rad=float(maxima[1]),
        max_command_xy_m_s=float(maxima[2]),max_command_yaw_rad_s=float(maxima[3]),
        max_tcp_xy_m_s=float(maxima[4]),max_tcp_yaw_rad_s=float(maxima[5]),
        continuous_contact_verified=False,action_contract=DEFAULT.action_contract)


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
            preflight=zero_residual_preflight(engine,initial_config['preflight'][:2])
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
                added=guard.residual_steps-before; training_steps+=added
                if result.row['outcome']=='budget_truncated':
                    if result.episode is not None: replay.add(result.episode)
                    break
                try:
                    learned=learn_after_episode(learner,replay,updates,result.episode,guard.poll)
                except BudgetStop as stop:
                    if str(stop)=='stop_collection': break
                    raise
                for values in learned: metrics.append(values)
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
            no_automatic_resume=True,training_steps=training_steps,
            policy_guard=getattr(error,'report',None)))
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
            actor_updates=learner.actor_updates if learner else 0,
            save=save,cleanup_error=cleanup_error,
            quantity_error_is_not_strict_success=True,small_validation_is_not_significance_proof=True))
        write_json(store.root/'status.json',dict(state=outcome,training_started=learner is not None,
            complete=outcome=='complete',elapsed_wall_seconds=time.monotonic()-started_at,
            save=save,cleanup_error=cleanup_error,
            no_automatic_resume=True))
        write_derived_report(store)
