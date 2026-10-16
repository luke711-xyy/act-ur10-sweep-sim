"""Explicit training-only lifecycle, using the unmodified Hybrid control path.

No preflight, baseline/final validation or video is executed here. The completed
actual-ACT preflight is referenced as evidence, never imported as replay data.
"""
import hashlib
import json
from pathlib import Path
import time


def training_schedule():
    return dict(preflight=[],validation=[],training=dict(kind='arithmetic_round_robin_v1',
        seed_base=5000000,max_opportunities=1000000,target_cycle=[1,2,3,4,5,6],
        phase='train'),validation_steps_in_training_budget=0)


def iter_training_specs(schedule,*,start_index=0):
    plan=schedule['training']
    if plan!=training_schedule()['training'] or schedule['preflight'] or schedule['validation']:
        raise ValueError('unexpected long training sequence')
    if type(start_index) is not int or not 0<=start_index<=plan['max_opportunities']:
        raise ValueError('invalid next training layout index')
    for i in range(start_index,plan['max_opportunities']):
        yield dict(episode_id=f'train_{i:07d}',seed=plan['seed_base']+i,
            target_count=plan['target_cycle'][i%6],phase='train')


def crossed_milestones(before,after):
    if type(before) is not int or type(after) is not int or not 0<=before<=after<=1000000:
        raise ValueError('invalid effective training step count')
    return list(range((before//50000+1)*50000,after+1,50000))


def verified_preflight():
    """Read small evidence/trace hashes only; no policy, trajectory or weight load."""
    from .runtime import EXPORTS,identity
    from .storage import sha256
    root=EXPORTS/'residual_relative80_preflight_20261016_115913/videos/edit/paired_run'
    proof=json.loads((root/'preflight.json').read_text())
    frozen=json.loads((root/'frozen_identity.json').read_text())
    if not proof['complete'] or not proof['actual_ACT'] or identity()!=frozen:
        raise RuntimeError('completed same-baseline actual ACT preflight required')
    zero,probe=proof['checks']
    if not (zero['zero_equivalent'] and zero['mismatches']==0 and zero['checked_steps']>0
            and probe['passed'] and probe['exercised_action_frames']>=50
            and probe['action_contract']=='relative_xy80_yaw45_v1' and probe['Z_changes']==0):
        raise RuntimeError('actual preflight evidence is not sufficient')
    evidence=json.loads((root.parent/'evidence_summary.json').read_text())
    for name,digest in evidence['sources_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=digest:
            raise RuntimeError('preflight source evidence changed')
    rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    if (len(rows)!=2 or any(r['actor_dim']!=572 or r['critic_dim']!=791 or r['error']
                           or r['physical_attempts']!=1 for r in rows)):
        raise RuntimeError('preflight observation/control contract mismatch')
    return dict(root=str(root),proof_sha256=sha256(root/'preflight.json'),
        evidence_sha256=sha256(root.parent/'evidence_summary.json'),actor_dim=572,critic_dim=791,
        replay_imported=False,no_new_preflight_execution=True)


def prior_layouts():
    """Only explicitly identified numeric attempt/layout evidence; no media walks."""
    from .runtime import EXPORTS
    from .pilot import prior_consumed_layouts,native_check_consumed_layout
    prior=[]
    names=['residual_sac_pilot_pre_dino_20261015_2143',
        'residual_sac_pilot_ensemble_fixed_20261015_2200','residual_sac_pilot_same_stream_20261015_2219',
        'residual_sac_stability_preflight_20261016_020731','residual_pose_preflight_20261016_034044',
        'residual_pose_pair_target1_20261016_035316/videos/edit/zero_run',
        'residual_relative80_preflight_20261016_115913/videos/edit/paired_run']
    for name in names:
        root=EXPORTS/name
        if root.is_dir() and any((root/'attempts').glob('*.json')):
            prior.extend(prior_consumed_layouts(root))
    root=EXPORTS/'hybrid_restore_goal1_20261015'
    if (root/'attempt.json').is_file():prior.extend(native_check_consumed_layout(root))
    return prior


def prepare(root):
    from .config import DEFAULT
    from .runtime import identity
    from .pilot import validate_output_path,own_code_hashes
    from .storage import RunStore,sha256,write_json
    if DEFAULT.run_profile!='long_train_1m_v1':raise RuntimeError('long profile not activated')
    root=validate_output_path(root);frozen=identity();proof=verified_preflight();schedule=training_schedule()
    schedule['prior_consumed_layouts']=prior_layouts()
    store=RunStore(root,create=True)
    write_json(root/'configuration.json',DEFAULT.to_dict())
    write_json(root/'seed_schedule.json',schedule)
    write_json(root/'frozen_identity.json',frozen)
    write_json(root/'code_hashes.json',own_code_hashes())
    write_json(root/'preflight_reference.json',proof)
    write_json(root/'authorization.json',dict(run='long_train_1m_v1',human_authorized=True,
        max_training_steps=1000000,max_wall_seconds=259200,max_output_bytes=20*2**30,
        min_disk_bytes=8*2**30,checkpoint_interval_steps=50000,
        success_terminal=0,failure_terminal=0,safety_terminal=-60,
        contact_reward_base=60,excess_penalty_base=-120,no_baseline_validation=True,
        no_final_validation=True,no_new_preflight=True,no_old_replay_or_checkpoint=True,
        one_geometry_plan_per_layout=True,no_automatic_resume=True))
    names=['configuration.json','seed_schedule.json','frozen_identity.json','code_hashes.json',
        'preflight_reference.json','authorization.json']
    write_json(root/'seal.json',{name:sha256(root/name) for name in names})
    return store


def effective_steps(result):
    """Count accepted recorded 25Hz contexts, not a begun-but-unexecuted action."""
    if result.episode is None:return 0
    return int(result.episode.learn_mask.sum())


def new_guard(root,started_at,stop_request=None):
    from .safety import LiveGuard
    class FrameBoundaryGuard(LiveGuard):
        """Collector.begin owns the step cap, so finish its last four controls.

        All wall/disk/memory/signal limits still apply on every normal poll.
        Only step-budget polling is deferred to the next actual 25Hz begin.
        """
        def poll(self):
            recorded=self.residual_steps
            try:
                self.residual_steps=0
                return super().poll()
            finally:self.residual_steps=recorded
    return FrameBoundaryGuard(root,started_at,stop_request=stop_request)


def execute(store,*,started_at,stop_request=None,resume=False,attempt_token=None):
    from .config import DEFAULT
    from .runtime import HybridRuntime,exclusive_inference
    from .learning import RecurrentSAC,EpisodicReplay
    from .safety import LiveGuard,BudgetStop
    from .storage import TraceWriter,write_json
    from .pilot import verify_run,checkpoint,shutdown_save,UpdateSchedule,learn_after_episode,summarize,write_derived_report
    if DEFAULT.run_profile!='long_train_1m_v1':raise RuntimeError('wrong training profile')
    recovered=None
    if resume:
        from .resume import load_recovery
        # Fail before opening or replacing ANY existing progress/checkpoint file.
        recovered=load_recovery(store.root)
    guard=new_guard(store.root,started_at,stop_request=stop_request);guard.phase='train'
    guard.attempt_token=attempt_token
    guard.training_decision_stop=1000000
    outcome='initializing';reason=None;learner=replay=None;last_checkpoint=None
    training_steps=0;updates=UpdateSchedule();slot=0;next_index=0
    if recovered:
        learner,replay,updates=(recovered[k] for k in ('learner','replay','schedule'))
        training_steps=recovered['training_steps'];guard.residual_steps=training_steps
        next_index=recovered['next_index']
    metrics=TraceWriter(store.root/'learning_metrics.jsonl.gz',append=resume)
    if not resume:write_json(store.root/'checkpoint_state.json',dict(initialized=False))
    def recovery_state():
        guard.resume_state=dict(training_steps=training_steps,next_index=next_index,
            update_schedule=vars(updates).copy(),original_started_at=started_at)
    def progress(current=None):
        value=dict(state=outcome,training_started=learner is not None,
            training_steps=training_steps,residual_decisions=guard.residual_steps,
            updates=learner.updates if learner else 0,actor_updates=learner.actor_updates if learner else 0,
            active_episode=current,elapsed_wall_seconds=time.monotonic()-started_at,
            next_checkpoint_step=min(1000000,(training_steps//50000+1)*50000),
            counters='completed accepted training contexts; active episode not yet included')
        write_json(store.root/'training_progress.json',value)
    try:
        verify_run(store)
        schedule=json.loads((store.root/'seed_schedule.json').read_text())
        reference=json.loads((store.root/'preflight_reference.json').read_text())
        with exclusive_inference():
            engine=HybridRuntime(store,guard)
            if resume:
                for row in store.rows():
                    if row.get('layout_fingerprint') is not None:
                        engine.registry.register(row['seed'],row['layout_fingerprint'],row['phase'])
            if not resume:
                learner=RecurrentSAC(reference['actor_dim'],reference['critic_dim'],seed=DEFAULT.random_seed)
                replay=EpisodicReplay(seed=DEFAULT.random_seed)
            else:
                from .resume import restore_global_rng
                restore_global_rng(recovered['payload'])
            write_json(store.root/'checkpoint_state.json',dict(initialized=True))
            recovery_state()
            if not resume:
                last_checkpoint=checkpoint(store,learner,replay,guard,'initial')
                last_checkpoint=checkpoint(store,learner,replay,guard,'recovery_0',compact=True,replace=True)
            else:
                # Publish the previously failed milestone from saved state only.
                label=f'step_{training_steps}_resume_verified'
                last_checkpoint=checkpoint(store,learner,replay,guard,label)
                write_json(store.root/'resume_loaded.json',dict(training_steps=training_steps,
                    updates=learner.updates,actor_updates=learner.actor_updates,next_index=next_index,
                    next_seed=DEFAULT.training_seed_base+next_index,checkpoint=str(last_checkpoint),
                    original_started_at=started_at,no_physical_replay=True))
            outcome='training'
            write_json(store.root/'status.json',dict(state='training',training_started=True,
                run_profile=DEFAULT.run_profile,baseline_validation=False,preflight_executed=False))
            progress()
            for spec in iter_training_specs(schedule,start_index=next_index):
                guard.phase='train';guard.poll()
                if training_steps>=1000000:break
                verify_run(store);progress(spec)
                before=training_steps
                result=engine.run(spec,'train',learner)
                added=effective_steps(result)
                if added>guard.residual_steps-before:raise RuntimeError('recorded steps exceed scheduled actions')
                training_steps+=added;guard.residual_steps=training_steps
                next_index+=1
                if result.row['outcome']=='budget_truncated':
                    if result.episode is not None:replay.add(result.episode)
                    recovery_state()
                    for milestone in crossed_milestones(before,training_steps):
                        last_checkpoint=checkpoint(store,learner,replay,guard,
                            f'step_{training_steps}_milestone_{milestone}')
                    reason='stop_collection';break
                learned=learn_after_episode(learner,replay,updates,result.episode,guard.poll)
                for values in learned:metrics.append(values)
                metrics.stream.flush()
                recovery_state()
                # Recovery is bounded/rotating; full milestones are never overwritten.
                slot=1-slot
                last_checkpoint=checkpoint(store,learner,replay,guard,f'recovery_{slot}',compact=True,replace=True)
                for milestone in crossed_milestones(before,training_steps):
                    last_checkpoint=checkpoint(store,learner,replay,guard,
                        f'step_{training_steps}_milestone_{milestone}')
                progress()
                print(json.dumps(dict(event='train_episode_complete',episode=spec['episode_id'],
                    target=spec['target_count'],actual=result.row['final_collected'],success=result.row['success'],
                    effective_training_steps=training_steps,updates=learner.updates,
                    actor_updates=learner.actor_updates,reason=result.row['reason'])),flush=True)
                if training_steps>=1000000:break
            verify_run(store)
            if training_steps==1000000:
                last_checkpoint=checkpoint(store,learner,replay,guard,'final')
                outcome='complete'
            else:
                outcome='budget_stopped';reason=reason or 'training_sequence_exhausted'
    except BudgetStop as stop:
        outcome='budget_stopped';reason=str(stop)
    except BaseException as error:
        outcome='needs_attention'
        write_json(store.root/'exception.json',dict(type=type(error).__name__,error=str(error),
            training_steps=training_steps,policy_guard=getattr(error,'report',None),no_automatic_resume=True))
        raise
    finally:
        cleanup_error=None
        try:metrics.close()
        except Exception as error:cleanup_error=f'{type(error).__name__}: {error}'
        if outcome=='budget_stopped':write_json(store.root/'stop.json',dict(reason=reason,
            training_steps=training_steps,no_automatic_resume=True))
        save=shutdown_save(store,learner,replay,guard,last_checkpoint,complete=outcome=='complete',
            interrupted_label='interrupted_resume_1' if resume else 'interrupted')
        write_json(store.root/'summary.json',dict(complete=outcome=='complete',by_phase=summarize(store.rows()),
            training_steps=training_steps,residual_decisions=guard.residual_steps,
            updates=learner.updates if learner else 0,actor_updates=learner.actor_updates if learner else 0,
            save=save,last_checkpoint=save['checkpoint'],baseline_validation=False,final_validation=False,
            no_automatic_resume=True,cleanup_error=cleanup_error))
        write_json(store.root/'status.json',dict(state=outcome,complete=outcome=='complete',
            training_started=learner is not None,reason=reason,save=save,no_automatic_resume=True))
        progress();write_derived_report(store)
