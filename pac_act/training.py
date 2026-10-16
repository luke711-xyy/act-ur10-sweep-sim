"""Explicit GPU experiment lifecycle; no entry point starts this implicitly."""
from dataclasses import replace
import json
from pathlib import Path
import random
import time
import uuid

from .config import DEFAULT,Config
from .identity import LayoutRegistry,seed_spec,verify_bundle,bundle_digest
from .lifecycle import Collection,cfg_for,experiment_lock,make_guard,resume_counters
from .storage import Checkpoints,RunStore,append_json,read_rows,write_json,iter_rows,audit_identity


def cpu_acceptance_authorization(bundle_identity,acceptance_report,*,waive_cpu_acceptance=False):
    if waive_cpu_acceptance:
        if acceptance_report is not None:
            raise ValueError('CPU acceptance waiver cannot be combined with an acceptance report')
        return dict(schema=1,status='waived',engineering_passed=None,
            bundle_identity=bundle_identity,acceptance_report=None,
            scope='this formal training run only',
            operator_direction='User directed formal training to start without CPU acceptance preflight.',
            launch_flag='--waive-cpu-acceptance',recorded_at=time.time())
    if acceptance_report is None:
        raise ValueError('a matching CPU acceptance report is required unless the one-run waiver is explicit')
    report_path=Path(acceptance_report).resolve()
    acceptance=json.loads(report_path.read_text())
    if acceptance.get('engineering_passed') is not True or acceptance.get('bundle_identity')!=bundle_identity:
        raise RuntimeError('a passed matching CPU acceptance report is required before GPU training')
    return dict(schema=1,status='passed',engineering_passed=True,bundle_identity=bundle_identity,
        acceptance_report=str(report_path))


def write_cpu_acceptance_waiver(out,authorization):
    if authorization.get('status')!='waived' or authorization.get('engineering_passed') is not None:
        raise ValueError('only an explicit non-passing waiver may be recorded here')
    path=Path(out)/'cpu_acceptance_waiver.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path,authorization)
    return path


def run_baseline_validation(validate,*,skip=False):
    if skip:
        return dict(status='skipped',scope='this formal training run only',
            operator_direction='User directed formal training to skip the baseline evaluation.')
    validate('baseline',media='all',baseline=True)
    return dict(status='completed')


def final_accepted_state(checkpoints,counters,registry):
    import copy
    value=dict(checkpoints.latest()['state'])
    value['counters']=copy.deepcopy(counters)
    value['registry']=copy.deepcopy(registry)
    return value


def restore_registry(store,registry):
    for marker in sorted((store.root/'attempts').glob('*.json')):
        spec=json.loads(marker.read_text()); path=store.root/'episodes'/store.key(spec)/'layout.json'
        if not path.exists(): continue
        fingerprint=json.loads(path.read_text())['layout_fingerprint']
        prior=registry.used.get(fingerprint)
        if prior and (prior['seed']!=spec['seed'] or prior['role']!=spec['role']):
            raise ValueError('saved layout registry conflict')
        registry.used[fingerprint]=spec


def validation_pending(store,specs):
    keys={store.key(s) for s in specs}
    rows=[r for r in store.rows() if store.key(r) in keys]
    completed={store.key(r) for r in rows}
    for spec in specs:
        key=store.key(spec)
        if key not in completed and (store.root/'attempts'/f'{key}.json').exists():
            raise RuntimeError('unresolved validation opportunity; explicit recovery required: '+key)
    return [s for s in specs if store.key(s) not in completed],rows


def reconcile_update_audit(root,state):
    audit=state.get('last_update_audit')
    if audit and not any(r.get('accepted') and audit_identity(r)==audit_identity(audit)
                         for r in read_rows(Path(root)/'updates.jsonl')):
        append_json(Path(root)/'updates.jsonl',audit)
    write_json(Path(root)/'active_update.json',dict(commit_id=audit_identity(audit)))


def commit_update(checkpoints,state,counts,audit,root):
    """A loss curve and used-action count refer only to a durable update."""
    audit.update(commit_id=uuid.uuid4().hex,parent_commit_id=audit_identity(state.get('last_update_audit')))
    state=dict(state,last_update_audit=audit.copy())
    record=checkpoints.save(state,state['counters']['sampled_actions'])
    counts.update(state['counters'])
    reconcile_update_audit(root,state)
    return record


def evaluate_set(bundle,policy,store,registry,guard,*,arm,media='selected',specs=None,baseline=False):
    previous=[r for r in read_rows(store.root/'validations.jsonl') if r['arm']==arm]
    if previous: return previous[-1]
    from .hybrid import HybridEpisode
    from .media import heatmap,render_episode
    collection=Collection(bundle,policy,store,registry,guard,{})
    specs=specs or [seed_spec('validation',i,arm) for i in range(24)]
    pending,rows=validation_pending(store,specs)
    for spec in pending:
        guard.poll()
        world=HybridEpisode(cfg_for(bundle,spec['target_count']),policy,spec,store,registry,guard)
        try:
            while not world.done: collection.block(world,deterministic=True,baseline=baseline,learning=False)
            rows.append(world.finish())
        except BaseException as error:
            world.interrupt(error); raise
    # Resume media-only work from saved states, never execute a completed seed.
    for i,spec in enumerate(specs):
        folder=store.root/'episodes'/store.key(spec)
        if (media=='all' or (media=='selected' and i in (0,5))) and not (folder/'media.json').exists():
            if (folder/'configuration.json').exists(): render_episode(folder,guard)
    known=[r for r in rows if r['final_collected'] is not None]
    summary=dict(arm=arm,episodes=len(rows),strict_success_rate=sum(r['success'] is True for r in rows)/len(rows),
        exact_count_rate=sum(r['final_collected']==r['target_count'] for r in known)/len(rows),
        mean_absolute_error=sum(abs(r['final_collected']-r['target_count']) for r in known)/len(known) if known else None,
        unknown_counts=len(rows)-len(known),safety_terminations=sum(r['outcome']=='safety' for r in rows),
        IK_failures=sum(r.get('IK_error') is not None for r in rows),
        protocol_complete=len(rows)==len(specs) and all(r.get('outcome')!='interrupted' for r in rows))
    heatmap(rows,store.root/f'validation_{arm}.png')
    append_json(store.root/'validations.jsonl',summary)
    return summary


def recover_orphans(store):
    """Explicit resume only: retire interrupted opportunities, never rerun them."""
    known={store.key(r) for r in store.rows()}
    from collections import Counter
    journal=Counter(r['seed'] for r in iter_rows(store.root/'action_journal.jsonl'))
    for marker in sorted((store.root/'attempts').glob('*.json')):
        spec=json.loads(marker.read_text()); key=store.key(spec)
        if key in known: continue
        folder=store.root/'episodes'/key
        result=folder/'result.json'
        if result.exists():
            # Measurement was durably written, only append-only ledger commit
            # was interrupted. Reuse the actual result verbatim.
            store.commit(json.loads(result.read_text())); continue
        blocks=read_rows(folder/'blocks.jsonl'); last_count=blocks[-1].get('final_collected') if blocks else None
        block_actions=sum(r['executed'] for r in blocks if r['learned'])
        store.commit(dict(**spec,outcome='interrupted',success=None,final_collected=None,
            last_known_collected=last_count,terminal_count_known=False,
            active_actions=max(block_actions,journal[spec['seed']]),active_actions_complete=False,
            active_actions_source='durable blocks and per-episode action journal; unfinished action may be unknown',
            reason='previous process interrupted; no physical continuation or retry'))


def benchmark(collection,trainer):
    """Actual six-world rendering and backward memory test; no optimizer step."""
    import torch
    from .ppo import capture_rng,restore_rng
    start=time.time(); rng=capture_rng('cuda')
    try:
        from sim.environments.sweep_env import SweepEnv
        from sim.act.interface import ACTObservationBuilder
        from .media import configure_replay_framebuffer
        worlds=[]
        observations=[]
        try:
            for goal in range(1,7):
                cfg=cfg_for(collection.bundle,goal); seed=8_100_000+goal-1
                env=SweepEnv(cfg,seed=seed); worlds.append(env); env.reset(seed=seed)
                import hashlib
                import numpy as np
                poses=np.c_[env.component_positions(),env.component_quats()].astype('<f4')
                collection.registry.register(dict(role='preflight',seed=seed,target_count=goal,arm='GPU_memory_only'),hashlib.sha256(poses.tobytes()).hexdigest())
                collection.guard.poll()
                builder=ACTObservationBuilder(cfg,camera_views=[k.removeprefix('observation.images.') for k in collection.policy.image_keys],environment_state_dim=1)
                raw=builder.observe(env,contact_latched=False)
                obs={k:raw[k] for k in collection.policy.obs_keys}
                for key in collection.policy.image_keys: obs[key]=np.rint(np.clip(obs[key],0,1)*255).astype(np.uint8)
                observations.append(obs)
                configure_replay_framebuffer(env.model,(960,640))
                env.render_rgb('overhead_cam',size=(960,640))
            # Keep all six actual rendering contexts alive during backward.
            return _benchmark_backward(collection,trainer,observations,start)
        finally:
            for env in worlds: env.close()
    finally:
        trainer.optimizer.zero_grad(set_to_none=True); restore_rng(rng)


def _benchmark_backward(collection,trainer,observations,start):
        import torch
        import numpy as np
        batched={k:torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}
        selected=None; attempts=[]
        for micro in (4,2,1):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                trainer.optimizer.zero_grad(set_to_none=True)
                for i in range(0,6,micro):
                    obs={k:v[i:i+micro] for k,v in batched.items()}
                    loss=collection.policy.means(obs).square().mean()+collection.policy.values(obs).square().mean()
                    loss.backward(); collection.guard.poll()
                torch.cuda.synchronize()
                selected=micro; attempts.append(dict(microbatch=micro,passed=True,
                    max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),
                    free_total=torch.cuda.mem_get_info()))
                break
            except torch.cuda.OutOfMemoryError as error:
                attempts.append(dict(microbatch=micro,passed=False,error=repr(error)))
                trainer.optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if selected is None: raise RuntimeError('six environments/full model cannot fit microbatch1; design not reduced')
        trainer.config=replace(trainer.config,microbatch=selected)
        return dict(environments=6,microbatch=selected,attempts=attempts,seconds=time.time()-start,
            examples_per_second=6/max(.001,time.time()-start),optimizer_steps=0)


def train(bundle,out,acceptance_report=None,*,resume=False,waive_cpu_acceptance=False,
          skip_baseline_validation=False):
    began=time.time()
    import torch
    from .policy import ChunkPolicy
    from .ppo import audited_trainer
    from .safety import StopExperiment
    from torch.utils.tensorboard import SummaryWriter
    if resume and waive_cpu_acceptance:
        raise ValueError('CPU acceptance can only be waived for a new training run, not resume')
    if resume and skip_baseline_validation:
        raise ValueError('baseline validation can only be skipped for a new training run, not resume')
    if not torch.cuda.is_available(): raise RuntimeError('formal PAC-ACT training requires a CUDA GPU; no silent Mac/CPU fallback')
    frozen=verify_bundle(bundle)
    authorization=cpu_acceptance_authorization(frozen['identity'],acceptance_report,
        waive_cpu_acceptance=waive_cpu_acceptance)
    out=Path(out).resolve()
    if not resume: out.mkdir(parents=True,exist_ok=False)
    elif not (out/'run.json').is_file(): raise ValueError('resume needs an existing experiment, not a new run')
    with experiment_lock(out):
        run=json.loads((out/'run.json').read_text()) if resume else None
        if resume and run['bundle_identity']!=frozen['identity']: raise ValueError('resume bundle identity differs')
        guard=make_guard(out,'gpu',run['budget'] if run else None,started_at=began)
        from .identity import watch_bundle
        guard.identity_check=watch_bundle(bundle,frozen)
        guard.poll()
        random.seed(42)
        import numpy as np
        np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        store=RunStore(out); registry=LayoutRegistry(json.loads((Path(bundle)/'exclusions.json').read_text()))
        policy=ChunkPolicy(frozen['model'],'cuda'); trainer=audited_trainer(policy,guard.poll,out,frozen['identity'])
        identity=bundle_digest(dict(bundle=frozen['identity'],config=DEFAULT.asdict()))
        cp=Checkpoints(out/'checkpoints',identity,guard)
        counts=dict(sampled_actions=0,used_actions=0,discarded_actions=0,updates=0,used_blocks=0)
        if resume:
            latest=cp.latest()['state']; trainer.config=Config(**latest['config']); trainer.restore(latest['learner'])
            counts=resume_counters(out,latest['counters'])
            reconcile_update_audit(out,latest)
            registry.used=latest['registry']
            recover_orphans(store)
            restore_registry(store,registry)
        else:
            if authorization['status']=='waived':
                write_cpu_acceptance_waiver(out,authorization)
            baseline_record=(dict(status='skipped',scope='this formal training run only',
                operator_direction='User directed formal training to skip the baseline evaluation.')
                if skip_baseline_validation else dict(status='pending'))
            write_json(out/'run.json',dict(bundle_identity=frozen['identity'],checkpoint_identity=identity,
                config=DEFAULT.asdict(),budget=guard.identity(),validation_seeds=[seed_spec('validation',i) for i in range(24)],
                cpu_acceptance=authorization,baseline_validation=baseline_record))
        execution_id=uuid.uuid4().hex
        # A recovery fork gets a separate TensorBoard run. Historical orphaned
        # updates remain visible as history, never mixed into the new curve.
        writer=SummaryWriter(str(out/'tensorboard'/execution_id))
        write_json(out/'active_learning_session.json',dict(execution_id=execution_id,
            resumed=resume,selected_commit_id=audit_identity(latest.get('last_update_audit')) if resume else None))
        collection=Collection(bundle,policy,store,registry,guard,counts)
        collection.version=trainer._n_updates
        collection.commit_id=audit_identity(latest.get('last_update_audit')) if resume else None
        milestones={r['threshold'] for r in cp.index()['milestones'] if r['threshold'] is not None}
        def publish(state='running',**extra):
            write_json(out/'progress.json',counts)
            write_json(out/'status.json',dict(state=state,updated_at=time.time(),pid=__import__('os').getpid(),**extra))
        collection.on_status=lambda row:publish(phase='collection',last_seed=row['seed'])
        last_audit=latest.get('last_update_audit') if resume else None
        def state(next_counts=None):
            return dict(learner=trainer.snapshot(),counters=(counts if next_counts is None else next_counts).copy(),
                config=trainer.config.asdict(),registry=registry.used.copy(),budget=guard.identity(),last_update_audit=last_audit)
        def validation(arm,**kwargs):
            result=evaluate_set(bundle,policy,store,registry,guard,arm=arm,**kwargs)
            if not result['protocol_complete']:
                raise RuntimeError('validation contains interrupted opportunities; no automatic retry: '+arm)
            return result
        try:
            if not resume:
                cp.save(state(),0,kind='initial')
            publish(phase='GPU_startup_benchmark')
            audit=benchmark(collection,trainer)
            append_json(out/'gpu_startup_attempts.jsonl',audit)
            write_json(out/'gpu_startup.json',audit)
            if not skip_baseline_validation:
                publish(phase='baseline_validation')
            baseline_record=run_baseline_validation(validation,skip=skip_baseline_validation)
            if not resume:
                run_metadata=json.loads((out/'run.json').read_text())
                run_metadata['baseline_validation']=baseline_record
                write_json(out/'run.json',run_metadata)
            # A checkpoint can precede its milestone validation. Resume that
            # protocol before sampling another rollout, without physical retries.
            for threshold in sorted(milestones):
                publish(phase=f'validation_{threshold}')
                validation(f'step_{threshold}',media='selected')
            while counts['sampled_actions']<DEFAULT.max_actions:
                guard.poll(collecting=True); publish(phase='collecting384')
                batch,records=collection.collect()
                publish(phase='updating')
                audit=trainer.train(batch)
                next_counts=dict(counts,used_actions=counts['used_actions']+int(batch.lengths.sum()),
                    used_blocks=counts['used_blocks']+len(batch),updates=trainer._n_updates)
                next_counts['discarded_actions']=counts['sampled_actions']-next_counts['used_actions']
                noise=policy.noise.physical_std().detach().cpu().numpy()
                audit.update(sampled_actions=counts['sampled_actions'],used_actions=next_counts['used_actions'],
                    prior_cost=float(np.mean([r['prior_cost'] for r in records])),
                    environment_return=float(np.mean([r['environment_reward'] for r in records])),
                    optimization_return=float(np.mean([r['reward'] for r in records])),
                    std_xy_mm=float(noise[:,:2].mean()*1000),std_yaw_deg=float(np.degrees(noise[:,2]).mean()))
                # Atomic checkpoint publication comes before curves/counters.
                commit_update(cp,state(next_counts),counts,audit,out)
                last_audit=audit.copy(); collection.version=trainer._n_updates
                collection.commit_id=audit['commit_id']
                for k,v in audit.items():
                    if isinstance(v,(int,float)): writer.add_scalar(k,v,trainer._n_updates)
                writer.flush()
                for threshold in range(DEFAULT.milestone,counts['sampled_actions']+1,DEFAULT.milestone):
                    if threshold in milestones: continue
                    cp.save(state(),counts['sampled_actions'],kind='milestone',threshold=threshold); milestones.add(threshold)
                    publish(phase=f'validation_{threshold}')
                    validation(f'step_{threshold}',media='selected')
                del batch,records
                publish(phase='update_committed')
        except StopExperiment as stop:
            cleanup_errors=collection.interrupt(stop)
            counts['discarded_actions']=counts['sampled_actions']-counts['used_actions']
            publish(state='needs_attention' if cleanup_errors else 'budget_stopped',reason=str(stop),cleanup_errors=cleanup_errors)
            if cleanup_errors: raise RuntimeError('physical cleanup incomplete; no automatic continuation') from stop
        except BaseException as error:
            collection.interrupt(error)
            counts['discarded_actions']=counts['sampled_actions']-counts['used_actions']
            from .update_audit import write_failure_reports,failure_audit_fields
            audit_fields=failure_audit_fields(trainer,error)
            if audit_fields['rejected_update_audit'] is not None:
                try:append_json(out/'updates.jsonl',dict(audit_fields['rejected_update_audit'],update=trainer._n_updates))
                except BaseException as failure:error.add_note('rejection history append: '+repr(failure))
            write_failure_reports(error,{out/'progress.json':counts,out/'status.json':dict(state='needs_attention',
                error=repr(error),cleanup_errors=collection.cleanup_errors,automatic_resume=False,
                **audit_fields)})
            raise
        else:
            stopped=StopExperiment('effective action budget completed')
            errors=collection.interrupt(stopped)
            if errors:
                publish(state='needs_attention',cleanup_errors=errors,automatic_resume=False)
                raise RuntimeError('physical cleanup incomplete') from stopped
        finally:
            import sys
            original=sys.exception()
            try:writer.close()
            except BaseException as close_error:
                if original is None:raise
                original.add_note('TensorBoard writer close: '+repr(close_error))
        # Persist the accepted learning state, not an uncommitted partial batch.
        # If the total deadline/resource reserve is already gone, the two
        # previously verified rolling points remain authoritative.
        try:
            guard.poll()
            cp.save(final_accepted_state(cp,counts,registry.used),counts['sampled_actions'],kind='final')
            validation('final',media='all')
            verify_bundle(bundle)
            publish(state='complete',final_checkpoint='last_complete_accepted_update')
        except StopExperiment as error:
            publish(state='budget_stopped',reason=str(error),final_validation_complete=False)
