"""Bounded local physical acceptance, never a GPU-training launcher."""
import json
from pathlib import Path
import time

from .identity import LayoutRegistry,seed_spec,verify_bundle
from .lifecycle import Collection,cfg_for,experiment_lock,make_guard,require_local_slot
from .storage import Checkpoints,RunStore,append_json,write_json


def preflight(bundle,out,device='cpu',continue_from=None,*,start_index=0):
    began=time.time()
    # A new authorized acceptance uses a fresh registered substream; the old
    # single-use media continuation cannot silently acquire a new stream.
    from .identity import RANGES
    if type(start_index) is not int or start_index<0 or start_index%6 or start_index+5>=RANGES['preflight'][1]-RANGES['preflight'][0]:
        raise ValueError('preflight start index must be six-aligned within its reserved stream')
    if start_index and device!='cpu':raise ValueError('fresh local acceptance is CPU-only')
    if continue_from is not None and start_index:raise ValueError('legacy continuation cannot change layout stream')
    # Check before model loading, output creation or any physics/layout claim.
    require_local_slot()
    import torch
    from .policy import ChunkPolicy
    from .ppo import audited_trainer
    from .hybrid import HybridEpisode
    from .media import render_episode
    frozen=verify_bundle(bundle); out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=continue_from is not None)
    with experiment_lock(out,local=True):
        continuation=None
        if continue_from is not None:
            from .preflight_continuation import continuation_state,verify_replacement,claim_continuation
            original=verify_replacement(continue_from,frozen)
            continuation=continuation_state(out,original['identity'])
        guard=make_guard(out,'mac',original=continuation['budget'] if continuation else None,started_at=began)
        from .identity import watch_bundle
        guard.identity_check=watch_bundle(bundle,frozen); guard.poll()
        torch.manual_seed(42)
        store=RunStore(out); registry=LayoutRegistry(json.loads((Path(bundle)/'exclusions.json').read_text()))
        if continuation:
            for row in continuation['rows']: registry.register(row,row['layout_fingerprint'])
            claim_continuation(out,continuation,frozen['identity'])
        policy=ChunkPolicy(frozen['model'],device); trainer=audited_trainer(policy,guard.poll,out,frozen['identity'])
        counters=dict(sampled_actions=0,used_actions=0,discarded_actions=0,updates=0)
        collection=Collection(bundle,policy,store,registry,guard,counters,preflight_start_index=start_index)
        checks=dict(continuation['checks']) if continuation else {}; summaries=list(continuation['rows']) if continuation else []
        from .preflight_continuation import pending_comparisons
        pending={RunStore.key(s) for s in pending_comparisons(summaries)} if continuation else None
        write_json(out/'preflight_design.json',dict(start_index=start_index,budget=guard.identity(),
            KL_stopping=False,KL_penalty=trainer.config.kl_coef,formal_training_authorized=False,
            comparisons=[seed_spec('preflight',start_index+i,a) for a in ('old_hybrid','new_zero','new_exploration') for i in (0,5)]))
        write_json(out/'status.json',dict(state='preflight',training_started=False))
        try:
            # Same new layouts for all three arms. Each arm has its own one-shot
            # marker; no failure causes a replacement or success cherry-pick.
            for arm in ('old_hybrid','new_zero','new_exploration'):
                for index in (0,5):
                    guard.poll()
                    spec=seed_spec('preflight',start_index+index,arm)
                    if pending is not None and RunStore.key(spec) not in pending: continue
                    world=HybridEpisode(cfg_for(bundle,spec['target_count']),policy,spec,store,registry,guard)
                    try:
                        if arm=='old_hybrid' and not world.done:
                            _old_hybrid(world,policy,frozen['model'])
                        else:
                            while not world.done:
                                obs=collection.tensor_obs(world.observe())
                                if 'initial_ACT_mean_equivalence' not in checks:
                                    with torch.no_grad():
                                        a=policy.means(obs); b=policy.means(obs,reference=True)
                                    checks['initial_ACT_mean_equivalence']=bool(torch.equal(a,b))
                                collection.block(world,deterministic=arm=='new_zero',learning=False)
                        row=world.finish(); summaries.append(row)
                        render_episode(world.path,guard)
                    except BaseException as error:
                        world.interrupt(error); raise
            for round_number in (1,2):
                guard.poll()
                collection.version=trainer._n_updates
                batch,records=collection.collect({1:16,6:16},role='preflight')
                audit=trainer.train(batch)
                counters['updates']=trainer._n_updates; counters['used_actions']+=int(batch.lengths.sum())
                append_json(out/'updates.jsonl',dict(audit,acceptance_round=round_number))
                # Every executed acceptance opportunity retains its media,
                # including failures. Close live partial tasks as administrative
                # cuts rather than inventing task failure rewards.
                from .safety import StopExperiment
                stopped=StopExperiment('32-block acceptance update boundary')
                if collection.interrupt(stopped):
                    raise RuntimeError('acceptance world cleanup incomplete') from stopped
                for row in store.rows():
                    folder=out/'episodes'/store.key(row)
                    if not (folder/'media.json').exists(): render_episode(folder,guard)
                del batch,records
            # Deterministic behavior after the two accepted actual PPO updates.
            for index in (0,5):
                spec=seed_spec('preflight',start_index+index,'post_update')
                world=HybridEpisode(cfg_for(bundle,spec['target_count']),policy,spec,store,registry,guard)
                try:
                    while not world.done: collection.block(world,deterministic=True,learning=False)
                    summaries.append(world.finish()); render_episode(world.path,guard)
                except BaseException as error: world.interrupt(error); raise
            checks.update(two_real_PPO_updates=trainer._n_updates==2,
                within_opportunity_cap=len(list((out/'attempts').glob('*.json')))<=24,
                actual_new_schedule_executed=any(r.get('active_actions',0)>0 and r['arm']=='new_zero' for r in summaries))
            _audit_saved_Z(out,checks)
            verify_bundle(bundle)
            report=dict(engineering_passed=all(checks.values()),checks=checks,bundle_identity=frozen['identity'],
                training_started=False,GPU_training_started=False,physical_rows=store.rows(),
                elapsed_s=time.time()-guard.started_at,budget=guard.identity(),
                note='Native task failures are retained. Engineering acceptance is not evidence of improved success rate.')
            write_json(out/'acceptance.json',report)
            Checkpoints(out/'checkpoints',frozen['identity'],guard).save(
                dict(learner=trainer.snapshot(),config=trainer.config.asdict()),counters['used_actions'],kind='acceptance')
            write_json(out/'status.json',dict(state='preflight_complete',engineering_passed=report['engineering_passed'],training_started=False))
            return report
        except BaseException as error:
            collection.interrupt(error)
            from .update_audit import write_failure_reports,failure_audit_fields
            write_failure_reports(error,{out/'acceptance.json':dict(engineering_passed=False,incomplete=True,error=repr(error),
                bundle_identity=frozen['identity'],training_started=False,budget=guard.identity(),checks=checks,
                cleanup_errors=collection.cleanup_errors,**failure_audit_fields(trainer,error)),
                out/'status.json':dict(state='needs_attention',error=repr(error),training_started=False,automatic_retry=False)})
            raise


def _old_hybrid(world,policy,model_path):
    """Run the genuinely unchanged temporal-ensemble native driver for comparison."""
    from .policy import actor_chunk
    from sim.act import evaluate
    driver,governor=world.formal.build_formal_act_driver(world.module,world.cfg,world.context)
    namespace=driver.__globals__
    class FrozenFacade:
        config=policy.reference.config
        def eval(self): return self
        def reset(self): pass
        def predict_action_chunk(self,batch):
            import torch
            batch=dict(batch); batch['observation.images']=[batch[k] for k in policy.image_keys]
            with torch.no_grad(): return actor_chunk(policy.reference,batch)
    namespace['build_act_policy']=lambda cfg,pretrained_path:(FrozenFacade(),policy.reference.config)
    namespace['build_act_processors']=lambda *args,**kwargs:(policy.preprocessor,policy.postprocessor)
    native_record=namespace['_trial_record']
    def record(row,env):
        world.record_measurement(row,native_record); world._save_row(row)
        if len(world.context['act_trace'])>2: del world.context['act_trace'][:-2]
    namespace['_trial_record']=record
    try:
        result=driver(world.cfg,seed=world.spec['seed'],model_path=str(model_path),preview=True,capture_replay=True)
        world.clock.frames=int(result.sampled_frames)-len(world.context['actions'])
        world._terminal('success' if result.success else 'safety' if world.context['safety_reason'] else 'failure',result.termination_reason)
    except BaseException as error:
        if world._known_physical(error): world._terminal(__import__('pac_act.rewards',fromlist=['classify_failure']).classify_failure(error),str(error))
        else: raise


def _audit_saved_Z(out,checks):
    import gzip
    max_error=0.; act_rows=0; invalid=0; missing_nominal=0
    for trace in (Path(out)/'episodes').glob('*/physical_trace.jsonl.gz'):
        with gzip.open(trace,'rt') as f:
            for line in f:
                row=json.loads(line)
                if row.get('phase') in ('PAC_ACT','ACT_warmup'):
                    act_rows+=1
                    invalid+=row.get('z_owner')!='admittance'
                    nominal=row.get('pac_nominal_z_m',row.get('z_follow',{}).get('nominal_z'))
                    if nominal is not None: max_error=max(max_error,abs(row['command'][2]-nominal))
                    else: missing_nominal+=1
    checks['Z_admittance_only']=act_rows>0 and invalid==0
    checks['Z_total_2mm_bound']=act_rows>0 and missing_nominal==0 and max_error<=.00200001
