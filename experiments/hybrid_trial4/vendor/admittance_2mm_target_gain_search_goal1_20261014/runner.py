"""THROWAWAY bounded 2-mm, target-force and M/B/K search with actual ACT.

No production source/default, frozen batch, model, training, or old evidence
is changed. ACT owns XY/yaw only; the two-stage admittance always owns Z.
"""
import argparse
import gc
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
from contextlib import contextmanager,ExitStack
import fcntl

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(ROOT))
from phases import AdmittanceSpec,TwoStageAdmittance,ZCommandLimiter,compose_z,attach_follow
from sim.act.evaluate import _contact_z_command as NATIVE_CONTACT_Z

spec = importlib.util.spec_from_file_location('sealed_simultaneous_helper',
    ROOT.parent/'hybrid_astar_act_new_admittance_3mm_20261014/trial.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
spec = importlib.util.spec_from_file_location('scoped_one_second_handoff',ROOT.parent/'hybrid_astar_act_gate1mm_acc50_2s_goal1_20261014/handoff.py')
handoff = importlib.util.module_from_spec(spec); spec.loader.exec_module(handoff)
BATCH = old.BATCH
SEEDS, EXPECTED_LAYOUTS = old.SEEDS, old.EXPECTED_LAYOUTS
write = old.write

TRIAL_SPEC=AdmittanceSpec()
TRIAL_NUMBER=None
TRIAL_ROOT=None

def overrides():
    m,b,k=TRIAL_SPEC.seek
    return {
        'controller.desired_force':TRIAL_SPEC.force_N,
        'controller.admittance.m':m,'controller.admittance.b':b,'controller.admittance.k':k,
        'controller.force_filter_cutoff_hz':5.,'controller.delta_z_limit':.002,
        'controller.delta_z_upward_rate_limit':.02,'controller.delta_z_downward_rate_limit':.1,
    }


def make_config(frozen,target):
    if target not in SEEDS: raise ValueError('only approved goal-1/goal-6 layouts')
    cfg = frozen.copy()
    for key,value in overrides().items(): cfg.set_path(key,value)
    cfg.set_path('task.target_count',target)
    validate_config(frozen,cfg,target)
    return cfg


def validate_config(frozen,actual,target):
    expected = frozen.copy()
    for key,value in overrides().items(): expected.set_path(key,value)
    expected.set_path('task.target_count',target)
    if actual.to_dict() != expected.to_dict():
        raise RuntimeError('configuration differs outside temporary force-loop scope')
    if (actual.controller.safe_max_force != 20.
            or actual.controller.safe_force_dwell_steps != 3):
        raise RuntimeError('native safety limits changed')


def wrist_readout(env):
    return {'filtered_wrist_N':float(env.normal_force()),
            'raw_tared_wrist_N':float(env.raw_normal_force()),
            'compensated_wrist_N':float(env.compensated_normal_force_raw())}


def bounded_contact_z(cfg,loop,nominal,force,relief,limiter,actual_z):
    return compose_z(nominal,actual_z,force,float(cfg.workspace.z_search_min),loop,limiter,relief)


def collection_feedback(env):
    return {'trial_partial_overlap_mask':np.asarray(env.partial_collection_overlap_mask(),dtype=bool).copy()}


def measure(env,controller=None):
    sensor = wrist_readout(env)
    state = np.r_[env.data.qpos,env.data.qvel,env.data.sensordata,
                  env.ee.wrist_ft_world(),env.ee.compensated_wrench_world(),
                  list(sensor.values())]
    if not np.isfinite(state).all(): raise RuntimeError('nonfinite physical wrist/state sample')
    if env.unsafe_robot_collision(): raise RuntimeError('unsafe robot collision')
    return {**sensor,**collection_feedback(env),'force_feedback_source':'existing_compensated_filtered_wrist_Fz',
            'admittance_phase':controller.phase if controller else 'Astar_prefix'}


def _reference(start,target,z,dt=.01,speed=.08,yaw_rate=.35):
    from sim.act.evaluate import _policy_reference_substep
    return _policy_reference_substep(start,target,1,1,dt,speed,.12,yaw_rate,True,z)


def act_reference(start,target,z):
    return _reference(start,target,z)


def acquisition_reference(start,landing_target,z):
    return _reference(start,landing_target,z,speed=.04)


def protections():
    paths = old.extra_protection()
    inherited_phase=ROOT.parent/'hybrid_astar_act_wrist1n_twostage_20261014/phases.py'
    paths[str(inherited_phase)]=old.precision.digest(inherited_phase)
    for name in ['runner.py','phases.py','test_tuning.py','metrics.py','test_metrics.py']:
        p=ROOT/name; paths[str(p)] = old.precision.digest(p)
    for name in ['runner.py','phases.py','test_phases.py','test_runner.py','goal_1/final.json']:
        p=ROOT.parent/'hybrid_astar_act_actual_reconnect_goal1_20261014'/name
        paths[str(p)]=old.precision.digest(p)
    for name in ('control.py','follow.py','test_follow.py','test_control.py','bench.py','test_bench.py'):
        path=ROOT.parent/'admittance_wrist1n_small_motion_20261014'/name
        paths[str(path)]=old.precision.digest(path)
    previous=ROOT.parent/'hybrid_astar_act_handoff_only_1s_goal1_20261014'
    for name in ('runner.py','phases.py','handoff.py','test_phases.py','test_runner.py','test_handoff.py','goal_1/final.json'):
        path=previous/name
        paths[str(path)]=old.precision.digest(path)
    rates_previous=ROOT.parent/'hybrid_astar_act_up20_handoff1s_goal1_20261014'
    for name in ('runner.py','phases.py','handoff.py','curves.py','test_phases.py','test_runner.py','test_handoff.py','test_upward20.py','test_curves.py','goal_1/final.json','verification.json'):
        path=rates_previous/name
        paths[str(path)]=old.precision.digest(path)
    current_previous=ROOT.parent/'hybrid_astar_act_gate1mm_acc50_2s_goal1_20261014'
    for name in ('runner.py','phases.py','handoff.py','curves.py','test_phases.py','test_runner.py','test_handoff.py','test_upward20.py','test_experiment.py','test_curves.py','goal_1/final.json','verification.json'):
        path=current_previous/name
        paths[str(path)]=old.precision.digest(path)
    metadata=json.loads((BATCH/'metadata.json').read_text())
    sources={str(p.relative_to(PROJECT)):old.precision.digest(p) for p in (PROJECT/'sim').rglob('*.py')}
    if sources!=metadata['source_fingerprints'] or len(sources)!=67:
        raise RuntimeError('complete frozen67 simulation fingerprint mismatch')
    paths.update({str(PROJECT/k):v for k,v in sources.items()})
    canonical=json.loads((ROOT.parent/'batch_fine_teached_100k_cleanup_review_20261014/precheck_blocked_061856.json').read_text())['protected_sha256']
    if any(old.precision.digest(Path(k))!=v for k,v in canonical.items()):
        raise RuntimeError('formal records changed; no trial permitted')
    paths.update(canonical)
    return paths


@contextmanager
def formal_guard():
    with ExitStack() as stack:
        for folder in (BATCH,ROOT.parent/'batch_astar_single_attempt_120_paired_20261014'):
            h=stack.enter_context((folder/'batch.lock').open('rb'))
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protections()
        yield


def claim_trial(root,number):
    if type(number) is not int or not 1<=number<=6:
        raise ValueError('approved search has at most six uniquely numbered trials')
    path=Path(root)/f'trial_{number:02d}'
    path.mkdir()
    return path

def run_one(target):
    global TRIAL_ROOT
    if target != 1:raise ValueError('this approved inference is goal1 only')
    from sim.config import load_config
    candidate=make_config(load_config(str(BATCH/'evaluation_config.yaml')),target)
    TRIAL_SPEC.check_clock(1./float(candidate.sim.control_hz))
    with formal_guard():
        if shutil.disk_usage(ROOT).free<6*2**30:raise RuntimeError('free disk below 6 GiB')
        TRIAL_ROOT=claim_trial(ROOT,TRIAL_NUMBER)
        write(TRIAL_ROOT/'parameters.json',dict(trial=TRIAL_NUMBER,target_force_N=TRIAL_SPEC.force_N,
            seek_MBK=TRIAL_SPEC.seek,maintenance_MBK=TRIAL_SPEC.maintain,total_correction_limit_m=.002))
        return _run_one_excluded(target)


def snapshot(env):
    import mujoco
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    values=np.empty(mujoco.mj_stateSize(env.model,state_spec))
    mujoco.mj_getState(env.model,env.data,values,state_spec)
    return {'spec':int(state_spec),'values':values}


def clear_geometric_handoff(c,now,tcp):
    c['ready']=False
    c.pop('handoff_actual_table_contact',None)
    c['proximity_start_time']=float(now)
    c['proximity_start_tcp']=np.asarray(tcp).copy()
    for key in ['handoff_time','handoff_force','handoff_tcp','contact_time','first_contact_position']:
        c[key]=None


def transactional_command(step,capture,restore,force,failures,invalid_samples,command):
    try:
        return old.hybrid.transactional_step(step,capture,restore,force,failures)
    except Exception:
        failure=failures[-1] if failures else {'attempted_state':capture()}
        invalid_samples.append({'command':np.asarray(command).copy(),**failure})
        raise


def acknowledge_z(c,row):
    limiter=c['z_limiter']
    if float(row['t'])<=limiter.last_time+1e-10:return
    accepted=(row['command'][2] if row.get('control_step_advanced',True) else row['tcp'][2])
    limiter.commit(float(accepted),float(row['t']))


def physical_row(env,command,builder,phase,contact=False):
    return {'t':float(env.time),'tcp':env.tcp().copy(),'command':command.copy(),
            'policy_reference':command.copy(),'policy_z':float(command[2]),
            'applied_z':float(command[2]),'z_owner':'admittance',
            'normal_force':float(env.normal_force()),'contact':contact,'phase':phase,
            'supervisor_owned':True,'collected':int(env.collected_mask().sum()),
            'replay_state':snapshot(env),'object_collected':env.collected_mask().copy(),
            'wrench':env.wrench().copy(),'raw_normal_force':float(env.raw_normal_force()),
            'compensated_normal_force_raw':float(env.compensated_normal_force_raw()),
            'tcp_pose':np.r_[env.tcp(),env.ee.tcp_yaw()],
            'object_pose':np.c_[env.component_positions(),env.component_quats()],
            'replay_observation':builder.observe(env,contact_latched=contact,include_images=False)}


def geometry_ready(tcp_z,table_z):
    return bool(np.isfinite([tcp_z,table_z]).all() and tcp_z<=table_z+.001)


def prepare_contact(env,cfg,plan):
    from sim.act.interface import ACTObservationBuilder
    from sim.act.evaluate import _contact_z_command
    from sim.controllers.hybrid import Command
    original_measure,original_gate=old.measure,old.geometry_ready
    try:
        old.measure=measure
        old.geometry_ready=geometry_ready
        c=old.prepare_prefix(env,cfg,plan)
    finally:
        old.measure=original_measure
        old.geometry_ready=original_gate
    # Geometric proximity only starts wrist-force acquisition, never ACT.
    if not c['ready']:
        if isinstance(c.get('error'),str) and '3 mm' in c['error']:
            c['original_helper_error']=c['error']
            c['error']=c['error'].replace('3 mm','1 mm')
        return c
    clear_geometric_handoff(c,env.time,env.tcp())
    c['admittance']=TwoStageAdmittance(env.control_dt,TRIAL_SPEC)
    c['relief'].reset()
    builder=ACTObservationBuilder(cfg); builder._history.extend(c['history'])
    reference=c['last_reference'].copy()
    c['z_limiter']=ZCommandLimiter(float(reference[2]),env.control_dt,initial_time=env.time)
    landing=reference.copy(); landing[:2]=np.asarray(plan.waypoints[1],dtype=float)
    period=int(round(float(cfg.sim.control_hz)/float(cfg.act.action_hz)))
    count=0
    def capture_attempt():
        return {'time':float(env.time),'integration_state':snapshot(env),'wrist':wrist_readout(env)}
    def restore_attempt(before):
        import mujoco
        state=before['integration_state']
        mujoco.mj_setState(env.model,env.data,np.asarray(state['values']),state['spec'])
        mujoco.mj_forward(env.model,env.data)
        mujoco.mj_rnePostConstraint(env.model,env.data)
    try:
        for _ in range(round(30./env.control_dt)):
            if len(c['actions']) >= 1999: raise RuntimeError('shared frame budget exhausted before stable wrist contact')
            z=bounded_contact_z(cfg,c['admittance'],c['z_nominal'],float(env.normal_force()),c['relief'],c['z_limiter'],float(env.tcp()[2]))
            reference=acquisition_reference(reference,landing,z)
            transactional_command(lambda:env.step_control(Command(*map(float,reference))),
                capture_attempt,restore_attempt,env.normal_force,c['step_failures'],c['invalid_samples'],reference)
            count+=1
            row=physical_row(env,reference,builder,'wrist_'+c['admittance'].phase,
                             contact=c['admittance'].phase!='seek')
            attach_follow(row,c['z_limiter'])
            old.record_row(c,row,env,lambda e:measure(e,c['admittance']),prefix=True)
            acknowledge_z(c,row)
            c['admittance'].observe(float(env.time),float(env.normal_force()))
            c['peak_force']=max(c['peak_force'],float(env.normal_force()))
            if count%period==0:
                c['actions'].append(reference.copy())
                builder.advance_state_history(env,contact_latched=c['admittance'].phase!='seek')
            if c['safety_reason'] or c['collection_reason']:
                c['terminal_reason']=c['safety_reason'] or c['collection_reason']
                raise RuntimeError('acquisition terminated: '+c['terminal_reason'])
            if c['admittance'].ready:
                if count%period:
                    c['actions'].append(reference.copy())
                    builder.advance_state_history(env,contact_latched=True)
                c.update(ready=True,handoff_time=float(env.time),handoff_force=float(env.normal_force()),
                    handoff_tcp=env.tcp().copy(),contact_time=float(env.time),
                    first_contact_position=env.tcp().copy(),last_reference=reference.copy(),
                    prefix_frames=len(c['actions']),history=[(t,a.copy()) for t,a in builder._history])
                return c
        raise RuntimeError('stable wrist-force band not established within 30 s; ACT sweep not started')
    except Exception as error:
        c['error']=repr(error)
        c['terminal_reason']=c.get('terminal_reason') or str(error)
        return c


def wrist_metrics(rows,handoff=None):
    valid=[r for r in rows if np.isfinite(r.get('normal_force',np.nan))]
    force=np.asarray([r['normal_force'] for r in valid])
    post=[r for r in valid if handoff is not None and r['t']>=handoff]
    tail=[r for r in valid if valid[-1]['t']-r['t']<=2.] if valid else []
    return {'wrist_force_peak_N':float(force.max()) if len(force) else None,
            'minimum_tcp_z_m':min(float(r['tcp'][2]) for r in valid) if valid else None,
            'post_ACT_wrist_above_0_5N_fraction':sum(r['normal_force']>=.5 for r in post)/len(post) if post else None,
            'tail2s_wrist_mean_N':float(np.mean([r['normal_force'] for r in tail])) if tail else None,
            'tail2s_wrist_std_N':float(np.std([r['normal_force'] for r in tail])) if tail else None,
            'tail2s_wrist_in_0_8_to_1_2N_fraction':sum(.8<=r['normal_force']<=1.2 for r in tail)/len(tail) if tail else None,
            'admittance_rows':sum(r.get('z_owner')=='admittance' for r in valid),
            'policy_Z_rows_after_handoff':sum(r.get('z_owner')=='policy' for r in post),
            'table_force_never_used_as_feedback':True,
            'stable_wrist_force_is_not_proof_of_table_only_contact':True}


def build_act_driver(cfg,c):
    """Connect the one-shot hook to the real, reused policy/scheduler loop."""
    from sim.act import evaluate
    env=c['env'];controller=c['admittance']
    original_contact_z=evaluate._contact_z_command
    def limited_contact_z(config,loop,nominal,force,relief=None):
        return bounded_contact_z(config,loop,nominal,force,relief,c['z_limiter'],float(env.tcp()[2]))
    try:
        evaluate._contact_z_command=limited_contact_z
        adapted=old.adapt_act(evaluate.run_act_episode,c,measure=lambda e:measure(e,controller))
    finally:
        evaluate._contact_z_command=original_contact_z
    original_capture=adapted.__globals__['_trial_record']
    def capture_and_ack(row,current_env):
        attach_follow(row,c['z_limiter'])
        original_capture(row,current_env)
        acknowledge_z(c,row)
    adapted.__globals__['_trial_record']=capture_and_ack
    previous=[r for r in c['trace'] if r.get('control_step_advanced',True)]
    if len(previous)<2:raise RuntimeError('no accepted prefix velocity history')
    dt=float(previous[-1]['t'])-float(previous[-2]['t'])
    if abs(dt-float(env.control_dt))>1e-8:raise RuntimeError('prefix clock differs from control step')
    v=(np.asarray(previous[-1]['command'])[:2]-np.asarray(previous[-2]['command'])[:2])/dt
    yaw=handoff.angle_delta(float(previous[-1]['command'][3])-float(previous[-2]['command'][3]))/dt
    governor=handoff.TwoSecondHandoff(c['last_reference'],env.control_dt,float(env.time),
        bounds=[[cfg.workspace.x_min,cfg.workspace.x_max],[cfg.workspace.y_min,cfg.workspace.y_max]],
        velocity_xy=v,yaw_velocity=yaw)
    handoff.install(adapted,env,governor)
    c['handoff_governor']=governor
    return adapted,governor


def finish_video_summary(destination,summary,video,validation,identity):
    summary.update(video=str(video),media_validation=validation,final_identity_verification=identity)
    write(destination/'final.json',summary)
    print(json.dumps({'event':'video_ready','target':summary['target'],'video':str(video),
                     'success':summary['success'],'final_collected':summary['final_collected']}),flush=True)
    return summary


def _run_one_excluded(target):
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.environments.ur10_interface import UR10CB3EndEffector
    from sim.act.expert import expert_plan_candidates
    from sim.act import evaluate
    from sim.act.rollout import ActRolloutResult
    from sim.act.inference_lock import inference_lease
    from sim.act.replay import persist_replay_input,save_inference_replay
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video
    if target not in SEEDS: raise ValueError('only approved goals')
    destination=TRIAL_ROOT/f'goal_{target}'; destination.mkdir()
    if shutil.disk_usage(ROOT).free < 6*2**30: raise RuntimeError('free disk below 6 GiB')
    protected=old.precision.reviewed.base.protected_snapshot(); extra=protections()
    write(destination/'before.json',{'protected':protected,'extra_sha256':extra})
    cfg=make_config(load_config(str(BATCH/'evaluation_config.yaml')),target)
    write(destination/'configuration.json',cfg.to_dict())
    model=protected['checkpoint_path'];seed=SEEDS[target]
    env=None;c=None;result=None;primary=None;summary=None
    native_solver=UR10CB3EndEffector._solve_ik
    print(json.dumps({'event':'waiting_for_inference_slot','target':target}),flush=True)
    try:
        with inference_lease(on_wait=lambda:print(json.dumps({'event':'slot_busy'}),flush=True)):
            old.precision.reviewed.base.verify_protected(protected)
            if old.precision.protected_changes(extra,protections()): raise RuntimeError('protected evidence drift')
            with old.precision.temporary_solver(UR10CB3EndEffector,1e-6):
                env=SweepEnv(cfg,seed=seed);env.record_raw_force_peaks=True;env.reset(seed=seed)
                initial=np.c_[env.component_positions(),env.component_quats()].astype('<f4')
                fingerprint=hashlib.sha256(initial.tobytes()).hexdigest()
                write(destination/'initial_layout.json',{'poses':initial,'sha256':fingerprint})
                if fingerprint!=EXPECTED_LAYOUTS[target]: raise RuntimeError('fixed layout mismatch')
                candidates=expert_plan_candidates(env,cfg,max_candidates=1)
                if not candidates or not candidates[0].feasible: raise RuntimeError('no single feasible A* plan; no retry')
                plan=candidates[0]
                write(destination/'planning.json',{'planning_calls':1,'physical_candidate_screening':False,
                    'strategy':plan.strategy,'waypoints':plan.waypoints,'landing_region':plan.landing_region,
                    'selected_indices':plan.target_indices})
                c=prepare_contact(env,cfg,plan)
                controller=c['admittance']
                evidence={k:c.get(k) for k in ['ready','handoff_time','handoff_force','handoff_tcp',
                    'prefix_frames','z_nominal','error','step_failures','proximity_start_time','proximity_start_tcp']}
                evidence.update(target=target,seed=seed,checkpoint=model,geometry_gate_m=.001,
                    desired_force_N=TRIAL_SPEC.force_N,downward_rate_limit_m_s=.1,correction_limit_m=.002,
                    seek_MBK=list(TRIAL_SPEC.seek),maintenance_MBK=list(TRIAL_SPEC.maintain),trial_number=TRIAL_NUMBER,
                    normal_upward_rate_limit_m_s=.02,tracking_gain=.5,tracking_bias_cap_m=.0002,
                    position_feedback='existing actual TCP Z kinematics',
                    control_scheme='two_stage_wrist_admittance_plus_bounded_Z_follow',
                    force_contact_threshold_N=.5*TRIAL_SPEC.force_N,force_contact_dwell_s=.1,
                    stable_wrist_band_N=[.8*TRIAL_SPEC.force_N,1.2*TRIAL_SPEC.force_N],stable_wrist_dwell_s=.2,
                    phase_transitions=getattr(controller,'transitions',[]),
                    XY_before_handoff='Astar_landing_approach_adjustment',XY_after_handoff='ACT',
                    Z_after_1mm='two_stage_wrist_admittance_plus_Z_follow_always',
                    only_wrist_force_feedback=True,table_force_feedback=False,
                    initial_layout_fingerprint=fingerprint,no_retries=True,formal_evaluation=False)
                evidence.update(ACT_handoff_smoothing=dict(
                    window_s=2.,timer_origin='first executable valid action from the real ACT scheduler',
                    XY_speed_limit_m_s=.08,XY_acceleration_limit_m_s2=.05,
                    yaw_speed_limit_rad_s=.35,yaw_acceleration_limit_rad_s2=.5,
                    after_two_seconds='exact native ACT interpolator; no continued smoothing',
                    Z_controller_not_reset=True,safety_unchanged=True,
                    release_step_acceleration_not_limited=True))
                write(destination/'handoff.json',evidence)
                print(json.dumps(old.precision.reviewed.base.json_safe({'event':'handoff','target':target,**evidence})),flush=True)
                if c['ready']:
                    try:
                        adapted,governor=build_act_driver(cfg,c)
                        result=adapted(cfg,seed=seed,model_path=model,preview=True,capture_replay=True)
                        result.trace=c['trace']+result.trace
                        result.actions=np.concatenate((np.asarray(c['actions']),result.actions),axis=0)
                        result.observations.clear()
                    except Exception as error:
                        c['error']=repr(error)
                        c['terminal_reason']=c.get('safety_reason') or c.get('collection_reason') or str(error)
                        write(destination/'ACT_exception.json',{'error':repr(error),'traceback':traceback.format_exc()})
                if result is None:
                    trace=c['trace']+c['act_trace'];term=c.get('terminal_reason','')
                    result=ActRolloutResult(success=False,failure_reason=c.get('error','no stable contact'),
                        target_count=target,total=6,trace=trace,
                        actions=handoff.partial_actions(c['actions'],c.get('handoff_governor')),
                        final_collected=int(env.collected_mask().sum()),collected=int(env.collected_mask().sum()),
                        peak_force=max((float(r['normal_force']) for r in trace),default=0.),elapsed=float(env.time),
                        sampled_frames=len(c['actions'])+c['act_outer_frames'],scheduler_queries=c['act_queries'],
                        termination_reason=term,first_entry_time=c['collection_timer'].first_entry_time,
                        peak_collected=c['collection_timer'].peak_collected,
                        collection_window_seconds=c['collection_timer'].window_seconds)
                result.initial_object_poses=initial
                result.planner_strategy=f'temporary_astar_gate1mm_wrist{TRIAL_SPEC.force_N:g}N_total2mm_ACT_acc50_handoff2s'
                smoothing_audit=handoff.metrics(result.trace)
                summary={**evidence,'ACT_executed':bool(smoothing_audit['accepted_ACT_control_steps']),
                    'ACT_control_loop_entered':bool(c['act_trace']),
                    'success':result.success,'failure_reason':result.failure_reason,
                    'termination_reason':result.termination_reason,'final_collected':result.final_collected,
                    'elapsed_s':result.elapsed,'sampled_frames':result.sampled_frames,
                    'accepted_ACT_chunks':c.get('accepted_chunks',0),
                    'first_entry_time':result.first_entry_time,
                    'invalid_samples':len(c['invalid_samples']),
                    **wrist_metrics(result.trace,c.get('handoff_time') if c['ready'] else None),
                    'handoff_smoothing_audit':smoothing_audit}
                write(destination/'physical_result.json',summary)
                saved=old.persist_with_backup(destination,result,c['invalid_samples'],
                    lambda:persist_replay_input(cfg,seed,result,destination/'replays',model=model))
                summary['replay_input']=str(saved)
    except BaseException as error:
        primary=error;raise
    finally:
        if env is not None:env.close()
        errors=[];identity=None
        try:
            identity=old.precision.reviewed.base.verify_protected(protected)
            changed=old.precision.protected_changes(extra,protections())
            if changed:errors.append('protected changes: '+repr(changed))
        except Exception as error:errors.append(repr(error))
        restored=UR10CB3EndEffector._solve_ik is native_solver
        if not restored:errors.append('solver not restored')
        write(destination/'physical_verification.json',{'identity':identity,'restored_native_solver':restored,
            'audit_errors':errors,'escaped_exception':repr(primary) if primary else None})
        if errors:
            if primary:primary.add_note(repr(errors))
            else:raise RuntimeError('protected identity audit failed: '+repr(errors))
    gc.collect()
    episode_id=save_inference_replay(cfg,seed,result,destination/'replays',model=model,image_format='webp-lossless')
    if not episode_id:raise RuntimeError('empty media; never rerun physics')
    episode=destination/'replays'/episode_id
    old.hybrid.annotate_replay(episode,old.precision.reviewed.base.json_safe(evidence))
    video=export_inference_video(episode,TRIAL_ROOT/'videos')
    final=video.with_name(f'trial_{TRIAL_NUMBER:02d}_ACT_goal1_force{TRIAL_SPEC.force_N:g}N_total2mm.mp4')
    if final.exists():raise RuntimeError('refuse to overwrite requested video')
    video.rename(final)
    validation=finalize_inference_video(episode,final)
    old.precision.reviewed.base.verify_protected(protected)
    if old.precision.protected_changes(extra,protections()):raise RuntimeError('protected evidence changed during media')
    return finish_video_summary(destination,summary,final,validation,identity)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=int,choices=[1],required=True)
    p.add_argument('--trial',type=int,choices=range(1,7),required=True)
    p.add_argument('--force',type=float,choices=[1.,.3,.2],required=True)
    p.add_argument('--seek',type=float,nargs=3,default=[25.,1200.,1.])
    p.add_argument('--maintain',type=float,nargs=3,default=[25.,2000.,1.])
    args=p.parse_args()
    TRIAL_SPEC=AdmittanceSpec(args.force,tuple(args.seek),tuple(args.maintain))
    TRIAL_NUMBER=args.trial
    run_one(args.target)
