"""THROWAWAY: one A* landing prefix, then simultaneous 3-mm ACT/new-Z handoff.

No production file, model, formal result, default or frozen config is edited.
Only the copied ACT initializer/telemetry/safety guard and process-local IK
threshold differ. The ACT predictor, scheduler and collection contract remain.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import gzip
import hashlib
import importlib.util
import inspect
import json
import linecache
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[4]
BATCH = ROOT.parent / 'batch_fine_teached_100k_120_unique_seeds_20261014'
sys.path.insert(0, str(PROJECT))


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


precision = load_helper('approved_ik_helper', ROOT.parent /
                        'admittance_ik_precision_comparison_20261014/precision_probe.py')
hybrid = load_helper('reviewed_hybrid_helper', ROOT.parent /
                     'contact_handoff_probe_20261014/probe.py')
legacy_runner = load_helper('reviewed_metrics_helper', ROOT.parent /
                            'act_geometry3mm_admittance5mm_k20_trial_20261014/runner.py')
geometry = legacy_runner.adapter
SEEDS = {1: 2120018, 6: 2120029}
EXPECTED_LAYOUTS = {1: '3c65aaa843635cc71cf176499aff8c04e834066cac7dcdf38ed176e3c6a7339f',
                    6: 'a19adb6e7a1580669ef108c4cb4326bceaef05657172ca1cf7e70950747b27b3'}


def geometry_ready(tcp_z, table_z):
    return bool(np.isfinite([tcp_z, table_z]).all() and tcp_z <= table_z + .003)


def validate_config(frozen, actual, target):
    base = frozen.copy()
    base.set_path('task.target_count', target)
    return precision.validate_effective_config(base, actual)


def make_config(frozen, target):
    if target not in SEEDS:
        raise ValueError('only the two approved fixed goal/layout pairs are allowed')
    cfg = precision.reviewed.base.make_config(frozen, dict(precision.PARAMETERS))
    cfg.set_path('task.target_count', target)
    validate_config(frozen, cfg, target)
    return cfg


def write(path, value):
    precision.reviewed.base.write(Path(path), value)


def measure(env):
    table_force, table_contact, part_contacts = precision.reviewed.base.table_measurement(env)
    sensor = {
        'trial_table_normal_force': table_force,
        'trial_brush_table_contact': table_contact,
        'trial_part_contact_count': part_contacts,
        'trial_raw_wrist_substep_peak': float(env.last_raw_force_peak or 0.),
        'trial_wrist_world': env.ee.wrist_ft_world().copy(),
        'trial_compensated_wrist_world': env.ee.compensated_wrench_world().copy(),
        'trial_partial_overlap_mask': env.partial_collection_overlap_mask().copy(),
    }
    state = np.r_[env.data.qpos, env.data.qvel, env.data.sensordata,
                  sensor['trial_wrist_world'], sensor['trial_compensated_wrist_world'],
                  env.normal_force(), env.raw_normal_force(), table_force]
    if not np.isfinite(state).all():
        raise RuntimeError('nonfinite physical sensor/state; terminate temporary trial')
    if env.unsafe_robot_collision():
        raise RuntimeError('unsafe robot collision; terminate temporary trial')
    return sensor


def observe_control_row(context, row):
    """One safety/collection observation per genuinely advanced physics time."""
    now, force = float(row['t']), float(row['normal_force'])
    if not np.isfinite([now, force]).all():
        context['safety_reason'] = 'temporary hybrid safety: force ceiling or nonfinite force'
        return
    previous = context['last_guard_time']
    if previous is not None and now <= previous + 1e-10:
        return
    context['last_guard_time'] = now
    if context['force_gate'].update(force):
        context['safety_reason'] = 'temporary hybrid safety: force ceiling or nonfinite force'
    context['collection_reason'] = context['collection_timer'].update(
        row['object_collected'], now,
        partial_overlap_mask=row.get('partial_overlap_mask',row.get('trial_partial_overlap_mask')))


def record_row(context, row, env, measurement, *, prefix=False):
    try:
        row.update(measurement(env))
    except Exception:
        context['invalid_samples'].append(copy.deepcopy(row))
        raise
    observe_control_row(context,row)
    context['trace' if prefix else 'act_trace'].append(copy.deepcopy(row))


def adapt_act(function, context, measure=measure):
    """Reuse reviewed initializers after narrowly instrumenting a source copy."""
    tree = ast.parse(inspect.getsource(function))
    counts = {'capture': 0, 'gate': 0, 'guard': 0, 'tracker':0,
              'continue':0,'frames':0,'queries':0,'accepted':0}

    class Instrument(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == 'capture_state':
                node.body.extend(ast.parse('_trial_record(row, env)').body)
                counts['capture'] += 1
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == 'overforce_relief':
                    counts['gate'] += 1
                    return [node, *ast.parse('_trial_gate = _trial_shared_force_gate').body]
                if name == 'collection_timer':
                    counts['tracker'] += 1
                    node.value = ast.parse('_trial_shared_collection_timer',mode='eval').body
                if name == 'accepted':
                    counts['accepted'] += 1
                    return [node,*ast.parse('_trial_chunk_ack(accepted)').body]
                if (name == 'termination_reason' and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == 'update'):
                    counts['guard'] += 1
                    node.value = ast.parse('_trial_terminal_reason()',mode='eval').body
            return node

        def visit_Continue(self,node):
            counts['continue'] += 1
            return [*ast.parse('if _trial_terminal_reason():\n    termination_reason = _trial_terminal_reason()\n    break').body,node]

        def visit_AugAssign(self,node):
            if isinstance(node.target,ast.Name) and node.target.id in ('sampled_frames','scheduler_queries'):
                kind = 'frames' if node.target.id=='sampled_frames' else 'queries'
                counts[kind] += 1
                return [node,*ast.parse(f'_trial_count("{kind}")').body]
            return node

    tree = Instrument().visit(tree)
    if counts != {'capture':1,'gate':1,'guard':1,'tracker':1,'continue':1,'frames':1,'queries':1,'accepted':1}:
        raise RuntimeError(f'native instrumentation structure changed: {counts}')

    def record(row, env):
        record_row(context,row,env,measure)

    def terminal_reason():
        return context['safety_reason'] or context['collection_reason']

    def count(kind):
        context['act_outer_frames' if kind=='frames' else 'act_queries'] += 1

    def chunk_ack(accepted):
        context['accepted_chunks'] = context.get('accepted_chunks',0) + int(bool(accepted))

    namespace = dict(function.__globals__)
    namespace.update(_trial_record=record,_trial_terminal_reason=terminal_reason,_trial_count=count,_trial_chunk_ack=chunk_ack,
                     _trial_shared_force_gate=context['force_gate'],
                     _trial_shared_collection_timer=context['collection_timer'])
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    filename = f'<throwaway_hybrid_instrument_{id(context)}>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), namespace)
    return hybrid.adapt_function(namespace[function.__name__], context)


def validate_prefix_budget(phases):
    count=sum(phase in ('approach','descent') for phase in phases)
    if count>=2000:
        raise RuntimeError('planned expert prefix exhausts shared 2000-action budget')
    return count


def prepare_prefix(env, cfg, plan):
    import mujoco
    from sim.controllers.hybrid import Command
    from sim.act.interface import ACTObservationBuilder
    from sim.act.evaluate import _policy_reference_substep
    from sim.act.rollout import _expert_execution_path, _force_loop, _overforce_relief_loop, CollectionTerminationTracker
    path, phases = _expert_execution_path(cfg, plan, table_top_z=env.table_top_z)
    validate_prefix_budget(phases)
    dt = 1 / float(cfg.sim.control_hz)
    period = int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz)))
    builder = ACTObservationBuilder(cfg)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(env.model, state_spec)
    reference = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    gate = geometry.ForceGate(cfg.controller.safe_max_force, cfg.controller.safe_force_dwell_steps)
    context = {'env': env, 'admittance': _force_loop(cfg), 'relief': _overforce_relief_loop(cfg),
               'trace': [], 'actions': [], 'act_trace': [], 'invalid_samples': [],
               'step_failures': [], 'peak_force': 0., 'ready': False,'force_gate':gate,
               'collection_timer':CollectionTerminationTracker(target_count=int(cfg.task.target_count)),
               'last_guard_time':None,'safety_reason':'','collection_reason':'',
               'act_outer_frames':0,'act_queries':0,'accepted_chunks':0}
    steps = 0

    def snapshot():
        state = np.empty(state_size)
        mujoco.mj_getState(env.model, env.data, state, state_spec)
        return state

    def restore(state):
        mujoco.mj_setState(env.model, env.data, state, state_spec)
        mujoco.mj_forward(env.model, env.data)
        mujoco.mj_rnePostConstraint(env.model, env.data)

    try:
        for action, phase in zip(path, phases):
            if phase not in ('approach', 'descent'):
                break
            start = reference.copy()
            for substep in range(1, period + 1):
                command = _policy_reference_substep(start, action, substep, period, dt,
                    float(cfg.controller.travel_speed), float(cfg.controller.z_speed),
                    float(cfg.controller.yaw_speed_limit), False)
                force = hybrid.transactional_step(lambda: env.step_control(Command(*map(float, command))),
                    snapshot, restore, env.normal_force, context['step_failures'])
                reference = command.copy()
                steps += 1
                row = {'t': float(env.time), 'tcp': env.tcp().copy(), 'command': command.copy(),
                       'policy_reference': command.copy(), 'policy_z': float(command[2]),
                       'applied_z': float(command[2]), 'z_owner': 'Astar_prefix',
                       'normal_force': force, 'contact': False, 'phase': 'expert_' + phase,
                       'supervisor_owned': True, 'collected': int(env.collected_mask().sum()),
                       'replay_state': {'spec': int(state_spec), 'values': snapshot()},
                       'object_collected': env.collected_mask().copy(), 'wrench': env.wrench().copy(),
                       'raw_normal_force': float(env.raw_normal_force()),
                       'compensated_normal_force_raw': float(env.compensated_normal_force_raw()),
                       'tcp_pose': np.r_[env.tcp(), env.ee.tcp_yaw()],
                       'object_pose': np.c_[env.component_positions(), env.component_quats()],
                       'replay_observation': builder.observe(env, contact_latched=False, include_images=False)}
                record_row(context,row,env,measure,prefix=True)
                context['peak_force'] = max(context['peak_force'], force)
                if steps % period == 0:
                    context['actions'].append(reference.copy())
                    builder.advance_state_history(env, contact_latched=False)
                if context['safety_reason'] or context['collection_reason']:
                    context['terminal_reason'] = context['safety_reason'] or context['collection_reason']
                    raise RuntimeError('prefix terminated before ACT handoff: '+context['terminal_reason'])
                if phase == 'descent' and geometry_ready(float(env.tcp()[2]), float(env.table_top_z)):
                    if steps % period:
                        context['actions'].append(reference.copy())
                        builder.advance_state_history(env, contact_latched=True)
                    context.update(ready=True, handoff_time=float(env.time),
                        handoff_force=float(env.normal_force()), handoff_tcp=env.tcp().copy(),
                        handoff_actual_table_contact=bool(env.brush_table_contact()),
                        contact_time=float(env.time), first_contact_position=env.tcp().copy(),
                        z_nominal=float(env.tcp()[2]), last_reference=reference.copy(),
                        prefix_frames=len(context['actions']),
                        history=[(t, a.copy()) for t, a in builder._history])
                    context['admittance'].reset()
                    context['relief'].reset()
                    return context
        raise RuntimeError('expert descent ended without reaching 3 mm; no force search/retry permitted')
    except Exception as error:
        context['error'] = repr(error)
        return context


def extra_protection():
    paths = precision.extra_protection()
    for path in [Path(__file__), Path(__file__).with_name('test_trial.py'),
                 Path(hybrid.__file__)]:
        paths[str(path)] = precision.digest(path)
    return paths


def persist_with_backup(destination, result, invalid_samples, native_persist):
    """Durably retain every valid and rejected sample before strict persistence."""
    with gzip.open(Path(destination)/'physical_trace_backup.json.gz','xt',compresslevel=3) as stream:
        payload={'trace':result.trace,'actions':result.actions,'invalid_samples':invalid_samples,
                 'result_fields':{k:v for k,v in vars(result).items()
                                  if k not in ('trace','actions','observations')}}
        json.dump(precision.reviewed.base.json_safe(payload),stream,allow_nan=False)
    return native_persist()


def run_one(target):
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.environments.ur10_interface import UR10CB3EndEffector
    from sim.act.expert import expert_plan_candidates
    from sim.act import evaluate
    from sim.act.rollout import ActRolloutResult
    from sim.act.inference_lock import inference_lease
    from sim.act.replay import persist_replay_input, save_inference_replay
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video
    destination = ROOT / f'goal_{target}'
    destination.mkdir()  # immutable attempt marker: never replace a failed trial
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('free disk below 6 GiB; refuse trial')
    protected = precision.reviewed.base.protected_snapshot()
    extra = extra_protection()
    write(destination / 'before.json', {'protected': protected, 'extra_sha256': extra})
    frozen = load_config(str(BATCH / 'evaluation_config.yaml'))
    cfg = make_config(frozen, target)
    write(destination / 'configuration.json', cfg.to_dict())
    model = protected['checkpoint_path']
    seed = SEEDS[target]
    native_solver = UR10CB3EndEffector._solve_ik
    env = None
    context = None
    result = None
    primary = None
    print(json.dumps({'event':'waiting_for_inference_slot','target':target,'seed':seed}), flush=True)
    try:
        with inference_lease(on_wait=lambda: print(json.dumps({'event':'slot_busy'}), flush=True)):
            precision.reviewed.base.verify_protected(protected)
            if precision.protected_changes(extra, extra_protection()):
                raise RuntimeError('protected inputs changed while waiting')
            with precision.temporary_solver(UR10CB3EndEffector, 1e-6):
                env = SweepEnv(cfg, seed=seed)
                env.record_raw_force_peaks = True
                env.reset(seed=seed)
                initial = np.c_[env.component_positions(), env.component_quats()].astype('<f4')
                fingerprint = hashlib.sha256(initial.tobytes()).hexdigest()
                write(destination / 'initial_layout.json', {'poses':initial,'sha256':fingerprint})
                if fingerprint != EXPECTED_LAYOUTS[target]:
                    raise RuntimeError('fixed layout mismatch; refuse replacement trial')
                candidates = expert_plan_candidates(env, cfg, max_candidates=1)
                if not candidates or not candidates[0].feasible:
                    raise RuntimeError('no feasible single A* plan; no replacement layout permitted')
                plan = candidates[0]
                write(destination / 'planning.json', {'planning_calls':1, 'physical_candidate_screening':False,
                    'strategy':plan.strategy,'waypoints':plan.waypoints,'landing_region':plan.landing_region,
                    'selected_indices':plan.target_indices})
                context = prepare_prefix(env, cfg, plan)
                evidence = {k:context.get(k) for k in ['ready','handoff_time','handoff_force','handoff_tcp',
                    'handoff_actual_table_contact','prefix_frames','z_nominal','error','step_failures']}
                evidence.update(target=target, seed=seed, checkpoint=model, geometry_gate_m=.003,
                    parameters=precision.PARAMETERS, ik_threshold=1e-6,
                    simultaneous_ACT_and_admittance=True, waited_for_force_contact=False,
                    extra_stabilization_seconds=0., correction_limit_m=.005, desired_force_N=.1,
                    contact_flag_semantics='geometric Z ownership, not measured table contact',
                    initial_layout_fingerprint=fingerprint, no_retries=True, formal_evaluation=False)
                write(destination / 'handoff.json', evidence)
                print(json.dumps(precision.reviewed.base.json_safe({'event':'handoff','target':target,**evidence})),flush=True)
                if context['ready']:
                    try:
                        result = adapt_act(evaluate.run_act_episode, context)(cfg, seed=seed,
                            model_path=model, preview=True, capture_replay=True)
                        result.trace = context['trace'] + result.trace
                        result.actions = np.concatenate((np.asarray(context['actions']),result.actions),axis=0)
                        result.observations.clear()
                    except Exception as error:
                        context['error'] = repr(error)
                        write(destination / 'ACT_exception.json', {'error':repr(error),
                            'traceback':traceback.format_exc(),'invalid_samples':context['invalid_samples']})
                if result is None:
                    rows = context['trace'] + context['act_trace']
                    finite_forces=[float(row['normal_force']) for row in rows if np.isfinite(row['normal_force'])]
                    terminal=context.get('terminal_reason','')
                    result = ActRolloutResult(success=terminal=='exact target count reached', failure_reason=context.get('error','prefix failed'),
                        target_count=target,total=6,trace=rows,
                        actions=np.asarray([row['policy_reference'] for row in rows[::4]],dtype=np.float32),
                        final_collected=int(env.collected_mask().sum()),collected=int(env.collected_mask().sum()),
                        peak_force=max(finite_forces,default=0.),elapsed=float(env.time),
                        sampled_frames=len(context['actions'])+context['act_outer_frames'],
                        scheduler_queries=context['act_queries'],termination_reason=terminal,
                        first_entry_time=context['collection_timer'].first_entry_time,
                        peak_collected=context['collection_timer'].peak_collected,
                        collection_window_seconds=context['collection_timer'].window_seconds)
                    env.close()
                result.initial_object_poses = initial
                result.planner_strategy = 'temporary_astar_3mm_ACT_new_admittance'
                metrics = legacy_runner.contact_metrics(result.trace,dt=1/float(cfg.sim.control_hz))
                summary = {**evidence,'ACT_handoff_attempted':bool(context['ready']),
                    'ACT_executed':bool(context['act_trace']), 'success':result.success,
                    'failure_reason':result.failure_reason,'termination_reason':result.termination_reason,
                    'final_collected':result.final_collected,'elapsed_s':result.elapsed,
                    'sampled_frames':result.sampled_frames,'scheduler_queries':result.scheduler_queries,
                    'accepted_ACT_chunks':context.get('accepted_chunks',0),
                    'scheduler_timeouts':result.scheduler_timeouts if not context.get('error') else None,
                    'first_entry_time':result.first_entry_time,'peak_collected':result.peak_collected,
                    'force_dwell_preserved_across_handoff':True,
                    'collection_tracker_preserved_across_handoff':True,
                    'invalid_physical_samples_retained':len(context['invalid_samples']),**metrics}
                write(destination / 'physical_result.json', summary)
                saved = persist_with_backup(destination,result,context['invalid_samples'],
                    lambda:persist_replay_input(cfg,seed,result,destination/'replays',model=model))
                summary['replay_input'] = str(saved)
    except BaseException as error:
        primary = error
        raise
    finally:
        if env is not None:
            env.close()
        audit_errors = []
        identity = None
        try:
            identity = precision.reviewed.base.verify_protected(protected)
            changed = precision.protected_changes(extra,extra_protection())
            if changed:
                audit_errors.append('protected changes: '+repr(changed))
        except Exception as error:
            audit_errors.append(repr(error))
        restored = UR10CB3EndEffector._solve_ik is native_solver
        if not restored:
            audit_errors.append('native process-local solver did not restore')
        write(destination/'physical_verification.json',{'identity':identity,'restored_native_solver':restored,
            'audit_errors':audit_errors,'escaped_exception':repr(primary) if primary else None})
        if audit_errors:
            if primary:
                primary.add_note(repr(audit_errors))
            else:
                raise RuntimeError('identity audit failed: '+repr(audit_errors))
    gc.collect()
    episode_id = save_inference_replay(cfg,seed,result,destination/'replays',model=model,image_format='webp-lossless')
    if not episode_id:
        raise RuntimeError('no replay frames; retain evidence, never rerun physics')
    episode = destination/'replays'/episode_id
    hybrid.annotate_replay(episode, evidence)
    video = export_inference_video(episode,ROOT/'videos')
    final_video = video.with_name(f'hybrid_Astar_ACT_new_admittance_3mm_goal_{target}_step_100000.mp4')
    if final_video.exists():
        raise RuntimeError('refuse to overwrite diagnostic video')
    video.rename(final_video)
    validation = finalize_inference_video(episode,final_video)
    identity = precision.reviewed.base.verify_protected(protected)
    changed = precision.protected_changes(extra,extra_protection())
    if changed:
        raise RuntimeError('formal evidence changed during rendering: '+repr(changed))
    summary.update(video=str(final_video),media_validation=validation,final_identity_verification=identity)
    write(destination/'final.json',summary)
    print(json.dumps({'event':'video_ready','target':target,'video':str(final_video),
        'success':result.success,'final_collected':result.final_collected}),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=int,choices=(1,6),required=True)
    args = parser.parse_args()
    run_one(args.target)
