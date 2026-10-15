"""DISPOSABLE diagnostic: expert landing/contact preparation -> unchanged 100K ACT.

No project files are changed. The existing inference function is adapted in
memory only at its initial conditions; prediction, action execution, contact Z
control and collection termination remain the original function body.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path('/Users/huazhi_luke/.codex/worktrees/act-five-view')
BATCH = Path('/Users/huazhi_luke/Documents/Codex/2026-09-08/zhe/exports/batch_fine_teached_100k_120_unique_seeds_20261014')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))


class StableContact:
    def __init__(self, threshold, dt, duration):
        self.threshold = float(threshold)
        self.required = max(1, int(math.ceil(duration / dt - 1e-9)))
        self.samples = 0

    def update(self, force):
        self.samples = (self.samples + 1 if np.isfinite(force) and force >= self.threshold else 0)
        return self.samples >= self.required


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, dict):
        return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(float(value))
    return value


def transactional_step(step_command, snapshot, restore, measure_force, failures):
    before = snapshot()
    invalid = None
    try:
        step_command()
        force = float(measure_force())
        if not np.isfinite(force):
            invalid = str(force)
            raise RuntimeError('nonfinite measured force; retain last valid frame, not fabricated telemetry')
        return force
    except Exception as exc:
        failures.append(safe_json({'error':f'{type(exc).__name__}: {exc}',
            'attempted_state':snapshot(), 'invalid_force':invalid,
            'diagnostic_restore':'last valid integration state; failed command is not a physical replay frame'}))
        restore(before)
        raise


def annotate_replay(episode, evidence):
    path = Path(episode).parent / 'manifest.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    matches = [r for r in rows if r['episode_id'] == Path(episode).name]
    if len(matches) != 1:
        raise RuntimeError('diagnostic replay provenance is ambiguous')
    matches[0].update(planner='hybrid_astar_contact_act',
                      planner_strategy='astar_contact_handoff_act', hybrid_handoff=evidence)
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))


def adapt_function(function, context):
    """Replace only cold-reset initializers, never the policy/control loop."""
    tree = ast.parse(inspect.getsource(function))
    node = tree.body[0]
    values = {
        'env': "_handoff['env']", 'admittance': "_handoff['admittance']",
        'overforce_relief': "_handoff['relief']", 'contact': 'True',
        'z_nominal': "_handoff['z_nominal']", 'last_action': "_handoff['last_reference'].copy()",
        'peak_force': "_handoff['peak_force']", 'sampled_frames': "_handoff['prefix_frames']",
        'contact_time': "_handoff['contact_time']",
        'first_contact_position': "_handoff['first_contact_position'].copy()",
    }
    counts = {key: 0 for key in values}
    counts.update(reset=0, history=0)
    body = []
    for statement in node.body:
        if isinstance(statement, ast.If):
            expanded = []
            for child in statement.body:
                expanded.append(child)
                if (isinstance(child, ast.Assign) and len(child.targets)==1
                        and isinstance(child.targets[0],ast.Name) and child.targets[0].id=='replay_builder'):
                    expanded.extend(ast.parse("replay_builder._history.extend(_handoff['history'])").body)
            statement.body = expanded
        target = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target = statement.target.id
        if target in values:
            statement.value = ast.parse(values[target], mode='eval').body
            counts[target] += 1
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and isinstance(statement.value.func.value, ast.Name)
                and statement.value.func.value.id == 'env' and statement.value.func.attr == 'reset'):
            statement = ast.Pass()
            counts['reset'] += 1
        body.append(statement)
        if target == 'builder':
            body.extend(ast.parse("builder._history.extend(_handoff['history'])").body)
            counts['history'] += 1
    if any(value != 1 for value in counts.values()):
        raise RuntimeError(f'cold-start structure changed; refuse unsafe adapter: {counts}')
    node.body = body
    namespace = dict(function.__globals__)
    namespace['_handoff'] = context
    exec(compile(ast.fix_missing_locations(tree), '<temporary_contact_handoff>', 'exec'), namespace)
    adapted = namespace[function.__name__]
    adapted.probe_initializer_changes = counts
    return adapted


def prepare_contact(env, cfg, plan):
    import mujoco
    from sim.controllers.hybrid import Command
    from sim.act.interface import ACTObservationBuilder
    from sim.act.evaluate import _policy_reference_substep, _contact_z_command
    from sim.act.rollout import _expert_execution_path, _force_loop, _overforce_relief_loop

    path, phases = _expert_execution_path(cfg, plan, table_top_z=env.table_top_z)
    builder = ACTObservationBuilder(cfg)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(env.model, spec)
    dt = 1. / float(cfg.sim.control_hz)
    period = int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz)))
    gate = StableContact(float(cfg.controller.contact_threshold), dt, .1)
    reference = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    context = {'env': env, 'trace': [], 'actions': [], 'peak_force': 0., 'step_failures': [],
               'contact_time': None, 'first_contact_position': None,
               'admittance': _force_loop(cfg), 'relief': _overforce_relief_loop(cfg)}
    control_steps = 0
    overforce_steps = 0

    def snapshot():
        values = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(env.model,env.data,values,spec)
        return values

    def restore(values):
        mujoco.mj_setState(env.model,env.data,values,spec)
        mujoco.mj_forward(env.model,env.data)
        mujoco.mj_rnePostConstraint(env.model,env.data)

    def step(command, phase, contact=False):
        nonlocal reference, control_steps, overforce_steps
        force = transactional_step(
            lambda: env.step_control(Command(*[float(v) for v in command])),
            snapshot,restore,env.normal_force,context['step_failures'])
        reference = np.asarray(command, dtype=np.float32).copy()
        control_steps += 1
        context['peak_force'] = max(context['peak_force'], force)
        overforce_steps = overforce_steps + 1 if force > float(cfg.controller.safe_max_force) else 0
        state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, state, spec)
        row = {'t': float(env.time), 'tcp': env.tcp().copy(), 'command': reference.copy(),
               'policy_reference': reference.copy(), 'policy_z': float(reference[2]),
               'applied_z': float(reference[2]), 'z_owner': 'expert_contact_preparation',
               'normal_force': force, 'contact': bool(contact), 'collected': int(env.collected_mask().sum()),
               'phase': phase, 'supervisor_owned': True,
               'replay_state': {'spec': int(spec), 'values': state},
               'object_collected': env.collected_mask().copy(), 'wrench': env.wrench().copy(),
               'raw_normal_force': float(env.raw_normal_force()),
               'compensated_normal_force_raw': float(env.compensated_normal_force_raw()),
               'tcp_pose': np.array([*env.tcp(), env.ee.tcp_yaw()]),
               'object_pose': np.concatenate((env.component_positions(), env.component_quats()), axis=1),
               'replay_observation': builder.observe(env, contact_latched=contact, include_images=False)}
        context['trace'].append(row)
        if control_steps % period == 0:
            context['actions'].append(reference.copy())
        if not np.isfinite(force) or overforce_steps >= int(cfg.controller.safe_force_dwell_steps):
            raise RuntimeError('expert prefix exceeded the unchanged measured-force safety gate')
        return force

    try:
        # Only fly to the first 7 cm dynamic-boundary waypoint, then descend.
        # No expert sweep or physical candidate screening is executed.
        for action, phase in zip(path, phases):
            if phase not in ('approach', 'descent'):
                break
            start = reference.copy()
            for substep in range(1, period + 1):
                command = _policy_reference_substep(start, action, substep, period, dt,
                    float(cfg.controller.travel_speed), float(cfg.controller.z_speed),
                    float(cfg.controller.yaw_speed_limit), False)
                step(command, 'expert_' + phase)
        search_started = float(env.time)
        while True:
            for _ in range(period):
                force = float(env.normal_force())
                command = reference.copy()
                if force < gate.threshold:
                    command[2] = max(float(cfg.workspace.z_search_min),
                                     float(command[2]) - float(cfg.controller.search_velocity) * dt)
                force = step(command, 'expert_force_search')
                gate.update(force)
            if gate.samples >= gate.required:
                break
            if float(reference[2]) <= float(cfg.workspace.z_search_min) or env.time - search_started >= 3.:
                raise RuntimeError('no stable measured table contact before unchanged search-depth/time limit')
        context['contact_time'] = float(env.time)
        context['first_contact_position'] = env.tcp().copy()
        context['z_nominal'] = float(env.tcp()[2])
        context['admittance'].reset()
        context['relief'].reset()
        steady = StableContact(float(cfg.controller.contact_threshold), dt, .1)
        build_steps = int(round(float(cfg.episode.contact_build_time) * float(cfg.sim.control_hz)))
        for _ in range(build_steps):
            command = reference.copy()
            command[2] = _contact_z_command(cfg, context['admittance'], context['z_nominal'],
                                          float(env.normal_force()), relief=context['relief'])
            steady.update(step(command, 'expert_contact_build', True))
        if steady.samples < steady.required:
            raise RuntimeError('measured contact was not stable at the end of the one-second build')
        context['handoff_time'] = float(env.time)
        context['handoff_force'] = float(env.normal_force())
        context['last_reference'] = reference.copy()
        context['prefix_frames'] = len(context['actions'])
        context['history'] = [(t, a.copy()) for t, a in builder._history]
        context['ready'] = True
    except Exception as exc:
        context['ready'] = False
        context['error'] = f'{type(exc).__name__}: {exc}'
    return context


def run_one(target, seed):
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.act.collect_dataset import _expert_config
    from sim.act.expert import expert_plan_candidates
    from sim.act import evaluate
    from sim.act.rollout import ActRolloutResult
    from sim.act.replay import persist_replay_input, save_inference_replay
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video
    from sim.act.inference_lock import inference_lease

    destination = ROOT / f'goal_{target}'
    destination.mkdir()
    cfg = _expert_config(load_config(str(BATCH / 'evaluation_config.yaml')), target)
    metadata = json.loads((BATCH / 'metadata.json').read_text())
    actual = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (PROJECT / 'sim').rglob('*.py')}
    if actual != metadata['source_fingerprints']:
        raise RuntimeError('frozen simulation source identity mismatch')
    model = metadata['models'][0]['path']
    if metadata['models'][0]['step'] != 100000:
        raise RuntimeError('this experiment only uses the explicitly frozen 100K model')
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('insufficient free disk for two diagnostic videos')
    print(json.dumps({'event': 'waiting_for_inference_slot', 'target': target}), flush=True)
    with inference_lease():
        env = SweepEnv(cfg, seed=seed)
        env.reset(seed=seed)
        initial = np.concatenate((env.component_positions(), env.component_quats()), axis=1).astype('<f4')
        candidates = expert_plan_candidates(env, cfg, max_candidates=1)
        if not candidates or not candidates[0].feasible:
            env.close()
            raise RuntimeError('no feasible single A* landing plan; no replacement trial is permitted')
        plan = candidates[0]
        context = prepare_contact(env, cfg, plan)
        evidence = {'method': 'temporary_astar_measured_contact_handoff_act', 'target': target, 'seed': seed,
                    'checkpoint': model, 'prefix_ready': context['ready'],
                    'landing_point': np.asarray(plan.waypoints[1]).tolist(),
                    'landing_region': plan.landing_region,
                    'prefix_frames': len(context['actions']), 'prefix_peak_force': context['peak_force'],
                    'handoff_time': context.get('handoff_time'), 'handoff_force': context.get('handoff_force'),
                    'contact_time': context['contact_time'], 'prefix_error': context.get('error', ''),
                    'failed_control_attempts': context['step_failures'],
                    'initial_layout_fingerprint': hashlib.sha256(initial.tobytes()).hexdigest()}
        (destination / 'handoff.json').write_text(json.dumps(evidence, indent=2, default=lambda x:np.asarray(x).tolist()))
        print(json.dumps({'event': 'contact_preparation', **evidence}, default=lambda x:np.asarray(x).tolist()), flush=True)
        if context['ready']:
            adapted = adapt_function(evaluate.run_act_episode, context)
            result = adapted(cfg, seed=seed, model_path=model, preview=True, capture_replay=True)
            result.trace = context['trace'] + result.trace
            result.actions = np.concatenate((np.asarray(context['actions']), result.actions), axis=0)
            result.observations.clear()
        else:
            result = ActRolloutResult(success=False, failure_reason=context['error'], total=6,
                target_count=target, trace=context['trace'], actions=np.asarray(context['actions']),
                peak_force=context['peak_force'], elapsed=float(env.time), collected=int(env.collected_mask().sum()))
            env.close()
        result.initial_object_poses = initial
        result.planner_strategy = 'astar_contact_handoff_act'
        saved = persist_replay_input(cfg, seed, result, destination / 'replays', model=model)
        summary = {**evidence, 'success': result.success, 'reason': result.failure_reason,
                   'final_collected': result.final_collected if result.final_collected is not None else result.collected,
                   'peak_force': result.peak_force, 'replay_input': str(saved)}
        (destination / 'result.json').write_text(json.dumps(summary, indent=2, default=lambda x:np.asarray(x).tolist()))
        print(json.dumps({'event': 'physical_result_saved', **summary}, default=lambda x:np.asarray(x).tolist()), flush=True)
    # Rendering outside the inference slot cannot acquire or load ACT weights.
    episode = save_inference_replay(cfg, seed, result, destination / 'replays', model=model,
                                    image_format='webp-lossless')
    annotate_replay(destination / 'replays' / episode, evidence)
    video = export_inference_video(destination / 'replays' / episode, ROOT / 'videos')
    final_video = video.with_name(f'hybrid_astar_contact_act_goal_{target}_step_100000.mp4')
    if final_video.exists():
        raise RuntimeError('refuse to replace an existing diagnostic video')
    video.rename(final_video)
    validation = finalize_inference_video(destination / 'replays' / episode, final_video)
    summary.update(video=str(final_video), media_validation=validation)
    (destination / 'final.json').write_text(json.dumps(summary, indent=2, default=lambda x:np.asarray(x).tolist()))
    print(json.dumps({'event': 'video_ready', 'target': target, 'video': str(final_video),
                      'success': result.success, 'final_collected': summary['final_collected']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, choices=(1, 6), required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    run_one(args.target, args.seed)
