"""Portable, fail-closed trial-4 A* → frozen-DINOv3 ACT inference.

The legacy helpers are source-preserving ports of the independent Hybrid
snapshot. Only their module-level machine paths changed. Historical probe
entry points are not called: this module owns checkpoint selection and the
single-plan/single-execution public entry.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
from collections.abc import Mapping

import numpy as np

from . import formal

_ROOT = Path(__file__).resolve().parent
_IMPORT_LOCK = threading.RLock()
PROFILE = json.loads((_ROOT / 'control_profile.json').read_text())


def runtime():
    """Load sealed helpers without depending on cwd or polluting generic imports."""
    def load(relative, name):
        spec = importlib.util.spec_from_file_location(name, _ROOT / 'legacy' / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses need a registered owning module
        spec.loader.exec_module(module)
        return module

    with _IMPORT_LOCK:
        aliases = ('runner', 'bench', 'bench_v2', 'phases')
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in aliases}
        previous_path = sys.path.copy()
        try:
            sys.modules['runner'] = load(
                'act_geometry3mm_admittance5mm_k20_trial_20261014/runner.py',
                '_portable_hybrid_base_runner')
            sys.modules['bench'] = load(
                'admittance_fixed_xy_tuning_20261014/bench.py', '_portable_hybrid_bench')
            sys.modules['bench_v2'] = load(
                'admittance_fixed_xy_tuning_20261014/bench_v2.py', '_portable_hybrid_bench_v2')
            phase = load('admittance_2mm_target_gain_search_goal1_20261014/phases.py',
                         '_portable_hybrid_phases')
            sys.modules['phases'] = phase
            module = load('admittance_2mm_target_gain_search_goal1_20261014/runner.py',
                          '_portable_hybrid_runtime')
            module.TRIAL_SPEC = phase.AdmittanceSpec(1., (10., 400., 10.), (10., 800., 10.))
            return module
        finally:
            sys.path[:] = previous_path
            for name, value in previous.items():
                if value is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


def require_dinov3_checkpoint(model_path):
    """No random policy, implicit old ResNet, or checkpoint conversion at handoff."""
    from ..evaluate import _resolve_model_path

    resolved = _resolve_model_path(str(model_path)) if model_path else None
    if not resolved or not (Path(resolved) / 'config.json').is_file():
        raise FileNotFoundError('Hybrid DINO inference requires a trained act_dinov3 checkpoint; '
                                'prepare/train it on the other computer first')
    path = Path(resolved).resolve()
    config = json.loads((path / 'config.json').read_text())
    if config.get('type') != 'act_dinov3' or config.get('vision_backbone') != 'dinov3_vits16':
        raise ValueError('Hybrid DINO requires act_dinov3 / dinov3_vits16, not a ResNet checkpoint')
    if not (path / 'model.safetensors').is_file():
        raise FileNotFoundError(f'incomplete ACT checkpoint: {path}')
    return str(path)


def configure_admittance(module, cfg):
    """Apply explicit contact gains before creating the continuous Z loop.

    Omission retains the archived trial-4 profile. Configured runs report the
    actual gains and keep the same wrist feedback, gate, Z bounds and XY path.
    """
    profile = dict(PROFILE)
    from dataclasses import asdict
    from ..evaluate import _overforce_relief_loop
    from .contact_motion import configure as configure_motion
    relief = asdict(_overforce_relief_loop(cfg).params)
    base = getattr(module, '_unconfigured_admittance_type', module.TwoStageAdmittance)
    module._unconfigured_admittance_type = base
    module.TwoStageAdmittance = base
    values = cfg.get_path('act.hybrid_admittance')
    if values is None:
        module.TRIAL_SPEC = type(module.TRIAL_SPEC)(PROFILE['target_force_N'],
            tuple(PROFILE['seek_MBK']), tuple(PROFILE['maintain_MBK']))
        archived_relief = dict(activation_force=10., release_force=7., upward_rate=.02,
                               return_rate=.1, max_offset=.00025)
        if cfg.get_path('act.hybrid_sweep_smoothing') is not None or relief != archived_relief:
            profile['overforce_relief'] = relief
        return configure_motion(module, cfg, profile)
    allowed = {'seek_MBK', 'maintain_MBK', 'force_compliance_m_per_N'}
    if not isinstance(values, Mapping) or set(values) - allowed:
        raise ValueError('unknown hybrid_admittance parameter')
    try:
        compliance = float(values.get('force_compliance_m_per_N', 0.))
        if not np.isfinite(compliance) or not 0. <= compliance <= .0003:
            raise ValueError('outside 0 to 0.3 mm/N')
    except (TypeError, ValueError) as error:
        raise ValueError(f'invalid force compliance: {error}') from error
    try:
        spec = type(module.TRIAL_SPEC)(
            PROFILE['target_force_N'],
            tuple(values.get('seek_MBK', PROFILE['seek_MBK'])),
            tuple(values.get('maintain_MBK', PROFILE['maintain_MBK'])))
        spec.check_clock(1. / float(cfg.sim.control_hz))
    except (TypeError, ValueError) as error:
        raise ValueError(f'invalid configured Hybrid admittance: {error}') from error
    module.TRIAL_SPEC = spec
    if compliance:
        class ContactCompliantAdmittance(base):
            """Bounded elastic wrist feedback alongside the continuous M/B/K loop.

            Acquisition and its gain-switch state remain intact. Once contact
            is maintained, this term reacts before the slower force integral
            builds pressure. It releases immediately as pressure falls so the
            brush can recover contact. The existing limiter owns actual Z rate.
            """
            def step(self, desired_force, wrist_force):
                correction = super().step(desired_force, wrist_force)
                if self.phase == 'maintain':
                    elastic = np.clip(compliance * (wrist_force - desired_force),
                                      -.0005, .0005)
                    correction = float(np.clip(correction + elastic, -.002, .002))
                return correction

        module.TwoStageAdmittance = ContactCompliantAdmittance
        profile['maintain_force_compliance'] = {
            'gain_m_per_N': compliance, 'offset_cap_m': .0005,
            'active_phase': 'maintain', 'feedback': 'compensated_filtered_wrist_Fz'}
    profile.update(seek_MBK=list(spec.seek), maintain_MBK=list(spec.maintain),
                   admittance_configuration='act.hybrid_admittance',
                   overforce_relief=relief)
    return configure_motion(module, cfg, profile)


def _audit_result(module, context, result):
    if context['step_failures'] or context['invalid_samples']:
        raise RuntimeError('invalid Hybrid execution; no fallback or retry permitted')
    limits = getattr(module, '_contact_sweep_limits', None)
    if limits is not None:
        governor = context.get('handoff_governor')
        for row in result.trace:
            p = row.get('handoff_smoothing', {})
            if governor is not None and governor.started_at is not None:
                proposal_time = float(row['t']) - governor.dt
                due = (row.get('control_step_advanced', True)
                       and proposal_time >= governor.started_at + 2. - 1e-8)
                elapsed = p.get('elapsed_s')
                if due and (elapsed is None or not np.isfinite(elapsed)
                            or abs(elapsed-(proposal_time-governor.started_at)) > 1e-8):
                    raise RuntimeError('invalid continuous contact sweep elapsed time')
                if due and (not p.get('advanced') or p.get('started_at') != governor.started_at):
                    raise RuntimeError('missing continuous contact sweep telemetry')
            if not p.get('advanced') or p.get('started_at') is None:
                continue
            elapsed = p.get('elapsed_s')
            if elapsed is None or not np.isfinite(elapsed):
                raise RuntimeError('invalid continuous contact sweep clock telemetry')
            if elapsed >= 2.-1e-9:
                if (p.get('mode') != 'contact_sweep' or not p.get('acceleration_bounds_apply')
                        or any(p.get(k) != v for k, v in limits.items())
                        or not np.isfinite(np.r_[p.get('velocity_xy_m_s', np.nan),
                            p.get('acceleration_xy_m_s2', np.nan),
                            p.get('yaw_velocity_rad_s', np.nan),
                            p.get('yaw_acceleration_rad_s2', np.nan)]).all()):
                    raise RuntimeError('invalid continuous contact sweep mode or limit telemetry')
    audit = module.handoff.metrics(result.trace)
    sweep = [r for r in result.trace if r.get('handoff_smoothing', {}).get('advanced')
             and r['handoff_smoothing']['mode'] == 'contact_sweep']
    if sweep:
        violations = []
        for row in sweep:
            p = row['handoff_smoothing']
            if (np.linalg.norm(p['acceleration_xy_m_s2']) > p['xy_acceleration_limit_m_s2']+1e-8
                    or abs(p['yaw_acceleration_rad_s2']) > p['yaw_acceleration_limit_rad_s2']+1e-8
                    or np.linalg.norm(p['velocity_xy_m_s']) > .08+1e-8
                    or abs(p['yaw_velocity_rad_s']) > .35+1e-8):
                violations.append(row['t'])
        audit.update(contact_sweep_steps=len(sweep), contact_sweep_limit_violations=violations,
                     contact_sweep_peak_wrist_N=max(float(r['normal_force']) for r in sweep),
                     acceleration_limit_applies_only_inside_two_seconds=False)
        audit['accepted_ACT_control_steps'] += sum(r['handoff_smoothing']['accepted'] for r in sweep)
        if violations:
            raise RuntimeError('continuous contact sweep acceleration audit failed')
    if audit['active_limit_violations'] or audit['rejected_advanced_rows']:
        raise RuntimeError('handoff acceleration/accepted-command audit failed')
    rows = [r for r in result.trace if r.get('handoff_smoothing', {}).get('advanced')]
    if any(r.get('z_owner') != 'admittance' for r in rows):
        raise RuntimeError('ACT stole Z ownership')
    if any(not np.array_equal(np.asarray(r['handoff_smoothing']['command']),
                              np.asarray(r['handoff_smoothing']['native_reference']))
           for r in rows if r['handoff_smoothing']['mode'] == 'native_restored'):
        raise RuntimeError('native ACT not exactly restored after transition')
    result.hybrid_audit = audit


def build_driver(module, cfg, context):
    """Assemble the reviewed driver, explicitly binding its copied Z namespace.

    strict_exception_function copies globals *before* the legacy factory
    temporarily patches evaluate._contact_z_command. Bind the final namespace
    explicitly so the existing trial-4 limiter/follow controller is not lost.
    No gains, ownership state, planner, timer or governor algorithm change.
    """
    driver, governor = formal.build_formal_act_driver(module, cfg, context)

    def contact_z(config, loop, nominal, force, relief=None):
        return module.bounded_contact_z(config, loop, nominal, force, relief,
                                        context['z_limiter'], float(context['env'].tcp()[2]))

    driver.__globals__['_contact_z_command'] = contact_z
    scheduler = driver.__globals__['ActionChunkScheduler']

    class ValidatedScheduler(scheduler):
        def accept(self, issued_at, actions, now):
            values = np.asarray(actions, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
                raise ValueError('invalid/nonfinite Hybrid ACT chunk; no hold or requery fallback')
            return super().accept(issued_at, values, now)

        def action_for(self, now):
            action = super().action_for(now)
            # None before a chunk's timestamp is normal scheduler waiting;
            # malformed executable predictions are not normal waiting.
            if action is not None:
                values = np.asarray(action)
                if values.shape != (4,) or not np.isfinite(values).all():
                    raise ValueError('invalid/nonfinite Hybrid ACT action')
            return action

    driver.__globals__['ActionChunkScheduler'] = ValidatedScheduler
    return driver, governor


def run_hybrid_episode(cfg, seed=0, model_path=None, preview=False, capture_replay=False):
    """One geometric opportunity; A* approach, wrist contact, ACT XY/yaw sweep.

    Always record control telemetry internally: the reviewed adapter's shared
    safety/termination hooks run with capture_replay=True. This does not render
    images or export media; the CLI's --record-replay controls those operations.
    Execution exceptions propagate, carrying the retained causal context.
    """
    from ..expert import expert_plan_candidates
    from ..rollout import ActRolloutResult
    from ...environments.sweep_env import SweepEnv
    from ...environments.ur10_interface import UR10CB3EndEffector

    checkpoint = require_dinov3_checkpoint(model_path)
    cfg = formal.candidate_config(cfg, int(cfg.task.target_count))
    module = runtime()
    control_profile = configure_admittance(module, cfg)
    module.TRIAL_SPEC.check_clock(1. / float(cfg.sim.control_hz))
    env = None
    context = None
    try:
        with module.old.precision.temporary_solver(UR10CB3EndEffector, 1e-6):
            env = SweepEnv(cfg, seed=seed)
            env.record_raw_force_peaks = True
            env.reset(seed=seed)
            initial = np.c_[env.component_positions(), env.component_quats()].astype('<f4')
            plans = expert_plan_candidates(env, cfg, max_candidates=1)
            if not plans or not plans[0].feasible:
                mask = env.collected_mask()
                result = ActRolloutResult(
                    success=False, failure_reason='no feasible single geometric plan',
                    termination_reason='no_plan', target_count=int(cfg.task.target_count),
                    total=len(mask), collected=int(mask.sum()), final_collected=int(mask.sum()),
                    planner_attempts=1, planner_status='no_plan')
                result.physical_attempts = 0
            else:
                context = module.prepare_contact(env, cfg, plans[0])
                formal.validate_acquisition(context)
                if context['ready']:
                    # Source adapters patch only process-local function globals;
                    # Z/admittance/filter/clock state is inherited, never reset.
                    driver, _governor = build_driver(module, cfg, context)
                    result = driver(cfg, seed=seed, model_path=checkpoint,
                                    preview=preview, capture_replay=True)
                    result.trace = context['trace'] + result.trace
                    result.actions = np.concatenate((np.asarray(context['actions']).reshape(-1, 4),
                                                     result.actions), axis=0)
                    result.observations.clear()
                else:
                    count = int(env.collected_mask().sum())
                    result = ActRolloutResult(
                        success=False, failure_reason=context.get('error', 'stable wrist contact not reached'),
                        target_count=int(cfg.task.target_count), total=len(env.collected_mask()),
                        trace=context['trace'], actions=np.asarray(context['actions'], dtype=np.float32).reshape(-1, 4),
                        final_collected=count, collected=count, peak_force=context['peak_force'],
                        elapsed=float(env.time), sampled_frames=len(context['actions']),
                        termination_reason=context.get('terminal_reason', 'no_handoff'),
                        first_entry_time=context['collection_timer'].first_entry_time,
                        peak_collected=context['collection_timer'].peak_collected, collection_window_seconds=2.)
                _audit_result(module, context, result)
                result.physical_attempts = 1
                result.planner_attempts = 1
            result.initial_object_poses = initial
            result.planner_strategy = 'hybrid_astar_act_trial4_dinov3_frozen'
            result.control_profile = control_profile
            result.inference_config = cfg
            result.model_checkpoint = checkpoint
            return result
    except BaseException as error:
        # Do not convert unknown physical/policy failures to ordinary scored
        # failures, issue a hold, or ask the policy for another action chunk.
        error.hybrid_context = context
        raise
    finally:
        if env is not None:
            env.close()
