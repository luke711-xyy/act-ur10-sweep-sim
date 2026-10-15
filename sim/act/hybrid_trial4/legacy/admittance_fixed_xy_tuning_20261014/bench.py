"""Throwaway fixed-XY physical admittance bench, no ACT/planner/task rollout.

Uses the existing arm, brush, scene and wrist-force controller unchanged.
Only approved controller parameters in config copies differ. All physics is
local, serialized via the existing lease; no RGB rendering or model loading.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[5]
BATCH = ROOT.parent / 'batch_fine_teached_100k_120_unique_seeds_20261014'
PREVIOUS = ROOT.parent / 'act_geometry3mm_admittance5mm_k20_trial_20261014'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PREVIOUS))
from runner import protected_snapshot, verify_protected, json_safe

FIELDS = {'m': 'controller.admittance.m', 'b': 'controller.admittance.b',
          'k': 'controller.admittance.k', 'cutoff': 'controller.force_filter_cutoff_hz',
          'up': 'controller.delta_z_upward_rate_limit',
          'down': 'controller.delta_z_downward_rate_limit'}
CASES = {
    'baseline_k20': {'m': .5, 'b': 70., 'k': 20.},
    'mass5_k20': {'m': 5., 'b': 70., 'k': 20.},
    'mass5_k0': {'m': 5., 'b': 70., 'k': 0.},
    'mass5_b140_k0': {'m': 5., 'b': 140., 'k': 0.},
    'mass10_b140_k0': {'m': 10., 'b': 140., 'k': 0.},
}


def make_config(base, values):
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise ValueError(f'outside approved controller scope: {sorted(unknown)}')
    limits = {'m': (.05, 100.), 'b': (1., 500.), 'k': (0., 100.),
              'cutoff': (.2, 40.), 'up': (.000001, .02), 'down': (.000001, .10)}
    cfg = base.copy()
    cfg.set_path('controller.delta_z_limit', .005)
    for key, value in values.items():
        value = float(value)
        lo, hi = limits[key]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'nonphysical/out-of-range {key}: {value}')
        cfg.set_path(FIELDS[key], value)
    if float(cfg.controller.desired_force) != .1 or float(cfg.controller.safe_max_force) != 20 or int(cfg.controller.safe_force_dwell_steps) != 3:
        raise ValueError('the approved force/safety contract has changed')
    a = cfg.controller.admittance
    dt = 1 / float(cfg.sim.control_hz)
    matrix = [[1-dt*dt*float(a.k)/float(a.m), dt*(1-dt*float(a.b)/float(a.m))],
              [-dt*float(a.k)/float(a.m), 1-dt*float(a.b)/float(a.m)]]
    if max(abs(np.linalg.eigvals(matrix))) > 1 + 1e-10:
        raise ValueError('these gains are unstable for the existing discrete integrator')
    return cfg


class SafetyGate:
    def __init__(self, ceiling, dwell):
        self.ceiling, self.dwell, self.count = float(ceiling), int(dwell), 0

    def observe(self, filtered_force, signed_sensor, state):
        if not np.isfinite(np.r_[filtered_force, signed_sensor, np.asarray(state).ravel()]).all():
            return 'nonfinite sensor or state'
        self.count = self.count + 1 if filtered_force > self.ceiling else 0
        return '20 N force dwell exceeded' if self.count >= self.dwell else None


def metrics(rows, *, dt, tail_seconds=10.):
    actual = [r for r in rows if r.get('advanced', True)]
    hold = [r for r in actual if r['phase'] == 'admittance']
    steady = [r for r in hold if r['t'] >= hold[-1]['t'] - tail_seconds] if hold else []
    longest = run = 0
    for row in hold:
        loaded = bool(row['table_geom_contact']) and row['table_normal_force'] > 1e-6
        run = run + 1 if loaded else 0
        longest = max(longest, run)
    forces = np.asarray([r['filtered_force'] for r in steady], dtype=float)
    signed = np.asarray([r['signed_compensated_force'] for r in steady], dtype=float)
    finite = bool(np.isfinite(forces).all() and np.isfinite(signed).all())
    z = np.asarray([r['tcp_z'] for r in steady], dtype=float)
    loaded = [bool(r['table_geom_contact']) and r['table_normal_force'] > 1e-6 for r in steady]
    return {
        'actual_steps': len(actual), 'admittance_steps': len(hold),
        'table_loaded_steps': sum(bool(r['table_geom_contact']) and r['table_normal_force'] > 1e-6 for r in hold),
        'longest_continuous_loaded_contact_s': longest * dt,
        'steady_window_actual_s': len(steady) * dt,
        'steady_loaded_contact_fraction': sum(loaded)/len(loaded) if loaded else 0.,
        'steady_filtered_force_mean_N': float(forces.mean()) if len(forces) and finite else None,
        'steady_filtered_force_std_N': float(forces.std()) if len(forces) and finite else None,
        'steady_filtered_force_quantiles_N': np.quantile(forces, [.05, .5, .95]).tolist() if len(forces) and finite else None,
        'steady_signed_compensated_force_mean_N': float(signed.mean()) if len(signed) and finite else None,
        'steady_table_normal_force_mean_N': float(np.mean([r['table_normal_force'] for r in steady])) if steady else None,
        'steady_tcp_z_range_mm': (np.array([z.min(), z.max()])*1000).tolist() if len(z) else None,
        'filtered_force_peak_N': max((r['filtered_force'] for r in actual), default=0.),
        'raw_substep_sample_peak_N': max((r['raw_peak'] for r in actual), default=0.),
        'table_normal_force_peak_N': max((r['table_normal_force'] for r in actual), default=0.),
        'all_recorded_sensor_values_finite': finite,
        'maximum_abs_command_correction_mm': max((abs(r['delta_z'])*1000 for r in hold), default=0.),
    }


def table_measurement(env):
    import mujoco
    total = 0.
    part_contacts = 0
    table_contacts = 0
    force = np.empty(6)
    for i in range(int(env.data.ncon)):
        c = env.data.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if not pair & env.brush_geom_ids:
            continue
        if pair & set(env.geom_to_component):
            part_contacts += 1
        if env.table_geom_id in pair:
            table_contacts += 1
            mujoco.mj_contactForce(env.model, env.data, i, force)
            total += max(0., float(force[0]))
    return total, table_contacts > 0, part_contacts


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(json_safe(value), f, indent=2, allow_nan=False)


def run_case(label, values, *, x=.42, y=0., handoff_height=.003, hold_seconds=20.):
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.controllers.hybrid import Command
    from sim.act.rollout import _force_loop, _overforce_relief_loop, _contact_z_command
    from sim.act.inference_lock import inference_lease

    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in label):
        raise ValueError('invalid artifact label')
    if not 0 < handoff_height <= .003 or not 10 <= hold_seconds <= 45:
        raise ValueError('outside the approved short contact bench')
    folder = ROOT / label
    folder.mkdir()  # each configuration/validation condition gets an honest separate record
    before = protected_snapshot()
    bench_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(folder / 'before.json', {'protected_inputs': before, 'bench_sha256': bench_sha})
    cfg = make_config(load_config(str(BATCH / 'evaluation_config.yaml')), values)
    write(folder / 'configuration.json', cfg.to_dict())
    rows = []
    failure = None
    env = None
    nominal = None
    hold_start = None
    invalid_sample = None
    identity_after = None
    elapsed_wall = time.monotonic()
    print(json.dumps({'event': 'waiting_for_slot', 'case': label, 'parameters': values,
                      'fixed_xy': [x,y], 'handoff_height_mm': handoff_height*1000}), flush=True)
    try:
        with inference_lease():
            verify_protected(before)
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != bench_sha:
                raise RuntimeError('bench changed while waiting')
            env = SweepEnv(cfg, seed=2120018)
            env.record_raw_force_peaks = True  # telemetry only, no physics/config change
            env.reset(seed=2120018)
            poses = np.c_[env.component_positions(), env.component_quats()].astype('<f4')
            write(folder / 'initial_layout.json', {'poses': poses, 'sha256': hashlib.sha256(poses.tobytes()).hexdigest()})
            if np.min(np.abs(env.component_positions()[:,0]-x)) < .08:
                raise RuntimeError('test location is not clear of components; do not move parts')
            dt = float(env.control_dt)
            loop = _force_loop(cfg)
            relief = _overforce_relief_loop(cfg)
            gate = SafetyGate(cfg.controller.safe_max_force, cfg.controller.safe_force_dwell_steps)
            phase = 'approach'
            command_z = float(cfg.end_effector.z_home)
            begin = float(env.time)
            print(json.dumps({'event': 'fixed_xy_test_started', 'case': label}), flush=True)
            for _ in range(round((hold_seconds + 15.) / dt)):
                sensor = float(env.ee.wrist_sign * env.ee.compensated_wrench_world()[2])
                raw_state = np.r_[env.data.qpos, env.data.qvel, env.data.sensordata,
                                  env.ee.wrist_ft_world(), env.ee.compensated_wrench_world()]
                reason = gate.observe(float(env.normal_force()), sensor, raw_state)
                if reason:
                    failure = reason
                    invalid_sample = {'filtered_force': float(env.normal_force()),
                                      'signed_sensor': sensor, 'state': raw_state}
                    break
                if phase == 'approach' and float(env.tcp()[2]) <= env.table_top_z + handoff_height:
                    phase = 'admittance'
                    nominal = float(env.tcp()[2])
                    hold_start = float(env.time)
                    loop.reset()
                    relief.reset()
                if phase == 'approach':
                    command_z = max(env.table_top_z + handoff_height - .001, command_z - .01*dt)
                else:
                    command_z = _contact_z_command(cfg, loop, relief, nominal, float(env.normal_force()))
                env.step_control(Command(x,y,command_z,0.))
                table_force, table_geom, part_contacts = table_measurement(env)
                sensor = float(env.ee.wrist_sign * env.ee.compensated_wrench_world()[2])
                row = {'t': float(env.time)-begin, 'advanced': True, 'phase': phase,
                       'filtered_force': float(env.normal_force()),
                       'signed_compensated_force': sensor,
                       'raw_tared_force': float(env.raw_normal_force()),
                       'raw_peak': float(env.last_raw_force_peak or 0.),
                       'table_normal_force': table_force, 'table_geom_contact': table_geom,
                       'part_contacts': part_contacts, 'tcp': env.tcp().tolist(),
                       'tcp_z': float(env.tcp()[2]), 'command_z': command_z,
                       'delta_z': 0. if nominal is None else command_z-nominal,
                       'wrist_ft_world': env.ee.wrist_ft_world().tolist(),
                       'compensated_wrench_world': env.ee.compensated_wrench_world().tolist(),
                       'wrench': env.wrench().tolist()}
                rows.append(row)
                raw_state = np.r_[env.data.qpos, env.data.qvel, env.data.sensordata,
                                  row['wrist_ft_world'], row['compensated_wrench_world'], row['wrench']]
                # Count safety dwell once per physical control step, before next command.
                if not np.isfinite(raw_state).all() or not math.isfinite(sensor):
                    failure = 'nonfinite sensor or state'
                    invalid_sample = {'signed_sensor': sensor, 'state': raw_state, 'row': row}
                    break
                if part_contacts:
                    failure = 'component contact invalidates isolated bench'
                    break
                if env.unsafe_robot_collision():
                    failure = 'unsafe robot collision'
                    break
                if hold_start is not None and float(env.time)-hold_start >= hold_seconds:
                    break
            if hold_start is None and failure is None:
                failure = '3 mm geometry handoff not reached'
            if hold_start is not None and float(env.time)-hold_start < hold_seconds-.02 and failure is None:
                failure = 'requested contact observation duration not reached'
    except Exception as error:
        failure = repr(error)
    finally:
        if env is not None:
            env.close()
        try:
            identity_after = verify_protected(before)
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != bench_sha:
                raise RuntimeError('bench identity changed during this attempt')
        except Exception as error:
            identity_after = {'verification_failure': repr(error)}
            failure = f'{failure or ""}; protected identity failure: {error}'
        with gzip.open(folder / 'trace.json.gz', 'xt') as f:
            json.dump(json_safe(rows), f, allow_nan=False)
        result = {'case': label, 'parameters': values, 'fixed_xy': [x,y],
                  'handoff_height_m': handoff_height, 'hold_seconds_requested': hold_seconds,
                  'latched_nominal_z_m': nominal if env is not None else None,
                  'failure': failure, 'wall_seconds': time.monotonic()-elapsed_wall,
                  'invalid_sample': invalid_sample,
                  'no_act_model_or_task_rollout': True, 'no_rgb_or_video': True,
                  'identity_after': identity_after,
                  'statistics': metrics(rows,dt=1/float(cfg.sim.control_hz))}
        write(folder / 'summary.json', result)
        print(json.dumps(json_safe({k:v for k,v in result.items() if k != 'identity_after'})), flush=True)
    if identity_after and 'verification_failure' in identity_after:
        raise RuntimeError('protected identity mismatch; stop tests, no automatic repair')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES))
    parser.add_argument('--label')
    for key in FIELDS:
        parser.add_argument(f'--{key}', type=float)
    parser.add_argument('--x',type=float,default=.42)
    parser.add_argument('--y',type=float,default=0.)
    parser.add_argument('--handoff-mm',type=float,default=3.)
    parser.add_argument('--hold-seconds',type=float,default=20.)
    args=parser.parse_args()
    if args.cases:
        for label in args.cases:
            run_case(label, CASES[label], x=args.x,y=args.y,handoff_height=args.handoff_mm/1000,hold_seconds=args.hold_seconds)
    elif args.label:
        run_case(args.label,{k:getattr(args,k) for k in FIELDS if getattr(args,k) is not None},
                 x=args.x,y=args.y,handoff_height=args.handoff_mm/1000,hold_seconds=args.hold_seconds)
    else:
        parser.error('declare cases or one uniquely named parameter trial')
