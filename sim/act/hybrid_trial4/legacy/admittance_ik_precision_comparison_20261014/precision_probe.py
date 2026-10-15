"""THROWAWAY: approved 1e-4 versus 1e-6 fixed-XY IK precision probe.

Only a copied solver's early-exit constant changes, in this process. No sim
file, shared default, task rollout, model or formal result is changed.
"""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import textwrap
import traceback

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'admittance_fixed_xy_tuning_20261014'
PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PREVIOUS))
import bench_v2 as reviewed

PARAMETERS = {'m': 25., 'b': 400., 'k': 1., 'cutoff': 5.,
              'up': .0001, 'down': .0002}
CASES = {'baseline_1e4': 1e-4, 'tight_1e6': 1e-6}


def tightened_solver(original, threshold):
    if threshold not in (1e-4, 1e-6):
        raise ValueError('only the approved original/tight IK thresholds are allowed')
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('native early-exit function structure changed')
    if tree.body[0].decorator_list:
        raise RuntimeError('native early-exit function has unexpected decorators')
    expected = ast.dump(ast.parse('np.linalg.norm(error)', mode='eval').body)
    matches = [node.test.comparators[0] for node in ast.walk(tree)
               if isinstance(node, ast.If)
               and isinstance(node.test, ast.Compare)
               and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Lt)
               and len(node.test.comparators) == 1
               and ast.dump(node.test.left) == expected
               and isinstance(node.test.comparators[0], ast.Constant)
               and node.test.comparators[0].value == 1e-4]
    if len(matches) != 1:
        raise RuntimeError('native early-exit contract changed; refuse broad replacement')
    if threshold == 1e-4:
        return original
    matches[0].value = 1e-6
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<temporary_single_constant_ik>',
                 'exec'), namespace)
    replacement = namespace[original.__name__]
    replacement.__qualname__ = original.__qualname__
    return replacement


@contextmanager
def temporary_solver(cls, threshold):
    original = cls._solve_ik
    replacement = tightened_solver(original, threshold)
    cls._solve_ik = replacement
    try:
        yield
    finally:
        cls._solve_ik = original


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_changes(before, after):
    return sorted(p for p in before.keys() | after.keys()
                  if before.get(p) != after.get(p))


def validate_effective_config(frozen, actual):
    expected = copy.deepcopy(frozen.to_dict())
    for field, value in PARAMETERS.items():
        keys = reviewed.base.FIELDS[field].split('.')
        node = expected
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    expected['controller']['delta_z_limit'] = .005
    if actual.to_dict() != expected:
        raise RuntimeError('effective configuration differs outside approved values')
    if (float(actual.controller.desired_force) != .1
            or float(actual.controller.safe_max_force) != 20.
            or int(actual.controller.safe_force_dwell_steps) != 3
            or str(actual.end_effector.type) != 'ur10_cb3_rudra'):
        raise RuntimeError('effective configuration violates force/safety/solver contract')
    return {'verified': True, 'all_unrelated_fields_identical_to_frozen_config': True,
            'end_effector_type': str(actual.end_effector.type),
            'desired_force_N': .1, 'correction_limit_m': .005,
            'safety_filtered_force_N': 20., 'safety_dwell_steps': 3}


def extra_protection():
    act = reviewed.base.BATCH
    star = ROOT.parent / 'batch_astar_single_attempt_120_paired_20261014'
    paths = [act / name for name in ('status.json', 'summary.json', 'episodes.jsonl',
                                    'runner.log', 'replays/manifest.jsonl')]
    paths += [star / name for name in ('status.json', 'summary.json', 'episodes.jsonl',
                                      'metadata.json', 'launch.json', 'runner.log')]
    paths += list((star / 'attempts').glob('seed_*.json'))
    paths += [PROJECT / 'scripts/evaluate_astar_baseline.py',
              PREVIOUS / 'bench.py', PREVIOUS / 'bench_v2.py',
              PREVIOUS / 'spring1_soft/trace.json.gz',
              PREVIOUS / 'best_parameter_only_candidate.json', Path(__file__)]
    return {str(path): digest(path) for path in paths}


def run_one(case):
    from sim.environments.ur10_interface import UR10CB3EndEffector
    if case not in CASES:
        raise ValueError('declare one of the two approved conditions')
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('free space below 6 GiB; refuse physical probe')
    threshold = CASES[case]
    protected = reviewed.base.protected_snapshot()
    extra_before = extra_protection()
    original = UR10CB3EndEffector._solve_ik
    native_source = inspect.getsource(original)
    original_root = reviewed.ROOT
    original_make_config = reviewed.base.make_config
    from sim.config import load_config
    frozen = load_config(str(reviewed.base.BATCH / 'evaluation_config.yaml'))
    initial_config = original_make_config(frozen, dict(PARAMETERS))
    config_preflight = validate_effective_config(frozen, initial_config)
    effective_check = None
    reviewed.base.write(ROOT / f'{case}_contract.json', {
        'case': case, 'seed': 2120018, 'fixed_xy': [.42, 0.],
        'ik_mixed_error_early_exit_threshold': threshold,
        'only_ik_solver_change': 'np.linalg.norm(error) early-exit constant',
        'unchanged_solver_settings': ['iteration budget', 'damping', 'joint limits',
                                     'actuator joint step limit', 'final residual safety tolerances'],
        'parameters': PARAMETERS, 'handoff_height_m': .003,
        'correction_limit_m': .005, 'desired_force_N': .1,
        'native_solver_source_sha256': hashlib.sha256(native_source.encode()).hexdigest(),
        'configuration_preflight': config_preflight,
        'protected_inputs': protected, 'extra_sha256': extra_before,
        'no_ACT_or_Astar_or_task_rollout': True,
        'throwaway_not_production': True,
    })
    result = None
    primary = None
    primary_traceback = None
    escaped_traceback = None
    audit_errors = []
    identity = None
    extra_after = None
    changed = []
    try:
        reviewed.ROOT = ROOT
        with temporary_solver(UR10CB3EndEffector, threshold):
            expected_solver = UR10CB3EndEffector._solve_ik
            def checked_make_config(base_config, values):
                nonlocal effective_check
                actual = original_make_config(base_config, values)
                effective_check = validate_effective_config(base_config, actual)
                if UR10CB3EndEffector._solve_ik is not expected_solver:
                    raise RuntimeError('expected native solver is not installed')
                effective_check['expected_native_solver_installed'] = True
                return actual
            reviewed.base.make_config = checked_make_config
            result = reviewed.run_case(case, dict(PARAMETERS), hold_seconds=45.)
    except BaseException as error:
        primary = error
        primary_traceback = error.__traceback__
        escaped_traceback = traceback.format_exc()
    finally:
        reviewed.ROOT = original_root
        reviewed.base.make_config = original_make_config
        restored = UR10CB3EndEffector._solve_ik is original
        if not restored:
            audit_errors.append('native in-process solver did not restore')
        try:
            identity = reviewed.base.verify_protected(protected)
        except Exception as error:
            audit_errors.append('protected input verification: ' + repr(error))
        try:
            extra_after = extra_protection()
            changed = protected_changes(extra_before, extra_after)
            if changed:
                audit_errors.append('protected formal/test evidence changed: ' + repr(changed))
        except Exception as error:
            audit_errors.append('extra protection readback: ' + repr(error))
        try:
            reviewed.base.write(ROOT / f'{case}_verification.json', {
                'restored_native_solver': restored, 'protected_verification': identity,
                'extra_protected_file_changes': changed,
                'extra_sha256_after': extra_after, 'audit_errors': audit_errors,
                'effective_configuration_check': effective_check,
                'escaped_exception': repr(primary) if primary is not None else None,
                'escaped_traceback': escaped_traceback,
                'ik_threshold': threshold,
                'physical_failure': result['failure'] if result is not None else None,
            })
        except Exception as error:
            audit_errors.append('outer verification write failed: ' + repr(error))
    if primary is not None:
        if audit_errors:
            primary.add_note('outer identity audit: ' + repr(audit_errors))
        raise primary.with_traceback(primary_traceback)
    if audit_errors:
        raise RuntimeError('outer identity audit failed: ' + repr(audit_errors))
    if result['failure'] is not None:
        raise RuntimeError('physical probe failed; evidence retained, do not retry')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    args = parser.parse_args()
    run_one(args.case)
