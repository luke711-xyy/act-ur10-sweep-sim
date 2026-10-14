"""Frozen trial-4 Hybrid A*→ACT evaluation on the original 120 paired layouts.

This explicit formal entry replaces the inference design for THIS batch only.
The original pure-ACT entry, frozen sim sources, defaults, data and old ledgers
stay intact. Existing reviewed trial-4 control functions are reused in isolated
episode processes; no training, physical screening, fallback or retry exists.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import lru_cache
import fcntl
import gzip
import hashlib
import importlib.util
import inspect
import json
import linecache
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EXPORTS = Path(os.environ.get('SWEEPSIM_EXPORTS', str(PROJECT / 'exports'))).resolve()
OLD = Path(os.environ.get('SWEEPSIM_ACT_BATCH', str(EXPORTS / 'batch_fine_teached_100k_120_unique_seeds_20261014'))).resolve()
RUNTIME_ROOT = PROJECT / 'experiments' / 'hybrid_trial4'
CANDIDATE = RUNTIME_ROOT / 'vendor' / 'admittance_2mm_target_gain_search_goal1_20261014'
LABEL = 'hybrid_astar_act_trial4_100k'
PROFILE = dict(target_force_N=1., total_Z_limit_m=.002,
               seek_MBK=[10., 400., 10.], maintain_MBK=[10., 800., 10.],
               gate_m=.001, contact_threshold_N=.5, contact_dwell_s=.1,
               stable_band_N=[.8, 1.2], stable_dwell_s=.2,
               handoff_window_s=2., XY_acceleration_m_s2=.05,
               XY_speed_m_s=.08, yaw_speed_rad_s=.35, yaw_acceleration_rad_s2=.5,
               up_m_s=.02, down_m_s=.1, tracking_gain=.5, tracking_bias_cap_m=.0002,
               IK_early_exit_threshold=1e-6, force_feedback='compensated_filtered_wrist_Fz_only',
               Z_owner='admittance_plus_bounded_actual_Z_follow', ACT_predicted_Z_executed=False,
               after_handoff_window='exact_native_ACT_without_continued_acceleration_limiting')
sys.path.insert(0, str(PROJECT))


def configure_locations(exports_root=None, act_batch=None):
    """Set new-run locations; never relocate or reseal old formal histories."""
    global EXPORTS, OLD
    if exports_root is not None: EXPORTS = Path(exports_root).resolve()
    if act_batch is not None: OLD = Path(act_batch).resolve()
    os.environ['SWEEPSIM_EXPORTS'] = str(EXPORTS)
    os.environ['SWEEPSIM_ACT_BATCH'] = str(OLD)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def numeric(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(f'unsupported JSON value: {type(value).__name__}')
    # Serialize completely before touching any existing artifact. Standard
    # Python JSON retains integer nanoseconds; nonfinite values still fail.
    payload = json.dumps(value, indent=2, allow_nan=False, default=numeric)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(payload)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def validate_plan(plan):
    if len(plan) != 120 or Counter(p['target_count'] for p in plan) != Counter({i: 20 for i in range(1, 7)}):
        raise ValueError('exactly 120 opportunities, twenty per target, required')
    if any(len({p[k] for p in plan}) != 120 for k in ('key', 'seed', 'layout_fingerprint')):
        raise ValueError('paired keys, seeds and layouts must be unique')
    if any(p['checkpoint_step'] != 100000 or Path(p['model_path']).name != 'step_100000'
           or np.asarray(p['initial_object_poses']).shape != (6, 7) for p in plan):
        raise ValueError('all paired episodes must use the frozen 100K model and six poses')
    if len({p['model_path'] for p in plan}) != 1:
        raise ValueError('cannot mix checkpoints')


def candidate_config(frozen, target):
    if type(target) is not int or not 1 <= target <= 6: raise ValueError('target outside 1–6')
    cfg = frozen.copy()
    changes = {'task.target_count': target, 'controller.desired_force': 1.,
               'controller.admittance.m': 10., 'controller.admittance.b': 400.,
               'controller.admittance.k': 10., 'controller.force_filter_cutoff_hz': 5.,
               'controller.delta_z_limit': .002,
               'controller.delta_z_upward_rate_limit': .02,
               'controller.delta_z_downward_rate_limit': .1}
    for key, value in changes.items(): cfg.set_path(key, value)
    return cfg


def validate_row(spec, row):
    for key in ('key', 'seed', 'target_count', 'layout_fingerprint', 'model_path', 'checkpoint_step'):
        if row.get(key) != spec[key]: raise ValueError(f'nonpaired result: {key}')
    if (row.get('opportunities') != 1 or row.get('planning_calls') != 1
            or row.get('physical_attempts') not in (0, 1) or row.get('frame_budget') != 2000
            or type(row.get('sampled_frames')) is not int or not 0 <= row['sampled_frames'] <= 2000
            or type(row.get('final_collected')) is not int or not 0 <= row['final_collected'] <= 6
            or type(row.get('success')) is not bool or row.get('media_complete') is not True
            or row.get('model_label') != LABEL):
        raise ValueError('incomplete, invalid or repeated opportunity result')
    if row['success'] and row['final_collected'] != spec['target_count']:
        raise ValueError('strict success cannot have an incorrect final count')
    if row.get('no_plan') and (row['success'] or row['physical_attempts'] != 0):
        raise ValueError('no-plan opportunity must be a nonexecuted failure')


def claim_attempt(root, spec):
    folder = Path(root) / 'attempts'; folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"seed_{spec['seed']}.json"
    with path.open('x') as stream:
        json.dump(dict(key=spec['key'], seed=spec['seed'], target_count=spec['target_count'],
                       layout_fingerprint=spec['layout_fingerprint'], started_at=time.time(),
                       opportunities=1, automatic_retry_forbidden=True), stream)
        stream.flush(); os.fsync(stream.fileno())
    return path


def require_clean_start(root):
    if list((Path(root) / 'attempts').glob('seed_*.json')) or read_rows(Path(root) / 'episodes.jsonl'):
        raise RuntimeError('existing attempt/result: automatic restart or retry forbidden')


def select_unattempted(root, plan):
    rows = read_rows(Path(root)/'episodes.jsonl')
    summarize(rows, plan)
    completed = {r['key'] for r in rows}
    by_seed = {s['seed']: s for s in plan}
    markers = {}
    for path in (Path(root)/'attempts').glob('seed_*.json'):
        marker = json.loads(path.read_text()); seed = marker['seed']
        if seed not in by_seed or marker['key'] != by_seed[seed]['key'] or seed in markers:
            raise RuntimeError('invalid or duplicate attempt marker')
        markers[seed] = marker['key']
    if any(r['seed'] not in markers for r in rows): raise RuntimeError('scored row lacks attempt marker')
    if set(markers.values()) != completed: raise RuntimeError('unfinished attempt requires a new user decision')
    return [s for s in plan if s['key'] not in completed]


def complete_json_prefix(text):
    """Read only fully encoded top-level fields; never repair a partial value."""
    decoder = json.JSONDecoder(); index = 0; values = {}
    def whitespace(pos):
        while pos < len(text) and text[pos].isspace(): pos += 1
        return pos
    index = whitespace(index)
    if index == len(text) or text[index] != '{': raise ValueError('not an original JSON object')
    index += 1
    while True:
        index = whitespace(index)
        if index == len(text) or text[index] == '}': return values
        try: key, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError: return values
        if not isinstance(key, str) or key in values: raise ValueError('invalid/duplicate original field')
        end = whitespace(end)
        if end == len(text): return values
        if text[end] != ':': raise ValueError('invalid original field separator')
        try: value, end = decoder.raw_decode(text, whitespace(end+1))
        except json.JSONDecodeError: return values
        values[key] = value; end = whitespace(end)
        if end == len(text) or text[end] == '}': return values
        if text[end] != ',': raise ValueError('invalid original object separator')
        index = end+1


def authorize_worker(root):
    root = Path(root)
    owner = root/'continuation_started.json' if (root/'continuation_started.json').exists() else root/'started.json'
    try: parent = json.loads(owner.read_text())['pid']
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError('worker has no authorized parent') from error
    if parent != os.getppid(): raise RuntimeError('worker is not owned by the batch parent')
    if not (root / 'batch.lock').exists(): raise RuntimeError('parent batch lock is missing')
    with (root / 'batch.lock').open('r') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise RuntimeError('parent does not own the exclusive batch lock')


def validate_metadata_seal(root):
    root = Path(root)
    seal = json.loads((root / 'seal.json').read_text())
    if sha(root / 'metadata.json') != seal['metadata_sha256']:
        raise RuntimeError('frozen metadata changed after preparation')


def commit_result(root, spec, row):
    validate_row(spec, row)
    marker = Path(root) / 'attempts' / f"seed_{spec['seed']}.json"
    if not marker.is_file() or json.loads(marker.read_text())['key'] != spec['key']:
        raise RuntimeError('result has no matching consumed opportunity')
    ledger = Path(root) / 'episodes.jsonl'
    if any(r['key'] == spec['key'] for r in read_rows(ledger)):
        raise RuntimeError('refuse to replace or repeat a scored result')
    with ledger.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def summarize(rows, plan):
    validate_plan(plan); by_key = {p['key']: p for p in plan}
    if len({r['key'] for r in rows}) != len(rows): raise ValueError('duplicate scored opportunity')
    for row in rows:
        if row['key'] not in by_key: raise ValueError('result outside paired plan')
        validate_row(by_key[row['key']], row)
    targets = {}
    for t in range(1, 7):
        selected = [r for r in rows if r['target_count'] == t]
        errors = [abs(r['final_collected']-t) for r in selected]
        targets[str(t)] = dict(episodes=len(selected), successes=sum(r['success'] for r in selected),
                              exact_quantity=sum(e == 0 for e in errors),
                              absolute_error_counts={str(e): errors.count(e) for e in range(7)},
                              mean_absolute_error=sum(errors)/len(errors) if errors else None)
    return dict(completed=len(rows), planned=120, complete=len(rows) == 120,
                opportunities=sum(r['opportunities'] for r in rows),
                physical_attempts=sum(r['physical_attempts'] for r in rows),
                no_plan_failures=sum(bool(r.get('no_plan')) for r in rows),
                successes=sum(r['success'] for r in rows), by_target=targets,
                tuning_layout_reused_seed=2120018, quantity_match_is_not_strict_success=True)


def remove_verified_video(root, video, verification):
    root, video = Path(root).resolve(), Path(video)
    resolved = video.resolve()
    relative = resolved.relative_to(root)
    if (video.is_symlink() or len(relative.parts) != 4 or relative.parts[0] != 'episodes'
            or not relative.parts[1].startswith('seed_') or relative.parts[2] != 'videos'
            or resolved.suffix != '.mp4' or not resolved.is_file()
            or resolved.stat().st_size != verification['bytes'] or sha(resolved) != verification['sha256']):
        raise ValueError('not this batch verified generated video')
    report = dict(deleted_files=1, freed_bytes=verification['bytes'],
                  sha256=verification['sha256'], path=str(resolved), permanent=True,
                  completed_at=time.time())
    # Evidence is durable before the authorized irreversible deletion.
    write_json(resolved.parent.parent / 'video_cleanup.json', report)
    resolved.unlink()
    return report


@lru_cache(maxsize=1)
def runtime():
    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec); sys.modules[name] = value
        spec.loader.exec_module(value); return value
    previous = sys.modules.get('phases')
    try:
        phase = load(CANDIDATE / 'phases.py', 'hybrid_formal_trial4_phases')
        sys.modules['phases'] = phase
        module = load(CANDIDATE / 'runner.py', 'hybrid_formal_trial4_runtime')
    finally:
        if previous is None: sys.modules.pop('phases', None)
        else: sys.modules['phases'] = previous
    module.TRIAL_SPEC = phase.AdmittanceSpec(1., (10., 400., 10.), (10., 800., 10.))
    return module


def dependency_hashes(module):
    # ``from phases import Class`` does not leave the phases module in the
    # runner's namespace. Its registered module and nested helpers still own
    # live execution code, so seed the traversal with all loaded local modules.
    pending = [module] + [v for v in list(sys.modules.values()) if isinstance(v, ModuleType)]
    visited, paths = set(), {Path(__file__).resolve()}
    while pending:
        value = pending.pop()
        if id(value) in visited: continue
        visited.add(id(value))
        path = Path(getattr(value, '__file__', '') or '').resolve()
        if not path.is_relative_to(RUNTIME_ROOT) or path.suffix != '.py': continue
        paths.add(path)
        pending.extend(v for v in vars(value).values() if isinstance(v, ModuleType))
    return {str(p): sha(p) for p in sorted(paths)}


def verify_identity(root):
    validate_metadata_seal(root)
    metadata = json.loads((Path(root) / 'metadata.json').read_text())
    if metadata['kind'] != LABEL or metadata['profile'] != PROFILE:
        raise RuntimeError('not the approved trial4 formal profile')
    paired = metadata.get('paired_act_batch')
    if (not isinstance(paired, str) or not Path(paired).is_absolute()
            or Path(paired).resolve() != OLD.resolve()):
        raise RuntimeError('configured paired baseline differs from sealed batch identity')
    for path, expected in metadata['frozen_sha256'].items():
        if sha(path) != expected: raise RuntimeError(f'frozen input changed: {path}')
    actual_sources = {str(p.relative_to(PROJECT)): sha(p) for p in (PROJECT / 'sim').rglob('*.py')}
    if actual_sources != metadata['source_fingerprints']: raise RuntimeError('frozen sim source set changed')
    model = metadata['checkpoint']
    for name, expected in model['checkpoint_files'].items():
        p = Path(model['path']) / name; stat = p.stat()
        if [stat.st_size, stat.st_mtime_ns] != expected: raise RuntimeError('frozen checkpoint changed: ' + name)
    return metadata


def prepare(root):
    from sim.config import load_config, save_config
    root = Path(root).resolve()
    if root.is_relative_to(OLD) or OLD.is_relative_to(root) or root.parent != EXPORTS:
        raise ValueError('new batch must be a separate direct exports child')
    old_meta = json.loads((OLD / 'metadata.json').read_text())
    plan = json.loads((OLD / 'episode_plan.json').read_text()); validate_plan(plan)
    pure_rows = read_rows(OLD / 'episodes.jsonl')
    if (len(pure_rows) != 120 or len({r['key'] for r in pure_rows}) != 120
            or json.loads((OLD / 'status.json').read_text())['state'] != 'complete'
            or json.loads((OLD / 'summary.json').read_text()).get('complete') is not True):
        raise RuntimeError('paired pure-ACT numeric results must be complete')
    by_key = {s['key']: s for s in plan}
    if any(r['key'] not in by_key or any(r[k] != by_key[r['key']][k]
           for k in ('seed', 'target_count', 'layout_fingerprint')) for r in pure_rows):
        raise RuntimeError('old pure-ACT ledger does not match plan')
    if (sha(OLD / 'episode_plan.json') != old_meta['episode_plan_sha256']
            or sha(OLD / 'evaluation_config.yaml') != old_meta['config_sha256']):
        raise RuntimeError('old frozen plan/config changed')
    model = old_meta['models'][0]
    if model['step'] != 100000 or sha(Path(model['path']) / 'model.safetensors') != model['weights_sha256']:
        raise RuntimeError('wrong frozen 100K weights')
    module = runtime()
    cfg = candidate_config(load_config(str(OLD / 'evaluation_config.yaml')), 1)
    module.TRIAL_SPEC.check_clock(1./float(cfg.sim.control_hz))
    if cfg.controller.force_sensor != 'wrist_ft' or cfg.controller.safe_max_force != 20 or cfg.controller.safe_force_dwell_steps != 3:
        raise RuntimeError('trial4 wrist/safety contract differs')
    frozen = dependency_hashes(module)
    # Bind actual completed baseline records in this new run, not a macOS-only
    # historical cleanup review. Already sealed histories are never rewritten.
    for name in ('episode_plan.json', 'evaluation_config.yaml', 'metadata.json',
                 'episodes.jsonl', 'summary.json', 'status.json'):
        frozen[str(OLD / name)] = sha(OLD / name)
    manifest = Path(old_meta['dataset']) / 'manifest.jsonl'
    if sha(manifest) != old_meta['dataset_manifest_sha256']: raise RuntimeError('dataset manifest changed')
    frozen[str(manifest)] = sha(manifest)
    if shutil.disk_usage(EXPORTS).free < 6*2**30: raise RuntimeError('free disk below 6 GiB')
    root.mkdir(exist_ok=False)
    for name in ('episodes', 'attempts', 'logs'): (root / name).mkdir()
    shutil.copyfile(OLD / 'episode_plan.json', root / 'episode_plan.json')
    save_config(cfg, str(root / 'evaluation_config.yaml'))
    write_json(root / 'control_profile.json', PROFILE)
    frozen.update({str(root / name): sha(root / name) for name in ('episode_plan.json', 'evaluation_config.yaml', 'control_profile.json')})
    write_json(root / 'metadata.json', dict(kind=LABEL, paired_act_batch=str(OLD),
               checkpoint=model, source_fingerprints=old_meta['source_fingerprints'],
               frozen_sha256=frozen, profile=PROFILE, created_at=time.time(),
               planned=120, targets=list(range(1, 7)), episodes_per_target=20,
               frame_budget=2000, budget_includes_Astar_prefix_and_force_acquisition=True,
               first_entry_window_seconds=2., tuning_layout_reused_seed=2120018,
               not_a_fresh_untuned_test_set=True, no_automatic_restart=True,
               media_policy='verify generated MP4 and images then delete; retain states, arrays, manifests and trace',
               native_success_contract='ACT native strict collection contract, shared two-second first-entry window'))
    write_json(root / 'seal.json', dict(metadata_sha256=sha(root / 'metadata.json')))
    verify_identity(root)
    write_json(root / 'summary.json', summarize([], plan))
    write_json(root / 'status.json', dict(state='prepared', completed=0, planned=120))
    return root


def result_row(spec, result=None, *, no_plan=False):
    return dict(**{k: spec[k] for k in ('key', 'seed', 'target_count', 'layout_fingerprint', 'model_path', 'checkpoint_step')},
                model_label=LABEL, paired_pure_ACT_key=spec['key'],
                success=bool(result is not None and result.success),
                final_collected=int(result.final_collected) if result is not None else 0,
                reason=str(result.failure_reason) if result is not None else 'no feasible A* plan',
                termination_reason=str(result.termination_reason) if result is not None else 'no_plan',
                opportunities=1, planning_calls=1, physical_attempts=0 if no_plan else 1,
                no_plan=no_plan, frame_budget=2000,
                sampled_frames=int(result.sampled_frames) if result is not None else 0,
                tuning_layout=spec['seed'] == 2120018, media_complete=no_plan)


def strict_exception_function(function):
    """Process-local copy: normal ACT code unchanged; all exceptions propagate.

    Keep the old handler statements unreachable after a bare raise so the
    reviewed adapter's structural checks still cover the exact original AST.
    No hold command or new prediction is attempted after an exception.
    """
    tree = ast.parse(inspect.getsource(function))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    if not handlers: raise RuntimeError('native ACT exception structure changed')
    for handler in handlers: handler.body.insert(0, ast.Raise())
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    filename = f'<formal_hybrid_fail_closed_{id(function)}>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(function.__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[function.__name__]


def build_formal_act_driver(module, cfg, context):
    from sim.act import evaluate
    original = evaluate.run_act_episode
    try:
        evaluate.run_act_episode = strict_exception_function(original)
        return module.build_act_driver(cfg, context)
    finally:
        evaluate.run_act_episode = original


def validate_acquisition(context):
    if context['step_failures'] or context['invalid_samples']:
        raise RuntimeError('unexpected invalid prefix execution')
    if context['ready']: return
    terminal = context.get('safety_reason') or context.get('collection_reason')
    if terminal:
        known = {repr(RuntimeError(prefix + terminal)) for prefix in (
            'prefix terminated before ACT handoff: ', 'acquisition terminated: ')}
        if context.get('terminal_reason') == terminal and context.get('error') in known: return
        raise RuntimeError('unexpected exception on a terminal acquisition step: ' + str(context.get('error')))
    expected = {
        'stable wrist-force band not established within 30 s; ACT sweep not started',
        'shared frame budget exhausted before stable wrist contact',
        'expert descent ended without reaching 1 mm; no force search/retry permitted',
    }
    if context.get('error') not in {repr(RuntimeError(reason)) for reason in expected}:
        raise RuntimeError('unexpected acquisition exception: ' + str(context.get('error')))


def require_saved_replay_states(trace):
    if not trace or any(not isinstance(r.get('replay_state'), dict)
                        or not {'spec', 'values'} <= r['replay_state'].keys() for r in trace):
        raise RuntimeError('replay requires saved integration states on every recorded row')


def recover_measurements(spec, header, trace, initial):
    """Recover recorded outcome and derived metrics only; no env or policy."""
    from sim.act.rollout import CollectionTerminationTracker
    required = {'key', 'seed', 'target_count', 'layout_fingerprint', 'model_path', 'checkpoint_step',
                'model_label', 'success', 'final_collected', 'reason', 'termination_reason',
                'opportunities', 'planning_calls', 'physical_attempts', 'no_plan', 'frame_budget',
                'sampled_frames', 'ACT_executed', 'handoff_ready', 'handoff_time', 'accepted_ACT_chunks'}
    if not required <= header.keys(): raise ValueError('incomplete native result prefix')
    if any(header[k] != spec[k] for k in ('key', 'seed', 'target_count', 'layout_fingerprint', 'model_path', 'checkpoint_step')):
        raise ValueError('saved outcome belongs to another paired layout')
    if spec['seed'] != 2120000 or header['success'] is not False or header['physical_attempts'] != 1:
        raise ValueError('authorization covers only the first recorded failed execution')
    if header['model_label'] != LABEL or header['no_plan'] or not header['ACT_executed'] or not header['handoff_ready']:
        raise ValueError('not the recorded completed ACT failure')
    require_saved_replay_states(trace)
    initial = np.asarray(initial, dtype='<f4')
    if initial.shape != (6, 7) or hashlib.sha256(initial.tobytes()).hexdigest() != spec['layout_fingerprint']:
        raise ValueError('recorded initial layout changed')
    times = np.asarray([r['t'] for r in trace], dtype=float)
    forces = np.asarray([r['normal_force'] for r in trace], dtype=float)
    if not np.isfinite(np.r_[times, forces]).all() or np.any(np.diff(times) <= 0):
        raise ValueError('nonfinite or noncausal saved trace')
    tracker = CollectionTerminationTracker(target_count=spec['target_count'])
    terminal = ''
    for r in trace:
        mask = np.asarray(r['object_collected'], dtype=bool)
        if mask.shape != (6,) or int(mask.sum()) != r['collected']:
            raise ValueError('saved count/mask mismatch')
        terminal = tracker.update(mask, float(r['t']), partial_overlap_mask=r.get('partial_overlap_mask', r.get('trial_partial_overlap_mask')))
    if header['final_collected'] != trace[-1]['collected'] or terminal != header['termination_reason']:
        raise ValueError('recorded outcome conflicts with retained causal trace')
    row = dict(header); row.pop('handoff_audit', None)
    module = runtime(); audit = module.handoff.metrics(trace)
    if audit['active_limit_violations'] or audit['rejected_advanced_rows']:
        raise ValueError('saved handoff audit failed')
    act = [r for r in trace if r.get('handoff_smoothing', {}).get('advanced')]
    if not act or any(r.get('z_owner') != 'admittance' for r in act): raise ValueError('missing recorded ACT/Z ownership')
    if any(not np.array_equal(np.asarray(r['handoff_smoothing']['command']), np.asarray(r['handoff_smoothing']['native_reference']))
           for r in act if r['handoff_smoothing']['mode'] == 'native_restored'):
        raise ValueError('saved native-restoration audit failed')
    act_force = np.asarray([r['normal_force'] for r in act], dtype=float)
    row.update(media_complete=False, handoff_audit=audit,
               mean_ACT_filtered_wrist_N=float(act_force.mean()), peak_filtered_wrist_N=float(forces.max()),
               peak_ACT_raw_tared_wrist_N=max(float(r['raw_normal_force']) for r in act),
               wrist_below_0_1N_fraction=float(np.mean(act_force < .1)),
               force_is_wrist_reaction_not_table_only_pressure=True, force_records_Hz=100.,
               raw_peak_is_not_all_physics_substeps=True, recovered_from_saved_states_only=True,
               scheduler_queries=None, scheduler_timeouts=None, scheduler_late_results=None,
               missing_scheduler_counters_not_fabricated=True,
               physics_runner_sha256='cfae2c55a5d00c5cd5b64356379e6f7d585459212a4ab1f13578e076290fa025')
    contact = next((r['tcp'] for r in trace if abs(float(r['t'])-float(header['handoff_time'])) < 1e-8), None)
    if contact is None: raise ValueError('original handoff pose not recorded')
    result = SimpleNamespace(success=header['success'], failure_reason=header['reason'],
        termination_reason=header['termination_reason'], trace=trace, total=6, target_count=spec['target_count'],
        final_collected=header['final_collected'], collected=tracker.peak_collected,
        peak_collected=tracker.peak_collected, first_entry_time=tracker.first_entry_time,
        collection_window_seconds=tracker.window_seconds, first_contact_position=np.asarray(contact),
        initial_object_poses=initial, peak_force=float(forces.max()), elapsed=float(times[-1]),
        sampled_frames=header['sampled_frames'])
    return row, result


def claim_continuation(root, plan):
    root = Path(root)
    try: authorization = json.loads((root/'recovery_authorization.json').read_text())
    except (OSError, ValueError) as error: raise RuntimeError('continuation requires explicit authorization') from error
    if authorization.get('recovered_seed') != 2120000 or authorization.get('continue_unattempted') != 119:
        raise RuntimeError('wrong continuation authorization')
    if (root/'continuation_started.json').exists(): raise RuntimeError('authorized continuation already claimed')
    rows = read_rows(root/'episodes.jsonl'); remaining = select_unattempted(root, plan)
    if len(rows) != 1 or rows[0]['key'] != plan[0]['key'] or len(remaining) != 119:
        raise RuntimeError('authorization requires the first recovered record and exactly119 unattempted layouts')
    write_json(root/'continuation_started.json', dict(pid=os.getpid(), started_at=time.time(),
               authorization_sha256=sha(root/'recovery_authorization.json'), unattempted=119))


def recover_first_record(root):
    """Single authorized record/media repair. Never calls evaluate_worker."""
    from sim.act.inference_lock import inference_lease
    from sim.act.media_retention import finalize_inference_video
    from sim.act.replay import persist_replay_input, save_inference_replay
    from sim.act.video import export_inference_video
    from sim.config import load_config
    root = Path(root).resolve(); metadata = verify_identity(root)
    authorization = json.loads((root/'recovery_authorization.json').read_text())
    if authorization.get('recovered_seed') != 2120000: raise RuntimeError('first-record recovery not authorized')
    with (root/'batch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read_rows(root/'episodes.jsonl'): raise RuntimeError('first record already committed; do not replace it')
        if len(list((root/'attempts').glob('seed_*.json'))) != 1: raise RuntimeError('unexpected attempted-layout set')
        for path, expected in authorization['evidence_sha256'].items():
            if sha(path) != expected: raise RuntimeError('saved recovery evidence changed: '+path)
        plan = json.loads((root/'episode_plan.json').read_text()); spec = plan[0]
        destination = root/'episodes/seed_2120000'
        error = json.loads((destination/'exception.json').read_text())
        if error['error'] != "TypeError('Object of type ndarray is not JSON serializable')":
            raise RuntimeError('not the authorized serialization-only failure')
        with gzip.open(destination/'exception_trace.json.gz', 'rt') as stream: backup = json.load(stream)
        if backup['invalid_samples'] or backup['step_failures']: raise RuntimeError('invalid physical samples cannot be recovered as a completed result')
        header = complete_json_prefix((destination/'.physical_result.json.tmp').read_text())
        initial = json.loads((destination/'initial_layout.json').read_text())['poses']
        row, result = recover_measurements(spec, header, backup['trace'], initial)
        cfg = candidate_config(load_config(str(OLD/'evaluation_config.yaml')), spec['target_count'])
        if cfg.to_dict() != json.loads((destination/'configuration.json').read_text()): raise RuntimeError('original physical config changed')
        write_json(destination/'physical_result_recovered.json', row)
        with inference_lease():
            verify_identity(root)
            if shutil.disk_usage(root).free < 6*2**30: raise RuntimeError('free disk below6GiB before recovery replay')
            saved = persist_replay_input(cfg, spec['seed'], result, destination/'replays', model=spec['model_path'])
            episode_id = save_inference_replay(cfg, spec['seed'], result, destination/'replays',
                                             model=spec['model_path'], image_format='webp-lossless')
            if not episode_id: raise RuntimeError('empty retained-state replay; do not rerun physics')
            episode = destination/'replays'/episode_id
            manifest = destination/'replays/manifest.jsonl'; records = read_rows(manifest)
            if len(records) != 1 or records[0]['replay_render_mode'] != 'recorded_physics_state':
                raise RuntimeError('ambiguous or non-state recovery replay')
            records[0].update(planner=LABEL, planner_strategy=LABEL, recovered_from_saved_states_only=True,
                scheduler_queries=None, scheduler_timeouts=None, scheduler_late_results=None,
                missing_scheduler_counters_not_fabricated=True)
            with manifest.open('w') as stream:
                stream.write(json.dumps(records[0], allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            video = export_inference_video(episode, destination/'videos')
            media = finalize_inference_video(episode, video)
            cleanup = remove_verified_video(root, video, media['video_verification'])
        row.update(media_complete=True, media_validation=media, video_cleanup=cleanup,
                   historical_video_path=str(video), replay_episode=str(episode), replay_input=str(saved),
                   record_recovery_runner_sha256=sha(Path(__file__)))
        verify_identity(root); validate_row(spec, row); write_json(destination/'result.json', row)
        commit_result(root, spec, row)
        write_json(root/'summary.json', summarize([row], plan))
        write_json(root/'status.json', dict(state='ready_for_authorized_continuation', completed=1,
                   planned=120, current=None, updated_at=time.time(), physics_first_seed_not_repeated=True))
        return row


def preserve_partial(destination, module, context, error):
    context = context or {}
    with gzip.open(destination / 'exception_trace.json.gz', 'wt', compresslevel=3) as stream:
        json.dump(module.old.precision.reviewed.base.json_safe(dict(
            error=repr(error), trace=context.get('trace', [])+context.get('act_trace', []),
            invalid_samples=context.get('invalid_samples', []), step_failures=context.get('step_failures', []))),
            stream, allow_nan=False)
    write_json(destination / 'exception.json', dict(error=repr(error), traceback=traceback.format_exc(),
               unknown_final_count_not_fabricated=True, automatic_retry_forbidden=True))


def evaluate_worker(root, index):
    from sim.act import evaluate
    from sim.act.expert import expert_plan_candidates
    from sim.act.inference_lock import inference_lease
    from sim.act.media_retention import finalize_inference_video
    from sim.act.replay import persist_replay_input, save_inference_replay
    from sim.act.rollout import ActRolloutResult
    from sim.act.video import export_inference_video
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.environments.ur10_interface import UR10CB3EndEffector
    root = Path(root); authorize_worker(root); metadata = verify_identity(root)
    plan = json.loads((root / 'episode_plan.json').read_text()); validate_plan(plan)
    if not 0 <= index < 120: raise ValueError('episode index outside paired plan')
    spec = plan[index]; module = runtime()
    cfg = candidate_config(load_config(str(OLD / 'evaluation_config.yaml')), spec['target_count'])
    module.TRIAL_SPEC.check_clock(1./float(cfg.sim.control_hz))
    destination = root / 'episodes' / f"seed_{spec['seed']}"
    context = None; env = None; result = None
    with inference_lease(on_wait=lambda: print(json.dumps(dict(event='waiting_for_manual_inference', seed=spec['seed'])), flush=True)):
        verify_identity(root)
        if shutil.disk_usage(root).free < 6*2**30: raise RuntimeError('free disk below 6 GiB')
        claim_attempt(root, spec); destination.mkdir()
        write_json(destination / 'configuration.json', cfg.to_dict())
        try:
            with module.old.precision.temporary_solver(UR10CB3EndEffector, 1e-6):
                env = SweepEnv(cfg, seed=spec['seed']); env.record_raw_force_peaks = True; env.reset(seed=spec['seed'])
                initial = np.c_[env.component_positions(), env.component_quats()].astype('<f4')
                fingerprint = hashlib.sha256(initial.tobytes()).hexdigest()
                if fingerprint != spec['layout_fingerprint']: raise RuntimeError('paired initial layout mismatch')
                write_json(destination / 'initial_layout.json', dict(sha256=fingerprint, poses=initial.tolist()))
                plans = expert_plan_candidates(env, cfg, max_candidates=1)
                if not plans or not plans[0].feasible:
                    row = result_row(spec, no_plan=True)
                    write_json(destination / 'planning.json', dict(planning_calls=1, feasible=False, physical_attempts=0))
                else:
                    geometric = plans[0]
                    write_json(destination / 'planning.json', module.old.precision.reviewed.base.json_safe(dict(
                        planning_calls=1, feasible=True, physical_candidate_screening=False,
                        selected_indices=geometric.target_indices, waypoints=geometric.waypoints,
                        strategy=geometric.strategy, landing_region=geometric.landing_region)))
                    context = module.prepare_contact(env, cfg, geometric)
                    validate_acquisition(context)
                    if context['ready']:
                        driver, governor = build_formal_act_driver(module, cfg, context)
                        result = driver(cfg, seed=spec['seed'], model_path=spec['model_path'], preview=True, capture_replay=True)
                        result.trace = context['trace'] + result.trace
                        result.actions = np.concatenate((np.asarray(context['actions']).reshape(-1, 4), result.actions), axis=0)
                        result.observations.clear()
                    else:
                        result = ActRolloutResult(success=False, failure_reason=context.get('error', 'stable wrist contact not reached'),
                            target_count=spec['target_count'], total=6, trace=context['trace'],
                            actions=np.asarray(context['actions'], dtype=np.float32).reshape(-1, 4),
                            final_collected=int(env.collected_mask().sum()), collected=int(env.collected_mask().sum()),
                            peak_force=context['peak_force'], elapsed=float(env.time), sampled_frames=len(context['actions']),
                            termination_reason=context.get('terminal_reason', 'no_handoff'),
                            first_entry_time=context['collection_timer'].first_entry_time,
                            peak_collected=context['collection_timer'].peak_collected, collection_window_seconds=2.)
                    if context['step_failures'] or context['invalid_samples']:
                        raise RuntimeError('invalid ACT execution; retain evidence, no unknown-count score')
                    result.initial_object_poses = initial
                    result.planner_strategy = LABEL
                    row = result_row(spec, result)
                    audit = module.handoff.metrics(result.trace)
                    if audit['active_limit_violations'] or audit['rejected_advanced_rows']:
                        raise RuntimeError('handoff acceleration/accepted-command audit failed')
                    act_rows = [r for r in result.trace if r.get('handoff_smoothing', {}).get('advanced')]
                    if any(r.get('z_owner') != 'admittance' for r in act_rows): raise RuntimeError('ACT stole Z ownership')
                    if any(not np.array_equal(np.asarray(r['handoff_smoothing']['command']), np.asarray(r['handoff_smoothing']['native_reference']))
                           for r in act_rows if r['handoff_smoothing']['mode'] == 'native_restored'):
                        raise RuntimeError('native ACT not exactly restored after transition')
                    force = [float(r['normal_force']) for r in act_rows]
                    row.update(ACT_executed=bool(audit['accepted_ACT_control_steps']),
                        handoff_ready=bool(context['ready']), handoff_time=context.get('handoff_time'),
                        accepted_ACT_chunks=context.get('accepted_chunks', 0), handoff_audit=audit,
                        mean_ACT_filtered_wrist_N=float(np.mean(force)) if force else None,
                        peak_filtered_wrist_N=max((float(r['normal_force']) for r in result.trace), default=None),
                        peak_ACT_raw_tared_wrist_N=max((float(r['raw_normal_force']) for r in act_rows), default=None),
                        wrist_below_0_1N_fraction=sum(f < .1 for f in force)/len(force) if force else None,
                        force_is_wrist_reaction_not_table_only_pressure=True,
                        force_records_Hz=float(cfg.sim.control_hz), raw_peak_is_not_all_physics_substeps=True)
                    write_json(destination / 'physical_result.json', row)
                    saved = module.old.persist_with_backup(destination, result, context['invalid_samples'],
                        lambda: persist_replay_input(cfg, spec['seed'], result, destination / 'replays', model=spec['model_path']))
                    row['replay_input'] = str(saved)
        except BaseException as error:
            preserve_partial(destination, module, context, error)
            raise
        finally:
            if env is not None: env.close()
            verify_identity(root)
    # Replay setup resets/settles the same-seed scene. Frames then restore the
    # saved integration states; no policy or episode command path is rerun.
    if result is not None and result.trace:
        require_saved_replay_states(result.trace)
        if shutil.disk_usage(root).free < 6*2**30: raise RuntimeError('free disk below 6 GiB before replay')
        episode_id = save_inference_replay(cfg, spec['seed'], result, destination / 'replays',
                                         model=spec['model_path'], image_format='webp-lossless')
        if not episode_id: raise RuntimeError('replay empty; no physical retry permitted')
        episode = destination / 'replays' / episode_id
        video = export_inference_video(episode, destination / 'videos')
        media = finalize_inference_video(episode, video)
        cleanup = remove_verified_video(root, video, media['video_verification'])
        row.update(media_complete=True, media_validation=media, video_cleanup=cleanup,
                   historical_video_path=str(video), replay_episode=str(episode))
    elif result is not None:
        row['media_complete'] = True
        row['no_physical_trace'] = True
    verify_identity(root); validate_row(spec, row)
    write_json(destination / 'result.json', row)
    return row


def run_batch(root, continue_authorized=False):
    root = Path(root).resolve()
    if root.parent != EXPORTS or root.is_symlink(): raise ValueError('batch must be its separate exports directory')
    verify_identity(root)
    with (root / 'batch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((root / 'episode_plan.json').read_text())
        if continue_authorized:
            claim_continuation(root, plan)
            rows = read_rows(root/'episodes.jsonl'); completed = {r['key'] for r in rows}
        else:
            require_clean_start(root)
            if (root / 'started.json').exists(): raise RuntimeError('batch already started; no automatic restart')
            write_json(root / 'started.json', dict(pid=os.getpid(), started_at=time.time()))
            rows = []; completed = set()
        try:
            for index, spec in enumerate(plan):
                if spec['key'] in completed: continue
                verify_identity(root)
                write_json(root / 'status.json', dict(state='running', completed=len(rows), planned=120,
                           current=spec['key'], seed=spec['seed'], pid=os.getpid(), updated_at=time.time()))
                with (root / 'logs' / f"seed_{spec['seed']}.log").open('x') as log:
                    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--out', str(root), '--worker-index', str(index)]
                    child = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                if child.returncode: raise RuntimeError(f"episode process exited {child.returncode}: seed {spec['seed']}; no retry")
                result = root / 'episodes' / f"seed_{spec['seed']}" / 'result.json'
                row = json.loads(result.read_text()); commit_result(root, spec, row); rows.append(row)
                write_json(root / 'summary.json', summarize(rows, plan))
                print(json.dumps(dict(event='hybrid_episode_finished', completed=len(rows), planned=120,
                                     seed=spec['seed'], success=row['success'])), flush=True)
            verify_identity(root)
            write_json(root / 'status.json', dict(state='complete', completed=120, planned=120, pid=os.getpid(), updated_at=time.time()))
        except BaseException as error:
            write_json(root / 'status.json', dict(state='needs_attention', completed=len(rows), planned=120,
                       current=spec['key'] if 'spec' in locals() else None, error=repr(error),
                       automatic_retry_forbidden=True, pid=os.getpid(), updated_at=time.time()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--exports-root', help='parent directory for a NEW isolated batch')
    parser.add_argument('--act-batch', help='completed same-machine pure ACT baseline to pair')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--recover-first-record-only', action='store_true')
    parser.add_argument('--continue-authorized', action='store_true')
    args = parser.parse_args(argv)
    configure_locations(args.exports_root, args.act_batch)
    if args.worker_index is not None:
        if args.prepare_only or args.recover_first_record_only or args.continue_authorized: parser.error('worker cannot prepare/recover/resume')
        evaluate_worker(Path(args.out), args.worker_index)
    elif args.recover_first_record_only:
        if args.prepare_only or args.continue_authorized: parser.error('record recovery cannot start physics')
        recover_first_record(Path(args.out))
    elif args.prepare_only: prepare(Path(args.out))
    else: run_batch(Path(args.out), continue_authorized=args.continue_authorized)


if __name__ == '__main__': main()
