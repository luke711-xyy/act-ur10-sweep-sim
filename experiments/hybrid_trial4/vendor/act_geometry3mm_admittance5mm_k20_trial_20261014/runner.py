"""Temporary fixed-seed ACT admittance trials; never modify the live project.

Uses the reviewed 3-mm geometry adapter from the earlier trial unchanged.
Only a deep config copy receives +/-5-mm correction and requested M/B/K.
Every variant/goal runs once, with raw failure evidence retained.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[4]
BATCH = ROOT.parent / 'batch_fine_teached_100k_120_unique_seeds_20261014'
ADAPTER = ROOT.parent / 'act_geometry3mm_admittance5mm_trial_20261014' / 'trial.py'
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location('reviewed_geometry_adapter', ADAPTER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
SEEDS = adapter.SEEDS
EXPECTED_LAYOUTS = {
    1: '3c65aaa843635cc71cf176499aff8c04e834066cac7dcdf38ed176e3c6a7339f',
    6: 'a19adb6e7a1580669ef108c4cb4326bceaef05657172ca1cf7e70950747b27b3',
}


def make_config(base, target, *, m=.5, b=70., k=20.):
    for name, value in (('m', m), ('b', b), ('k', k)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'admittance {name} must be finite and positive')
    if target not in SEEDS:
        raise ValueError('only the two approved fixed layouts are allowed')
    cfg = adapter.trial_config(base, target)
    for name, value in (('m', m), ('b', b), ('k', k)):
        cfg.set_path(f'controller.admittance.{name}', float(value))
    return cfg


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_evidence(path, value):
    with Path(path).open('x') as f:
        json.dump(json_safe(value), f, indent=2, allow_nan=False)


def contact_metrics(rows, *, dt):
    actual = [r for r in rows if r.get('control_step_advanced', True)]
    contact = np.asarray([bool(r.get('trial_brush_table_contact')) for r in actual], dtype=bool)
    force = np.asarray([float(r['normal_force']) for r in actual])
    finite = force[np.isfinite(force)]
    table_force = force[contact & np.isfinite(force)]
    longest = current = 0
    for hit in contact:
        current = current + 1 if hit else 0
        longest = max(longest, current)
    handoff = next((r for r in actual if r.get('contact')), None)
    table_first = next((r for r in actual if r.get('trial_brush_table_contact')), None)
    post = [r for r in actual if handoff and r['t'] >= handoff['t']]
    z = np.asarray([r.get('tcp', [None, None, float('nan')])[2] for r in actual], dtype=float)
    return {
        'physical_control_steps': len(actual),
        'rejected_attempt_rows': len(rows) - len(actual),
        'table_contact_steps': int(contact.sum()),
        'table_contact_fraction_after_handoff': (sum(bool(r.get('trial_brush_table_contact')) for r in post) / len(post)) if post else None,
        'longest_continuous_table_contact_s': longest * dt,
        'table_contact_force_max_N': float(table_force.max()) if len(table_force) else None,
        'table_contact_filtered_force_quantiles_N': np.quantile(table_force, [.1, .5, .9]) if len(table_force) else None,
        'filtered_force_peak_N': float(finite.max()) if len(finite) else None,
        'nonfinite_force_rows': int((~np.isfinite(force)).sum()),
        'force_over_20N_steps': int((force > 20).sum()),
        'force_over_0p05N_steps': int((force >= .05).sum()),
        'minimum_actual_tcp_z_m': float(z[np.isfinite(z)].min()) if np.isfinite(z).any() else None,
        'geometry_handoff': handoff and {'t': handoff['t'], 'tcp': handoff.get('tcp'), 'normal_force': handoff['normal_force']},
        'first_real_table_contact_time_s': table_first and table_first['t'],
        'table_contact_semantics': 'actual brush/table geom contact, not logical Z ownership or force on parts',
        'force_semantics': 'native filtered normal force; may include brush/part contact',
    }


def protected_snapshot():
    result = adapter.protect_snapshot()
    result.update(runner_sha256=adapter.digest(__file__), adapter_sha256=adapter.digest(ADAPTER))
    return result


def verify_protected(before):
    after = protected_snapshot()
    stable = ('sources_sha256', 'checkpoint_path', 'checkpoint_files', 'metadata_sha256',
              'default_config_sha256', 'frozen_config_sha256', 'plan_sha256',
              'dataset_manifest_sha256', 'runner_sha256', 'adapter_sha256')
    mismatches = [k for k in stable if before[k] != after[k]]
    ledger = (BATCH / 'episodes.jsonl').read_bytes()
    prefix = ledger[:before['formal_ledger_prefix_bytes']]
    if hashlib.sha256(prefix).hexdigest() != before['formal_ledger_prefix_sha256']:
        mismatches.append('existing_formal_ledger_prefix')
    if mismatches:
        raise RuntimeError(f'protected inputs changed: {mismatches}')
    return {'unchanged': list(stable), 'original_formal_ledger_prefix_unchanged': True, 'after': after}


def run_one(target, *, variant, m, b, k):
    from sim.config import load_config
    from sim.act import evaluate
    from sim.act.inference_lock import inference_lease
    from sim.act.replay import persist_replay_input, save_inference_replay
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video

    if not variant or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in variant):
        raise ValueError('variant must be an explicit safe artifact name')
    variant_root = ROOT / variant
    variant_root.mkdir(exist_ok=True)
    destination = variant_root / f'goal_{target}'
    destination.mkdir()  # one physical trial per variant/goal; refuse overwrite/retry
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('free disk below 6 GiB; refuse trial')
    before = protected_snapshot()
    write_evidence(destination / 'before.json', before)
    cfg = make_config(load_config(str(BATCH / 'evaluation_config.yaml')), target, m=m, b=b, k=k)
    override = dict(adapter.OVERRIDE, admittance_m=m, admittance_b=b, admittance_k=k)
    model = before['checkpoint_path']
    seed = SEEDS[target]
    write_evidence(destination / 'configuration.json', cfg.to_dict())
    print(json.dumps({'event': 'waiting_for_inference_slot', 'target': target, 'variant': variant, 'override': override}), flush=True)
    with inference_lease():
        verify_protected(before)
        result = adapter.adapt_function(evaluate.run_act_episode)(cfg, seed=seed, model_path=model, preview=True, capture_replay=True)
        result.observations.clear()
        fingerprint = hashlib.sha256(np.asarray(result.initial_object_poses, dtype='<f4').tobytes()).hexdigest()
        summary = {
            'method': 'temporary_ACT_geometry3mm_admittance5mm', 'variant': variant,
            'target': target, 'seed': seed, 'checkpoint': model, 'act_executed': True,
            'override': override, 'success': result.success, 'failure_reason': result.failure_reason,
            'termination_reason': result.termination_reason, 'sampled_frames': result.sampled_frames,
            'elapsed_s': result.elapsed, 'final_collected': result.final_collected,
            'initial_layout_fingerprint': fingerprint, 'no_retries': True,
            **contact_metrics(result.trace, dt=1 / float(cfg.sim.control_hz)),
        }
        # Save the outcome before strict native replay serialization can reject NaN/Inf.
        write_evidence(destination / 'physical_result.json', summary)
        print(json.dumps(json_safe({'event': 'physical_result_saved', **summary})), flush=True)
        if fingerprint != EXPECTED_LAYOUTS[target]:
            write_evidence(destination / 'physical_trace_failure.json', {'reason': 'fixed layout mismatch', 'result': vars(result)})
            raise RuntimeError('fixed layout identity mismatch; retained raw evidence, no retry')
        result.planner_strategy = f'act_geometry3mm_admittance5mm_{variant}'
        try:
            replay_input = persist_replay_input(cfg, seed, result, destination / 'replays', model=model)
        except Exception as error:
            write_evidence(destination / 'physical_trace_failure.json', {'reason': repr(error), 'result': vars(result)})
            raise RuntimeError('native replay persistence failed; invalid raw values retained, no retry') from error
        summary['replay_input'] = str(replay_input)
    episode_id = save_inference_replay(cfg, seed, result, destination / 'replays', model=model, image_format='webp-lossless')
    if not episode_id:
        raise RuntimeError('no replay frames; physical evidence retained, no replacement trial')
    episode = destination / 'replays' / episode_id
    manifest = episode.parent / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [r for r in records if r['episode_id'] == episode_id]
    if len(selected) != 1:
        raise RuntimeError('ambiguous trial media provenance')
    selected[0].update(planner='act_geometry_admittance_trial',
                       planner_strategy=result.planner_strategy,
                       temporary_control_override=override,
                       contact_flag_semantics='geometric ownership latch; NOT proof of physical contact')
    # Only newly created trial manifest; never production or previous trial records.
    manifest.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    video = export_inference_video(episode, variant_root / 'videos')
    final_video = video.with_name(f'temporary_ACT_{variant}_goal_{target}_step_100000.mp4')
    if final_video.exists():
        raise RuntimeError('refuse to overwrite trial video')
    video.rename(final_video)
    validation = finalize_inference_video(episode, final_video)
    summary.update(video=str(final_video), media_validation=validation,
                   identity_verification=verify_protected(before))
    write_evidence(destination / 'final.json', summary)
    print(json.dumps({'event': 'video_ready', 'target': target, 'variant': variant, 'video': str(final_video),
                      'success': result.success, 'final_collected': result.final_collected}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, choices=(1, 6), required=True)
    parser.add_argument('--variant', default='k20_m0p5_b70')
    parser.add_argument('--m', type=float, default=.5)
    parser.add_argument('--b', type=float, default=70.)
    parser.add_argument('--k', type=float, default=20.)
    args = parser.parse_args()
    run_one(args.target, variant=args.variant, m=args.m, b=args.b, k=args.k)
