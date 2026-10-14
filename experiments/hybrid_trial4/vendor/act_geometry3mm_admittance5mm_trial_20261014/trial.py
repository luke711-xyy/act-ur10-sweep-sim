"""THROWAWAY: 3-mm geometric Z handoff, +/-5-mm admittance correction.

Only a copied function AST and an in-memory config overlay are changed.
The frozen project, evaluation files, model and demonstrations stay untouched.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
ROOT = Path(__file__).resolve().parent
BATCH = ROOT.parent / 'batch_fine_teached_100k_120_unique_seeds_20261014'
sys.path.insert(0, str(PROJECT))
OVERRIDE = {'geometry_handoff_height_m': .003, 'admittance_correction_limit_m': .005,
            'force_safety_ceiling_N': 20, 'force_safety_dwell_steps': 3}
SEEDS = {1: 2120018, 6: 2120029}


def geometry_contact(tcp_z, table_z):
    return bool(np.isfinite(tcp_z) and np.isfinite(table_z)
                and float(tcp_z) <= float(table_z) + .003)


def trial_config(base, target):
    cfg = base.copy()
    cfg.set_path('task.target_count', int(target))
    cfg.set_path('controller.delta_z_limit', .005)
    return cfg


class ForceGate:
    def __init__(self, threshold, dwell):
        self.threshold = float(threshold)
        self.dwell = int(dwell)
        self.steps = 0

    def update(self, force):
        if not np.isfinite(force):
            return True
        self.steps = self.steps + 1 if force > self.threshold else 0
        return self.steps >= self.dwell


def adapt_function(function):
    """Adapt the gate in copied globals; do not monkey-patch the sim module."""
    tree = ast.parse(inspect.getsource(function))
    counts = {'contact_condition': 0, 'safety_initialization': 0, 'safety_update': 0}

    class Adapter(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if (isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And)
                    and len(node.test.values) == 2
                    and isinstance(node.test.values[0], ast.Name)
                    and node.test.values[0].id == 'near_table'
                    and any(isinstance(v, ast.Attribute) and v.attr == 'contact_threshold'
                            for v in ast.walk(node.test.values[1]))):
                node.test = ast.parse('_trial_geometry_contact(float(tcp[2]), float(env.table_top_z))', mode='eval').body
                counts['contact_condition'] += 1
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return node
            target = node.targets[0].id
            if target == 'overforce_relief':
                counts['safety_initialization'] += 1
                return [node, *ast.parse('_trial_gate = _trial_force_gate(float(cfg.controller.safe_max_force), int(cfg.controller.safe_force_dwell_steps))').body]
            if (target == 'termination_reason' and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'update'):
                counts['safety_update'] += 1
                return [node, *ast.parse("if _trial_gate.update(post_force):\n    termination_reason = 'temporary trial safety: force ceiling or nonfinite force'").body]
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == 'capture_state':
                node.body.extend(ast.parse('row["trial_brush_table_contact"] = bool(env.brush_table_contact())').body)
            return node

    tree = Adapter().visit(tree)
    if counts != {'contact_condition': 1, 'safety_initialization': 1, 'safety_update': 1}:
        raise RuntimeError(f'frozen native control structure changed; refuse adapter: {counts}')
    namespace = dict(function.__globals__)
    namespace.update(_trial_geometry_contact=geometry_contact, _trial_force_gate=ForceGate)
    exec(compile(ast.fix_missing_locations(tree), '<temporary_geometry_handoff_trial>', 'exec'), namespace)
    result = namespace[function.__name__]
    result.__doc__ = __doc__
    return result


def annotate_replay(episode):
    manifest = Path(episode).parent / 'manifest.jsonl'
    records = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    found = [r for r in records if r['episode_id'] == Path(episode).name]
    if len(found) != 1:
        raise RuntimeError('temporary replay provenance is ambiguous')
    found[0].update(planner='act_geometry_height_trial',
                    planner_strategy='act_geometry3mm_admittance5mm',
                    temporary_control_override=OVERRIDE,
                    contact_flag_semantics='geometric Z-ownership latch, NOT proof of measured contact')
    manifest.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_evidence(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False,
                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))


def protect_snapshot():
    meta = json.loads((BATCH / 'metadata.json').read_text())
    sources = {str(p.relative_to(PROJECT)): digest(p) for p in (PROJECT / 'sim').rglob('*.py')}
    if sources != meta['source_fingerprints'] or len(sources) != 67:
        raise RuntimeError('frozen simulation identity mismatch')
    if digest(BATCH / 'episode_plan.json') != '3f7b5c0eddf23f3c9f018366de6a5504c41b7fa5d549b525c79b57091c85ff64':
        raise RuntimeError('frozen plan identity mismatch')
    if digest(BATCH / 'evaluation_config.yaml') != 'c962907ea1051d8554687b7292f90dcf4e4d2ddf6c32029fc211d08bbba57fd8':
        raise RuntimeError('frozen config identity mismatch')
    model = meta['models'][0]
    cp = Path(model['path'])
    stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in cp.iterdir() if p.is_file()}
    if model['step'] != 100000 or stats != model['checkpoint_files']:
        raise RuntimeError('100K checkpoint identity mismatch')
    ledger = (BATCH / 'episodes.jsonl').read_bytes()
    return {'created_at_epoch': time.time(), 'sources_sha256': sources,
            'checkpoint_path': model['path'], 'checkpoint_files': stats,
            'metadata_sha256': digest(BATCH / 'metadata.json'),
            'default_config_sha256': digest(PROJECT / 'configs/default.yaml'),
            'frozen_config_sha256': digest(BATCH / 'evaluation_config.yaml'),
            'plan_sha256': digest(BATCH / 'episode_plan.json'),
            'dataset_manifest_sha256': digest(PROJECT / 'runs/act_dataset_five_view_rudra_300_smooth320_v1/manifest.jsonl'),
            'formal_ledger_prefix_bytes': len(ledger),
            'formal_ledger_prefix_sha256': hashlib.sha256(ledger).hexdigest(),
            'trial_sha256': digest(ROOT / 'trial.py'), 'temporary_override': OVERRIDE,
            'not_formal_evaluation': True, 'no_training': True}


def run_one(target):
    from sim.config import load_config
    from sim.act import evaluate
    from sim.act.inference_lock import inference_lease
    from sim.act.replay import persist_replay_input, save_inference_replay
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video

    destination = ROOT / f'goal_{target}'
    destination.mkdir()
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('free disk is below the 6-GiB safety boundary')
    cfg = trial_config(load_config(str(BATCH / 'evaluation_config.yaml')), target)
    model = json.loads((BATCH / 'metadata.json').read_text())['models'][0]['path']
    seed = SEEDS[target]
    write_evidence(destination / 'configuration.json', cfg.to_dict())
    print(json.dumps({'event': 'waiting_for_inference_slot', 'target': target, 'seed': seed, 'override': OVERRIDE}), flush=True)
    with inference_lease():
        protect_snapshot()
        result = adapt_function(evaluate.run_act_episode)(cfg, seed=seed, model_path=model, preview=True, capture_replay=True)
        result.observations.clear()
        result.planner_strategy = 'act_geometry3mm_admittance5mm'
        replay_input = persist_replay_input(cfg, seed, result, destination / 'replays', model=model)
        trace = result.trace
        first_handoff = next((r for r in trace if r.get('contact')), None)
        first_measured = next((r for r in trace if float(r['normal_force']) >= .05), None)
        first_physical = next((r for r in trace if r.get('trial_brush_table_contact')), None)
        summary = {'method': 'temporary_ACT_geometry3mm_admittance5mm', 'target': target, 'seed': seed,
                   'checkpoint': model, 'act_executed': True, 'override': OVERRIDE,
                   'success': result.success, 'failure_reason': result.failure_reason,
                   'termination_reason': result.termination_reason,
                   'sampled_frames': result.sampled_frames, 'elapsed_s': result.elapsed,
                   'final_collected': result.final_collected, 'peak_force_N': result.peak_force,
                   'geometric_handoff_time': None if first_handoff is None else first_handoff['t'],
                   'geometric_handoff_tcp': None if first_handoff is None else first_handoff['tcp'],
                   'geometric_handoff_force_N': None if first_handoff is None else first_handoff['normal_force'],
                   'first_force_threshold_time': None if first_measured is None else first_measured['t'],
                   'first_brush_table_contact_time': None if first_physical is None else first_physical['t'],
                   'brush_table_contact_control_steps': sum(bool(r.get('trial_brush_table_contact')) for r in trace),
                   'measured_force_contact_control_steps': sum(float(r['normal_force']) >= .05 for r in trace),
                   'initial_layout_fingerprint': hashlib.sha256(np.asarray(result.initial_object_poses, dtype='<f4').tobytes()).hexdigest(),
                   'replay_input': str(replay_input), 'no_retries': True}
        write_evidence(destination / 'physical_result.json', summary)
        print(json.dumps({'event': 'physical_result_saved', **summary}, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), flush=True)
    episode_id = save_inference_replay(cfg, seed, result, destination / 'replays', model=model, image_format='webp-lossless')
    if not episode_id:
        raise RuntimeError('no replay frames; physical evidence retained, no replacement trial')
    episode = destination / 'replays' / episode_id
    annotate_replay(episode)
    video = export_inference_video(episode, ROOT / 'videos')
    final_video = video.with_name(f'temporary_ACT_geometry3mm_admittance5mm_goal_{target}_step_100000.mp4')
    if final_video.exists():
        raise RuntimeError('refuse to overwrite an existing trial video')
    video.rename(final_video)
    validation = finalize_inference_video(episode, final_video)
    summary.update(video=str(final_video), media_validation=validation)
    write_evidence(destination / 'final.json', summary)
    print(json.dumps({'event': 'video_ready', 'target': target, 'video': str(final_video),
                      'success': result.success, 'final_collected': result.final_collected}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, choices=(1, 6), required=True)
    args = parser.parse_args()
    if not (ROOT / 'before.json').exists():
        write_evidence(ROOT / 'before.json', protect_snapshot())
    run_one(args.target)
