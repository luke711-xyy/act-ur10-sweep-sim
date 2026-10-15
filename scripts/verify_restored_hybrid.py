"""Inspect, or explicitly run ONE original trial-4 target-1 rollout without RL.

The frozen controller/driver is reused without residual instrumentation. This is
a restoration check, not a formal benchmark, retry loop, or training entry.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import evaluate_hybrid_paired_batch as formal

REFERENCE = 'eb887843c9849d4a72be116bb87c44519dc7ef74'
HISTORY = formal.EXPORTS / 'batch_hybrid_trial4_100k_120_paired_20261014'


def inspect_baseline():
    manifest = json.loads((PROJECT / 'hybrid_trial4_snapshot/source_manifest.json').read_text())
    if manifest['branch'] != 'feat/hybrid-astar-act-trial4':
        raise RuntimeError('wrong published Hybrid baseline')
    for name, entry in manifest['files'].items():
        if formal.sha(PROJECT / name) != entry['sha256']:
            raise RuntimeError('published Hybrid snapshot drift: ' + name)
    metadata = formal.verify_identity(HISTORY)
    if manifest['source_fingerprints'] != metadata['source_fingerprints']:
        raise RuntimeError('published simulator is not the formal Hybrid simulator')
    return dict(reference_branch=manifest['branch'], reference_commit=REFERENCE,
                sim_source_count=len(manifest['source_fingerprints']),
                profile=metadata['profile'], checkpoint=metadata['checkpoint']['path'],
                execution_started=False, RL_enabled=False, training_started=False)


def validate_restoration(row):
    return bool(row['handoff_ready'] and row['ACT_executed'] and row['native_restored_steps'] > 0
                and row['invalid_samples'] == row['step_failures'] == row['Z_policy_rows'] == 0
                and not row['handoff_limit_violations'] and not row['rejected_advanced_rows']
                and not row['safety_reason'])


def finish_media(root, cfg, seed, result, checkpoint):
    from sim.act.replay import save_inference_replay
    from sim.act.video import export_inference_video
    from sim.act.media_retention import verify_inference_video

    formal.write_json(root / 'status.json', dict(state='media_pending',
                      physical_execution_complete=True, training_started=False))
    try:
        episode_id = save_inference_replay(cfg, seed, result, root / 'replays',
                                          model=checkpoint, image_format='webp-lossless')
        if not episode_id:
            raise RuntimeError('saved-state replay is empty; no physical retry')
        video = export_inference_video(root / 'replays' / episode_id, root / 'videos')
        record = next(json.loads(line) for line in (root / 'replays/manifest.jsonl').read_text().splitlines()
                      if json.loads(line)['episode_id'] == episode_id)
        verification = verify_inference_video(video, frames=record['frame_count'], fps=25.)
        formal.write_json(root / 'media.json', dict(video=str(video), verification=verification,
                          frames_retained=True, replay_episode=str(root / 'replays' / episode_id)))
        return video
    except BaseException as error:
        formal.write_json(root / 'status.json', dict(state='needs_attention', stage='media',
            error=repr(error), physical_execution_complete=True, training_started=False,
            automatic_retry_forbidden=True))
        raise


def run_once(root, seed):
    import numpy as np
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.environments.ur10_interface import UR10CB3EndEffector
    from sim.act.expert import expert_plan_candidates
    from sim.act.replay import persist_replay_input
    from sim.act.rollout import ActRolloutResult

    if any(n == 'residual_rl' or n.startswith('residual_rl.') for n in sys.modules):
        raise RuntimeError('native restoration check must not load the RL adapter')
    before = inspect_baseline()
    root = Path(root).resolve()
    if root.parent != formal.EXPORTS or not root.name.startswith('hybrid_restore_goal1_'):
        raise ValueError('use a new direct exports/hybrid_restore_goal1_* directory')
    if type(seed) is not int or seed < 0 or 2120000 <= seed < 2120120:
        raise ValueError('use a new layout, not the historical120')
    lock_path = PROJECT / 'runs/.act_inference.lock'
    with lock_path.open('a+') as lock:
        # Fail instead of waiting for, or stopping, a user's manual task.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        formal.write_json(root / 'baseline.json', before)
        formal.write_json(root / 'attempt.json', dict(seed=seed, target_count=1,
                          opportunities=1, retry_allowed=False, RL_enabled=False))
        formal.write_json(root / 'status.json', dict(state='running', training_started=False))
        cfg = formal.candidate_config(load_config(str(formal.OLD / 'evaluation_config.yaml')), 1)
        formal.write_json(root / 'configuration.json', cfg.to_dict())
        module = formal.runtime()
        module.TRIAL_SPEC.check_clock(1. / float(cfg.sim.control_hz))
        env = None
        context = None
        try:
            import shutil
            if shutil.disk_usage(root).free < 8 * 2**30:
                raise RuntimeError('free disk below8GiB')
            with module.old.precision.temporary_solver(UR10CB3EndEffector, 1e-6):
                env = SweepEnv(cfg, seed=seed)
                env.record_raw_force_peaks = True
                env.reset(seed=seed)
                initial = np.c_[env.component_positions(), env.component_quats()].astype('<f4')
                fingerprint = hashlib.sha256(initial.tobytes()).hexdigest()
                old = json.loads((HISTORY / 'episode_plan.json').read_text())
                if fingerprint in {p['layout_fingerprint'] for p in old}:
                    raise RuntimeError('new layout duplicates old120')
                formal.write_json(root / 'initial_layout.json', dict(poses=initial, sha256=fingerprint))
                plans = expert_plan_candidates(env, cfg, max_candidates=1)
                formal.write_json(root / 'planning.json', dict(planning_calls=1,
                    feasible=bool(plans and plans[0].feasible), physical_screening=False))
                if not plans or not plans[0].feasible:
                    raise RuntimeError('one geometry opportunity produced no plan; no retry')
                context = module.prepare_contact(env, cfg, plans[0])
                formal.validate_acquisition(context)
                if context['ready']:
                    driver, _ = formal.build_formal_act_driver(module, cfg, context)
                    result = driver(cfg, seed=seed, model_path=before['checkpoint'],
                                    preview=True, capture_replay=True)
                    result.trace = context['trace'] + result.trace
                    result.actions = np.concatenate((np.asarray(context['actions']).reshape(-1, 4),
                                                     result.actions), axis=0)
                    result.observations.clear()
                else:
                    result = ActRolloutResult(success=False,
                        failure_reason=context.get('error', 'no stable contact'),
                        target_count=1, total=6, trace=context['trace'],
                        actions=np.asarray(context['actions'], dtype=np.float32).reshape(-1, 4),
                        final_collected=int(env.collected_mask().sum()),
                        collected=int(env.collected_mask().sum()), peak_force=context['peak_force'],
                        elapsed=float(env.time), sampled_frames=len(context['actions']),
                        termination_reason=context.get('terminal_reason', 'no_handoff'),
                        first_entry_time=context['collection_timer'].first_entry_time,
                        peak_collected=context['collection_timer'].peak_collected,
                        collection_window_seconds=2.)
                result.initial_object_poses = initial
                formal.require_saved_replay_states(result.trace)
                saved = persist_replay_input(cfg, seed, result, root / 'replays',
                                              model=before['checkpoint'])
                audit = module.handoff.metrics(result.trace)
                act = [r for r in result.trace if r.get('handoff_smoothing', {}).get('advanced')]
                forces = [float(r['normal_force']) for r in result.trace]
                errors = [abs(float(r['tcp'][2]) - float(r['command'][2])) for r in result.trace]
                row = dict(seed=seed, target_count=1, success=bool(result.success),
                    final_collected=int(result.final_collected), reason=result.termination_reason,
                    failure_reason=str(result.failure_reason),
                    opportunities=1, physical_attempts=1, RL_enabled=False, training_started=False,
                    handoff_ready=bool(context['ready']), ACT_executed=bool(act),
                    native_restored_steps=sum(r['handoff_smoothing']['mode'] == 'native_restored' for r in act),
                    invalid_samples=len(context['invalid_samples']), step_failures=len(context['step_failures']),
                    Z_policy_rows=sum(r.get('z_owner') != 'admittance' for r in act),
                    handoff_limit_violations=audit['active_limit_violations'],
                    rejected_advanced_rows=audit['rejected_advanced_rows'],
                    safety_reason=context.get('safety_reason'), handoff_audit=audit,
                    peak_filtered_wrist_N=max(forces), mean_filtered_wrist_N=float(np.mean(forces)),
                    mean_ACT_filtered_wrist_N=float(np.mean([r['normal_force'] for r in act])) if act else None,
                    max_Z_error_m=max(errors), sampled_frames=int(result.sampled_frames),
                    trace_rows=len(result.trace), replay_input=str(saved), layout_fingerprint=fingerprint)
                row['restoration_verified'] = validate_restoration(row)
                formal.write_json(root / 'result.json', row)
                np.savez_compressed(root / 'telemetry.npz',
                    t=np.asarray([r['t'] for r in result.trace]),
                    wrist_force_N=np.asarray(forces),
                    actual_tcp=np.asarray([r['tcp'] for r in result.trace]),
                    command=np.asarray([r['command'] for r in result.trace]))
        except BaseException as error:
            formal.preserve_partial(root, module, context, error)
            formal.write_json(root / 'status.json', dict(state='needs_attention', error=repr(error),
                actual_final_count=int(env.collected_mask().sum()) if env is not None else None,
                training_started=False, automatic_retry_forbidden=True))
            raise
        finally:
            if env is not None:
                env.close()
            if inspect_baseline() != before:
                raise RuntimeError('baseline changed during restoration check')
    # Media restores saved integration states; no ACT query or command execution.
    video = finish_media(root, cfg, seed, result, before['checkpoint'])
    formal.write_json(root / 'status.json', dict(state='complete' if row['restoration_verified'] else 'needs_attention',
                      restoration_verified=row['restoration_verified'], training_started=False,
                      complete=True, automatic_retry_forbidden=True))
    print(json.dumps(dict(row, handoff_audit='saved in result.json', video=str(video))))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-target1', action='store_true')
    parser.add_argument('--seed', type=int, default=3152200)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not args.run_target1:
        print(json.dumps(inspect_baseline(), indent=2))
        return 0
    if args.out is None:
        parser.error('--run-target1 requires a new --out')
    return 0 if run_once(args.out, args.seed)['restoration_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
