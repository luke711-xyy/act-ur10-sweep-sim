"""One authorized preflight after a verified batch; never a trainer."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .storage import RunStore, read_rows, sha256, write_json


LEGACY_PROFILE = 'legacy_100k'
DINO_PROFILE = 'dinov3_sync5_200k'
DINO_LABEL = 'hybrid_dinov3_200k'
DINO_STEP = 200000
DINO_SEEDS = set(range(2120000, 2120120))
DINO_SCHEDULE = dict(prediction_steps=25, execution_steps=5, action_hz=25.0,
    query_hz=5.0, temporal_ensemble=False, inference_advances_simulation=False,
    deterministic=True)
_BATCH_PROCESS_NAMES = {
    'runner.py', 'evaluate_new_schedule_batch.py', 'recovery_launcher.py',
    'recovery_episode_entrypoint.py', 'batch_evaluate.py',
}
_BATCH_MODULES = {'sim.act.evaluate', 'sim.act.batch_evaluate'}


def _path_belongs_to_root(argument, root):
    value = str(argument)
    return value == root or value.startswith(root + os.sep)


def _is_batch_execution(command):
    names = {Path(argument).name for argument in command}
    if names & _BATCH_PROCESS_NAMES or any(name.startswith('ffmpeg') for name in names):
        return True
    return any(argument in _BATCH_MODULES for argument in command)


def dependency_pids(root):
    """Find batch/recovery/encoder processes without signaling or stopping them."""
    import psutil

    root = str(Path(root).resolve())
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid == os.getpid():
            continue
        try:
            command = process.info['cmdline'] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if _is_batch_execution(command) and any(_path_belongs_to_root(arg, root) for arg in command):
            found.append(process.pid)
    return found


def local_slot_available():
    from .safety import local_inference_available
    return local_inference_available()[0]


def _legacy_dependency_ready(root, *, active_pids):
    """Validate the original paired 100K comparison batch."""
    root = Path(root)
    status = json.loads((root / 'status.json').read_text())
    if status.get('state') == 'needs_attention':
        raise RuntimeError('dependency needs_attention; no automatic recovery')
    if status.get('state') != 'complete':
        if not active_pids:
            raise RuntimeError('dependency exited before verified completion')
        return None
    if active_pids:
        return None
    metadata = json.loads((root / 'metadata.json').read_text())
    if (metadata.get('label') != 'new_schedule_no_noise_100k' or metadata.get('training') is not False
            or metadata.get('noise') is not False or metadata.get('checkpoint_step') != 100000):
        raise RuntimeError('wrong dependency protocol')
    plan_path = root / 'episode_plan.json'
    if sha256(plan_path) != metadata['plan_sha256']:
        raise RuntimeError('dependency plan changed')
    plan = json.loads(plan_path.read_text())
    planned = {r['seed']: r for r in plan}
    rows = read_rows(root / 'episodes.jsonl')
    seeds = {r['seed'] for r in rows}
    if (len(plan) != 120 or len(planned) != 120 or seeds != DINO_SEEDS or len(rows) != 120
            or Counter(r['target_count'] for r in rows) != Counter({n: 20 for n in range(1, 7)})):
        raise RuntimeError('dependency has incomplete/duplicate/unpaired result ledger')
    summary = json.loads((root / 'summary.json').read_text())
    if (summary.get('complete') is not True or summary.get('episodes') != 120
            or summary.get('media_pending') is not False or summary.get('verified_videos') != 120
            or summary.get('training_updates') != 0 or status.get('completed') != 120
            or summary.get('strict_successes') != sum(r.get('success') is True for r in rows)):
        raise RuntimeError('dependency summary/status/media disagree')
    keys = {RunStore.key(r) for r in rows}
    if {p.stem for p in (root / 'attempts').glob('*.json')} != keys:
        raise RuntimeError('dependency orphan/duplicate opportunity')
    for row in rows:
        expected = planned.get(row['seed'])
        key = RunStore.key(row)
        folder = root / 'episodes' / key
        if (expected is None or row['role'] != 'comparison' or row['arm'] != 'new_schedule_zero'
                or row['target_count'] != expected['target_count']
                or row['paired_original_key'] != expected['key']
                or row.get('layout_fingerprint') != expected['layout_fingerprint']
                or row.get('outcome') not in ('success', 'failure', 'safety')
                or type(row.get('success')) is not bool or row.get('opportunities') != 1
                or row.get('planning_calls') not in (0, 1) or row.get('physical_attempts') not in (0, 1)):
            raise RuntimeError('dependency pairing/one-shot result differs')
        if json.loads((folder / 'result.json').read_text()) != row:
            raise RuntimeError('dependency result differs from ledger')
        marker = json.loads((root / 'attempts' / f'{key}.json').read_text())
        if any(row.get(k) != v for k, v in marker.items()):
            raise RuntimeError('dependency attempt marker differs')
        media = json.loads((folder / 'media.json').read_text())
        video = folder / media['file']
        verified = media.get('verification', {})
        if (video.resolve().parent != folder.resolve() or video.is_symlink()
                or video.stat().st_size != media['bytes'] or sha256(video) != media['sha256']
                or verified.get('frames', 0) <= 0 or verified.get('sha256') != media['sha256']
                or verified.get('bytes') != media['bytes']):
            raise RuntimeError('dependency retained media is unverified or changed')
    return dict(episodes=120, verified_videos=120, ledger_sha256=sha256(root / 'episodes.jsonl'),
        summary_sha256=sha256(root / 'summary.json'), metadata_sha256=sha256(root / 'metadata.json'),
        status_sha256=sha256(root / 'status.json'), plan_sha256=sha256(plan_path),
        strict_successes=summary['strict_successes'])


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _decode_one_video_frame(video):
    try:
        import imageio_ffmpeg
        frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        try:
            details = next(frames)
            frame = next(frames)
            width, height = details['size']
            if not width or not height or len(frame) != width * height * 3:
                raise RuntimeError('decoded video frame has invalid dimensions')
            return 1
        finally:
            try:
                frames.close()
            except (BrokenPipeError, OSError):
                pass
    except (StopIteration, OSError, ValueError, KeyError) as error:
        raise RuntimeError(f'retained video cannot decode a frame: {video}') from error


def _resolve_dino_video(root, video_value):
    if not isinstance(video_value, str) or not video_value:
        raise RuntimeError('DINO result is missing its retained video path')
    video_root = root / 'videos'
    if video_root.is_symlink() or not video_root.is_dir():
        raise RuntimeError('DINO videos directory is missing or is a symlink')
    video_root = video_root.resolve(strict=True)
    candidate = Path(video_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise RuntimeError('DINO retained video is a symlink')
    try:
        video = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError('DINO retained video is missing') from error
    if video.parent != video_root or not video.is_file() or video.suffix.lower() != '.mp4':
        raise RuntimeError('DINO retained video escapes the batch videos directory')
    if video.stat().st_size <= 0:
        raise RuntimeError('DINO retained video is empty')
    return video


def _verify_dino_checkpoint(metadata, model, model_path):
    checkpoint = Path(model_path)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError('DINO checkpoint directory is missing or is a symlink')
    checkpoint = checkpoint.resolve(strict=True)
    files = model.get('checkpoint_files')
    digests = metadata.get('checkpoint_files_sha256')
    if not isinstance(files, dict) or not files or not isinstance(digests, dict) or set(files) != set(digests):
        raise RuntimeError('DINO checkpoint file identity map is incomplete')
    observed_files = {item.name for item in checkpoint.iterdir() if item.is_file()}
    if observed_files != set(files) or any(item.is_symlink() for item in checkpoint.iterdir()):
        raise RuntimeError('DINO checkpoint file set changed')
    for name, identity in files.items():
        item = checkpoint / name
        if (not isinstance(name, str) or Path(name).name != name or item.is_symlink()
                or not item.is_file() or not isinstance(identity, list) or len(identity) != 2
                or type(identity[0]) is not int or type(identity[1]) is not int
                or [item.stat().st_size, item.stat().st_mtime_ns] != identity
                or not _is_sha256(digests.get(name)) or sha256(item) != digests[name]):
            raise RuntimeError(f'DINO checkpoint file changed: {name}')
    if digests.get('model.safetensors') != model.get('weights_sha256'):
        raise RuntimeError('DINO model weight digest disagrees with checkpoint file identity')


def _matrix(rows, field):
    values = [[0 for _ in range(6)] for _ in range(7)]
    for row in rows:
        actual = row.get(field)
        target = row.get('target_count')
        if type(actual) is not int or not 0 <= actual <= 6 or type(target) is not int or not 1 <= target <= 6:
            raise RuntimeError(f'DINO ledger has an unknown or invalid {field} value')
        values[actual][target - 1] += 1
    return values


def _validate_dino_summary(root, status, rows):
    summary = json.loads((root / 'summary.json').read_text())
    successes = sum(row.get('success') is True for row in rows)
    if (summary.get('complete') is not True or summary.get('completed') != 120
            or summary.get('planned') != 120 or status.get('completed') != 120
            or status.get('planned') != 120 or status.get('current') is not None
            or status.get('media_pending') not in (None, False, '')):
        raise RuntimeError('DINO batch summary/status is incomplete or media is still pending')
    model_stats = summary.get('models', {}).get(DINO_LABEL)
    exact_matches = sum(row.get('peak_collected') == row.get('target_count') for row in rows)
    if (not isinstance(model_stats, dict)
            or any(type(model_stats.get(field)) is not int for field in (
                'episodes', 'successes', 'exact_peak_matches'))
            or model_stats.get('episodes') != 120 or model_stats.get('successes') != successes
            or model_stats.get('exact_peak_matches') != exact_matches):
        raise RuntimeError('DINO summary totals disagree with the result ledger')
    for field in ('peak_count_matrix', 'final_count_matrix'):
        observed = model_stats.get(field)
        if (not isinstance(observed, list) or len(observed) != 7
                or any(not isinstance(line, list) or len(line) != 6
                    or any(type(value) is not int for value in line) for line in observed)
                or observed != _matrix(rows, field.removesuffix('_count_matrix') + '_collected')):
            raise RuntimeError('DINO summary density matrices disagree with the result ledger')
    by_target = model_stats.get('by_target')
    if not isinstance(by_target, dict):
        raise RuntimeError('DINO summary is missing per-target totals')
    for target in range(1, 7):
        selected = [row for row in rows if row['target_count'] == target]
        entry = by_target.get(str(target), {})
        target_successes = sum(row['success'] is True for row in selected)
        if (type(entry.get('episodes')) is not int or entry.get('episodes') != 20
                or type(entry.get('successes')) is not int or entry.get('successes') != target_successes
                or type(entry.get('success_rate')) not in (int, float)
                or not math.isclose(entry.get('success_rate', -1), target_successes / 20,
                    rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError('DINO per-target summary disagrees with the result ledger')
    return summary, successes


def _dino_dependency_ready(root, *, active_pids):
    root = Path(root).resolve(strict=True)
    status = json.loads((root / 'status.json').read_text())
    if status.get('state') == 'needs_attention':
        raise RuntimeError('dependency needs_attention; no automatic recovery')
    if status.get('state') != 'complete':
        if not active_pids:
            raise RuntimeError('dependency exited before verified completion')
        return None
    if active_pids:
        return None
    metadata = json.loads((root / 'metadata.json').read_text())
    model_list = metadata.get('models')
    if (metadata.get('planned') != 120 or metadata.get('layouts') != 120
            or metadata.get('targets') != [1, 2, 3, 4, 5, 6]
            or metadata.get('layout_seed_mode') != 'independent_per_episode'
            or metadata.get('episodes_per_target_per_model') != 20
            or metadata.get('preview') is not False
            or not isinstance(model_list, list) or len(model_list) != 1):
        raise RuntimeError('wrong DINO sync5 dependency protocol')
    model = model_list[0]
    model_path = model.get('path')
    if (model.get('label') != DINO_LABEL or model.get('step') != DINO_STEP
            or not isinstance(model_path, str) or Path(model_path).name != 'step_200000'
            or not _is_sha256(model.get('weights_sha256'))):
        raise RuntimeError('wrong DINO checkpoint identity')
    _verify_dino_checkpoint(metadata, model, model_path)
    config_path = root / 'evaluation_config.yaml'
    if (not _is_sha256(metadata.get('config_sha256')) or not config_path.is_file()
            or sha256(config_path) != metadata['config_sha256']):
        raise RuntimeError('DINO frozen evaluation config changed')
    if (metadata.get('schedule_adapter') != DINO_SCHEDULE
            or not _is_sha256(metadata.get('schedule_adapter_sha256'))):
        raise RuntimeError('wrong DINO sync5 action schedule')
    control_profile = metadata.get('hybrid_control_profile')
    if not isinstance(control_profile, dict):
        raise RuntimeError('DINO hybrid control profile is missing')

    plan_path = root / 'episode_plan.json'
    plan_hash = sha256(plan_path)
    if plan_hash != metadata.get('episode_plan_sha256'):
        raise RuntimeError('DINO dependency plan changed')
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan, list) or len(plan) != 120:
        raise RuntimeError('DINO plan is incomplete')
    plan_by_seed = {}
    plan_keys = set()
    fingerprints = set()
    target_plan_counts = Counter()
    for item in plan:
        seed = item.get('seed')
        target = item.get('target_count')
        index = item.get('layout_index')
        fingerprint = item.get('layout_fingerprint')
        key = item.get('key')
        if (type(seed) is not int or seed in plan_by_seed or type(target) is not int
                or target not in range(1, 7) or type(index) is not int or index not in range(120)
                or not isinstance(fingerprint, str) or not _is_sha256(fingerprint)
                or not isinstance(key, str) or key in plan_keys
                or item.get('model_label') != DINO_LABEL or item.get('model_path') != model_path
                or item.get('checkpoint_step') != DINO_STEP):
            raise RuntimeError('DINO plan has duplicate or mismatched opportunities')
        expected_key = f'{DINO_LABEL}:n{target}:l{index:02d}'
        if key != expected_key:
            raise RuntimeError('DINO plan key does not match its target and layout index')
        plan_by_seed[seed] = item
        plan_keys.add(key)
        fingerprints.add(fingerprint)
        target_plan_counts[target] += 1
    if (set(plan_by_seed) != DINO_SEEDS or len({r['layout_index'] for r in plan}) != 120
            or len(fingerprints) != 120
            or target_plan_counts != Counter({target: 20 for target in range(1, 7)})):
        raise RuntimeError('DINO plan does not contain exactly 120 independent target/seed pairs')

    rows = read_rows(root / 'episodes.jsonl')
    rows_by_seed = {}
    if len(rows) != 120:
        raise RuntimeError('DINO result ledger is incomplete or duplicated')
    target_row_counts = Counter()
    seen_keys = set()
    seen_video_paths = set()
    videos = []
    for row in rows:
        seed = row.get('seed')
        expected = plan_by_seed.get(seed)
        key = row.get('key')
        if (expected is None or seed in rows_by_seed or key in seen_keys
                or type(row.get('layout_index')) is not int or type(row.get('target_count')) is not int
                or any(row.get(field) != expected[field] for field in (
                    'layout_index', 'layout_fingerprint', 'target_count', 'model_label', 'model_path',
                    'checkpoint_step', 'key'))
                or row.get('model_checkpoint') != model_path
                or type(row.get('success')) is not bool
                or type(row.get('planning_calls')) is not int or row['planning_calls'] not in (0, 1)
                or type(row.get('physical_attempts')) is not int
                or row['physical_attempts'] not in (0, 1)
                or (row['success'] and row['physical_attempts'] != 1)):
            raise RuntimeError('DINO ledger has a duplicate, mismatched or retried opportunity')
        if (row.get('inference_strategy') != 'hybrid_astar_act_trial4_dinov3_frozen'
                or row.get('act_chunk_schedule') != DINO_SCHEDULE
                or row.get('hybrid_control_profile') != control_profile):
            raise RuntimeError('DINO ledger action schedule or hybrid controller profile differs')
        if (row.get('video_error') or row.get('replay_error') or row.get('image_cleanup_error')
                or row.get('frame_images_deleted') is not True):
            raise RuntimeError('DINO retained media is incomplete or has an export error')
        # The child evaluator sets video_deferred when it hands encoding to the batch's
        # bounded media worker. The batch appends a row only after that worker has
        # materialized, decoded and finalized the MP4; status.media_pending is the
        # authoritative pending marker, while this field records the child path.
        if row.get('video_deferred') is not True:
            raise RuntimeError('DINO result did not use the expected bounded media pipeline')
        video = _resolve_dino_video(root, row.get('video_path'))
        if video in seen_video_paths:
            raise RuntimeError('DINO results share a retained video path')
        seen_video_paths.add(video)
        evidence = dict(key=key, path=video.relative_to(root).as_posix(),
            bytes=video.stat().st_size, sha256=sha256(video), frames=_decode_one_video_frame(video))
        videos.append(evidence)
        rows_by_seed[seed] = row
        seen_keys.add(key)
        target_row_counts[row['target_count']] += 1
    if set(rows_by_seed) != DINO_SEEDS or target_row_counts != Counter({target: 20 for target in range(1, 7)}):
        raise RuntimeError('DINO result ledger does not cover each planned seed exactly once')

    summary, successes = _validate_dino_summary(root, status, rows)
    videos.sort(key=lambda item: item['key'])
    return dict(profile=DINO_PROFILE, model_label=DINO_LABEL, checkpoint_step=DINO_STEP,
        model_path=model_path, weights_sha256=model['weights_sha256'], episodes=120,
        verified_videos=len(videos), strict_successes=successes, videos=videos,
        ledger_sha256=sha256(root / 'episodes.jsonl'), summary_sha256=sha256(root / 'summary.json'),
        metadata_sha256=sha256(root / 'metadata.json'), status_sha256=sha256(root / 'status.json'),
        plan_sha256=plan_hash, schedule_adapter_sha256=metadata['schedule_adapter_sha256'])


def dependency_ready(root, *, active_pids, profile=LEGACY_PROFILE):
    """Return a completion certificate for one explicit dependency protocol."""
    if profile == LEGACY_PROFILE:
        return _legacy_dependency_ready(root, active_pids=active_pids)
    if profile == DINO_PROFILE:
        return _dino_dependency_ready(root, active_pids=active_pids)
    raise ValueError(f'unknown dependency profile: {profile}')


def _dependency_json_snapshot(root):
    root = Path(root)
    names = ('metadata.json', 'episode_plan.json', 'episodes.jsonl', 'summary.json', 'status.json')
    return {name: sha256(root / name) for name in names}


def _verify_video_certificate(root, certificate):
    if certificate.get('profile') != DINO_PROFILE:
        return
    root = Path(root).resolve(strict=True)
    videos_root = root / 'videos'
    if videos_root.is_symlink() or not videos_root.is_dir():
        raise RuntimeError('dependency videos directory changed after verification')
    for item in certificate.get('videos', []):
        video = _resolve_dino_video(root, item.get('path'))
        if (video.stat().st_size != item.get('bytes') or sha256(video) != item.get('sha256')):
            raise RuntimeError('dependency video changed after verification')


def run_queue(bundle, after_batch, out, queue, *, start_index=120, device='cpu',
              poll_seconds=30., dependency_profile=LEGACY_PROFILE):
    bundle, after_batch, out, queue = map(lambda p: Path(p).resolve(),
        (bundle, after_batch, out, queue))
    if type(start_index) is not int or start_index < 6 or start_index % 6 or start_index >= 1_000_000 - 5:
        raise ValueError('new preflight needs a fresh six-aligned registered start index')
    if device != 'cpu':
        raise ValueError('authorized acceptance queue is CPU-only')
    if dependency_profile not in (LEGACY_PROFILE, DINO_PROFILE):
        raise ValueError(f'unknown dependency profile: {dependency_profile}')

    def overlap(a, b):
        return a == b or a.is_relative_to(b) or b.is_relative_to(a)

    if overlap(out, queue) or any(overlap(p, protected) for p in (out, queue)
            for protected in (bundle, after_batch)):
        raise ValueError('preflight/queue destinations must not overlap each other, bundle or running batch')
    if out.exists():
        raise FileExistsError('preflight output must be new; no automatic resume')
    queue.mkdir(parents=True, exist_ok=False)

    def publish(state, **extra):
        value = dict(state=state, pid=os.getpid(), time=time.time(), training_started=False, **extra)
        write_json(queue / 'status.json', value)
        return value

    try:
        started = time.time()
        deadline = started + 24 * 3600
        protected = {p: sha256(p) for p in (
            bundle / 'manifest.json', after_batch / 'metadata.json', after_batch / 'episode_plan.json')}
        identity = dict(bundle=str(bundle), after_batch=str(after_batch), out=str(out),
            start_index=start_index, dependency_profile=dependency_profile, device=device,
            queued_at=started, queue_deadline=deadline, waiting_counts_against_preflight_budget=False,
            preflight_limits=dict(hours=2, opportunities=24, output_gib=5, free_gib=8),
            formal_training_authorized=False,
            inference_lock_paths=os.environ.get('PAC_ACT_INFERENCE_LOCKS', '').split(os.pathsep),
            protected_sha256={str(p): h for p, h in protected.items()})
        write_json(queue / 'authorization.json', identity)
        pinned_ready = None
        pinned_files = None

        while True:
            if time.time() >= deadline:
                raise RuntimeError('queue deadline reached; no automatic extension')
            if any(sha256(p) != h for p, h in protected.items()):
                raise RuntimeError('queued identity changed')
            active = dependency_pids(after_batch)
            status = json.loads((after_batch / 'status.json').read_text())

            if pinned_ready is None:
                candidate = dependency_ready(after_batch, active_pids=active, profile=dependency_profile)
                if candidate is None:
                    publish('waiting_for_120_batch', dependency_pids=active)
                    time.sleep(poll_seconds)
                    continue
                pinned_ready = candidate
                pinned_files = _dependency_json_snapshot(after_batch)
            else:
                if status.get('state') == 'needs_attention':
                    raise RuntimeError('dependency needs_attention; no automatic recovery')
                if status.get('state') != 'complete' or active:
                    dependency_ready(after_batch, active_pids=active, profile=dependency_profile)
                    publish('waiting_for_120_batch', dependency_pids=active)
                    time.sleep(poll_seconds)
                    continue
                if _dependency_json_snapshot(after_batch) != pinned_files:
                    raise RuntimeError('dependency changed while waiting for inference slot')

            if not local_slot_available():
                publish('waiting_for_local_slot')
                time.sleep(poll_seconds)
                continue

            # Re-read the complete ledger, summary, status, and every video once the
            # inference slot is free. A valid but changed ledger is still refused.
            active = dependency_pids(after_batch)
            candidate = dependency_ready(after_batch, active_pids=active, profile=dependency_profile)
            if candidate is None:
                publish('waiting_for_clear_handoff', dependency_pids=active)
                time.sleep(poll_seconds)
                continue
            if candidate != pinned_ready or _dependency_json_snapshot(after_batch) != pinned_files:
                raise RuntimeError('dependency changed after initial verification')
            write_json(queue / 'dependency_verified.json', candidate)

            if any(sha256(p) != h for p, h in protected.items()):
                raise RuntimeError('queued identity changed before launch')
            if _dependency_json_snapshot(after_batch) != pinned_files:
                raise RuntimeError('dependency changed immediately before preflight')
            _verify_video_certificate(after_batch, candidate)

            if (_dependency_json_snapshot(after_batch) != pinned_files
                    or any(sha256(p) != h for p, h in protected.items())):
                raise RuntimeError('dependency changed immediately before preflight')

            # These are the final checks immediately before the exclusive marker.
            # Never signal or terminate a process if a new batch/encoder or lease appears.
            final_pids = dependency_pids(after_batch)
            slot_available = local_slot_available()
            if final_pids or not slot_available:
                publish('waiting_for_clear_handoff', dependency_pids=final_pids)
                time.sleep(poll_seconds)
                continue

            command = [sys.executable, '-B', str(bundle / 'source/scripts/train_pac_act.py'),
                '--mode', 'preflight', '--bundle', str(bundle), '--out', str(out), '--device', device,
                '--preflight-start-index', str(start_index)]
            with (queue / 'preflight_started.json').open('x') as stream:
                json.dump(dict(command=command, time=time.time(), authorized_once=True,
                    dependency_profile=dependency_profile), stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            break

        publish('preflight_running')
        with (queue / 'preflight.log').open('xb') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f'preflight exited {completed.returncode}; no automatic retry')
        report = json.loads((out / 'acceptance.json').read_text())
        if (report.get('engineering_passed') is not True or report.get('training_started') is not False
                or report.get('GPU_training_started') is not False):
            raise RuntimeError('preflight did not pass engineering checks')
        return publish('preflight_complete', engineering_passed=True, report=str(out / 'acceptance.json'))
    except BaseException as error:
        try:
            write_json(queue / 'status.json', dict(state='needs_attention',
                error=f'{type(error).__name__}: {error}', original_exception_repr=repr(error),
                automatic_retry=False, training_started=False, pid=os.getpid()))
        except BaseException as reporting:
            error.add_note('queue failure report: ' + repr(reporting))
        raise
