"""Independent saved-state validation movies; never execute recorded commands.

The caller supplies the exact episode configuration and seed. Only the initial
SweepEnv.reset() settles physics. Subsequent images restore mjSTATE_INTEGRATION.
No runtime, policy, learner, or checkpoint module is imported here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import tempfile
import time

import numpy as np


_PROCESS_TIMEOUT_SECONDS = 300.
_WIDTH, _HEIGHT, _FPS = 960, 640, 25
_VIEWS = ('corner_xminus_yminus', 'corner_xminus_yplus',
          'corner_xplus_yminus', 'corner_xplus_yplus', 'wrist')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('trace values must be finite JSON numbers')
    return value


def _vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'trace vector must have length {length}')
    for item in value:
        _number(item)
    return np.asarray(value, dtype=np.float64)


def _state_values(model, row):
    import mujoco

    state = row.get('replay_state')
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    if not isinstance(state, dict) or type(state.get('spec')) is not int or state['spec'] != spec:
        raise ValueError('every trace row requires an exact mjSTATE_INTEGRATION state')
    values = _vector(state.get('values'), mujoco.mj_stateSize(model, spec))
    t = _number(row.get('t'))
    # Both clocks originate from the very same mjData.time double. JSON
    # round-trips them exactly; accepting a tolerance could hide a backward
    # saved-state clock behind an increasing metadata clock.
    if t < 0 or t != values[0]:
        raise ValueError('trace clock differs from saved integration clock')
    return values


def _validate_row(model, row, previous_time):
    if not isinstance(row, dict):
        raise ValueError('trace rows must be objects')
    _state_values(model, row)
    if row['t'] <= previous_time:
        raise ValueError('trace clock must be strictly increasing')
    _vector(row.get('command'), 4)
    _vector(row.get('tcp'), 3)
    _number(row.get('normal_force'))
    _number(row.get('raw_normal_force'))
    count = row.get('final_collected')
    if type(count) is not int or not 0 <= count <= 6:
        raise ValueError('invalid collected count')
    if type(row.get('active')) is not bool or not isinstance(row.get('phase'), str) or not row['phase']:
        raise ValueError('invalid active flag or phase')


def _sample_rows(trace_path, model, guard):
    """Validate ALL rows, yield zero-based 0,4,8,... and the terminal row once."""
    previous_time, last, last_selected = -math.inf, None, -1
    with gzip.open(trace_path, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            guard.poll()
            try:
                row = json.loads(line)
                _validate_row(model, row, previous_time)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f'invalid trace row {index}: {error}') from error
            previous_time = row['t']
            last = index, row
            if index % 4 == 0:
                last_selected = index
                yield last
    if last is None:
        raise ValueError('validation trace is empty')
    if last[0] != last_selected:
        guard.poll()
        yield last


def _create_env(cfg, seed, guard):
    from sim.environments.sweep_env import SweepEnv

    guard.poll()
    env = SweepEnv(cfg, seed=seed)
    try:
        env.reset(seed=seed)  # reset already performs the one initial settle
        guard.poll()
    except BaseException:
        env.close()
        raise
    return env


def _restore_state(env, row):
    import mujoco

    values = _state_values(env.model, row)
    mujoco.mj_setState(env.model, env.data, values, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_rnePostConstraint(env.model, env.data)


def _compose_frame(env, row, seed, guard):
    from PIL import Image, ImageDraw

    frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    positions = ((0, 0), (320, 0), (0, 320), (320, 320), (640, 0))
    for view, (x, y) in zip(_VIEWS, positions):
        guard.poll()
        rgb = (env.render_wrist_rgb(size=(320, 320)) if view == 'wrist'
               else env.render_rgb(view, size=(320, 320)))
        if rgb.shape != (320, 320, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'{view} must render 320x320 uint8 RGB')
        frame[y:y+320, x:x+320] = rgb
        guard.poll()
    panel = Image.new('RGB', (320, 320), (18, 22, 29))
    draw = ImageDraw.Draw(panel)
    lines = [
        'SAVED-STATE VALIDATION', f'seed: {seed}', f't: {row["t"]:.3f} s',
        f'phase: {row["phase"]}', f'active: {row["active"]}',
        f'collected: {row["final_collected"]}',
        f'normal force: {row["normal_force"]:.4f}',
        f'raw force: {row["raw_normal_force"]:.4f}',
        'command XYZYaw:', '  ' + '  '.join(f'{v:.4f}' for v in row['command']),
        'TCP: ' + '  '.join(f'{v:.4f}' for v in row['tcp']),
        '100 Hz trace / stride 4 / 25 fps',
    ]
    for index, line in enumerate(lines):
        draw.text((12, 12 + 24 * index), line, fill=(225, 232, 242))
    frame[320:, 640:] = np.asarray(panel)
    return frame


def _poll(guard, deadline):
    guard.poll()
    if time.monotonic() >= deadline:
        raise TimeoutError('owned ffmpeg process exceeded its bounded timeout')


def _wait(process, guard, deadline):
    while True:
        _poll(guard, deadline)
        try:
            return process.wait(timeout=.1)
        except subprocess.TimeoutExpired:
            pass


def _cleanup_process(process):
    """Only this Popen's child is owned; never signal a process group or peers."""
    if process.poll() is None:
        process.kill()
        process.wait(timeout=1.)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _process_error(log):
    log.seek(0, os.SEEK_END)
    log.seek(max(0, log.tell() - 4096))
    return RuntimeError('ffmpeg failed: ' + log.read().decode('utf-8', errors='replace'))


def _encode_frames(path, frames, guard):
    import imageio_ffmpeg

    guard.poll()
    deadline = time.monotonic() + min(300., _PROCESS_TIMEOUT_SECONDS)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-v', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x640', '-r', '25',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-threads', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               '-f', 'mp4', '-n', str(path)]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            os.set_blocking(process.stdin.fileno(), False)
            count = 0
            for frame in frames:
                _poll(guard, deadline)
                if frame.shape != (_HEIGHT, _WIDTH, 3) or frame.dtype != np.uint8:
                    raise ValueError('encoder frames must be 960x640 uint8 RGB')
                remaining = memoryview(np.ascontiguousarray(frame)).cast('B')
                while remaining:
                    _poll(guard, deadline)
                    if process.poll() is not None:
                        raise _process_error(log)
                    if select.select([], [process.stdin], [], .1)[1]:
                        try:
                            sent = os.write(process.stdin.fileno(), remaining[:65536])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError as error:
                            raise _process_error(log) from error
                        remaining = remaining[sent:]
                count += 1
            if count == 0:
                raise ValueError('cannot encode an empty validation movie')
            process.stdin.close()
            if _wait(process, guard, deadline):
                raise _process_error(log)
            return count
        finally:
            _cleanup_process(process)


def _verify_video(path, expected_frames, guard):
    """Decode every frame without rescaling/resampling; verify actual timing/size."""
    import imageio_ffmpeg

    guard.poll()
    path = Path(path)
    if type(expected_frames) is not int or expected_frames < 1 or path.stat().st_size <= 0:
        raise ValueError('empty movie or invalid expected frame count')
    deadline = time.monotonic() + min(300., _PROCESS_TIMEOUT_SECONDS)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-v', 'info',
               '-xerror', '-err_detect', 'explode', '-i', str(path), '-map', '0:v:0',
               '-an', '-sn', '-dn', '-vf', 'showinfo', '-vsync', '0', '-threads', '1',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
        try:
            os.set_blocking(process.stdout.fileno(), False)
            decoded_bytes = 0
            while True:
                _poll(guard, deadline)
                if not select.select([process.stdout], [], [], .1)[0]:
                    continue
                block = os.read(process.stdout.fileno(), 1024 * 1024)
                if not block:
                    break
                decoded_bytes += len(block)
            if _wait(process, guard, deadline):
                raise _process_error(log)
        finally:
            _cleanup_process(process)
        log.seek(0)
        frames, found_rate = 0, False
        for raw_line in log:
            _poll(guard, deadline)
            line = raw_line.decode('utf-8', errors='replace')
            if 'showinfo' not in line:
                continue
            rate = re.search(r'config in .*frame_rate:\s*(\d+)/(\d+)', line)
            if rate:
                numerator, denominator = map(int, rate.groups())
                if denominator == 0 or numerator != _FPS * denominator:
                    raise ValueError('decoded video is not 25 fps')
                found_rate = True
            info = re.search(r'\bn:\s*(\d+).*?pts_time:\s*([-+0-9.eE]+).*?\bs:(\d+)x(\d+)', line)
            if info:
                index, pts, width, height = info.groups()
                if (int(index) != frames or (int(width), int(height)) != (_WIDTH, _HEIGHT)
                        or not math.isclose(float(pts), frames / _FPS, rel_tol=0., abs_tol=1e-5)):
                    raise ValueError('decoded video dimensions or frame timestamps differ')
                frames += 1
        if not found_rate or frames != expected_frames or decoded_bytes != frames * _WIDTH * _HEIGHT * 3:
            raise ValueError('full decode did not match expected video frames')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            guard.poll()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return dict(frames=frames, fps=_FPS, width=_WIDTH, height=_HEIGHT,
                size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def export_validation_video(cfg, seed, trace_path, output_path, guard) -> dict:
    """Publish a new MP4 and sibling ``.verification.json``, returning that JSON.

    ``cfg`` is the original SweepEnv configuration, not PilotConfig. The sidecar
    is the atomic validation commit marker. Existing files (including dangling
    symlinks) are never replaced. Failed exports have no validation marker.
    """
    output = Path(os.path.abspath(output_path))
    verification = output.with_suffix('.verification.json')
    for path in (output, verification):
        if os.path.lexists(path):
            raise FileExistsError(path)
    if output.suffix.lower() != '.mp4' or type(seed) is not int:
        raise ValueError('validation export requires an MP4 path and integer seed')
    trace = Path(trace_path).absolute()
    if not trace.is_file():
        raise FileNotFoundError(trace)
    guard.poll()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = _create_env(cfg, seed, guard)
    try:
        with tempfile.TemporaryDirectory(prefix='.' + output.stem + '-', dir=output.parent) as directory:
            staged = Path(directory) / 'video.mp4'
            indices = []

            def frames():
                for index, row in _sample_rows(trace, env.model, guard):
                    guard.poll()
                    _restore_state(env, row)
                    guard.poll()
                    frame = _compose_frame(env, row, seed, guard)
                    indices.append(index)
                    yield frame

            count = _encode_frames(staged, frames(), guard)
            evidence = _verify_video(staged, count, guard)
            evidence.update(schema=1, verified=True, source='saved_mjSTATE_INTEGRATION',
                            seed=seed, trace_path=str(trace), trace_rows=indices[-1] + 1,
                            sampled_row_indices=indices, video_path=str(output),
                            verification_path=str(verification))
            pending = Path(directory) / 'verification.json'
            with pending.open('x', encoding='utf-8') as stream:
                json.dump(evidence, stream, allow_nan=False, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            with staged.open('rb') as stream:
                os.fsync(stream.fileno())
            guard.poll()
            # Same-filesystem hard links publish complete files atomically and
            # fail on a concurrent creator; replace()/ffmpeg -y could overwrite.
            os.link(staged, output)
            os.link(pending, verification)
            return evidence
    finally:
        env.close()
