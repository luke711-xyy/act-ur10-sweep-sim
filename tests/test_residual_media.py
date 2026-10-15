"""Saved-state media only: tiny MuJoCo states and in-memory RGB, no rollout."""
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import imageio_ffmpeg
import mujoco
import numpy as np
import pytest


@pytest.fixture
def media():
    try:
        return importlib.import_module('residual_rl.media')
    except ModuleNotFoundError:
        return SimpleNamespace()


class Guard:
    def __init__(self):
        self.polls = 0

    def poll(self):
        self.polls += 1


def tiny_env():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <size nuserdata="2"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .08"><joint name="slide" type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody><actuator><motor joint="slide"/></actuator>
    </mujoco>''')
    return SimpleNamespace(model=model, data=mujoco.MjData(model))


def rows_for(env, count):
    rows = []
    for i in range(count):
        env.data.time = .5 + .01 * i
        env.data.qpos[:] = .001 * i
        env.data.qvel[:] = .2
        env.data.ctrl[:] = .3
        env.data.qfrc_applied[:] = .4
        env.data.userdata[:] = [7., 8.]
        values = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(env.model, env.data, values, mujoco.mjtState.mjSTATE_INTEGRATION)
        rows.append(dict(t=env.data.time, command=[.1, .2, .3, .4], tcp=[.1, .2, .3],
                         normal_force=.7, raw_normal_force=.8, final_collected=2,
                         active=True, phase='sweep', replay_state=dict(
                             spec=int(mujoco.mjtState.mjSTATE_INTEGRATION), values=values.tolist())))
    return rows


def write_trace(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return path


@pytest.mark.parametrize('count,indices', [
    (1, [0]), (4, [0, 3]), (5, [0, 4]), (6, [0, 4, 5]), (9, [0, 4, 8]),
])
def test_sampling_keeps_every_fourth_row_and_terminal_without_duplicate(media, tmp_path, count, indices):
    env = tiny_env()
    path = write_trace(tmp_path / 'trace.jsonl.gz', rows_for(env, count))
    guard = Guard()
    selected = list(media._sample_rows(path, env.model, guard))
    assert [i for i, _ in selected] == indices
    assert selected[-1][1]['t'] == pytest.approx(.5 + .01 * (count - 1))
    assert guard.polls >= count


@pytest.mark.parametrize('bad', [
    'missing_state', 'missing_values', 'wrong_spec', 'bool_spec', 'wrong_size', 'nested_state',
    'nan_state', 'infinite_state', 'string_state', 'bool_state', 'negative_clock',
    'nan_clock', 'clock_mismatch', 'subtle_clock_mismatch', 'repeated_clock', 'backward_clock', 'short_command',
    'bad_tcp', 'nan_force', 'bad_count', 'bad_active', 'missing_phase',
])
def test_sampling_rejects_bad_unsampled_row_instead_of_silently_skipping(media, tmp_path, bad):
    env = tiny_env()
    rows = rows_for(env, 6)
    row = rows[2]  # deliberately not selected for rendering
    state = row['replay_state']
    if bad == 'missing_state': del row['replay_state']
    elif bad == 'missing_values': del state['values']
    elif bad == 'wrong_spec': state['spec'] = int(mujoco.mjtState.mjSTATE_PHYSICS)
    elif bad == 'bool_spec': state['spec'] = True
    elif bad == 'wrong_size': state['values'].pop()
    elif bad == 'nested_state': state['values'] = [state['values']]
    elif bad == 'nan_state': state['values'][2] = float('nan')
    elif bad == 'infinite_state': state['values'][2] = float('inf')
    elif bad == 'string_state': state['values'][2] = '1.0'
    elif bad == 'bool_state': state['values'][2] = True
    elif bad == 'negative_clock': row['t'] = state['values'][0] = -1.
    elif bad == 'nan_clock': row['t'] = float('nan')
    elif bad == 'clock_mismatch': state['values'][0] += .02
    elif bad == 'subtle_clock_mismatch': state['values'][0] += 1e-9
    elif bad == 'repeated_clock': row['t'] = state['values'][0] = rows[1]['t']
    elif bad == 'backward_clock': row['t'] = state['values'][0] = .1
    elif bad == 'short_command': row['command'].pop()
    elif bad == 'bad_tcp': row['tcp'][0] = float('inf')
    elif bad == 'nan_force': row['normal_force'] = float('nan')
    elif bad == 'bad_count': row['final_collected'] = True
    elif bad == 'bad_active': row['active'] = 'true'
    elif bad == 'missing_phase': del row['phase']
    path = write_trace(tmp_path / 'bad.jsonl.gz', rows)
    with pytest.raises(ValueError):
        list(media._sample_rows(path, env.model, Guard()))


@pytest.mark.parametrize('payload', ['', '\n', '{}\n', '{not json}\n'])
def test_empty_or_malformed_trace_fails_closed(media, tmp_path, payload):
    path = tmp_path / 'bad.gz'
    with gzip.open(path, 'wt') as stream:
        stream.write(payload)
    with pytest.raises(ValueError):
        list(media._sample_rows(path, tiny_env().model, Guard()))


def test_restore_uses_real_integration_state_and_rebuilds_contact_forces_without_step(media, monkeypatch):
    env = tiny_env()
    row = rows_for(env, 1)[0]
    env.data.qpos[:] = 4.
    env.data.qvel[:] = 9.
    env.data.ctrl[:] = 0.
    env.data.userdata[:] = 0.
    env.data.time = 99.

    def forbidden(*args, **kwargs):
        pytest.fail('saved-state restoration must never integrate physics')

    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    media._restore_state(env, row)
    assert env.data.time == .5
    assert env.data.qpos[0] == 0.
    assert env.data.qvel[0] == .2
    assert env.data.ctrl[0] == .3
    assert env.data.qfrc_applied[0] == .4
    assert env.data.userdata.tolist() == [7., 8.]
    assert env.data.ncon > 0
    assert np.linalg.norm(env.data.cfrc_ext[1]) > 0  # requires mj_rnePostConstraint


def test_composite_places_all_five_views_and_metadata_without_image_files(media):
    colors = {
        'corner_xminus_yminus': (20, 0, 0), 'corner_xminus_yplus': (0, 40, 0),
        'corner_xplus_yminus': (0, 0, 60), 'corner_xplus_yplus': (80, 80, 0),
        'wrist': (90, 0, 90),
    }

    def render(camera, size):
        assert size == (320, 320)
        return np.full((320, 320, 3), colors[camera], dtype=np.uint8)

    env = tiny_env()
    env.render_rgb = render
    env.render_wrist_rgb = lambda size: render('wrist', size)
    row = rows_for(env, 1)[0]
    guard = Guard()
    frame = media._compose_frame(env, row, 3140001, guard)
    assert frame.shape == (640, 960, 3) and frame.dtype == np.uint8
    assert frame[160, 160].tolist() == [20, 0, 0]
    assert frame[160, 480].tolist() == [0, 40, 0]
    assert frame[480, 160].tolist() == [0, 0, 60]
    assert frame[480, 480].tolist() == [80, 80, 0]
    assert frame[160, 800].tolist() == [90, 0, 90]
    assert len(np.unique(frame[320:, 640:].reshape(-1, 3), axis=0)) > 1
    assert guard.polls >= 5


def test_actual_streaming_encoder_and_full_decoder_verify_counts_hash_and_shape(media, tmp_path):
    path = tmp_path / 'tiny.mp4'
    guard = Guard()
    frames = (np.full((640, 960, 3), i * 50, np.uint8) for i in range(3))
    assert media._encode_frames(path, frames, guard) == 3
    evidence = media._verify_video(path, 3, guard)
    assert (evidence['frames'], evidence['fps'], evidence['width'], evidence['height']) == (3, 25, 960, 640)
    assert evidence['size_bytes'] == path.stat().st_size > 0
    assert evidence['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        media._verify_video(path, 4, guard)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('size,fps', [('960x640', 24), ('320x320', 25)])
def test_verifier_rejects_actual_video_with_wrong_fps_or_dimensions(media, tmp_path, size, fps):
    path = tmp_path / 'wrong.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                    f'color=c=blue:s={size}:r={fps}', '-frames:v', '2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=10)
    with pytest.raises(ValueError):
        media._verify_video(path, 2, Guard())


def test_verifier_rejects_truncated_mp4(media, tmp_path):
    path = tmp_path / 'broken.mp4'
    path.write_bytes(b'not a complete MP4')
    with pytest.raises((ValueError, RuntimeError)):
        media._verify_video(path, 1, Guard())


def test_export_restores_sampled_states_once_and_publishes_verification_atomically(media, tmp_path, monkeypatch):
    env = tiny_env()
    rows = rows_for(env, 6)
    trace = write_trace(tmp_path / 'trace.gz', rows)
    cfg = object()
    resets, rendered = [], []

    def create(config, seed, guard):
        assert config is cfg
        resets.append(seed)
        return env

    def compose(current, row, seed, guard):
        assert current is env and seed == 42
        rendered.append((current.data.time, current.data.qpos[0]))
        return np.zeros((640, 960, 3), np.uint8)

    env.close = lambda: None
    env.step_control = lambda *args: pytest.fail('commands must never be replayed')
    monkeypatch.setattr(media, '_create_env', create)
    monkeypatch.setattr(media, '_compose_frame', compose)
    monkeypatch.setattr(mujoco, 'mj_step', lambda *args: pytest.fail('no integration after reset'))
    original_link = media.os.link

    def check_atomic(source, destination, *args, **kwargs):
        if str(destination).endswith('.json'):
            assert not Path(destination).exists()
            assert json.loads(Path(source).read_text())['verified'] is True
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(media.os, 'link', check_atomic)
    output = tmp_path / 'validation.mp4'
    result = media.export_validation_video(cfg, 42, trace, output, Guard())
    assert resets == [42]
    assert rendered == [(rows[i]['t'], .001 * i) for i in (0, 4, 5)]
    verification = output.with_suffix('.verification.json')
    assert json.loads(verification.read_text()) == result
    assert result['video_path'] == str(output)
    assert result['verification_path'] == str(verification)
    assert result['sampled_row_indices'] == [0, 4, 5]
    assert result['trace_rows'] == 6 and result['frames'] == 3
    assert result['seed'] == 42 and result['verified'] is True
    assert {p.name for p in tmp_path.iterdir()} == {'trace.gz', 'validation.mp4', 'validation.verification.json'}


@pytest.mark.parametrize('existing', ['video', 'verification', 'dangling_symlink'])
def test_export_refuses_existing_destinations_before_initializing_env(media, tmp_path, monkeypatch, existing):
    output = tmp_path / 'validation.mp4'
    target = output if existing != 'verification' else output.with_suffix('.verification.json')
    if existing == 'dangling_symlink': target.symlink_to(tmp_path / 'missing')
    else: target.write_text('preserve')
    monkeypatch.setattr(media, '_create_env', lambda *args: pytest.fail('must reject before reset'))
    with pytest.raises(FileExistsError):
        media.export_validation_video(object(), 1, tmp_path / 'trace.gz', output, Guard())
    if existing != 'dangling_symlink': assert target.read_text() == 'preserve'


def test_guard_abort_during_stream_removes_only_own_partial_media(media, tmp_path):
    path = tmp_path / 'aborted.mp4'
    unrelated = tmp_path / 'existing.mp4'
    unrelated.write_bytes(b'preserve')

    class Stop(RuntimeError): pass
    class StopGuard(Guard):
        def poll(self):
            super().poll()
            if self.polls >= 6: raise Stop('budget')

    def frames():
        while True: yield np.zeros((640, 960, 3), np.uint8)

    with pytest.raises(Stop):
        media._encode_frames(path, frames(), StopGuard())
    assert unrelated.read_bytes() == b'preserve'
    # The failed private helper may leave its own partial video, never evidence.
    assert not list(tmp_path.glob('*.json'))


def test_encoder_stall_has_bounded_timeout_and_reaps_only_its_child(media, tmp_path, monkeypatch):
    import sys
    import time

    owned = []
    real_popen = subprocess.Popen

    def stalled_encoder(*args, **kwargs):
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(20)'], **kwargs)
        owned.append(child)
        return child

    monkeypatch.setattr(media.subprocess, 'Popen', stalled_encoder)
    monkeypatch.setattr(media, '_PROCESS_TIMEOUT_SECONDS', .08)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        media._encode_frames(tmp_path / 'stalled.mp4', iter([np.zeros((640, 960, 3), np.uint8)]), Guard())
    assert time.monotonic() - started < 3.
    assert len(owned) == 1 and owned[0].poll() is not None


def test_create_env_initializes_and_resets_same_seed_once(media, monkeypatch):
    from sim.environments import sweep_env
    calls = []

    class Environment:
        def __init__(self, config, seed): calls.append(('init', config, seed))
        def reset(self, seed): calls.append(('reset', seed))
        def close(self): calls.append(('close',))

    monkeypatch.setattr(sweep_env, 'SweepEnv', Environment)
    cfg = object()
    guard = Guard()
    env = media._create_env(cfg, 13, guard)
    assert isinstance(env, Environment)
    assert calls == [('init', cfg, 13), ('reset', 13)]
    assert guard.polls >= 2


def test_tracewriter_numpy_rows_are_accepted_without_conversion_or_reexecution(media, tmp_path):
    from residual_rl.storage import TraceWriter

    env = tiny_env()
    rows = rows_for(env, 2)
    rows[0]['replay_state']['values'] = np.asarray(rows[0]['replay_state']['values'])
    rows[0]['command'] = np.asarray(rows[0]['command'], dtype=np.float32)
    trace = tmp_path / 'trace.jsonl.gz'
    writer = TraceWriter(trace)
    try:
        for row in rows: writer.append(row)
    finally:
        writer.close()
    before = trace.read_bytes()
    selected = list(media._sample_rows(trace, env.model, Guard()))
    assert [i for i, _ in selected] == [0, 1]
    assert trace.read_bytes() == before


@pytest.mark.parametrize('reason', ['malformed', 'cancelled'])
def test_failed_export_closes_env_and_leaves_no_movie_or_evidence(media, tmp_path, monkeypatch, reason):
    env = tiny_env()
    rows = rows_for(env, 6)
    if reason == 'malformed': del rows[2]['replay_state']
    trace = write_trace(tmp_path / 'trace.gz', rows)
    before = trace.read_bytes()
    closed = []
    env.close = lambda: closed.append(True)
    monkeypatch.setattr(media, '_create_env', lambda *args: env)

    class Cancelled(RuntimeError): pass

    def compose(*args):
        if reason == 'cancelled': raise Cancelled('guard cancelled during rendering')
        return np.zeros((640, 960, 3), np.uint8)

    monkeypatch.setattr(media, '_compose_frame', compose)
    output = tmp_path / 'validation.mp4'
    with pytest.raises(ValueError if reason == 'malformed' else Cancelled):
        media.export_validation_video(object(), 1, trace, output, Guard())
    assert closed == [True]
    assert list(tmp_path.iterdir()) == [trace]
    assert trace.read_bytes() == before


def test_concurrent_output_creator_is_never_overwritten(media, tmp_path, monkeypatch):
    env = tiny_env()
    trace = write_trace(tmp_path / 'trace.gz', rows_for(env, 1))
    env.close = lambda: None
    output = tmp_path / 'validation.mp4'
    monkeypatch.setattr(media, '_create_env', lambda *args: env)
    monkeypatch.setattr(media, '_compose_frame', lambda *args: np.zeros((640, 960, 3), np.uint8))
    real_verify = media._verify_video

    def racing_verify(*args):
        result = real_verify(*args)
        output.write_bytes(b'concurrent creator')
        return result

    monkeypatch.setattr(media, '_verify_video', racing_verify)
    with pytest.raises(FileExistsError):
        media.export_validation_video(object(), 1, trace, output, Guard())
    assert output.read_bytes() == b'concurrent creator'
    assert not output.with_suffix('.verification.json').exists()
    assert {p.name for p in tmp_path.iterdir()} == {'trace.gz', 'validation.mp4'}


def test_cancelled_guard_during_full_decode_reaps_its_child(media, tmp_path, monkeypatch):
    path = tmp_path / 'tiny.mp4'
    media._encode_frames(path, iter([np.zeros((640, 960, 3), np.uint8)]), Guard())
    before = path.read_bytes()
    owned = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        owned.append(child)
        return child

    class Cancelled(RuntimeError): pass
    class StopGuard(Guard):
        def poll(self):
            super().poll()
            if self.polls == 3: raise Cancelled('budget')

    monkeypatch.setattr(media.subprocess, 'Popen', capture)
    with pytest.raises(Cancelled):
        media._verify_video(path, 1, StopGuard())
    assert len(owned) == 1 and owned[0].poll() is not None
    assert path.read_bytes() == before


def test_create_env_closes_its_renderer_resources_if_post_reset_guard_aborts(media, monkeypatch):
    from sim.environments import sweep_env
    closed = []

    class Environment:
        def __init__(self, config, seed): pass
        def reset(self, seed): pass
        def close(self): closed.append(True)

    class StopGuard(Guard):
        def poll(self):
            super().poll()
            if self.polls == 2: raise RuntimeError('budget')

    monkeypatch.setattr(sweep_env, 'SweepEnv', Environment)
    with pytest.raises(RuntimeError, match='budget'):
        media._create_env(object(), 1, StopGuard())
    assert closed == [True]
