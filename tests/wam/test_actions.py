"""Hand-derived action geometry, lifecycle and frozen-origin regressions."""
import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wam_sweep.contracts import CandidateProgram


def api():
    assert importlib.util.find_spec("wam_sweep.actions"), "Task3 action codec is absent"
    from wam_sweep import actions
    return actions


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_action_preserves_native_bytes(dtype):
    q = np.array([.1, -0., .2, math.pi], dtype=dtype)
    decoded = api().ActionCodec().decode(np.zeros(4), q, np.zeros(3), .04, True)
    assert decoded.target.tobytes() == q.tobytes()
    assert decoded.target.dtype == q.dtype
    assert not np.shares_memory(decoded.target, q)
    np.testing.assert_array_equal(decoded.residual, [0., 0., 0.])


@pytest.mark.parametrize("actual_x,want_x,radius", [(0., .18, .08), (.05, .14, .04), (.2, .02, .08)])
def test_mapping_uses_current_actual_tcp_without_historical_anchor(actual_x, want_x, radius):
    codec = api().ActionCodec()
    q = np.array([.1, 0., .2, 0.])
    result = codec.decode([actual_x, 0., .9, 2.], q, [1., 0., 0.], 10., True)
    np.testing.assert_allclose(result.target, [want_x, 0., .2, 0.], atol=1e-15)
    assert result.audit["correction_radius"] == pytest.approx(radius)
    assert result.audit["helper_sha256"] == "f185c594c0a539c16e100ef7bf3f6c88cd2393ee2276fdc5e857d425108a41d8"


def test_zero_displacement_removes_xy_but_keeps_bounded_yaw_and_z():
    q = np.array([.1, .2, .3, 0.])
    result = api().ActionCodec().decode([.1, .2, 8., 0.], q, [1., -1., 1.], 10., True)
    np.testing.assert_allclose(result.target, [.1, .2, .3, math.pi / 4], atol=1e-15)


def test_yaw_wraps_across_pi_without_large_residual():
    q = np.array([.1, 0., .2, math.radians(179)])
    result = api().ActionCodec().decode(np.zeros(4), q, [0., 0., 1.], 10., True)
    assert result.target[3] == pytest.approx(math.radians(-136))
    assert result.residual[2] == pytest.approx(math.pi / 4)


def test_repeated_decode_never_accumulates_or_mutates_nominal_chunk():
    codec = api().ActionCodec()
    q = np.array([.1, 0., .2, 0.], dtype=np.float32)
    original = q.tobytes()
    for _ in range(25):
        result = codec.decode(np.zeros(4), q, [1., 0., 0.], 10., True)
        assert result.target[0] == pytest.approx(.18)
        assert q.tobytes() == original


def test_filter_response_is_exponential_in_normalized_raw_coordinates():
    codec = api().ActionCodec()
    q = np.array([.1, 0., .2, 0.])
    first = codec.decode(np.zeros(4), q, [1., -1., .5], .04, True)
    np.testing.assert_allclose(first.filtered, [.18126924692201818, -.18126924692201818, .09063462346100909])
    second = codec.decode(np.zeros(4), q, [1., -1., .5], .04, True)
    np.testing.assert_allclose(second.filtered, [.3296799539643607, -.3296799539643607, .16483997698218035])


def test_hold_emits_native_bytes_and_clears_filter_and_raw_anchor():
    codec = api().ActionCodec()
    q = np.array([.1, -0., .2, 4.], dtype=np.float32)
    codec.decode(np.zeros(4), q, [1., 1., 1.], .04, True)
    codec.begin_program()
    held = codec.decode(np.zeros(4), q, [1., 1., 1.], .04, False)
    assert held.target.tobytes() == q.tobytes()
    assert held.active is False
    np.testing.assert_array_equal(held.residual, [0., 0., 0.])
    np.testing.assert_array_equal(codec.current_raw, [0., 0., 0.])
    assert codec.snapshot()["program_anchor"] == [0., 0., 0.]
    resumed = codec.decode(np.zeros(4), q, [0., 0., 0.], .04, True)
    assert resumed.target.tobytes() == q.tobytes()


def test_pchip_anchors_now_and_interpolates_channels_independently():
    codec = api().ActionCodec()
    program = CandidateProgram(np.array([[1., 1., .2], [0., 1., .4], [0., 1., .6], [0., 1., .8], [0., 1., 1.]]))
    anchor = np.array([0., 1., 0.])
    before = codec.snapshot()
    np.testing.assert_array_equal(codec.sample(program, 0., anchor), anchor)
    np.testing.assert_allclose(codec.sample(program, .1, anchor), [.75, 1., .1], atol=1e-15)
    np.testing.assert_allclose(codec.sample(program, .3, anchor), [.5, 1., .3], atol=1e-15)
    for j, t in enumerate([.2, .4, .6, .8, 1.]):
        np.testing.assert_allclose(codec.sample(program, t, anchor), program.knots[j], atol=1e-15)
    assert codec.snapshot() == before


def test_program_anchor_is_frozen_until_next_real_program_start():
    codec = api().ActionCodec()
    q = np.array([.1, 0., .2, 0.])
    codec.decode(np.zeros(4), q, [.6, -.3, .2], .04, True)
    anchor = codec.begin_program()
    np.testing.assert_array_equal(anchor, [.6, -.3, .2])
    root = codec.snapshot()
    assert root["program_anchor"] == [.6, -.3, .2]
    program = CandidateProgram(np.ones((5, 3)))
    for j in range(5):
        raw = codec.sample(program, j * .04, anchor)
        codec.decode(np.zeros(4), q, raw, .04, True)
        assert codec.snapshot()["program_anchor"] == [.6, -.3, .2]
        np.testing.assert_array_equal(codec.sample(program, 0., anchor), [.6, -.3, .2])
    assert codec.current_raw.tolist() != root["current_raw"]
    branch = api().ActionCodec()
    branch.restore(root)
    np.testing.assert_allclose(branch.sample(program, .1, branch.snapshot()["program_anchor"]), [.875, .59375, .75], atol=1e-15)
    newest_raw = codec.current_raw
    np.testing.assert_array_equal(codec.begin_program(), newest_raw)
    assert codec.snapshot()["program_anchor"] == newest_raw.tolist()
    assert root["program_anchor"] == [.6, -.3, .2]


def test_safe_initial_snapshot_whitelists_normalized_state_and_round_trips():
    codec = api().ActionCodec()
    state = codec.snapshot()
    assert set(state) == {"protocol", "helper_sha256", "bounds", "program_anchor", "current_raw", "filtered", "residual", "frames"}
    for key in ("program_anchor", "current_raw", "filtered", "residual"):
        assert state[key] == [0., 0., 0.]
    clone = api().ActionCodec()
    clone.restore(state)
    assert clone.snapshot() == state
    q = np.array([.1, -0., .2, 0.], dtype=np.float32)
    assert clone.decode(np.zeros(4), q, [0., 0., 0.], .04, True).target.tobytes() == q.tobytes()


def test_zero_physical_xy_does_not_erase_unobservable_normalized_filter():
    codec = api().ActionCodec()
    still = np.array([.1, 0., .2, 0.])
    prior = codec.decode(still, still, [1., 0., 0.], .04, True)
    np.testing.assert_array_equal(prior.residual, [0., 0., 0.])
    root = codec.snapshot()
    assert root["filtered"][0] == pytest.approx(.18126924692201818)
    clone = api().ActionCodec()
    clone.restore(root)
    continued = clone.decode(np.zeros(4), still, [0., 0., 0.], .04, True)
    assert continued.filtered[0] == pytest.approx(.14841070704234255)
    assert continued.target[0] == pytest.approx(.1118728565633874)
    fresh = api().ActionCodec().decode(np.zeros(4), still, [0., 0., 0.], .04, True)
    assert fresh.target[0] == .1


def test_five_candidates_clone_filter_and_raw_anchor_without_execution_pollution():
    codec = api().ActionCodec()
    q = np.array([.1, 0., .2, 0.])
    codec.decode(np.zeros(4), q, [.2, -.1, .3], .04, True)
    codec.begin_program()
    initial = codec.snapshot()
    outputs = []
    for sign in [1., -1., .5, 0., 1.]:
        branch = api().ActionCodec()
        branch.restore(initial)
        anchor = initial["program_anchor"]
        program = CandidateProgram(np.full((5, 3), sign))
        outputs.append([branch.decode(np.zeros(4), q, branch.sample(program, j * .04, anchor), .04, True).target for j in range(25)])
        assert codec.snapshot() == initial
    np.testing.assert_array_equal(outputs[0], outputs[4])
    assert not np.array_equal(outputs[0], outputs[1])
    replay = api().ActionCodec()
    replay.restore(initial)
    expected = replay.decode(np.zeros(4), q, [0., 0., 0.], .04, True)
    actual = codec.decode(np.zeros(4), q, [0., 0., 0.], .04, True)
    np.testing.assert_array_equal(actual.target, expected.target)
    np.testing.assert_array_equal(actual.residual_rate, expected.residual_rate)


@pytest.mark.parametrize("raw", [[1., 1., 1.], [-1., 1., -1.], [1., -1., .5], [-1., -1., 0.]])
def test_all_25_steps_generate_within_workspace_disk_and_local_forward(raw):
    codec = api().ActionCodec(bounds=[[0., 1.], [0., 1.]])
    for j in range(25):
        p = np.array([.7 + j * .005, .45 + j * .001, .7, 0.])
        q = np.array([.999, .001, .2, math.radians(179)])
        result = codec.decode(p, q, raw, .04, True)
        delta = result.target[:2] - q[:2]
        direction = q[:2] - p[:2]
        assert np.linalg.norm(delta) <= .8 * np.linalg.norm(direction) + 1e-15
        assert np.dot(result.target[:2] - p[:2], direction) > 0.
        assert np.all(result.target[:2] >= 0.) and np.all(result.target[:2] <= 1.)
        assert result.target[2] == q[2]
        assert abs(result.residual[2]) <= math.pi / 4 + 1e-15


def test_boundary_generation_scales_the_ray_instead_of_clipping_coordinates():
    q = np.array([.99, .5, .2, 0.])
    result = api().ActionCodec(bounds=[[0., 1.], [0., 1.]]).decode(
        [.8, .5, 0., 0.], q, [1., 1., 0.], 10., True)
    # At 45 degrees the X boundary is .01 away: BOTH coordinates move .01.
    np.testing.assert_allclose(result.target, [1., .51, .2, 0.], atol=1e-15)


@pytest.mark.parametrize("field,value", [("helper_sha256", "0" * 64), ("protocol", "old"), ("bounds", [[0., 1.], [0., 1.]]), ("filtered", [2., 0., 0.]), ("current_raw", [float("nan"), 0., 0.]), ("program_anchor", [-2., 0., 0.])])
def test_restore_rejects_incompatible_or_poisoned_state_atomically(field, value):
    codec = api().ActionCodec()
    before = codec.snapshot()
    poisoned = copy.deepcopy(before)
    poisoned[field] = value
    with pytest.raises(ValueError):
        codec.restore(poisoned)
    assert codec.snapshot() == before


@pytest.mark.parametrize("case", ["raw", "tcp", "target", "clock", "inactive_raw"])
def test_invalid_decode_does_not_change_filter_state(case):
    codec = api().ActionCodec()
    p, q, raw, dt, active = np.zeros(4), np.array([.1, 0., .2, 0.]), np.zeros(3), .04, True
    if case in {"raw", "inactive_raw"}:
        raw[0] = 1.1
        active = case != "inactive_raw"
    elif case == "tcp":
        p[0] = np.nan
    elif case == "target":
        q = np.zeros(4, dtype=int)
    else:
        dt = 0.
    before = codec.snapshot()
    with pytest.raises(ValueError):
        codec.decode(p, q, raw, dt, active)
    assert codec.snapshot() == before


@pytest.mark.parametrize("t", [-.01, 1.01, float("nan")])
def test_sample_rejects_out_of_horizon_clock(t):
    with pytest.raises(ValueError):
        api().ActionCodec().sample(CandidateProgram(np.zeros((5, 3))), t, np.zeros(3))


def test_sample_revalidates_mutated_candidate_payload():
    program = CandidateProgram(np.zeros((5, 3)))
    program.knots[0, 0] = 2.
    with pytest.raises(ValueError):
        api().ActionCodec().sample(program, .1, np.zeros(3))


def test_helper_drift_fails_closed_without_executing_legacy_package(tmp_path, monkeypatch):
    module = api()
    drifted = tmp_path / "relative_action.py"
    drifted.write_bytes(module.HELPER_PATH.read_bytes() + b"\nraise RuntimeError('must not execute')\n")
    monkeypatch.setattr(module, "HELPER_PATH", drifted)
    with pytest.raises(ValueError, match="helper.*hash"):
        module.ActionCodec()


def test_child_codec_preserves_actual_native_origin_guard_and_numeric_behavior():
    api()
    code = """
import sys
import numpy as np
from wam_sweep.inputs import verify_inputs, assert_native_origin
from wam_sweep.actions import ActionCodec
identity = verify_inputs('/Users/huazhi_luke/.local/share/wam-sweep-v1/inputs/r8', allow_unlisted=False)
assert_native_origin(identity)
codec = ActionCodec()
q = np.array([.1, -0., .2, 0.], dtype=np.float32)
assert codec.decode(np.zeros(4), q, np.zeros(3), .04, True).target.tobytes() == q.tobytes()
np.testing.assert_allclose(codec.decode(np.zeros(4), q, [1.,0.,0.], 10., True).target, [.18,0.,.2,0.], atol=1e-8)
assert not any(n == 'residual_rl' or n.startswith('residual_rl.') for n in sys.modules)
assert_native_origin(identity)
print('codec: zero bytes, known geometry, no residual_rl imports, actual guard passed')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert "actual guard passed" in result.stdout
