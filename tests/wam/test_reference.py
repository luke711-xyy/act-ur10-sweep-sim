"""Causal preview tests against the verified frozen scheduler, no physics."""
import importlib.util
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wam_sweep.inputs import assert_native_origin, verify_inputs


def api():
    assert importlib.util.find_spec("wam_sweep.reference"), "Task3 causal reference preview is absent"
    from wam_sweep import reference
    return reference


class ActSettings(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture(scope="module")
def frozen():
    identity = verify_inputs("/Users/huazhi_luke/.local/share/wam-sweep-v1/inputs/r8", allow_unlisted=False)
    alias = "_wam_task3_verified_scheduler"
    assert alias not in sys.modules
    assert_native_origin(identity, native_aliases=(alias,))
    path = Path(identity.root) / "source/sim/act/realtime.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    assert_native_origin(identity, native_aliases=(alias,))
    yield module
    # Keep the checked alias registered, exactly as a real backend would.
    assert_native_origin(identity, native_aliases=(alias,))


def scheduler(frozen, ensemble=False):
    cfg = SimpleNamespace(act=ActSettings(query_hz=5, action_hz=25, execute_steps=5, realtime_budget_seconds=.2, temporal_ensemble_coeff=.1 if ensemble else None))
    return frozen.ActionChunkScheduler(cfg)


def marked_chunk(length=25):
    values = np.zeros((length, 4), dtype=np.float32)
    for j in range(length):
        values[j] = [float(j), 0., .2, 0.]
    return values


@pytest.mark.parametrize("ensemble", [False, True])
def test_valid_from_uses_age_index_five(frozen, ensemble):
    module = api()
    live = scheduler(frozen, ensemble)
    assert live.accept(0., marked_chunk(), .2)
    before = pickle.dumps(live)
    preview = module.preview_reference(live, np.array([-1., 0., .2, 0.]), .2)
    selected = preview.targets[0]
    assert selected[0] == 5.0
    assert preview.targets[19, 0] == 24.
    assert preview.valid.tolist() == [1.] * 20 + [0.] * 5
    np.testing.assert_array_equal(preview.targets[20:], np.tile([-1., 0., .2, 0.], (5, 1)))
    assert pickle.dumps(live) == before


def test_command_times_start_now_and_hold_before_valid_and_after_expiry(frozen):
    live = scheduler(frozen)
    assert live.accept(0., marked_chunk(10), .2)
    before = pickle.dumps(live)
    preview = api().preview_reference(live, np.array([-.5, 0., .3, 0.]), 0.)
    np.testing.assert_allclose(preview.times[:6], [0., .04, .08, .12, .16, .2], atol=1e-15)
    assert preview.times[-1] == pytest.approx(.96)
    assert preview.issued_at == 0.
    assert preview.valid.tolist() == [0.] * 5 + [1.] * 5 + [0.] * 15
    np.testing.assert_array_equal(preview.targets[:5], np.tile([-.5, 0., .3, 0.], (5, 1)))
    assert preview.targets[5, 0] == 5.
    assert preview.targets[-1, 0] == -.5
    assert pickle.dumps(live) == before


def test_preview_does_not_query_or_consume_expired_ensemble(frozen):
    live = scheduler(frozen, True)
    assert live.accept(0., marked_chunk(5), .19)
    live.mark_query(.1)
    before = pickle.dumps(live)
    preview = api().preview_reference(live, np.array([-.5, 0., .3, 4.]), .4)
    assert preview.valid.tolist() == [0.] * 25
    np.testing.assert_array_equal(preview.targets, np.tile([-.5, 0., .3, 4.], (25, 1)))
    assert len(live.temporal_ensemble.chunks) == 1
    assert live.next_query == pytest.approx(.3)
    assert pickle.dumps(live) == before


def test_expired_preview_keeps_root_held_reference_without_nominal_shadow(frozen):
    module = api()
    live = scheduler(frozen)
    assert live.accept(0., marked_chunk(10), .2)
    held = np.array([-.5, .5, .9, 4.])
    context = module.NativeTargetContext(xy_bounds=((0., 1.), (0., 1.)), z_bounds=(.1, .3))
    before = pickle.dumps(live)
    preview = module.preview_reference(live, held, .2, native_context=context)
    assert preview.valid.tolist() == [1.] * 5 + [0.] * 20
    # No simulated accepted-reference/control integration exists in a preview.
    # Its invalid placeholders must preserve the supplied reference verbatim.
    np.testing.assert_array_equal(preview.targets[5:], np.tile([-.5, .5, .9, 4.], (20, 1)))
    assert pickle.dumps(live) == before


@pytest.mark.parametrize("ensemble", [False, True])
def test_preview_rejects_future_issued_chunks_at_root(frozen, ensemble):
    live = scheduler(frozen, ensemble)
    assert live.accept(.4, marked_chunk(), .6)
    before = pickle.dumps(live)
    with pytest.raises(ValueError, match="future"):
        api().preview_reference(live, np.zeros(4), .2)
    assert pickle.dumps(live) == before


def test_temporal_fusion_wraps_yaw_across_pi(frozen):
    live = scheduler(frozen, True)
    for yaw in [np.deg2rad(179), np.deg2rad(-179)]:
        values = np.tile([.5, .5, .2, yaw], (25, 1))
        assert live.accept(0., values, .2)
    preview = api().preview_reference(live, np.zeros(4), .2)
    assert abs(preview.targets[0, 3]) == pytest.approx(np.pi, abs=2e-7)


def test_native_normalization_occurs_after_fusion_and_zero_codec_keeps_bytes(frozen):
    module = api()
    from wam_sweep.actions import ActionCodec
    live = scheduler(frozen, True)
    for x in [1.1, .5]:
        assert live.accept(0., np.tile([x, -2., .9, 4.], (25, 1)), .2)
    before = pickle.dumps(live)
    context = module.NativeTargetContext(xy_bounds=((0., 1.), (0., 1.)), z_bounds=(.1, .3))
    preview = module.preview_reference(live, np.zeros(4), .2, native_context=context)
    # Literal native result: float32 fuse, then clip/wrap in float32.
    expected = np.array([.8, 0., .3, -2.2831852], dtype=np.float32)
    q = preview.targets[0].astype(np.float32)
    assert q.tobytes() == expected.tobytes()
    assert q[0] != np.float32(.75)
    decoded = ActionCodec(bounds=context.xy_bounds).decode(np.zeros(4), q, np.zeros(3), .04, True)
    assert decoded.target.tobytes() == expected.tobytes()
    assert pickle.dumps(live) == before
    diagnostic = module.preview_reference(live, np.zeros(4), .2)
    assert preview.source_hash != diagnostic.source_hash


def test_provenance_binds_queue_clock_held_target_and_native_context(frozen):
    module = api()
    live = scheduler(frozen)
    assert live.accept(0., marked_chunk(), .2)
    held = np.array([0., 0., .2, 0.])
    original = module.preview_reference(live, held, .2)
    assert original.source_hash == module.preview_reference(live, held, .2).source_hash
    assert original.source_hash != module.preview_reference(live, held, .24).source_hash
    assert original.source_hash != module.preview_reference(live, held + .1, .2).source_hash
    live.chunk.actions[6, 0] += 1.
    assert original.source_hash != module.preview_reference(live, held, .2).source_hash


@pytest.mark.parametrize("bad", ["now", "held", "period", "context"])
def test_invalid_preview_context_is_rejected_without_scheduler_mutation(frozen, bad):
    module = api()
    live = scheduler(frozen)
    held, now = np.zeros(4), .2
    if bad == "now":
        now = -.1
    elif bad == "held":
        held[0] = np.nan
    elif bad == "period":
        live.action_period = .05
    before = pickle.dumps(live)
    with pytest.raises(ValueError):
        if bad == "context":
            module.NativeTargetContext(xy_bounds=((0., 0.), (0., 1.)), z_bounds=(.3, .1))
        else:
            module.preview_reference(live, held, now)
    assert pickle.dumps(live) == before
