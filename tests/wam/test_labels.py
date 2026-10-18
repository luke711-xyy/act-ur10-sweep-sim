"""Temporal evidence fixtures catch fabricated negatives and stale contact IDs."""
import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest

from wam_sweep.contracts import supervision


def api():
    assert importlib.util.find_spec("wam_sweep.labels"), "Task4 carry labels are absent"
    from wam_sweep import labels, truth
    return labels, truth


def frame(t, *, x=None, contacts=((-1, 0), (0, 1)), visible=None, contact_valid=True,
          collected=(False, False), partial=(False, False), brush_speed=.08):
    _, truth = api()
    bx = brush_speed * t
    centers = x if x is not None else [bx + .019, bx + .038]
    objects = tuple(truth.CollisionBody((truth.CollisionGeom("sphere", np.array([.01, 0., 0.]),
        np.array([cx, 0., 0.]), np.eye(3)),), 0., "circle") for cx in centers)
    brush = truth.CollisionBody((truth.CollisionGeom("box", np.array([.01, .05, .01]),
        np.array([bx, 0., 0.]), np.eye(3)),), 0., "rectangle")
    raw = supervision.RawTruth(dict(objects=objects, brush=brush, contacts=contacts,
        contacts_valid=contact_valid, native_collected=np.array(collected, bool),
        native_partial=np.array(partial, bool), native_valid=np.ones(2, bool),
        visibility=np.zeros(2, bool) if visible is None else np.array(visible, bool),
        visibility_valid=np.zeros(2, bool) if visible is None else np.ones(2, bool),
        object_ids=(0, 1), target_bounds=None))
    return truth.TruthFrame(t, raw)


def run_frames(labeler, stop=.6, **kwargs):
    result = None
    for tick in range(round(stop / .04) + 1):
        result = labeler.update(frame(round(tick * .04, 8), **kwargs))
    return result


def test_direct_and_contact_chain_indirect_push_become_stable():
    labels, _ = api()
    result = run_frames(labels.CarriedLabeler())
    assert result.carried.tolist() == [True, True]
    assert result.carried_valid.tolist() == [True, True]
    assert result.state.tolist() == [1, 1]
    assert result.state_valid.all()


def test_transient_contact_is_not_stable():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    labeler.update(frame(0.))
    label = labeler.update(frame(.04, contacts=()))
    assert not label.carried_valid[1]
    label = labeler.update(frame(.08, contacts=()))
    assert not label.carried_valid.any()


def test_join_requires_full_window_then_120ms_hysteresis():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    for t in (0., .04, .08, .12, .16, .20, .24, .28):
        assert not labeler.update(frame(t)).carried_valid.any()
    label = labeler.update(frame(.32))
    assert label.carried.all() and label.carried_valid.all()


def test_dropout_transition_is_unknown_then_free_and_contact_identity_expires():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    assert run_frames(labeler).carried.all()
    for tick in range(16, 24):
        label = labeler.update(frame(tick * .04, x=[.067, .086], contacts=()))
        assert not label.carried_valid.any()
    for tick in range(24, 32):
        label = labeler.update(frame(tick * .04, x=[.067, .086], contacts=()))
    assert label.carried_valid.all()
    assert not label.carried.any()
    assert label.state.tolist() == [0, 0]


def test_short_occlusion_preserves_physical_carry_but_visibility_has_own_mask():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    run_frames(labeler, visible=(True, True))
    label = labeler.update(frame(.64, visible=(False, False)))
    assert label.carried.all() and label.carried_valid.all()
    assert not label.targets.visibility.any()
    assert label.targets.visibility_valid.all()
    label = labeler.update(frame(.68))
    assert label.carried_valid.all()
    assert not label.targets.visibility_valid.any()


def test_stationary_brush_and_missing_contact_evidence_are_unknown():
    labels, _ = api()
    for options in (dict(brush_speed=0.), dict(contact_valid=False)):
        label = run_frames(labels.CarriedLabeler(), **options)
        assert not label.carried_valid.any()


def test_contact_behind_brush_is_unknown_not_stable_negative():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    for tick in range(16):
        t = tick * .04
        label = labeler.update(frame(t, x=[.08 * t - .019, .08 * t - .038]))
    assert not label.carried_valid.any()


@pytest.mark.parametrize("delta", [.025, -.025])
def test_velocity_threshold_boundary_is_unknown(delta):
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    for tick in range(16):
        t = tick * .04
        label = labeler.update(frame(t, x=[.019 + (.08 + delta) * t, .038 + (.08 + delta) * t]))
    assert not label.carried_valid.any()


def test_displacement_threshold_boundary_is_unknown():
    labels, _ = api()
    # Speed tolerance raised solely to isolate the independently required 8mm gate.
    labeler = labels.CarriedLabeler(labels.LabelConfig(speed_difference_mps=.1))
    for tick in range(16):
        t = tick * .04
        label = labeler.update(frame(t, x=[.019 + .12 * t, .038 + .12 * t]))
    assert not label.carried_valid.any()


def test_full_native_collection_and_extra_partial_overlap_remain_separate():
    labels, _ = api()
    label = labels.CarriedLabeler().update(frame(0., collected=(True, False), partial=(False, True)))
    assert label.state[0] == 2 and label.state_valid[0]
    predicate = getattr(label, "native_count_matches", None)
    assert predicate is not None, "spatial count predicate must be distinct from terminal success"
    assert predicate(1) is False
    label = labels.CarriedLabeler().update(frame(0., collected=(True, False)))
    assert label.native_count_matches(1) is True


def test_early_termination_does_not_pad_future():
    labels, _ = api()
    target = labels.future_targets([frame(.04), frame(.08), frame(.12)])
    assert target.valid_steps.sum() == 3
    assert target.center_xy.shape == (25, 2, 2)
    assert not target.velocity_valid[3:].any()
    assert not target.visibility_valid[3:].any()
    assert not target.center_xy[3:].any()
    assert not target.velocity_valid[0].any()
    assert target.velocity_valid[1:3].all()


def test_snapshot_restore_is_json_serializable_and_replays_pending_hysteresis():
    labels, _ = api()
    original = labels.CarriedLabeler()
    run_frames(original, stop=.24)
    state = json.loads(json.dumps(original.snapshot(), allow_nan=False))
    restored = labels.CarriedLabeler()
    restored.restore(state)
    assert restored.state_hash == original.state_hash
    for t in (.28, .32, .36):
        a, b = original.update(frame(t)), restored.update(frame(t))
        np.testing.assert_array_equal(a.carried, b.carried)
        np.testing.assert_array_equal(a.carried_valid, b.carried_valid)
        assert restored.state_hash == original.state_hash
    with pytest.raises(ValueError, match="config"):
        labels.CarriedLabeler(labels.LabelConfig(speed_difference_mps=.02)).restore(state)


def test_config_metadata_hash_covers_thresholds_and_version():
    labels, _ = api()
    a, b = labels.CarriedLabeler(), labels.CarriedLabeler(labels.LabelConfig(window_seconds=.24))
    ma, mb = a.update(frame(0.)).metadata, b.update(frame(0.)).metadata
    assert ma["config_hash"] != mb["config_hash"]
    assert ma["config"]["version"] == "wam-carry-v1"
    assert ma["config"]["window_seconds"] == .2
    assert ma["config"]["join_seconds"] == ma["config"]["leave_seconds"] == .12
    assert ma["config"]["relative_displacement_m"] == .008
    assert ma["config"]["speed_difference_mps"] == .025


def test_duplicate_time_and_evidence_gap_cannot_reuse_stable_label():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    run_frames(labeler)
    with pytest.raises(ValueError, match="increasing"):
        labeler.update(frame(.6))
    label = labeler.update(frame(1.))
    assert not label.carried_valid.any()


def test_insufficient_brush_translation_cannot_label_stationary_contact_as_carried():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    for tick in range(16):
        t = tick * .04
        label = labeler.update(frame(t, x=[.019, .038], brush_speed=.01))
    assert not label.carried_valid.any()


@pytest.mark.parametrize("corrupt", ["future_timer", "negative_timer", "gap", "nonboolean_contact"])
def test_restore_rejects_evidence_that_can_bypass_hysteresis(corrupt):
    labels, _ = api()
    original = labels.CarriedLabeler()
    run_frames(original, stop=.24)
    state = original.snapshot()
    if corrupt == "future_timer":
        state["since"][0] = 10.
    elif corrupt == "negative_timer":
        state["since"][0] = -1.
    elif corrupt == "gap":
        state["history"][0]["time"] = 0.
    else:
        state["history"][0]["contacts_valid"] = "false"
    restored = labels.CarriedLabeler()
    before = restored.state_hash
    with pytest.raises(ValueError, match="snapshot"):
        restored.restore(state)
    assert restored.state_hash == before


def test_rotating_rectangles_supply_masked_angular_velocity_after_restore():
    labels, truth = api()
    labeler = labels.CarriedLabeler()

    def rotating(t, yaw):
        f = frame(t)
        fields = dict(f.raw.fields)
        objects = list(fields["objects"])
        objects[0] = replace(objects[0], shape="rectangle", yaw=yaw)
        fields["objects"] = tuple(objects)
        return truth.TruthFrame(t, supervision.RawTruth(fields))

    first = labeler.update(rotating(0., np.pi - .01))
    assert not first.targets.angular_velocity_valid.any()
    restored = labels.CarriedLabeler()
    restored.restore(labeler.snapshot())
    label = restored.update(rotating(.04, .01))
    np.testing.assert_allclose(label.targets.angular_velocity, [.5, 0.], atol=1e-10)
    assert label.targets.angular_velocity_valid.tolist() == [True, False]


def test_missing_native_evidence_does_not_become_task_success_or_free_state():
    labels, truth = api()
    f = frame(0., collected=(True, False))
    fields = dict(f.raw.fields, native_valid=np.zeros(2, bool))
    label = labels.CarriedLabeler().update(truth.TruthFrame(0., supervision.RawTruth(fields)))
    predicate = getattr(label, "native_count_matches", None)
    assert predicate is not None, "spatial count predicate must be distinct from terminal success"
    assert predicate(1) is None
    assert not label.state_valid.any()


def test_contact_edges_from_different_ticks_do_not_form_an_indirect_push():
    labels, _ = api()
    labeler = labels.CarriedLabeler()
    for tick in range(16):
        edges = ((-1, 0),) if tick % 2 else ((0, 1),)
        label = labeler.update(frame(tick * .04, contacts=edges))
    assert not label.carried_valid.any()


def test_relative_displacement_excursion_cannot_be_hidden_by_matching_endpoints():
    labels, _ = api()
    config = labels.LabelConfig(speed_difference_mps=1.)
    labeler = labels.CarriedLabeler(config)
    for tick in range(16):
        t = tick * .04
        offset = .009 if tick % 5 == 2 else 0.
        label = labeler.update(frame(t, x=[.08 * t + .019 + offset, .08 * t + .038 + offset]))
    assert not label.carried_valid.any()


def test_future_prefix_cannot_relabel_a_missing_25hz_capture_as_the_next_step():
    labels, _ = api()
    with pytest.raises(ValueError, match="25 Hz"):
        labels.future_targets([frame(.04), frame(.08), frame(.20)])
