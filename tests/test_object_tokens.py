import numpy as np

from sim.perception.object_tokens import (
    OBJECT_TOKEN_SLICES,
    TrackedObject,
    pack_object_tokens,
    symmetry_aware_yaw,
    track_to_token,
)


def make_track(track_id=1, confidence=0.9):
    return TrackedObject(
        track_id=track_id,
        class_probs=np.array([0.1, 0.2, 0.7], dtype=np.float32),
        xy=np.array([0.20, -0.10]),
        extent=np.array([0.04, 0.01]),
        yaw=0.30,
        previous_xy=np.array([0.18, -0.11]),
        previous_extent=np.array([0.039, 0.011]),
        previous_yaw=0.25,
        velocity_xy=np.array([0.10, -0.05]),
        angular_velocity=0.40,
        tray_overlap=0.25,
        full_inside=False,
        confidence=confidence,
        overhead_visibility=0.8,
        wrist_visibility=0.6,
        staleness=0.1,
    )


def test_symmetry_aware_yaw_uses_shape_periodicity():
    yaw = np.pi / 12.0
    np.testing.assert_allclose(
        symmetry_aware_yaw(yaw, class_name="nut"),
        [np.cos(6.0 * yaw), np.sin(6.0 * yaw)],
    )
    np.testing.assert_allclose(
        symmetry_aware_yaw(yaw, class_name="fastener"),
        [np.cos(2.0 * yaw), np.sin(2.0 * yaw)],
    )
    np.testing.assert_allclose(symmetry_aware_yaw(yaw, class_name="round"), [0.0, 0.0])


def test_symmetry_embedding_can_use_predicted_class_probabilities():
    yaw = 0.25
    probs = np.array([0.25, 0.50, 0.25])
    expected = np.array([
        0.25 * np.cos(6.0 * yaw) + 0.25 * np.cos(2.0 * yaw),
        0.25 * np.sin(6.0 * yaw) + 0.25 * np.sin(2.0 * yaw),
    ])
    np.testing.assert_allclose(symmetry_aware_yaw(yaw, class_probs=probs), expected)


def test_track_to_token_has_the_declared_29_fields():
    track = make_track()
    token = track_to_token(track, brush_xy=np.array([0.05, -0.20]), tray_mouth_xy=np.array([-0.4, 0.0]))

    assert token.shape == (29,)
    assert token.dtype == np.float32
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["class_probs"]], track.class_probs)
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["xy"]], track.xy)
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["extent"]], track.extent)
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["previous_xy"]], track.previous_xy)
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["velocity_xy"]], track.velocity_xy)
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["relative_brush"]], track.xy - [0.05, -0.20])
    np.testing.assert_allclose(token[OBJECT_TOKEN_SLICES["relative_tray"]], track.xy - [-0.4, 0.0])
    assert token[OBJECT_TOKEN_SLICES["valid"]][0] == 1.0


def test_pack_object_tokens_is_invariant_to_input_order_and_pads_slots():
    first = make_track(track_id=10, confidence=0.8)
    second = make_track(track_id=2, confidence=0.7)
    tokens_a, valid_a = pack_object_tokens(
        [first, second], brush_xy=np.zeros(2), tray_mouth_xy=np.zeros(2)
    )
    tokens_b, valid_b = pack_object_tokens(
        [second, first], brush_xy=np.zeros(2), tray_mouth_xy=np.zeros(2)
    )

    np.testing.assert_array_equal(tokens_a, tokens_b)
    np.testing.assert_array_equal(valid_a, valid_b)
    np.testing.assert_array_equal(valid_a, [True, True, False, False, False, False])
    np.testing.assert_array_equal(tokens_a[2:], np.zeros((4, 29), dtype=np.float32))


def test_pack_object_tokens_keeps_the_six_most_confident_tracks_deterministically():
    tracks = [make_track(track_id=i, confidence=i / 10.0) for i in range(1, 8)]
    tokens, valid = pack_object_tokens(
        list(reversed(tracks)), brush_xy=np.zeros(2), tray_mouth_xy=np.zeros(2)
    )

    assert valid.tolist() == [True] * 6
    # The retained rows are confidence-ranked, with track id used as the tie-breaker.
    retained = [tracks[i].track_id for i in range(1, 7)]
    assert all(np.isclose(row[OBJECT_TOKEN_SLICES["confidence"]][0], tid / 10.0)
               for row, tid in zip(tokens, retained))
