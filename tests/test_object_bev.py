import numpy as np

from sim.perception.camera import PinholeCamera
from sim.perception.object_bev import (
    BEVSpec,
    PredictedInstance,
    TrackManager,
    annotate_tray_features,
    build_bev_channels,
    fuse_projected_instances,
    pack_instance_masks,
    project_image_instance,
    rectangle_signed_distance,
)


def make_instance(spec, track_id, xy, selected=False):
    iy, ix, valid = spec.to_index(np.asarray([xy], dtype=float))
    assert bool(valid[0])
    mask = np.zeros(spec.shape, dtype=bool)
    mask[iy[0], ix[0]] = True
    return PredictedInstance(
        track_id=track_id,
        mask_bev=mask,
        class_probs=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        confidence=0.9,
        xy=np.asarray(xy, dtype=float),
        extent=np.array([0.02, 0.02]),
        yaw=0.0,
        overhead_visibility=1.0,
        wrist_visibility=0.0,
    )


def test_bev_spec_has_fixed_full_table_contract_and_round_trips_centres():
    spec = BEVSpec()
    assert spec.shape == (128, 160)
    assert np.isclose(spec.resolution, 0.00625)
    iy, ix, valid = spec.to_index(np.array([[0.0, 0.0]]))
    assert bool(valid[0])
    np.testing.assert_allclose(spec.to_xy(iy, ix)[0], [0.003125, 0.003125], atol=1e-9)


def test_project_image_instance_backprojects_pixels_without_simulator_truth():
    spec = BEVSpec()
    camera = PinholeCamera(
        position=np.array([0.0, 0.0, 1.0]),
        rotation=np.eye(3),
        width=100,
        height=100,
        fovy_deg=90.0,
    )
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:55, 45:55] = True
    instance = project_image_instance(
        mask,
        camera=camera,
        plane_z=0.0,
        bev_spec=spec,
        class_probs=np.array([1.0, 0.0, 0.0]),
        confidence=0.8,
    )
    assert instance.mask_bev.shape == spec.shape
    assert instance.mask_bev.any()
    np.testing.assert_allclose(instance.xy, [0.0, 0.0], atol=0.02)
    assert np.all(instance.extent > 0.0)
    assert instance.class_probs.tolist() == [1.0, 0.0, 0.0]


def test_bev_channels_encode_selection_tray_brush_and_free_space():
    spec = BEVSpec()
    first = make_instance(spec, 1, [0.10, 0.0])
    second = make_instance(spec, 2, [0.20, 0.0])
    tray = (-0.48, -0.30, -0.18, 0.18)
    forbidden = [(-0.50, -0.48, -0.40, 0.40)]
    channels = build_bev_channels(
        [first, second],
        spec=spec,
        selected_track_ids={1},
        brush_xy=np.array([0.10, 0.0]),
        brush_size=np.array([0.12, 0.02]),
        brush_yaw=0.0,
        tray_bounds=tray,
        forbidden_rectangles=forbidden,
    )
    assert channels.shape == (6, 128, 160)
    iy1, ix1, _ = spec.to_index(np.array([[0.10, 0.0]]))
    iy2, ix2, _ = spec.to_index(np.array([[0.20, 0.0]]))
    assert channels[0, iy1[0], ix1[0]] > 0.0
    assert channels[1, iy1[0], ix1[0]] > 0.0
    assert channels[2, iy2[0], ix2[0]] > 0.0
    assert channels[3, *spec.to_index(np.array([[-0.35, 0.0]]))[:2]] > 0.0
    assert channels[3, *spec.to_index(np.array([[0.35, 0.0]]))[:2]] < 0.0
    assert channels[4, iy1[0], ix1[0]] > 0.0
    assert channels[5, *spec.to_index(np.array([[0.0, 0.0]]))[:2]] > 0.0
    assert channels[5, *spec.to_index(np.array([[-0.49, 0.0]]))[:2]] < 0.0


def test_rectangle_signed_distance_has_positive_inside_sign():
    bounds = (-0.4, -0.2, -0.1, 0.1)
    points = np.array([[-0.3, 0.0], [-0.5, 0.0], [-0.3, 0.2]])
    distance = rectangle_signed_distance(points, bounds)
    assert distance[0] > 0.0
    assert distance[1] < 0.0
    assert distance[2] < 0.0


def test_dual_camera_fusion_merges_same_part_but_keeps_separate_parts():
    spec = BEVSpec()
    first = make_instance(spec, -1, [0.10, 0.0])
    duplicate = make_instance(spec, -1, [0.105, 0.0])
    other = make_instance(spec, -1, [0.30, 0.0])
    fused = fuse_projected_instances([other, duplicate, first], spec=spec)
    assert len(fused) == 2
    assert fused[0].confidence > 0.9
    assert np.isclose(fused[0].xy[0], 0.10, atol=0.02)


def test_tray_features_are_derived_from_projected_instance_mask():
    spec = BEVSpec()
    instance = make_instance(spec, 4, [-0.35, 0.0])
    annotated = annotate_tray_features(
        instance, spec=spec, tray_bounds=(-0.40, -0.25, -0.10, 0.10)
    )
    assert annotated.tray_overlap > 0.0
    assert annotated.full_inside is True


def test_instance_masks_use_the_same_slot_order_as_tokens():
    spec = BEVSpec()
    low = make_instance(spec, 10, [0.10, 0.0])
    high = make_instance(spec, 2, [0.20, 0.0])
    low = PredictedInstance(**{**low.__dict__, "confidence": 0.2})
    high = PredictedInstance(**{**high.__dict__, "confidence": 0.9})
    masks, valid, ids = pack_instance_masks([low, high], spec=spec)
    assert valid.tolist() == [True, True, False, False, False, False]
    assert ids == [2, 10]
    iy, ix, _ = spec.to_index(np.array([[0.20, 0.0]]))
    assert masks[0, iy[0], ix[0]]


def test_track_manager_keeps_ids_and_estimates_velocity_under_detection_order_changes():
    spec = BEVSpec()
    manager = TrackManager(max_match_distance=0.05)
    first = make_instance(spec, -1, [0.10, 0.0])
    second = make_instance(spec, -1, [0.20, 0.0])
    tracks0 = manager.update([first, second], timestamp=0.0)
    ids0 = {tuple(np.round(track.xy, 3)): track.track_id for track in tracks0}

    moved_first = make_instance(spec, -1, [0.11, 0.0])
    moved_second = make_instance(spec, -1, [0.20, 0.01])
    tracks1 = manager.update([moved_second, moved_first], timestamp=0.1)
    ids1 = {tuple(np.round(track.xy, 3)): track for track in tracks1}
    assert ids1[(0.11, 0.0)].track_id == ids0[(0.10, 0.0)]
    assert ids1[(0.20, 0.01)].track_id == ids0[(0.20, 0.0)]
    np.testing.assert_allclose(ids1[(0.11, 0.0)].velocity_xy, [0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(ids1[(0.11, 0.0)].previous_xy, [0.10, 0.0], atol=1e-6)
