import json

import numpy as np
import torch
from PIL import Image


def test_dense_detector_targets_encode_instances_centres_and_offsets():
    from sim.perception.detector_training import dense_targets_from_instance_map

    instance_map = np.zeros((8, 10), dtype=np.int32)
    instance_map[2:5, 3:6] = 1
    instance_map[5:7, 7:9] = 2
    class_map = np.zeros_like(instance_map)
    class_map[instance_map == 1] = 2
    class_map[instance_map == 2] = 3

    targets = dense_targets_from_instance_map(instance_map, class_map)

    assert targets["semantic"].shape == (8, 10)
    assert targets["center"].shape == (1, 8, 10)
    assert targets["offset"].shape == (2, 8, 10)
    assert targets["offset_valid"].shape == (1, 8, 10)
    assert targets["semantic"][3, 4] == 2
    assert targets["semantic"][6, 8] == 3
    assert targets["center"][0, 3, 4] == 1.0
    assert targets["center"][0, 6, 8] == 1.0
    # Decoder uses row + offset[0], col + offset[1] to recover a centre.
    np.testing.assert_allclose(targets["offset"][:, 2, 3], [1.0, 1.0])
    np.testing.assert_allclose(targets["offset"][:, 6, 8], [-0.5, -0.5])
    assert np.count_nonzero(targets["center"] > 0.1) > 2


def test_detector_loss_reweights_sparse_positive_center_pixels():
    from sim.perception.detector import detector_loss

    outputs = {
        "semantic_logits": torch.zeros(1, 4, 4, 4, requires_grad=True),
        "center_logits": torch.zeros(1, 1, 4, 4, requires_grad=True),
        "offset": torch.zeros(1, 2, 4, 4, requires_grad=True),
    }
    targets = {
        "semantic": torch.zeros(1, 4, 4, dtype=torch.long),
        "center": torch.zeros(1, 1, 4, 4),
        "offset": torch.zeros(1, 2, 4, 4),
        "offset_valid": torch.zeros(1, 1, 4, 4),
    }
    targets["center"][0, 0, 1, 1] = 1.0
    detector_loss(outputs, targets).backward()
    positive = abs(float(outputs["center_logits"].grad[0, 0, 1, 1]))
    negative = abs(float(outputs["center_logits"].grad[0, 0, 0, 0]))
    assert positive > negative


def test_detector_quality_metrics_match_instances_and_count_errors():
    from sim.perception.detector import ImageInstancePrediction
    from sim.perception.detector_training import detector_quality_metrics

    truth = np.zeros((8, 10), dtype=np.int32)
    truth[2:5, 3:6] = 1
    truth[5:7, 7:9] = 2
    predictions = [
        ImageInstancePrediction(
            mask=truth == 1,
            class_probs=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            confidence=0.9,
            center_rc=np.array([3.0, 4.0], dtype=np.float32),
        ),
        ImageInstancePrediction(
            mask=np.zeros_like(truth, dtype=bool),
            class_probs=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            confidence=0.6,
            center_rc=np.array([1.0, 1.0], dtype=np.float32),
        ),
    ]

    metrics = detector_quality_metrics(predictions, truth, iou_threshold=0.5)
    assert metrics["sample_count"] == 1
    assert metrics["matched_count"] == 1
    assert np.isclose(metrics["mask_iou"], 1.0)
    assert np.isclose(metrics["center_error_px"], 0.0)
    assert np.isclose(metrics["miss_rate"], 0.5)
    assert np.isclose(metrics["false_positive_rate"], 0.5)
    assert np.isclose(metrics["mean_abs_count_error"], 0.0)


def test_center_quality_metrics_are_not_penalized_by_mask_boundary_error():
    from sim.perception.detector import ImageInstancePrediction
    from sim.perception.detector_training import detector_center_quality_metrics

    truth = np.zeros((12, 12), dtype=np.int32)
    truth[2:5, 2:5] = 1
    truth[7:10, 7:10] = 2
    predictions = [
        ImageInstancePrediction(
            mask=np.zeros_like(truth, dtype=bool),
            class_probs=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            confidence=0.9,
            center_rc=np.array([3.2, 3.1], dtype=np.float32),
        ),
        ImageInstancePrediction(
            mask=np.zeros_like(truth, dtype=bool),
            class_probs=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            confidence=0.8,
            center_rc=np.array([8.1, 8.0], dtype=np.float32),
        ),
    ]
    metrics = detector_center_quality_metrics(predictions, truth, distance_threshold_px=2.0)
    assert metrics["matched_count"] == 2
    assert metrics["miss_rate"] == 0.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["center_error_px"] < 0.5


def test_detector_dataset_writer_is_atomic_and_rejects_truth_policy_fields(tmp_path):
    from sim.perception.detector_training import DetectorDatasetWriter, DetectorTrainingDataset

    writer = DetectorDatasetWriter(tmp_path)
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    instance = np.zeros((8, 10), dtype=np.int32)
    instance[2:5, 3:6] = 1
    class_map = np.zeros_like(instance)
    class_map[instance == 1] = 1
    writer.add_frame("frame_000", image, instance, class_map, split="train")

    record = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[0])
    assert record["schema_version"] == 1
    assert "mujoco_truth" not in record
    dataset = DetectorTrainingDataset(tmp_path, split="train")
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 10)
    assert sample["semantic"].shape == (8, 10)
    assert sample["instance"].shape == (8, 10)


def test_detector_quality_gate_requires_all_declared_thresholds():
    from sim.perception.detector_training import detector_quality_gate

    passing = {
        "mask_iou": 0.80,
        "center_error_px": 2.0,
        "miss_rate": 0.05,
        "false_positive_rate": 0.05,
        "mean_abs_count_error": 0.1,
    }
    assert detector_quality_gate(passing, min_mask_iou=0.75, max_center_error_px=3.0,
                                 max_miss_rate=0.10, max_false_positive_rate=0.10,
                                 max_count_error=0.25)
    failing = dict(passing, miss_rate=0.11)
    assert not detector_quality_gate(failing, min_mask_iou=0.75, max_center_error_px=3.0,
                                     max_miss_rate=0.10, max_false_positive_rate=0.10,
                                     max_count_error=0.25)


def test_detector_quality_gate_can_use_temporal_center_metrics():
    from sim.perception.detector_training import detector_quality_gate

    metrics = {
        "mask_iou": 0.72,
        "center_error_px": 0.8,
        "miss_rate": 0.4,  # raw mask miss rate, diagnostic only
        "false_positive_rate": 0.3,  # raw mask FP rate, diagnostic only
        "mean_abs_count_error": 0.6,  # raw frame count, diagnostic only
        "center_miss_rate": 0.08,
        "center_false_positive_rate": 0.04,
        "warmup_track_coverage": 1.0,
        "warmup_mean_abs_count_error": 0.0,
    }
    assert detector_quality_gate(
        metrics,
        min_mask_iou=0.70,
        max_center_error_px=3.0,
        max_miss_rate=0.10,
        max_false_positive_rate=0.10,
        max_count_error=0.25,
        min_warmup_track_coverage=0.95,
        max_warmup_count_error=0.25,
    )


def test_segmentation_ids_are_converted_to_grouped_instance_and_class_maps():
    from sim.perception.detector_training import instance_maps_from_geom_ids

    segmentation = np.array([[7, 7, 0, 8], [7, 9, 9, -1]], dtype=np.int32)
    instances, classes = instance_maps_from_geom_ids(
        segmentation,
        geom_to_component={7: 0, 8: 1, 9: 2},
        component_class_indices=[1, 3, 2],
    )
    np.testing.assert_array_equal(instances, [[1, 1, 0, 2], [1, 3, 3, 0]])
    np.testing.assert_array_equal(classes, [[1, 1, 0, 3], [1, 2, 2, 0]])


def test_detector_checkpoint_round_trip_keeps_schema_and_model_config(tmp_path):
    from sim.perception.detector import FrozenResNet18FPN
    from sim.perception.detector_training import (
        load_detector_checkpoint,
        save_detector_checkpoint,
    )

    model = FrozenResNet18FPN(pretrained=False, freeze_backbone=True, fpn_dim=16)
    path = save_detector_checkpoint(
        model, tmp_path, step=12, config={"fpn_dim": 16, "pretrained": False}
    )
    restored = load_detector_checkpoint(path, device="cpu")
    assert restored["schema_version"] == 1
    assert restored["step"] == 12
    assert restored["config"]["fpn_dim"] == 16
    assert set(restored["state_dict"]) == set(model.state_dict())


def test_detector_training_parser_has_explicit_quality_gate_and_mps_safe_defaults():
    from sim.act.train_detector import build_detector_train_parser

    args = build_detector_train_parser().parse_args([])
    assert args.steps == 5000
    assert args.batch_size == 4
    assert args.checkpoint_every == 500
    assert args.device == "auto"
    assert args.pretrained is True
    assert args.foreground_threshold == 0.60
    assert args.min_mask_iou == 0.70
    assert args.max_miss_rate == 0.10


def test_detector_source_split_is_grouped_by_layout_not_frame_or_episode():
    from sim.act.collect_detector_dataset import assign_layout_splits

    records = [
        {"episode_id": "a", "layout_id": "paired_000"},
        {"episode_id": "b", "layout_id": "paired_000"},
        {"episode_id": "c", "layout_id": "independent_001"},
        {"episode_id": "d", "layout_id": "independent_002"},
    ]
    split = assign_layout_splits(records, val_group_modulo=2)
    assert split["a"] == split["b"]
    assert set(split.values()) == {"train", "val"}


def test_unverified_detector_checkpoint_is_rejected_before_policy_data_generation(tmp_path):
    from sim.perception.detector_training import verified_detector_checkpoint

    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"placeholder")
    (tmp_path / "detector_quality.json").write_text(
        json.dumps({"quality_gate_passed": False}), encoding="utf-8"
    )
    try:
        verified_detector_checkpoint(checkpoint)
    except RuntimeError as error:
        assert "quality gate" in str(error)
    else:  # pragma: no cover - assertion form keeps the expected failure explicit
        raise AssertionError("unverified detector checkpoint was accepted")
