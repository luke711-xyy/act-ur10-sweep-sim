import numpy as np
import torch

from sim.perception.detector import (
    FrozenResNet18FPN,
    decode_detector_output,
    detector_loss,
)


def test_frozen_resnet18_fpn_returns_dense_rgb_heads_and_freezes_backbone():
    model = FrozenResNet18FPN(pretrained=False, freeze_backbone=True, fpn_dim=32)
    image = torch.randn(2, 3, 64, 64)
    outputs = model(image)
    assert outputs["semantic_logits"].shape == (2, 4, 64, 64)
    assert outputs["objectness_logits"].shape == (2, 1, 64, 64)
    assert outputs["center_logits"].shape == (2, 1, 64, 64)
    assert outputs["offset"].shape == (2, 2, 64, 64)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.semantic_head.parameters())


def test_detector_loss_uses_semantic_center_and_offset_targets():
    model = FrozenResNet18FPN(pretrained=False, freeze_backbone=True, fpn_dim=16)
    outputs = model(torch.randn(1, 3, 64, 64))
    targets = {
        "semantic": torch.zeros(1, 64, 64, dtype=torch.long),
        "center": torch.zeros(1, 1, 64, 64),
        "offset": torch.zeros(1, 2, 64, 64),
        "offset_valid": torch.zeros(1, 1, 64, 64),
    }
    loss = detector_loss(outputs, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_decoder_returns_grouped_instance_masks_without_environment_access():
    semantic = torch.full((1, 4, 16, 16), -10.0)
    semantic[:, 1, 2:5, 2:5] = 10.0
    semantic[:, 3, 10:13, 10:14] = 10.0
    center = torch.full((1, 1, 16, 16), -10.0)
    center[:, :, 3, 3] = 10.0
    center[:, :, 11, 11] = 10.0
    output = {
        "semantic_logits": semantic,
        "objectness_logits": torch.full((1, 1, 16, 16), -10.0),
        "center_logits": center,
        "offset": torch.zeros(1, 2, 16, 16),
    }
    output["objectness_logits"][0, 0, 2:5, 2:5] = 10.0
    output["objectness_logits"][0, 0, 10:13, 10:14] = 10.0
    predictions = decode_detector_output(output, batch_index=0, foreground_threshold=0.5)
    assert len(predictions) == 2
    assert all(prediction.mask.shape == (16, 16) for prediction in predictions)
    assert all(prediction.class_probs.shape == (3,) for prediction in predictions)
    assert np.argmax(predictions[0].class_probs) in {0, 2}


def test_decoder_prefers_binary_objectness_over_background_semantic_argmax():
    semantic = torch.full((1, 4, 8, 8), -10.0)
    semantic[:, 0] = 10.0  # the semantic head is uncertain/wrong here
    objectness = torch.full((1, 1, 8, 8), -10.0)
    objectness[:, :, 3:5, 3:5] = 10.0
    center = torch.full((1, 1, 8, 8), -10.0)
    center[:, :, 4, 4] = 10.0
    predictions = decode_detector_output(
        {
            "semantic_logits": semantic,
            "objectness_logits": objectness,
            "center_logits": center,
            "offset": torch.zeros(1, 2, 8, 8),
        },
        foreground_threshold=0.5,
    )
    assert len(predictions) == 1
    assert int(np.count_nonzero(predictions[0].mask)) == 4


def test_decoder_uses_center_offsets_to_split_touching_instances():
    semantic = torch.full((1, 4, 16, 16), -10.0)
    # Two instances share a one-pixel bridge, so semantic connected
    # components alone would incorrectly return one object.
    pixels = [(r, c) for r in range(3, 8) for c in range(3, 8)]
    pixels += [(r, c) for r in range(3, 8) for c in range(8, 13)]
    for row, col in pixels:
        semantic[0, 1, row, col] = 10.0
    center = torch.full((1, 1, 16, 16), -10.0)
    center[0, 0, 5, 5] = 10.0
    center[0, 0, 5, 10] = 10.0
    offset = torch.zeros(1, 2, 16, 16)
    for row, col in pixels:
        target = (5, 5) if col <= 7 else (5, 10)
        offset[0, :, row, col] = torch.tensor([target[0] - row, target[1] - col])
    predictions = decode_detector_output(
        {"semantic_logits": semantic, "center_logits": center, "offset": offset},
        foreground_threshold=0.5,
        center_threshold=0.35,
        center_nms_radius=2,
        max_center_distance=2.0,
    )
    assert len(predictions) == 2
    assert sorted(int(np.count_nonzero(item.mask)) for item in predictions) == [25, 25]
