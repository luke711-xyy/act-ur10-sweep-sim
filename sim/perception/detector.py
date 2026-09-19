"""RGB-only grouped-instance detector used before ObjectACT policy encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ImageInstancePrediction:
    """One detector instance in image coordinates."""

    mask: np.ndarray
    class_probs: np.ndarray
    confidence: float
    center_rc: np.ndarray


class FrozenResNet18FPN(nn.Module):
    """ResNet-18 FPN with semantic, centre-heatmap and offset heads.

    The ImageNet backbone is frozen by default for the initial ObjectACT
    iteration.  The three dense heads remain trainable for RGB supervision.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        fpn_dim: int = 128,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        source = resnet18(weights=weights)
        self.backbone = nn.ModuleDict({
            "stem": nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool),
            "layer1": source.layer1,
            "layer2": source.layer2,
            "layer3": source.layer3,
            "layer4": source.layer4,
        })
        self.lateral2 = nn.Conv2d(128, int(fpn_dim), kernel_size=1)
        self.lateral3 = nn.Conv2d(256, int(fpn_dim), kernel_size=1)
        self.lateral4 = nn.Conv2d(512, int(fpn_dim), kernel_size=1)
        self.fpn_smoothing = nn.Sequential(
            nn.Conv2d(int(fpn_dim), int(fpn_dim), kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(32, int(fpn_dim))), int(fpn_dim)),
            nn.GELU(),
        )
        self.semantic_head = self._head(int(fpn_dim), 4)
        self.center_head = self._head(int(fpn_dim), 1)
        self.offset_head = self._head(int(fpn_dim), 2)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _head(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (B, 3, H, W)")
        stem = self.backbone["stem"](image)
        c1 = self.backbone["layer1"](stem)
        c2 = self.backbone["layer2"](c1)
        c3 = self.backbone["layer3"](c2)
        c4 = self.backbone["layer4"](c3)
        p2 = self.lateral2(c2)
        p2 = p2 + F.interpolate(self.lateral3(c3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = p2 + F.interpolate(self.lateral4(c4), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        features = self.fpn_smoothing(p2)
        target_size = image.shape[-2:]
        return {
            "semantic_logits": F.interpolate(self.semantic_head(features), size=target_size, mode="bilinear", align_corners=False),
            "center_logits": F.interpolate(self.center_head(features), size=target_size, mode="bilinear", align_corners=False),
            "offset": F.interpolate(self.offset_head(features), size=target_size, mode="bilinear", align_corners=False),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BN statistics should not drift when only the dense heads are
        # trained on a small simulator dataset.
        if all(not parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self


def detector_loss(
    outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Semantic CE + centre BCE + masked offset SmoothL1."""

    semantic = F.cross_entropy(outputs["semantic_logits"], targets["semantic"].long())
    center = F.binary_cross_entropy_with_logits(outputs["center_logits"], targets["center"].to(dtype=outputs["center_logits"].dtype))
    prediction = outputs["offset"]
    target = targets["offset"].to(dtype=prediction.dtype)
    valid = targets["offset_valid"].to(dtype=prediction.dtype)
    if valid.shape[1] == 1:
        valid = valid.expand(-1, 2, -1, -1)
    if torch.any(valid > 0.0):
        offset = F.smooth_l1_loss(prediction[valid > 0.0], target[valid > 0.0])
    else:
        offset = prediction.sum() * 0.0
    return semantic + center + offset


def decode_detector_output(
    output: dict[str, torch.Tensor],
    *,
    batch_index: int = 0,
    foreground_threshold: float = 0.5,
    max_instances: int = 6,
) -> list[ImageInstancePrediction]:
    """Decode grouped semantic pixels into RGB-derived instance masks."""

    semantic_logits = output["semantic_logits"].detach().cpu()
    center_logits = output["center_logits"].detach().cpu()
    offsets = output["offset"].detach().cpu()
    if semantic_logits.ndim != 4 or semantic_logits.shape[1] != 4:
        raise ValueError("semantic_logits must have shape (B, 4, H, W)")
    if center_logits.shape[:2] != (semantic_logits.shape[0], 1) or center_logits.shape[-2:] != semantic_logits.shape[-2:]:
        raise ValueError("center_logits has incompatible shape")
    if offsets.shape[:2] != (semantic_logits.shape[0], 2) or offsets.shape[-2:] != semantic_logits.shape[-2:]:
        raise ValueError("offset has incompatible shape")
    index = int(batch_index)
    if index < 0 or index >= semantic_logits.shape[0]:
        raise IndexError("batch_index is outside detector output")
    probabilities = torch.softmax(semantic_logits[index], dim=0).numpy()
    labels = np.argmax(probabilities, axis=0)
    confidence = np.max(probabilities, axis=0)
    foreground = (labels > 0) & (confidence >= float(foreground_threshold))
    components, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=int))
    center_probability = torch.sigmoid(center_logits[index, 0]).numpy()
    offset = offsets[index].numpy()
    predictions: list[ImageInstancePrediction] = []
    for component_id in range(1, int(count) + 1):
        rows, cols = np.nonzero(components == component_id)
        if len(rows) == 0:
            continue
        class_probs = probabilities[1:, rows, cols].mean(axis=1)
        class_probs = class_probs / max(float(class_probs.sum()), 1e-12)
        centre_rc = np.array([
            np.mean(rows + offset[0, rows, cols]),
            np.mean(cols + offset[1, rows, cols]),
        ], dtype=np.float32)
        score = float(np.mean(confidence[rows, cols]) * np.max(center_probability[rows, cols]))
        predictions.append(
            ImageInstancePrediction(
                mask=(components == component_id),
                class_probs=class_probs.astype(np.float32),
                confidence=score,
                center_rc=centre_rc,
            )
        )
    predictions.sort(key=lambda item: (-item.confidence, float(item.center_rc[0]), float(item.center_rc[1])))
    return predictions[: int(max_instances)]
