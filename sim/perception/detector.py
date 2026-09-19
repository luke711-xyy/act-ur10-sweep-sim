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
        normalize_input: bool | None = None,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        source = resnet18(weights=weights)
        self.normalize_input = bool(pretrained) if normalize_input is None else bool(normalize_input)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.backbone = nn.ModuleDict({
            "stem": nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool),
            "layer1": source.layer1,
            "layer2": source.layer2,
            "layer3": source.layer3,
            "layer4": source.layer4,
        })
        # The parts are only a few dozen pixels wide in the 320x320 policy
        # view.  Starting the decoder at layer2 (1/8 resolution) discards too
        # much instance detail before the dense heads see it.  Keep a 1/4
        # resolution lateral path and let the coarse pyramid add context.
        self.lateral1 = nn.Conv2d(64, int(fpn_dim), kernel_size=1)
        self.lateral2 = nn.Conv2d(128, int(fpn_dim), kernel_size=1)
        self.lateral3 = nn.Conv2d(256, int(fpn_dim), kernel_size=1)
        self.lateral4 = nn.Conv2d(512, int(fpn_dim), kernel_size=1)
        self.fpn_smoothing = nn.Sequential(
            nn.Conv2d(int(fpn_dim), int(fpn_dim), kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(32, int(fpn_dim))), int(fpn_dim)),
            nn.GELU(),
        )
        self.semantic_head = self._head(int(fpn_dim), 4)
        # Push-Wiper's useful abstraction is a spatial occupancy signal.  Keep
        # it as a learned RGB-derived head instead of thresholding raw pixels:
        # shadows, the brush, and the checkerboard are not object occupancy.
        self.objectness_head = self._head(int(fpn_dim), 1)
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
        image = image.to(dtype=torch.float32)
        if self.normalize_input:
            image = (image - self.imagenet_mean.to(device=image.device)) / self.imagenet_std.to(device=image.device)
        stem = self.backbone["stem"](image)
        c1 = self.backbone["layer1"](stem)
        c2 = self.backbone["layer2"](c1)
        c3 = self.backbone["layer3"](c2)
        c4 = self.backbone["layer4"](c3)
        p1 = self.lateral1(c1)
        p2 = self.lateral2(c2)
        p2 = p2 + F.interpolate(self.lateral3(c3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = p2 + F.interpolate(self.lateral4(c4), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = p1 + F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.fpn_smoothing(p1)
        target_size = image.shape[-2:]
        return {
            "semantic_logits": F.interpolate(self.semantic_head(features), size=target_size, mode="bilinear", align_corners=False),
            "objectness_logits": F.interpolate(self.objectness_head(features), size=target_size, mode="bilinear", align_corners=False),
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
    """Occupancy focal loss, class-balanced semantic/centre losses, and offsets.

    Tabletop parts occupy far fewer pixels than the background, and a centre
    heatmap has only a handful of positive locations.  Unweighted BCE/CE
    therefore learns the all-background solution even when its scalar loss
    looks small.  The fixed background down-weight and bounded dynamic centre
    ``pos_weight`` keep gradients useful without making false positives free.
    The separate occupancy head mirrors a binary tabletop map while retaining
    semantic classes for object-token construction.
    """

    semantic_logits = outputs["semantic_logits"]
    semantic_target = targets["semantic"].long()
    class_weights = torch.tensor(
        [0.05, 1.0, 1.0, 1.0], dtype=semantic_logits.dtype, device=semantic_logits.device
    )
    semantic = F.cross_entropy(semantic_logits, semantic_target, weight=class_weights)
    objectness_logits = outputs.get("objectness_logits")
    if objectness_logits is None:
        objectness = semantic_logits.sum() * 0.0
    else:
        objectness_target = (semantic_target > 0).to(dtype=objectness_logits.dtype)
        objectness_target = objectness_target.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(
            objectness_logits, objectness_target, reduction="none"
        )
        probability = torch.sigmoid(objectness_logits)
        p_t = probability * objectness_target + (1.0 - probability) * (1.0 - objectness_target)
        alpha_t = 0.75 * objectness_target + 0.25 * (1.0 - objectness_target)
        objectness = (alpha_t * (1.0 - p_t).pow(2.0) * bce).mean()
    center_logits = outputs["center_logits"]
    center_target = targets["center"].to(dtype=center_logits.dtype)
    positive = float(torch.count_nonzero(center_target > 1e-3).detach().cpu())
    total = float(center_target.numel())
    pos_weight = min(100.0, max(1.0, (total - positive) / max(positive, 1.0)))
    center = F.binary_cross_entropy_with_logits(
        center_logits,
        center_target,
        pos_weight=torch.full((1,), pos_weight, dtype=center_logits.dtype, device=center_logits.device),
    )
    prediction = outputs["offset"]
    target = targets["offset"].to(dtype=prediction.dtype)
    valid = targets["offset_valid"].to(dtype=prediction.dtype)
    if valid.shape[1] == 1:
        valid = valid.expand(-1, 2, -1, -1)
    if torch.any(valid > 0.0):
        offset = F.smooth_l1_loss(prediction[valid > 0.0], target[valid > 0.0])
    else:
        offset = prediction.sum() * 0.0
    return semantic + objectness + center + offset


def decode_detector_output(
    output: dict[str, torch.Tensor],
    *,
    batch_index: int = 0,
    foreground_threshold: float = 0.5,
    center_threshold: float = 0.35,
    center_nms_radius: int = 5,
    max_center_distance: float = 14.0,
    min_mask_pixels: int = 3,
    max_instances: int = 6,
) -> list[ImageInstancePrediction]:
    """Decode semantic pixels into RGB-derived instance masks.

    A connected-component decoder is insufficient when two small parts touch
    in the projected image.  The detector therefore predicts a center heatmap
    and per-pixel center offsets; foreground pixels are assigned to local
    center peaks in the offset-predicted center space.  Connected components
    remain the conservative fallback when no reliable center peak exists.
    """

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
    semantic_confidence = np.max(probabilities, axis=0)
    objectness_logits = output.get("objectness_logits")
    if objectness_logits is None:
        confidence = semantic_confidence
        foreground = (labels > 0) & (confidence >= float(foreground_threshold))
    else:
        objectness_probability = torch.sigmoid(objectness_logits.detach().cpu()[index, 0]).numpy()
        confidence = objectness_probability
        foreground = confidence >= float(foreground_threshold)
    center_probability = torch.sigmoid(center_logits[index, 0]).numpy()
    offset = offsets[index].numpy()
    peaks = _center_peaks(
        center_probability,
        threshold=float(center_threshold),
        nms_radius=int(center_nms_radius),
        max_peaks=int(max_instances),
    )
    assignments = _assign_foreground_to_centers(
        foreground,
        offset,
        peaks,
        max_distance=float(max_center_distance),
    )
    components, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=int))
    predictions: list[ImageInstancePrediction] = []
    if assignments is not None:
        groups = [assignments == peak_index for peak_index in range(len(peaks))]
        # Keep residual connected components if there is no center-supported
        # assignment for them.  This avoids silently dropping an occluded or
        # weakly peaked part while still splitting touching parts.
        assigned = np.zeros_like(foreground, dtype=bool)
        for group in groups:
            assigned |= group
        for component_id in range(1, int(count) + 1):
            residual = (components == component_id) & ~assigned
            if np.count_nonzero(residual) >= int(min_mask_pixels):
                groups.append(residual)
    else:
        groups = [components == component_id for component_id in range(1, int(count) + 1)]

    for group in groups:
        rows, cols = np.nonzero(group)
        if len(rows) == 0:
            continue
        if len(rows) < int(min_mask_pixels):
            continue
        class_probs = probabilities[1:, rows, cols].mean(axis=1)
        class_probs = class_probs / max(float(class_probs.sum()), 1e-12)
        centre_rc = np.array([
            np.mean(rows + offset[0, rows, cols]),
            np.mean(cols + offset[1, rows, cols]),
        ], dtype=np.float32)
        if peaks and assignments is not None:
            # The mean center probability is more stable than taking a random
            # foreground pixel from a small object as the confidence anchor.
            peak_index = int(np.argmax([
                -np.linalg.norm(np.asarray(peak, dtype=float) - centre_rc)
                for peak in peaks
            ]))
            center_score = float(center_probability[tuple(peaks[peak_index])])
        else:
            center_score = float(np.max(center_probability[rows, cols]))
        score = float(np.mean(confidence[rows, cols]) * center_score)
        predictions.append(
            ImageInstancePrediction(
                mask=group,
                class_probs=class_probs.astype(np.float32),
                confidence=score,
                center_rc=centre_rc,
            )
        )
    predictions.sort(key=lambda item: (-item.confidence, float(item.center_rc[0]), float(item.center_rc[1])))
    return predictions[: int(max_instances)]


def _center_peaks(
    probability: np.ndarray,
    *,
    threshold: float,
    nms_radius: int,
    max_peaks: int,
) -> list[tuple[int, int]]:
    """Return spatially separated local maxima from a center heatmap."""

    values = np.asarray(probability, dtype=float)
    if values.ndim != 2:
        raise ValueError("center probability must be a 2-D array")
    if int(nms_radius) < 1 or int(max_peaks) < 1:
        raise ValueError("nms_radius and max_peaks must be positive")
    local_max = ndimage.maximum_filter(values, size=2 * int(nms_radius) + 1, mode="nearest")
    candidates = np.argwhere((values >= float(threshold)) & (values >= local_max - 1e-8))
    candidates = sorted(
        candidates.tolist(),
        key=lambda rc: (-float(values[tuple(rc)]), int(rc[0]), int(rc[1])),
    )
    peaks: list[tuple[int, int]] = []
    for row, col in candidates:
        if any(np.hypot(float(row - other_row), float(col - other_col)) < float(nms_radius) for other_row, other_col in peaks):
            continue
        peaks.append((int(row), int(col)))
        if len(peaks) >= int(max_peaks):
            break
    return peaks


def _assign_foreground_to_centers(
    foreground: np.ndarray,
    offset: np.ndarray,
    peaks: list[tuple[int, int]],
    *,
    max_distance: float,
) -> np.ndarray | None:
    """Assign foreground pixels to predicted centers, or return no grouping."""

    mask = np.asarray(foreground, dtype=bool)
    offsets = np.asarray(offset, dtype=float)
    if offsets.shape[0] != 2 or offsets.shape[1:] != mask.shape:
        raise ValueError("offset has incompatible shape")
    if not peaks:
        return None
    if float(max_distance) <= 0.0:
        raise ValueError("max_distance must be positive")
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    predicted_centers = np.stack((rows + offsets[0, rows, cols], cols + offsets[1, rows, cols]), axis=1)
    peak_array = np.asarray(peaks, dtype=float)
    distances = np.linalg.norm(predicted_centers[:, None, :] - peak_array[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(rows)), nearest]
    assignment = np.full(mask.shape, -1, dtype=np.int32)
    accepted = nearest_distance <= float(max_distance)
    assignment[rows[accepted], cols[accepted]] = nearest[accepted].astype(np.int32)
    return assignment
