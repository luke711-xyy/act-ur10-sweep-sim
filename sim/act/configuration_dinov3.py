"""Serialized contract for frozen pretrained DINOv3 + the pinned LeRobot ACT."""

from dataclasses import dataclass, field, fields

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


DINO_MODEL_ID = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINO_WEIGHT_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"


def base_act_config(config):
    """Reuse upstream ACT validation/construction without changing its source.

    The upstream constructor only knows ResNet. Its temporary, unpretrained
    visual module is discarded by our policy, while all ACT/VAE settings are
    retained. This construction config is never the serialized policy identity.
    """
    values = {item.name: getattr(config, item.name) for item in fields(ACTConfig)
              if item.init}
    values.update(vision_backbone="resnet18", pretrained_backbone_weights=None)
    return ACTConfig(**values)


@PreTrainedConfig.register_subclass("act_dinov3")
@dataclass
class DINOv3ACTConfig(ACTConfig):
    vision_backbone: str = "dinov3_vits16"
    pretrained_backbone_weights: str | None = None
    dino_model_id: str = DINO_MODEL_ID
    dino_revision: str = DINO_REVISION
    dino_weight_sha256: str = DINO_WEIGHT_SHA256
    dino_pool_stride: int = 2
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        validated = base_act_config(self)
        self.device, self.use_amp = validated.device, validated.use_amp
        if (self.vision_backbone != "dinov3_vits16"
                or self.dino_model_id != DINO_MODEL_ID
                or self.dino_revision != DINO_REVISION
                or self.dino_weight_sha256 != DINO_WEIGHT_SHA256):
            raise ValueError("DINOv3 requires the pinned pretrained ViT-S/16 identity")
        if self.pretrained_backbone_weights is not None or self.use_peft:
            raise ValueError("DINOv3 is frozen; ResNet weights and PEFT are not supported")
        if self.dino_pool_stride != 2:
            raise ValueError("this DINOv3 experiment requires 2x2 spatial pooling")
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError("DINOv3 images must not receive dataset normalization")
        for feature in self.image_features.values():
            if (len(feature.shape) != 3 or feature.shape[0] != 3
                    or any(size <= 0 or size % 32 for size in feature.shape[1:])):
                raise ValueError("DINOv3 RGB image dimensions must be multiples of 32")
