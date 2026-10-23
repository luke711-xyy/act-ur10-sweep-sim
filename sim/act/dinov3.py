"""Frozen DINOv3 spatial features with the unchanged LeRobot ACT/VAE core."""

from __future__ import annotations

import hashlib

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from lerobot.policies.act.modeling_act import ACTPolicy

from .configuration_dinov3 import DINOv3ACTConfig, base_act_config


def load_dino_encoder(config: DINOv3ACTConfig, *, pretrained: bool = True):
    """Load a pinned, checksum-verified encoder; never fall back to random weights.

    ``pretrained=False`` only constructs the architecture for strict restoration
    of a full ACT checkpoint, which already contains all frozen encoder weights.
    """
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("DINOv3 requires the optional extra: pip install -e '.[act,dino]'") from exc

    encoder = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=False, num_classes=0,
    )
    if pretrained:
        path = hf_hub_download(
            repo_id=config.dino_model_id, filename="model.safetensors",
            revision=config.dino_revision,
        )
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != config.dino_weight_sha256:
            raise ValueError("DINOv3 pretrained weight checksum mismatch")
        encoder.load_state_dict(load_file(path, device="cpu"), strict=True)
    return encoder


class FrozenDINOv3Backbone(nn.Module):
    """Return ACT's feature_map contract while keeping only spatial patch tokens."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image):
        if (image.ndim != 4 or image.shape[1] != 3
                or not image.is_floating_point()):
            raise ValueError("DINOv3 expects floating-point B,C,H,W RGB images")
        batch, _, height, width = image.shape
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("DINOv3 RGB dimensions must be positive multiples of 32")
        invalid = (~torch.isfinite(image)) | (image < 0) | (image > 1)
        if bool(invalid.any()):
            raise ValueError("DINOv3 expects finite, unnormalized RGB values in [0,1]")
        with torch.no_grad():
            tokens = self.encoder.forward_features((image - self.image_mean) / self.image_std)
            patches = tokens[:, self.encoder.num_prefix_tokens:, :]
            grid_height, grid_width = height // 16, width // 16
            if patches.shape[1] != grid_height * grid_width:
                raise ValueError("DINOv3 patch count does not match the RGB spatial grid")
            feature_map = patches.transpose(1, 2).reshape(
                batch, self.encoder.num_features, grid_height, grid_width,
            )
            feature_map = F.avg_pool2d(feature_map, kernel_size=2, stride=2)
        return {"feature_map": feature_map}


class DINOv3ACTPolicy(ACTPolicy):
    """Original ACT/VAE, a trainable image projection, and a frozen DINO encoder."""

    config_class = DINOv3ACTConfig
    name = "act_dinov3"

    def __init__(self, config: DINOv3ACTConfig, *, load_backbone_weights: bool = True):
        if not isinstance(config, DINOv3ACTConfig):
            raise TypeError("DINOv3ACTPolicy requires its serialized DINOv3ACTConfig")
        # Reuse the pinned upstream ACT implementation, not a fork of its
        # Transformer, VAE, losses, chunking or scheduler. The unpretrained
        # ResNet construction stub is immediately replaced and not saved.
        super().__init__(base_act_config(config))
        self.config = config
        self.model.config = config
        encoder = load_dino_encoder(config, pretrained=load_backbone_weights)
        self.model.backbone = FrozenDINOv3Backbone(encoder)
        self.model.encoder_img_feat_input_proj = nn.Conv2d(
            encoder.num_features, config.dim_model, kernel_size=1,
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        if config is None:
            config = DINOv3ACTConfig.from_pretrained(
                pretrained_name_or_path,
                local_files_only=kwargs.get("local_files_only", False),
            )
        # Frozen weights are part of this checkpoint. Missing weights are an
        # error, never a reason to download another encoder or use random init.
        kwargs["strict"] = True
        kwargs["load_backbone_weights"] = False
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)
