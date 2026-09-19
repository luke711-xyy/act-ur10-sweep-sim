"""ObjectACT-BEV adapter built on the pinned LeRobot 0.6.1 ACT modules.

The official ACT CVAE, transformer encoder/decoder, action loss and temporal
ensembler are reused.  Object tokens and the table BEV enter as separate token
groups; they are not flattened into the robot state or environment vector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F
from torch import nn

from .selection import (
    PermutationInvariantSelectionHead,
    SelectionTeacherSchedule,
    scheduled_selection,
    selection_bce_loss,
    straight_through_top_n,
)


@dataclass(frozen=True)
class ObjectACTConfig:
    image_size: tuple[int, int] = (320, 320)
    chunk_size: int = 25
    robot_state_dim: int = 36
    task_state_dim: int = 6
    object_slots: int = 6
    object_token_dim: int = 29
    bev_channels: int = 6
    bev_height: int = 128
    bev_width: int = 160
    action_dim: int = 4
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 1024
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    use_vae: bool = True
    kl_weight: float = 10.0
    dropout: float = 0.1
    temporal_ensemble_coeff: float | None = 0.01
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    device: str = "mps"

    def __post_init__(self) -> None:
        if self.robot_state_dim != 36:
            raise ValueError("ObjectACT requires a 36D robot state")
        if self.action_dim != 4:
            raise ValueError("ObjectACT requires a 4D [dx, dy, dz, dyaw] action")
        if self.object_slots != 6 or self.object_token_dim != 29:
            raise ValueError("ObjectACT requires six 29D object tokens")
        if self.bev_channels != 6 or (self.bev_height, self.bev_width) != (128, 160):
            raise ValueError("ObjectACT requires a [6, 128, 160] BEV")
        if self.chunk_size != 25:
            raise ValueError("ObjectACT v5 uses a 25-step action chunk")
        if self.temporal_ensemble_coeff is not None and self.chunk_size < 1:
            raise ValueError("temporal ensembling requires a positive chunk size")

    def lerobot_config(self):
        try:
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.policies.act.configuration_act import ACTConfig
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ObjectACT requires lerobot==0.6.1") from exc
        import torch

        height, width = int(self.image_size[1]), int(self.image_size[0])
        inputs = {
            "observation.images.overhead": PolicyFeature(FeatureType.VISUAL, (3, height, width)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, height, width)),
            "observation.state": PolicyFeature(FeatureType.STATE, (self.robot_state_dim,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (self.task_state_dim,)),
        }
        outputs = {"action": PolicyFeature(FeatureType.ACTION, (self.action_dim,))}
        device = str(self.device)
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        return ACTConfig(
            n_obs_steps=1,
            chunk_size=self.chunk_size,
            n_action_steps=1,
            input_features=inputs,
            output_features=outputs,
            vision_backbone="resnet18",
            pretrained_backbone_weights=self.pretrained_backbone_weights,
            dim_model=self.dim_model,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            n_vae_encoder_layers=self.n_vae_encoder_layers,
            latent_dim=self.latent_dim,
            use_vae=self.use_vae,
            kl_weight=self.kl_weight,
            dropout=self.dropout,
            temporal_ensemble_coeff=self.temporal_ensemble_coeff,
            device=device,
            push_to_hub=False,
        )


def _require_lerobot():
    try:
        from lerobot.policies.act.modeling_act import ACT, ACTTemporalEnsembler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ObjectACT requires lerobot==0.6.1") from exc
    return ACT, ACTTemporalEnsembler


class ObjectACTModel(nn.Module):
    """ACT's CVAE/decoder with separate object and BEV token branches."""

    def __init__(self, config, object_config: ObjectACTConfig):
        ACT, _ = _require_lerobot()
        super().__init__()
        self.config = config
        self.object_config = object_config
        self.base = ACT(config)
        for parameter in self.base.backbone.parameters():
            parameter.requires_grad_(False)
        self.base.backbone.eval()

        dim = int(config.dim_model)
        self.object_projection = nn.Sequential(
            nn.Linear(29, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.object_modality = nn.Parameter(torch.zeros(1, 1, dim))
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(6, max(16, dim // 4), kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(max(16, dim // 4), max(16, dim // 2), kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(max(16, dim // 2), max(16, (3 * dim) // 4), kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(max(16, (3 * dim) // 4), dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        bev_height = int(object_config.bev_height) // 16
        bev_width = int(object_config.bev_width) // 16
        self.bev_tokens = bev_height * bev_width
        self.bev_modality = nn.Parameter(torch.zeros(1, 1, dim))
        self.extra_position = nn.Parameter(
            torch.zeros(int(object_config.object_slots) + self.bev_tokens, dim)
        )
        self.camera_modality = nn.Parameter(torch.zeros(2, dim, 1, 1))

    @staticmethod
    def _key(batch: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name not in batch:
            raise ValueError(f"ObjectACT batch is missing {name}")
        return batch[name]

    def _latent(self, batch: dict[str, torch.Tensor], robot_state: torch.Tensor):
        batch_size = robot_state.shape[0]
        if self.config.use_vae and self.training and "action" in batch:
            cls_embed = self.base.vae_encoder_cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            robot_embed = self.base.vae_encoder_robot_state_input_proj(robot_state).unsqueeze(1)
            action_embed = self.base.vae_encoder_action_input_proj(batch["action"])
            vae_input = torch.cat([cls_embed, robot_embed, action_embed], dim=1)
            padding = torch.cat(
                [
                    torch.zeros((batch_size, 2), dtype=torch.bool, device=robot_state.device),
                    batch["action_is_pad"].to(dtype=torch.bool),
                ],
                dim=1,
            )
            encoded = self.base.vae_encoder(
                vae_input.permute(1, 0, 2),
                pos_embed=self.base.vae_encoder_pos_enc.detach().permute(1, 0, 2),
                key_padding_mask=padding,
            )[0]
            params = self.base.vae_encoder_latent_output_proj(encoded)
            mu = params[:, : self.config.latent_dim]
            log_sigma_x2 = params[:, self.config.latent_dim :]
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            return latent, mu, log_sigma_x2
        latent = torch.zeros(
            (batch_size, self.config.latent_dim), dtype=robot_state.dtype, device=robot_state.device
        )
        return latent, None, None

    def forward(self, batch: dict[str, torch.Tensor]):
        robot_state = self._key(batch, "observation.robot_state")
        task_state = self._key(batch, "observation.task_state")
        object_tokens = self._key(batch, "observation.object_tokens")
        object_valid = self._key(batch, "observation.object_valid")
        bev = self._key(batch, "observation.bev")
        if robot_state.ndim != 2 or robot_state.shape[1] != 36:
            raise ValueError("observation.robot_state must have shape (B, 36)")
        if task_state.ndim != 2 or task_state.shape[1] != 6:
            raise ValueError("observation.task_state must have shape (B, 6)")
        if object_tokens.ndim != 3 or object_tokens.shape[1:] != (6, 29):
            raise ValueError("observation.object_tokens must have shape (B, 6, 29)")
        if object_valid.shape != object_tokens.shape[:2] or object_valid.dtype != torch.bool:
            raise ValueError("observation.object_valid must be boolean with shape (B, 6)")
        if bev.ndim != 4 or bev.shape[1:] != (6, 128, 160):
            raise ValueError("observation.bev must have shape (B, 6, 128, 160)")
        batch_size = robot_state.shape[0]
        latent, mu, log_sigma_x2 = self._latent(batch, robot_state)

        base = self.base
        token_parts = [
            base.encoder_latent_input_proj(latent).unsqueeze(0),
            base.encoder_robot_state_input_proj(robot_state).unsqueeze(0),
            base.encoder_env_state_input_proj(task_state).unsqueeze(0),
        ]
        pos_parts = [
            base.encoder_1d_feature_pos_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
        ]
        padding_parts = [torch.zeros((batch_size, 3), dtype=torch.bool, device=robot_state.device)]

        objects = self.object_projection(object_tokens) + self.object_modality
        token_parts.append(objects.transpose(0, 1))
        pos_parts.append(self.extra_position[:6].unsqueeze(1).expand(-1, batch_size, -1))
        padding_parts.append(~object_valid)

        bev_features = self.bev_encoder(bev)
        if bev_features.shape[-2:] != (8, 10):
            raise ValueError(f"BEV encoder must produce 8x10 tokens, got {bev_features.shape[-2:]}")
        bev_tokens = einops.rearrange(bev_features, "b c h w -> (h w) b c") + self.bev_modality
        token_parts.append(bev_tokens)
        pos_parts.append(self.extra_position[6:].unsqueeze(1).expand(-1, batch_size, -1))
        padding_parts.append(torch.zeros((batch_size, self.bev_tokens), dtype=torch.bool, device=robot_state.device))

        image_keys = ("observation.images.overhead", "observation.images.wrist")
        for camera_index, key in enumerate(image_keys):
            image = self._key(batch, key)
            feature_map = base.backbone(image)["feature_map"]
            position = base.encoder_cam_feat_pos_embed(feature_map).to(dtype=feature_map.dtype)
            if position.shape[0] == 1 and batch_size > 1:
                position = position.expand(batch_size, -1, -1, -1)
            feature_map = base.encoder_img_feat_input_proj(feature_map)
            feature_map = feature_map + self.camera_modality[camera_index]
            token_parts.append(einops.rearrange(feature_map, "b c h w -> (h w) b c"))
            pos_parts.append(einops.rearrange(position, "b c h w -> (h w) b c"))
            padding_parts.append(torch.zeros((batch_size, feature_map.shape[-2] * feature_map.shape[-1]), dtype=torch.bool, device=robot_state.device))

        encoder_tokens = torch.cat(token_parts, dim=0)
        encoder_positions = torch.cat(pos_parts, dim=0)
        key_padding_mask = torch.cat(padding_parts, dim=1)
        encoded = base.encoder(
            encoder_tokens,
            pos_embed=encoder_positions,
            key_padding_mask=key_padding_mask,
        )
        decoder_input = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_positions.dtype,
            device=encoder_positions.device,
        )
        decoded = base.decoder(
            decoder_input,
            encoded,
            encoder_pos_embed=encoder_positions,
            decoder_pos_embed=base.decoder_pos_embed.weight.unsqueeze(1),
        ).transpose(0, 1)
        return base.action_head(decoded), (mu, log_sigma_x2)


class ObjectACTPolicy(nn.Module):
    """Trainable ObjectACT policy with the ordinary ACT public call pattern."""

    def __init__(self, config: ObjectACTConfig):
        super().__init__()
        self.object_config = config
        self.config = config.lerobot_config()
        self.model = ObjectACTModel(self.config, config)
        self.selection_head = PermutationInvariantSelectionHead(
            token_dim=config.object_token_dim,
            hidden_dim=config.dim_model,
            heads=config.n_heads,
        )
        self.selection_schedule = SelectionTeacherSchedule()
        self.selection_step = 0
        self.last_selection_logits = None
        self.last_selection = None
        _, temporal_ensembler = _require_lerobot()
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = temporal_ensembler(config.temporal_ensemble_coeff, config.chunk_size)
        else:
            self._action_queue = deque([], maxlen=1)
        self.reset()

    def _validate_batch(self, batch: dict[str, torch.Tensor], *, training: bool) -> None:
        if "observation.state" in batch or "observation.environment_state" in batch:
            raise ValueError("ObjectACT batch must use explicit robot_state/task_state keys")
        required = (
            "observation.images.overhead", "observation.images.wrist",
            "observation.robot_state", "observation.task_state",
            "observation.object_tokens", "observation.object_valid", "observation.bev",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"ObjectACT batch is missing {missing}")
        if training and self.object_config.use_vae and "action" not in batch:
            raise ValueError("training ObjectACT batches need action targets")

    def get_optim_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def _target_counts(task_state: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = task_state[:, 1]
        # v5 stores counts normalized to the six-object maximum.  Accept raw
        # counts as well for a debugging batch, but never let a malformed
        # visual observation request more slots than the detector provides.
        counts = torch.where(values.abs() <= 1.5, torch.round(values * 6.0), torch.round(values))
        return counts.to(dtype=torch.long).clamp(min=1, max=valid.shape[1])

    def _prepare_selection(self, batch: dict[str, torch.Tensor], *, training: bool):
        tokens = batch["observation.object_tokens"]
        valid = batch["observation.object_valid"]
        counts = self._target_counts(batch["observation.task_state"], valid)
        logits = self.selection_head(tokens, valid, counts)
        target = batch.get("selection_target")
        if target is not None:
            selection = scheduled_selection(
                logits,
                target_mask=target.to(dtype=torch.bool),
                target_count=counts,
                valid_mask=valid,
                step=self.selection_step,
                schedule=self.selection_schedule,
            )
            selection_loss = selection_bce_loss(logits, target, valid)
        else:
            selection = straight_through_top_n(
                logits, target_count=counts, valid_mask=valid
            )
            selection_loss = logits.sum() * 0.0
        prepared = dict(batch)
        if "observation.instance_bev" in batch:
            masks = batch["observation.instance_bev"].to(dtype=torch.bool)
            if masks.shape[1:] != (6, 128, 160):
                raise ValueError("observation.instance_bev must have shape (B, 6, 128, 160)")
            all_mask = (masks & valid[:, :, None, None]).any(dim=1).to(dtype=prepared["observation.bev"].dtype)
            selected_mask = (
                masks & valid[:, :, None, None] & selection.detach().to(dtype=torch.bool)[:, :, None, None]
            ).any(dim=1).to(dtype=prepared["observation.bev"].dtype)
            bev = prepared["observation.bev"].clone()
            bev[:, 0] = all_mask
            bev[:, 1] = selected_mask
            bev[:, 2] = torch.clamp(all_mask - selected_mask, min=0.0)
            prepared["observation.bev"] = bev
        self.last_selection_logits = logits.detach()
        self.last_selection = selection.detach()
        return prepared, selection_loss

    def reset(self) -> None:
        if hasattr(self, "temporal_ensembler"):
            self.temporal_ensembler.reset()
        elif hasattr(self, "_action_queue"):
            self._action_queue.clear()

    def forward(self, batch: dict[str, torch.Tensor]):
        self._validate_batch(batch, training=True)
        batch, selection_loss = self._prepare_selection(batch, training=True)
        actions_hat, (mu, log_sigma_x2) = self.model(batch)
        if batch["action"].shape != actions_hat.shape:
            raise ValueError(f"action target must have shape {tuple(actions_hat.shape)}")
        abs_err = F.l1_loss(batch["action"], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].to(dtype=torch.bool).unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        losses = {"l1_loss": float(l1_loss.detach().item())}
        losses["selection_bce_loss"] = float(selection_loss.detach().item())
        if self.object_config.use_vae and log_sigma_x2 is not None:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            losses["kld_loss"] = float(kld.detach().item())
            total = l1_loss + kld * float(self.object_config.kl_weight) + selection_loss
            self.selection_step += 1
            return total, losses
        self.selection_step += 1
        return l1_loss + selection_loss, losses

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._validate_batch(batch, training=False)
        self.eval()
        batch, _ = self._prepare_selection(batch, training=False)
        return self.model(batch)[0]

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        if hasattr(self, "temporal_ensembler"):
            return self.temporal_ensembler.update(self.predict_action_chunk(batch))
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions[:, :1].transpose(0, 1))
        return self._action_queue.popleft()
