"""Pinned LeRobot ACT adapter.

The base MuJoCo simulator remains usable without ML dependencies.  Training or
policy rollout asks explicitly for the optional ``act`` extra and fails with a
useful installation message when it is absent.
"""

from __future__ import annotations


def require_act_dependencies():
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "ACT dependencies are missing. Install with: "
            "uv pip install -e '.[act]'"
        ) from exc
    return FeatureType, PolicyFeature, ACTConfig, ACTPolicy


def build_act_policy(cfg, pretrained_path=None):
    FeatureType, PolicyFeature, ACTConfig, ACTPolicy = require_act_dependencies()
    import torch

    image_shape = (3, int(cfg.act.image_size[1]), int(cfg.act.image_size[0]))
    inputs = {
        "observation.images.overhead": PolicyFeature(FeatureType.VISUAL, image_shape),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, image_shape),
        "observation.state": PolicyFeature(FeatureType.STATE, (26,)),
        "observation.environment_state": PolicyFeature(FeatureType.ENV, (2,)),
    }
    outputs = {"action": PolicyFeature(FeatureType.ACTION, (int(cfg.act.action_dim),))}
    device = str(cfg.act.get("device", "mps"))
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    act_cfg = ACTConfig(
        n_obs_steps=1,
        chunk_size=int(cfg.act.chunk_size),
        n_action_steps=int(cfg.act.execute_steps),
        input_features=inputs,
        output_features=outputs,
        device=device,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        temporal_ensemble_coeff=None,
        push_to_hub=False,
        pretrained_path=pretrained_path,
    )
    policy = ACTPolicy(act_cfg)
    policy.to(device)
    return policy, act_cfg
