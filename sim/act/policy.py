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


def build_act_config(cfg):
    FeatureType, PolicyFeature, ACTConfig, ACTPolicy = require_act_dependencies()

    image_shape = (3, int(cfg.act.image_size[1]), int(cfg.act.image_size[0]))
    inputs = {
        "observation.images.overhead": PolicyFeature(FeatureType.VISUAL, image_shape),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, image_shape),
        "observation.state": PolicyFeature(
            FeatureType.STATE, (int(cfg.act.state_dim),)
        ),
        "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,)),
    }
    outputs = {"action": PolicyFeature(FeatureType.ACTION, (int(cfg.act.action_dim),))}
    device = str(cfg.act.get("device", "mps"))
    import torch

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
    )
    return act_cfg


def validate_act_policy_contract(policy_cfg, cfg) -> None:
    """Reject checkpoints built for the retired 40D/3D interface."""
    state_width = int(
        policy_cfg.input_features["observation.state"].shape[0]
    )
    action_width = int(policy_cfg.output_features["action"].shape[0])
    expected_state = int(cfg.act.state_dim)
    expected_action = int(cfg.act.action_dim)
    if state_width != expected_state or action_width != expected_action:
        raise ValueError(
            "ACT checkpoint/interface mismatch: "
            f"checkpoint state/action={state_width}/{action_width}, "
            f"required={expected_state}/{expected_action} (schema v4)"
        )


def build_act_policy(cfg, pretrained_path=None):
    _, _, _, ACTPolicy = require_act_dependencies()
    act_cfg = build_act_config(cfg)
    if pretrained_path:
        # Loading through LeRobot's official API restores the saved policy
        # config and safetensors weights instead of merely attaching a path to
        # a freshly initialised random network.
        policy = ACTPolicy.from_pretrained(
            pretrained_path,
            config=act_cfg,
            local_files_only=True,
        )
        validate_act_policy_contract(policy.config, cfg)
        return policy, policy.config
    policy = ACTPolicy(act_cfg)
    policy.to(act_cfg.device)
    validate_act_policy_contract(act_cfg, cfg)
    return policy, act_cfg


def build_act_processors(policy_cfg, dataset_stats=None, pretrained_path=None):
    """Build LeRobot's official ACT pre/post-processing pipelines.

    Training creates fresh pipelines from the training-set statistics.  When a
    saved model is supplied, the serialized processor files are loaded so
    inference uses exactly the same normalization contract as training.
    """
    if pretrained_path:
        from lerobot.policies import make_pre_post_processors

        preprocessor_overrides = {
            # A processor saved on MPS/CUDA must still follow the device on
            # which the current policy was loaded (for example CPU in a
            # validation job).  This mirrors LeRobot's official train/eval
            # entrypoints.
            "device_processor": {"device": str(policy_cfg.device)},
        }
        if dataset_stats is not None:
            preprocessor_overrides["normalizer_processor"] = {"stats": dataset_stats}
        postprocessor_overrides = {}
        if dataset_stats is not None:
            postprocessor_overrides["unnormalizer_processor"] = {"stats": dataset_stats}
        return make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            dataset_stats=dataset_stats,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

    from lerobot.policies.act import make_act_pre_post_processors

    return make_act_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)
