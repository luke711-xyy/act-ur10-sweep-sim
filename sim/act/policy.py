"""Pinned LeRobot ACT adapter.

The base MuJoCo simulator remains usable without ML dependencies.  Training or
policy rollout asks explicitly for the optional ``act`` extra and fails with a
useful installation message when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

from .variants import (
    GOAL_COUNT_CLASSES,
    GOAL_TOKEN_DIM,
    GOAL_TOKEN_POLICY,
    ORDINARY_POLICY,
    TASK_ID_FEATURE,
    normalize_policy_variant,
)


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


def build_act_config(cfg, policy_variant: str = ORDINARY_POLICY):
    FeatureType, PolicyFeature, ACTConfig, ACTPolicy = require_act_dependencies()
    policy_variant = normalize_policy_variant(policy_variant)

    image_shape = (3, int(cfg.act.image_size[1]), int(cfg.act.image_size[0]))
    inputs = {
        "observation.images.overhead": PolicyFeature(FeatureType.VISUAL, image_shape),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, image_shape),
        "observation.state": PolicyFeature(
            FeatureType.STATE, (int(cfg.act.state_dim),)
        ),
    }
    if policy_variant == GOAL_TOKEN_POLICY:
        inputs[TASK_ID_FEATURE] = PolicyFeature(FeatureType.ENV, (1,))
    else:
        inputs["observation.environment_state"] = PolicyFeature(FeatureType.ENV, (3,))
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
    """Reject checkpoints built for an incompatible ACT state/action contract."""
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
            f"required={expected_state}/{expected_action} (schema v5)"
        )


def build_act_policy(cfg, pretrained_path=None,
                     policy_variant: str = ORDINARY_POLICY):
    policy_variant = normalize_policy_variant(policy_variant)
    if policy_variant == GOAL_TOKEN_POLICY:
        act_cfg = build_act_config(cfg, policy_variant=policy_variant)
        if int(act_cfg.dim_model) != GOAL_TOKEN_DIM:
            raise ValueError(
                f"goal-token ACT requires dim_model={GOAL_TOKEN_DIM}, "
                f"got {act_cfg.dim_model}"
            )
        if pretrained_path:
            policy = load_goal_token_policy(pretrained_path, device=act_cfg.device)
            validate_goal_token_contract(policy.config, cfg)
            return policy, policy.config
        policy = create_goal_token_policy(act_cfg)
        policy.to(act_cfg.device)
        validate_goal_token_contract(act_cfg, cfg)
        return policy, act_cfg

    _, _, _, ACTPolicy = require_act_dependencies()
    act_cfg = build_act_config(cfg, policy_variant=policy_variant)
    if pretrained_path:
        saved_variant = policy_variant_from_checkpoint(pretrained_path)
        if saved_variant != ORDINARY_POLICY:
            raise ValueError(
                f"checkpoint is {saved_variant!r}, but ordinary ACT was requested"
            )
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


def validate_goal_token_contract(policy_cfg, cfg=None) -> None:
    FeatureType, _, _, _ = require_act_dependencies()
    validate_act_policy_contract(policy_cfg, cfg) if cfg is not None else None
    task_feature = policy_cfg.input_features.get(TASK_ID_FEATURE)
    if (task_feature is None or task_feature.type is not FeatureType.ENV
            or tuple(task_feature.shape) != (1,)):
        raise ValueError(
            "goal-token checkpoint must expose observation.task_id as a "
            "one-value categorical ENV feature"
        )
    if "observation.environment_state" in policy_cfg.input_features:
        raise ValueError(
            "goal-token checkpoint must not include total/collected environment counts"
        )
    if int(policy_cfg.dim_model) != GOAL_TOKEN_DIM:
        raise ValueError(
            f"goal-token checkpoint dim_model must be {GOAL_TOKEN_DIM}, "
            f"got {policy_cfg.dim_model}"
        )


def _goal_token_policy_classes():
    import torch
    from lerobot.policies.act.modeling_act import (
        ACT,
        ACTPolicy,
        ACTTemporalEnsembler,
    )
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

    _, _, ACTConfig, _ = require_act_dependencies()

    class GoalTokenACT(ACT):
        def __init__(self, config):
            super().__init__(config)
            if config.env_state_feature is None:
                raise ValueError("goal-token ACT requires a categorical ENV feature")
            self.encoder_env_state_input_proj = torch.nn.Embedding(
                GOAL_COUNT_CLASSES, int(config.dim_model)
            )

        def forward(self, batch):
            task_ids = batch.get(TASK_ID_FEATURE)
            if task_ids is None:
                raise KeyError(f"missing required ACT input {TASK_ID_FEATURE!r}")
            task_ids = task_ids.to(
                device=batch[OBS_STATE].device, dtype=torch.long
            )
            if task_ids.ndim == 2 and task_ids.shape[-1] == 1:
                task_ids = task_ids.squeeze(-1)
            if task_ids.ndim != 1:
                raise ValueError(
                    f"task IDs must have shape (batch,) or (batch, 1), "
                    f"got {tuple(task_ids.shape)}"
                )
            token_batch = dict(batch)
            # LeRobot ACT already has a dedicated ENV token slot in its encoder.
            # Route the categorical IDs through that slot; no numeric count
            # vector is created or exposed at the policy boundary.
            token_batch[OBS_ENV_STATE] = task_ids
            return super().forward(token_batch)

    class GoalTokenACTPolicy(ACTPolicy):
        config_class = ACTConfig
        name = "act_goal_token"

        def __init__(self, config, **kwargs):
            # Mirror the pinned ACTPolicy initialization while swapping in the
            # task-token model, avoiding a temporary second ResNet allocation.
            PreTrainedPolicy.__init__(self, config)
            config.validate_features()
            self.config = config
            self.model = GoalTokenACT(config)
            if config.temporal_ensemble_coeff is not None:
                self.temporal_ensembler = ACTTemporalEnsembler(
                    config.temporal_ensemble_coeff, config.chunk_size
                )
            self.reset()

    return GoalTokenACTPolicy


def create_goal_token_policy(policy_cfg):
    """Construct ACT with a learned six-class token in its ENV encoder slot."""
    GoalTokenACTPolicy = _goal_token_policy_classes()
    validate_goal_token_contract(policy_cfg)
    return GoalTokenACTPolicy(policy_cfg)


def policy_variant_from_checkpoint(checkpoint_path: str | Path) -> str:
    marker_path = Path(checkpoint_path) / "policy_variant.json"
    if not marker_path.exists():
        # Historical v10 ACT checkpoints predate the marker and are ordinary.
        return ORDINARY_POLICY
    try:
        metadata = json.loads(marker_path.read_text(encoding="utf-8"))
        return normalize_policy_variant(metadata.get("policy_variant"))
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"invalid policy variant marker: {marker_path}") from exc


def write_policy_variant_metadata(checkpoint_path: str | Path,
                                  policy_variant: str) -> Path:
    variant = normalize_policy_variant(policy_variant)
    path = Path(checkpoint_path) / "policy_variant.json"
    metadata = {"policy_variant": variant}
    if variant == GOAL_TOKEN_POLICY:
        metadata.update({
            "task_id_encoding": "target_count_minus_one",
            "goal_count_classes": GOAL_COUNT_CLASSES,
            "embedding_dim": GOAL_TOKEN_DIM,
        })
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def load_goal_token_policy(checkpoint_path: str | Path, device=None):
    """Load a tagged goal-token checkpoint with the pinned LeRobot ACT config."""
    from safetensors.torch import load_file

    path = Path(checkpoint_path)
    if policy_variant_from_checkpoint(path) != GOAL_TOKEN_POLICY:
        raise ValueError(
            f"checkpoint does not declare policy_variant={GOAL_TOKEN_POLICY!r}: {path}"
        )
    _, _, ACTConfig, _ = require_act_dependencies()
    policy_cfg = ACTConfig.from_pretrained(path, local_files_only=True)
    if device is not None:
        policy_cfg.device = str(device)
    validate_goal_token_contract(policy_cfg)
    policy = create_goal_token_policy(policy_cfg)
    weights_path = path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"goal-token ACT weights not found: {weights_path}")
    state_dict = load_file(str(weights_path), device="cpu")
    policy.load_state_dict(state_dict, strict=True)
    policy.to(policy_cfg.device)
    return policy


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
