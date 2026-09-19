import numpy as np
import pytest
import torch

from sim.act.object_policy import ObjectACTConfig, ObjectACTPolicy


def make_policy(temporal_ensemble_coeff=None):
    config = ObjectACTConfig(
        image_size=(64, 64),
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )
    return ObjectACTPolicy(config)


def make_batch(batch_size=2, include_action=True):
    batch = {
        "observation.images.overhead": torch.randn(batch_size, 3, 64, 64),
        "observation.images.wrist": torch.randn(batch_size, 3, 64, 64),
        "observation.robot_state": torch.randn(batch_size, 36),
        "observation.task_state": torch.randn(batch_size, 6),
        "observation.object_tokens": torch.randn(batch_size, 6, 29),
        "observation.object_valid": torch.ones(batch_size, 6, dtype=torch.bool),
        "observation.bev": torch.randn(batch_size, 6, 128, 160),
    }
    if include_action:
        batch["action"] = torch.randn(batch_size, 25, 4)
        batch["action_is_pad"] = torch.zeros(batch_size, 25, dtype=torch.bool)
    return batch


def test_objectact_forward_preserves_action_chunk_and_standard_act_losses():
    policy = make_policy()
    batch = make_batch()
    loss, logs = policy(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert {"l1_loss", "kld_loss"} <= set(logs)
    with torch.no_grad():
        actions = policy.predict_action_chunk(make_batch(include_action=False))
    assert actions.shape == (2, 25, 4)


def test_objectact_eval_inference_uses_deterministic_zero_latent():
    policy = make_policy()
    batch = make_batch(batch_size=1, include_action=False)
    with torch.no_grad():
        first = policy.predict_action_chunk(batch)
        second = policy.predict_action_chunk(batch)
    torch.testing.assert_close(first, second)


def test_objectact_rejects_legacy_v4_state_and_wrong_action_width():
    with pytest.raises(ValueError, match="36D robot state"):
        ObjectACTConfig(robot_state_dim=42)
    policy = make_policy()
    bad = make_batch(batch_size=1)
    bad["observation.state"] = bad.pop("observation.robot_state")
    with pytest.raises(ValueError, match="robot_state"):
        policy.predict_action_chunk(bad)


def test_objectact_select_action_can_use_official_temporal_ensemble():
    policy = make_policy(temporal_ensemble_coeff=0.01)
    action = policy.select_action(make_batch(batch_size=1, include_action=False))
    assert action.shape == (1, 4)


def test_selector_rewrites_bev_selection_channels_and_reports_selection_loss():
    policy = make_policy()
    batch = make_batch(batch_size=1)
    batch["observation.task_state"] = torch.tensor([[1.0, 0.5, 0.0, 1.0, 0.5, 0.0]])
    batch["observation.instance_bev"] = torch.zeros(1, 6, 128, 160, dtype=torch.bool)
    batch["observation.instance_bev"][0, 0, 64, 80] = True
    batch["observation.instance_bev"][0, 1, 64, 90] = True
    batch["selection_target"] = torch.tensor([[True, False, False, False, False, False]])
    loss, logs = policy(batch)
    assert torch.isfinite(loss)
    assert "selection_bce_loss" in logs
    assert policy.last_selection.shape == (1, 6)


def test_objectact_policy_factory_can_bootstrap_before_first_checkpoint():
    from sim.act.policy import build_objectact_policy
    from sim.config import load_config

    cfg = load_config(overrides=[
        "act.image_size=[64,64]",
        "act.device=cpu",
        "act.objectact_pretrained_weights=null",
    ])
    policy, object_config, checkpoint = build_objectact_policy(cfg)
    assert checkpoint is None
    assert object_config.robot_state_dim == 36
    assert policy.config.device == "cpu"
