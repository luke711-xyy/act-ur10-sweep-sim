"""Contracts for a frozen DINOv3 encoder feeding the existing ACT policy."""

import hashlib
import json

import pytest
import torch

from sim.act.policy import build_act_config
from sim.config import load_config


def dino_sim_config():
    return load_config(
        "configs/five-view-fine-teached-300.yaml",
        overrides=["act.device=cpu", "act.vision_encoder=dinov3"],
    )


def test_dino_config_identifies_the_new_encoder_without_relabeling_resnet():
    # Catches accidentally retaining the ResNet factory or serialized identity.
    config = build_act_config(dino_sim_config())

    assert config.type == "act_dinov3"
    assert config.vision_backbone == "dinov3_vits16"
    assert config.pretrained_backbone_weights is None
    assert config.input_features["observation.images.wrist"].shape == (3, 320, 320)
    assert len(config.image_features) == 5
    assert config.chunk_size == 25
    assert config.n_action_steps == 5
    assert config.use_vae


def test_dino_processors_must_not_apply_the_old_image_dataset_normalization():
    # DINO applies its pretrained ImageNet transform exactly once internally.
    config = build_act_config(dino_sim_config())

    assert config.normalization_mapping["VISUAL"].value == "IDENTITY"
    assert config.normalization_mapping["STATE"].value == "MEAN_STD"
    assert config.normalization_mapping["ACTION"].value == "MEAN_STD"


def dino_module():
    try:
        from sim.act import dinov3
    except ImportError:
        pytest.fail("the frozen DINOv3 ACT adapter has not been implemented")
    return dinov3


def tiny_encoder():
    import timm

    # Real DINO architecture, shortened only for offline unit tests. This is
    # not a pretrained-model validation; that is a separate real-data smoke run.
    return timm.create_model(
        "vit_small_patch16_dinov3", pretrained=False, depth=1,
        img_size=32, num_classes=0,
    )


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def small_policy_config():
    cfg = dino_sim_config()
    cfg.act.image_size = [32, 32]
    config = build_act_config(cfg)
    config.dim_model = 32
    config.n_heads = 4
    config.dim_feedforward = 64
    config.n_encoder_layers = 1
    config.n_decoder_layers = 1
    config.n_vae_encoder_layers = 1
    config.chunk_size = 3
    config.n_action_steps = 1
    return config


def policy_batch(config):
    batch = {
        key: torch.rand(2, *feature.shape)
        for key, feature in config.image_features.items()
    }
    batch.update({
        "observation.state": torch.randn(2, 36),
        "observation.environment_state": torch.ones(2, 1),
        "action": torch.randn(2, config.chunk_size, 4),
        "action_is_pad": torch.zeros(2, config.chunk_size, dtype=torch.bool),
    })
    return batch


def test_backbone_preserves_spatial_map_and_does_not_unfreeze_in_train_mode():
    module = dino_module()
    encoder = tiny_encoder()
    backbone = module.FrozenDINOv3Backbone(encoder)
    before = parameter_digest(encoder)
    backbone.train()
    features = backbone(torch.rand(2, 3, 32, 32))["feature_map"]
    projection = torch.nn.Conv2d(384, 512, kernel_size=1)
    projection(features).square().mean().backward()

    assert features.shape == (2, 384, 1, 1)
    assert not features.requires_grad
    assert not encoder.training
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert projection.weight.grad.norm() > 0
    assert parameter_digest(encoder) == before


def test_backbone_applies_pretrained_rgb_normalization_exactly_once():
    encoder = tiny_encoder()
    seen = []
    hook = encoder.patch_embed.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    dino_module().FrozenDINOv3Backbone(encoder)(torch.zeros(1, 3, 32, 32))
    hook.remove()

    expected = torch.tensor([-2.11790393, -2.03571429, -1.80444444])
    torch.testing.assert_close(seen[0][0, :, 0, 0], expected)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 3, 33, 32)])
def test_backbone_rejects_wrong_rgb_or_patch_grid_shapes(shape):
    with pytest.raises(ValueError, match="RGB|multiple"):
        dino_module().FrozenDINOv3Backbone(tiny_encoder())(torch.zeros(*shape))


def test_real_act_update_changes_adapter_and_action_head_but_not_dino():
    module = dino_module()
    config = small_policy_config()
    policy = module.DINOv3ACTPolicy(config, load_backbone_weights=False)
    policy.train()
    backbone = policy.model.backbone
    frozen_before = parameter_digest(backbone)
    projection_before = policy.model.encoder_img_feat_input_proj.weight.detach().clone()
    head_before = policy.model.action_head.weight.detach().clone()
    groups = policy.get_optim_params()
    optimizer_ids = {id(p) for group in groups for p in group["params"]}
    assert not optimizer_ids.intersection(id(p) for p in backbone.parameters())
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    loss, metrics = policy(policy_batch(config))
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.model.encoder_img_feat_input_proj.weight.grad.norm() > 0
    assert policy.model.action_head.weight.grad.norm() > 0
    optimizer.step()

    assert parameter_digest(backbone) == frozen_before
    assert not torch.equal(policy.model.encoder_img_feat_input_proj.weight, projection_before)
    assert not torch.equal(policy.model.action_head.weight, head_before)
    assert not backbone.encoder.training


def test_saved_dino_policy_reloads_offline_and_keeps_its_encoder_identity(tmp_path, monkeypatch):
    from sim.act.policy import build_act_policy

    module = dino_module()
    config = small_policy_config()
    policy = module.DINOv3ACTPolicy(config, load_backbone_weights=False).eval()
    batch = policy_batch(config)
    expected = policy.predict_action_chunk(dict(batch))
    policy.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["type"] == "act_dinov3"
    assert saved["vision_backbone"] == "dinov3_vits16"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def no_download(*args, **kwargs):
        raise AssertionError("saved DINO policy must not download its backbone again")

    monkeypatch.setattr(module, "hf_hub_download", no_download)
    restored, restored_config = build_act_policy(
        load_config(overrides=["act.device=cpu"]), pretrained_path=str(tmp_path),
    )

    assert restored_config.type == "act_dinov3"
    assert not restored.model.backbone.encoder.training
    torch.testing.assert_close(restored.predict_action_chunk(dict(batch)), expected)


def test_missing_backbone_weight_is_rejected_instead_of_using_random_weights(tmp_path):
    from safetensors.torch import load_file, save_file

    module = dino_module()
    policy = module.DINOv3ACTPolicy(small_policy_config(), load_backbone_weights=False)
    policy.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = load_file(str(path))
    weights.pop(next(key for key in weights if key.startswith("model.backbone.encoder.")))
    save_file(weights, str(path))

    with pytest.raises(RuntimeError, match="Missing|missing"):
        module.DINOv3ACTPolicy.from_pretrained(tmp_path, local_files_only=True)


def test_processors_preserve_raw_rgb_and_saved_state_action_scaling(tmp_path):
    from sim.act.policy import build_act_processors

    config = small_policy_config()
    batch = policy_batch(config)
    stats = {}
    for key, feature in config.image_features.items():
        # Deliberately incompatible image statistics: these must be ignored.
        stats[key] = {"mean": torch.full((3, 1, 1), 0.9),
                      "std": torch.full((3, 1, 1), 0.001)}
    stats["observation.state"] = {"mean": torch.ones(36), "std": torch.full((36,), 2.)}
    stats["action"] = {"mean": torch.full((4,), 2.), "std": torch.full((4,), 4.)}
    batch["observation.state"] = torch.zeros(2, 36)
    batch["action"] = torch.zeros(2, 3, 4)
    pre, post = build_act_processors(config, dataset_stats=stats)
    processed = pre(dict(batch))
    for key in config.image_features:
        torch.testing.assert_close(processed[key], batch[key])
    torch.testing.assert_close(processed["observation.state"], torch.full((2, 36), -0.5))
    torch.testing.assert_close(processed["action"], torch.full((2, 3, 4), -0.5))
    torch.testing.assert_close(post(processed["action"]), batch["action"])
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    restored_pre, restored_post = build_act_processors(config, pretrained_path=str(tmp_path))
    restored = restored_pre(dict(batch))
    for key in config.image_features:
        torch.testing.assert_close(restored[key], batch[key])
    torch.testing.assert_close(restored_post(restored["action"]), batch["action"])


def test_pretrained_loader_rejects_a_corrupted_weight_file(tmp_path, monkeypatch):
    module = dino_module()
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not-the-pinned-pretrained-weights")
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: str(broken))
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_dino_encoder(small_policy_config())


def test_pretrained_loader_does_not_require_python311_hash_api(tmp_path, monkeypatch):
    module = dino_module()
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not-the-pinned-pretrained-weights")
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: str(broken))
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_dino_encoder(small_policy_config())


@pytest.mark.parametrize("profile,saved_type", [
    ("configs/five-view-dinov3-frozen-300.yaml", "act"),
    ("configs/five-view-fine-teached-300.yaml", "act_dinov3"),
])
def test_training_rejects_encoder_mismatch_before_loading_data_or_weights(tmp_path, profile, saved_type):
    from sim.act.train import main

    checkpoint = tmp_path / "other-encoder"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"type": saved_type}))
    out = tmp_path / "must-not-start-training"
    with pytest.raises(ValueError, match="training encoder.*mismatch"):
        main([
            "--config", profile, "--resume", str(checkpoint),
            "--dataset", str(tmp_path / "not-read"), "--out", str(out),
            "--steps", "1", "--no-trackio",
        ])
    assert not out.exists()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_backbone_rejects_non_raw_rgb_values(value):
    with pytest.raises(ValueError, match="finite, unnormalized RGB"):
        dino_module().FrozenDINOv3Backbone(tiny_encoder())(
            torch.full((1, 3, 32, 32), value),
        )
