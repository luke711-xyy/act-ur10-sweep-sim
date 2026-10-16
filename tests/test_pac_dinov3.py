"""Offline contracts for the frozen DINOv3 ACT base policy."""

from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from sim.act.policy import build_act_config, build_act_processors
from sim.config import load_config


PROJECT = Path(__file__).resolve().parents[1]
CAMERA_ORDER = (
    "corner_xminus_yminus",
    "corner_xminus_yplus",
    "corner_xplus_yminus",
    "corner_xplus_yplus",
    "wrist",
)


def _dino_config():
    try:
        from sim.act.configuration_dinov3 import DINOv3ACTConfig
    except ModuleNotFoundError as error:
        if error.name == "sim.act.configuration_dinov3":
            pytest.fail("the serialized DINOv3 ACT config has not been implemented")
        raise
    from lerobot.configs import NormalizationMode
    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = load_config(str(PROJECT / "configs/five-view-fine-teached-300.yaml"),
                      overrides=["act.device=cpu", "act.vision_encoder=dinov3"])
    cfg.act.camera_views = list(CAMERA_ORDER)
    cfg.act.ordinary_target_count_only = True
    base = build_act_config(cfg)
    values = {field.name: getattr(base, field.name)
              for field in fields(ACTConfig) if field.init}
    values.update(
        vision_backbone="dinov3_vits16",
        pretrained_backbone_weights=None,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    return DINOv3ACTConfig(**values)


def _dino_module():
    try:
        return importlib.import_module("sim.act.dinov3")
    except ModuleNotFoundError as error:
        if error.name == "sim.act.dinov3":
            pytest.fail("the frozen DINOv3 ACT model adapter has not been implemented")
        raise


def _tiny_encoder():
    import timm

    # Offline architecture exercise only; the full checkpoint is verified at
    # the later readiness gate, without downloading or substituting weights.
    return timm.create_model("vit_small_patch16_dinov3", pretrained=False,
                             depth=1, img_size=32, num_classes=0)


def _parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _install_test_policy_base(monkeypatch):
    """Use a tiny BasePolicy shim only when the optional SB3 extra is absent."""
    try:
        import stable_baselines3.common.policies  # noqa: F401
        return
    except ImportError:
        pass

    package = types.ModuleType("stable_baselines3")
    package.__path__ = []
    common = types.ModuleType("stable_baselines3.common")
    common.__path__ = []
    policies = types.ModuleType("stable_baselines3.common.policies")

    class BasePolicy(nn.Module):
        def __init__(self, observation_space, action_space, normalize_images=False):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
            self.normalize_images = normalize_images

        @property
        def device(self):
            parameter = next(self.parameters(), None)
            return parameter.device if parameter is not None else torch.device("cpu")

    policies.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "stable_baselines3", package)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.policies", policies)


class _TinyACTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(1, 4)

    def forward(self, batch):
        state = batch["observation.state"]
        value = self.output(state[:, :1])
        chunk = value[:, None, :].expand(-1, 25, -1)
        return chunk, None


def _processor_stats(config):
    stats = {}
    for key in config.image_features:
        stats[key] = {
            "mean": torch.full((3, 1, 1), 0.9),
            "std": torch.full((3, 1, 1), 0.001),
        }
    stats["observation.state"] = {
        "mean": torch.ones(36), "std": torch.full((36,), 2.0),
    }
    stats["action"] = {
        "mean": torch.full((4,), 2.0), "std": torch.full((4,), 4.0),
    }
    return stats


def _fake_local_checkpoint(root, monkeypatch):
    from safetensors.torch import save_file

    config = _dino_config()
    config.save_pretrained(root)
    save_file({"sentinel": torch.zeros(1)}, str(root / "model.safetensors"))
    pre, post = build_act_processors(config, dataset_stats=_processor_stats(config))
    pre.save_pretrained(root, config_filename="policy_preprocessor.json")
    post.save_pretrained(root, config_filename="policy_postprocessor.json")

    dino = _dino_module()
    loads = []

    def from_pretrained(cls, path, *, config, **kwargs):
        loads.append((str(path), kwargs.copy()))
        return types.SimpleNamespace(config=config, model=_TinyACTModel(config))

    monkeypatch.setattr(dino.DINOv3ACTPolicy, "from_pretrained",
                        classmethod(from_pretrained))
    _install_test_policy_base(monkeypatch)
    pac_policy = importlib.import_module("pac_act.policy")
    return config, pac_policy, loads


def _observation(config):
    observation = {
        key: torch.full(tuple(feature.shape), 127, dtype=torch.uint8)
        for key, feature in config.image_features.items()
    }
    observation["observation.state"] = torch.zeros(36, dtype=torch.float32)
    observation["observation.environment_state"] = torch.ones(1, dtype=torch.float32)
    return observation


def test_dinov3_config_keeps_pinned_identity_and_five_view_contract():
    config = _dino_config()

    assert config.type == "act_dinov3"
    assert config.vision_backbone == "dinov3_vits16"
    assert config.pretrained_backbone_weights is None
    assert config.dino_model_id == "timm/vit_small_patch16_dinov3.lvd1689m"
    assert config.dino_revision == "3bf4720a82ec2066db88137180ff1f83a675cef0"
    assert config.dino_weight_sha256 == "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
    assert tuple(key.removeprefix("observation.images.")
                 for key in config.image_features) == CAMERA_ORDER
    assert all(tuple(feature.shape) == (3, 320, 320)
               for feature in config.image_features.values())
    assert tuple(config.input_features["observation.state"].shape) == (36,)
    assert tuple(config.input_features["observation.environment_state"].shape) == (1,)
    assert tuple(config.output_features["action"].shape) == (4,)
    assert (config.chunk_size, config.n_action_steps, config.n_obs_steps) == (25, 5, 1)
    assert config.normalization_mapping["VISUAL"].value == "IDENTITY"


def test_frozen_backbone_keeps_dino_eval_and_rgb_feature_map():
    dino = _dino_module()
    encoder = _tiny_encoder()
    seen = []
    hook = encoder.patch_embed.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone()))
    backbone = dino.FrozenDINOv3Backbone(encoder)
    before = _parameter_digest(encoder)

    backbone.train()
    features = backbone(torch.zeros(2, 3, 32, 32))["feature_map"]
    projection = nn.Conv2d(384, 32, kernel_size=1)
    projection(features).square().mean().backward()
    hook.remove()

    assert features.shape == (2, 384, 1, 1)
    assert not features.requires_grad
    assert not encoder.training
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in encoder.parameters())
    assert projection.weight.grad is not None and projection.weight.grad.norm() > 0
    assert _parameter_digest(encoder) == before
    expected = torch.tensor([-2.11790393, -2.03571429, -1.80444444])
    torch.testing.assert_close(seen[0][0, :, 0, 0], expected)


def test_chunk_policy_loads_dinov3_locally_and_rejects_wrong_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "valid"
    checkpoint.mkdir()
    config, pac_policy, loads = _fake_local_checkpoint(checkpoint, monkeypatch)

    policy = pac_policy.ChunkPolicy(checkpoint, device="cpu", build_critic=False)
    assert loads == [(str(checkpoint), {"local_files_only": True})]
    assert tuple(policy.image_keys) == tuple(config.image_features)
    assert policy.means(_observation(config)).shape == (1, 5, 3)
    assert torch.isfinite(policy.means(_observation(config))).all()

    source = json.loads((checkpoint / "config.json").read_text())
    for name, change in (
        ("wrong-type", {"type": "act"}),
        ("wrong-backbone", {"vision_backbone": "resnet18"}),
        ("wrong-chunk", {"chunk_size": 24}),
        ("wrong-execute", {"n_action_steps": 4}),
    ):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({**source, **change}))
        with pytest.raises(ValueError, match="DINOv3|contract|chunk|action"):
            pac_policy.ChunkPolicy(path, device="cpu", build_critic=False)


def test_dinov3_processor_preserves_camera_order_and_identity_visual_stats(tmp_path, monkeypatch):
    checkpoint = tmp_path / "processor-checkpoint"
    checkpoint.mkdir()
    config, pac_policy, _loads = _fake_local_checkpoint(checkpoint, monkeypatch)
    policy = pac_policy.ChunkPolicy(checkpoint, device="cpu", build_critic=False)
    batch = policy.prepare(_observation(config))

    assert tuple(policy.image_keys) == tuple(config.image_features)
    assert len(batch["observation.images"]) == 5
    for index, key in enumerate(policy.image_keys):
        assert batch["observation.images"][index].data_ptr() == batch[key].data_ptr()
        torch.testing.assert_close(batch[key], torch.full((1, 3, 320, 320), 127 / 255))
    torch.testing.assert_close(batch["observation.state"], torch.full((1, 36), -0.5))
    torch.testing.assert_close(policy.action_mean, torch.full((4,), 2.0))
    torch.testing.assert_close(policy.action_scale, torch.full((4,), 4.0))
    physical = policy.physical_actions(torch.zeros(1, 5, 3))
    torch.testing.assert_close(physical[..., 0], torch.full((1, 5), 2.0))
    torch.testing.assert_close(physical[..., 1], torch.full((1, 5), 2.0))
    torch.testing.assert_close(physical[..., 2], torch.zeros((1, 5)))
    torch.testing.assert_close(physical[..., 3], torch.full((1, 5), 2.0))
