import numpy as np
import torch

from sim.act.dataset import ActDataset, ActDatasetWriter
from sim.act.interface import ACTObservationBuilder
from sim.act.policy import build_act_config
from sim.config import load_config


def _write_dataset(root, *, target_count=4):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observations = [{
        "overhead": image,
        "wrist": image,
        "state": np.zeros(36, dtype=np.float32),
        "environment_state": np.asarray([6, target_count, 0], dtype=np.float32),
        "phase": "sweep",
        "policy_mask": True,
    }]
    ActDatasetWriter(str(root)).add_episode(
        "expert_goal_4",
        observations,
        np.zeros((1, 4), dtype=np.float32),
        True,
        {"episode_kind": "expert", "split": "train", "target_count": target_count},
    )


def test_goal_token_config_uses_one_categorical_environment_feature_and_four_actions():
    from lerobot.configs.types import FeatureType

    config = build_act_config(load_config(overrides=["act.device=cpu"]),
                              policy_variant="goal_token")

    assert set(config.input_features) == {
        "observation.images.overhead",
        "observation.images.wrist",
        "observation.state",
        "observation.task_id",
    }
    assert config.input_features["observation.task_id"].type is FeatureType.ENV
    assert config.input_features["observation.task_id"].shape == (1,)
    assert "ENV" not in config.normalization_mapping
    assert config.dim_model == 512
    assert config.output_features["action"].shape == (4,)


def test_goal_token_dataset_maps_manifest_target_to_id_without_environment_counts(tmp_path):
    _write_dataset(tmp_path, target_count=4)

    dataset = ActDataset(str(tmp_path), chunk_size=2, policy_variant="goal_token",
                         image_stat_samples=1)
    sample = dataset[0]

    assert int(sample["observation.task_id"]) == 3
    assert sample["observation.task_id"].dtype == np.int64
    assert "observation.environment_state" not in sample
    assert "observation.environment_state" not in dataset.stats


def test_goal_token_train_cli_requires_explicit_variant_and_preview_set():
    from sim.act.train import build_arg_parser

    args = build_arg_parser().parse_args([
        "--policy-variant", "goal_token", "--preview-training",
        "--dataset", "runs/act_dataset_curve_v10",
        "--out", "runs/act_model_goal_token_v1", "--steps", "100000",
    ])
    assert args.policy_variant == "goal_token"
    assert args.preview_training is True
    assert args.steps == 100000


def test_goal_token_inference_batch_contains_only_selected_target_id():
    observation = {
        "observation.state": np.zeros(36, dtype=np.float32),
        "observation.environment_state": np.asarray([6, 4, 2], dtype=np.float32),
        "t": 1.0,
        "contact_latched": True,
    }

    batch = ACTObservationBuilder.torch_batch(
        observation, policy_variant="goal_token", target_count=4
    )

    assert set(batch) == {"observation.state", "observation.task_id"}
    assert batch["observation.task_id"].dtype == torch.long
    assert batch["observation.task_id"].tolist() == [3]


def _tiny_goal_token_config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        n_obs_steps=1,
        chunk_size=3,
        n_action_steps=1,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (36,)),
            "observation.task_id": PolicyFeature(FeatureType.ENV, (1,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        device="cpu",
        pretrained_backbone_weights=None,
        dim_model=512,
        n_heads=8,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )


def test_goal_embedding_gets_action_loss_gradient_and_changes_conditioned_prediction():
    import torch

    from sim.act.policy import create_goal_token_policy

    torch.manual_seed(23)
    policy = create_goal_token_policy(_tiny_goal_token_config())
    policy.train()
    batch = {
        "observation.state": torch.zeros((2, 36)),
        "observation.task_id": torch.tensor([0, 5], dtype=torch.long),
        "action": torch.zeros((2, 3, 4)),
        "action_is_pad": torch.zeros((2, 3), dtype=torch.bool),
    }
    loss, _ = policy(batch)
    loss.backward()

    embedding = policy.model.encoder_env_state_input_proj
    assert isinstance(embedding, torch.nn.Embedding)
    assert tuple(embedding.weight.shape) == (6, 512)
    assert embedding.weight.grad is not None
    assert float(embedding.weight.grad.abs().sum()) > 0.0

    policy.eval()
    base = {"observation.state": torch.zeros((1, 36))}
    with torch.no_grad():
        action_1 = policy.predict_action_chunk({
            **base, "observation.task_id": torch.tensor([0], dtype=torch.long)
        })
        action_6 = policy.predict_action_chunk({
            **base, "observation.task_id": torch.tensor([5], dtype=torch.long)
        })
    assert tuple(action_1.shape) == (1, 3, 4)
    assert not torch.allclose(action_1, action_6)


def test_goal_token_checkpoint_round_trip_preserves_embedding_and_predictions(tmp_path):
    import torch

    from sim.act.policy import (
        create_goal_token_policy,
        load_goal_token_policy,
        write_policy_variant_metadata,
    )

    torch.manual_seed(29)
    policy = create_goal_token_policy(_tiny_goal_token_config()).eval()
    observation = {
        "observation.state": torch.zeros((1, 36)),
        "observation.task_id": torch.tensor([2], dtype=torch.long),
    }
    with torch.no_grad():
        expected = policy.predict_action_chunk(observation)
    policy.save_pretrained(tmp_path)
    write_policy_variant_metadata(tmp_path, "goal_token")

    restored = load_goal_token_policy(tmp_path, device="cpu").eval()
    with torch.no_grad():
        actual = restored.predict_action_chunk(observation)

    torch.testing.assert_close(actual, expected)


def test_goal_token_preprocessor_preserves_task_ids_as_unscaled_integer_tokens():
    from sim.act.policy import build_act_processors

    preprocessor, _ = build_act_processors(_tiny_goal_token_config())
    task_ids = torch.tensor([0, 5], dtype=torch.long)
    batch = preprocessor({
        "observation.state": torch.zeros((2, 36)),
        "observation.task_id": task_ids,
    })

    assert batch["observation.task_id"].dtype == torch.long
    torch.testing.assert_close(batch["observation.task_id"], task_ids)


def test_goal_token_eval_layout_plan_excludes_training_fingerprints_and_pairs_goals(
        tmp_path, monkeypatch):
    import json
    import sim.act.compare as compare

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    records = [{"seed": 10, "layout_fingerprint": "train-a"},
               {"seed": 11, "layout_fingerprint": "train-b"}]
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        compare, "_layout_fingerprint",
        lambda _cfg, seed, _target: f"layout-{seed}",
    )

    plan = compare.make_layout_plan(
        object(), dataset, validation_layouts=2, test_layouts=2, seed_start=10
    )
    all_specs = plan["validation"] + plan["test"]
    assert len(all_specs) == 4
    assert len({spec["seed"] for spec in all_specs}) == 4
    assert all(spec["seed"] not in {10, 11} for spec in all_specs)
    assert all(spec["paired_target_counts"] == [1, 2, 3, 4, 5, 6]
               for spec in all_specs)
