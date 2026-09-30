import json

import numpy as np
import torch

from sim.act.dataset import ActDataset, ActDatasetWriter
from sim.act.interface import ACTObservationBuilder
from sim.act.policy import build_act_config
from sim.config import load_config


def _write_episode(root, *, target_count=4):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "overhead": image,
        "wrist": image,
        "state": np.zeros(36, dtype=np.float32),
        "environment_state": np.asarray([6, target_count, 2], dtype=np.float32),
        "phase": "sweep",
        "policy_mask": True,
    }
    ActDatasetWriter(str(root)).add_episode(
        "expert_goal_4",
        [observation],
        np.zeros((1, 4), dtype=np.float32),
        True,
        {"episode_kind": "expert", "split": "train", "target_count": target_count},
    )


def test_ordinary_target_count_dataset_contains_only_the_selected_scalar(tmp_path):
    _write_episode(tmp_path, target_count=4)

    dataset = ActDataset(
        str(tmp_path), chunk_size=2, image_stat_samples=1,
        ordinary_target_count_only=True,
    )
    sample = dataset[0]

    np.testing.assert_array_equal(
        sample["observation.environment_state"], np.asarray([4.0], dtype=np.float32)
    )
    assert dataset.stats["observation.environment_state"]["mean"].shape == (1,)


def test_ordinary_target_count_policy_config_is_scalar_and_legacy_stays_three_wide():
    from sim.act.policy import ordinary_target_count_only_from_config

    legacy_config = build_act_config(load_config(overrides=["act.device=cpu"]))
    target_count_config = build_act_config(load_config(overrides=[
        "act.device=cpu", "act.ordinary_target_count_only=true",
    ]))

    assert legacy_config.input_features["observation.environment_state"].shape == (3,)
    assert target_count_config.input_features["observation.environment_state"].shape == (1,)
    assert not ordinary_target_count_only_from_config(legacy_config)
    assert ordinary_target_count_only_from_config(target_count_config)
    assert target_count_config.output_features["action"].shape == (4,)


def test_loaded_ordinary_checkpoint_keeps_its_saved_goal_input_width(tmp_path):
    from sim.act.policy import load_ordinary_act_config

    target_count_checkpoint_cfg = build_act_config(load_config(overrides=[
        "act.device=cpu", "act.ordinary_target_count_only=true",
    ]))
    target_count_checkpoint_cfg.save_pretrained(tmp_path)
    runtime_cfg = load_config(overrides=["act.device=cpu"])

    loaded = load_ordinary_act_config(tmp_path, runtime_cfg)

    assert loaded.input_features["observation.environment_state"].shape == (1,)


def test_ordinary_target_count_inference_batch_does_not_expose_other_counts():
    observation = {
        "observation.state": np.zeros(36, dtype=np.float32),
        "observation.environment_state": np.asarray([6, 4, 2], dtype=np.float32),
        "t": 1.0,
        "contact_latched": True,
    }

    target_only = ACTObservationBuilder.torch_batch(
        observation, target_count=4, ordinary_target_count_only=True
    )
    legacy = ACTObservationBuilder.torch_batch(observation)

    assert target_only["observation.environment_state"].shape == (1,)
    torch.testing.assert_close(
        target_only["observation.environment_state"], torch.tensor([4.0])
    )
    torch.testing.assert_close(
        legacy["observation.environment_state"], torch.tensor([6.0, 4.0, 2.0])
    )


def test_training_manifest_validator_uses_run_configured_episodes_per_target(tmp_path):
    from sim.act.train import validate_configured_training_manifest

    records = []
    for target_count in range(1, 7):
        for index in range(50):
            layout_index = len(records)
            records.append({
                "episode_id": f"expert_{target_count}_{index:03d}",
                "schema_version": 5,
                "action_dim": 4,
                "state_dim": 36,
                "episode_kind": "expert",
                "split": "train",
                "success": True,
                "target_count": target_count,
                "frame_count": 100,
                "max_frames": 500,
                "layout_kind": "independent",
                "layout_id": f"train_layout_{layout_index:03d}",
                "seed": 900000 + layout_index,
                "layout_fingerprint": f"fingerprint_{layout_index:03d}",
            })
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    cfg = load_config(overrides=["act.training_episodes_per_target=50"])

    summary = validate_configured_training_manifest(tmp_path, cfg)

    assert summary["total"] == 300
    assert summary["per_target"] == {str(target): 50 for target in range(1, 7)}
