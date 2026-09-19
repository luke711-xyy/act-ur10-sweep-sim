import json

import numpy as np
import pytest

from sim.act.dataset import ActDataset, ActDatasetWriter
from sim.act.interface import action_deltas_to_absolute
from sim.act.policy import (
    build_act_config,
    build_act_processors,
    validate_act_policy_contract,
)
from sim.act.realtime import ActionChunkScheduler
from sim.config import load_config


def _write_dataset(root):
    image_a = np.zeros((8, 8, 3), dtype=np.uint8)
    image_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    writer = ActDatasetWriter(str(root))
    observations = []
    actions = []
    for i, (image, state_value, action_value) in enumerate(
        ((image_a, 1.0, 0.1), (image_b, 3.0, 0.3))
    ):
        observations.append({
            "overhead": image,
            "wrist": image,
            "state": np.full(42, state_value, dtype=np.float32),
            "environment_state": np.array([6, 1, i], dtype=np.float32),
            "policy_mask": True,
            "phase": "approach" if i == 0 else "sweep",
        })
        actions.append([
            action_value, -action_value, action_value / 2, action_value / 4
        ])
    writer.add_episode(
        "success",
        observations,
        np.asarray(actions, dtype=np.float32),
        True,
        {
            "target_count": 1,
            "total_count": 6,
            "episode_kind": "expert",
            "split": "train",
        },
    )


def test_act_dataset_exposes_lerobot_compatible_feature_stats(tmp_path):
    _write_dataset(tmp_path)

    dataset = ActDataset(str(tmp_path), chunk_size=2)
    sample = dataset[0]
    assert sample["observation.images.overhead"].dtype == np.uint8
    assert sample["observation.images.wrist"].dtype == np.uint8
    stats = dataset.stats

    assert stats["observation.images.overhead"]["mean"].shape == (3, 1, 1)
    np.testing.assert_allclose(
        stats["observation.images.overhead"]["mean"].reshape(3), 0.5
    )
    np.testing.assert_allclose(stats["observation.state"]["mean"], 2.0)
    np.testing.assert_allclose(stats["action"]["mean"], [0.2, -0.2, 0.1, 0.05])
    assert np.all(stats["action"]["std"] > 0)


def test_writer_removes_partial_episode_when_image_write_fails(tmp_path, monkeypatch):
    from PIL import Image

    root = tmp_path / "dataset"
    writer = ActDatasetWriter(str(root))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "overhead": image,
        "wrist": image,
        "inspection": image,
        "state": np.zeros(42, dtype=np.float32),
        "environment_state": np.zeros(3, dtype=np.float32),
    }
    original_save = Image.Image.save

    def fail_on_wrist(self, fp, *args, **kwargs):
        if "wrist_" in str(fp):
            raise OSError("injected image write failure")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_on_wrist)
    with pytest.raises(OSError, match="injected image write failure"):
        writer.add_episode(
            "preview_goal_6_0001", [observation], np.zeros((1, 4), dtype=np.float32),
            True, {"episode_kind": "expert_preview"},
        )

    assert not (root / "preview_goal_6_0001").exists()
    assert not (root / "manifest.jsonl").exists()
    assert not any(path.name.startswith(".preview_goal_6_0001.tmp-")
                   for path in root.iterdir())


def test_official_act_processors_normalize_and_unnormalize_same_action_contract(tmp_path):
    _write_dataset(tmp_path)
    dataset = ActDataset(str(tmp_path), chunk_size=2)
    cfg = load_config(overrides=["act.device=cpu"])
    policy_cfg = build_act_config(cfg)
    preprocessor, postprocessor = build_act_processors(policy_cfg, dataset.stats)

    torch = __import__("torch")
    samples = [dataset[0], dataset[1]]
    batch = {
        key: torch.utils.data.default_collate([sample[key] for sample in samples])
        for key in samples[0]
    }
    for key in ("observation.images.overhead", "observation.images.wrist"):
        batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    processed = preprocessor(batch)
    assert processed["observation.state"].shape == (2, 42)
    assert processed["action"].shape == (2, 2, 4)
    assert float(processed["observation.images.overhead"].abs().max()) < 10.0
    assert float(processed["observation.images.wrist"].abs().max()) < 10.0
    assert not __import__("torch").allclose(processed["action"], batch["action"])

    restored = postprocessor(processed["action"])
    __import__("torch").testing.assert_close(restored, batch["action"], atol=1e-5, rtol=1e-5)


def test_action_delta_interface_converts_model_four_dimensional_output_to_absolute_targets():
    start = np.array([0.40, -0.10, 0.02, 0.20], dtype=np.float32)
    deltas = np.array([
        [0.01, 0.02, -0.005, 0.10],
        [-0.02, 0.00, -0.003, -0.20],
    ], dtype=np.float32)

    absolute = action_deltas_to_absolute(start, deltas)

    assert absolute.shape == (2, 4)
    np.testing.assert_allclose(absolute[0], [0.41, -0.08, 0.015, 0.30])
    np.testing.assert_allclose(absolute[1], [0.39, -0.08, 0.012, 0.10])


def test_training_output_and_inference_output_dimensions_match_config(tmp_path):
    _write_dataset(tmp_path)
    cfg = load_config(overrides=["act.device=cpu"])
    policy_cfg = build_act_config(cfg)
    assert policy_cfg.input_features["observation.state"].shape == (42,)
    assert policy_cfg.output_features["action"].shape == (4,)
    assert policy_cfg.chunk_size == cfg.act.chunk_size
    assert policy_cfg.n_action_steps == cfg.act.execute_steps


def test_policy_contract_rejects_retired_checkpoint_widths():
    cfg = load_config(overrides=["act.device=cpu"])
    policy_cfg = build_act_config(cfg)
    policy_cfg.input_features["observation.state"].shape = (40,)

    with pytest.raises(ValueError, match="schema v4"):
        validate_act_policy_contract(policy_cfg, cfg)


def test_sparse_scheduler_temporally_ensembles_overlapping_chunks():
    cfg = load_config(overrides=[
        "act.device=cpu", "act.temporal_ensemble_coeff=0.01",
    ])
    scheduler = ActionChunkScheduler(cfg)
    scheduler.reset(0.0)
    first = np.zeros((25, 4), dtype=np.float32)
    first[:, 0] = 1.0
    second = np.zeros((25, 4), dtype=np.float32)
    second[:, 0] = 3.0
    assert scheduler.accept(0.0, first, 0.0)
    assert scheduler.accept(0.2, second, 0.2)

    # After the second chunk's own 200 ms budget, both chunks predict the same
    # execution instant.  The result must be a stable weighted blend, not a
    # hard jump to the newest chunk.
    action = scheduler.action_for(0.4)
    assert action is not None
    assert 1.0 < float(action[0]) < 3.0
