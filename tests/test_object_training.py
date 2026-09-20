import numpy as np
import torch


def test_objectact_preprocessor_normalizes_declared_modalities_separately(tmp_path):
    from sim.act.object_training import ObjectACTPreprocessor

    stats = {
        "observation.images.overhead": {"mean": np.zeros((3, 1, 1)), "std": np.ones((3, 1, 1))},
        "observation.images.wrist": {"mean": np.zeros((3, 1, 1)), "std": np.ones((3, 1, 1))},
        "observation.robot_state": {"mean": np.zeros(36), "std": np.ones(36)},
        "observation.task_state": {"mean": np.zeros(6), "std": np.ones(6)},
        "observation.object_tokens": {"mean": np.zeros((6, 29)), "std": np.ones((6, 29))},
        "observation.bev": {"mean": np.zeros((6, 1, 1)), "std": np.ones((6, 1, 1))},
        "action": {"mean": np.zeros(4), "std": np.ones(4)},
    }
    preprocessor = ObjectACTPreprocessor(stats, device="cpu")
    batch = {
        "observation.images.overhead": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8),
        "observation.images.wrist": torch.zeros((1, 3, 4, 4), dtype=torch.uint8),
        "observation.robot_state": torch.ones((1, 36)),
        "observation.task_state": torch.ones((1, 6)),
        "observation.object_tokens": torch.ones((1, 6, 29)),
        "observation.object_valid": torch.ones((1, 6), dtype=torch.bool),
        "observation.instance_bev": torch.zeros((1, 6, 128, 160), dtype=torch.bool),
        "observation.bev": torch.ones((1, 6, 128, 160)),
        "action": torch.ones((1, 25, 4)),
        "action_is_pad": torch.zeros((1, 25), dtype=torch.bool),
    }
    processed = preprocessor(batch)
    assert processed["observation.images.overhead"].dtype == torch.float32
    assert torch.allclose(processed["observation.images.overhead"], torch.ones((1, 3, 4, 4)))
    assert torch.allclose(processed["observation.robot_state"], torch.ones((1, 36)))
    assert processed["observation.object_valid"].dtype == torch.bool
    assert processed["observation.instance_bev"].dtype == torch.bool

    path = tmp_path / "stats.json"
    preprocessor.save(path)
    restored = ObjectACTPreprocessor.load(path, device="cpu")
    assert restored.stats["observation.object_tokens"]["std"].shape == (6, 29)


def test_objectact_dataset_statistics_keep_slot_and_channel_axes():
    from sim.act.object_training import compute_object_dataset_stats

    class TinyDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "observation.images.overhead": np.zeros((3, 4, 4), dtype=np.uint8),
                "observation.images.wrist": np.zeros((3, 4, 4), dtype=np.uint8),
                "observation.robot_state": np.zeros(36, dtype=np.float32),
                "observation.task_state": np.zeros(6, dtype=np.float32),
                "observation.object_tokens": np.zeros((6, 29), dtype=np.float32),
                "observation.bev": np.zeros((6, 128, 160), dtype=np.float32),
                "action": np.zeros((25, 4), dtype=np.float32),
            }

    stats = compute_object_dataset_stats(TinyDataset())
    assert np.asarray(stats["observation.object_tokens"]["mean"]).shape == (6, 29)
    assert np.asarray(stats["observation.bev"]["mean"]).shape == (6, 1, 1)
    assert np.asarray(stats["action"]["mean"]).shape == (4,)


def test_objectact_checkpoint_has_schema5_contract_and_prunes_only_non_milestones(tmp_path):
    from sim.act.object_training import (
        ObjectACTPreprocessor,
        load_objectact_checkpoint,
        prune_objectact_checkpoints,
        save_objectact_checkpoint,
    )

    model = torch.nn.Linear(2, 2)
    model.selection_step = 123
    optimizer = torch.optim.AdamW(model.parameters())
    preprocessor = ObjectACTPreprocessor({
        key: {"mean": np.zeros(1), "std": np.ones(1)}
        for key in (
            "observation.images.overhead", "observation.images.wrist",
            "observation.robot_state", "observation.task_state",
            "observation.object_tokens", "observation.bev", "action",
        )
    })
    checkpoint = save_objectact_checkpoint(
        model, optimizer, preprocessor, tmp_path, step=2500,
        config_payload={"name": "test"}, dataset_root="dataset",
    )
    payload = load_objectact_checkpoint(checkpoint)
    assert payload["schema_version"] == 5
    assert payload["action_dim"] == 4
    assert payload["selection_step"] == 123
    for step in (5000, 7500, 10000, 12500):
        save_objectact_checkpoint(
            model, optimizer, preprocessor, tmp_path, step=step,
            config_payload={"name": "test"}, dataset_root="dataset",
        )
    prune_objectact_checkpoints(tmp_path, keep_latest=1, milestones={2500, 10000})
    assert (tmp_path / "checkpoints" / "step_002500").exists()
    assert (tmp_path / "checkpoints" / "step_010000").exists()
    assert not (tmp_path / "checkpoints" / "step_005000").exists()


def test_objectact_training_parser_defaults_to_formal_80k_schedule():
    from sim.act.object_training import build_objectact_train_parser

    args = build_objectact_train_parser().parse_args([])
    assert args.steps == 80000
    assert args.checkpoint_every == 2500
    assert args.keep_checkpoints == 3
    assert args.expected_per_target == 20
