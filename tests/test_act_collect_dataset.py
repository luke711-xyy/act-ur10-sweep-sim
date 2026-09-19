import pytest
import numpy as np

import sim.act.collect_dataset as collect_dataset
import sim.act.generate_preview_batch as preview_batch
from sim.act.collect_dataset import parse_episode_plan
from sim.act.dataset import ActDataset, ActDatasetWriter


def test_episode_plan_preserves_target_seed_pairs():
    assert parse_episode_plan("1:2001,2:2000,5:2010") == [
        (1, 2001), (2, 2000), (5, 2010),
    ]


def test_preview_batch_plan_has_one_episode_for_each_exact_target_count():
    assert hasattr(collect_dataset, "preview_batch_plan")
    assert [tuple(item) for item in collect_dataset.preview_batch_plan(7300)] == [
        (1, 7300), (2, 7300), (3, 7300),
        (4, 7300), (5, 7300), (6, 7300),
    ]


def test_preview_batch_cli_delegates_exactly_one_preview_per_target(tmp_path, monkeypatch):
    calls = []

    class FakeWorkbenchState:
        def __init__(self, cfg, dataset_root, preview_root):
            calls.append(("init", dataset_root, preview_root))

        def build_preview(self, seed, target_count, max_attempts=1, **kwargs):
            calls.append(("preview", seed, target_count))
            return {
                "episode_id": f"preview_{target_count}",
                "success": True,
                "total_count": 6,
                "collected": target_count,
                "target_collected": target_count,
                "unexpected_collected": 0,
                "failure_reason": "",
            }

    monkeypatch.setattr(preview_batch, "WorkbenchState", FakeWorkbenchState)
    monkeypatch.setattr(
        preview_batch, "find_shared_layout_seed",
        lambda cfg, requested_seed, max_attempts, **kwargs: (requested_seed, 1),
    )
    assert preview_batch.main([
        "--out", str(tmp_path), "--seed-base", "7300",
        "--set", "sim.real_time=false",
    ]) == 0

    assert [item[1:] for item in calls if item[0] == "preview"] == [
        (7300, 1), (7300, 2), (7300, 3),
        (7300, 4), (7300, 5), (7300, 6),
    ]


@pytest.mark.parametrize("value", ["0:1", "7:1", "1", "1:not-a-seed", ""])
def test_episode_plan_rejects_invalid_entries(value):
    with pytest.raises(ValueError):
        parse_episode_plan(value)


def test_act_training_dataset_excludes_failed_episodes_by_default(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "overhead": image,
        "wrist": image,
        "state": np.zeros(42, dtype=np.float32),
        "environment_state": np.array([6, 1, 0], dtype=np.float32),
        "policy_mask": True,
        "phase": "sweep",
    }
    writer = ActDatasetWriter(str(tmp_path))
    for episode_id, success in (("success", True), ("failure", False)):
        writer.add_episode(
            episode_id, [observation], np.zeros((1, 4), dtype=np.float32),
            success, {
                "target_count": 1,
                "total_count": 6,
                "episode_kind": "expert",
                "split": "train",
            },
        )

    training = ActDataset(str(tmp_path), chunk_size=1)
    audit = ActDataset(str(tmp_path), chunk_size=1, include_failures=True)

    assert [record["episode_id"] for record in training.records] == ["success"]
    assert len(training) == 1
    assert len(audit) == 2
