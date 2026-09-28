import json

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


def test_training_episode_plan_starts_with_goal_six_and_keeps_balanced_slots():
    plan = collect_dataset.training_episode_plan(
        seed_base=610000, episodes_per_target=2)

    assert [item.target_count for item in plan[:3]] == [6, 6, 5]
    assert {count: sum(item.target_count == count for item in plan)
            for count in range(1, 7)} == {count: 2 for count in range(1, 7)}
    assert len({item.seed for item in plan}) == len(plan)


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
        "state": np.zeros(36, dtype=np.float32),
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


def test_approved_generator_retries_if_expert_rollout_rejects_seed(
    tmp_path, monkeypatch, capsys,
):
    spec = collect_dataset.EpisodeSpec(
        "train", 1, 410000, "train_n1_layout_000", "independent")
    captures = []

    def capture(_cfg, _writer, _root, _spec, _requested_seed, _attempt):
        captures.append(_spec.seed)
        if len(captures) == 1:
            raise RuntimeError("screened layout changed during capture: test rejection")
        return {"episode_id": "expert_goal_1_0001", "success": True}

    monkeypatch.setattr(
        collect_dataset, "training_episode_plan",
        lambda _seed, _episodes_per_target=20: [spec],
    )
    monkeypatch.setattr(collect_dataset, "_capture_training_episode", capture)
    monkeypatch.setattr(
        collect_dataset, "validate_training_manifest",
        lambda _root, episodes_per_target=20: {"total": 1},
    )
    monkeypatch.setattr(
        collect_dataset, "write_split_manifests", lambda _root: {})

    summary = collect_dataset.generate_approved_training_dataset(
        object(), tmp_path, seed_base=410000, max_attempts=3,
        episodes_per_target=1,
    )

    assert captures == [410000, 410001]
    assert summary["generated"] == 1
    failure = (tmp_path / "generation_failures.jsonl").read_text(
        encoding="utf-8")
    assert '"stage": "rollout"' in failure
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "expert_attempt_started", "expert_attempt_failed",
        "expert_attempt_started", "expert_episode_saved",
    ]
    assert [event["seed"] for event in events
            if event["event"] == "expert_attempt_started"] == [410000, 410001]


def test_approved_generator_resume_skips_previously_failed_seeds(
    tmp_path, monkeypatch, capsys,
):
    spec = collect_dataset.EpisodeSpec(
        "train", 6, 940000, "train_n6_layout_000", "independent")
    (tmp_path / "generation_failures.jsonl").write_text(
        json.dumps({
            "layout_id": spec.layout_id,
            "target_count": 6,
            "seed": 940000,
            "stage": "rollout",
            "reason": "previous target-six rollout failed",
        }) + "\n",
        encoding="utf-8",
    )
    captures = []

    def capture(_cfg, _writer, _root, attempted, _requested_seed, _attempt):
        captures.append(attempted.seed)
        return {"episode_id": "expert_goal_6_0001", "success": True}

    monkeypatch.setattr(
        collect_dataset, "training_episode_plan",
        lambda _seed, _episodes_per_target=20: [spec],
    )
    monkeypatch.setattr(collect_dataset, "_capture_training_episode", capture)
    monkeypatch.setattr(
        collect_dataset, "validate_training_manifest",
        lambda _root, episodes_per_target=20: {"total": 1},
    )
    monkeypatch.setattr(
        collect_dataset, "write_split_manifests", lambda _root: {})

    summary = collect_dataset.generate_approved_training_dataset(
        object(), tmp_path, seed_base=940000, max_attempts=3,
        episodes_per_target=1,
    )

    assert captures == [940001]
    assert summary["generated"] == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == ["expert_attempt_started",
                                                    "expert_episode_saved"]
    assert events[0]["target_count"] == 6
    assert events[0]["attempt"] == 2
    assert events[0]["seed"] == 940001


def test_approved_plan_cli_passes_custom_episodes_per_target(tmp_path, monkeypatch):
    calls = {}

    def generate(_cfg, root, seed_base, max_attempts, episodes_per_target):
        calls.update({
            "root": root,
            "seed_base": seed_base,
            "max_attempts": max_attempts,
            "episodes_per_target": episodes_per_target,
        })
        return {"total": 90}

    monkeypatch.setattr(collect_dataset, "save_config", lambda *_args: None)
    monkeypatch.setattr(
        collect_dataset, "generate_approved_training_dataset", generate)

    result = collect_dataset.main([
        "--approved-plan",
        "--episodes-per-target", "15",
        "--max-attempts-per-episode", "64",
        "--seed-base", "910000",
        "--out", str(tmp_path),
    ])

    assert result == 0
    assert calls == {
        "root": tmp_path,
        "seed_base": 910000,
        "max_attempts": 64,
        "episodes_per_target": 15,
    }
