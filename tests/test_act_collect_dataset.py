import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import numpy as np

import sim.act.collect_dataset as collect_dataset
import sim.act.generate_preview_batch as preview_batch
from sim.act.collect_dataset import parse_episode_plan
from sim.act.dataset import ActDataset, ActDatasetWriter
from sim.config import Config


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


def test_parallel_worker_config_payload_is_plain_serializable_data():
    cfg = Config({"planner": {"expert_validation_workers": 4},
                  "act": {"camera_views": ["wrist"]}})

    payload = collect_dataset._worker_config_payload(cfg)

    assert type(payload) is dict
    assert type(payload["planner"]) is dict
    assert payload["planner"]["expert_validation_workers"] == 4
    assert payload["act"]["camera_views"] == ["wrist"]


def test_approved_generator_parallelizes_independent_episode_rollouts(
    tmp_path, monkeypatch, capsys,
):
    plan = [
        collect_dataset.EpisodeSpec(
            "train", 4, 710000, "train_n4_layout_000", "independent"),
        collect_dataset.EpisodeSpec(
            "train", 4, 710256, "train_n4_layout_001", "independent"),
    ]
    active = 0
    peak_active = 0
    captured_ids = []
    state_lock = threading.Lock()

    def capture(_cfg, _writer, _root, spec, _requested_seed, _attempt,
                episode_id=None, **_kwargs):
        nonlocal active, peak_active
        with state_lock:
            active += 1
            peak_active = max(peak_active, active)
            captured_ids.append((spec.layout_id, episode_id))
        time.sleep(0.08)
        with state_lock:
            active -= 1
        return {"episode_id": episode_id, "success": True}

    monkeypatch.setattr(
        collect_dataset, "training_episode_plan",
        lambda _seed, _episodes_per_target=20: plan,
    )
    monkeypatch.setattr(collect_dataset, "_capture_training_episode", capture)
    class ThreadPoolAdapter(ThreadPoolExecutor):
        def __init__(self, max_workers, mp_context=None):
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(collect_dataset, "ProcessPoolExecutor", ThreadPoolAdapter)
    reserved_counts = []
    next_serial = collect_dataset.next_demo_serial

    def record_serial_scan(roots, target_count):
        reserved_counts.append(target_count)
        return next_serial(roots, target_count)

    monkeypatch.setattr(collect_dataset, "next_demo_serial", record_serial_scan)
    monkeypatch.setattr(
        collect_dataset, "validate_training_manifest",
        lambda _root, episodes_per_target=20, **_kwargs: {"total": 2},
    )
    monkeypatch.setattr(collect_dataset, "write_split_manifests", lambda _root: {})

    summary = collect_dataset.generate_approved_training_dataset(
        SimpleNamespace(act={"camera_views": ["wrist"]}), tmp_path,
        seed_base=710000, max_attempts=2, episodes_per_target=1,
        workers=2,
    )

    assert peak_active == 2
    assert reserved_counts == [4]
    assert sorted(episode_id for _layout, episode_id in captured_ids) == [
        "preview_goal_4_0001", "preview_goal_4_0002",
    ]
    assert summary["generated"] == 2
    events = [json.loads(line)["event"]
              for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert events.count("expert_attempt_started") == 2
    assert events.count("expert_episode_saved") == 2


def test_dataset_writer_does_not_lose_manifest_updates_from_two_processes(tmp_path):
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    worker = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        import numpy as np
        from sim.act.dataset import ActDatasetWriter

        root = Path(sys.argv[1])
        episode_id = sys.argv[2]
        original_read_text = Path.read_text

        def delayed_read_text(path, *args, **kwargs):
            value = original_read_text(path, *args, **kwargs)
            if path.name == "manifest.jsonl":
                time.sleep(0.5)
            return value

        Path.read_text = delayed_read_text
        (root / ("ready-" + episode_id)).touch()
        deadline = time.monotonic() + 30
        while len(list(root.glob("ready-*"))) < 2:
            if time.monotonic() > deadline:
                raise TimeoutError("both writer processes did not become ready")
            time.sleep(0.01)

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = {
            "images": {"wrist": image},
            "state": np.zeros(36, dtype=np.float32),
            "environment_state": np.array([6, 1, 0], dtype=np.float32),
        }
        ActDatasetWriter(str(root)).add_episode(
            episode_id, [observation], np.zeros((1, 4), dtype=np.float32),
            True, {"target_count": 1, "episode_kind": "expert", "split": "train"},
        )
    """)
    project_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (project_root, env.get("PYTHONPATH", "")) if part
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(tmp_path), episode_id],
            cwd=project_root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        for episode_id in ("preview_goal_1_0001", "preview_goal_1_0002")
    ]
    outputs = [process.communicate(timeout=60) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs

    records = [
        json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    assert {record["episode_id"] for record in records} == {
        "preview_goal_1_0001", "preview_goal_1_0002",
    }


def test_approved_cli_forwards_episode_worker_count(tmp_path, monkeypatch):
    calls = []
    cfg = SimpleNamespace(act=SimpleNamespace(dataset_dir=str(tmp_path)))
    monkeypatch.setattr(collect_dataset, "load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(collect_dataset, "save_config", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        collect_dataset, "generate_approved_training_dataset",
        lambda *_a, **kwargs: calls.append(kwargs) or {"total": 6},
    )

    assert collect_dataset.main([
        "--approved-plan", "--episodes-per-target", "1",
        "--workers", "2", "--out", str(tmp_path),
    ]) == 0
    assert calls[0]["workers"] == 2


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


def test_capture_training_episode_delegates_candidate_screening_to_expert_rollout(
    tmp_path, monkeypatch,
):
    calls = []
    writes = []

    def result(success=True):
        return SimpleNamespace(
            success=success,
            failure_reason="contact lost for 100 ms" if not success else "",
            collected=1,
            target_count=1,
            planner_status="feasible",
            planner_strategy="coverage_astar",
            planner_turn_count=2,
            planner_attempts=1,
            sampled_frames=2,
            observations=[{"frame": 0}, {"frame": 1}],
            actions=np.zeros((2, 4), dtype=np.float32),
            trace=[],
        )

    def rollout(_cfg, *, seed, collect_observations, **_kwargs):
        calls.append((seed, collect_observations))
        return result()

    class Writer:
        def add_episode(self, *args, **kwargs):
            writes.append((args, kwargs))

    spec = collect_dataset.EpisodeSpec(
        "train", 1, 5123, "train_n1_layout_000", "independent")
    monkeypatch.setattr(collect_dataset, "_expert_config", lambda *_args: "cfg")
    monkeypatch.setattr(collect_dataset, "run_expert_episode", rollout)
    monkeypatch.setattr(collect_dataset, "next_demo_name", lambda *_args: "expert_goal_1_0001")
    monkeypatch.setattr(collect_dataset, "_expert_metadata", lambda *_args: {"success": True})

    episode = collect_dataset._capture_training_episode(
        object(), Writer(), tmp_path, spec, requested_seed=5123,
        generation_attempt=1,
    )

    assert calls == [(5123, True)]
    assert episode["episode_id"] == "expert_goal_1_0001"
    assert len(writes) == 1


def test_capture_training_episode_rejects_failed_expert_rollout_without_writing(
    tmp_path, monkeypatch,
):
    calls = []
    writes = []

    def rollout(_cfg, *, seed, collect_observations, **_kwargs):
        calls.append((seed, collect_observations))
        return SimpleNamespace(
            success=False,
            failure_reason="contact lost for 100 ms",
            collected=0,
            target_count=1,
            planner_status="feasible",
            planner_strategy="coverage_astar",
            planner_turn_count=2,
            planner_attempts=1,
            sampled_frames=30,
        )

    class Writer:
        def add_episode(self, *args, **kwargs):
            writes.append((args, kwargs))

    monkeypatch.setattr(collect_dataset, "_expert_config", lambda *_args: "cfg")
    monkeypatch.setattr(collect_dataset, "run_expert_episode", rollout)
    spec = collect_dataset.EpisodeSpec(
        "train", 1, 5123, "train_n1_layout_000", "independent")

    with pytest.raises(RuntimeError, match="contact lost for 100 ms"):
        collect_dataset._capture_training_episode(
            object(), Writer(), tmp_path, spec, requested_seed=5123,
            generation_attempt=1,
        )
    assert calls == [(5123, True)]
    assert writes == []


def test_approved_plan_cli_passes_custom_episodes_per_target(tmp_path, monkeypatch):
    calls = {}

    def generate(_cfg, root, seed_base, max_attempts, episodes_per_target,
                 image_format, workers):
        calls.update({
            "root": root,
            "seed_base": seed_base,
            "max_attempts": max_attempts,
            "episodes_per_target": episodes_per_target,
            "image_format": image_format,
            "workers": workers,
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
        "--image-format", "webp-lossless",
    ])

    assert result == 0
    assert calls == {
        "root": tmp_path,
        "seed_base": 910000,
        "max_attempts": 64,
        "episodes_per_target": 15,
        "image_format": "webp-lossless",
        "workers": 1,
    }
