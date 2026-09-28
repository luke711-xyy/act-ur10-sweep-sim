"""Tests for optional Hugging Face Trackio training visibility."""

import json
from pathlib import Path

import pytest

from sim.act import train


def test_training_parser_accepts_remote_trackio_options():
    args = train.build_arg_parser().parse_args([
        "--dataset", "runs/workbench_previews",
        "--out", "runs/act_model",
        "--trackio-project", "act-ur10-sweep",
        "--trackio-space", "Luke711/act-ur10-sweep-tracking",
    ])

    assert args.trackio_project == "act-ur10-sweep"
    assert args.trackio_space == "Luke711/act-ur10-sweep-tracking"
    assert args.trackio_private is None


def test_training_parser_accepts_promoted_preview_set():
    args = train.build_arg_parser().parse_args(["--preview-training"])

    assert args.preview_training is True


def test_training_parser_accepts_checkpoint_and_resume_options():
    args = train.build_arg_parser().parse_args([
        "--batch-size", "8",
        "--amp",
        "--checkpoint-every", "250",
        "--keep-checkpoints", "2",
        "--resume", "runs/act_model/checkpoints/step_000500",
    ])

    assert args.batch_size == 8
    assert args.amp is True
    assert args.checkpoint_every == 250
    assert args.keep_checkpoints == 2
    assert args.resume.endswith("step_000500")


def test_remote_tracking_requires_project_and_space_together():
    with pytest.raises(ValueError, match="project and space"):
        train.tracking_settings("act-ur10-sweep", None, True)

    with pytest.raises(ValueError, match="project and space"):
        train.tracking_settings(None, "Luke711/act-ur10-sweep-tracking", True)


def test_tracking_settings_include_private_hf_space():
    assert train.tracking_settings(
        "act-ur10-sweep", "Luke711/act-ur10-sweep-tracking", True
    ) == {
        "project": "act-ur10-sweep",
        "space_id": "Luke711/act-ur10-sweep-tracking",
        "private": True,
        "static": False,
    }


def test_public_static_tracking_mode_is_explicit():
    args = train.build_arg_parser().parse_args([
        "--trackio-project", "act-ur10-sweep",
        "--trackio-space", "Luke711/act-ur10-sweep-tracking",
        "--trackio-static",
    ])

    assert args.trackio_static is True
    assert train.tracking_settings(
        args.trackio_project, args.trackio_space, args.trackio_private,
        static=args.trackio_static,
    ) == {
        "project": "act-ur10-sweep",
        "space_id": "Luke711/act-ur10-sweep-tracking",
        "private": False,
        "static": True,
    }


def test_hub_init_failure_falls_back_to_local_trackio(capsys):
    class FakeTrackio:
        def __init__(self):
            self.calls = []

        def init(self, **kwargs):
            self.calls.append(kwargs)
            if "space_id" in kwargs:
                raise RuntimeError("Hub returned 402")

    tracker = FakeTrackio()
    settings = train.tracking_settings(
        "act-ur10-sweep", "Luke711/act-ur10-sweep-tracking", False
    )
    result, active_tracking = train.initialize_tracker(
        tracker, settings, {"max_steps": 100000}
    )

    assert result is tracker
    assert active_tracking is None
    assert tracker.calls[0]["private"] is False
    assert "space_id" in tracker.calls[0]
    assert "space_id" not in tracker.calls[1]
    assert tracker.calls[1]["config"] == {"max_steps": 100000}
    output = capsys.readouterr().out
    assert '"event": "trackio_remote_init_failed"' in output
    assert '"event": "trackio_local_fallback"' in output


def test_tracker_failure_does_not_abort_training_setup(capsys):
    class BrokenTrackio:
        def init(self, **kwargs):
            raise RuntimeError("local database unavailable")

    settings = train.tracking_settings(
        "act-ur10-sweep", "Luke711/act-ur10-sweep-tracking", False
    )
    result, active_tracking = train.initialize_tracker(
        BrokenTrackio(), settings, {"max_steps": 100000}
    )
    assert result is None
    assert active_tracking is None
    assert '"event": "trackio_local_init_failed"' in capsys.readouterr().out


def test_static_tracking_sync_targets_the_public_static_space():
    class FakeTrackio:
        def __init__(self):
            self.calls = []

        def sync(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["space_id"]

    tracker = FakeTrackio()
    settings = train.tracking_settings(
        "act-ur10-sweep-full-episode-v1",
        "Luke711/act-ur10-sweep-tracking",
        static=True,
    )

    assert train.sync_static_tracking(tracker, settings)
    assert tracker.calls == [{
        "project": "act-ur10-sweep-full-episode-v1",
        "space_id": "Luke711/act-ur10-sweep-tracking",
        "sdk": "static",
    }]


def test_resumable_checkpoint_persists_cumulative_training_time(tmp_path):
    torch = pytest.importorskip("torch")

    class SavedComponent:
        def save_pretrained(self, root, **kwargs):
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)
            (root / kwargs.get("config_filename", "component.json")).write_text(
                "{}", encoding="utf-8"
            )

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    checkpoint = train.save_checkpoint(
        SavedComponent(), SavedComponent(), SavedComponent(), optimizer,
        tmp_path, step=2500, dataset_root="runs/act_dataset", device="mps",
        keep_checkpoints=3, elapsed_seconds=1234.5,
    )

    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    assert state["step"] == 2500
    assert state["schema_version"] == 5
    assert state["state_dim"] == 36
    assert state["elapsed_seconds"] == pytest.approx(1234.5)


def test_small_sample_training_gate_requires_approved_same_layout(tmp_path):
    records = [{
        "episode_id": f"preview_goal_{target}_0001",
        "schema_version": 5,
        "action_dim": 4,
        "state_dim": 36,
        "frame_count": 300,
        "episode_kind": "expert",
        "split": "train",
        "success": True,
        "target_count": target,
        "max_frames": 800,
        "layout_id": f"train_layout_{target:03d}",
        "layout_kind": "independent",
        "layout_fingerprint": f"fp_{target:03d}",
        "seed": 4100 + target,
    } for target in range(1, 7)]
    records[-1]["frame_count"] = 529
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = train.validate_small_sample_gate_manifest(tmp_path)

    assert summary["total"] == 6
    records[-1]["seed"] = records[0]["seed"]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="independent unique layouts"):
        train.validate_small_sample_gate_manifest(tmp_path)
