"""Bounded replay memory and one-inference/one-export batch scheduling."""

import gc
import json
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sim.act.dataset import ActDatasetWriter


def test_stream_writer_releases_images_before_requesting_next_frame(tmp_path):
    references = []

    def frames():
        for index in range(6):
            gc.collect()
            assert all(reference() is None for reference in references)
            image = np.full((8, 8, 3), index * 20, dtype=np.uint8)
            references.append(weakref.ref(image))
            observation = {"images": {"wrist": image}, "wrist": image,
                           "state": np.full(36, index), "environment_state": [2],
                           "t": index / 25, "normal_force": index / 10}
            yield observation, np.full(4, index / 100)
            del image, observation

    writer = ActDatasetWriter(str(tmp_path), image_format="webp-lossless")
    writer.add_episode_stream("inference_stream", frames(), False,
                              {"episode_kind": "inference", "target_count": 2},
                              causal_trace=[{"t": 0.01, "command": [0, 0, 0, 0]}])
    record = json.loads(writer.manifest.read_text())
    assert record["frame_count"] == 6
    assert record["causal_trace_rows"] == 1
    with np.load(tmp_path / record["arrays"]) as arrays:
        np.testing.assert_array_equal(arrays["state"], np.repeat(np.arange(6)[:, None], 36, axis=1))
        np.testing.assert_allclose(arrays["action"][:, 0], np.arange(6) / 100)
        np.testing.assert_allclose(arrays["normal_force"], np.arange(6) / 10)
    for index, relative in enumerate(record["images"]["wrist"]):
        with Image.open(tmp_path / relative) as image:
            np.testing.assert_array_equal(np.asarray(image), np.full((8, 8, 3), index * 20))


@pytest.mark.parametrize("bad", ["empty", "action", "state", "views", "exception"])
def test_stream_writer_failure_never_publishes_partial_episode(tmp_path, bad):
    def frames():
        if bad == "empty":
            return
        for index in range(2):
            if bad == "exception" and index == 1:
                raise OSError("render failed")
            yield {"images": {"other" if bad == "views" and index else "wrist":
                               np.zeros((8, 8, 3), dtype=np.uint8)},
                   "state": np.zeros(35 if bad == "state" else 36),
                   "environment_state": [2]}, np.zeros(3 if bad == "action" else 4)
    with pytest.raises((ValueError, OSError)):
        ActDatasetWriter(str(tmp_path)).add_episode_stream(
            "inference_bad", frames(), False, {"episode_kind": "inference"})
    assert not (tmp_path / "manifest.jsonl").exists()
    assert not (tmp_path / "inference_bad").exists()
    assert not list(tmp_path.glob(".inference_bad.tmp-*"))


def _batch_fixture(root, monkeypatch, *, fail_export=False):
    from sim.act import batch_evaluate as batch
    layouts = [{"layout_index": i, "seed": 2110000 + i,
                "layout_fingerprint": f"layout-{i}"} for i in range(120)]
    plan = batch.make_episode_plan(layouts, [{"label": "five_view_100k", "path": "/checkpoint"}])
    (root / "episode_plan.json").write_text(json.dumps(plan))
    existing = root / "existing.mp4"
    existing.write_bytes(b"verified")
    previous = [{**spec, "video_path": str(existing), "success": False,
                 "peak_collected": 0, "final_collected": 0} for spec in plan[:117]]
    (root / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in previous))
    (root / "evaluation_config.yaml").write_text("{}")
    (root / "logs").mkdir()
    metadata = {"project": str(root), "models": [], "source_fingerprints": {},
                "config_sha256": batch._sha256(root / "evaluation_config.yaml"),
                "episode_plan_sha256": batch._sha256(root / "episode_plan.json"),
                "layout_seed_mode": "independent_per_episode"}
    events = []
    exporting = threading.Event()
    second_started = threading.Event()

    def export(root, physical):
        key = physical["replay_episode_id"]
        events.append(("export_start", key))
        if key == plan[117]["key"]:
            exporting.set()
            assert second_started.wait(5), "next inference never overlapped the export"
        if fail_export:
            raise OSError("injected export failure")
        mp4 = root / f"{key.replace(':', '_')}.mp4"
        mp4.write_bytes(b"verified")
        physical.update(video_path=str(mp4), frame_images_deleted=True,
                        image_cleanup_error="", video_error="")
        events.append(("export_done", key))

    class Child:
        pid = 12345
        def __init__(self, args, **kwargs):
            index = int(args[args.index("--seed") + 1]) - 2110000
            self.spec = plan[index]
            events.append(("inference_start", self.spec["key"]))
            assert "--defer-video" in args
            if index == 118:
                assert exporting.wait(5)
                status = json.loads((root / "status.json").read_text())
                assert status["completed"] == 117
                assert status["media_pending"] == plan[117]["key"]
                second_started.set()
            if index == 119:
                assert ("export_done", plan[117]["key"]) in events
            self.stream = kwargs["stdout"]
        def wait(self):
            self.stream.write(json.dumps({"success": False, "target_count": self.spec["target_count"],
                "peak_collected": 0, "final_collected": 0, "model_checkpoint": "/checkpoint",
                "replay_episode_id": self.spec["key"], "replay_error": "", "video_path": None,
                "video_error": "", "frame_images_deleted": False}) + "\n")
            self.stream.flush()
            return 1

    monkeypatch.setattr(batch.subprocess, "Popen", Child)
    monkeypatch.setattr(batch.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(batch, "recover_episode_media", export)
    monkeypatch.setattr(batch, "validate_episode_identity", lambda *a, **k: {"source": "test"})
    monkeypatch.setattr(batch, "write_artifacts", lambda *a, **k: None)
    return batch, metadata, plan, events


def test_batch_overlaps_export_without_scoring_or_queuing_unfinished_media(tmp_path, monkeypatch):
    batch, metadata, plan, events = _batch_fixture(tmp_path, monkeypatch)
    batch.run_batch(tmp_path, metadata, pipeline_media=True)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [row["key"] for row in rows] == [spec["key"] for spec in plan]
    assert len([event for event in events if event[0] == "inference_start"]) == 3
    assert all(row["frame_images_deleted"] for row in rows[117:])
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "complete" and status["completed"] == 120


def test_batch_export_failure_keeps_measured_log_and_stops_without_resampling(tmp_path, monkeypatch):
    batch, metadata, plan, events = _batch_fixture(tmp_path, monkeypatch, fail_export=True)
    with pytest.raises(OSError, match="injected export failure"):
        batch.run_batch(tmp_path, metadata, pipeline_media=True)
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 117
    assert (tmp_path / "logs" / (plan[118]["key"].replace(":", "_") + ".log")).is_file()
    assert len([event for event in events if event[0] == "inference_start"]) == 2
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "needs_attention"


def test_pipeline_resume_finishes_saved_replay_without_running_inference_again(tmp_path, monkeypatch):
    batch, metadata, plan, events = _batch_fixture(tmp_path, monkeypatch)
    # This run has already measured the last three episodes. Only media is pending.
    for spec in plan[117:]:
        (tmp_path / "logs" / (spec["key"].replace(":", "_") + ".log")).write_text(json.dumps({
            "success": False, "target_count": spec["target_count"], "peak_collected": 0,
            "final_collected": 0, "model_checkpoint": "/checkpoint", "replay_episode_id": spec["key"]}) + "\n")
    calls = []
    def recover(root, physical):
        calls.append(physical["replay_episode_id"])
        physical.update(video_path=str(root / "existing.mp4"), frame_images_deleted=True)
    monkeypatch.setattr(batch, "recover_episode_media", recover)
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("measured episode must not run again"))
    batch.run_batch(tmp_path, metadata, pipeline_media=True)
    assert calls == [spec["key"] for spec in plan[117:]]
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 120


def test_pipeline_recovers_durable_physics_input_without_new_inference(tmp_path, monkeypatch):
    from sim.act import replay
    from sim.config import load_config
    from types import SimpleNamespace
    batch, metadata, plan, events = _batch_fixture(tmp_path, monkeypatch)
    (tmp_path / "replays").mkdir()
    first = plan[117]
    cfg = load_config(str(tmp_path / "evaluation_config.yaml"))
    cfg.set_path("task.target_count", first["target_count"])
    result = SimpleNamespace(success=False, failure_reason="original physical failure",
                             peak_collected=0, final_collected=0, target_count=first["target_count"],
                             collected=0, total=6, trace=[{"t": 0.41}])
    saved = replay.persist_replay_input(cfg, first["seed"], result, tmp_path / "replays",
                                        model="/checkpoint")
    for spec in plan[117:]:
        physical = {"success": False, "target_count": spec["target_count"], "peak_collected": 0,
                    "final_collected": 0, "model_checkpoint": "/checkpoint", "replay_episode_id": spec["key"]}
        if spec == first:
            physical.update(replay_episode_id=None, replay_error="render failed",
                            replay_input_path=str(saved))
        (tmp_path / "logs" / (spec["key"].replace(":", "_") + ".log")).write_text(json.dumps(physical) + "\n")
    restored = []
    def recover_input(path, cfg, *, seed, model, **kwargs):
        measured = replay.load_replay_input(path, cfg, seed=seed, model=model)
        assert measured.failure_reason == "original physical failure"
        restored.append(seed)
        return first["key"]
    monkeypatch.setattr(replay, "recover_replay_input", recover_input)
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("saved physics must not be inferred again"))
    monkeypatch.setattr(batch, "recover_episode_media", lambda root, physical: physical.update(
        video_path=str(root / "existing.mp4"), frame_images_deleted=True))
    batch.run_batch(tmp_path, metadata, pipeline_media=True)
    assert restored == [first["seed"]]
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 120


def test_later_replay_failure_still_commits_previous_verified_media(tmp_path, monkeypatch):
    batch, metadata, plan, events = _batch_fixture(tmp_path, monkeypatch)
    original_child = batch.subprocess.Popen
    class BrokenReplay(original_child):
        def wait(self):
            if self.spec == plan[118]:
                self.stream.write(json.dumps({"success": False, "target_count": self.spec["target_count"],
                    "peak_collected": 0, "final_collected": 0, "model_checkpoint": "/checkpoint",
                    "replay_episode_id": None, "replay_error": "injected replay failure"}) + "\n")
                self.stream.flush()
                return 1
            return super().wait()
    monkeypatch.setattr(batch.subprocess, "Popen", BrokenReplay)
    with pytest.raises(RuntimeError, match="unfinished replay"):
        batch.run_batch(tmp_path, metadata, pipeline_media=True)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 118
    assert rows[-1]["key"] == plan[117]["key"]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "needs_attention"


def test_derived_artifact_failure_cannot_append_the_same_result_twice(tmp_path, monkeypatch):
    batch, metadata, plan, events = _batch_fixture(tmp_path, monkeypatch)
    def failed_summary(root, rows, planned):
        if len(rows) == 118:
            raise OSError("derived summary write failed")
    monkeypatch.setattr(batch, "write_artifacts", failed_summary)
    with pytest.raises(OSError, match="derived summary"):
        batch.run_batch(tmp_path, metadata, pipeline_media=True)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 118
    assert len({row["key"] for row in rows}) == 118


@pytest.mark.parametrize("tamper", [None, "physics", "metadata", "unknown_revision"])
def test_media_revision_pins_all_sources_and_original_metadata(tmp_path, tamper):
    from sim.act import batch_evaluate as batch
    sources = {"sim/act/replay.py": "old-replay", "sim/environments/physics.py": "physics"}
    for name, contents in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    baseline = batch._source_fingerprints(tmp_path)
    metadata = {"project": str(tmp_path), "source_fingerprints": baseline}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "sim/act/replay.py").write_text("repaired-replay")
    revision = {"kind": "recorded_state_replay_v1", "name": "replay_fix",
                "metadata_sha256": batch._sha256(tmp_path / "metadata.json"),
                "before_sources": baseline, "after_sources": batch._source_fingerprints(tmp_path)}
    if tamper == "physics":
        (tmp_path / "sim/environments/physics.py").write_text("changed physical protocol")
        revision["after_sources"] = batch._source_fingerprints(tmp_path)
    elif tamper == "metadata":
        (tmp_path / "metadata.json").write_text("{}")
    elif tamper == "unknown_revision":
        revision["kind"] = "unknown"
    if tamper:
        with pytest.raises(RuntimeError, match="media revision"):
            batch.validate_media_revision(tmp_path, metadata, revision)
    else:
        assert batch.validate_media_revision(tmp_path, metadata, revision) == revision["after_sources"]
