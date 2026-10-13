"""Inference produces a playable video without hiding its physical result."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from sim.act import evaluate, replay
from sim.act.dataset import ActDatasetWriter
from sim.config import load_config, save_config


VIEWS = (
    "corner_xminus_yminus", "corner_xminus_yplus",
    "corner_xplus_yminus", "corner_xplus_yplus", "wrist",
)
COLORS = ((220, 30, 30), (30, 220, 30), (30, 30, 220),
          (220, 220, 30), (220, 30, 220))


def _inference_fixture(tmp_path, monkeypatch, *, success=False, image_format="png"):
    root = tmp_path / "replays"
    episode_id = "inference_fixture_n2_s17"
    observations = [{
        "images": {view: np.full((12, 12, 3), color, dtype=np.uint8)
                   for view, color in zip(VIEWS, COLORS)},
        "state": np.zeros(36, dtype=np.float32),
        "environment_state": np.array([2], dtype=np.float32),
        "t": i / 10,
        "normal_force": 0.75,
        "raw_normal_force": 1.5,
        "tcp_pose": np.array([0, 0, 0.002, 0]),
        "reference": np.array([0, 0, 0.003, 0]),
        "fully_collected": 2 if success else 0,
        "z_owner": "admittance",
    } for i in range(2)]
    ActDatasetWriter(str(root), image_format=image_format).add_episode(
        episode_id, observations, np.zeros((2, 4), dtype=np.float32), success,
        {"episode_kind": "inference", "split": "inference", "fps": 10,
         "target_count": 2, "desired_force": 0.75,
         "model": "model/checkpoints/step_100000"},
    )
    result = SimpleNamespace(
        success=success, failure_reason="" if success else "2000-frame timeout before task completion",
        collected=2 if success else 0, target_count=2, total=6,
        elapsed=80.0, actions=[0] * 2000, scheduler_queries=1,
        scheduler_timeouts=0, scheduler_late_results=0, peak_force=1.5,
        termination_reason="", trace=[{}],
    )
    # Only model/simulator execution is expensive. The dataset and video encoder
    # remain real so these tests catch a missing/invalid exported artifact.
    monkeypatch.setattr(evaluate, "run_act_episode", lambda *args, **kwargs: result)
    monkeypatch.setattr(replay, "save_inference_replay",
                        lambda *args, **kwargs: episode_id)
    cfg = load_config()
    cfg.set_path("web.inference_video_dir", str(tmp_path / "videos"))
    config = tmp_path / "config.yaml"
    save_config(cfg, str(config))
    argv = ["--config", str(config), "--record-replay", "--replay-root", str(root)]
    return argv, root / episode_id, tmp_path / "videos"


@pytest.mark.parametrize("success,image_format", [(True, "png"), (False, "webp_lossless")])
def test_inference_exports_all_five_views_for_success_and_timeout(
        tmp_path, monkeypatch, capsys, success, image_format):
    argv, _, video_root = _inference_fixture(
        tmp_path, monkeypatch, success=success, image_format=image_format)
    code = evaluate.main(argv)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert "video_path" in summary, "inference did not publish its automatic video"
    video = Path(summary["video_path"])
    assert video.parent == video_root
    assert video.is_file() and video.stat().st_size > 100
    assert summary["video_error"] == ""
    assert summary["success"] is success
    assert code == (0 if success else 1)
    with imageio.get_reader(video, format="ffmpeg") as reader:
        assert reader.get_meta_data()["fps"] == 10.0
        frame = reader.get_data(0)
        assert frame.shape == (640, 960, 3)
        for (x, y), color in zip(((160, 160), (480, 160), (160, 480),
                                 (480, 480), (800, 160)), COLORS):
            assert np.max(np.abs(frame[y, x].astype(int) - color)) < 12
        assert len(list(reader.iter_data())) == 2


def test_deferred_video_publishes_replay_but_keeps_images_for_batch_worker(
        tmp_path, monkeypatch, capsys):
    from sim.act import video
    argv, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(video, "export_inference_video", lambda *a, **k: pytest.fail("deferred export must not encode in inference"))
    assert evaluate.main(argv + ["--defer-video"]) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["replay_episode_id"] == episode.name
    assert summary["video_path"] is None
    assert summary["video_deferred"] is True
    assert summary["frame_images_deleted"] is False
    assert list(episode.glob("*.png"))
    assert not video_root.exists()


def test_export_failure_preserves_timeout_reason_and_replay(
        tmp_path, monkeypatch, capsys):
    argv, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    (episode / "image_wrist_00000.png").unlink()
    assert evaluate.main(argv) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary.get("video_path") is None
    assert summary.get("video_error"), "missing images must be reported as a video error"
    assert summary["reason"] == "2000-frame timeout before task completion"
    assert summary["replay_episode_id"] == "inference_fixture_n2_s17"
    assert list(video_root.glob("*")) == []


def test_repeated_export_does_not_overwrite_existing_video(
        tmp_path, monkeypatch, capsys):
    from sim.act.video import export_inference_video
    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    first = export_inference_video(episode, video_root)
    original = first.read_bytes()
    second = export_inference_video(episode, video_root)
    assert second != first
    assert second.is_file()
    assert first.read_bytes() == original


@pytest.mark.parametrize("image_format", ["png", "webp-lossless"])
def test_verified_video_replaces_images_without_losing_workbench_playback(
        tmp_path, monkeypatch, capsys, image_format):
    from sim.web.workbench import WorkbenchState
    argv, episode, _ = _inference_fixture(
        tmp_path, monkeypatch, image_format=image_format)
    assert evaluate.main(argv) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary.get("frame_images_deleted") is True
    assert not list(episode.glob("*.png")) and not list(episode.glob("*.webp"))
    assert (episode / "arrays.npz").is_file()
    state = WorkbenchState(load_config(), dataset_root=tmp_path / "missing",
                           preview_root=episode.parent)
    metadata = state.episode_metadata(episode.name)
    assert metadata["video_path"] == summary["video_path"]
    frame = state.load_episode_frame(episode.name, 1)
    assert set(frame["images"]) == set(VIEWS)
    for view, color in zip(VIEWS, COLORS):
        assert np.max(np.abs(frame["images"][view][160, 160].astype(int) - color)) < 12
    assert state.load_episode_frame_data(episode.name, 0)["target_count"] == 2


def test_cleanup_refuses_expert_records_and_unsafe_frame_paths(
        tmp_path, monkeypatch):
    from sim.act import media_retention
    from sim.act.video import export_inference_video
    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    video = export_inference_video(episode, video_root)
    manifest = episode.parent / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    original_images = list(episode.glob("*.png"))
    record["episode_kind"] = "expert"
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="inference"):
        media_retention.finalize_inference_video(episode, video)
    assert all(p.is_file() for p in original_images)
    record["episode_kind"] = "inference"
    record["images"]["wrist"][0] = "../unrelated.png"
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="outside"):
        media_retention.finalize_inference_video(episode, video)
    assert all(p.is_file() for p in original_images)


def test_invalid_video_is_not_a_license_to_delete_images(tmp_path, monkeypatch):
    from sim.act import media_retention
    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    video_root.mkdir()
    invalid = video_root / "broken.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises((RuntimeError, OSError)):
        media_retention.finalize_inference_video(episode, invalid)
    assert len(list(episode.glob("*.png"))) == 10


def test_legacy_three_view_replay_is_exported_without_inventing_raw_force(
        tmp_path, monkeypatch):
    from sim.act.video import export_inference_video
    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    manifest = episode.parent / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    record["overhead"] = record["images"][VIEWS[0]]
    record["inspection"] = record["images"][VIEWS[1]]
    record["wrist"] = record["images"]["wrist"]
    record.pop("images")
    manifest.write_text(json.dumps(record) + "\n")
    with np.load(episode / "arrays.npz") as data:
        values = {k: data[k] for k in data.files if k != "raw_normal_force"}
    np.savez_compressed(episode / "arrays.npz", **values)
    video = export_inference_video(episode, video_root)
    assert "three_view" in video.name
    with imageio.get_reader(video, format="ffmpeg") as reader:
        frame = reader.get_data(0)
        for (x, y), color in zip(((160, 160), (480, 160), (800, 160)),
                                 (COLORS[0], COLORS[1], COLORS[4])):
            assert np.max(np.abs(frame[y, x].astype(int) - color)) < 12


@pytest.mark.parametrize("inspection", [False, True])
def test_legacy_camera_map_exports_its_actual_views(tmp_path, monkeypatch, inspection):
    from sim.act.video import export_inference_video
    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    manifest = episode.parent / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    old = record["images"]
    record["images"] = {"overhead": old[VIEWS[0]], "wrist": old["wrist"]}
    if inspection:
        record["images"]["inspection"] = old[VIEWS[1]]
    manifest.write_text(json.dumps(record) + "\n")
    exported = export_inference_video(episode, video_root)
    assert ("three_view" if inspection else "two_view") in exported.name


def test_corrupted_required_causal_trace_preserves_all_images(tmp_path, monkeypatch):
    from sim.act.media_retention import finalize_inference_video
    from sim.act.video import export_inference_video
    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    video = export_inference_video(episode, video_root)
    manifest = episode.parent / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    trace = episode / "causal_trace_100hz.json.gz"
    trace.write_bytes(b"corrupted")
    record.update(causal_trace_required=True, causal_trace=str(trace.relative_to(episode.parent)),
                  causal_trace_rows=1, causal_trace_sha256="bad")
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises((ValueError, RuntimeError, OSError), match="causal"):
        finalize_inference_video(episode, video)
    assert len(list(episode.glob("*.png"))) == 10


def test_stalled_encoder_is_killed_and_does_not_leave_a_partial_video(
        tmp_path, monkeypatch):
    import imageio_ffmpeg
    from sim.act import video

    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    real_popen = subprocess.Popen
    children = []

    def stalled_encoder(_argv, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(3)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: sys.executable)
    monkeypatch.setattr(video.subprocess, "Popen", stalled_encoder)
    monkeypatch.setattr(video, "ENCODING_TIMEOUT_SECONDS", 0.1, raising=False)
    with pytest.raises(TimeoutError, match="video encoding"):
        video.export_inference_video(episode, video_root)
    assert children[0].poll() is not None
    assert list(video_root.iterdir()) == []


def test_early_encoder_failure_reports_stderr_and_cleans_partial_video(
        tmp_path, monkeypatch):
    import imageio_ffmpeg
    from sim.act import video

    _, episode, video_root = _inference_fixture(tmp_path, monkeypatch)
    real_popen = subprocess.Popen

    def failing_encoder(_argv, **kwargs):
        return real_popen([
            sys.executable, "-c",
            "import sys; sys.stderr.write('encoder disk full'); sys.exit(1)",
        ], **kwargs)

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: sys.executable)
    monkeypatch.setattr(video.subprocess, "Popen", failing_encoder)
    with pytest.raises(RuntimeError, match="encoder disk full"):
        video.export_inference_video(episode, video_root)
    assert list(video_root.iterdir()) == []


def test_physical_result_is_published_before_video_encoding(
        tmp_path, monkeypatch, capsys):
    from sim.act import video

    argv, _, _ = _inference_fixture(tmp_path, monkeypatch)
    observed = {}

    def encoding_failure(*_args, **_kwargs):
        observed["output_before_encoding"] = capsys.readouterr().out
        raise RuntimeError("test encoding failure")

    monkeypatch.setattr(video, "export_inference_video", encoding_failure)
    assert evaluate.main(argv) == 1
    lines = observed["output_before_encoding"].splitlines()
    assert lines, "completed physical inference must be visible before encoding starts"
    physical = json.loads(lines[-1])
    assert physical["success"] is False
    assert physical["reason"] == "2000-frame timeout before task completion"
    assert physical["replay_episode_id"] == "inference_fixture_n2_s17"
    final = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert "test encoding failure" in final["video_error"]


def test_video_records_the_checkpoint_used_before_latest_pointer_changes(
        tmp_path, monkeypatch, capsys):
    argv, episode, _ = _inference_fixture(tmp_path, monkeypatch)
    model = tmp_path / "family_a"
    first = model / "checkpoints" / "step_100000"
    second = model / "checkpoints" / "step_200000"
    for checkpoint in (first, second):
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"fixture-weights")
    pointer = model / "latest_checkpoint.txt"
    pointer.write_text("checkpoints/step_100000", encoding="utf-8")
    fake_run = evaluate.run_act_episode

    def run_and_advance_latest(cfg, seed=0, model_path=None, preview=False, capture_replay=False):
        # A concurrent trainer can publish 200k after inference already loaded
        # 100k. The saved artifact must identify the loaded snapshot, not latest.
        result = fake_run(cfg, seed=seed, model_path=model_path, preview=preview,
                         capture_replay=capture_replay)
        pointer.write_text("checkpoints/step_200000", encoding="utf-8")
        return result

    def publish_snapshot(cfg, seed, result, root, model=None, image_format="png"):
        manifest = episode.parent / "manifest.jsonl"
        record = json.loads(manifest.read_text(encoding="utf-8"))
        record["model"] = model
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return episode.name

    monkeypatch.setattr(evaluate, "run_act_episode", run_and_advance_latest)
    monkeypatch.setattr(replay, "save_inference_replay", publish_snapshot)
    assert evaluate.main([*argv, "--model", str(model)]) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary.get("model_checkpoint") == str(first), "video must identify the frozen loaded checkpoint"
    record = json.loads((episode.parent / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["model"] == str(first)
    assert Path(summary["video_path"]).is_file()
