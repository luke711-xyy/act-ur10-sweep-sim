"""Batch scoring keeps physical failures and never mixes manual artifacts."""

import importlib
import json
from pathlib import Path
import pytest


def batch_module():
    assert importlib.util.find_spec("sim.act.batch_evaluate") is not None, (
        "serial checkpoint batch evaluator is not implemented")
    return importlib.import_module("sim.act.batch_evaluate")


def _independent_layouts(count=120):
    return [{"layout_index": i, "seed": 2110000 + i,
             "layout_fingerprint": f"layout-{i}"} for i in range(count)]


def _write_complete_ledger(root, plan):
    video = root / "existing.mp4"
    video.write_bytes(b"video")
    rows = [{**spec, "video_path": str(video), "success": False,
             "peak_collected": 0, "final_collected": 0} for spec in plan]
    (root / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_batch_plan_has_twenty_independent_layouts_for_each_goal():
    batch = batch_module()
    plan = batch.make_episode_plan(_independent_layouts(), [{"label": "five_view_100k"}])
    assert len(plan) == 120
    assert len({x["seed"] for x in plan}) == 120
    assert len({x["layout_fingerprint"] for x in plan}) == 120
    for target in range(1, 7):
        assert [x["seed"] for x in plan if x["target_count"] == target] == list(
            range(2110000 + target - 1, 2110120, 6))
    assert len({x["key"] for x in plan}) == 120


def test_multiple_checkpoints_do_not_reuse_seeds_or_layouts():
    batch = batch_module()
    plan = batch.make_episode_plan(_independent_layouts(240), [
        {"label": "five_view_100k"}, {"label": "five_view_200k"}])
    assert len(plan) == 240
    assert len({x["seed"] for x in plan}) == 240
    assert len({x["layout_fingerprint"] for x in plan}) == 240
    for label in ("five_view_100k", "five_view_200k"):
        for target in range(1, 7):
            assert sum(x["model_label"] == label and x["target_count"] == target
                       for x in plan) == 20


@pytest.mark.parametrize("field", ["seed", "layout_fingerprint"])
def test_batch_plan_rejects_duplicate_independent_layouts(field):
    batch = batch_module()
    layouts = _independent_layouts()
    layouts[-1][field] = layouts[0][field]
    with pytest.raises(ValueError, match="unique"):
        batch.make_episode_plan(layouts, [{"label": "five_view_100k"}])


def test_batch_plan_rejects_shared_twenty_layout_protocol():
    batch = batch_module()
    with pytest.raises(ValueError, match="120"):
        batch.make_episode_plan(_independent_layouts(20), [{"label": "five_view_100k"}])


def test_resume_rejects_modified_frozen_episode_plan_before_launch(tmp_path, monkeypatch):
    batch = batch_module()
    plan_path = tmp_path / "episode_plan.json"
    plan_path.write_text(json.dumps(_independent_layouts()))
    fingerprint = batch._sha256(plan_path)
    plan_path.write_text("[]")
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("must reject changed plan before inference"))
    with pytest.raises(RuntimeError, match="episode plan changed"):
        batch.run_batch(tmp_path, {"project": str(tmp_path), "episode_plan_sha256": fingerprint,
                                  "layout_seed_mode": "independent_per_episode"})


@pytest.mark.parametrize("missing", ["episode_plan_sha256", "layout_seed_mode"])
def test_resume_rejects_missing_independent_protocol_even_with_complete_ledger(tmp_path, missing):
    batch = batch_module()
    plan = batch.make_episode_plan(_independent_layouts(), [{"label": "five_view_100k"}])
    (tmp_path / "episode_plan.json").write_text(json.dumps(plan))
    _write_complete_ledger(tmp_path, plan)
    metadata = {"project": str(tmp_path), "episode_plan_sha256": batch._sha256(tmp_path / "episode_plan.json"),
                "layout_seed_mode": "independent_per_episode"}
    metadata.pop(missing)
    with pytest.raises(RuntimeError, match="independent.*protocol"):
        batch.run_batch(tmp_path, metadata)
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("field", ["seed", "layout_fingerprint", "key"])
def test_resume_rejects_legacy_repeated_seed_plan_even_when_hash_matches(tmp_path, field):
    batch = batch_module()
    plan = batch.make_episode_plan(_independent_layouts(), [{"label": "five_view_100k"}])
    plan[-1][field] = plan[0][field]
    (tmp_path / "episode_plan.json").write_text(json.dumps(plan))
    _write_complete_ledger(tmp_path, plan)
    with pytest.raises(RuntimeError, match="unique"):
        batch.run_batch(tmp_path, {"project": str(tmp_path),
                                  "episode_plan_sha256": batch._sha256(tmp_path / "episode_plan.json"),
                                  "layout_seed_mode": "independent_per_episode"})
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("field,value", [("seed", 2100000), ("layout_fingerprint", "old-layout"),
                                         ("target_count", 6), ("model_label", "old-model"),
                                         ("model_path", "/old-checkpoint")])
def test_resume_rejects_scored_identity_that_does_not_match_the_frozen_plan(tmp_path, field, value):
    batch = batch_module()
    plan = batch.make_episode_plan(_independent_layouts(), [{"label": "five_view_100k"}])
    (tmp_path / "episode_plan.json").write_text(json.dumps(plan))
    _write_complete_ledger(tmp_path, plan)
    ledger = tmp_path / "episodes.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0][field] = value
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="ledger.*frozen plan"):
        batch.run_batch(tmp_path, {"project": str(tmp_path),
                                  "episode_plan_sha256": batch._sha256(tmp_path / "episode_plan.json"),
                                  "layout_seed_mode": "independent_per_episode"})
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("field", [None, "seed", "fingerprint", "model", "target_count"])
def test_stale_log_cannot_relabel_a_different_measured_replay(tmp_path, monkeypatch, field):
    import hashlib
    import numpy as np
    batch = batch_module()
    layouts = _independent_layouts()
    poses = [[0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0]] * 6
    layouts[0]["initial_object_poses"] = poses
    layouts[0]["layout_fingerprint"] = hashlib.sha256(np.asarray(poses, dtype="<f4").tobytes()).hexdigest()
    plan = batch.make_episode_plan(layouts, [{"label": "five_view_100k", "path": "/checkpoint"}])
    (tmp_path / "episode_plan.json").write_text(json.dumps(plan))
    _write_complete_ledger(tmp_path, plan[1:])
    config = tmp_path / "evaluation_config.yaml"
    config.write_text("{}")
    replay = tmp_path / "replays"
    replay.mkdir()
    measured = {"episode_id": "inference_stale", "seed": 2110000, "target_count": 1,
                "model": "/checkpoint", "initial_object_poses": poses,
                "layout_fingerprint": layouts[0]["layout_fingerprint"]}
    if field == "fingerprint":
        measured["initial_object_poses"] = [[0.2, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0]] * 6
        measured["layout_fingerprint"] = hashlib.sha256(
            np.asarray(measured["initial_object_poses"], dtype="<f4").tobytes()).hexdigest()
    elif field:
        measured[field] = {"seed": 2100000, "model": "/old-checkpoint", "target_count": 2}[field]
    (replay / "manifest.jsonl").write_text(json.dumps(measured) + "\n")
    log = tmp_path / "logs"
    log.mkdir()
    physical = {"success": False, "target_count": 1, "peak_collected": 0, "final_collected": 0,
                "model_checkpoint": "/checkpoint", "replay_episode_id": "inference_stale"}
    (log / "five_view_100k_n1_l00.log").write_text(json.dumps(physical) + "\n")
    def finish_media(root, result):
        result.update(video_path=str(root / "existing.mp4"), frame_images_deleted=True)
    monkeypatch.setattr(batch, "recover_episode_media", finish_media)
    monkeypatch.setattr(batch.subprocess, "check_output", lambda *a, **k: "")
    metadata = {"project": str(tmp_path), "models": [], "source_fingerprints": {},
                "config_sha256": batch._sha256(config),
                "episode_plan_sha256": batch._sha256(tmp_path / "episode_plan.json"),
                "layout_seed_mode": "independent_per_episode"}
    if field:
        with pytest.raises(RuntimeError, match="measured.*identity"):
            batch.run_batch(tmp_path, metadata)
        assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 119
        assert not (tmp_path / "summary.json").exists()
    else:
        batch.run_batch(tmp_path, metadata)
        rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
        assert len(rows) == 120
        assert rows[-1]["layout_identity_verification"]["source"] == "initial_replay_pose_snapshot"
        assert rows[-1]["layout_identity_verification"]["measured_pose_fingerprint"] == layouts[0]["layout_fingerprint"]


@pytest.mark.parametrize("variant", ["near_initial", "late_frame", "different_layout", "unapproved", "nan_time"])
def test_legacy_identity_requires_an_early_near_initial_recorded_frame(tmp_path, variant):
    import hashlib
    import numpy as np
    batch = batch_module()
    assert callable(getattr(batch, "validate_episode_identity", None))
    replay = tmp_path / "replays"
    replay.mkdir()
    (replay / "inference_legacy").mkdir()
    initial = np.asarray([[0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0]] * 6, dtype="<f4")
    observed = initial.copy()
    observed[0, 0] += 0.01 if variant == "different_layout" else 0.0001
    frame_time = float("nan") if variant == "nan_time" else 0.8 if variant == "late_frame" else 0.44
    np.savez_compressed(replay / "inference_legacy/arrays.npz", object_pose=observed[None],
                        t=np.array([frame_time]))
    (tmp_path / "evaluation_config.yaml").write_text("episode:\n  settle_time: 0.4\nact:\n  action_hz: 25\n")
    (replay / "manifest.jsonl").write_text(json.dumps({
        "episode_id": "inference_legacy", "seed": 2110000, "target_count": 1,
        "model": "/checkpoint", "arrays": "inference_legacy/arrays.npz"}) + "\n")
    spec = {"seed": 2110000, "target_count": 1, "model_path": "/checkpoint",
            "initial_object_poses": initial.tolist(),
            "layout_fingerprint": hashlib.sha256(initial.tobytes()).hexdigest()}
    physical = {"replay_episode_id": "inference_legacy"}
    if variant == "near_initial":
        verification = batch.validate_episode_identity(tmp_path, spec, physical, allow_legacy_first_frame=True)
        assert verification["source"] == "legacy_first_replay_frame"
        assert verification["measured_pose_fingerprint"] != spec["layout_fingerprint"]
    else:
        with pytest.raises(RuntimeError, match="measured.*identity"):
            batch.validate_episode_identity(tmp_path, spec, physical,
                                           allow_legacy_first_frame=variant != "unapproved")


@pytest.mark.parametrize("training_overlap", [None, "seed", "fingerprint"])
def test_preparation_materializes_independent_layouts_and_rejects_training_overlap(
        tmp_path, monkeypatch, training_overlap):
    import hashlib
    import numpy as np
    from sim.environments import sweep_env
    batch = batch_module()
    calls = []

    class LayoutFixture:
        def __init__(self, cfg, seed):
            self.cfg, self.seed = cfg, seed

        def reset(self, seed):
            calls.append((seed, self.cfg.task.target_count))

        def component_positions(self):
            return np.asarray([[self.seed, 0.0, 0.0]] * 6)

        def component_quats(self):
            return np.asarray([[1.0, 0.0, 0.0, 0.0]] * 6)

        def close(self):
            pass

    monkeypatch.setattr(sweep_env, "SweepEnv", LayoutFixture)
    monkeypatch.setattr(batch, "select_models", lambda root: [{"label": "five_view_100k"}])
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    first_layout = np.asarray([[2110000, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 6, dtype="<f4")
    training_record = {"seed": 2110000 if training_overlap == "seed" else 99,
                       "layout_fingerprint": (hashlib.sha256(first_layout.tobytes()).hexdigest()
                                              if training_overlap == "fingerprint" else "training-layout")}
    (dataset / "manifest.jsonl").write_text(json.dumps(training_record) + "\n")
    root = tmp_path / "new-batch"
    arguments = dict(project=tmp_path, config=Path(__file__).resolve().parents[1] / "configs/default.yaml",
                     dataset=dataset, model_root=tmp_path / "model", seed_start=2110000)
    if training_overlap:
        with pytest.raises(ValueError, match="training"):
            batch.prepare_batch(root, **arguments)
        assert not root.exists()
        return
    metadata = batch.prepare_batch(root, **arguments)
    plan = json.loads((root / "episode_plan.json").read_text())
    assert metadata["layouts"] == metadata["planned"] == 120
    assert metadata["layout_seed_mode"] == "independent_per_episode"
    assert metadata["episode_plan_sha256"] == hashlib.sha256((root / "episode_plan.json").read_bytes()).hexdigest()
    assert calls[:6] == [(2110000, 1), (2110001, 2), (2110002, 3),
                         (2110003, 4), (2110004, 5), (2110005, 6)]
    assert calls[-1] == (2110119, 6) and len(calls) == 120
    assert [(row["seed"], row["target_count"]) for row in plan] == calls
    assert len({row["layout_fingerprint"] for row in plan}) == 120
    assert not (root / "episodes.jsonl").exists()


def _checkpoint(root, step):
    path = root / "checkpoints" / f"step_{step:06d}"
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "config.json").write_text("{}")
    (path / "training_state.pt").write_bytes(b"training")
    for name in ("policy_preprocessor", "policy_postprocessor"):
        (path / f"{name}.json").write_text(json.dumps({"steps": [{"state_file": f"{name}.safetensors"}]}))
        (path / f"{name}.safetensors").write_bytes(b"state")
    return path


def test_checkpoint_selection_excludes_an_unfinished_200k_directory(tmp_path):
    batch = batch_module()
    _checkpoint(tmp_path, 100000)
    second = _checkpoint(tmp_path, 200000)
    pointer = tmp_path / "latest_checkpoint.txt"
    pointer.write_text("checkpoints/step_100000")
    assert [m["step"] for m in batch.select_models(tmp_path)] == [100000]
    pointer.write_text("checkpoints/step_200000")
    (second / "policy_preprocessor.safetensors").unlink()
    assert [m["step"] for m in batch.select_models(tmp_path)] == [100000]
    (second / "policy_preprocessor.safetensors").write_bytes(b"complete")
    assert [m["step"] for m in batch.select_models(tmp_path)] == [100000, 200000]


def test_density_uses_peak_count_but_keeps_strict_success_and_final_count():
    batch = batch_module()
    rows = [
        {"model_label": "five_view_100k", "target_count": 2,
         "peak_collected": 2, "final_collected": 1, "success": False},
        {"model_label": "five_view_100k", "target_count": 2,
         "peak_collected": 2, "final_collected": 2, "success": True},
        {"model_label": "five_view_100k", "target_count": 2,
         "peak_collected": 0, "final_collected": 0, "success": False},
    ]
    stats = batch.summarize_rows(rows)["five_view_100k"]
    assert stats["episodes"] == 3
    assert stats["successes"] == 1
    assert stats["exact_peak_matches"] == 2
    assert stats["peak_count_matrix"][2][1] == 2
    assert stats["final_count_matrix"][1][1] == 1
    assert stats["final_count_matrix"][2][1] == 1
    assert stats["by_target"]["2"]["success_rate"] == pytest.approx(1 / 3)
    assert stats["by_target"]["1"]["success_rate"] is None


def test_command_uses_frozen_config_and_independent_batch_paths(tmp_path):
    batch = batch_module()
    spec = {"model_path": "/models/step_100000", "seed": 2100000,
            "target_count": 4}
    args = batch.build_episode_command(tmp_path, spec)
    assert args[args.index("--model") + 1] == "/models/step_100000"
    assert args[args.index("--config") + 1] == str(tmp_path / "evaluation_config.yaml")
    assert args[args.index("--replay-root") + 1] == str(tmp_path / "replays")
    assert args[args.index("--video-root") + 1] == str(tmp_path / "videos")
    assert args[args.index("--replay-image-format") + 1] == "webp-lossless"
    assert "--preview" in args


def test_parser_accepts_timeout_as_a_measured_outcome_not_process_error():
    batch = batch_module()
    output = ('loader status\n'
              '{"success":false,"target_count":4,"peak_collected":0,'
              '"final_collected":0,"model_checkpoint":"/models/step_100000",'
              '"reason":"2000-frame timeout before task completion",'
              '"replay_episode_id":"inference_fixture","replay_error":"",'
              '"video_path":"/batch/videos/a.mp4","video_error":""}\n')
    result = batch.parse_episode_output(output, returncode=1)
    assert result["success"] is False
    assert result["reason"] == "2000-frame timeout before task completion"
    with pytest.raises(RuntimeError, match="physical result"):
        batch.parse_episode_output("Traceback: load failed", returncode=1)


def test_partial_checkpoint_success_cannot_hide_video_export_failure():
    batch = batch_module()
    with pytest.raises(RuntimeError, match="video"):
        batch.parse_episode_output(
            '{"success":false,"target_count":2,"peak_collected":0,'
            '"final_collected":0,"model_checkpoint":"/models/step_100000",'
            '"replay_episode_id":"inference_fixture","replay_error":"",'
            '"video_path":null,"video_error":"disk full"}', returncode=1)


def test_written_artifacts_include_empty_goals_without_nan(tmp_path):
    batch = batch_module()
    row = {"key": "five_view_100k:n2:l0", "model_label": "five_view_100k",
           "target_count": 2, "seed": 2100000, "success": False,
           "peak_collected": 0, "final_collected": 0}
    batch.write_artifacts(tmp_path, [row], planned=120)
    import json
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] == 1 and summary["complete"] is False
    assert (tmp_path / "count_density.png").stat().st_size > 1000
    assert "NaN" not in (tmp_path / "summary.json").read_text()
    assert "five_view_100k" in (tmp_path / "episodes.csv").read_text()


def test_recovery_uses_published_video_after_images_were_deleted(tmp_path, monkeypatch):
    batch = batch_module()
    from sim.act import media_retention, video
    episode = tmp_path / "replays" / "inference_fixture"
    episode.mkdir(parents=True)
    mp4 = tmp_path / "existing.mp4"
    mp4.write_bytes(b"existing validated video")
    (episode.parent / "manifest.jsonl").write_text(json.dumps({
        "episode_id": episode.name, "episode_kind": "inference",
        "video_path": str(mp4), "frame_images_deleted": True}) + "\n")
    monkeypatch.setattr(video, "export_inference_video", lambda *args: pytest.fail("must not encode deleted images"))
    calls = []
    monkeypatch.setattr(media_retention, "finalize_inference_video", lambda e, v: calls.append((e, v)))
    physical = {"replay_episode_id": episode.name}
    batch.recover_episode_media(tmp_path, physical)
    assert physical["video_path"] == str(mp4)
    assert calls == [(episode, mp4)]


def test_complete_ledger_regenerates_stale_derived_artifacts(tmp_path, monkeypatch):
    batch = batch_module()
    mp4 = tmp_path / "existing.mp4"
    mp4.write_bytes(b"video")
    row = {"key": "five_view_100k:n2:l0", "model_label": "five_view_100k",
           "target_count": 2, "seed": 2100000, "success": False,
           "peak_collected": 0, "final_collected": 0, "video_path": str(mp4)}
    plan = batch.make_episode_plan(_independent_layouts(), [{"label": "five_view_100k"}])
    rows = [{**row, **spec} for spec in plan]
    (tmp_path / "episode_plan.json").write_text(json.dumps(plan))
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    (tmp_path / "summary.json").write_text('{"completed":0}')
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("completed episodes must not run again"))
    batch.run_batch(tmp_path, {"project": str(tmp_path),
                              "episode_plan_sha256": batch._sha256(tmp_path / "episode_plan.json"),
                              "layout_seed_mode": "independent_per_episode"})
    assert json.loads((tmp_path / "summary.json").read_text())["complete"] is True
    assert (tmp_path / "count_density.png").is_file()
