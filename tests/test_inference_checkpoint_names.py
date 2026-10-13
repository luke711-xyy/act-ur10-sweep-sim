from pathlib import Path

from sim.act.replay import _episode_id


def test_checkpoint_replaces_seed_and_keeps_collision_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr("sim.act.replay.time.strftime", lambda fmt: "20261014_001000")
    model = "/models/ordinary/checkpoints/step_100000"
    first = _episode_id(tmp_path, 6, model)
    assert first == "inference_20261014_001000_n6_step_100000"
    (tmp_path / first).mkdir()
    assert _episode_id(tmp_path, 6, model) == first + "_02"


def test_custom_model_name_is_filename_safe(tmp_path, monkeypatch):
    monkeypatch.setattr("sim.act.replay.time.strftime", lambda fmt: "20261014_001000")
    name = _episode_id(tmp_path, 1, "/models/custom model:one")
    assert name == "inference_20261014_001000_n1_custom_model_one"
    assert Path(name).name == name


def test_model_less_preview_does_not_claim_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("sim.act.replay.time.strftime", lambda fmt: "20261014_001000")
    assert _episode_id(tmp_path, 1, None) == "inference_20261014_001000_n1_no_checkpoint"
