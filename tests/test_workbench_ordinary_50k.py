"""The workbench must pin its default to the new ordinary-ACT checkpoint."""

from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sim.act.evaluate import build_arg_parser
from sim.config import load_config
from sim.web.app import create_app
from sim.web.jobs import JobRegistry


EXPECTED_MODEL = (
    "runs/act_model_five_view_ordinary_disk14_landing7_300_200k_20261013"
    "/checkpoints/step_050000"
)


class _InputParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and "id" in attrs:
            self.inputs[attrs["id"]] = attrs


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    # Isolate persistent job records and prevent real inference subprocesses.
    registry = JobRegistry(tmp_path)
    monkeypatch.setattr("sim.web.jobs.JobRegistry", lambda root: registry)
    # Never depend on the live training pointer while testing defaults.
    monkeypatch.setattr("sim.web.app.latest_inference_model",
                        lambda root: EXPECTED_MODEL, raising=False)
    cfg = load_config()
    cfg.set_path("web.inference_video_dir", str(tmp_path / "videos"))
    before = cfg.to_dict()
    app = create_app(cfg)
    calls = []

    def record_job(kind, argv):
        calls.append((kind, argv))
        return {"kind": kind, "argv": argv}

    registry.start = record_job
    with TestClient(app) as client:
        yield client, cfg, before, calls


def test_served_page_pins_new_ordinary_50k(workbench):
    client, _, _, _ = workbench
    response = client.get("/")
    assert response.status_code == 200
    parser = _InputParser()
    parser.feed(response.text)
    assert parser.inputs["inferModel"]["value"] == EXPECTED_MODEL


@pytest.mark.parametrize("payload", [{}, {"model": None}, {"model": ""},
                                     {"model": "  \t  "}])
def test_default_inference_loads_fixed_50k_and_records_video(workbench, payload):
    client, cfg, before, calls = workbench
    response = client.post("/api/jobs/inference", json={
        **payload, "seed": 42, "target_count": 6, "preview": True,
    })
    assert response.status_code == 200
    job = response.json()
    assert job["kind"] == "inference"
    args = build_arg_parser().parse_args(job["argv"][3:])
    assert args.model == EXPECTED_MODEL
    assert args.config == cfg.get_path("_source_config")
    assert args.seed == 42
    assert args.target_count == 6
    assert args.preview and args.record_replay
    assert args.video_root == cfg.web.inference_video_dir
    assert cfg.to_dict() == before, "mounting a model must not alter layout rules"
    assert len(calls) == 1


def test_explicit_older_checkpoint_remains_selectable(workbench):
    client, _, _, _ = workbench
    old = "runs/act_model_five_view_rudra_300_smooth320_v1/checkpoints/step_100000"
    response = client.post("/api/jobs/inference", json={
        "model": old, "seed": 7, "target_count": 3,
    })
    assert response.status_code == 200
    args = build_arg_parser().parse_args(response.json()["argv"][3:])
    assert args.model == old


@pytest.mark.parametrize("model", [4, False, ["checkpoint"], {"path": "checkpoint"}])
def test_invalid_model_type_does_not_spawn_a_job(workbench, model):
    client, _, _, calls = workbench
    assert client.post("/api/jobs/inference", json={"model": model}).status_code == 422
    assert calls == [], "invalid model values must be rejected before job creation"


def test_workbench_tests_cannot_restore_live_user_jobs(tmp_path, monkeypatch):
    # Suppress restoration during the red test so reproducing the isolation bug
    # cannot write or attach a watcher to the user's live inference records.
    monkeypatch.setattr(JobRegistry, "_restore", lambda self: None)
    app = create_app(load_config())
    assert app.state.jobs.project_root != Path(__file__).resolve().parents[1]
    # Explicit temporary registries retain their existing path semantics.
    assert JobRegistry(tmp_path).project_root == tmp_path


CHECKPOINT_FILES = (
    "config.json", "model.safetensors", "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "training_state.pt",
)


def _checkpoint(root, name):
    folder = root / "checkpoints" / name
    folder.mkdir(parents=True)
    for filename in CHECKPOINT_FILES:
        (folder / filename).write_bytes(b"fixture")
    return folder


def test_refresh_and_empty_api_follow_published_checkpoint(tmp_path, monkeypatch):
    from sim.web import app as module
    model_root = tmp_path / "current_model"
    cp50 = _checkpoint(model_root, "step_050000")
    cp100 = _checkpoint(model_root, "step_100000")
    pointer = model_root / "latest_checkpoint.txt"
    pointer.write_text("checkpoints/step_050000")
    monkeypatch.setattr(module, "DEFAULT_INFERENCE_MODEL", str(cp50))
    registry = JobRegistry(tmp_path)
    monkeypatch.setattr("sim.web.jobs.JobRegistry", lambda root: registry)
    registry.start = lambda kind, argv: {"kind": kind, "argv": argv}
    with TestClient(create_app(load_config())) as client:
        parser = _InputParser()
        parser.feed(client.get("/").text)
        assert parser.inputs["inferModel"]["value"] == str(cp50)
        pointer.write_text("checkpoints/step_100000")
        parser.feed(client.get("/").text)
        assert parser.inputs["inferModel"]["value"] == str(cp100)
        args = build_arg_parser().parse_args(
            client.post("/api/jobs/inference", json={}).json()["argv"][3:])
        assert args.model == str(cp100)
        # An unfinished higher directory must not displace the published model.
        cp200 = _checkpoint(model_root, "step_200000")
        parser.feed(client.get("/").text)
        assert parser.inputs["inferModel"]["value"] == str(cp100)
        pointer.write_text("checkpoints/step_200000")
        parser.feed(client.get("/").text)
        assert parser.inputs["inferModel"]["value"] == str(cp200)
        args = build_arg_parser().parse_args(
            client.post("/api/jobs/inference", json={"model": " "}).json()["argv"][3:])
        assert args.model == str(cp200)


@pytest.mark.parametrize("bad_pointer", ["../other_model/step_200000",
                                        "checkpoints/step_200000", ""])
def test_invalid_or_incomplete_pointer_keeps_complete_50k(tmp_path, monkeypatch, bad_pointer):
    from sim.web import app as module
    root = tmp_path / "current_model"
    cp50 = _checkpoint(root, "step_050000")
    incomplete = _checkpoint(root, "step_200000")
    (incomplete / "training_state.pt").write_bytes(b"")
    (root / "latest_checkpoint.txt").write_text(bad_pointer)
    monkeypatch.setattr(module, "DEFAULT_INFERENCE_MODEL", str(cp50))
    with TestClient(create_app(load_config())) as client:
        parser = _InputParser()
        parser.feed(client.get("/").text)
        assert parser.inputs["inferModel"]["value"] == str(cp50)


def test_no_complete_default_rejects_job_but_explicit_model_still_works(tmp_path, monkeypatch):
    from sim.web import app as module
    monkeypatch.setattr(module, "DEFAULT_INFERENCE_MODEL",
                        str(tmp_path / "model/checkpoints/step_050000"))
    registry = JobRegistry(tmp_path)
    monkeypatch.setattr("sim.web.jobs.JobRegistry", lambda root: registry)
    registry.start = lambda kind, argv: {"kind": kind, "argv": argv}
    with TestClient(create_app(load_config())) as client:
        assert client.post("/api/jobs/inference", json={}).status_code == 503
        assert client.post("/api/jobs/inference", json={"model": "/explicit/model"}).status_code == 200


@pytest.mark.parametrize("escape", ["directory", "file"])
def test_default_rejects_symlink_escape(tmp_path, monkeypatch, escape):
    from sim.web import app as module
    root = tmp_path / "current_model"
    cp50 = _checkpoint(root, "step_050000")
    cp100 = _checkpoint(root, "step_100000")
    external = tmp_path / "external"
    external.mkdir()
    if escape == "directory":
        (root / "checkpoints").rename(external / "checkpoints")
        (root / "checkpoints").symlink_to(external / "checkpoints", target_is_directory=True)
        expected = ""
    else:
        weights = external / "model.safetensors"
        weights.write_bytes(b"fixture")
        (cp100 / "model.safetensors").unlink()
        (cp100 / "model.safetensors").symlink_to(weights)
        expected = str(cp50)
    (root / "latest_checkpoint.txt").write_text("checkpoints/step_100000")
    monkeypatch.setattr(module, "DEFAULT_INFERENCE_MODEL", str(cp50))
    assert module.latest_inference_model(tmp_path) == expected
