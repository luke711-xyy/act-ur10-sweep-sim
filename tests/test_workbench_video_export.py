"""Real saved media exports; never substitute a new simulation for a replay."""
import hashlib
import json
import time
from pathlib import Path
from threading import Event
import subprocess
import shutil
import sys
from threading import Timer

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sim.act.dataset import ActDatasetWriter
from sim.config import load_config
from sim.web.app import create_app
from sim.web.jobs import JobRegistry
from sim.web.workbench import WorkbenchState

VIEWS = ("corner_xminus_yminus", "corner_xminus_yplus",
         "corner_xplus_yminus", "corner_xplus_yplus", "wrist")
COLORS = ((210, 30, 30), (30, 210, 30), (30, 30, 210),
          (210, 210, 30), (210, 30, 210))


def saved(root, *, name="demo", kind="expert", success=True, offset=0):
    observations = [{
        "images": {v: np.full((16, 16, 3), np.array(c) + offset, dtype=np.uint8)
                   for v, c in zip(VIEWS, COLORS)},
        "state": np.zeros(36, dtype=np.float32),
        "environment_state": np.array([6, 2, 2]), "phase": "sweep",
        "t": i / 10, "fully_collected": 2 if success else 0,
        "normal_force": .7,
    } for i in range(3)]
    ActDatasetWriter(str(root)).add_episode(
        name, observations, np.zeros((3, 4)), success,
        {"seed": 17, "target_count": 2, "count": 6, "total_count": 6,
         "fps": 10., "episode_kind": kind, "split": "train"})
    return json.loads((root / "manifest.jsonl").read_text())


def fingerprint(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def make_app(tmp_path):
    cfg = load_config(overrides=["sim.real_time=false"])
    cfg.set_path("web.inference_video_dir", str(tmp_path / "videos"))
    app = create_app(cfg)
    app.state.workbench = WorkbenchState(
        cfg, dataset_root=tmp_path / "dataset", preview_root=tmp_path / "previews")
    app.state.jobs = JobRegistry(tmp_path)
    return app


def wait_job(app, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = next(j for j in app.state.jobs.list() if j["job_id"] == job_id)
        if job["state"] not in {"starting", "running", "stopping"}:
            return job
        time.sleep(.02)
    pytest.fail("video export did not finish")


def test_same_name_selects_exact_source_and_keeps_raw_id_compatibility(tmp_path):
    saved(tmp_path / "dataset")
    saved(tmp_path / "previews", success=False, offset=10)
    state = make_app(tmp_path).state.workbench
    records = state.list_episodes()
    assert {r["playback_id"] for r in records} == {"dataset:demo", "preview:demo"}
    assert state.episode_metadata("demo")["success"] is True
    assert state.episode_metadata("preview:demo")["success"] is False
    image = state.load_episode_frame("preview:demo", 0)["images"][VIEWS[0]]
    np.testing.assert_array_equal(image[0, 0], np.array(COLORS[0]) + 10)
    root, record = state.export_source("preview:demo")
    assert root == (tmp_path / "previews").resolve() and record["episode_id"] == "demo"


def test_full_five_view_export_has_original_speed_pixels_and_unchanged_sources(tmp_path):
    from sim.web.video_export import export_saved_episode
    root = tmp_path / "dataset"
    record = saved(root)
    before = fingerprint(root)
    progress = []
    result = export_saved_episode(root, record, tmp_path / "out",
                                  progress=lambda p: progress.append(p))
    video = Path(result["video_path"])
    with imageio.get_reader(video, format="ffmpeg") as reader:
        assert reader.get_meta_data()["fps"] == 10.
        frames = list(reader.iter_data())
    assert len(frames) == 3 and frames[0].shape == (688, 960, 3)
    for view, color in zip(VIEWS, COLORS):
        x, y, w, h = result["video_layout"][view]
        pixel = frames[0][y + h // 2, x + w // 2]
        assert np.max(np.abs(pixel.astype(int) - color)) < 12
    assert result["episode_kind"] == "expert" and result["success"] is True
    assert result["frames"] == 3 and result["reused"] is False
    assert progress[-1]["stage"] == "complete"
    assert fingerprint(root) == before
    again = export_saved_episode(root, record, tmp_path / "out")
    assert again["video_path"] != str(video) and video.is_file()


@pytest.mark.parametrize("damage", ["camera", "path", "fps", "empty", "cancel"])
def test_invalid_or_cancelled_exports_never_publish_or_change_source(tmp_path, damage):
    from sim.web.video_export import export_saved_episode
    root = tmp_path / "dataset"
    record = saved(root)
    if damage == "camera":
        record["images"].pop("wrist")
    elif damage == "path":
        record["images"]["wrist"][0] = "../outside.png"
    elif damage == "fps":
        record["fps"] = 0.
    elif damage == "empty":
        record["images"]["wrist"] = []
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    cancelled = Event()
    if damage == "cancel":
        cancelled.set()
    before = fingerprint(root)
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        export_saved_episode(root, record, tmp_path / "out", cancel_event=cancelled)
    assert fingerprint(root) == before
    assert not list((tmp_path / "out").glob("*.mp4"))


def test_cleaned_inference_reuses_verified_video_without_regeneration(tmp_path):
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video
    from sim.web.video_export import export_saved_episode
    root = tmp_path / "previews"
    saved(root, name="inference_demo", kind="inference", success=False)
    video_root = tmp_path / "videos"
    video = export_inference_video(root / "inference_demo", video_root)
    finalize_inference_video(root / "inference_demo", video)
    record = json.loads((root / "manifest.jsonl").read_text())
    before = fingerprint(root)
    result = export_saved_episode(root, record, video_root / "workbench_exports",
                                  video_root=video_root)
    assert result["reused"] is True and result["success"] is False
    assert Path(result["video_path"]).read_bytes() == video.read_bytes()
    assert fingerprint(root) == before and not list((root / "inference_demo").glob("*.png"))
    video.write_bytes(b"broken")
    with pytest.raises((ValueError, RuntimeError)):
        export_saved_episode(root, record, video_root / "workbench_exports",
                              video_root=video_root)


def test_http_job_exports_captured_source_and_offers_safe_view_download(tmp_path):
    saved(tmp_path / "dataset")
    saved(tmp_path / "previews", success=False, offset=10)
    app = make_app(tmp_path)
    client = TestClient(app)
    before = fingerprint(tmp_path / "previews")
    response = client.post("/api/jobs/video-export", json={"episode_id": "preview:demo"})
    assert response.status_code == 200
    job = wait_job(app, response.json()["job_id"])
    assert job["state"] == "done", job
    p = job["progress"]
    assert p["episode_id"] == "demo" and p["source"] == "preview"
    assert p["success"] is False
    view = client.get(p["video_url"])
    download = client.get(p["download_url"])
    assert view.status_code == download.status_code == 200
    assert view.content == download.content == Path(p["video_path"]).read_bytes()
    assert 'inline;' in view.headers["content-disposition"]
    assert 'attachment;' in download.headers["content-disposition"]
    assert fingerprint(tmp_path / "previews") == before
    assert client.get("/api/episode-videos/unrelated.mp4").status_code == 404
    assert client.get("/api/episode-videos/%2E%2E%2Foutside.mp4").status_code == 404
    assert client.post("/api/jobs/video-export", json={"episode_id": "missing"}).status_code == 404
    assert client.post("/api/jobs/video-export", json={}).status_code == 422


def test_http_export_busy_conflict_and_failed_media_are_visible(tmp_path):
    record = saved(tmp_path / "dataset")
    app = make_app(tmp_path)
    client = TestClient(app)
    release = Event()
    running = app.state.jobs.start_callable("video_export", lambda _: release.wait(5))
    try:
        assert client.post("/api/jobs/video-export", json={"episode_id": "dataset:demo"}).status_code == 409
    finally:
        release.set()
        wait_job(app, running["job_id"])
    (tmp_path / "dataset" / record["images"]["wrist"][0]).unlink()
    response = client.post("/api/jobs/video-export", json={"episode_id": "dataset:demo"})
    assert response.status_code == 200
    job = wait_job(app, response.json()["job_id"])
    assert job["state"] == "failed" and not job["progress"].get("video_url")
    assert any("wrist" in line for line in job["output"])


def test_ui_polling_restores_export_result_without_changing_current_selection():
    from sim.web.app import HTML
    node = shutil.which("node")
    assert node, "Node is required for the real inline-JavaScript behavior test"
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    script = script[:script.rindex("toggleFailureMode();initParameters();")]
    harness = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
let elements={};const context={document:{getElementById:id=>elements[id]??=(
 {textContent:'',innerHTML:'',className:'',disabled:false,replaceChildren(){this.innerHTML=''}})},
 fetch:async()=>({ok:true,text:async()=>JSON.stringify([{job_id:'saved-export',
 kind:'video_export',state:'done',progress:{episode_id:'old-selected',frames:3,fps:10,
 video_url:'/api/episode-videos/episode_a.mp4',download_url:'/api/episode-videos/episode_a.mp4?download=true'}}])})};
for(const id of ['episodeVideoStatus','episodeVideoLinks','exportVideoButton'])context.document.getElementById(id);
vm.createContext(context);vm.runInContext(SCRIPT,context);
vm.runInContext("current={episode_id:'new-selection',playback_id:'dataset:new-selection'}",context);
vm.runInContext('refreshJobs()',context).then(()=>{
 assert.match(elements.episodeVideoStatus.textContent,/old-selected.*导出完成/);
 assert.match(elements.episodeVideoLinks.innerHTML,/下载 MP4/);
 assert.equal(elements.exportVideoButton.disabled,false);
 assert.equal(vm.runInContext('current.episode_id',context),'new-selection');
}).catch(e=>{console.error(e);process.exitCode=1});
'''.replace("SCRIPT", json.dumps(script))
    done = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=10)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("damage", ["identity", "rectangles"])
def test_cleaned_video_rejects_other_episode_and_overlapping_views(tmp_path, damage):
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video
    from sim.web.video_export import export_saved_episode
    root, video_root = tmp_path / "replays", tmp_path / "videos"
    saved(root, name="inference_demo", kind="inference")
    video = export_inference_video(root / "inference_demo", video_root)
    finalize_inference_video(root / "inference_demo", video)
    record = json.loads((root / "manifest.jsonl").read_text())
    if damage == "identity":
        other = video_root / "inference_other_five_view.mp4"
        shutil.copyfile(video, other)
        record["video_path"] = str(other)
    else:
        record["video_layout"] = {v: [0, 0, 320, 320] for v in VIEWS}
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="identity|layout"):
        export_saved_episode(root, record, video_root / "out", video_root=video_root)
    assert not list((video_root / "out").glob("*.mp4"))


@pytest.mark.parametrize("cancel", [False, True])
def test_decoder_timeout_and_cancel_stop_the_actual_child_promptly(cancel):
    from sim.web import video_export
    event = Event()
    timer = Timer(.15, event.set) if cancel else None
    if timer:
        timer.start()
    started = time.monotonic()
    try:
        with pytest.raises((TimeoutError, RuntimeError), match="deadline|cancel"):
            video_export._run_media(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cancel_event=event, timeout=.15 if not cancel else 10)
    finally:
        if timer:
            timer.cancel()
    assert time.monotonic() - started < 2


def test_single_generation_selects_captured_latest_key_despite_filter_refresh():
    from sim.web.app import HTML
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    script = script[:script.rindex("toggleFailureMode();initParameters();")]
    harness = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');let elements={};
const context={document:{getElementById:id=>elements[id]??=({value:'',textContent:'',className:''})},
fetch:async()=>({ok:true,text:async()=>JSON.stringify({episode_id:'new-demo'})})};
for(const [id,value]of Object.entries({previewTarget:'3',previewOutcome:'success',previewCount:'1',previewFailureMode:''}))context.document.getElementById(id).value=value;
vm.createContext(context);vm.runInContext(SCRIPT,context);
vm.runInContext("loadEpisodes=async()=>{selectedEpisodeId='dataset:old-demo'}; selectEpisode=async id=>{selectedEpisodeId=id}",context);
vm.runInContext('makePreview()',context).then(()=>{
assert.equal(vm.runInContext('selectedEpisodeId',context),'preview:new-demo');
}).catch(e=>{console.error(e);process.exitCode=1});
'''.replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_pending_post_is_not_replaced_by_polling_previous_completed_export():
    from sim.web.app import HTML
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    script = script[:script.rindex("toggleFailureMode();initParameters();")]
    harness = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');let elements={};
const context={document:{getElementById:id=>elements[id]??=({textContent:'',innerHTML:'',className:'',disabled:false,replaceChildren(){this.innerHTML=''}})},fetch:()=>new Promise(()=>{})};
vm.createContext(context);vm.runInContext(SCRIPT,context);
vm.runInContext("current={episode_id:'new-demo',playback_id:'dataset:new-demo'}; exportEpisodeVideo(); renderVideoExport({job_id:'old',state:'done',progress:{episode_id:'old-demo',frames:3,fps:10,video_url:'/old.mp4',download_url:'/old.mp4'}})",context);
assert.equal(elements.exportVideoButton.disabled,true);
assert.match(elements.episodeVideoStatus.textContent,/new-demo/);
assert.equal(elements.episodeVideoLinks.innerHTML,'');
'''.replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_cancel_interrupts_blocked_encoder_pipe_without_waiting_for_deadline(tmp_path, monkeypatch):
    from sim.web import video_export
    root = tmp_path / "dataset"
    record = saved(root)
    stalled = tmp_path / "stalled_encoder"
    stalled.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    stalled.chmod(0o700)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(stalled))
    monkeypatch.setattr(video_export, "ENCODING_TIMEOUT_SECONDS", .8)
    event = Event()
    timer = Timer(.15, event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="cancel"):
            video_export.export_saved_episode(root, record, tmp_path / "out", cancel_event=event)
    finally:
        timer.cancel()
    assert time.monotonic() - started < .7
    assert not list((tmp_path / "out").glob("*.mp4"))
