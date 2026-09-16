"""Local MuJoCo robot-learning workbench."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACT MuJoCo Sweep Workbench</title>
<style>
:root{--bg:#10151b;--panel:#18222b;--line:#30434f;--text:#eaf1f4;--muted:#94aab4;--accent:#62d6b0}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#20373b 0,#10151b 42%);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:22px}h1{margin:0;font-size:27px;letter-spacing:-.03em}h2{font-size:15px;margin:0 0 14px;color:#fff}.muted,.hint{color:var(--muted)}.eyebrow{color:var(--accent);text-transform:uppercase;font-size:11px;letter-spacing:.14em}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{background:linear-gradient(145deg,rgba(30,43,53,.96),rgba(20,29,36,.96));border:1px solid var(--line);border-radius:13px;padding:16px;box-shadow:0 12px 28px #0002}.simulation{grid-column:span 8}.demonstrations{grid-column:span 4}.training,.inference,.parameters{grid-column:span 4}.wide{grid-column:1/-1}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.view{margin:0;background:#0e1419;border:1px solid #263742;border-radius:9px;overflow:hidden}.view img{display:block;width:100%;aspect-ratio:1.15;object-fit:cover;background:#0a0f12}.view figcaption{padding:8px 10px;color:var(--muted);font-size:12px}.tag{float:right;color:var(--accent);font-size:10px;text-transform:uppercase}
.demo-views{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:12px 0 8px}.demo-view{margin:0;background:#0e1419;border:1px solid #263742;border-radius:7px;overflow:hidden}.demo-view img{display:block;width:100%;aspect-ratio:1.1;object-fit:cover;background:#0a0f12}.demo-view figcaption{padding:5px 6px;color:var(--muted);font-size:10px}
button,select,input{font:inherit;color:var(--text);background:#23343d;border:1px solid #3d5863;border-radius:7px;padding:8px 10px}button{cursor:pointer}button:hover{border-color:var(--accent);color:#fff}button.primary{background:#237c6b;border-color:#4fc19e}button.danger{border-color:#915454;color:#ffb2b2}input,select{width:100%;margin:4px 0 9px}label{display:block;color:var(--muted);font-size:12px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.row>*{flex:0 0 auto}.meta{background:#0e1419;border-radius:8px;padding:10px;min-height:74px;color:#a9c0c9;white-space:pre-wrap}.job{background:#0e1419;border-radius:8px;padding:9px;margin-top:10px;font-size:12px}.job strong{color:var(--accent)}.job pre{max-height:100px;overflow:auto;margin:7px 0 0;color:#b9c8ce}.status{margin-top:13px;display:flex;gap:9px;flex-wrap:wrap}.pill{border:1px solid var(--line);border-radius:99px;padding:5px 9px;color:var(--muted);font-size:12px}.ok{color:var(--accent)}.bad{color:#f08f8f}a{color:var(--accent)}
@media(max-width:1000px){.simulation,.demonstrations,.training,.inference,.parameters{grid-column:1/-1}}@media(max-width:650px){.views{grid-template-columns:1fr}}
</style></head><body><main>
<header><div><div class="eyebrow">local embodied-learning lab</div><h1>ACT · UR10 MuJoCo Sweep Workbench</h1><div class="muted">Observe the simulator, inspect expert trajectories, then launch reproducible local ACT jobs.</div></div><div class="row"><button onclick="resetScene()">Reset scene</button><button onclick="refreshAll()">Refresh</button></div></header>
<div class="grid">
<section class="panel simulation"><h2>Simulation <span class="muted">· live MuJoCo state</span></h2><div class="views">
<figure class="view"><img id="overheadImage"><figcaption>Perception / overhead</figcaption></figure>
<figure class="view"><img id="wristImage"><figcaption>ACT input / wrist</figcaption></figure>
<figure class="view"><img id="inspectionImage"><figcaption>斜前上方观察 <span class="tag">inspection-only</span></figcaption></figure>
</div><div id="healthPills" class="status"></div></section>
<section class="panel demonstrations"><h2>Demonstrations <span class="muted">· episode playback</span></h2>
<label for="episodeSelect">Saved or preview episode</label><select id="episodeSelect" onchange="selectEpisode(this.value)"></select>
<div class="row"><button class="primary" onclick="makePreview()">Generate expert preview</button><button onclick="togglePlay()" id="playButton">Play</button><button onclick="stepFrame(-1)">‹</button><button onclick="stepFrame(1)">›</button></div>
<div class="demo-views"><figure class="demo-view"><img id="demoOverheadImage"><figcaption>overhead</figcaption></figure><figure class="demo-view"><img id="demoWristImage"><figcaption>wrist</figcaption></figure><figure class="demo-view"><img id="demoInspectionImage"><figcaption>inspection</figcaption></figure></div>
<input id="frameSlider" type="range" min="0" max="0" value="0" oninput="seekFrame(this.value)"><div id="episodeMeta" class="meta">No episode loaded.</div>
</section>
<section class="panel training"><h2>Training <span class="muted">· local ACT</span></h2>
<label>Dataset directory<input id="trainDataset" value="runs/act_dataset"></label><label>Output directory<input id="trainOut" value="runs/act_model"></label><label>Steps (optional)<input id="trainSteps" type="number" min="1" placeholder="use config default"></label>
<div class="row"><button class="primary" onclick="startTraining()">Start training job</button><button class="danger" onclick="stopLatest('train')">Stop</button></div><div id="trainJob" class="job">No training job.</div>
</section>
<section class="panel inference"><h2>Inference <span class="muted">· MuJoCo rollout</span></h2>
<label>Model checkpoint<input id="inferModel" value="runs/act_model/act_policy.pt"></label><label>Seed<input id="inferSeed" type="number" value="0"></label>
<div class="row"><button class="primary" onclick="startInference()">Run inference</button><button class="danger" onclick="stopLatest('inference')">Stop</button></div><div id="inferJob" class="job">No inference job.</div>
</section>
<section class="panel parameters"><h2>Parameters <span class="muted">· explicit, editable knobs</span></h2>
<div class="grid"><div style="grid-column:span 6"><label>Desired force (N)<input id="pForce" type="number" step=".05" min="0" max="5"></label><label>Sweep speed (m/s)<input id="pSpeed" type="number" step=".01" min=".01" max="1"></label><label>Components<input id="pCount" type="number" min="1" max="10"></label></div><div style="grid-column:span 6"><label>Safe max force (N)<input id="pSafe" type="number" step=".05" min=".1" max="20"></label><label>Episode max time (s)<input id="pTime" type="number" step="1" min="1" max="600"></label><label>ACT batch size<input id="pBatch" type="number" min="1" max="256"></label></div></div>
<button class="primary" onclick="applyParameters()">Apply and reset scene</button><div id="parameterStatus" class="hint"></div>
</section>
<section class="panel wide"><h2>Dataset tooling <span class="muted">· reference integration</span></h2><div class="row"><span class="hint">For detailed episode videos, action plots, filtering, and annotations:</span><a href="https://huggingface.co/spaces/lerobot/visualize_dataset" target="_blank">Open LeRobot Dataset Visualizer ↗</a><span class="hint">or run <code>lerobot-dataset-viz --repo-id ... --mode local</code>.</span></div><div id="globalStatus" class="hint" style="margin-top:10px"></div></section>
</div></main><script>
const $=id=>document.getElementById(id);let episodes=[];let current=null;let selectedEpisodeId=null;let timer=null;
function image(id,b64){$(id).src=b64?'data:image/png;base64,'+b64:''}
async function json(url,opts){const r=await fetch(url,opts);const data=await r.json();if(!r.ok)throw new Error(data.detail||r.statusText);return data}
async function refreshAll(){try{const [h,f]=await Promise.all([json('/api/health'),json('/api/frames')]);image('overheadImage',f.overhead);image('wristImage',f.wrist);image('inspectionImage',f.inspection);$('healthPills').innerHTML=`<span class="pill ok">sim ${h.time.toFixed(2)} s</span><span class="pill">components ${h.collected}/${h.components}</span><span class="pill">Fz ${h.normal_force.toFixed(2)} N</span><span class="pill">${h.model}</span>`;await loadEpisodes();await refreshJobs()}catch(e){$('globalStatus').textContent='Dashboard error: '+e.message;$('globalStatus').className='bad'}}
async function loadEpisodes(){const data=await json('/api/episodes');episodes=data;const select=$('episodeSelect');const old=select.value;select.innerHTML=data.length?data.map(e=>`<option value="${e.episode_id}">${e.preview?'preview':'dataset'} · ${e.episode_id} · ${e.success?'success':'failure'}</option>`).join(''):'<option value="">No episodes yet</option>';if(data.length){const preferred=selectedEpisodeId||old;const id=data.some(e=>e.episode_id===preferred)?preferred:data[data.length-1].episode_id;select.value=id;if(id!==current?.episode_id)await selectEpisode(id)}}
async function selectEpisode(id){if(!id)return;selectedEpisodeId=id;current=episodes.find(e=>e.episode_id===id)||null;if(!current)return;$('frameSlider').max=Math.max(0,current.length-1);$('frameSlider').value=0;await seekFrame(0)}
async function seekFrame(index){if(!current)return;index=Math.max(0,Math.min(Number(index),current.length-1));$('frameSlider').value=index;try{const f=await json(`/api/episodes/${encodeURIComponent(current.episode_id)}/frame/${index}`);image('demoOverheadImage',f.overhead);image('demoWristImage',f.wrist);image('demoInspectionImage',f.inspection);$('episodeMeta').textContent=`${current.episode_id}\nframe ${index+1}/${current.length} · seed ${current.seed??'?'} · ${current.success?'SUCCESS':'FAILURE'}${current.failure_reason?'\nreason: '+current.failure_reason:''}`}catch(e){$('episodeMeta').textContent=e.message}}
function stepFrame(delta){seekFrame(Number($('frameSlider').value)+delta)}
function togglePlay(){if(timer){clearInterval(timer);timer=null;$('playButton').textContent='Play';return}if(!current)return;$('playButton').textContent='Pause';timer=setInterval(()=>{let n=Number($('frameSlider').value)+1;if(n>=current.length){clearInterval(timer);timer=null;$('playButton').textContent='Play';return}seekFrame(n)},120)}
async function makePreview(){try{$('globalStatus').textContent='Generating a fresh expert preview in MuJoCo…';const e=await json('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:Date.now()%100000,count:null})});selectedEpisodeId=e.episode_id;await loadEpisodes();$('episodeSelect').value=e.episode_id;await selectEpisode(e.episode_id);$('globalStatus').textContent=`Preview ${e.episode_id} ready; formal dataset unchanged.`}catch(e){$('globalStatus').textContent='Preview failed: '+e.message}}
async function resetScene(){await json('/api/reset',{method:'POST'});await refreshAll()}
function fillParameters(c){$('pForce').value=c.controller.desired_force;$('pSpeed').value=c.controller.sweep_speed;$('pSafe').value=c.controller.safe_max_force;$('pTime').value=c.episode.max_time;$('pCount').value=c.components.count;$('pBatch').value=c.act.batch_size}
async function initParameters(){try{fillParameters(await json('/api/config'))}catch(e){$('parameterStatus').textContent=e.message}}
async function applyParameters(){const values={"controller.desired_force":Number($('pForce').value),"controller.sweep_speed":Number($('pSpeed').value),"controller.safe_max_force":Number($('pSafe').value),"episode.max_time":Number($('pTime').value),"components.count":Number($('pCount').value),"act.batch_size":Number($('pBatch').value)};try{await json('/api/config/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});$('parameterStatus').textContent='Applied and saved to runs/workbench/config.yaml.';await refreshAll()}catch(e){$('parameterStatus').textContent='Rejected: '+e.message}}
async function startTraining(){try{const body={dataset:$('trainDataset').value,out:$('trainOut').value};if($('trainSteps').value)body.steps=Number($('trainSteps').value);await json('/api/jobs/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await refreshJobs()}catch(e){$('trainJob').textContent=e.message}}
async function startInference(){try{await json('/api/jobs/inference',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:$('inferModel').value,seed:Number($('inferSeed').value)})});await refreshJobs()}catch(e){$('inferJob').textContent=e.message}}
async function refreshJobs(){const jobs=await json('/api/jobs');for(const kind of ['train','inference']){const j=jobs.find(x=>x.kind===kind);if(j){$(kind==='train'?'trainJob':'inferJob').innerHTML=`<strong>${j.state}</strong> · ${j.job_id}<pre>${(j.output||[]).join('\n')}</pre>`}}}
async function stopLatest(kind){const jobs=await json('/api/jobs');const j=jobs.find(x=>x.kind===kind&&['starting','running'].includes(x.state));if(j){await json(`/api/jobs/${j.job_id}/stop`,{method:'POST'});await refreshJobs()}}
initParameters();refreshAll();setInterval(refreshAll,3000);
</script></body></html>"""


def _png_bytes(image):
    from PIL import Image

    if image is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(cfg):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dashboard dependencies are missing: uv pip install -e '.[web]'") from exc
    from ..config import save_config
    from ..environments.sweep_env import SweepEnv
    from .jobs import JobRegistry, build_inference_argv, build_train_argv
    from .workbench import EpisodeNotFound, FrameNotFound, WorkbenchState

    project_root = Path(__file__).resolve().parents[2]
    app = FastAPI(title="ACT MuJoCo Sweep Workbench")
    app.state.cfg = cfg
    app.state.env = None
    dataset_root = Path(str(cfg.act.dataset_dir))
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    app.state.workbench = WorkbenchState(
        cfg, dataset_root=dataset_root,
        preview_root=project_root / "runs" / "workbench_previews")
    app.state.jobs = JobRegistry(project_root)

    def env():
        if app.state.env is None:
            current = app.state.cfg
            app.state.env = SweepEnv(current, seed=int(current.seed))
            app.state.env.reset(seed=int(current.seed))
        return app.state.env

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/config")
    def config():
        return {**app.state.cfg.to_dict(), "_workbench_allowed": [
            "controller.desired_force", "controller.sweep_speed",
            "controller.safe_max_force", "episode.max_time",
            "components.count", "act.batch_size"]}

    @app.get("/api/health")
    def health():
        e = env()
        current = app.state.cfg
        return {"time": e.time, "components": len(e.layout),
                "collected": int(e.collected_mask().sum()),
                "tcp": e.tcp().tolist(), "normal_force": e.normal_force(),
                "model": str(current.end_effector.type),
                "camera_roles": {"overhead": "perception", "wrist": "act_input",
                                 "inspection": "inspection_only"}}

    @app.post("/api/reset")
    def reset():
        if app.state.env is not None:
            app.state.env.close()
        app.state.env = None
        env()
        return {"ok": True}

    @app.get("/api/frames")
    def frames():
        e = env()
        size = tuple(int(v) for v in app.state.cfg.act.image_size)
        return {"overhead": _png_bytes(e.render_rgb("overhead_cam", size=size)),
                "wrist": _png_bytes(e.render_wrist_rgb(size=size)),
                "inspection": _png_bytes(e.render_rgb("inspection_cam", size=size)),
                "camera_roles": {"overhead": "perception", "wrist": "act_input",
                                 "inspection": "inspection_only"}}

    @app.get("/api/episodes")
    def episodes():
        return app.state.workbench.list_episodes()

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str):
        try:
            return app.state.workbench.episode_metadata(episode_id)
        except (EpisodeNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/episodes/{episode_id}/frame/{frame_index}")
    def episode_frame(episode_id: str, frame_index: int):
        try:
            return {key: _png_bytes(value) for key, value in
                    app.state.workbench.load_episode_frame(episode_id, frame_index).items()}
        except (EpisodeNotFound, FrameNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/preview/{preview_id}")
    def preview_metadata(preview_id: str):
        try:
            metadata = app.state.workbench.episode_metadata(preview_id)
            if not metadata.get("preview"):
                raise EpisodeNotFound(preview_id)
            return metadata
        except (EpisodeNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/preview/{preview_id}/frame/{frame_index}")
    def preview_frame(preview_id: str, frame_index: int):
        try:
            metadata = app.state.workbench.episode_metadata(preview_id)
            if not metadata.get("preview"):
                raise EpisodeNotFound(preview_id)
            return {key: _png_bytes(value) for key, value in
                    app.state.workbench.load_episode_frame(preview_id, frame_index).items()}
        except (EpisodeNotFound, FrameNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/preview")
    def preview(payload: dict):
        seed = int(payload.get("seed", app.state.cfg.seed))
        count = payload.get("count")
        if count is not None and not 1 <= int(count) <= 10:
            raise HTTPException(status_code=422, detail="count must be between 1 and 10")
        return app.state.workbench.build_preview(seed, int(count) if count is not None else None)

    @app.get("/api/jobs")
    def jobs():
        return app.state.jobs.list()

    @app.post("/api/jobs/train")
    def train(payload: dict):
        argv = build_train_argv(project_root, payload.get("config", str(app.state.cfg.get_path("_source_config"))),
                                payload.get("dataset", str(app.state.cfg.act.dataset_dir)),
                                payload.get("out", str(app.state.cfg.act.model_dir)), payload.get("steps"))
        try:
            return app.state.jobs.start("train", argv)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/inference")
    def inference(payload: dict):
        argv = build_inference_argv(project_root,
                                    payload.get("config", str(app.state.cfg.get_path("_source_config"))),
                                    payload.get("seed", app.state.cfg.seed), payload.get("model"))
        try:
            return app.state.jobs.start("inference", argv)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        try:
            return app.state.jobs.stop(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    allowed = {
        "controller.desired_force": (0.0, 5.0),
        "controller.sweep_speed": (0.01, 1.0),
        "controller.safe_max_force": (0.1, 20.0),
        "episode.max_time": (1.0, 600.0),
        "components.count": (1, 10),
        "act.batch_size": (1, 256),
    }

    @app.post("/api/config/apply")
    def apply_config(payload: dict):
        unknown = sorted(set(payload) - set(allowed))
        if unknown:
            raise HTTPException(status_code=422, detail=f"unsupported parameter(s): {unknown}")
        updated = app.state.cfg.copy()
        for path, (low, high) in allowed.items():
            if path not in payload:
                continue
            try:
                value = float(payload[path])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{path} must be numeric") from exc
            if not math.isfinite(value) or value < low or value > high or (path == "components.count" and value % 1):
                raise HTTPException(status_code=422, detail=f"{path} outside [{low}, {high}]")
            updated.set_path(path, int(value) if path in {"components.count", "act.batch_size"} else value)
        target = project_root / "runs" / "workbench" / "config.yaml"
        updated.set_path("_source_config", str(target))
        save_config(updated, str(target))
        app.state.cfg = updated
        app.state.workbench.cfg = updated
        reset()
        return {"ok": True, "config_path": str(target), "config": updated.to_dict()}

    return app
