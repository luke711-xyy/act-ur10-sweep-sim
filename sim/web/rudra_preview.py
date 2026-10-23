"""Native Rudra scene inspection using the existing workbench shell and RGB encoder."""
from contextlib import asynccontextmanager
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import copy

from .app import HTML as WORKBENCH_HTML, _png_bytes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party" / "rudra_ur10_mujoco_sim"
DEFAULT_CONFIG = ROOT / "configs" / "rudra-native-scene.yaml"
COMMIT = "436058dde9098c077f14d14d11e633fee9a02dee"


@lru_cache(maxsize=1)
def upstream():
    path = SOURCE / "ur10sim" / "build_scene.py"
    spec = spec_from_file_location("rudra_native_scene", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RudraPreview:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        self.scene = None
        self._images = None
        self.seed_override = None

    def _get_scene(self):
        if self.scene is None:
            builder = upstream()
            cfg = builder.load_config(self.config_path)
            if cfg["robot"]["source"] not in {"home", "joints"}:
                raise ValueError("native preview requires home/joints; external datasets are not loaded")
            width, height = cfg["render"]["width"], cfg["render"]["height"]
            if not (1 <= width <= 1280 and 1 <= height <= 960):
                raise ValueError("preview resolution exceeds the native render buffer")
            if cfg.get("sweep_task",{}).get("enabled"):
                from ..model.rudra_sweep import make_sweep_scene
                if self.seed_override is not None:
                    cfg["sweep_task"]["seed"] = self.seed_override
                self.scene = make_sweep_scene(builder,cfg)
            else:
                self.scene = builder.Scene(cfg)
        return self.scene

    def metadata(self):
        scene = self._get_scene()
        cameras = {}
        for name in ("top", "wrist", "overview"):
            i = scene.cid(name)
            cfg = scene.cfg["cameras"][name]
            mounted = name == "wrist" and cfg.get("mount",{}).get("enabled",False)
            cameras[name] = dict(position=None if mounted else cfg.get("pos"),
                                 pose_source="cad_mount" if mounted else "configured_pose",
                                 configuration=copy.deepcopy(cfg),
                                 world_position=scene.data.cam_xpos[i].tolist(),
                                 world_rotation=scene.data.cam_xmat[i].reshape(3,3).tolist(),
                                 look_at=None if mounted else cfg.get("look_at"),
                                 horizontal_fov=cfg.get("fov_h_deg",69),
                                 vertical_fov=float(scene.model.cam_fovy[i]),
                                 mount=copy.deepcopy(cfg.get("mount")))
        task = scene.task_metadata() if hasattr(scene,"task_metadata") else None
        return dict(source_commit=COMMIT, config=str(self.config_path),
                    mode="rudra_sweep_preview" if task else "native_scene_preview",
                    sweep_task=task, time=float(scene.data.time),
                    table=copy.deepcopy(scene.cfg["table"]),
                    robot=copy.deepcopy(scene.cfg["robot"]), cameras=cameras,
                    render=copy.deepcopy(scene.cfg["render"]))

    def frames(self):
        scene = self._get_scene()
        if self._images is None:
            rendered = scene.render(scene.cfg["render"]["width"], scene.cfg["render"]["height"])
            self._images = {name:_png_bytes(pixels) for name,pixels in rendered.items()}
        return dict(images=self._images, time=float(scene.data.time))

    def reset(self, seed=None):
        if seed is not None:
            self.seed_override = seed
        self.close()
        return self.metadata()

    def close(self):
        if self.scene is not None:
            self.scene.close()
            self.scene = None
        self._images = None


def preview_html():
    # Reuse the workbench's established theme, cards, controls and RGB wire format.
    style = WORKBENCH_HTML.split("<style>",1)[1].split("</style>",1)[0]
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UR10 · Rudra 场景工作台</title><style>''' + style + r'''
.native-layout{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.native-overview,.native-details{grid-column:1/-1}.view img{aspect-ratio:4/3;object-fit:contain}
.scene-head{display:flex;justify-content:space-between;gap:16px;align-items:center}
.row[hidden]{display:none}
.native-label{font-size:13px;color:#b3c9ce}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media(min-width:1100px){.native-layout{grid-template-columns:2fr 1fr}.native-overview{grid-column:1;grid-row:1/3}}
@media(max-width:650px){.native-layout{grid-template-columns:1fr}.native-overview{grid-column:1}}
</style></head><body><main>
<header><div><div class="eyebrow">local embodied-learning lab · native scene</div>
<h1>UR10 · Rudra 场景工作台</h1>
<div class="muted" id="sceneSubtitle">原生桌面、PincOpen 夹爪与相机机位 · 新工作树独立预览</div></div>
<div class="row"><button id="resetScene" onclick="resetScene()">重置并重新加载配置</button>
<button onclick="refreshAll()">刷新画面</button></div></header>
<div class="status"><span class="pill ok">Rudra UR10 CB3</span>
<span class="pill">双相机 + 调试总览</span><span class="pill" id="sceneTime">正在加载</span></div>
<p class="hint" id="taskDescription">相机画面保持原生 640×480，重置会重新读取本地 YAML 并完成物理静置。</p>
<div class="row" id="layoutControls" hidden><label for="layoutSeed">布局种子</label>
<input id="layoutSeed" type="number" min="0" max="4294967295" step="1" value="42">
<button onclick="resetScene(true)">生成该布局</button><span class="muted">相同种子复现相同布局</span></div>
<div class="native-layout">
<section class="panel native-overview"><h2>场景总览 <span class="muted">· overview / 调试相机</span></h2>
<figure class="view"><img id="overviewImage" alt="原生场景总览"><figcaption id="overviewCaption">UR10、木桌板、PincOpen 夹爪与腕部 D435i</figcaption></figure></section>
<section class="panel"><h2><span id="topCameraTitle">俯视相机</span> <span class="muted">· top / D415</span></h2>
<figure class="view"><img id="topImage" alt="桌面相机"><figcaption id="topCameraCaption">保留上游位置、朝向、视场角与外观校准</figcaption></figure></section>
<section class="panel"><h2>腕部相机 <span class="muted">· wrist / D435i RGB</span></h2>
<figure class="view"><img id="wristImage" alt="原生腕部相机"><figcaption>V6_REINFORCED 安装架，RGB 镜头随夹爪运动</figcaption></figure></section>
<section class="panel native-details"><div class="scene-head"><h2>场景配置</h2>
<a href="https://github.com/rudra-8000/ur10-mujoco-sim/tree/436058dde9098c077f14d14d11e633fee9a02dee/ur10sim" target="_blank" rel="noopener">上游源代码 ↗</a></div>
<div class="native-label" id="summary"></div><p class="hint" id="configPath"></p>
<details><summary>查看相机位置、视场角与桌面尺寸</summary><pre id="sceneConfig"></pre></details>
<p class="hint">腕部机位由 CAD 支架与 RGB 镜头位置决定；配置详情同时显示编译后的世界坐标与朝向。</p>
<p class="hint" id="taskScope">当前页面复用原工作台的外观与渲染接口，提供场景检查；策略执行接口尚未接入此新环境。</p></section></div>
<p class="hint" role="status" id="status">正在构建原生场景…</p>
</main><script>
let busy=false;
async function refreshAll(){if(busy)return;busy=true;try{
 const stateResponse=await fetch('/api/scene',{cache:'no-store'});
 if(!stateResponse.ok)throw Error(await stateResponse.text());const state=await stateResponse.json();
 const response=await fetch('/api/frames',{cache:'no-store'});
 if(!response.ok)throw Error(await response.text());const data=await response.json();
 for(const [name,pixels]of Object.entries(data.images))document.getElementById(name+'Image').src='data:image/png;base64,'+pixels;
 document.getElementById('sceneTime').textContent='仿真 '+state.time.toFixed(3)+' s · 预览已静置';
 document.getElementById('summary').textContent=`桌板 ${state.table.size_xy[0]} × ${state.table.size_xy[1]} m · 桌面 Z ${state.table.z} m · 源码 ${state.source_commit.slice(0,7)}`;
 document.getElementById('configPath').textContent='本地配置：'+state.config;
 document.getElementById('sceneConfig').textContent=JSON.stringify({table:state.table,cameras:state.cameras,sweep_task:state.sweep_task},null,2);
 const topCamera=state.cameras.top.configuration;
 document.getElementById('topCameraTitle').textContent=topCamera.view_label||'俯视相机';
 document.getElementById('topCameraCaption').textContent=topCamera.view_description||'保留上游位置、朝向、视场角与外观校准';
 document.getElementById('layoutControls').hidden=!state.sweep_task;
 if(state.sweep_task){
  const task=state.sweep_task;
  document.getElementById('layoutSeed').value=task.seed;
  document.getElementById('sceneSubtitle').textContent='原扫零件任务 · Rudra 木桌、UR10 与双相机';
  document.getElementById('taskDescription').textContent=`收集区、${task.component_count} 个零件与 14 cm 刷板已迁移。复用原示教的簇状布局规则，当前种子 ${task.seed}。刷柄由 PincOpen 两指夹住，刚性随末端运动。`;
  document.getElementById('overviewCaption').textContent='UR10 夹持刷板 · 原尺寸三面收集槽 · 六零件簇状布局';
 }else{document.getElementById('taskDescription').textContent='上游插孔实验场景。相机画面保持原生 640×480。'}
 document.getElementById('status').textContent='三个视角已加载。修改 YAML 后点击“重置并重新加载配置”。';
 }catch(e){document.getElementById('status').textContent='场景加载失败：'+e.message}finally{busy=false}}
async function resetScene(useSeed=false){if(busy)return;busy=true;document.getElementById('resetScene').disabled=true;
 try{const options={method:'POST'};if(useSeed){options.headers={'Content-Type':'application/json'};options.body=JSON.stringify({seed:Number(document.getElementById('layoutSeed').value)})}
 const response=await fetch('/api/reset',options);if(!response.ok)throw Error(await response.text());busy=false;await refreshAll()}
 catch(e){document.getElementById('status').textContent=e.message}
 finally{busy=false;document.getElementById('resetScene').disabled=false}}
refreshAll();
</script></body></html>'''


def create_rudra_preview_app(config_path=None):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field, StrictInt

    class ResetRequest(BaseModel):
        seed: StrictInt | None = Field(default=None,ge=0,le=4294967295)
    viewer = RudraPreview(config_path)

    @asynccontextmanager
    async def lifespan(app):
        yield
        viewer.close()

    app = FastAPI(title="Rudra native scene workbench", lifespan=lifespan)

    @app.get("/",response_class=HTMLResponse)
    async def index():
        return preview_html()

    # All native scene/GL operations stay on the same event-loop thread.
    @app.get("/api/scene")
    async def metadata():
        return viewer.metadata()

    @app.get("/api/frames")
    async def frames():
        return viewer.frames()

    @app.post("/api/reset")
    async def reset(payload: ResetRequest | None = None):
        return viewer.reset(None if payload is None else payload.seed)

    return app
