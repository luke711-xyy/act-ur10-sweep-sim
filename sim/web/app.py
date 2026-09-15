"""Minimal local control panel API; no cloud service or telemetry."""

from __future__ import annotations

import base64
import io
from pathlib import Path


HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>ACT MuJoCo Sweep</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:24px}button{padding:9px 14px;margin-right:8px}img{max-width:48%;margin:8px;background:#222}pre{background:#1c1c1c;padding:12px;white-space:pre-wrap}</style></head>
<body><h1>ACT · UR10 MuJoCo Sweep</h1>
<button onclick="post('/api/reset')">Reset scene</button>
<button onclick="post('/api/health')">Refresh</button>
<div><img id='overhead'><img id='wrist'></div><pre id='status'>loading…</pre>
<script>
async function post(url){await fetch(url,{method:'POST'}); await refresh()}
async function refresh(){let h=await (await fetch('/api/health')).json();status.textContent=JSON.stringify(h,null,2);let f=await (await fetch('/api/frames')).json();overhead.src='data:image/png;base64,'+f.overhead;wrist.src='data:image/png;base64,'+f.wrist}
setInterval(refresh,1000);refresh();
</script></body></html>"""


def _png_bytes(image):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(cfg):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dashboard dependencies are missing: uv pip install -e '.[web]'") from exc
    from ..environments.sweep_env import SweepEnv

    app = FastAPI(title="ACT MuJoCo Sweep")
    app.state.cfg = cfg
    app.state.env = None

    def env():
        if app.state.env is None:
            app.state.env = SweepEnv(cfg, seed=int(cfg.seed))
            app.state.env.reset(seed=int(cfg.seed))
        return app.state.env

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/config")
    def config():
        return cfg.to_dict()

    @app.get("/api/health")
    def health():
        e = env()
        return {"time": e.time, "components": len(e.layout),
                "collected": int(e.collected_mask().sum()),
                "tcp": e.tcp().tolist(), "normal_force": e.normal_force(),
                "model": str(cfg.end_effector.type)}

    @app.post("/api/reset")
    def reset():
        if app.state.env is not None:
            app.state.env.close()
        app.state.env = SweepEnv(cfg, seed=int(cfg.seed))
        app.state.env.reset(seed=int(cfg.seed))
        return {"ok": True}

    @app.get("/api/frames")
    def frames():
        e = env()
        size = tuple(int(v) for v in cfg.act.image_size)
        return {"overhead": _png_bytes(e.render_rgb("overhead_cam", size=size)),
                "wrist": _png_bytes(e.render_wrist_rgb(size=size))}

    return app
