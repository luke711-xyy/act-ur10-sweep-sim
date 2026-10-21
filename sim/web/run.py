"""Run the local ACT MuJoCo dashboard."""

from __future__ import annotations

import argparse

from ..config import load_config
from pathlib import Path


def diffusion_workspace_config(cfg, project_root):
    from sim.diffusion.config import DEFAULT_DATASET
    cfg = cfg.copy()
    cfg.set_path("act.dataset_dir", DEFAULT_DATASET)
    cfg.set_path("web.readonly_dataset", True)
    cfg.set_path("web.inference_video_dir", str(
        Path(project_root)/"runs/diffusion_policy_mvp/videos"))
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--diffusion-workspace",action="store_true",
                        help="browse original demonstrations read-only; keep new media local")
    args = parser.parse_args(argv)
    import uvicorn

    cfg = load_config(args.config)
    if args.diffusion_workspace:
        cfg = diffusion_workspace_config(cfg,Path(__file__).resolve().parents[2])
    from .app import create_app

    uvicorn.run(create_app(cfg), host=args.host or str(cfg.web.host),
                port=args.port or int(cfg.web.port))


if __name__ == "__main__":
    main()
