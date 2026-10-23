"""Run the local ACT MuJoCo dashboard."""

from __future__ import annotations

import argparse

from ..config import load_config
from pathlib import Path


def diffusion_workspace_config(cfg, project_root, training_config=None):
    from sim.diffusion.config import DEFAULT_DATASET, DEFAULT_OUT, read_launch_config
    launch=read_launch_config(training_config) if training_config else {}
    cfg = cfg.copy()
    cfg.set_path("act.dataset_dir", launch.get("dataset",DEFAULT_DATASET))
    cfg.set_path("web.diffusion_dataset", launch.get("dataset",DEFAULT_DATASET))
    cfg.set_path("web.diffusion_out", launch.get("out",DEFAULT_OUT))
    cfg.set_path("web.inference_policy", "diffusion")
    cfg.set_path("web.diffusion_backbone", launch.get("training",{}).get("backbone","unet"))
    if training_config:
        cfg.set_path("web.diffusion_training_config",str(Path(training_config).resolve()))
    cfg.set_path("web.readonly_dataset", True)
    cfg.set_path("web.inference_video_dir", str(
        Path(project_root)/launch.get("out",DEFAULT_OUT)/"videos"))
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rudra-scene", action="store_true", help="inspect the original Rudra tabletop scene")
    parser.add_argument("--scene-config", help="native Rudra scene YAML for --rudra-scene")
    parser.add_argument("--diffusion-workspace",action="store_true",
                        help="browse original demonstrations read-only; keep new media local")
    parser.add_argument("--diffusion-training-config",help="same frozen launch JSON used by train/resume")
    args = parser.parse_args(argv)
    import uvicorn

    if args.rudra_scene:
        from .rudra_preview import create_rudra_preview_app
        uvicorn.run(create_rudra_preview_app(args.scene_config), host=args.host or "127.0.0.1",
                    port=args.port or 8780)
        return
    if args.scene_config:
        parser.error("--scene-config requires --rudra-scene")
    cfg = load_config(args.config)
    if args.diffusion_workspace or args.diffusion_training_config:
        cfg = diffusion_workspace_config(cfg,Path(__file__).resolve().parents[2],args.diffusion_training_config)
    from .app import create_app

    uvicorn.run(create_app(cfg), host=args.host or str(cfg.web.host),
                port=args.port or int(cfg.web.port))


if __name__ == "__main__":
    main()
