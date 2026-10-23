"""Run the local ACT MuJoCo dashboard."""

from __future__ import annotations

import argparse

from ..config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rudra-scene", action="store_true", help="inspect the original Rudra tabletop scene")
    parser.add_argument("--scene-config", help="native Rudra scene YAML for --rudra-scene")
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
    from .app import create_app

    uvicorn.run(create_app(cfg), host=args.host or str(cfg.web.host),
                port=args.port or int(cfg.web.port))


if __name__ == "__main__":
    main()
