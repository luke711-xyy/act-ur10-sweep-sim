"""Run the local ACT MuJoCo dashboard."""

from __future__ import annotations

import argparse

from ..config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    import uvicorn

    cfg = load_config(args.config)
    from .app import create_app

    uvicorn.run(create_app(cfg), host=args.host or str(cfg.web.host),
                port=args.port or int(cfg.web.port))


if __name__ == "__main__":
    main()
