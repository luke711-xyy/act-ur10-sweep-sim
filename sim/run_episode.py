"""Run a single episode.

Examples
--------
    python -m sim.run_episode --planner fixed --num-components 1 --seed 0
    python -m sim.run_episode --planner visual_greedy --num-components 5 --seed 0
    python -m sim.run_episode --planner visual_greedy --perception vision \
        --num-components 5 --seed 3 --save-observations
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_config, save_config
from .logging_utils import make_run_dir, print_metrics, save_episode_bundle
from .plotting import plot_episode, plot_observation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one sweeping episode.")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--planner", default=None,
                        choices=["fixed", "global_sweep", "visual_greedy"])
    parser.add_argument("--perception", default=None, choices=["ground_truth", "vision"])
    parser.add_argument("--num-components", type=int, default=None)
    parser.add_argument("--geometry", default=None,
                        choices=["hex_nut", "cylinder", "screw", "bolt", "washer", "mixed"])
    parser.add_argument("--spawn-mode", default=None, choices=["uniform", "cluster"],
                        help="uniform scatter (planner comparison) or one loose cluster")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output directory (default: runs/episode-*)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-observations", action="store_true",
                        help="also save the camera frame / mask / occupancy grid per stroke")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                        help="override any config entry, e.g. --set controller.desired_force=5")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    if args.num_components is not None:
        cfg.set_path("components.count", int(args.num_components))
    if args.geometry is not None:
        cfg.set_path("components.geometry", args.geometry)
    if args.spawn_mode is not None:
        cfg.set_path("components.spawn_mode", args.spawn_mode)
    if args.planner is not None:
        cfg.set_path("planner.name", args.planner)
    if args.perception is not None:
        cfg.set_path("perception.backend", args.perception)
    cfg.set_path("seed", int(args.seed))

    from .environments.episode import run_episode  # imports MuJoCo lazily

    out_dir = args.out or make_run_dir(str(cfg.logging.out_dir), tag="episode")
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    result = run_episode(
        cfg,
        seed=int(args.seed),
        planner_name=str(cfg.planner.name),
        perception_name=str(cfg.perception.backend),
        keep_images=bool(args.save_observations),
        verbose=not args.quiet,
    )

    save_episode_bundle(result, out_dir)
    if not args.no_plots and bool(cfg.logging.save_plots):
        paths = plot_episode(result, out_dir)
        if args.save_observations:
            obs_dir = os.path.join(out_dir, "observations")
            for i, obs in enumerate(result.observations):
                plot_observation(obs, result.config, os.path.join(obs_dir, f"obs_{i:03d}.png"))
        if not args.quiet:
            for path in paths:
                print(f"  plot: {path}")

    if not args.quiet:
        print_metrics(result.metrics)
        print(f"  results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
