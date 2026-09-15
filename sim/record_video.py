"""Record a multi-camera video of one episode.

    # default two-camera view: the perception camera + a top-down overhead shot
    python -m sim.record_video --planner visual_greedy --num-components 5 --seed 0

    # three panes, including a camera that tracks the tool
    python -m sim.record_video --planner fixed --num-components 10 --seed 1 \
        --cameras scene_cam side_cam follow_cam

    # side by side comparison of two planners on the SAME layout
    python -m sim.record_video --compare fixed visual_greedy --num-components 5 --seed 0

    # 2-D preview instead of rendered cameras -- no OpenGL, works anywhere
    python -m sim.record_video --preview --planner visual_greedy --speed 3
    python -m sim.record_video --preview --speed 8 \
        --compare fixed global_sweep visual_greedy --num-components 5 --seed 2

Available cameras: ``scene_cam`` (the fixed perception camera -- what the planner
actually sees), plus whatever is defined under ``video.extra_cameras`` in the
config: ``overhead_cam``, ``side_cam``, ``follow_cam``.

Rendering needs a GL backend.  On macOS the default works; on a headless Linux
box use ``MUJOCO_GL=egl`` (or ``osmesa`` without a GPU).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .config import load_config, save_config
from .logging_utils import make_run_dir, print_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record an episode as video.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--planner", default="visual_greedy",
                        choices=["fixed", "global_sweep", "visual_greedy"])
    parser.add_argument("--compare", nargs="+", default=None,
                        help="record one video per planner on the same layout")
    parser.add_argument("--perception", default="ground_truth",
                        choices=["ground_truth", "vision"])
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--spawn-mode", default=None, choices=["uniform", "cluster"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="camera names, left to right (default: video.cameras)")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--every-n", type=int, default=None,
                        help="render one frame every N control steps")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--gif", action="store_true", help="write a GIF instead of an MP4")
    parser.add_argument("--preview", action="store_true",
                        help="2-D Matplotlib animation instead of rendered cameras "
                             "(no OpenGL needed; with --compare, one shared-clock figure)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="preview playback speed relative to simulated time")
    parser.add_argument("--track-every", type=int, default=4,
                        help="preview: record component positions every N control steps")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    return parser


def preview_one(cfg, planner: str, args):
    """Run an episode with component tracking, for the 2-D preview."""
    from .environments.episode import run_episode

    return run_episode(cfg, seed=int(args.seed), planner_name=planner,
                       perception_name=str(cfg.perception.backend),
                       verbose=True, track_every=int(args.track_every))


def record_one(cfg, planner: str, args, out_path: str):
    from .environments.episode import run_episode
    from .video import EpisodeRecorder

    recorder = EpisodeRecorder(
        cfg, out_path,
        cameras=args.cameras, fps=args.fps, every_n=args.every_n,
        hud=not args.no_hud,
        size=(args.width or cfg.get_path("video.width", 640),
              args.height or cfg.get_path("video.height", 480)),
    )

    from .environments.sweep_env import SweepEnv

    env = SweepEnv(cfg, seed=int(args.seed))
    env.reset(seed=int(args.seed))
    recorder.attach(env)
    recorder.capture(env, {"t": 0.0, "phase": "START", "stroke": -1,
                           "collected": 0, "total": len(env.layout),
                           "planner": planner}, force=True)
    try:
        result = run_episode(cfg, seed=int(args.seed), planner_name=planner,
                             perception_name=str(cfg.perception.backend),
                             env=env, verbose=True, recorder=recorder)
        for _ in range(int(recorder.fps)):        # hold the last frame for ~1 s
            recorder.capture(env, {"t": env.time, "phase": result.metrics.success and "SUCCESS"
                                   or "FAILURE", "stroke": -1,
                                   "collected": int(env.collected_mask().sum()),
                                   "total": len(env.layout), "planner": planner}, force=True)
    finally:
        path = recorder.close()
        env.close()
    return result, path, recorder.n_frames


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg.set_path("components.count", int(args.num_components))
    cfg.set_path("perception.backend", args.perception)
    if args.geometry:
        cfg.set_path("components.geometry", args.geometry)
    if args.spawn_mode:
        cfg.set_path("components.spawn_mode", args.spawn_mode)

    out_dir = args.out if (args.out and os.path.isdir(args.out)) else None
    if args.out and not os.path.isdir(args.out) and os.path.splitext(args.out)[1]:
        base_dir, explicit = os.path.dirname(args.out) or ".", args.out
    else:
        base_dir = out_dir or make_run_dir(str(cfg.logging.out_dir), tag="video")
        explicit = None
    os.makedirs(base_dir, exist_ok=True)
    save_config(cfg, os.path.join(base_dir, "config.yaml"))

    planners: List[str] = list(args.compare) if args.compare else [args.planner]
    extension = "gif" if args.gif else str(cfg.get_path("video.format", "mp4"))

    if args.preview:
        from .preview import animate_comparison, animate_episode

        results = []
        for planner in planners:
            print(f"\n  running {planner} for preview")
            result = preview_one(cfg, planner, args)
            print_metrics(result.metrics)
            results.append(result)
        # Episodes here always come from the real simulator, so the provenance
        # banner says so. Pass a different string only when the episode did not.
        source = "MuJoCo"
        if len(results) > 1:
            path = explicit or os.path.join(
                base_dir, f"compare_n{args.num_components}_seed{args.seed}.{extension}")
            written = animate_comparison(results, path, speed=max(args.speed, 1.0),
                                         labels=planners, source=source)
            print(f"  comparison preview written to {written}")
        else:
            path = explicit or os.path.join(
                base_dir,
                f"{planners[0]}_n{args.num_components}_seed{args.seed}_preview.{extension}")
            written = animate_episode(results[0], path, speed=max(args.speed, 1.0),
                                      source=source)
            print(f"  preview written to {written}")
        return 0

    for planner in planners:
        path = explicit if (explicit and len(planners) == 1) else os.path.join(
            base_dir, f"{planner}_n{args.num_components}_seed{args.seed}.{extension}")
        print(f"\n  recording {planner} -> {path}")
        result, written, frames = record_one(cfg, planner, args, path)
        print_metrics(result.metrics)
        print(f"  {frames} frames written to {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
