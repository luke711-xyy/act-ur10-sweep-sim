"""Batch experiments with paired seeds across planners.

Every ``(n_components, seed)`` pair produces the *same* initial layout for every
planner, so planners can be compared on identical scenes.  The low-level
controller, target force, velocity limits, workspace and episode budget are
taken from one config and are therefore identical by construction.

Examples
--------
    python -m sim.run_experiment --planners fixed visual_greedy \
        --counts 1 2 3 5 10 --episodes 20
    python -m sim.run_experiment --planners fixed visual_greedy --counts 3 \
        --episodes 10 --perception vision --jobs 4
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from typing import List, Optional

from .config import load_config, save_config
from .logging_utils import config_hash, make_run_dir, save_csv, save_json
from .metrics import aggregate, paired_comparison
from .plotting import plot_experiment, plot_force_quality, plot_paired_delta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch of paired episodes.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--planners", nargs="+", default=["fixed", "visual_greedy"],
                        choices=["fixed", "global_sweep", "visual_greedy"])
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    parser.add_argument("--episodes", type=int, default=20, help="episodes per (planner, count)")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--perception", default=None, choices=["ground_truth", "vision"])
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--spawn-mode", default=None, choices=["uniform", "cluster"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--save-traces", action="store_true",
                        help="also store the dense per-episode control trace")
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    return parser


def _run_one(job):
    cfg_dict, seed, count, planner, perception, out_dir, save_traces = job
    from .config import Config
    from .environments.episode import run_episode
    from .logging_utils import save_episode_bundle

    cfg = Config(cfg_dict)
    cfg.set_path("components.count", int(count))
    result = run_episode(cfg, seed=int(seed), planner_name=planner,
                         perception_name=perception, keep_images=False, verbose=False)
    row = result.metrics.to_dict()
    row["config_hash"] = config_hash(cfg_dict)
    if save_traces:
        episode_dir = os.path.join(out_dir, "episodes",
                                   f"{planner}_n{count}_s{seed}")
        save_episode_bundle(result, episode_dir)
    return row


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    if args.perception is not None:
        cfg.set_path("perception.backend", args.perception)
    if args.geometry is not None:
        cfg.set_path("components.geometry", args.geometry)
    if args.spawn_mode is not None:
        cfg.set_path("components.spawn_mode", args.spawn_mode)
    perception = str(cfg.perception.backend)

    out_dir = args.out or make_run_dir(str(cfg.logging.out_dir), tag="experiment")
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    cfg_dict = cfg.to_dict()
    jobs = []
    for count, episode in itertools.product(args.counts, range(args.episodes)):
        seed = int(args.seed_base) + episode
        for planner in args.planners:
            jobs.append((cfg_dict, seed, count, planner, perception, out_dir,
                         bool(args.save_traces)))

    print(f"running {len(jobs)} episodes "
          f"({len(args.planners)} planners x {len(args.counts)} counts x "
          f"{args.episodes} seeds), perception={perception}")
    t0 = time.time()
    rows: List[dict] = []
    if int(args.jobs) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(int(args.jobs)) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_one, jobs), start=1):
                rows.append(row)
                _progress(i, len(jobs), row, t0)
    else:
        for i, job in enumerate(jobs, start=1):
            row = _run_one(job)
            rows.append(row)
            _progress(i, len(jobs), row, t0)

    rows.sort(key=lambda r: (r["n_components"], r["seed"], r["planner"]))
    save_csv(rows, os.path.join(out_dir, "episodes.csv"))
    summary = aggregate(rows, group_keys=("planner", "n_components"))
    save_csv(summary, os.path.join(out_dir, "summary.csv"))
    save_json({"rows": len(rows), "planners": args.planners, "counts": args.counts,
               "episodes": args.episodes, "perception": perception,
               "wall_time_s": time.time() - t0},
              os.path.join(out_dir, "experiment.json"))

    if len(args.planners) >= 2:
        for metric in ("collection_rate", "completion_time", "n_strokes", "path_length"):
            pairs = paired_comparison(rows, args.planners[0], args.planners[1], metric)
            save_csv(pairs, os.path.join(out_dir, f"paired_{metric}.csv"))
            plot_paired_delta(pairs, metric, out_dir)

    plot_experiment(summary, out_dir)
    plot_force_quality(summary, out_dir)
    _print_summary(summary)
    print(f"\n  results written to {out_dir}")
    return 0


def _progress(i: int, total: int, row: dict, t0: float) -> None:
    elapsed = time.time() - t0
    eta = elapsed / i * (total - i)
    print(f"  [{i:4d}/{total}] {row['planner']:13s} n={row['n_components']:2d} "
          f"seed={row['seed']:3d}  rate={row['collection_rate']:.2f} "
          f"strokes={row['n_strokes']:2d} t={row['completion_time']:6.1f}s "
          f"| elapsed {elapsed/60:5.1f} min, eta {eta/60:5.1f} min", flush=True)


def _print_summary(summary) -> None:
    print("\n  planner        n   episodes  success  rate   strokes  time[s]  path[m]  "
          "peakF[N]  rmsF[N]")
    for row in summary:
        print(f"  {row['planner']:<13s} {row['n_components']:<3} {row['episodes']:<8} "
              f"{row['success_rate']:<8.2f} {row['collection_rate_mean']:<6.2f} "
              f"{row['n_strokes_mean']:<8.1f} {row['completion_time_mean']:<8.1f} "
              f"{row['path_length_mean']:<8.2f} {row['peak_normal_force_mean']:<9.2f} "
              f"{row['rms_force_error_mean']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
