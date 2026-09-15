"""Desired-force sweep: where does a single fixed F_z* stop working?

The hybrid controller regulates the normal force to one fixed setpoint. That
setpoint is an engineering parameter, and this script is how it gets chosen: it
sweeps ``controller.desired_force`` against component geometry and reports the
two failure modes that bracket the usable range.

* **too little force** -- the tip *rides over* flat parts (thin washers) instead
  of pushing them: ``n_ride_over`` rises, collection rate falls
* **too much force** -- parts *jam* against the tip and the tool loads up in
  plane: ``n_jam_events`` and the touchdown overshoot rise, and parts get
  ejected out of the workspace

A good F_z* is the widest plateau where both stay near zero for every geometry.

    python -m sim.run_force_sweep --forces 1 2 3 5 8 12 \
        --geometries washer hex_nut bolt --episodes 5
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from typing import List

import numpy as np

from .config import load_config, save_config
from .logging_utils import make_run_dir, save_csv, save_json
from .metrics import aggregate
from .plotting import plot_force_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep the desired normal force.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--forces", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    parser.add_argument("--geometries", nargs="+", default=["washer", "hex_nut", "bolt"])
    parser.add_argument("--episodes", type=int, default=5, help="seeds per (force, geometry)")
    parser.add_argument("--num-components", type=int, default=3)
    parser.add_argument("--planner", default="visual_greedy",
                        choices=["fixed", "global_sweep", "visual_greedy"])
    parser.add_argument("--perception", default="ground_truth",
                        choices=["ground_truth", "vision"])
    parser.add_argument("--spawn-mode", default="cluster", choices=["uniform", "cluster"])
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    return parser


def _run_one(job):
    cfg_dict, force, geometry, seed, planner, perception = job
    from .config import Config
    from .environments.episode import run_episode

    cfg = Config(cfg_dict)
    cfg.set_path("controller.desired_force", float(force))
    cfg.set_path("components.geometry", geometry)
    result = run_episode(cfg, seed=int(seed), planner_name=planner,
                         perception_name=perception, keep_images=False, verbose=False)
    row = result.metrics.to_dict()
    row["desired_force"] = float(force)
    row["geometry"] = geometry
    return row


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg.set_path("components.count", int(args.num_components))
    cfg.set_path("components.spawn_mode", args.spawn_mode)
    cfg.set_path("perception.backend", args.perception)

    out_dir = args.out or make_run_dir(str(cfg.logging.out_dir), tag="force-sweep")
    save_config(cfg, f"{out_dir}/config.yaml")
    cfg_dict = cfg.to_dict()

    jobs = [
        (cfg_dict, force, geometry, int(args.seed_base) + episode, args.planner, args.perception)
        for force, geometry, episode in itertools.product(
            args.forces, args.geometries, range(int(args.episodes))
        )
    ]
    print(f"  force sweep: {len(args.forces)} forces x {len(args.geometries)} geometries "
          f"x {args.episodes} seeds = {len(jobs)} episodes")

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

    rows.sort(key=lambda r: (r["geometry"], r["desired_force"], r["seed"]))
    save_csv(rows, f"{out_dir}/episodes.csv")
    summary = aggregate(rows, group_keys=("geometry", "desired_force"))
    save_csv(summary, f"{out_dir}/summary.csv")
    save_json({"forces": args.forces, "geometries": args.geometries,
               "episodes": args.episodes, "planner": args.planner,
               "wall_time_s": time.time() - t0}, f"{out_dir}/force_sweep.json")

    for path in plot_force_sweep(summary, out_dir):
        print(f"  plot: {path}")
    _print_summary(summary)
    _recommend(summary)
    print(f"\n  results written to {out_dir}")
    return 0


def _progress(i, total, row, t0):
    elapsed = time.time() - t0
    print(f"  [{i:4d}/{total}] F={row['desired_force']:5.1f} N {row['geometry']:9s} "
          f"seed={row['seed']:3d} rate={row['collection_rate']:.2f} "
          f"ride_over={row['n_ride_over']} jams={row['n_jam_events']} "
          f"overshoot={row['touchdown_overshoot_max']:.2f} N "
          f"| {elapsed/60:5.1f} min", flush=True)


def _print_summary(summary):
    print("\n  geometry    F[N]   rate   ride_over  jams   overshoot[N]  ejected")
    for row in summary:
        print(f"  {row['geometry']:<11s} {row['desired_force']:<6.1f} "
              f"{row['collection_rate_mean']:<6.2f} {row['n_ride_over_mean']:<10.2f} "
              f"{row['n_jam_events_mean']:<6.2f} {row['touchdown_overshoot_max_mean']:<13.2f} "
              f"{row['n_pushed_out_mean']:.2f}")


def _recommend(summary) -> None:
    """Pick the force that maximises the worst-case collection rate across geometries."""
    forces = sorted({row["desired_force"] for row in summary})
    best, best_score = None, -np.inf
    for force in forces:
        rows = [r for r in summary if r["desired_force"] == force]
        if not rows:
            continue
        worst_rate = min(r["collection_rate_mean"] for r in rows)
        penalty = max(r["n_ride_over_mean"] + r["n_jam_events_mean"] for r in rows)
        score = worst_rate - 0.05 * penalty
        if score > best_score:
            best, best_score = force, score
    if best is not None:
        print(f"\n  suggested F_z* = {best:.1f} N "
              f"(best worst-case collection rate across geometries)")
        print("  Treat this as a starting point, not a result: re-check it on hardware.")


if __name__ == "__main__":
    sys.exit(main())
