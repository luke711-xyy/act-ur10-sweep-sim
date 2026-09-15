"""Re-render figures and tables from a saved experiment directory.

    python -m sim.visualize_results --run runs/experiment-20260911-101500
    python -m sim.visualize_results --run runs/experiment-* --metric collection_rate
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

from .logging_utils import load_csv, save_csv
from .metrics import aggregate, paired_comparison
from .plotting import plot_experiment, plot_force_quality, plot_paired_delta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved experiment.")
    parser.add_argument("--run", required=True,
                        help="experiment directory (or a glob matching one)")
    parser.add_argument("--metric", default="collection_rate",
                        help="metric for the paired-difference box plot")
    parser.add_argument("--planners", nargs=2, default=["fixed", "visual_greedy"])
    parser.add_argument("--out", default=None, help="where to write figures (default: --run)")
    return parser


def _resolve(run: str) -> str:
    matches = sorted(glob.glob(run))
    if not matches:
        raise SystemExit(f"no experiment directory matches {run!r}")
    return matches[-1]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = _resolve(args.run)
    out_dir = args.out or run_dir
    episodes_csv = os.path.join(run_dir, "episodes.csv")
    if not os.path.exists(episodes_csv):
        raise SystemExit(f"{episodes_csv} not found -- run sim.run_experiment first")

    rows: List[dict] = load_csv(episodes_csv)
    summary = aggregate(rows, group_keys=("planner", "n_components"))
    save_csv(summary, os.path.join(out_dir, "summary.csv"))

    paths = plot_experiment(summary, out_dir)
    paths += plot_force_quality(summary, out_dir)
    pairs = paired_comparison(rows, args.planners[0], args.planners[1], args.metric)
    path = plot_paired_delta(pairs, args.metric, out_dir)
    if path:
        paths.append(path)

    print(f"  {len(rows)} episodes from {run_dir}")
    for row in summary:
        print(f"    {row['planner']:<13s} n={row['n_components']:<3} "
              f"rate={row['collection_rate_mean']:.2f} +/- {row['collection_rate_std']:.2f}  "
              f"strokes={row['n_strokes_mean']:.1f}  t={row['completion_time_mean']:.1f}s")
    for path in paths:
        print(f"  plot: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
