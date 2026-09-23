"""Generate automatic MuJoCo demonstrations for ACT."""

from __future__ import annotations

import argparse
import json

from ..config import load_config, save_config
from .dataset import ActDatasetWriter
from .rollout import run_expert_episode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--split", choices=["train", "val", "test", "failures"], default="train")
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="explicit seed per episode; useful for reproducible diagnostic failures",
    )
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument(
        "--failures-only", action="store_true",
        help="write only failed expert attempts; intended for a separate diagnostic dataset",
    )
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    root = args.out or str(cfg.act.dataset_dir)
    writer = ActDatasetWriter(root)
    save_config(cfg, f"{root}/{args.split}_config.yaml")
    for i in range(int(args.episodes)):
        count = 1 + (i % 6)
        local_cfg = cfg.copy()
        local_cfg.set_path("components.count", count)
        if args.seeds is not None and i >= len(args.seeds):
            raise ValueError("--seeds must provide at least one seed per episode")
        seed = int(args.seeds[i]) if args.seeds is not None else int(args.seed_base) + i
        result = run_expert_episode(local_cfg, seed=seed, collect_observations=True)
        if args.failures_only and result.success:
            should_write = False
        else:
            should_write = result.success or args.all_episodes or args.failures_only
        if should_write:
            writer.add_episode(
                f"{args.split}_{i:06d}", result.observations, result.actions,
                result.success, {"split": args.split, "seed": seed,
                                  "count": count, "failure_reason": result.failure_reason,
                                  "episode_kind": "expert_failure" if not result.success else "expert"},
            )
        print(json.dumps({"split": args.split, "index": i, "seed": seed,
                          "success": result.success, "collected": result.collected,
                          "total": result.total, "failure_reason": result.failure_reason}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
