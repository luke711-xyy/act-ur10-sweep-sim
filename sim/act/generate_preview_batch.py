"""Generate one local workbench preview for each exact target count.

This is intentionally a thin CLI around :class:`WorkbenchState`: the
workbench and this batch generator therefore share the same MuJoCo expert
rollout, A* one-pass planner, admittance controller, observation schema, and
preview manifest.  It runs entirely on the local machine and does not start
ACT training or call any remote service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config, save_config
from .collect_dataset import find_shared_layout_seed, preview_batch_plan
from ..web.workbench import WorkbenchState


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate six workbench previews, targeting exactly 1..6 parts.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--out", default=None,
        help="preview manifest directory; defaults to runs/workbench_previews",
    )
    parser.add_argument("--seed-base", type=int, default=4100)
    parser.add_argument(
        "--max-attempts-per-preview", type=int, default=64,
        help="shared-layout seed attempts before giving up (default: 64)",
    )
    parser.add_argument("--set", action="append", default=[],
                        help="configuration override key.path=value (repeatable)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    preview_root = Path(args.out) if args.out else Path(cfg.logging.out_dir) / "workbench_previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(preview_root / "preview_batch_config.yaml"))

    state = WorkbenchState(
        cfg,
        dataset_root=cfg.act.dataset_dir,
        preview_root=preview_root,
    )
    accepted_seed, shared_attempt = find_shared_layout_seed(
        cfg, int(args.seed_base), int(args.max_attempts_per_preview),
        failure_root=preview_root, layout_id="paired_000",
    )
    summaries = []
    generated_ids = []
    try:
        for target_count, _requested_seed in preview_batch_plan(args.seed_base):
            record = state.build_preview(
                seed=accepted_seed, target_count=target_count,
                max_attempts=1, persist_failed=False,
                layout_id="paired_000", layout_kind="paired",
            )
            if not bool(record.get("success", False)):
                raise RuntimeError(
                    f"screened shared layout failed during target {target_count} capture: "
                    f"{record.get('failure_reason', '')}"
                )
            generated_ids.append(str(record["episode_id"]))
            summary = {
                "episode_id": record["episode_id"],
                "target_count": target_count,
                "total_count": record.get("total_count", 6),
                "seed": record.get("seed", accepted_seed),
                "requested_seed": int(args.seed_base),
                "shared_generation_attempt": shared_attempt,
                "layout_id": "paired_000",
                "success": bool(record.get("success", False)),
                "collected": record.get("collected"),
                "planner_status": record.get("planner_status", "unknown"),
                "planner_strategy": record.get("planner_strategy", "unknown"),
                "planner_failure_reason": record.get("planner_failure_reason", ""),
                "failure_reason": record.get("failure_reason", ""),
            }
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))
    except Exception:
        for episode_id in generated_ids:
            state.delete_episode(episode_id)
        raise

    print(json.dumps({
        "preview_root": str(preview_root),
        "episodes": len(summaries),
        "layout_id": "paired_000",
        "shared_seed": accepted_seed,
        "shared_generation_attempt": shared_attempt,
        "targets": [item["target_count"] for item in summaries],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
