"""Paired ordinary-ACT versus goal-token rollout evaluation.

The simulator supervisor is used only for reset/layout validation and scoring;
no object truth is placed in either policy batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..environments.sweep_env import SweepEnv
from .evaluate import load_act_runtime, run_act_episode
from .variants import GOAL_TOKEN_POLICY, ORDINARY_POLICY


def _layout_fingerprint(cfg, seed: int, target_count: int = 1) -> str:
    layout_cfg = cfg.copy()
    layout_cfg.set_path("task.target_count", int(target_count))
    env = SweepEnv(layout_cfg, seed=int(seed))
    try:
        env.reset(seed=int(seed))
        poses = np.concatenate((
            np.asarray(env.component_positions(), dtype="<f4"),
            np.asarray(env.component_quats(), dtype="<f4"),
        ), axis=1)
        return hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
    finally:
        env.close()


def make_layout_plan(cfg, dataset_root: str | Path, *, validation_layouts: int = 6,
                     test_layouts: int = 20, seed_start: int = 1_000_000) -> dict:
    """Create paired, target-independent unseen layouts with held-out fingerprints."""
    manifest = Path(dataset_root) / "manifest.jsonl"
    train_records = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training_seeds = {int(record["seed"]) for record in train_records}
    training_fingerprints = {
        str(record["layout_fingerprint"]) for record in train_records
    }
    reserved_fingerprints = set(training_fingerprints)
    reserved_seeds = set(training_seeds)
    plan = {}
    next_seed = int(seed_start)
    for split, count in (("validation", int(validation_layouts)),
                         ("test", int(test_layouts))):
        if count < 1:
            raise ValueError(f"{split} layout count must be positive")
        specs = []
        attempts = 0
        max_attempts = max(1000, count * 100)
        while len(specs) < count and attempts < max_attempts:
            seed = next_seed
            next_seed += 1
            attempts += 1
            if seed in reserved_seeds:
                continue
            fingerprints = {
                _layout_fingerprint(cfg, seed, target)
                for target in range(1, 7)
            }
            if len(fingerprints) != 1:
                continue
            fingerprint = fingerprints.pop()
            if fingerprint in reserved_fingerprints:
                continue
            reserved_seeds.add(seed)
            reserved_fingerprints.add(fingerprint)
            specs.append({
                "split": split,
                "layout_index": len(specs),
                "seed": seed,
                "layout_fingerprint": fingerprint,
                "paired_target_counts": list(range(1, 7)),
            })
        if len(specs) != count:
            raise RuntimeError(
                f"could not find {count} unique {split} layouts after {attempts} seeds"
            )
        plan[split] = specs
    return plan


def _result_row(split: str, layout: dict, target_count: int, model_label: str,
                model_path: str, result) -> dict:
    collected = int(result.collected)
    return {
        "split": split,
        "layout_index": int(layout["layout_index"]),
        "layout_seed": int(layout["seed"]),
        "layout_fingerprint": str(layout["layout_fingerprint"]),
        "target_count": int(target_count),
        "model_label": model_label,
        "model_path": model_path,
        "policy_variant": (
            GOAL_TOKEN_POLICY if model_label == "candidate" else ORDINARY_POLICY
        ),
        "success": bool(result.success),
        "failure_reason": str(result.failure_reason),
        "collected": collected,
        "under_count": max(0, int(target_count) - collected),
        "over_count": max(0, collected - int(target_count)),
        "collection_ratio": float(collected / max(1, int(result.total))),
        "elapsed_sim_seconds": float(result.elapsed),
        "sampled_frames": int(result.sampled_frames),
        "peak_force": float(result.peak_force),
        "termination_reason": str(result.termination_reason),
    }


def _aggregate(rows: list[dict]) -> dict:
    output = {}
    for label in ("baseline", "candidate"):
        model_rows = [row for row in rows if row["model_label"] == label]
        output[label] = {}
        for split in ("validation", "test"):
            split_rows = [row for row in model_rows if row["split"] == split]
            if not split_rows:
                continue
            output[label][split] = {"overall": {
                "episodes": len(split_rows),
                "successes": sum(bool(row["success"]) for row in split_rows),
                "success_rate": float(np.mean([row["success"] for row in split_rows])),
                "under_count_sum": sum(row["under_count"] for row in split_rows),
                "over_count_sum": sum(row["over_count"] for row in split_rows),
                "mean_collection_ratio": float(np.mean([
                    row["collection_ratio"] for row in split_rows
                ])),
            }}
            for target in range(1, 7):
                target_rows = [row for row in split_rows
                               if row["target_count"] == target]
                output[label][split][str(target)] = {
                    "episodes": len(target_rows),
                    "successes": sum(bool(row["success"]) for row in target_rows),
                    "success_rate": float(np.mean([row["success"] for row in target_rows]))
                    if target_rows else None,
                    "under_count_sum": sum(row["under_count"] for row in target_rows),
                    "over_count_sum": sum(row["over_count"] for row in target_rows),
                    "mean_collection_ratio": float(np.mean([
                        row["collection_ratio"] for row in target_rows
                    ])) if target_rows else None,
                }
    paired = {}
    key_fields = ("split", "layout_seed", "target_count")
    by_key = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        by_key.setdefault(key, {})[row["model_label"]] = bool(row["success"])
    for split in ("validation", "test"):
        pairs = [entry for key, entry in by_key.items()
                 if key[0] == split and {"baseline", "candidate"} <= entry.keys()]
        paired[split] = {
            "pairs": len(pairs),
            "candidate_wins": sum(
                entry["candidate"] and not entry["baseline"] for entry in pairs
            ),
            "candidate_losses": sum(
                entry["baseline"] and not entry["candidate"] for entry in pairs
            ),
            "ties": sum(
                entry["baseline"] == entry["candidate"] for entry in pairs
            ),
        }
    return {"models": output, "paired_comparison": paired}


def run_comparison(cfg, *, dataset_root: str | Path, baseline_model: str,
                   candidate_model: str, output_root: str | Path,
                   validation_layouts: int = 6, test_layouts: int = 20,
                   seed_start: int = 1_000_000,
                   preview: bool = True) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = make_layout_plan(
        cfg, dataset_root, validation_layouts=validation_layouts,
        test_layouts=test_layouts, seed_start=seed_start,
    )
    (output_root / "layout_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    rows = []
    result_path = output_root / "episodes.jsonl"
    if result_path.exists():
        raise FileExistsError(
            f"comparison output already contains episodes: {result_path}"
        )
    with result_path.open("w", encoding="utf-8") as stream:
        for model_label, model_path, variant in (
            ("baseline", baseline_model, ORDINARY_POLICY),
            ("candidate", candidate_model, GOAL_TOKEN_POLICY),
        ):
            runtime = load_act_runtime(
                cfg, model_path=model_path, policy_variant=variant
            )
            try:
                for split in ("validation", "test"):
                    for layout in plan[split]:
                        for target_count in range(1, 7):
                            episode_cfg = cfg.copy()
                            episode_cfg.set_path("task.target_count", target_count)
                            result = run_act_episode(
                                episode_cfg,
                                seed=int(layout["seed"]),
                                preview=preview,
                                policy_variant=variant,
                                runtime=runtime,
                            )
                            row = _result_row(
                                split, layout, target_count, model_label,
                                model_path, result,
                            )
                            rows.append(row)
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            print(json.dumps({
                                "event": "evaluation_episode",
                                "completed": len(rows),
                                "total": 6 * (validation_layouts + test_layouts) * 2,
                                "split": split,
                                "model": model_label,
                                "target_count": target_count,
                                "seed": layout["seed"],
                                "success": result.success,
                                "collected": result.collected,
                                "failure_reason": result.failure_reason,
                            }, ensure_ascii=False), flush=True)
            finally:
                del runtime
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except (ImportError, AttributeError):
                    pass
    summary = _aggregate(rows)
    summary.update({
        "dataset": str(dataset_root),
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "validation_layouts": int(validation_layouts),
        "test_layouts": int(test_layouts),
        "episodes_per_model": 6 * (int(validation_layouts) + int(test_layouts)),
        "paired_total": 6 * (int(validation_layouts) + int(test_layouts)),
        "policy_input_guard": (
            "task count token and ordinary environment counts are separate; "
            "supervisor layout truth is used only for evaluation"
        ),
    })
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/act_dataset_curve_v10/train_config.yaml")
    parser.add_argument("--dataset", default="runs/act_dataset_curve_v10")
    parser.add_argument("--baseline-model",
                        default="runs/act_model_curve_v10/checkpoints/step_100000")
    parser.add_argument("--candidate-model", default="runs/act_model_goal_token_v1")
    parser.add_argument("--out", default="runs/evaluations/act_goal_token_v1")
    parser.add_argument("--validation-layouts", type=int, default=6)
    parser.add_argument("--test-layouts", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--strict-deadline", action="store_true",
                        help="disable late-result preview mode")
    return parser


def main(argv=None) -> int:
    from ..config import load_config

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    summary = run_comparison(
        cfg,
        dataset_root=args.dataset,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        output_root=args.out,
        validation_layouts=args.validation_layouts,
        test_layouts=args.test_layouts,
        seed_start=args.seed_start,
        preview=not args.strict_deadline,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
