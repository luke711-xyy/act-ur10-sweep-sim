"""Acceptance gate for the six-episode ACT overfit stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import run_act_episode


def _paired_references(dataset_root: str | Path,
                       layout_id: str = "paired_000") -> dict[int, dict]:
    root = Path(dataset_root)
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    selected = {
        int(record.get("target_count", 0)): record
        for record in records
        if str(record.get("layout_id", "")) == str(layout_id)
        and str(record.get("episode_kind", "")) == "expert"
        and str(record.get("split", "")) == "train"
        and bool(record.get("success", False))
    }
    if set(selected) != set(range(1, 7)):
        raise ValueError(
            f"{layout_id} must contain one successful expert for targets 1..6"
        )
    return selected


def summarize_small_sample_gate(expert_records: dict[int, dict],
                                results: dict[int, object],
                                frame_limit: int = 500,
                                contact_xy_limit: float = 0.03,
                                force_limit: float = 20.0,
                                required_successes: int = 5) -> dict:
    """Evaluate contact, force and exact-success requirements."""
    episodes = []
    contact_errors = []
    for target_count in range(1, 7):
        record = expert_records[target_count]
        result = results[target_count]
        expected = np.asarray(
            record.get("first_contact_position", []), dtype=float
        ).reshape(-1)
        actual = np.asarray(
            getattr(result, "first_contact_position", []), dtype=float
        ).reshape(-1)
        contact_established = bool(
            expected.size >= 2 and actual.size >= 2
            and np.all(np.isfinite(actual[:2]))
        )
        error = (float(np.linalg.norm(actual[:2] - expected[:2]))
                 if contact_established else float("inf"))
        contact_errors.append(error)
        frame_count = int(getattr(result, "sampled_frames", 0)
                          or len(getattr(result, "actions", ())))
        episodes.append({
            "target_count": target_count,
            "success": bool(getattr(result, "success", False)),
            "contact_established": contact_established,
            "frames": frame_count,
            "contact_xy_error_m": error,
            "peak_force_n": float(getattr(result, "peak_force", float("inf"))),
            "reason": str(getattr(result, "failure_reason", "")),
        })
    median_error = float(np.median(contact_errors))
    successes = sum(item["success"] for item in episodes)
    all_contact_in_budget = all(
        item["contact_established"] and item["frames"] <= int(frame_limit)
        for item in episodes
    )
    force_ok = all(item["peak_force_n"] <= float(force_limit) for item in episodes)
    passed = bool(
        all_contact_in_budget
        and median_error <= float(contact_xy_limit)
        and force_ok
        and successes >= int(required_successes)
    )
    return {
        "passed": passed,
        "successes": successes,
        "required_successes": int(required_successes),
        "all_contact_in_budget": all_contact_in_budget,
        "median_contact_xy_error_m": median_error,
        "contact_xy_limit_m": float(contact_xy_limit),
        "force_ok": force_ok,
        "force_limit_n": float(force_limit),
        "frame_limit": int(frame_limit),
        "episodes": episodes,
    }


def run_small_sample_gate(cfg, model_path: str,
                          dataset_root: str | Path) -> dict:
    references = _paired_references(dataset_root)
    results = {}
    for target_count in range(1, 7):
        record = references[target_count]
        local_cfg = cfg.copy()
        local_cfg.set_path("task.target_count", target_count)
        results[target_count] = run_act_episode(
            local_cfg, seed=int(record["seed"]),
            model_path=model_path, preview=True,
        )
    return summarize_small_sample_gate(
        references, results,
        frame_limit=int(cfg.episode.get("max_frames", 500)),
        force_limit=float(cfg.controller.safe_max_force),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the six-target ACT autonomous-contact overfit gate"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from ..config import load_config

    cfg = load_config(args.config)
    summary = run_small_sample_gate(
        cfg, args.model, args.dataset or str(cfg.act.dataset_dir)
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
