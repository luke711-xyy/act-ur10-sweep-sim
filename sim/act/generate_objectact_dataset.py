"""Generate RGB-derived schema-v5 expert data without A* in the reader.

The expert rollout may use A* to produce its action path.  The observation
side of the saved episode is built by the RGB detector frontend, and the only
MuJoCo-derived selection signal is written as an offline supervision label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..perception.detector_training import verified_detector_checkpoint
from .object_dataset import ObjectActDatasetWriter
from .object_interface import ObjectACTObservationBuilder, RGBObjectPerceptionFrontend
from .rollout import run_expert_episode
from ..perception.object_tokens import OBJECT_TOKEN_SLICES


def _selection_targets(result) -> np.ndarray:
    """Match predicted slots to current physical parts for offline labels only."""
    observations = list(getattr(result, "observations", ()))
    predicted = list(getattr(result, "object_observations", ()))
    if len(observations) != len(predicted):
        raise ValueError("v4 and v5 expert observation lengths do not match")
    final_mask = np.asarray(result.final_collected_mask, dtype=bool).reshape(-1)
    labels = np.zeros((len(predicted), 6), dtype=bool)
    for frame, (legacy, v5) in enumerate(zip(observations, predicted)):
        poses = np.asarray(legacy.get("object_pose", np.zeros((0, 7))), dtype=float)
        if poses.ndim != 2 or poses.shape[1] < 2:
            continue
        token_xy = np.asarray(v5["object_tokens"], dtype=float)[:, OBJECT_TOKEN_SLICES["xy"]]
        valid = np.asarray(v5["object_valid"], dtype=bool)
        if len(final_mask) != len(poses):
            raise ValueError("final collection mask and object pose count differ")
        used: set[int] = set()
        for slot in np.flatnonzero(valid):
            distances = np.linalg.norm(poses[:, :2] - token_xy[slot][None, :], axis=1)
            order = np.argsort(distances)
            match = next((int(index) for index in order if int(index) not in used), None)
            if match is not None and float(distances[match]) <= 0.06:
                labels[frame, int(slot)] = bool(final_mask[match])
                used.add(match)
    return labels


def generate_one(
    cfg,
    *,
    root: str | Path,
    target_count: int,
    seed: int,
    detector_checkpoint: str | None,
    split: str = "train",
) -> dict:
    """Generate and persist one successful expert episode."""
    if not detector_checkpoint:
        raise ValueError(
            "an RGB detector checkpoint is required; use the explicit detector "
            "training step before generating v5 policy data"
        )
    checkpoint = verified_detector_checkpoint(detector_checkpoint)
    local_cfg = cfg.copy()
    local_cfg.set_path("task.target_count", int(target_count))

    def builder_factory():
        return ObjectACTObservationBuilder(
            local_cfg,
            frontend=RGBObjectPerceptionFrontend(
                local_cfg,
                device="cpu",
                detector_checkpoint=str(checkpoint),
            ),
        )

    result = run_expert_episode(
        local_cfg,
        seed=int(seed),
        collect_observations=True,
        object_builder_factory=builder_factory,
    )
    if not result.success:
        raise RuntimeError(f"expert generation failed: {result.failure_reason}")
    labels = _selection_targets(result)
    episode_id = f"expert_v5_n{int(target_count)}_s{int(seed)}"
    metadata = {
        "split": str(split),
        "seed": int(seed),
        "total_count": int(result.total),
        "target_count": int(target_count),
        "collected": int(result.collected),
        "fps": float(local_cfg.act.action_hz),
        "perception_source": "rgb_detector",
        "detector_checkpoint": str(checkpoint),
        "selection_label_source": "actual_final_collected_ids_offline_only",
        "planner": "astar_expert_only",
        "planner_status": str(result.planner_status),
        "planner_strategy": str(result.planner_strategy),
        "planner_turn_count": int(result.planner_turn_count),
        "peak_force": float(result.peak_force),
        "first_contact_position": (
            np.asarray(result.first_contact_position, dtype=float).tolist()
            if result.first_contact_position is not None else None
        ),
    }
    ObjectActDatasetWriter(str(root)).add_episode(
        episode_id,
        result.object_observations,
        labels,
        True,
        metadata,
    )
    return {"episode_id": episode_id, **metadata, "frames": len(labels)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate one RGB-derived v5 expert episode")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/objectact_dataset")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    if not 1 <= int(args.target_count) <= 6:
        parser.error("--target-count must be between 1 and 6")
    result = generate_one(
        load_config(args.config),
        root=args.out,
        target_count=args.target_count,
        seed=args.seed,
        detector_checkpoint=args.detector_checkpoint,
        split=args.split,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
