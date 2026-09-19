"""Generate RGB-derived schema-v5 expert data without A* in the reader.

The expert rollout may use A* to produce its action path.  The observation
side of the saved episode is built by the RGB detector frontend, and the only
MuJoCo-derived selection signal is written as an offline supervision label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import load_config
from ..perception.detector_training import verified_detector_checkpoint
from .collect_dataset import (
    find_shared_layout_seed,
    training_episode_plan,
    write_split_manifests,
)
from .object_dataset import ObjectActDatasetWriter
from .object_interface import ObjectACTObservationBuilder, RGBObjectPerceptionFrontend
from .rollout import run_expert_episode
from ..perception.object_tokens import OBJECT_TOKEN_SLICES


def objectact_training_episode_plan(seed_base: int = 4100):
    """Return the approved 8-paired + 12-independent v5 training plan."""

    return training_episode_plan(seed_base)


def resolve_detector_device(value: str = "auto") -> str:
    """Resolve the detector inference device without changing policy semantics."""

    value = str(value).lower()
    if value == "auto":
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    if value not in {"cpu", "mps", "cuda"}:
        raise ValueError("detector device must be auto, cpu, mps, or cuda")
    return value


def _read_v5_manifest(root: str | Path) -> list[dict]:
    path = Path(root) / "manifest_v5.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_objectact_training_manifest(root: str | Path) -> dict:
    """Validate the exact 120-record v5 training composition."""

    records = _read_v5_manifest(root)
    invalid = []
    for record in records:
        if "target_indices" in record:
            raise ValueError(
                f"v5 training manifest must not expose target_indices: "
                f"{record.get('episode_id', '<missing>')}"
            )
        if not (
            int(record.get("schema_version", 0)) == 5
            and str(record.get("episode_kind", "")) == "expert"
            and str(record.get("split", "")) == "train"
            and bool(record.get("success", False))
            and int(record.get("action_dim", 0)) == 4
            and int(record.get("robot_state_dim", 0)) == 36
            and 1 <= int(record.get("target_count", 0)) <= 6
            and str(record.get("layout_id", ""))
            and str(record.get("layout_kind", "")) in {"paired", "independent"}
        ):
            invalid.append(str(record.get("episode_id", "<missing>")))
    if invalid:
        raise ValueError(f"v5 training manifest contains incompatible records: {invalid}")
    slots = [
        (str(record["layout_id"]), int(record["target_count"]))
        for record in records
    ]
    if len(set(slots)) != len(slots):
        raise ValueError("v5 training manifest contains duplicate layout/target slots")
    counts = Counter(int(record["target_count"]) for record in records)
    per_target = {str(target): int(counts.get(target, 0)) for target in range(1, 7)}
    if len(records) != 120 or any(value != 20 for value in per_target.values()):
        raise ValueError(
            f"v5 training manifest must contain 120 records and 20 per target; "
            f"found total={len(records)}, per_target={per_target}"
        )
    paired = [record for record in records if record["layout_kind"] == "paired"]
    independent = [record for record in records if record["layout_kind"] == "independent"]
    paired_ids = sorted({str(record["layout_id"]) for record in paired})
    if len(paired) != 48 or len(paired_ids) != 8:
        raise ValueError("v5 training manifest must contain 8 paired layouts x 6 goals")
    for layout_id in paired_ids:
        group = [record for record in paired if str(record["layout_id"]) == layout_id]
        if {int(record["target_count"]) for record in group} != set(range(1, 7)):
            raise ValueError(f"paired layout {layout_id} does not contain goals 1..6")
        if len({int(record.get("seed", -1)) for record in group}) != 1:
            raise ValueError(f"paired layout {layout_id} does not share one seed")
    if len(independent) != 72:
        raise ValueError("v5 training manifest must contain 72 independent records")
    for target in range(1, 7):
        group = [
            record for record in independent
            if int(record["target_count"]) == target
        ]
        if len(group) != 12 or len({str(record["layout_id"]) for record in group}) != 12:
            raise ValueError(f"target {target} does not contain 12 independent layouts")
    return {
        "total": len(records),
        "per_target": per_target,
        "paired_records": len(paired),
        "paired_layouts": len(paired_ids),
        "independent_records": len(independent),
    }


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
    layout_id: str | None = None,
    layout_kind: str | None = None,
    requested_seed: int | None = None,
    generation_attempt: int = 1,
    detector_device: str = "auto",
) -> dict:
    """Generate and persist one successful expert episode."""
    if not detector_checkpoint:
        raise ValueError(
            "an RGB detector checkpoint is required; use the explicit detector "
            "training step before generating v5 policy data"
        )
    checkpoint = verified_detector_checkpoint(detector_checkpoint)
    detector_device = resolve_detector_device(detector_device)
    local_cfg = cfg.copy()
    local_cfg.set_path("task.target_count", int(target_count))

    def builder_factory():
        return ObjectACTObservationBuilder(
            local_cfg,
            frontend=RGBObjectPerceptionFrontend(
                local_cfg,
                device=detector_device,
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
    if layout_id:
        episode_id = (
            f"expert_v5_{str(layout_id)}_n{int(target_count)}_s{int(seed)}"
        )
    else:
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
        "detector_device": str(detector_device),
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
    if layout_id is not None:
        metadata.update({
            "layout_id": str(layout_id),
            "layout_kind": str(layout_kind or "independent"),
            "requested_seed": int(requested_seed if requested_seed is not None else seed),
            "generation_attempt": int(generation_attempt),
        })
    ObjectActDatasetWriter(str(root)).add_episode(
        episode_id,
        result.object_observations,
        labels,
        True,
        metadata,
    )
    return {"episode_id": episode_id, **metadata, "frames": len(labels)}


def _v5_expert_config(cfg, target_count: int):
    local_cfg = cfg.copy()
    local_cfg.set_path("components.count", 6)
    local_cfg.set_path("components.geometry", "mixed")
    local_cfg.set_path("task.total_count", 6)
    local_cfg.set_path("task.target_count", int(target_count))
    return local_cfg


def _append_v5_failure(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "generation_failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_layout_assignment(root: Path, payload: dict) -> None:
    path = root / "layout_assignments.jsonl"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(
        old + json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_layout_assignments(root: Path) -> dict[str, dict]:
    path = root / "layout_assignments.jsonl"
    if not path.exists():
        return {}
    assignments = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        layout_id = str(payload["layout_id"])
        if layout_id in assignments and assignments[layout_id] != payload:
            raise ValueError(f"conflicting layout assignment for {layout_id}")
        assignments[layout_id] = payload
    return assignments


def _progress(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _existing_v5_slots(root: Path) -> tuple[list[dict], set[tuple[str, int]]]:
    records = _read_v5_manifest(root)
    slots: set[tuple[str, int]] = set()
    for record in records:
        if not (
            int(record.get("schema_version", 0)) == 5
            and str(record.get("episode_kind", "")) == "expert"
            and str(record.get("split", "")) == "train"
            and bool(record.get("success", False))
            and str(record.get("layout_id", ""))
            and str(record.get("layout_kind", "")) in {"paired", "independent"}
        ):
            raise ValueError(
                "existing v5 root contains a record outside the resumable training contract"
            )
        slot = (str(record["layout_id"]), int(record["target_count"]))
        if slot in slots:
            raise ValueError(f"duplicate v5 training slot {slot}")
        slots.add(slot)
    return records, slots


def _find_independent_seed(
    cfg,
    *,
    target_count: int,
    requested_seed: int,
    max_attempts: int,
    root: Path,
    layout_id: str,
) -> tuple[int, int]:
    for attempt in range(1, int(max_attempts) + 1):
        seed = int(requested_seed) + attempt - 1
        result = run_expert_episode(
            _v5_expert_config(cfg, target_count),
            seed=seed,
            collect_observations=False,
        )
        if result.success:
            return seed, attempt
        _append_v5_failure(root, {
            "layout_id": str(layout_id),
            "layout_kind": "independent",
            "target_count": int(target_count),
            "requested_seed": int(requested_seed),
            "seed": int(seed),
            "generation_attempt": int(attempt),
            "collected": int(result.collected),
            "reason": str(result.failure_reason),
            "planner_status": str(result.planner_status),
            "planner_strategy": str(result.planner_strategy),
        })
    raise RuntimeError(
        f"no successful layout for target {target_count} after {max_attempts} seeds"
    )


def generate_objectact_training_dataset(
    cfg,
    *,
    root: str | Path,
    detector_checkpoint: str,
    seed_base: int = 4100,
    max_attempts: int = 64,
    detector_device: str = "auto",
) -> dict:
    """Generate the resumable 120-success RGB-derived v5 training set.

    Layout screening uses the expert-only MuJoCo planner with observations
    disabled.  Only a successful RGB-derived capture is handed to the v5
    writer, so rejected attempts never create episode images or arrays.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = str(verified_detector_checkpoint(detector_checkpoint))
    detector_device = resolve_detector_device(detector_device)
    _, slots = _existing_v5_slots(root)
    assignments = _read_layout_assignments(root)
    plan = objectact_training_episode_plan(seed_base)

    for paired_index in range(8):
        layout_id = f"paired_{paired_index:03d}"
        missing = [
            target for target in range(1, 7)
            if (layout_id, target) not in slots
        ]
        if not missing:
            continue
        assignment = assignments.get(layout_id)
        if assignment is None:
            requested_seed = int(seed_base) + paired_index * 1_000
            _progress(
                "screen_shared_layout",
                layout_id=layout_id,
                requested_seed=requested_seed,
                target_counts=list(range(1, 7)),
            )
            accepted_seed, attempt = find_shared_layout_seed(
                cfg,
                requested_seed,
                int(max_attempts),
                failure_root=root,
                layout_id=layout_id,
            )
            assignment = {
                "layout_id": layout_id,
                "layout_kind": "paired",
                "requested_seed": requested_seed,
                "seed": int(accepted_seed),
                "generation_attempt": int(attempt),
                "target_counts": list(range(1, 7)),
            }
            _append_layout_assignment(root, assignment)
            assignments[layout_id] = assignment
            _progress("layout_assigned", **assignment)
        elif set(int(value) for value in assignment.get("target_counts", [])) != set(range(1, 7)):
            raise ValueError(f"paired assignment {layout_id} does not cover targets 1..6")
        for target_count in missing:
            try:
                _progress(
                    "capture_start",
                    layout_id=layout_id,
                    layout_kind="paired",
                    target_count=int(target_count),
                    seed=int(assignment["seed"]),
                )
                generate_one(
                    cfg,
                    root=root,
                    target_count=target_count,
                    seed=int(assignment["seed"]),
                    detector_checkpoint=checkpoint,
                    split="train",
                    layout_id=layout_id,
                    layout_kind="paired",
                    requested_seed=int(assignment["requested_seed"]),
                    generation_attempt=int(assignment["generation_attempt"]),
                    detector_device=detector_device,
                )
                _progress(
                    "capture_success",
                    layout_id=layout_id,
                    layout_kind="paired",
                    target_count=int(target_count),
                    seed=int(assignment["seed"]),
                )
            except Exception as exc:
                _append_v5_failure(root, {
                    "layout_id": layout_id,
                    "layout_kind": "paired",
                    "target_count": int(target_count),
                    "seed": int(assignment["seed"]),
                    "reason": f"capture: {exc}",
                })
                raise
            slots.add((layout_id, target_count))

    for planned in plan:
        if planned.layout_kind != "independent":
            continue
        slot = (str(planned.layout_id), int(planned.target_count))
        if slot in slots:
            continue
        layout_id = str(planned.layout_id)
        assignment = assignments.get(layout_id)
        if assignment is None:
            _progress(
                "screen_independent_layout",
                layout_id=layout_id,
                target_count=int(planned.target_count),
                requested_seed=int(planned.seed),
            )
            accepted_seed, attempt = _find_independent_seed(
                cfg,
                target_count=int(planned.target_count),
                requested_seed=int(planned.seed),
                max_attempts=int(max_attempts),
                root=root,
                layout_id=layout_id,
            )
            assignment = {
                "layout_id": layout_id,
                "layout_kind": "independent",
                "target_count": int(planned.target_count),
                "requested_seed": int(planned.seed),
                "seed": int(accepted_seed),
                "generation_attempt": int(attempt),
                "target_counts": [int(planned.target_count)],
            }
            _append_layout_assignment(root, assignment)
            assignments[layout_id] = assignment
            _progress("layout_assigned", **assignment)
        try:
            _progress(
                "capture_start",
                layout_id=layout_id,
                layout_kind="independent",
                target_count=int(planned.target_count),
                seed=int(assignment["seed"]),
            )
            generate_one(
                cfg,
                root=root,
                target_count=int(planned.target_count),
                seed=int(assignment["seed"]),
                detector_checkpoint=checkpoint,
                split="train",
                layout_id=layout_id,
                layout_kind="independent",
                requested_seed=int(assignment["requested_seed"]),
                generation_attempt=int(assignment["generation_attempt"]),
                detector_device=detector_device,
            )
            _progress(
                "capture_success",
                layout_id=layout_id,
                layout_kind="independent",
                target_count=int(planned.target_count),
                seed=int(assignment["seed"]),
            )
        except Exception as exc:
            _append_v5_failure(root, {
                "layout_id": layout_id,
                "layout_kind": "independent",
                "target_count": int(planned.target_count),
                "seed": int(assignment["seed"]),
                "reason": f"capture: {exc}",
            })
            raise
        slots.add(slot)

    summary = validate_objectact_training_manifest(root)
    split_manifests = write_split_manifests(root)
    return {
        **summary,
        "detector_checkpoint": checkpoint,
        "detector_device": detector_device,
        "split_manifests": split_manifests,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate one RGB-derived v5 expert episode")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/objectact_dataset")
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch", action="store_true", help="generate the approved 120-record training set")
    parser.add_argument("--seed-base", type=int, default=4100)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--detector-device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.batch:
        if not args.detector_checkpoint:
            parser.error("--batch requires --detector-checkpoint")
        result = generate_objectact_training_dataset(
            cfg,
            root=args.out,
            detector_checkpoint=args.detector_checkpoint,
            seed_base=args.seed_base,
            max_attempts=args.max_attempts,
            detector_device=args.detector_device,
        )
    else:
        if args.target_count is None or args.seed is None:
            parser.error("single-episode mode requires --target-count and --seed")
        if not 1 <= int(args.target_count) <= 6:
            parser.error("--target-count must be between 1 and 6")
        result = generate_one(
            cfg,
            root=args.out,
            target_count=args.target_count,
            seed=args.seed,
            detector_checkpoint=args.detector_checkpoint,
            split=args.split,
            detector_device=args.detector_device,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
