"""Generate automatic MuJoCo demonstrations for ACT."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import load_config, save_config
from ..model.geometries import GEOMETRY_NAMES
from .dataset import ActDatasetWriter
from .naming import next_demo_name
from .rollout import run_expert_episode


@dataclass(frozen=True)
class EpisodeSpec:
    """One deterministic layout/goal slot in a dataset split."""

    split: str
    target_count: int
    seed: int
    layout_id: str
    layout_kind: str

    def __iter__(self):
        # Preserve the historical ``for target_count, seed in plan`` API.
        yield self.target_count
        yield self.seed


def parse_episode_plan(value: str) -> list[tuple[int, int]]:
    """Parse reproducible ``target_count:seed`` pairs for a curated batch."""
    if not value or not value.strip():
        raise ValueError("episode plan must not be empty")
    plan = []
    for raw_entry in value.split(","):
        fields = raw_entry.strip().split(":")
        if len(fields) != 2:
            raise ValueError(f"invalid episode-plan entry: {raw_entry!r}")
        try:
            target_count, seed = (int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"invalid episode-plan entry: {raw_entry!r}") from exc
        if not 1 <= target_count <= 6:
            raise ValueError("episode-plan target counts must be between 1 and 6")
        if seed < 0:
            raise ValueError("episode-plan seeds must be non-negative")
        plan.append((target_count, seed))
    return plan


def preview_batch_plan(seed_base: int) -> list[EpisodeSpec]:
    """Return goals 1..6 on one shared layout for causal preview review."""
    seed_base = int(seed_base)
    if seed_base < 0:
        raise ValueError("preview batch seed base must be non-negative")
    return [EpisodeSpec(
        split="pilot",
        target_count=target_count,
        seed=seed_base,
        layout_id="paired_000",
        layout_kind="paired",
    ) for target_count in range(1, 7)]


def training_episode_plan(seed_base: int = 4100) -> list[EpisodeSpec]:
    """Build the approved 120-demo composition without sampling MuJoCo.

    Eight paired layouts each appear with all six exact target counts (48
    slots).  The remaining twelve layouts per target are independent (72
    slots).  The first paired layout is intentionally identical to
    :func:`preview_batch_plan`, so an approved pilot can be promoted without
    being regenerated.
    """
    seed_base = int(seed_base)
    if seed_base < 0:
        raise ValueError("training seed base must be non-negative")
    plan: list[EpisodeSpec] = []
    for paired_index in range(8):
        # Reserve a local retry window for each paired layout while keeping
        # paired_000 anchored at the user-reviewed preview seed base.
        seed = seed_base + paired_index * 1_000
        layout_id = f"paired_{paired_index:03d}"
        for target_count in range(1, 7):
            plan.append(EpisodeSpec(
                split="train", target_count=target_count, seed=seed,
                layout_id=layout_id, layout_kind="paired"))
    independent_base = seed_base + 10_000
    for target_count in range(1, 7):
        for independent_index in range(12):
            seed = (independent_base + (target_count - 1) * 7_000
                    + independent_index * 100)
            plan.append(EpisodeSpec(
                split="train", target_count=target_count, seed=seed,
                layout_id=f"independent_n{target_count}_{independent_index:03d}",
                layout_kind="independent"))
    return plan


def evaluation_layout_plan(split: str, seed_base: int,
                           layouts_per_target: int) -> list[EpisodeSpec]:
    """Create a seed-isolated validation or test layout manifest."""
    split = str(split)
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be 'val' or 'test'")
    seed_base = int(seed_base)
    layouts_per_target = int(layouts_per_target)
    if seed_base < 0 or layouts_per_target < 1:
        raise ValueError("evaluation seeds must be non-negative and count positive")
    return [EpisodeSpec(
        split=split,
        target_count=target_count,
        seed=seed_base + (target_count - 1) * 1_000 + index,
        layout_id=f"{split}_n{target_count}_{index:03d}",
        layout_kind="independent",
    ) for target_count in range(1, 7) for index in range(layouts_per_target)]


def validate_training_manifest(root: str | Path) -> dict:
    """Raise unless ``root`` is exactly the approved 120-record train set."""
    root = Path(root)
    path = root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    invalid = [record.get("episode_id", "<missing>") for record in records if not (
        int(record.get("schema_version", 0)) == 4
        and int(record.get("action_dim", 0)) == 4
        and int(record.get("state_dim", 0)) == 42
        and str(record.get("episode_kind", "")) == "expert"
        and str(record.get("split", "")) == "train"
        and bool(record.get("success", False))
        and 1 <= int(record.get("target_count", 0)) <= 6
        and 1 <= int(record.get("frame_count", 0)) <= 500
        and "target_indices" not in record
    )]
    per_target = {
        str(target_count): sum(
            int(record.get("target_count", 0)) == target_count
            for record in records
        )
        for target_count in range(1, 7)
    }
    slots = [
        (str(record.get("layout_id", "")), int(record.get("target_count", 0)))
        for record in records
    ]
    paired = [
        record for record in records
        if str(record.get("layout_kind", "")) == "paired"
    ]
    independent = [
        record for record in records
        if str(record.get("layout_kind", "")) == "independent"
    ]
    paired_ids = sorted({str(record.get("layout_id", "")) for record in paired})
    paired_ok = len(paired) == 48 and len(paired_ids) == 8
    for layout_id in paired_ids:
        group = [record for record in paired
                 if str(record.get("layout_id", "")) == layout_id]
        paired_ok = paired_ok and {
            int(record.get("target_count", 0)) for record in group
        } == set(range(1, 7))
        paired_ok = paired_ok and len({
            int(record.get("seed", -1)) for record in group
        }) == 1
    independent_ok = len(independent) == 72
    for target_count in range(1, 7):
        group = [record for record in independent
                 if int(record.get("target_count", 0)) == target_count]
        independent_ok = independent_ok and len(group) == 12
        independent_ok = independent_ok and len({
            str(record.get("layout_id", "")) for record in group
        }) == 12
    if invalid:
        raise ValueError(f"training manifest contains incompatible records: {invalid}")
    if len(records) != 120 or any(value != 20 for value in per_target.values()):
        raise ValueError(
            f"training manifest must contain 120 records and 20 per target; "
            f"found total={len(records)}, per_target={per_target}"
        )
    if len(set(slots)) != len(slots) or not paired_ok or not independent_ok:
        raise ValueError(
            "training manifest composition must be 8 paired layouts x 6 goals "
            "plus 12 independent layouts per goal"
        )
    return {
        "total": len(records),
        "per_target": per_target,
        "paired_records": len(paired),
        "paired_layouts": len(paired_ids),
        "independent_records": len(independent),
    }


def _read_manifest(root: Path) -> list[dict]:
    path = root / "manifest.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]


def _append_failure_log(root: Path, payload: dict) -> None:
    """Record a rejected generation attempt without images or trajectories."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "generation_failures.jsonl").open(
            "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def validate_preview_training_manifest(root: str | Path,
                                       per_target: int = 15) -> dict:
    """Validate a promoted 6-by-N successful preview training set."""
    root = Path(root)
    per_target = int(per_target)
    if per_target < 1:
        raise ValueError("per_target must be positive")
    records = _read_manifest(root)
    invalid = [record.get("episode_id", "<missing>") for record in records
               if not (
                   int(record.get("schema_version", 0)) == 4
                   and int(record.get("action_dim", 0)) == 4
                   and int(record.get("state_dim", 0)) == 42
                   and str(record.get("episode_kind", "")) == "expert"
                   and str(record.get("split", "")) == "train"
                   and bool(record.get("success", False))
                   and 1 <= int(record.get("target_count", 0)) <= 6
                   and 1 <= int(record.get("frame_count", 0)) <= 500
                   and "target_indices" not in record
               )]
    per_target_counts = {
        str(target): sum(int(record.get("target_count", 0)) == target
                         for record in records)
        for target in range(1, 7)
    }
    expected_total = 6 * per_target
    if invalid or len(records) != expected_total or any(
            value != per_target for value in per_target_counts.values()):
        raise ValueError(
            f"preview training manifest must contain {expected_total} successful "
            f"records and {per_target} per target; invalid={invalid}, "
            f"per_target={per_target_counts}"
        )
    return {"total": len(records), "per_target": per_target_counts,
            "source": "promoted_success_previews"}


def promote_success_preview_set(preview_root: str | Path,
                                dataset_root: str | Path,
                                expected_per_target: int = 15) -> list[dict]:
    """Promote all successful preview records into a formal train dataset.

    This is intentionally separate from ``promote_preview_batch``: the latter
    handles the six-record same-layout approval batch, while this path uses the
    user's completed 15-per-target preview set for ACT training.  The source is
    never modified and the destination is published only after all episode
    directories and the manifest are complete.
    """
    preview_root = Path(preview_root)
    dataset_root = Path(dataset_root)
    expected_per_target = int(expected_per_target)
    if expected_per_target < 1:
        raise ValueError("expected_per_target must be positive")
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise ValueError("training dataset destination must be empty")

    source_records = [
        record for record in _read_manifest(preview_root)
        if bool(record.get("success", False))
        and str(record.get("episode_kind", "")) == "expert_preview"
    ]
    by_target = {
        target: [record for record in source_records
                 if int(record.get("target_count", 0)) == target]
        for target in range(1, 7)
    }
    if (len(source_records) != 6 * expected_per_target
            or any(len(records) != expected_per_target
                   for records in by_target.values())):
        counts = {str(target): len(records)
                  for target, records in by_target.items()}
        raise ValueError(
            f"successful preview set must contain {6 * expected_per_target} "
            f"records, {expected_per_target} per target; found {counts}"
        )

    temporary_root = dataset_root.parent / (
        f".{dataset_root.name}.tmp-{uuid.uuid4().hex}"
    )
    if temporary_root.exists():
        raise FileExistsError(temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=False)
    promoted = []
    try:
        for record in sorted(source_records,
                             key=lambda item: str(item.get("episode_id", ""))):
            episode_id = str(record["episode_id"])
            source = preview_root / episode_id
            destination = temporary_root / episode_id
            if not source.is_dir():
                raise FileNotFoundError(source)
            shutil.copytree(source, destination)
            updated = dict(record)
            updated.update({
                "episode_kind": "expert",
                "split": "train",
                "preview": False,
                "promoted_from_preview": True,
                "layout_id": str(record.get("layout_id") or f"preview_{episode_id}"),
                "layout_kind": str(record.get("layout_kind") or "independent"),
            })
            # Planner-selected object identities are private expert metadata,
            # not an ACT input or a training label.
            updated.pop("target_indices", None)
            promoted.append(updated)
        (temporary_root / "manifest.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n"
                    for record in promoted),
            encoding="utf-8",
        )
        # Validate the staged copy before it becomes the training root.
        validate_preview_training_manifest(temporary_root, expected_per_target)
        temporary_root.rename(dataset_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return promoted


def _expert_config(cfg, target_count: int):
    local_cfg = cfg.copy()
    local_cfg.set_path("components.count", 6)
    local_cfg.set_path("components.geometry", "mixed")
    local_cfg.set_path("task.total_count", 6)
    local_cfg.set_path("task.target_count", int(target_count))
    return local_cfg


def screen_shared_layout(cfg, seed: int,
                         target_counts=range(1, 7)) -> tuple[bool, list[dict]]:
    """Physically screen one seed for every requested exact cardinality.

    Screening disables observation capture, so rejected layouts create no RGB
    files or trajectory arrays.  A paired layout is accepted only if all six
    expert executions succeed with the same seed and component geometry.
    """
    outcomes = []
    accepted = True
    # Reject on the hardest cardinalities first.  A seed that cannot collect
    # all six should not spend time screening five easier goals.
    ordered_targets = sorted(
        (int(value) for value in target_counts), reverse=True
    )
    for target_count in ordered_targets:
        try:
            result = run_expert_episode(
                _expert_config(cfg, int(target_count)),
                seed=int(seed), collect_observations=False,
            )
            item = {
                "target_count": int(target_count),
                "success": bool(result.success),
                "collected": int(result.collected),
                "reason": str(result.failure_reason),
                "planner_status": str(result.planner_status),
                "planner_strategy": str(result.planner_strategy),
            }
        except Exception as exc:
            # IK/collision failures are rejected layout attempts, not batch
            # process failures.  Keep the reason in the lightweight screening
            # result so the caller can advance to the next seed.
            item = {
                "target_count": int(target_count),
                "success": False,
                "collected": 0,
                "reason": f"{type(exc).__name__}: {exc}",
                "planner_status": "exception",
                "planner_strategy": "screening_rejected",
            }
        outcomes.append(item)
        accepted = accepted and bool(item["success"])
        if not accepted:
            break
    return accepted, outcomes


def find_shared_layout_seed(cfg, requested_seed: int, max_attempts: int,
                            failure_root: str | Path | None = None,
                            layout_id: str = "paired_000") -> tuple[int, int]:
    """Find a same-layout seed that succeeds for all targets 1..6."""
    requested_seed = int(requested_seed)
    max_attempts = int(max_attempts)
    if requested_seed < 0 or max_attempts < 1:
        raise ValueError("seed must be non-negative and max_attempts positive")
    for attempt in range(max_attempts):
        seed = requested_seed + attempt
        accepted, outcomes = screen_shared_layout(cfg, seed)
        if accepted:
            return seed, attempt + 1
        if failure_root is not None:
            _append_failure_log(Path(failure_root), {
                "layout_id": str(layout_id),
                "layout_kind": "paired",
                "requested_seed": requested_seed,
                "seed": seed,
                "generation_attempt": attempt + 1,
                "outcomes": outcomes,
            })
    raise RuntimeError(
        f"no shared layout succeeded for targets 1..6 after {max_attempts} seeds"
    )


def promote_preview_batch(preview_root: str | Path,
                          dataset_root: str | Path,
                          layout_id: str = "paired_000") -> list[dict]:
    """Promote one approved same-layout preview batch into the train set.

    The copy is lossless: synchronized images, telemetry and actions are kept,
    while only provenance changes from ``expert_preview/pilot`` to
    ``expert/train``.  It refuses mixed seeds, duplicate goals, failures or a
    non-empty destination so stale data cannot leak into the rebuilt set.
    """
    preview_root = Path(preview_root)
    dataset_root = Path(dataset_root)
    destination_has_files = (
        dataset_root.exists() and any(dataset_root.iterdir())
    )
    if destination_has_files:
        raise ValueError("training dataset destination must be empty before preview promotion")
    records = [
        record for record in _read_manifest(preview_root)
        if bool(record.get("success", False))
        and int(record.get("schema_version", 0)) == 4
        and str(record.get("episode_kind", "")) == "expert_preview"
        and str(record.get("layout_id", "")) == str(layout_id)
    ]
    by_target = {int(record.get("target_count", 0)): record for record in records}
    if set(by_target) != set(range(1, 7)) or len(records) != 6:
        raise ValueError(
            "approved preview batch must contain exactly one successful target 1..6 record"
        )
    seeds = {int(record.get("seed", -1)) for record in records}
    if len(seeds) != 1:
        raise ValueError("approved preview batch does not share one physical layout seed")
    dataset_root.mkdir(parents=True, exist_ok=True)
    promoted = []
    for target_count in range(1, 7):
        record = dict(by_target[target_count])
        episode_id = str(record["episode_id"])
        source = preview_root / episode_id
        destination = dataset_root / episode_id
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, destination)
        record.update({
            "episode_kind": "expert",
            "split": "train",
            "preview": False,
            "layout_id": str(layout_id),
            "layout_kind": "paired",
            "promoted_from_preview": True,
        })
        # A* may expose the chosen identities while building a review preview,
        # but identity is neither a policy input nor part of the approved
        # training contract.  Promotion deliberately removes it so the formal
        # dataset cannot encode the planner's private selection decision.
        record.pop("target_indices", None)
        promoted.append(record)
    with (dataset_root / "manifest.jsonl").open(
            "w", encoding="utf-8") as handle:
        for record in promoted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return promoted


def write_split_manifests(root: str | Path,
                          validation_seed: int = 60_000,
                          test_seed: int = 90_000) -> dict:
    """Write the frozen, disjoint validation/test layout assignments."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = {}
    for split, seed, count in (
        ("val", validation_seed, 10),
        ("test", test_seed, 20),
    ):
        specs = evaluation_layout_plan(split, seed, count)
        path = root / f"{split}_layouts.jsonl"
        path.write_text("".join(
            json.dumps(spec.__dict__, ensure_ascii=False) + "\n"
            for spec in specs
        ), encoding="utf-8")
        output[split] = {"path": str(path), "count": len(specs)}
    return output


def _expert_metadata(spec: EpisodeSpec, result, requested_seed: int,
                     generation_attempt: int, cfg) -> dict:
    planner_score = float(getattr(result, "planner_score", float("inf")))
    return {
        "episode_kind": "expert",
        "split": "train",
        "seed": int(spec.seed),
        "requested_seed": int(requested_seed),
        "generation_attempt": int(generation_attempt),
        "layout_id": str(spec.layout_id),
        "layout_kind": str(spec.layout_kind),
        "count": 6,
        "total_count": 6,
        "target_count": int(spec.target_count),
        "collected": int(result.collected),
        "geometry": "mixed",
        "fps": float(cfg.act.action_hz),
        "planner": "astar_one_pass",
        "planner_status": str(result.planner_status),
        "planner_failure_reason": str(result.planner_failure_reason),
        "planner_strategy": str(result.planner_strategy),
        "planner_turn_count": int(result.planner_turn_count),
        "planner_attempts": int(result.planner_attempts),
        "planner_score": planner_score if np.isfinite(planner_score) else None,
        "target_mode": "exact",
        "spawn_mode": str(cfg.components.get("spawn_mode", "cluster")),
        "generation_outcome": "success",
        "failure_reason": "",
        "peak_force": float(result.peak_force),
        "first_contact_position": np.asarray(
            result.first_contact_position, dtype=float
        ).tolist(),
    }


def _capture_training_episode(cfg, writer: ActDatasetWriter,
                              root: Path, spec: EpisodeSpec,
                              requested_seed: int,
                              generation_attempt: int) -> dict:
    local_cfg = _expert_config(cfg, spec.target_count)
    result = run_expert_episode(
        local_cfg, seed=int(spec.seed), collect_observations=True
    )
    if not result.success:
        raise RuntimeError(
            f"screened layout changed during capture: {result.failure_reason}"
        )
    episode_id = next_demo_name([root], spec.target_count)
    metadata = _expert_metadata(
        spec, result, requested_seed, generation_attempt, local_cfg
    )
    writer.add_episode(
        episode_id, result.observations, result.actions, True, metadata
    )
    return {"episode_id": episode_id, **metadata}


def _screen_single_layout(cfg, target_count: int, requested_seed: int,
                          max_attempts: int, failure_root: Path,
                          layout_id: str) -> tuple[int, int]:
    for attempt in range(int(max_attempts)):
        seed = int(requested_seed) + attempt
        result = run_expert_episode(
            _expert_config(cfg, target_count), seed=seed,
            collect_observations=False,
        )
        if result.success:
            return seed, attempt + 1
        _append_failure_log(failure_root, {
            "layout_id": str(layout_id),
            "layout_kind": "independent",
            "target_count": int(target_count),
            "requested_seed": int(requested_seed),
            "seed": seed,
            "generation_attempt": attempt + 1,
            "collected": int(result.collected),
            "reason": str(result.failure_reason),
            "planner_status": str(result.planner_status),
            "planner_strategy": str(result.planner_strategy),
        })
    raise RuntimeError(
        f"no successful layout for target {target_count} after {max_attempts} seeds"
    )


def generate_approved_training_dataset(cfg, root: str | Path,
                                       seed_base: int = 4100,
                                       max_attempts: int = 64) -> dict:
    """Complete the approved 120-record dataset after preview promotion.

    ``paired_000`` must already be the approved six-record preview batch.
    Every rejected attempt is kept only in ``generation_failures.jsonl``.
    The function is restartable: already completed layout/target slots are
    skipped after strict schema/provenance checks.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest(root)
    slots = {
        (str(record.get("layout_id", "")), int(record.get("target_count", 0)))
        for record in existing
        if bool(record.get("success", False))
        and int(record.get("schema_version", 0)) == 4
        and str(record.get("episode_kind", "")) == "expert"
        and str(record.get("split", "")) == "train"
    }
    required_preview = {("paired_000", target) for target in range(1, 7)}
    if not required_preview.issubset(slots):
        raise ValueError(
            "paired_000 approved preview batch must be promoted before full generation"
        )
    writer = ActDatasetWriter(str(root))
    generated = []

    # Complete paired layouts 1..7 atomically at the seed-selection level.
    for paired_index in range(1, 8):
        layout_id = f"paired_{paired_index:03d}"
        missing = [
            target for target in range(1, 7)
            if (layout_id, target) not in slots
        ]
        if not missing:
            continue
        if len(missing) != 6:
            raise ValueError(
                f"partial paired layout {layout_id} found; remove it before resume"
            )
        requested_seed = int(seed_base) + paired_index * 1_000
        accepted_seed, attempt = find_shared_layout_seed(
            cfg, requested_seed, max_attempts,
            failure_root=root, layout_id=layout_id,
        )
        for target_count in range(1, 7):
            spec = EpisodeSpec(
                split="train", target_count=target_count,
                seed=accepted_seed, layout_id=layout_id,
                layout_kind="paired",
            )
            generated.append(_capture_training_episode(
                cfg, writer, root, spec, requested_seed, attempt
            ))
            slots.add((layout_id, target_count))

    # Add twelve seed-independent layouts for each target cardinality.
    for planned in training_episode_plan(seed_base):
        if planned.layout_kind != "independent":
            continue
        slot = (planned.layout_id, planned.target_count)
        if slot in slots:
            continue
        accepted_seed, attempt = _screen_single_layout(
            cfg, planned.target_count, planned.seed, max_attempts,
            root, planned.layout_id,
        )
        spec = EpisodeSpec(
            split="train", target_count=planned.target_count,
            seed=accepted_seed, layout_id=planned.layout_id,
            layout_kind="independent",
        )
        generated.append(_capture_training_episode(
            cfg, writer, root, spec, planned.seed, attempt
        ))
        slots.add(slot)

    validation = validate_training_manifest(root)
    split_manifests = write_split_manifests(root)
    return {
        **validation,
        "generated": len(generated),
        "split_manifests": split_manifests,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument(
        "--max-attempts-per-episode", type=int, default=6,
        help="MuJoCo/layout retries before retaining a failure (default: 6)",
    )
    parser.add_argument("--split", choices=["pilot", "train", "val", "test"], default="train")
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument(
        "--episode-plan", default=None,
        help="comma-separated target_count:seed pairs; overrides --episodes/--seed-base",
    )
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument(
        "--promote-preview",
        default=None,
        metavar="PREVIEW_DIR",
        help="promote the approved paired_000 six-preview batch into an empty train set",
    )
    parser.add_argument(
        "--approved-plan",
        action="store_true",
        help="complete the approved 120-record composition after preview promotion",
    )
    parser.add_argument(
        "--write-split-manifests-only",
        action="store_true",
        help="write the frozen 60 validation and 120 test layout assignments",
    )
    parser.add_argument("--composition", choices=["balanced", "mixed", "same"],
                        default="balanced")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    if args.max_attempts_per_episode < 1:
        parser.error("--max-attempts-per-episode must be positive")
    cfg = load_config(args.config, overrides=args.set)
    root = args.out or str(cfg.act.dataset_dir)
    root_path = Path(root)
    if args.promote_preview is not None:
        promoted = promote_preview_batch(args.promote_preview, root_path)
        print(json.dumps({
            "promoted": len(promoted),
            "dataset": str(root_path),
            "layout_id": "paired_000",
        }, ensure_ascii=False))
        if not args.approved_plan and not args.write_split_manifests_only:
            return 0
    if args.write_split_manifests_only:
        output = write_split_manifests(root_path)
        print(json.dumps(output, ensure_ascii=False))
        if not args.approved_plan:
            return 0
    if args.approved_plan:
        save_config(cfg, str(root_path / "train_config.yaml"))
        output = generate_approved_training_dataset(
            cfg, root_path, seed_base=int(args.seed_base),
            max_attempts=int(args.max_attempts_per_episode),
        )
        print(json.dumps(output, ensure_ascii=False))
        return 0
    writer = ActDatasetWriter(root)
    preview_root = Path(root).parent / "workbench_previews"
    save_config(cfg, f"{root}/{args.split}_config.yaml")
    if args.episode_plan is not None:
        try:
            episode_plan = parse_episode_plan(args.episode_plan)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        episode_plan = [
            (1 + (i % 6), int(args.seed_base) + i)
            for i in range(int(args.episodes))
        ]
    for i, (target_count, seed) in enumerate(episode_plan):
        local_cfg = cfg.copy()
        local_cfg.set_path("components.count", 6)
        local_cfg.set_path("task.total_count", 6)
        local_cfg.set_path("task.target_count", target_count)
        if args.composition == "mixed" or (args.composition == "balanced" and i % 2 == 0):
            geometry = "mixed"
        else:
            geometry = GEOMETRY_NAMES[(i // 2 if args.composition == "balanced" else i)
                                      % len(GEOMETRY_NAMES)]
        local_cfg.set_path("components.geometry", geometry)
        # A fixed seed is the first reproducible proposal, not a reason to
        # waste a dataset slot on a layout that is geometrically valid but
        # dynamically unlucky.  Each retry changes only the layout seed; the
        # requested target count and component composition remain fixed.
        max_attempts = 1 if args.all_episodes else int(args.max_attempts_per_episode)
        accepted_seed = int(seed)
        generation_attempt = 0
        while True:
            accepted_seed = int(seed) + generation_attempt * 1000003
            result = run_expert_episode(
                local_cfg, seed=accepted_seed, collect_observations=True)
            generation_attempt += 1
            if result.success or args.all_episodes or generation_attempt >= max_attempts:
                break
        if result.success or args.all_episodes:
            episode_id = next_demo_name([Path(root), preview_root], target_count)
            writer.add_episode(
                episode_id, result.observations, result.actions,
                result.success, {"split": args.split, "seed": accepted_seed,
                                  "requested_seed": seed,
                                  "generation_attempt": generation_attempt,
                                  "episode_kind": "expert",
                                  "layout_id": f"legacy_{args.split}_{i:04d}",
                                  "layout_kind": "independent",
                                  "total_count": 6, "target_count": target_count,
                                  "count": 6, "geometry": geometry,
                                  "fps": float(local_cfg.act.action_hz),
                                  "planner": "astar_one_pass",
                                  "planner_status": result.planner_status,
                                  "planner_failure_reason": result.planner_failure_reason,
                                  "planner_strategy": result.planner_strategy,
                                  "planner_turn_count": result.planner_turn_count,
                                  "planner_attempts": result.planner_attempts,
                                  "planner_score": (result.planner_score
                                                     if result.planner_score != float("inf")
                                                     else None),
                                  "target_mode": str(local_cfg.task.target_mode),
                                  "spawn_mode": str(local_cfg.components.get("spawn_mode", "uniform")),
                                  "target_indices": result.target_indices,
                                  "collected": result.collected,
                                  "recovery_used": any(
                                      row.get("phase") == "recovery" for row in result.trace),
                                  "generation_outcome": (
                                      "success" if result.success else "failed"),
                                  "failure_reason": result.failure_reason},
            )
        print(json.dumps({"split": args.split, "index": i, "seed": accepted_seed,
                          "requested_seed": seed,
                          "generation_attempt": generation_attempt,
                          "success": result.success, "collected": result.collected,
                          "target": target_count, "total": result.total,
                          "geometry": geometry,
                          "planner_status": result.planner_status,
                          "planner_strategy": result.planner_strategy,
                          "planner_failure_reason": result.planner_failure_reason,
                          "failure_reason": result.failure_reason}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
