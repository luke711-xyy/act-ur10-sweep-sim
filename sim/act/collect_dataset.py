"""Generate automatic MuJoCo demonstrations for ACT."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config, load_config, save_config
from ..model.geometries import GEOMETRY_NAMES
from .dataset import ActDatasetWriter
from .expert import (expert_selection_audit as _expert_selection_audit,
                     expert_target_subset_mode, expert_generation_protocol)
from .interface import ACT_SCHEMA_VERSION, ACT_STATE_DIM
from .naming import demo_name, next_demo_name, next_demo_serial
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


def training_episode_plan(seed_base: int = 410000,
                          episodes_per_target: int = 20) -> list[EpisodeSpec]:
    """Build independent randomized layouts with an equal count per goal.

    Each training episode gets its own layout seed and ID.  Retry windows are
    spaced apart so a failed candidate for one episode cannot collide with a
    later episode's proposed seed.  The six user-reviewed previews are kept
    outside this training plan.
    """
    seed_base = int(seed_base)
    episodes_per_target = int(episodes_per_target)
    if seed_base < 0 or episodes_per_target < 1:
        raise ValueError(
            "training seed base must be non-negative and episodes per target positive"
        )
    plan: list[EpisodeSpec] = []
    slot_index = 0
    # Start with the hardest all-six capture goal so planner regressions surface
    # before spending generation time on easier partial-count layouts.
    for target_count in range(6, 0, -1):
        for layout_index in range(episodes_per_target):
            seed = seed_base + slot_index * 256
            plan.append(EpisodeSpec(
                split="train", target_count=target_count, seed=seed,
                layout_id=f"train_n{target_count}_layout_{layout_index:03d}",
                layout_kind="independent"))
            slot_index += 1
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


def validate_training_manifest(root: str | Path,
                               episodes_per_target: int = 20,
                               camera_views=None,
                               target_subset_mode: str = "any") -> dict:
    """Require a balanced set of successful schema-v5 experts."""
    root = Path(root)
    episodes_per_target = int(episodes_per_target)
    if episodes_per_target < 1:
        raise ValueError("episodes_per_target must be positive")
    expected_total = 6 * episodes_per_target
    path = root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    _validate_existing_target_selection(records, target_subset_mode)
    expected_views = (set(str(view) for view in camera_views)
                      if camera_views is not None else None)
    invalid = [record.get("episode_id", "<missing>") for record in records if not (
        int(record.get("schema_version", 0)) == ACT_SCHEMA_VERSION
        and int(record.get("action_dim", 0)) == 4
        and int(record.get("state_dim", 0)) == ACT_STATE_DIM
        and str(record.get("episode_kind", "")) == "expert"
        and str(record.get("split", "")) == "train"
        and bool(record.get("success", False))
        and 1 <= int(record.get("target_count", 0)) <= 6
        and 1 <= int(record.get("frame_count", 0))
        <= int(record.get("max_frames", 500))
        and "target_indices" not in record
        and str(record.get("layout_kind", "")) == "independent"
        and bool(str(record.get("layout_id", "")))
        and int(record.get("seed", -1)) >= 0
        and bool(record.get("layout_fingerprint"))
        and (expected_views is None or set(
            str(view) for view in record.get("camera_views", [])
        ) == expected_views)
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
    independent = [
        record for record in records
        if str(record.get("layout_kind", "")) == "independent"
    ]
    layout_ids = [str(record.get("layout_id", "")) for record in independent]
    seeds = [int(record.get("seed", -1)) for record in independent]
    fingerprints = [str(record.get("layout_fingerprint", ""))
                    for record in independent]
    if invalid:
        raise ValueError(f"training manifest contains incompatible records: {invalid}")
    if (len(records) != expected_total
            or any(value != episodes_per_target for value in per_target.values())):
        raise ValueError(
            f"training manifest must contain {expected_total} records and "
            f"{episodes_per_target} per target; "
            f"found total={len(records)}, per_target={per_target}"
        )
    if (len(independent) != expected_total
            or len(set(slots)) != expected_total
            or len(set(layout_ids)) != expected_total
            or len(set(seeds)) != expected_total
            or len(set(fingerprints)) != expected_total):
        raise ValueError(
            f"training manifest must contain {expected_total} mutually unique "
            "independent layouts"
        )
    return {
        "total": len(records),
        "per_target": per_target,
        "independent_records": len(independent),
        "unique_layouts": len(set(fingerprints)),
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
    failure_path = root / "generation_failures.jsonl"
    with (root / ".generation_failures.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_preview_training_manifest(root: str | Path,
                                       per_target: int = 15) -> dict:
    """Validate a balanced successful training set with independent layouts."""
    root = Path(root)
    per_target = int(per_target)
    if per_target < 1:
        raise ValueError("per_target must be positive")
    records = _read_manifest(root)
    invalid = [record.get("episode_id", "<missing>") for record in records
               if not (
                   int(record.get("schema_version", 0)) == ACT_SCHEMA_VERSION
                   and int(record.get("action_dim", 0)) == 4
                   and int(record.get("state_dim", 0)) == ACT_STATE_DIM
                   and str(record.get("episode_kind", "")) == "expert"
                   and str(record.get("split", "")) == "train"
                   and bool(record.get("success", False))
                   and int(record.get("count", 0)) == 6
                   and int(record.get("total_count", 0)) == 6
                   and 1 <= int(record.get("target_count", 0)) <= 6
                   and 1 <= int(record.get("frame_count", 0))
                   <= int(record.get("max_frames", 500))
                   and "target_indices" not in record
                   and str(record.get("layout_kind", "")) == "independent"
                   and bool(str(record.get("layout_id", "")))
                   and int(record.get("seed", -1)) >= 0
                   and bool(record.get("layout_fingerprint"))
               )]
    per_target_counts = {
        str(target): sum(int(record.get("target_count", 0)) == target
                         for record in records)
        for target in range(1, 7)
    }
    expected_total = 6 * per_target
    layout_ids = [str(record.get("layout_id", "")) for record in records]
    seeds = [int(record.get("seed", -1)) for record in records]
    fingerprints = [str(record.get("layout_fingerprint", ""))
                    for record in records]
    unique_layouts = (
        len(set(layout_ids)) == expected_total
        and len(set(seeds)) == expected_total
        and len(set(fingerprints)) == expected_total
    )
    if (invalid or len(records) != expected_total or not unique_layouts
            or any(value != per_target
                   for value in per_target_counts.values())):
        raise ValueError(
            f"preview training manifest must contain {expected_total} successful "
            f"records and {per_target} per target on unique independent layouts; "
            f"invalid={invalid}, unique_layouts={unique_layouts}, "
            f"per_target={per_target_counts}"
        )
    return {"total": len(records), "per_target": per_target_counts,
            "unique_layouts": expected_total,
            "source": "successful_independent_experts"}


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


def _generation_protocol_options(cfg, composition: str = "mixed") -> list[dict | None]:
    """Compare identities of configurations actually used for six-part capture."""
    if not hasattr(cfg, "get_path"):
        return [None]
    local_cfg = _expert_config(cfg, 6)
    if expert_generation_protocol(local_cfg) is None:
        return [None]
    geometries = (["mixed"] if composition == "mixed" else
                  list(GEOMETRY_NAMES) + (["mixed"] if composition == "balanced" else []))
    protocols = []
    for geometry in geometries:
        local_cfg.set_path("components.geometry", geometry)
        protocols.append(expert_generation_protocol(local_cfg))
    return protocols


def _validate_existing_generation_protocol(records: list[dict],
                                           protocols: list[dict | None]) -> None:
    """Exact identity, plus the explicit stronger v2-success -> v3 upgrade."""
    def compatible_stricter_success(record):
        # A previously published success with height checks throughout also
        # satisfies the new post-count waiver. Never relabel its metadata, and
        # never admit old previews, failures, another layout or a reverse run.
        if (record.get("success") is not True
                or record.get("episode_kind") != "expert"
                or record.get("split") != "train"):
            return False
        for protocol in protocols:
            if (not isinstance(protocol, dict)
                    or protocol.get("name") != "coverage_astar_layout_clearance_v3"
                    or protocol.get("height_guard") != "until_exact_count"):
                continue
            strict = {k:v for k,v in protocol.items() if k != "height_guard"}
            strict["name"] = "coverage_astar_layout_clearance_v2"
            if record.get("expert_generation_protocol") == strict:
                return True
        return False

    wrong = [r.get("episode_id", "<missing>") for r in records
             if r.get("expert_generation_protocol") not in protocols
             and not compatible_stricter_success(r)]
    if wrong:
        raise ValueError("existing data has an incompatible generation protocol; "
                         "preserve it and explicitly choose the new batch transition: "
                         f"{wrong[:5]}")


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
        outcomes.append(item)
        accepted = accepted and bool(result.success)
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
        and int(record.get("schema_version", 0)) == ACT_SCHEMA_VERSION
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
                          validation_seed: int = 1_000_000,
                          test_seed: int = 2_000_000) -> dict:
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


def _validate_existing_target_selection(records: list[dict], mode: str) -> None:
    """Reject missing/invalid selection provenance before counting resumed slots."""
    if mode == "any":
        return
    from types import SimpleNamespace
    cfg = Config({"planner": {"expert_target_subset_mode": mode}})
    for record in records:
        selection = record.get("target_selection")
        try:
            if not isinstance(selection, dict) or selection.get("mode") != mode:
                raise ValueError("missing or incompatible selection audit")
            positions = np.asarray(selection["initial_positions"], dtype=float)
            collected = selection["collected_component_ids"]
            result = SimpleNamespace(
                initial_object_positions=positions,
                target_indices=selection["planned_component_ids"],
                final_collected_mask=np.isin(np.arange(6), collected),
            )
            checked = _expert_selection_audit(result, cfg, int(record["target_count"]))
            if (selection["y_order"] != checked["y_order"]
                    or collected != checked["collected_component_ids"]):
                raise ValueError("inconsistent selection identities or Y ranks")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"invalid target selection for {record.get('episode_id', '<missing>')}: {exc}"
            ) from exc


def _expert_metadata(spec: EpisodeSpec, result, requested_seed: int,
                     generation_attempt: int, cfg) -> dict:
    planner_score = float(getattr(result, "planner_score", float("inf")))
    initial_pose = np.asarray(
        result.observations[0]["object_pose"], dtype="<f4"
    )
    layout_fingerprint = hashlib.sha256(
        np.ascontiguousarray(initial_pose).tobytes()
    ).hexdigest()
    selection = _expert_selection_audit(result, cfg, spec.target_count)
    protocol = expert_generation_protocol(cfg)
    return {
        **({"expert_generation_protocol": protocol} if protocol is not None else {}),
        **({"expert_landing_region": result.expert_landing_region}
           if getattr(result, "expert_landing_region", None) is not None else {}),
        **({"target_selection": selection} if selection is not None else {}),
        "episode_kind": "expert",
        "split": "train",
        "seed": int(spec.seed),
        "requested_seed": int(requested_seed),
        "generation_attempt": int(generation_attempt),
        "layout_id": str(spec.layout_id),
        "layout_kind": str(spec.layout_kind),
        "layout_fingerprint": layout_fingerprint,
        "count": 6,
        "total_count": 6,
        "target_count": int(spec.target_count),
        "collected": int(result.collected),
        "geometry": "mixed",
        "fps": float(cfg.act.action_hz),
        "max_frames": int(cfg.episode.get("max_frames", 500)),
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
        "desired_force": float(cfg.controller.desired_force),
        "peak_force": float(result.peak_force),
        "peak_raw_force": float(getattr(result, "peak_raw_force", 0.0)),
        "force_source": (
            "wrist_ft_sensor_world_z_tared"
            if str(cfg.controller.get("force_sensor", "contact")) == "wrist_ft"
            else "simulated_contact_geometry"
        ),
        "force_processing": (
            (
                "raw wrist world-Z retained; reset-pose tare; distal payload "
                "inertial-load delta estimated from modeled rigid-body kinematics "
                "and removed for force control; pressing-positive; low-pass; "
                "raw and compensated channels logged separately"
            )
            if str(cfg.controller.get("force_sensor", "contact")) == "wrist_ft"
            else "contact-normal projection; low-pass"
        ),
        "first_contact_position": np.asarray(
            result.first_contact_position, dtype=float
        ).tolist(),
    }


def _capture_training_episode(cfg, writer: ActDatasetWriter,
                              root: Path, spec: EpisodeSpec,
                              requested_seed: int,
                              generation_attempt: int,
                              episode_id: str | None = None,
                              candidate_workers: int | None = None) -> dict:
    local_cfg = _expert_config(cfg, spec.target_count)
    if candidate_workers is not None:
        local_cfg.set_path(
            "planner.expert_validation_workers", max(1, int(candidate_workers))
        )
    result = run_expert_episode(
        local_cfg, seed=int(spec.seed), collect_observations=True
    )
    if not result.success:
        raise RuntimeError(
            f"expert rollout failed: {result.failure_reason}; "
            f"collected={result.collected}/{result.target_count}; "
            f"planner={result.planner_status}/{result.planner_strategy}; "
            f"turns={result.planner_turn_count}; "
            f"candidate={result.planner_attempts}; "
            f"frames={result.sampled_frames}"
        )
    episode_id = (str(episode_id) if episode_id is not None
                  else next_demo_name([root], spec.target_count))
    metadata = _expert_metadata(
        spec, result, requested_seed, generation_attempt, local_cfg
    )
    writer.add_episode(
        episode_id, result.observations, result.actions, True, metadata,
        control_trace=result.trace,
    )
    return {"episode_id": episode_id, **metadata}


def _capture_approved_training_slot(cfg, root: str, planned: EpisodeSpec,
                                    max_attempts: int, image_format: str,
                                    episode_id: str,
                                    previously_failed_seeds: list[int],
                                    candidate_workers: int) -> dict:
    """Generate one independent layout slot, retrying only its seed window."""
    if isinstance(cfg, dict):
        cfg = Config(cfg)
    root_path = Path(root)
    writer = ActDatasetWriter(root, image_format=image_format)
    failed_seeds = set(int(seed) for seed in previously_failed_seeds)
    for attempt in range(1, int(max_attempts) + 1):
        seed = int(planned.seed) + attempt - 1
        if seed in failed_seeds:
            continue
        spec = EpisodeSpec(
            split="train", target_count=planned.target_count,
            seed=seed, layout_id=planned.layout_id,
            layout_kind="independent",
        )
        print(json.dumps({
            "event": "expert_attempt_started",
            "target_count": int(planned.target_count),
            "layout_id": str(planned.layout_id),
            "attempt": attempt,
            "seed": seed,
            "episode_id": episode_id,
        }, ensure_ascii=False), flush=True)
        try:
            episode = _capture_training_episode(
                cfg, writer, root_path, spec, int(planned.seed), attempt,
                episode_id=episode_id,
                candidate_workers=candidate_workers,
            )
        except RuntimeError as exc:
            failure = {
                "layout_id": str(planned.layout_id),
                "layout_kind": "independent",
                "target_count": int(planned.target_count),
                "requested_seed": int(planned.seed),
                "seed": seed,
                "generation_attempt": attempt,
                "stage": "rollout",
                "reason": str(exc),
            }
            _append_failure_log(root_path, failure)
            print(json.dumps({
                "event": "expert_attempt_failed",
                **failure,
            }, ensure_ascii=False), flush=True)
            continue
        return episode
    raise RuntimeError(
        f"could not record layout {planned.layout_id} for target "
        f"{planned.target_count} after {max_attempts} seeds"
    )


def _reserve_episode_ids(root: Path, pending: list[EpisodeSpec]) -> dict[tuple[str, int], str]:
    """Reserve distinct canonical names before concurrent writers start."""
    occupied = {str(record.get("episode_id", "")) for record in _read_manifest(root)}
    if root.is_dir():
        occupied.update(child.name for child in root.iterdir() if child.is_dir())
    next_serial_by_target: dict[int, int] = {}
    episode_ids = {}
    for planned in pending:
        target_count = int(planned.target_count)
        if target_count not in next_serial_by_target:
            next_serial_by_target[target_count] = next_demo_serial(
                [root], target_count
            )
        serial = next_serial_by_target[target_count]
        while demo_name(target_count, serial) in occupied:
            serial += 1
        episode_id = demo_name(target_count, serial)
        occupied.add(episode_id)
        next_serial_by_target[target_count] = serial + 1
        episode_ids[(planned.layout_id, planned.target_count)] = episode_id
    return episode_ids


def _worker_config_payload(cfg):
    """Convert nested Config objects to spawn-safe plain mappings."""
    to_dict = getattr(cfg, "to_dict", None)
    return to_dict() if callable(to_dict) else cfg


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
                                       seed_base: int = 410000,
                                       max_attempts: int = 64,
                                       episodes_per_target: int = 20,
                                       image_format: str = "png",
                                       workers: int = 1) -> dict:
    """Generate a balanced set of successful demonstrations on unique layouts.

    The accepted six-record pilot remains in the preview directory and is not
    promoted. Rejected candidates are recorded only as lightweight failure
    reasons. Completed v5 slots are skipped on restart.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest(root)
    incompatible = [record.get("episode_id", "<missing>") for record in existing
                    if int(record.get("schema_version", 0)) != ACT_SCHEMA_VERSION
                    or int(record.get("state_dim", 0)) != ACT_STATE_DIM
                    or str(record.get("episode_kind", "")) != "expert"
                    or str(record.get("split", "")) != "train"
                    or str(record.get("layout_kind", "")) != "independent"
                    or not str(record.get("layout_fingerprint", ""))]
    if incompatible:
        raise ValueError(
            "formal dataset root contains preview/legacy records; use a fresh "
            f"training directory instead: {incompatible[:5]}"
        )
    _validate_existing_generation_protocol(existing, _generation_protocol_options(cfg))
    subset_mode = (expert_target_subset_mode(cfg)
                   if hasattr(cfg, "get_path") else "any")
    _validate_existing_target_selection(existing, subset_mode)
    if not 1 <= int(max_attempts) < 256:
        raise ValueError("max_attempts must be between 1 and 255")
    episodes_per_target = int(episodes_per_target)
    if episodes_per_target < 1:
        raise ValueError("episodes_per_target must be positive")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    slots = {
        (str(record.get("layout_id", "")), int(record.get("target_count", 0)))
        for record in existing
        if bool(record.get("success", False))
        and int(record.get("schema_version", 0)) == ACT_SCHEMA_VERSION
        and str(record.get("episode_kind", "")) == "expert"
        and str(record.get("split", "")) == "train"
    }
    failure_path = root / "generation_failures.jsonl"
    failed_seeds: dict[tuple[str, int], set[int]] = {}
    if failure_path.is_file():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            failure = json.loads(line)
            slot = (str(failure.get("layout_id", "")),
                    int(failure.get("target_count", 0)))
            failed_seeds.setdefault(slot, set()).add(int(failure["seed"]))
    writer = ActDatasetWriter(str(root), image_format=image_format)
    generated = []
    completed = len(slots)
    plan = training_episode_plan(seed_base, episodes_per_target)
    pending = [planned for planned in plan
               if (planned.layout_id, planned.target_count) not in slots]
    if workers == 1:
        scheduled = pending
        for planned in scheduled:
            slot = (planned.layout_id, planned.target_count)
            if slot in slots:
                continue
            episode = None
            accepted_seed = int(planned.seed)
            for attempted_seeds in range(1, int(max_attempts) + 1):
                accepted_seed = int(planned.seed) + attempted_seeds - 1
                if accepted_seed in failed_seeds.get(slot, set()):
                    continue
                spec = EpisodeSpec(
                    split="train", target_count=planned.target_count,
                    seed=accepted_seed, layout_id=planned.layout_id,
                    layout_kind="independent",
                )
                print(json.dumps({
                    "event": "expert_attempt_started",
                    "completed": completed,
                    "total": 6 * episodes_per_target,
                    "target_count": int(planned.target_count),
                    "layout_id": str(planned.layout_id),
                    "attempt": attempted_seeds,
                    "seed": int(accepted_seed),
                }, ensure_ascii=False), flush=True)
                try:
                    episode = _capture_training_episode(
                        cfg, writer, root, spec, planned.seed, attempted_seeds
                    )
                except RuntimeError as exc:
                    failure = {
                        "layout_id": str(planned.layout_id),
                        "layout_kind": "independent",
                        "target_count": int(planned.target_count),
                        "requested_seed": int(planned.seed),
                        "seed": int(accepted_seed),
                        "generation_attempt": int(attempted_seeds),
                        "stage": "rollout",
                        "reason": str(exc),
                    }
                    _append_failure_log(root, failure)
                    print(json.dumps({
                        "event": "expert_attempt_failed",
                        "completed": completed,
                        **failure,
                    }, ensure_ascii=False), flush=True)
                    continue
                generated.append(episode)
                break
            if episode is None:
                raise RuntimeError(
                    f"could not record layout {planned.layout_id} for target "
                    f"{planned.target_count} after {max_attempts} seeds"
                )
            slots.add(slot)
            completed += 1
            print(json.dumps({
                "event": "expert_episode_saved",
                "completed": completed,
                "total": 6 * episodes_per_target,
                "target_count": int(planned.target_count),
                "layout_id": str(planned.layout_id),
                "seed": int(accepted_seed),
            }), flush=True)
    else:
        episode_ids = _reserve_episode_ids(root, pending)
        try:
            candidate_worker_budget = max(1, int(
                cfg.get_path("planner.expert_validation_workers", 1)
            ) // workers)
        except AttributeError:
            candidate_worker_budget = 1
        worker_cfg = _worker_config_payload(cfg)
        with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            # Parallelize layouts within a target, never across the 6→1 stages.
            for target_count in range(6, 0, -1):
                futures = {
                    pool.submit(
                        _capture_approved_training_slot,
                        worker_cfg, str(root), planned, int(max_attempts), image_format,
                        episode_ids[(planned.layout_id, planned.target_count)],
                        sorted(failed_seeds.get(
                            (planned.layout_id, planned.target_count), set())),
                        candidate_worker_budget,
                    ): planned
                    for planned in pending if planned.target_count == target_count
                }
                for future in as_completed(futures):
                    planned = futures[future]
                    episode = future.result()
                    generated.append(episode)
                    slots.add((planned.layout_id, planned.target_count))
                    completed += 1
                    print(json.dumps({
                        "event": "expert_episode_saved",
                        "completed": completed,
                        "total": 6 * episodes_per_target,
                        "target_count": int(planned.target_count),
                        "layout_id": str(planned.layout_id),
                        "seed": int(episode.get("seed", planned.seed)),
                        "episode_id": episode["episode_id"],
                    }), flush=True)

    act_config = getattr(cfg, "act", None)
    configured_views = (act_config.get("camera_views", None)
                        if act_config is not None else None)
    if configured_views is None:
        validation = validate_training_manifest(
            root, episodes_per_target=episodes_per_target,
            **({"target_subset_mode": subset_mode} if subset_mode != "any" else {}),
        )
    else:
        validation = validate_training_manifest(
            root, episodes_per_target=episodes_per_target,
            camera_views=list(configured_views),
            **({"target_subset_mode": subset_mode} if subset_mode != "any" else {}),
        )
    if episodes_per_target == 15:
        validate_preview_training_manifest(root, per_target=15)
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
    parser.add_argument("--seed-base", type=int, default=410000)
    parser.add_argument(
        "--episode-plan", default=None,
        help="comma-separated target_count:seed pairs; overrides --episodes/--seed-base",
    )
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument(
        "--promote-preview",
        default=None,
        metavar="PREVIEW_DIR",
        help="legacy utility to copy a preview set; cannot be combined with --approved-plan",
    )
    parser.add_argument(
        "--approved-plan",
        action="store_true",
        help="generate successful experts on unique randomized layouts",
    )
    parser.add_argument(
        "--episodes-per-target", type=int, default=None,
        help="successful independent training layouts per target with --approved-plan (default: 20)",
    )
    parser.add_argument(
        "--image-format", choices=["png", "webp-lossless"], default="png",
        help="lossless image storage format; WebP reduces disk use without changing RGB pixels",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel full-episode workers for --approved-plan (default: 1)",
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
    if args.approved_plan and args.promote_preview is not None:
        parser.error("an approved randomized training plan cannot promote pilot previews")
    if args.episodes_per_target is not None:
        if not args.approved_plan:
            parser.error("--episodes-per-target requires --approved-plan")
        if args.episodes_per_target < 1:
            parser.error("--episodes-per-target must be positive")
    if args.max_attempts_per_episode < 1:
        parser.error("--max-attempts-per-episode must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers > 1 and not args.approved_plan:
        parser.error("--workers greater than 1 requires --approved-plan")
    cfg = load_config(args.config, overrides=args.set)
    root = args.out or str(cfg.act.dataset_dir)
    root_path = Path(root)
    # Reject incompatible history before saving configs, reserving names, or
    # replacing validation/test assignments. This also covers the normal CLI.
    if args.promote_preview is None and (args.approved_plan or not args.write_split_manifests_only):
        _validate_existing_generation_protocol(
            _read_manifest(root_path),
            _generation_protocol_options(cfg, "mixed" if args.approved_plan else args.composition),
        )
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
            episodes_per_target=(
                20 if args.episodes_per_target is None
                else int(args.episodes_per_target)
            ),
            image_format=args.image_format,
            workers=args.workers,
        )
        print(json.dumps(output, ensure_ascii=False))
        return 0
    writer = ActDatasetWriter(root, image_format=args.image_format)
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
            protocol = expert_generation_protocol(local_cfg)
            writer.add_episode(
                episode_id, result.observations, result.actions,
                result.success, {**({"expert_generation_protocol": protocol}
                                    if protocol is not None else {}),
                                  **({"expert_landing_region": result.expert_landing_region}
                                     if getattr(result, "expert_landing_region", None) is not None else {}),
                                  "split": args.split, "seed": accepted_seed,
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
                control_trace=result.trace,
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
