"""Local state and episode inspection helpers for the MuJoCo workbench.

The workbench deliberately reads the existing ACT manifest format.  Preview
episodes live in a separate directory and therefore cannot accidentally enter
the training dataset.
"""

from __future__ import annotations

import json
import shutil
from threading import RLock
from pathlib import Path

import numpy as np

from ..act.dataset import ActDatasetWriter
from ..act.rollout import run_expert_episode, stall_failure_sidewall_ok
from ..act.naming import next_demo_name


class EpisodeNotFound(LookupError):
    """Requested episode does not exist in the selected local store."""


class FrameNotFound(IndexError):
    """Requested episode frame is outside the recorded range."""


def _resolve_failure_mode(mode: str, target_count: int, ordinal: int = 0) -> str:
    """Resolve the UI's family-level failure choice to a physical variant."""
    mode = str(mode or "")
    if mode != "wrong_count":
        return mode
    target_count = int(target_count)
    # With six physical parts, an over-count at goal 6 cannot exist.  Keep the
    # generated record honest by using the only available wrong-count side.
    if target_count >= 6:
        return "wrong_count_under"
    return "wrong_count_over" if int(ordinal) % 2 == 0 else "wrong_count_under"


def _read_manifest(root: Path) -> list[dict]:
    path = root / "manifest.jsonl"
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A background preview may append the final JSONL line while
                # the dashboard is reading it.  Ignore only that incomplete
                # tail; malformed committed lines must still surface.
                if index == len(lines) - 1:
                    break
                raise
    return records


def _write_manifest(root: Path, records: list[dict]) -> None:
    """Replace a manifest atomically so polling cannot see a half-file."""
    manifest = root / "manifest.jsonl"
    if not records:
        manifest.unlink(missing_ok=True)
        return
    temporary = root / f".{manifest.name}.tmp"
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(manifest)


def _frame_count(root: Path, record: dict) -> int:
    arrays = np.load(root / record["arrays"], mmap_mode="r")
    return int(len(arrays["action"]))


def _image(root: Path, relative: str):
    from PIL import Image

    return np.asarray(Image.open(root / relative).convert("RGB"), dtype=np.uint8)


class WorkbenchState:
    """Expose saved demonstrations and isolated expert previews."""

    def __init__(self, cfg, dataset_root=None, preview_root=None):
        self.cfg = cfg
        self.dataset_root = Path(dataset_root or cfg.act.dataset_dir)
        self.preview_root = Path(
            preview_root or Path(cfg.logging.out_dir) / "workbench_previews")
        # Episode refreshes run while the UI can delete a record or a preview
        # job can publish one.  Keep manifest reads and directory mutations
        # from observing a half-committed episode.
        self._io_lock = RLock()
        self._preview_records: dict[str, dict] = {}

    def _records(self) -> list[tuple[Path, dict, bool]]:
        with self._io_lock:
            records = [(self.dataset_root, rec, False)
                       for rec in _read_manifest(self.dataset_root)]
            preview_records = _read_manifest(self.preview_root)
            self._preview_records = {
                rec["episode_id"]: rec for rec in preview_records
            }
            records.extend((self.preview_root, rec, True)
                           for rec in preview_records)
            return records

    def list_episodes(self) -> list[dict]:
        with self._io_lock:
            output = []
            for root, rec, preview in self._records():
                output.append({
                    "episode_id": rec["episode_id"],
                    "success": bool(rec.get("success", False)),
                    "seed": rec.get("seed"),
                    "count": rec.get("total_count", rec.get("count")),
                    "total_count": rec.get("total_count", rec.get("count")),
                    "target_count": rec.get("target_count"),
                    "collected": rec.get("collected"),
                    "geometry": rec.get("geometry", "unknown"),
                    "split": rec.get("split", "preview" if preview else "unknown"),
                    "fps": rec.get("fps", 25.0),
                    "planner": rec.get("planner", "legacy"),
                    "planner_status": rec.get("planner_status", "unknown"),
                    "planner_failure_reason": rec.get("planner_failure_reason", ""),
                    "planner_strategy": rec.get("planner_strategy", "unknown"),
                    "planner_turn_count": rec.get("planner_turn_count", 0),
                    "planner_attempts": rec.get("planner_attempts", 0),
                    "planner_score": rec.get("planner_score"),
                    "generation_outcome": rec.get("generation_outcome", "success"),
                    "failure_mode": rec.get("failure_mode", ""),
                    "deliberate_failure": bool(rec.get("deliberate_failure", False)),
                    "target_mode": rec.get("target_mode", "unknown"),
                    "target_indices": rec.get("target_indices", []),
                    "target_collected": rec.get("target_collected"),
                    "unexpected_collected": rec.get("unexpected_collected"),
                    "recovery_used": bool(rec.get("recovery_used", False)),
                    "schema_version": rec.get("schema_version", 1),
                    "episode_kind": rec.get("episode_kind", "legacy"),
                    "action_dim": rec.get("action_dim"),
                    "state_dim": rec.get("state_dim"),
                    "layout_id": rec.get("layout_id", ""),
                    "layout_kind": rec.get("layout_kind", ""),
                    "first_contact_position": rec.get("first_contact_position"),
                    "peak_force": rec.get("peak_force"),
                    "length": _frame_count(root, rec),
                    "duration": _frame_count(root, rec) / max(float(rec.get("fps", 25.0)), 1.0),
                    "failure_reason": rec.get("failure_reason", ""),
                    "preview": preview,
                })
            return output

    def _find(self, episode_id: str) -> tuple[Path, dict, bool]:
        for root, rec, preview in self._records():
            if rec.get("episode_id") == episode_id:
                return root, rec, preview
        raise EpisodeNotFound(episode_id)

    def episode_metadata(self, episode_id: str) -> dict:
        with self._io_lock:
            root, rec, preview = self._find(episode_id)
            return {**rec, "length": _frame_count(root, rec), "preview": preview}

    def delete_episode(self, episode_id: str) -> dict:
        """Delete exactly one episode directory and its manifest record."""
        with self._io_lock:
            root, rec, preview = self._find(episode_id)
            episode_id = str(episode_id)
            episode_dir = root / episode_id
            if episode_dir.resolve().parent != root.resolve() or not episode_dir.is_dir():
                raise EpisodeNotFound(episode_id)

            # Resolve the record again from the manifest so a malformed or stale
            # directory cannot cause an unrelated entry to be removed.
            records = _read_manifest(root)
            if not any(str(item.get("episode_id")) == episode_id for item in records):
                raise EpisodeNotFound(episode_id)
            shutil.rmtree(episode_dir)
            remaining = [item for item in records
                         if str(item.get("episode_id")) != episode_id]
            _write_manifest(root, remaining)
            self._preview_records.pop(episode_id, None)
            return {"episode_id": episode_id, "preview": bool(preview),
                    "success": bool(rec.get("success", False)), "deleted": True}

    def load_episode_frame(self, episode_id: str, frame_index: int) -> dict:
        with self._io_lock:
            root, rec, _ = self._find(episode_id)
            length = _frame_count(root, rec)
            frame_index = int(frame_index)
            if frame_index < 0 or frame_index >= length:
                raise FrameNotFound(f"{episode_id}:{frame_index}")
            frame = {
                "overhead": _image(root, rec["overhead"][frame_index]),
                "wrist": _image(root, rec["wrist"][frame_index]),
                "inspection": None,
            }
            inspection = rec.get("inspection")
            if inspection and inspection[frame_index]:
                frame["inspection"] = _image(root, inspection[frame_index])
            return frame

    def load_episode_signals(self, episode_id: str) -> dict:
        """Load the persisted full-run force trace for an episode.

        Older dataset episodes do not have a trace sidecar.  Returning an
        empty, well-shaped response keeps the workbench useful for those
        episodes while making the provenance of the plotted signal explicit.
        """
        with self._io_lock:
            root, rec, _ = self._find(episode_id)
            arrays = np.load(root / rec["arrays"], allow_pickle=False)
        if "t" in arrays.files:
            def matrix(name, width):
                values = (np.asarray(arrays[name], dtype=float)
                          if name in arrays.files else np.zeros((len(arrays["action"]), width)))
                return [[float(item) for item in row] for row in values]

            env_state = matrix("environment_state", 3)
            return {
                "episode_id": rec["episode_id"],
                "t": [float(value) for value in arrays["t"]],
                "phase": [str(value) for value in arrays["phase"]],
                "policy_mask": [bool(value) for value in arrays["action_valid"]],
                "contact": [bool(value) for value in arrays["contact"]],
                "contact_latched": [
                    bool(value) for value in (
                        arrays["contact_latched"]
                        if "contact_latched" in arrays.files else arrays["contact"]
                    )
                ],
                "fz": [float(value) for value in arrays["normal_force"]],
                "tcp": matrix("tcp_pose", 4),
                "joint_position": matrix("joint_position", 6),
                "wrench": matrix("wrench", 6),
                "action": matrix("action", int(arrays["action"].shape[1])),
                "reference": matrix("reference", 4),
                "policy_reference": matrix("policy_reference", 4),
                "policy_z": [
                    float(value) for value in (
                        arrays["policy_z"] if "policy_z" in arrays.files
                        else arrays["reference"][:, 2]
                    )
                ],
                "applied_z": [
                    float(value) for value in (
                        arrays["applied_z"] if "applied_z" in arrays.files
                        else arrays["reference"][:, 2]
                    )
                ],
                "z_owner": [
                    str(value) for value in (
                        arrays["z_owner"] if "z_owner" in arrays.files
                        else np.asarray(["unknown"] * len(arrays["action"]))
                    )
                ],
                "environment_state": env_state,
                "fully_collected": [int(value) for value in arrays["fully_collected"]],
            }

        # Backward-compatible fallback for the one legacy preview format.
        path = root / rec["episode_id"] / "signals.json"
        if not path.is_file():
            return {"episode_id": rec["episode_id"], "t": [], "fz": [],
                    "contact": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"episode_id": rec["episode_id"],
                "t": [float(value) for value in data.get("t", [])],
                "fz": [float(value) for value in data.get("fz", [])],
                "contact": [bool(value) for value in data.get("contact", [])]}

    def load_episode_frame_data(self, episode_id: str, frame_index: int) -> dict:
        with self._io_lock:
            root, rec, _ = self._find(episode_id)
            arrays = np.load(root / rec["arrays"], allow_pickle=False)
        length = len(arrays["action"])
        index = int(frame_index)
        if index < 0 or index >= length:
            raise FrameNotFound(f"{episode_id}:{index}")

        def vector(name, width):
            if name not in arrays.files:
                return [0.0] * width
            return [float(value) for value in np.asarray(arrays[name][index]).reshape(width)]

        env_state = vector("environment_state", int(arrays["environment_state"].shape[1]))
        objects = []
        if "object_pose" in arrays.files:
            poses = np.asarray(arrays["object_pose"][index], dtype=float)
            flags = (np.asarray(arrays["object_collected"][index], dtype=bool)
                     if "object_collected" in arrays.files
                     else np.zeros(len(poses), dtype=bool))
            objects = [{"index": int(i),
                        "position": [float(value) for value in pose[:3]],
                        "quaternion": [float(value) for value in pose[3:7]],
                        "collected": bool(flags[i])}
                       for i, pose in enumerate(poses)]
        return {
            "episode_id": episode_id,
            "frame": index,
            "t": float(arrays["t"][index]) if "t" in arrays.files else index / 25.0,
            "phase": str(arrays["phase"][index]) if "phase" in arrays.files else "unknown",
            "policy_mask": bool(arrays["action_valid"][index]) if "action_valid" in arrays.files else True,
            "contact": bool(arrays["contact"][index]) if "contact" in arrays.files else False,
            "contact_latched": bool(
                arrays["contact_latched"][index]
                if "contact_latched" in arrays.files else arrays["contact"][index]
            ) if "contact" in arrays.files else False,
            "joint_position": vector("joint_position", 6),
            "tcp_pose": vector("tcp_pose", 4),
            "wrench": vector("wrench", 6),
            "normal_force": float(arrays["normal_force"][index]) if "normal_force" in arrays.files else 0.0,
            "action": vector("action", int(arrays["action"].shape[1])),
            "reference": vector("reference", 4),
            "policy_reference": vector("policy_reference", 4),
            "policy_z": float(
                arrays["policy_z"][index]
                if "policy_z" in arrays.files else arrays["reference"][index, 2]
            ),
            "applied_z": float(
                arrays["applied_z"][index]
                if "applied_z" in arrays.files else arrays["reference"][index, 2]
            ),
            "z_owner": str(
                arrays["z_owner"][index]
                if "z_owner" in arrays.files else "unknown"
            ),
            "environment_state": env_state,
            "fully_collected": int(arrays["fully_collected"][index]) if "fully_collected" in arrays.files else 0,
            "objects": objects,
            "target_count": int(rec.get("target_count", env_state[1] if len(env_state) > 1 else 0)),
            "total_count": int(rec.get("total_count", rec.get("count", env_state[0] if env_state else 0))),
            "target_collected": int(rec.get("target_collected", -1)),
            "unexpected_collected": int(rec.get("unexpected_collected", -1)),
            "target_indices": rec.get("target_indices", []),
        }

    def _clone_success_as_wrong_count(self, seed: int, target_count: int,
                                      failure_mode: str,
                                      source_slot: int | None = None) -> dict:
        """Reuse an adjacent exact-success rollout under a different goal.

        This is the intended wrong-count construction: a trajectory that
        physically collected ``n+1`` is a valid over-count failure for goal
        ``n``; likewise a trajectory that collected ``n-1`` is an under-count
        failure for goal ``n``.  Only the task-count channels are relabelled;
        images, object poses, force trace and actions stay exactly physical.
        """
        target_count = int(target_count)
        if failure_mode == "wrong_count_over":
            if target_count >= 6:
                raise ValueError(
                    "wrong_count_over is physically impossible for target 6")
            source_target = target_count + 1
        elif failure_mode == "wrong_count_under":
            if target_count <= 1:
                raise ValueError("target 1 under-count has no adjacent success source")
            source_target = target_count - 1
        else:
            raise ValueError(f"not a wrong-count mode: {failure_mode}")
        records = [record for root, record, preview in self._records()
                   if preview and bool(record.get("success", False))
                   and int(record.get("target_count", -1)) == source_target]
        if not records:
            raise RuntimeError(
                f"no successful source episode available for target {source_target}")
        records.sort(key=lambda record: str(record.get("episode_id", "")))
        # A wrong-count preview is a relabelled physical rollout, but two
        # records of the same target/mode must not be relabelled copies of the
        # same rollout.  Exclude source episodes already used by this target
        # and failure family whenever possible.  ``source_slot`` is supplied
        # by build_previews so the first two records are deterministic and
        # distinct even when their seed stride is a multiple of len(records).
        used_sources = {
            str(record.get("derived_from"))
            for _root, record, preview in self._records()
            if preview
            and not bool(record.get("success", False))
            and int(record.get("target_count", -1)) == target_count
            and str(record.get("failure_mode", "")) == failure_mode
            and record.get("derived_from")
        }
        available = [record for record in records
                     if str(record.get("episode_id")) not in used_sources]
        if not available:
            # More records than adjacent successful sources is unusual, but
            # do not fail generation solely because the pool is exhausted.
            available = records
        if source_slot is None:
            source_index = int(seed) % len(available)
        else:
            source_index = int(source_slot) % len(available)
        source = available[source_index]
        source_id = str(source["episode_id"])
        episode_id = next_demo_name(
            [self.dataset_root, self.preview_root], target_count)
        source_dir = self.preview_root / source_id
        dest_dir = self.preview_root / episode_id
        shutil.copytree(source_dir, dest_dir)

        arrays_path = dest_dir / "arrays.npz"
        arrays = np.load(arrays_path, allow_pickle=False)
        payload = {key: np.asarray(arrays[key]) for key in arrays.files}
        payload["environment_state"][:, 1] = float(target_count)
        # observation.state is [current(20), previous(20)], and the target
        # count is the middle element of each 3-value count triplet.
        payload["state"][:, 18] = float(target_count)
        payload["state"][:, 39] = float(target_count)
        np.savez_compressed(arrays_path, **payload)

        def replace_path(value):
            if isinstance(value, str):
                return value.replace(source_id, episode_id)
            if isinstance(value, list):
                return [replace_path(item) for item in value]
            return value

        record = {key: replace_path(value) for key, value in source.items()}
        actual = int(record.get("collected", record.get("target_collected", 0)))
        record.update({
            "episode_id": episode_id,
            "success": False,
            "preview": True,
            "generation_outcome": "failed",
            "failure_mode": failure_mode,
            "failure_family": "wrong_count",
            "failure_actual_count": actual,
            "deliberate_failure": True,
            "target_count": target_count,
            "requested_seed": int(seed),
            "seed": int(record.get("seed", seed)),
            "derived_from": source_id,
            "failure_reason": (
                f"wrong count: collected {actual} instead of target "
                f"{target_count} ({'over' if failure_mode.endswith('over') else 'under'})"),
        })
        with (self.preview_root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.episode_metadata(episode_id)

    @staticmethod
    def _failure_quality_ok(result, target_count: int, failure_mode: str,
                            cfg=None) -> bool:
        """Gate a deliberate failure before it is written to the manifest."""
        # A failure example is still a demonstration of an attempted sweep.
        # Reject planner probes and touchdown-only traces before they can enter
        # the persistent preview set.  This is deliberately shared by every
        # failure family, including the target-1 zero-collected boundary.
        if str(getattr(result, "planner_status", "")) != "feasible":
            return False
        if len(getattr(result, "observations", ())) < 100:
            return False
        actual = int(getattr(result, "collected", 0))
        target_count = int(target_count)
        if failure_mode == "wrong_count_over":
            return actual == target_count + 1
        if failure_mode == "wrong_count_under":
            return actual == max(0, target_count - 1)
        if failure_mode == "stall_outside_tray":
            if target_count == 1:
                return actual == 0
            if not (0 < actual < target_count):
                return False
            if cfg is None:
                return True
            return stall_failure_sidewall_ok(result, cfg)
        if failure_mode == "misroute":
            return len(getattr(result, "observations", ())) >= 100
        return False

    def build_preview(self, seed: int, target_count: int | None = None,
                      max_attempts: int = 6, outcome: str = "success",
                      failure_mode: str = "", persist_failed: bool = True,
                      source_slot: int | None = None,
                      layout_id: str | None = None,
                      layout_kind: str | None = None) -> dict:
        local_cfg = self.cfg.copy()
        local_cfg.set_path("components.count", 6)
        local_cfg.set_path("task.total_count", 6)
        if target_count is not None:
            local_cfg.set_path("task.target_count", int(target_count))
            # Keep the six physical parts in the same concentrated scene for
            # every target count.  ``exact_partial_spawn_mode`` remains an
            # explicit escape hatch for experiments that need a wider layout.
            if int(target_count) < 6 and str(local_cfg.components.get("spawn_mode", "cluster")) == "cluster":
                local_cfg.set_path(
                    "components.spawn_mode",
                    str(local_cfg.components.get("exact_partial_spawn_mode", "uniform")),
                )
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be positive")
        outcome = str(outcome or "success").lower()
        if outcome not in {"success", "failed"}:
            raise ValueError("outcome must be success or failed")
        if outcome == "failed" and not failure_mode:
            failure_mode = "stall_outside_tray"
        failure_mode = _resolve_failure_mode(
            failure_mode, int(local_cfg.task.target_count), seed)
        if outcome == "failed" and failure_mode.startswith("wrong_count_"):
            if int(local_cfg.task.target_count) > 1:
                return self._clone_success_as_wrong_count(
                    seed, int(local_cfg.task.target_count), failure_mode,
                    source_slot=source_slot)
        requested_seed = int(seed)
        generation_attempt = 0
        while True:
            accepted_seed = requested_seed + generation_attempt * 1000003
            if outcome == "failed":
                result = run_expert_episode(
                    local_cfg, seed=accepted_seed, collect_observations=True,
                    failure_mode=failure_mode)
            else:
                result = run_expert_episode(
                    local_cfg, seed=accepted_seed, collect_observations=True)
            generation_attempt += 1
            if outcome == "failed" or result.success or generation_attempt >= int(max_attempts):
                break
        self.preview_root.mkdir(parents=True, exist_ok=True)
        episode_id = next_demo_name(
            [self.dataset_root, self.preview_root], int(local_cfg.task.target_count))
        planner_score = float(getattr(result, "planner_score", float("inf")))
        metadata = {
            "schema_version": 4,
            "episode_kind": "expert_preview",
            "split": "pilot",
            "layout_id": str(layout_id or ""),
            "layout_kind": str(layout_kind or "independent"),
            "preview": True,
            "seed": int(accepted_seed),
            "requested_seed": requested_seed,
            "generation_attempt": generation_attempt,
            "generation_outcome": outcome,
            "failure_mode": str(failure_mode),
            "failure_family": (
                "wrong_count" if str(failure_mode).startswith("wrong_count_")
                else str(failure_mode)),
            "failure_actual_count": int(getattr(result, "collected", 0)),
            "deliberate_failure": bool(outcome == "failed"),
            "count": 6,
            "total_count": 6,
            "target_count": int(local_cfg.task.target_count),
            "collected": int(getattr(result, "collected", 0)),
            "fps": float(local_cfg.act.action_hz),
            "planner": "astar_one_pass",
            "planner_status": str(getattr(result, "planner_status", "unknown")),
            "planner_failure_reason": str(
                getattr(result, "planner_failure_reason", "")),
            "planner_strategy": str(getattr(result, "planner_strategy", "unknown")),
            "planner_turn_count": int(getattr(result, "planner_turn_count", 0)),
            "planner_attempts": int(getattr(result, "planner_attempts", 0)),
            "planner_score": (planner_score if np.isfinite(planner_score) else None),
            "target_mode": str(local_cfg.task.target_mode),
            "spawn_mode": str(local_cfg.components.get("spawn_mode", "cluster")),
            "target_indices": list(getattr(result, "target_indices", [])),
            "recovery_used": any(row.get("phase") == "recovery" for row in result.trace),
            "failure_reason": result.failure_reason,
            "peak_force": float(getattr(result, "peak_force", 0.0)),
            "first_contact_position": (
                np.asarray(result.first_contact_position, dtype=float).tolist()
                if getattr(result, "first_contact_position", None) is not None else None
            ),
        }
        if (outcome == "failed"
                and not self._failure_quality_ok(
                result, int(local_cfg.task.target_count), failure_mode,
                cfg=local_cfg)):
            # Failed attempts are probes until their physical outcome matches
            # the requested family.  Never leave a rejected attempt in the
            # manifest for a later cleanup pass.
            return {
                **metadata,
                "episode_id": "",
                "preview": True,
                "persisted": False,
                "length": len(result.observations),
                "duration": len(result.observations) / max(
                    float(local_cfg.act.action_hz), 1.0),
            }
        if not result.success and not persist_failed:
            # A success batch must not consume a slot with an exhausted
            # quality-gate failure.  The caller can advance the seed and try
            # another layout without leaving an unlabeled failure record.
            return {
                **metadata,
                "episode_id": "",
                "preview": True,
                "persisted": False,
                "length": len(result.observations),
                "duration": len(result.observations) / max(
                    float(local_cfg.act.action_hz), 1.0),
            }
        ActDatasetWriter(str(self.preview_root)).add_episode(
            episode_id, result.observations, result.actions, result.success, metadata)
        signals = {
            "t": [float(row["t"]) for row in result.trace],
            "fz": [float(row["normal_force"]) for row in result.trace],
            "contact": [bool(row["contact"]) for row in result.trace],
        }
        (self.preview_root / episode_id / "signals.json").write_text(
            json.dumps(signals, ensure_ascii=False), encoding="utf-8")
        return self.episode_metadata(episode_id)

    def build_previews(self, seed: int, target_count: int,
                       outcome: str = "success", count: int = 1,
                       failure_mode: str = "", progress_callback=None,
                       cancel_event=None) -> list[dict]:
        """Generate exactly ``count`` records of the requested outcome.

        Success batches count only physically accepted exact-target rollouts.
        A rejected layout or planner exception consumes a replacement seed,
        never a user-visible success slot.  Failed batches use the same retry
        discipline but persist only deliberate failure rollouts.
        """
        count = int(count)
        if count < 1 or count > 100:
            raise ValueError("count must be between 1 and 100")
        outcome = str(outcome or "success").lower()
        if outcome not in {"success", "failed"}:
            raise ValueError("outcome must be success or failed")
        records = []
        failure_modes = ["stall_outside_tray", "wrong_count", "misroute"]
        if progress_callback is not None:
            progress_callback({"completed": 0, "total": count,
                               "attempted": 0, "retry": 0,
                               "last_error": "", "latest_episode_id": ""})
        # Each build_preview may consume up to six replacement seeds. Leave a
        # larger window between logical records so retries cannot collide with
        # a later record's initial seed.
        seed_stride = 1000003 * 64
        for index in range(count):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("preview generation stopped")
            selected_failure_mode = (failure_mode or failure_modes[index % len(failure_modes)])
            selected_failure_mode = _resolve_failure_mode(
                selected_failure_mode, int(target_count), index)
            accepted = None
            last_error = ""
            for retry in range(32):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("preview generation stopped")
                requested_seed = (int(seed) + index * seed_stride
                                  + retry * 1000003)
                if progress_callback is not None:
                    progress_callback({
                        "completed": len(records),
                        "total": count,
                        "attempted": index + 1,
                        "retry": retry + 1,
                        "last_error": last_error,
                        "latest_episode_id": str(
                            accepted.get("episode_id", "") if accepted else ""),
                    })
                try:
                    candidate = self.build_preview(
                        seed=requested_seed,
                        target_count=int(target_count),
                        outcome=outcome,
                        failure_mode=(selected_failure_mode
                                      if outcome == "failed" else ""),
                        max_attempts=(6 if outcome == "success" else 1),
                        persist_failed=(outcome == "failed"),
                        source_slot=index,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"preview persistence failed on target {target_count}, "
                        f"record {index + 1}/{count}: {exc}") from exc
                except Exception as exc:  # keep one bad layout from killing a batch
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue
                if outcome == "success" and not candidate.get("success", False):
                    last_error = str(candidate.get("failure_reason", "quality gate rejected"))
                    continue
                if outcome == "failed":
                    if candidate.get("persisted", True) is False:
                        last_error = str(candidate.get("failure_reason", "failure quality gate rejected"))
                        continue
                    actual = int(candidate.get("failure_actual_count",
                                             candidate.get("collected", 0)))
                    target = int(target_count)
                    if selected_failure_mode == "wrong_count_over" and actual != target + 1:
                        last_error = f"wrong-count over quality gate: actual={actual}, target={target}"
                        continue
                    if selected_failure_mode == "wrong_count_under" and actual != max(0, target - 1):
                        last_error = f"wrong-count under quality gate: actual={actual}, target={target}"
                        continue
                    if (selected_failure_mode == "stall_outside_tray"
                            and not ((target == 1 and actual == 0)
                                     or (0 < actual < target))):
                        last_error = f"partial-tray quality gate: actual={actual}, target={target}"
                        continue
                accepted = candidate
                break
            if accepted is None:
                raise RuntimeError(
                    f"could not generate {outcome} record {index + 1}/{count} "
                    f"for target {target_count} after 32 replacement seeds; {last_error}")
            records.append(accepted)
            if progress_callback is not None:
                progress_callback({
                    "completed": len(records),
                    "total": count,
                    "attempted": index + 1,
                    "retry": retry + 1,
                    "last_error": "",
                    "latest_episode_id": str(accepted.get("episode_id", "")),
                })
        return records

    def _target_outcome_counts(self, target_count: int) -> tuple[int, int]:
        success = failed = 0
        for _root, record, _preview in self._records():
            if int(record.get("target_count", -1)) != int(target_count):
                continue
            if bool(record.get("success", False)):
                success += 1
            else:
                failed += 1
        return success, failed

    def fill_target_counts(self, seed: int, target_count: int,
                           success_count: int = 8,
                           failure_count: int = 2) -> dict:
        """Fill one target bucket to exact success and failure counts."""
        target_count = int(target_count)
        if not 1 <= target_count <= 6:
            raise ValueError("target_count must be between 1 and 6")
        success_count, failure_count = int(success_count), int(failure_count)
        if success_count < 0 or failure_count < 0:
            raise ValueError("requested outcome counts must be non-negative")
        current_success, current_failed = self._target_outcome_counts(target_count)
        need_success = max(0, success_count - current_success)
        need_failed = max(0, failure_count - current_failed)
        generated = []
        if need_success:
            generated.extend(self.build_previews(
                seed=int(seed), target_count=target_count,
                outcome="success", count=need_success))
        if need_failed:
            generated.extend(self.build_previews(
                seed=int(seed) + 1000003 * 2048,
                target_count=target_count,
                outcome="failed", count=need_failed))
        final_success, final_failed = self._target_outcome_counts(target_count)
        if final_success != success_count or final_failed != failure_count:
            raise RuntimeError(
                f"target {target_count} ended at success={final_success}, "
                f"failed={final_failed}; expected {success_count}/{failure_count}")
        return {
            "target_count": target_count,
            "success": final_success,
            "failed": final_failed,
            "generated": generated,
        }

    def fill_failure_mix(self, seed: int, target_count: int,
                         stall_count: int = 2,
                         wrong_over_count: int = 2,
                         wrong_under_count: int = 2,
                         misroute_count: int = 2) -> dict:
        """Fill the persistent three-family failure mix for one target.

        ``wrong_count_over`` is rejected for target 6 because the scene has
        exactly six physical components; silently fabricating a seventh would
        invalidate the task contract.
        """
        target_count = int(target_count)
        if not 1 <= target_count <= 6:
            raise ValueError("target_count must be between 1 and 6")
        counts = {
            "stall_outside_tray": int(stall_count),
            "wrong_count_over": int(wrong_over_count),
            "wrong_count_under": int(wrong_under_count),
            "misroute": int(misroute_count),
        }
        if target_count == 6 and counts["wrong_count_over"]:
            raise ValueError(
                "target 6 cannot have wrong_count_over with six physical components")
        current = {mode: 0 for mode in counts}
        for _root, record, _preview in self._records():
            if int(record.get("target_count", -1)) != target_count:
                continue
            mode = str(record.get("failure_mode", ""))
            if not bool(record.get("success", False)) and mode in current:
                current[mode] += 1
        generated = []
        offset = 0
        for mode, desired in counts.items():
            missing = max(0, desired - current[mode])
            if missing:
                generated.extend(self.build_previews(
                    seed=int(seed) + offset * 1000003,
                    target_count=target_count, outcome="failed",
                    count=missing, failure_mode=mode))
            offset += max(1, desired)
        final = {mode: 0 for mode in counts}
        for _root, record, _preview in self._records():
            if int(record.get("target_count", -1)) != target_count:
                continue
            mode = str(record.get("failure_mode", ""))
            if not bool(record.get("success", False)) and mode in final:
                final[mode] += 1
        if final != counts:
            raise RuntimeError(
                f"target {target_count} failure mix ended at {final}; expected {counts}")
        return {"target_count": target_count, "counts": final, "generated": generated}
