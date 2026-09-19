"""Dataset reader for ObjectACT schema-v5 episodes.

The legacy :class:`sim.act.dataset.ActDataset` remains schema-v4-only.  This
reader intentionally uses a separate manifest so old ACT runs are still
reproducible while v5 perception sidecars are added incrementally.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from .v5_bev import build_bev_from_sidecar
from .v5 import (
    read_v5_sidecar,
    write_v5_sidecar,
)


class ObjectActDatasetWriter:
    """Atomic writer for RGB-derived schema-v5 expert episodes."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest_v5.jsonl"

    def add_episode(
        self,
        episode_id: str,
        observations: list[dict],
        selection_target: np.ndarray,
        success: bool,
        metadata: dict,
    ) -> Path:
        from PIL import Image

        if not success:
            raise ValueError("v5 training writer accepts successful expert episodes only")
        if not observations:
            raise ValueError("v5 episode must contain observations")
        labels = np.asarray(selection_target, dtype=bool)
        if labels.shape != (len(observations), 6):
            raise ValueError("selection_target must have shape (frames, 6)")
        actions = np.stack([
            np.asarray(item["action_delta"], dtype=np.float32).reshape(4)
            for item in observations
        ])
        robot_state = np.stack([
            np.asarray(item["robot_state"], dtype=np.float32).reshape(36)
            for item in observations
        ])
        task_state = np.stack([
            np.asarray(item["task_state"], dtype=np.float32).reshape(6)
            for item in observations
        ])
        object_tokens = np.stack([
            np.asarray(item["object_tokens"], dtype=np.float32).reshape(6, 29)
            for item in observations
        ])
        object_valid = np.stack([
            np.asarray(item["object_valid"], dtype=bool).reshape(6)
            for item in observations
        ])
        instance_bev = np.stack([
            np.asarray(item["instance_bev"], dtype=bool).reshape(6, 128, 160)
            for item in observations
        ])
        action_valid = np.asarray([
            bool(item.get("policy_mask", True)) for item in observations
        ], dtype=bool)
        episode_dir = self.root / str(episode_id)
        if episode_dir.exists():
            raise FileExistsError(episode_dir)
        temporary = self.root / f".{episode_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        created = False
        committed = False
        try:
            overhead, wrist, inspection = [], [], []
            for index, item in enumerate(observations):
                for key, paths in (("overhead", overhead), ("wrist", wrist)):
                    path = temporary / f"{key}_{index:05d}.png"
                    Image.fromarray(np.asarray(item[key], dtype=np.uint8)).save(path)
                    paths.append(str(Path(str(episode_id)) / path.name))
                image = item.get("inspection")
                if image is None:
                    inspection.append("")
                else:
                    path = temporary / f"inspection_{index:05d}.png"
                    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
                    inspection.append(str(Path(str(episode_id)) / path.name))

            def matrix(name: str, width: int) -> np.ndarray:
                return np.stack([
                    np.asarray(item.get(name, np.zeros(width)), dtype=np.float32).reshape(width)
                    for item in observations
                ])

            np.savez_compressed(
                temporary / "arrays.npz",
                robot_state=robot_state,
                action=actions,
                action_valid=action_valid,
                t=np.asarray([float(item.get("t", index / 25.0)) for index, item in enumerate(observations)], dtype=np.float64),
                phase=np.asarray([str(item.get("phase", "unknown")) for item in observations], dtype="U16"),
                joint_position=matrix("joint_position", 6),
                tcp_pose=matrix("tcp_pose", 4),
                wrench=matrix("wrench", 6),
                normal_force=np.asarray([float(item.get("normal_force", 0.0)) for item in observations], dtype=np.float32),
                contact=np.asarray([bool(item.get("contact", False)) for item in observations], dtype=bool),
                contact_latched=np.asarray([bool(item.get("contact_latched", item.get("contact", False))) for item in observations], dtype=bool),
                fully_collected=np.asarray([int(item.get("fully_collected", 0)) for item in observations], dtype=np.int16),
                reference=np.stack([np.asarray(item.get("reference", np.zeros(4)), dtype=np.float32).reshape(4) for item in observations]),
                policy_reference=np.stack([np.asarray(item.get("policy_reference", np.zeros(4)), dtype=np.float32).reshape(4) for item in observations]),
                policy_z=np.asarray([float(item.get("policy_z", 0.0)) for item in observations], dtype=np.float32),
                applied_z=np.asarray([float(item.get("applied_z", 0.0)) for item in observations], dtype=np.float32),
                z_owner=np.asarray([str(item.get("z_owner", "unknown")) for item in observations], dtype="U24"),
            )
            sidecar = temporary / "perception_v5.npz"
            write_v5_sidecar(
                sidecar,
                object_tokens,
                object_valid,
                instance_bev,
                task_state,
                selection_target=labels,
                visual_count=np.rint(task_state[:, 0] * 6.0).astype(np.int16),
            )
            record = {
                **metadata,
                "perception_source": str(metadata.get("perception_source", "rgb_detector")),
                "episode_id": str(episode_id),
                "schema_version": 5,
                "episode_kind": "expert",
                "split": str(metadata.get("split", "train")),
                "success": True,
                "frame_count": len(observations),
                "action_dim": 4,
                "robot_state_dim": 36,
                "task_state_dim": 6,
                "object_slots": 6,
                "object_token_dim": 29,
                "bev_shape": [6, 128, 160],
                "arrays": str(Path(str(episode_id)) / "arrays.npz"),
                "perception_sidecar": str(Path(str(episode_id)) / "perception_v5.npz"),
                "overhead": overhead,
                "wrist": wrist,
                "inspection": inspection,
            }
            temporary.rename(episode_dir)
            created = True
            old_manifest = self.manifest.read_text(encoding="utf-8") if self.manifest.exists() else ""
            manifest_tmp = self.root / f".{self.manifest.name}.tmp-{uuid.uuid4().hex}"
            try:
                manifest_tmp.write_text(
                    old_manifest + json.dumps(record, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                manifest_tmp.replace(self.manifest)
            finally:
                manifest_tmp.unlink(missing_ok=True)
            committed = True
            return episode_dir
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            if created and not committed and episode_dir.exists():
                shutil.rmtree(episode_dir, ignore_errors=True)


def validate_v5_manifest(
    records: list[dict],
    *,
    expected_per_target: int = 20,
    target_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> None:
    """Validate the exact successful-expert composition required by v5."""

    required = set(int(value) for value in target_counts)
    for record in records:
        if int(record.get("schema_version", 0)) != 5:
            raise ValueError("v5 manifest contains a non-v5 record")
        if str(record.get("episode_kind", "")) != "expert" or not bool(record.get("success")):
            raise ValueError("v5 training manifest accepts only successful expert records")
        if int(record.get("action_dim", 0)) != 4 or int(record.get("robot_state_dim", 0)) != 36:
            raise ValueError("v5 manifest record has an incompatible action/state shape")
    counts = Counter(int(record.get("target_count", -1)) for record in records)
    unexpected = sorted(set(counts) - required)
    if unexpected:
        raise ValueError(f"v5 manifest has unexpected target counts {unexpected}")
    for target in sorted(required):
        if counts[target] != int(expected_per_target):
            raise ValueError(
                f"v5 manifest target {target} has {counts[target]} records; "
                f"expected {expected_per_target}"
            )


class ObjectActDataset:
    """Return one v5 observation and a padded 25-step action chunk."""

    def __init__(
        self,
        root: str,
        *,
        manifest_name: str = "manifest_v5.jsonl",
        chunk_size: int = 25,
        split: str = "train",
        tray_bounds: tuple[float, float, float, float] = (-0.45, -0.31, -0.18, 0.18),
        brush_size: tuple[float, float] = (0.12, 0.01),
    ):
        self.root = Path(root)
        self.chunk_size = int(chunk_size)
        self.tray_bounds = tuple(float(value) for value in tray_bounds)
        self.brush_size = tuple(float(value) for value in brush_size)
        manifest_path = self.root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.records = [
            record for record in records
            if int(record.get("schema_version", 0)) == 5
            and int(record.get("action_dim", 0)) == 4
            and int(record.get("robot_state_dim", 0)) == 36
            and str(record.get("episode_kind", "")) == "expert"
            and str(record.get("split", "")) == str(split)
            and bool(record.get("success", False))
        ]
        self.index: list[tuple[dict, int]] = []
        self._sidecars: dict[str, dict[str, np.ndarray]] = {}
        for record in self.records:
            with np.load(self.root / record["arrays"], mmap_mode="r") as arrays:
                actions = np.asarray(arrays["action"])
                states = np.asarray(arrays["robot_state"])
                valid = np.asarray(arrays["action_valid"], dtype=bool)
                if actions.ndim != 2 or actions.shape[1] != 4:
                    raise ValueError(f"{record['episode_id']} action array is not (T, 4)")
                if states.ndim != 2 or states.shape[1] != 36:
                    raise ValueError(f"{record['episode_id']} robot_state array is not (T, 36)")
                if valid.shape != (len(actions),):
                    raise ValueError(f"{record['episode_id']} action_valid length mismatch")
                self.index.extend((record, int(frame)) for frame in np.flatnonzero(valid))

    def _sidecar(self, record: dict) -> dict[str, np.ndarray]:
        episode_id = str(record["episode_id"])
        if episode_id not in self._sidecars:
            self._sidecars[episode_id] = read_v5_sidecar(
                self.root / record["perception_sidecar"]
            )
        return self._sidecars[episode_id]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        from PIL import Image

        record, frame = self.index[index]
        sidecar = self._sidecar(record)
        with Image.open(self.root / record["overhead"][frame]) as image:
            overhead = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(self.root / record["wrist"][frame]) as image:
            wrist = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with np.load(self.root / record["arrays"], mmap_mode="r") as arrays:
            robot_state = np.asarray(arrays["robot_state"][frame], dtype=np.float32)
            actions = np.asarray(
                arrays["action"][frame:frame + self.chunk_size], dtype=np.float32
            )
            valid = np.asarray(
                arrays["action_valid"][frame:frame + self.chunk_size], dtype=bool
            )
        if actions.shape[0] == 0:
            raise ValueError("v5 sample cannot start without an action")
        pad = self.chunk_size - actions.shape[0]
        if pad > 0:
            actions = np.concatenate((actions, np.repeat(actions[-1:], pad, axis=0)))
            valid = np.concatenate((valid, np.zeros(pad, dtype=bool)))
        sample = {
            "observation.images.overhead": np.transpose(overhead, (2, 0, 1)),
            "observation.images.wrist": np.transpose(wrist, (2, 0, 1)),
            "observation.robot_state": robot_state,
            "observation.task_state": np.asarray(sidecar["task_state"][frame], dtype=np.float32),
            "observation.object_tokens": np.asarray(
                sidecar["object_tokens"][frame], dtype=np.float32
            ),
            "observation.object_valid": np.asarray(
                sidecar["object_valid"][frame], dtype=bool
            ),
            "observation.instance_bev": np.asarray(
                sidecar["instance_bev"][frame], dtype=bool
            ),
            "observation.bev": build_bev_from_sidecar(
                object_tokens=np.asarray(sidecar["object_tokens"][frame], dtype=np.float32),
                object_valid=np.asarray(sidecar["object_valid"][frame], dtype=bool),
                instance_bev=np.asarray(sidecar["instance_bev"][frame], dtype=bool),
                robot_state=robot_state,
                selection_target=(
                    np.asarray(sidecar["selection_target"][frame], dtype=bool)
                    if "selection_target" in sidecar else None
                ),
                tray_bounds=self.tray_bounds,
                brush_size=self.brush_size,
            ),
            "action": actions,
            "action_is_pad": ~valid,
        }
        if "selection_target" in sidecar:
            sample["selection_target"] = np.asarray(
                sidecar["selection_target"][frame], dtype=bool
            )
        return sample

    def phase_sampling_weights(self, approach_fraction: float = 0.35) -> np.ndarray:
        """Return the configured 35/65 approach versus contact weights."""

        fraction = float(approach_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("approach_fraction must be strictly between zero and one")
        approach, contact = [], []
        for item, (record, frame) in enumerate(self.index):
            with np.load(self.root / record["arrays"], mmap_mode="r") as arrays:
                phase = str(arrays["phase"][frame])
            if phase in {"approach", "descent"}:
                approach.append(item)
            elif phase in {"contact_build", "sweep"}:
                contact.append(item)
            else:
                raise ValueError(f"unsupported v5 training phase {phase!r}")
        if not approach or not contact:
            raise ValueError("v5 phase sampling needs approach and contact frames")
        weights = np.zeros(len(self.index), dtype=np.float64)
        weights[approach] = fraction / len(approach)
        weights[contact] = (1.0 - fraction) / len(contact)
        return weights
