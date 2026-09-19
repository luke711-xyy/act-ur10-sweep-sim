"""Small local dataset format for MuJoCo ACT demonstrations."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np


class ActDatasetWriter:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"

    def add_episode(self, episode_id: str, observations: list[dict], actions: np.ndarray,
                    success: bool, metadata: dict) -> Path:
        from PIL import Image

        if len(observations) != len(actions):
            raise ValueError("one ACT action target is required per observation")
        action_values = np.asarray(actions, dtype=np.float32)
        if action_values.ndim != 2 or action_values.shape[1] != 4:
            raise ValueError(
                f"schema v4 ACT actions must have shape (frames, 4), got {action_values.shape}"
            )
        if not observations:
            raise ValueError("an ACT episode must contain at least one observation")
        state_values = np.stack([o["state"] for o in observations]).astype(np.float32)
        if state_values.ndim != 2 or state_values.shape[1] != 42:
            raise ValueError(
                f"schema v4 ACT state must have shape (frames, 42), got {state_values.shape}"
            )
        episode_dir = self.root / episode_id
        if episode_dir.exists():
            raise FileExistsError(f"episode directory already exists: {episode_dir}")
        temporary_dir = self.root / f".{episode_id}.tmp-{uuid.uuid4().hex}"
        temporary_dir.mkdir(parents=True, exist_ok=False)
        committed = False
        final_created = False
        try:
            overhead, wrist, inspection = [], [], []
            for i, obs in enumerate(observations):
                overhead_path = temporary_dir / f"overhead_{i:05d}.png"
                wrist_path = temporary_dir / f"wrist_{i:05d}.png"
                Image.fromarray(np.asarray(obs["overhead"], dtype=np.uint8)).save(overhead_path)
                Image.fromarray(np.asarray(obs["wrist"], dtype=np.uint8)).save(wrist_path)
                overhead.append(str((episode_dir / overhead_path.name).relative_to(self.root)))
                wrist.append(str((episode_dir / wrist_path.name).relative_to(self.root)))
                inspection_image = obs.get("inspection")
                if inspection_image is None:
                    inspection.append("")
                else:
                    inspection_path = temporary_dir / f"inspection_{i:05d}.png"
                    Image.fromarray(np.asarray(inspection_image, dtype=np.uint8)).save(inspection_path)
                    inspection.append(str((episode_dir / inspection_path.name).relative_to(self.root)))
            states = state_values
            env_states = np.stack([o["environment_state"] for o in observations]).astype(np.float32)
            n_frames = len(observations)

            def vectors(key, width, dtype=np.float32):
                return np.stack([
                    np.asarray(o.get(key, np.zeros(width)), dtype=dtype).reshape(width)
                    for o in observations
                ])

            times = np.asarray([float(o.get("t", i)) for i, o in enumerate(observations)],
                               dtype=np.float64)
            phases = np.asarray([str(o.get("phase", "unknown")) for o in observations],
                                dtype="U16")
            action_valid = np.asarray([bool(o.get("policy_mask", True)) for o in observations],
                                      dtype=bool)
            contact_latched = np.asarray([
                bool(o.get("contact_latched", o.get("contact", False)))
                for o in observations
            ], dtype=bool)
            object_pose = np.stack([
                np.asarray(o.get("object_pose", np.zeros((0, 7))), dtype=np.float32)
                for o in observations
            ])
            object_collected = np.stack([
                np.asarray(o.get("object_collected", np.zeros(object_pose.shape[1])), dtype=bool)
                for o in observations
            ])
            z_owner = np.asarray([
                str(o.get("z_owner", "unknown")) for o in observations
            ], dtype="U24")
            policy_reference = np.stack([
                np.asarray(o.get("policy_reference", o.get("reference", np.zeros(4))),
                           dtype=np.float32).reshape(4)
                for o in observations
            ])
            applied_reference = np.stack([
                np.asarray(o.get("applied_reference", o.get("reference", np.zeros(4))),
                           dtype=np.float32).reshape(4)
                for o in observations
            ])
            np.savez_compressed(temporary_dir / "arrays.npz", state=states,
                                environment_state=env_states,
                                action=action_values,
                                action_valid=action_valid,
                                t=times,
                                phase=phases,
                                joint_position=vectors("joint_position", 6),
                                tcp_pose=vectors("tcp_pose", 4),
                                wrench=vectors("wrench", 6),
                                normal_force=np.asarray([
                                    float(o.get("normal_force", 0.0)) for o in observations
                                ], dtype=np.float32),
                                contact=np.asarray([
                                    bool(o.get("contact", False)) for o in observations
                                ], dtype=bool),
                                contact_latched=contact_latched,
                                fully_collected=np.asarray([
                                    int(o.get("fully_collected", 0)) for o in observations
                                ], dtype=np.int16),
                                object_pose=object_pose,
                                object_collected=object_collected,
                                reference=applied_reference,
                                policy_reference=policy_reference,
                                policy_z=np.asarray([
                                    float(o.get("policy_z", policy_reference[i, 2]))
                                    for i, o in enumerate(observations)
                                ], dtype=np.float32),
                                applied_z=np.asarray([
                                    float(o.get("applied_z", applied_reference[i, 2]))
                                    for i, o in enumerate(observations)
                                ], dtype=np.float32),
                                z_owner=z_owner)
            metadata = {
                **metadata,
                "schema_version": 4,
                "frame_count": n_frames,
                "action_dim": 4,
                "state_dim": 42,
                "episode_kind": str(metadata.get("episode_kind", "unknown")),
            }
            record = {"episode_id": episode_id, "success": bool(success),
                      "overhead": overhead, "wrist": wrist, "inspection": inspection,
                      "arrays": str((episode_dir / "arrays.npz").relative_to(self.root)),
                      **metadata}

            # Publish the completed episode directory first, then atomically
            # replace the manifest.  Readers see either the old manifest or a
            # complete new record, never a half-written episode.
            temporary_dir.rename(episode_dir)
            final_created = True
            manifest_contents = self.manifest.read_text(encoding="utf-8") \
                if self.manifest.exists() else ""
            manifest_temp = self.root / f".{self.manifest.name}.tmp-{uuid.uuid4().hex}"
            try:
                manifest_temp.write_text(
                    manifest_contents + json.dumps(record, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                manifest_temp.replace(self.manifest)
            finally:
                manifest_temp.unlink(missing_ok=True)
            committed = True
            return episode_dir
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
            if final_created and not committed and episode_dir.exists():
                shutil.rmtree(episode_dir, ignore_errors=True)


class ActDataset:
    """Returns one observation and a padded future action chunk."""

    def __init__(self, root: str, chunk_size: int = 25,
                 include_failures: bool = False,
                 image_stat_samples: int = 1000,
                 split: str = "train"):
        self.root = Path(root)
        self.chunk_size = int(chunk_size)
        self.image_stat_samples = max(1, int(image_stat_samples))
        self._stats = None
        manifest_records = [
            json.loads(line)
            for line in (self.root / "manifest.jsonl").read_text(
                encoding="utf-8").splitlines()
            if line.strip()
        ]
        # The training reader is deliberately strict.  Workbench previews and
        # inference replays may share the same on-disk format, but only
        # successful schema-v4 expert records from the requested split are
        # behavior-cloning inputs.
        self.records = [
            record for record in manifest_records
            if int(record.get("schema_version", 0)) == 4
            and int(record.get("action_dim", 0)) == 4
            and int(record.get("state_dim", 0)) == 42
            and str(record.get("episode_kind", "")) == "expert"
            and str(record.get("split", "")) == str(split)
            and (include_failures or bool(record.get("success", False)))
        ]
        self.index = []
        for rec in self.records:
            with np.load(self.root / rec["arrays"], mmap_mode="r") as arrays:
                if arrays["action"].ndim != 2 or arrays["action"].shape[1] != 4:
                    raise ValueError(f"{rec['episode_id']} does not contain 4-D ACT actions")
                if arrays["state"].ndim != 2 or arrays["state"].shape[1] != 42:
                    raise ValueError(f"{rec['episode_id']} does not contain 42-D ACT states")
                valid = (np.asarray(arrays["action_valid"], dtype=bool)
                         if "action_valid" in arrays.files
                         else np.ones(len(arrays["action"]), dtype=bool))
                self.index.extend((len(self.index), rec, i)
                                  for i in np.flatnonzero(valid))

    def phase_sampling_weights(self, approach_fraction: float = 0.35) -> np.ndarray:
        """Return normalized weights for 35/65 phase-stratified sampling.

        ``approach`` and ``descent`` share the first mass; ``contact_build``
        and ``sweep`` share the remainder.  Weight is divided by the number of
        available frames in each group so long sweeps do not erase the
        approach/descent supervision.  Masked stability/unload frames are not
        present in ``self.index`` and therefore cannot receive weight.
        """
        fraction = float(approach_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("approach_fraction must be strictly between zero and one")
        approach_ids, contact_ids = [], []
        for item_index, (_, record, frame_id) in enumerate(self.index):
            with np.load(self.root / record["arrays"], mmap_mode="r") as arrays:
                phase = str(arrays["phase"][frame_id])
            if phase in {"approach", "descent"}:
                approach_ids.append(item_index)
            elif phase in {"contact_build", "sweep"}:
                contact_ids.append(item_index)
            else:
                raise ValueError(
                    f"valid ACT frame has unsupported training phase {phase!r}"
                )
        if not approach_ids or not contact_ids:
            raise ValueError("phase-balanced sampling needs both approach and contact frames")
        weights = np.zeros(len(self.index), dtype=np.float64)
        weights[approach_ids] = fraction / len(approach_ids)
        weights[contact_ids] = (1.0 - fraction) / len(contact_ids)
        return weights

    @staticmethod
    def _vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("cannot compute ACT statistics from an empty feature array")
        std = np.std(values, axis=0)
        # LeRobot's MEAN_STD normalizer adds an epsilon to the denominator.  A
        # unit fallback is still useful for constant simulator channels and
        # makes the exported stats explicit and numerically stable.
        std = np.where(std < 1e-8, 1.0, std)
        return {
            "min": values.min(axis=0).astype(np.float32),
            "max": values.max(axis=0).astype(np.float32),
            "mean": values.mean(axis=0).astype(np.float32),
            "std": std.astype(np.float32),
            "count": np.asarray([values.shape[0]], dtype=np.int64),
        }

    def _compute_image_stats(self, records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
        """Compute per-channel image stats without materialising all images.

        This follows the important part of LeRobot's dataset-statistics
        contract: RGB images are converted to float values in [0, 1], and the
        resulting mean/std have shape (C, 1, 1) for broadcasting.
        """
        sums = {
            key: np.zeros(3, dtype=np.float64)
            for key in ("observation.images.overhead", "observation.images.wrist")
        }
        sums_sq = {key: value.copy() for key, value in sums.items()}
        mins = {key: np.full(3, np.inf, dtype=np.float64) for key in sums}
        maxs = {key: np.full(3, -np.inf, dtype=np.float64) for key in sums}
        pixel_count = 0
        selected_count = 0
        for record in records:
            with np.load(self.root / record["arrays"], mmap_mode="r") as arrays:
                valid = (np.asarray(arrays["action_valid"], dtype=bool)
                         if "action_valid" in arrays.files
                         else np.ones(len(arrays["action"]), dtype=bool))
                frame_ids = np.flatnonzero(valid)
            if len(frame_ids) == 0:
                continue
            remaining = max(0, self.image_stat_samples - selected_count)
            if remaining == 0:
                break
            stride = max(1, int(np.ceil(len(frame_ids) / remaining)))
            frame_ids = frame_ids[::stride][:remaining]
            selected_count += len(frame_ids)
            for frame_id in frame_ids:
                for key, manifest_key in (
                    ("observation.images.overhead", "overhead"),
                    ("observation.images.wrist", "wrist"),
                ):
                    from PIL import Image

                    with Image.open(self.root / record[manifest_key][int(frame_id)]) as pil_image:
                        image = np.asarray(
                            pil_image.convert("RGB"), dtype=np.float32
                        ).transpose(2, 0, 1) / 255.0
                    pixels = image.reshape(3, -1).astype(np.float64)
                    sums[key] += pixels.sum(axis=1)
                    sums_sq[key] += np.square(pixels).sum(axis=1)
                    mins[key] = np.minimum(mins[key], pixels.min(axis=1))
                    maxs[key] = np.maximum(maxs[key], pixels.max(axis=1))
                pixel_count += int(image.shape[1] * image.shape[2])

        if pixel_count == 0:
            raise ValueError("cannot compute ACT image statistics without valid frames")
        result = {}
        for key in sums:
            mean = sums[key] / pixel_count
            variance = np.maximum(0.0, sums_sq[key] / pixel_count - np.square(mean))
            std = np.sqrt(variance)
            std = np.where(std < 1e-8, 1.0, std)
            result[key] = {
                "min": mins[key].astype(np.float32).reshape(3, 1, 1),
                "max": maxs[key].astype(np.float32).reshape(3, 1, 1),
                "mean": mean.astype(np.float32).reshape(3, 1, 1),
                "std": std.astype(np.float32).reshape(3, 1, 1),
                "count": np.asarray([pixel_count], dtype=np.int64),
            }
        return result

    @property
    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Return LeRobot-compatible statistics over valid ACT samples.

        Statistics are computed only over the records and frames that the
        behavior-cloning dataset actually indexes.  Failed episodes and
        masked stability/unload frames therefore cannot leak into training
        normalization.
        """
        if self._stats is not None:
            return self._stats
        if not self.index:
            raise ValueError("ACT dataset has no valid behavior-cloning frames")
        vectors = {"observation.state": [], "observation.environment_state": [], "action": []}
        for _, record, frame_id in self.index:
            with np.load(self.root / record["arrays"], mmap_mode="r") as arrays:
                vectors["observation.state"].append(np.asarray(arrays["state"][frame_id], dtype=np.float32))
                vectors["observation.environment_state"].append(
                    np.asarray(arrays["environment_state"][frame_id], dtype=np.float32)
                )
                vectors["action"].append(np.asarray(arrays["action"][frame_id], dtype=np.float32))
        self._stats = {
            key: self._vector_stats(np.stack(values, axis=0))
            for key, values in vectors.items()
        }
        self._stats.update(self._compute_image_stats(self.records))
        return self._stats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        from PIL import Image

        _, rec, i = self.index[index]
        episode_dir = self.root / rec["episode_id"]
        # Keep camera observations as uint8 at the dataset boundary.  The
        # training loop converts them to [0, 1] before LeRobot's official
        # normalizer runs.  Returning float32 here would preserve 0-255
        # values and make MEAN_STD image normalization numerically wrong.
        with Image.open(self.root / rec["overhead"][i]) as overhead_image:
            overhead = np.asarray(overhead_image.convert("RGB"), dtype=np.uint8)
        with Image.open(self.root / rec["wrist"][i]) as wrist_image:
            wrist = np.asarray(wrist_image.convert("RGB"), dtype=np.uint8)
        with np.load(self.root / rec["arrays"]) as arrays:
            state = np.asarray(arrays["state"][i], dtype=np.float32)
            environment_state = np.asarray(
                arrays["environment_state"][i], dtype=np.float32
            )
            actions = np.asarray(arrays["action"][i:i + self.chunk_size], dtype=np.float32)
            valid = (np.asarray(arrays["action_valid"][i:i + self.chunk_size], dtype=bool)
                     if "action_valid" in arrays.files else np.ones(len(actions), dtype=bool))
        pad = self.chunk_size - len(actions)
        if pad > 0:
            actions = np.concatenate((actions, np.repeat(actions[-1:], pad, axis=0)), axis=0)
            valid = np.concatenate((valid, np.zeros(pad, dtype=bool)))
        return {
            # Keep uint8 at the dataset boundary, matching LeRobot's dataset
            # contract.  The official training loop converts camera values to
            # [0, 1] before the policy preprocessor applies dataset stats.
            "observation.images.overhead": np.transpose(overhead, (2, 0, 1)),
            "observation.images.wrist": np.transpose(wrist, (2, 0, 1)),
            "observation.state": state,
            "observation.environment_state": environment_state,
            "action": actions,
            "action_is_pad": ~valid,
        }
