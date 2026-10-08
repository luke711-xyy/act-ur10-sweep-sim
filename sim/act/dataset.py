"""Small local dataset format for MuJoCo ACT demonstrations."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np

from .interface import ACT_SCHEMA_VERSION, ACT_STATE_DIM


def _observation_images(observation: dict) -> dict[str, np.ndarray]:
    """Return a frame's policy images, accepting legacy two-view records."""
    images = observation.get("images")
    if images is not None:
        return {str(name): np.asarray(value, dtype=np.uint8)
                for name, value in images.items()}
    legacy = {name: np.asarray(observation[name], dtype=np.uint8)
              for name in ("overhead", "wrist") if name in observation}
    return legacy


def _record_image_paths(record: dict) -> dict[str, list[str]]:
    """Resolve a manifest's dynamic view map or its legacy aliases."""
    images = record.get("images")
    if isinstance(images, dict) and images:
        return {str(name): list(paths) for name, paths in images.items()}
    return {name: list(record[name]) for name in ("overhead", "wrist")
            if name in record}


def normalize_feature(values: np.ndarray, stats: dict, feature: str) -> np.ndarray:
    """Apply the MEAN_STD transform used by LeRobot to one feature array.

    Training and inference use LeRobot's serialized pre/post-processors; this
    small NumPy helper is for diagnostics and contract tests at the raw-data
    boundary.
    """
    if feature not in stats:
        raise KeyError(f"normalization statistics are missing feature {feature!r}")
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats[feature]["mean"], dtype=np.float32)
    std = np.asarray(stats[feature]["std"], dtype=np.float32)
    return ((values - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def unnormalize_action(values: np.ndarray, stats: dict) -> np.ndarray:
    """Map normalized ACT actions back to their physical data units."""
    if "action" not in stats:
        raise KeyError("normalization statistics are missing feature 'action'")
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats["action"]["mean"], dtype=np.float32)
    std = np.asarray(stats["action"]["std"], dtype=np.float32)
    return (values * std + mean).astype(np.float32)


class ActDatasetWriter:
    def __init__(self, root: str, image_format: str = "png"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"
        image_format = str(image_format).lower().replace("_", "-")
        if image_format not in {"png", "webp-lossless"}:
            raise ValueError("image_format must be 'png' or 'webp-lossless'")
        self.image_format = image_format
        self._pillow_image_format = "PNG" if image_format == "png" else "WEBP"
        self._image_extension = ".png" if image_format == "png" else ".webp"
        self._image_save_options = (
            {} if image_format == "png"
            else {"lossless": True, "quality": 100, "method": 4}
        )

    def add_episode(self, episode_id: str, observations: list[dict], actions: np.ndarray,
                    success: bool, metadata: dict,
                    control_trace: list[dict] | None = None) -> Path:
        from PIL import Image

        if len(observations) != len(actions):
            raise ValueError("one ACT action target is required per observation")
        action_values = np.asarray(actions, dtype=np.float32)
        if action_values.ndim != 2 or action_values.shape[1] != 4:
            raise ValueError(
                f"schema v{ACT_SCHEMA_VERSION} ACT actions must have shape "
                f"(frames, 4), got {action_values.shape}"
            )
        if not observations:
            raise ValueError("an ACT episode must contain at least one observation")
        state_values = np.stack([o["state"] for o in observations]).astype(np.float32)
        if state_values.ndim != 2 or state_values.shape[1] != ACT_STATE_DIM:
            raise ValueError(
                f"schema v{ACT_SCHEMA_VERSION} ACT state must have shape "
                f"(frames, {ACT_STATE_DIM}), got {state_values.shape}"
            )
        episode_dir = self.root / episode_id
        if episode_dir.exists():
            raise FileExistsError(f"episode directory already exists: {episode_dir}")
        temporary_dir = self.root / f".{episode_id}.tmp-{uuid.uuid4().hex}"
        temporary_dir.mkdir(parents=True, exist_ok=False)
        committed = False
        final_created = False
        try:
            image_views = {}
            expected_views = None
            inspection = []
            for i, obs in enumerate(observations):
                frame_images = _observation_images(obs)
                if not frame_images:
                    raise ValueError("each ACT frame must contain at least one policy image")
                views = tuple(frame_images)
                if expected_views is None:
                    expected_views = views
                    image_views = {view: [] for view in views}
                elif set(views) != set(expected_views):
                    raise ValueError(
                        "all frames in an episode must contain the same ACT views"
                    )
                for view in expected_views:
                    safe_view = "".join(
                        character if character.isalnum() or character in "-_" else "_"
                        for character in view
                    )
                    image_name = (
                        f"image_{safe_view}_{i:05d}{self._image_extension}"
                    )
                    image_path = temporary_dir / image_name
                    image = frame_images[view]
                    if image.ndim != 3 or image.shape[-1] != 3:
                        raise ValueError(
                            f"ACT view {view!r} must be HxWx3 RGB, got {image.shape}"
                        )
                    Image.fromarray(image).save(
                        image_path, format=self._pillow_image_format,
                        **self._image_save_options,
                    )
                    image_views[view].append(str(
                        (episode_dir / image_name).relative_to(self.root)
                    ))
                inspection_image = obs.get("inspection")
                if inspection_image is None:
                    inspection.append("")
                else:
                    inspection_path = temporary_dir / (
                        f"inspection_{i:05d}{self._image_extension}"
                    )
                    Image.fromarray(
                        np.asarray(inspection_image, dtype=np.uint8)
                    ).save(
                        inspection_path, format=self._pillow_image_format,
                        **self._image_save_options,
                    )
                    inspection.append(str((episode_dir / inspection_path.name).relative_to(self.root)))
            states = state_values
            env_states = np.stack([o["environment_state"] for o in observations]).astype(np.float32)
            n_frames = len(observations)

            def supervisor_vectors():
                values = []
                for obs in observations:
                    value = obs.get("supervisor_state")
                    if value is None:
                        policy_state = np.asarray(
                            obs["environment_state"], dtype=np.float32
                        ).reshape(-1)
                        if policy_state.size == 3:
                            value = policy_state
                        else:
                            value = [
                                int(metadata.get("total_count", metadata.get("count", 6))),
                                int(metadata.get("target_count", 0)),
                                int(obs.get("fully_collected", 0)),
                            ]
                    values.append(np.asarray(value, dtype=np.float32).reshape(3))
                return np.stack(values)

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
                                supervisor_state=supervisor_vectors(),
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
                                raw_normal_force_peak=np.asarray([
                                    float(o.get("raw_normal_force_peak", 0.0))
                                    for o in observations
                                ], dtype=np.float32),
                                raw_normal_force_peak_t=np.asarray([
                                    float(o.get("raw_normal_force_peak_t", times[i]))
                                    for i, o in enumerate(observations)
                                ], dtype=np.float64),
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
            control_trace_path = ""
            if control_trace:
                trace = list(control_trace)
                trace_values = {
                    "t": np.asarray([float(row.get("t", 0.0)) for row in trace],
                                    dtype=np.float64),
                    "phase": np.asarray([str(row.get("phase", "unknown"))
                                          for row in trace], dtype="U16"),
                    "tcp": np.stack([np.asarray(row.get("tcp", np.zeros(3)),
                                                dtype=np.float32).reshape(3)
                                     for row in trace]),
                    "normal_force": np.asarray([
                        float(row.get("normal_force", 0.0)) for row in trace
                    ], dtype=np.float32),
                    "raw_normal_force_peak": np.asarray([
                        float(row.get("raw_normal_force_peak", 0.0)) for row in trace
                    ], dtype=np.float32),
                    "raw_normal_force_peak_t": np.asarray([
                        float(row.get("raw_normal_force_peak_t", row.get("t", 0.0)))
                        for row in trace
                    ], dtype=np.float64),
                    "policy_z": np.asarray([
                        float(row.get("policy_z", 0.0)) for row in trace
                    ], dtype=np.float32),
                    "applied_z": np.asarray([
                        float(row.get("applied_z", 0.0)) for row in trace
                    ], dtype=np.float32),
                    "z_owner": np.asarray([
                        str(row.get("z_owner", "unknown")) for row in trace
                    ], dtype="U24"),
                    "contact": np.asarray([
                        bool(row.get("contact", False)) for row in trace
                    ], dtype=bool),
                    "table_contact": np.asarray([
                        bool(row.get("table_contact", False)) for row in trace
                    ], dtype=bool),
                }
                np.savez_compressed(
                    temporary_dir / "control_trace_100hz.npz", **trace_values)
                control_trace_path = str(
                    (episode_dir / "control_trace_100hz.npz").relative_to(self.root)
                )
            metadata = {
                **metadata,
                "schema_version": ACT_SCHEMA_VERSION,
                "frame_count": n_frames,
                "action_dim": 4,
                "state_dim": ACT_STATE_DIM,
                "episode_kind": str(metadata.get("episode_kind", "unknown")),
            }
            record = {"episode_id": episode_id, "success": bool(success),
                      "camera_views": list(expected_views or ()),
                      "images": image_views, "inspection": inspection,
                      "image_storage_format": self.image_format,
                      "arrays": str((episode_dir / "arrays.npz").relative_to(self.root)),
                      **({"control_trace": control_trace_path}
                         if control_trace_path else {}),
                      **metadata}
            # Keep aliases for the older workbench and two-camera records.
            for legacy_view in ("overhead", "wrist"):
                if legacy_view in image_views:
                    record[legacy_view] = image_views[legacy_view]

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
                 split: str = "train", camera_views=None):
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
        # successful schema-v5 expert records from the requested split are
        # behavior-cloning inputs.
        self.records = [
            record for record in manifest_records
            if int(record.get("schema_version", 0)) == ACT_SCHEMA_VERSION
            and int(record.get("action_dim", 0)) == 4
            and int(record.get("state_dim", 0)) == ACT_STATE_DIM
            and str(record.get("episode_kind", "")) == "expert"
            and str(record.get("split", "")) == str(split)
            and (include_failures or bool(record.get("success", False)))
        ]
        if self.records:
            inferred = tuple(_record_image_paths(self.records[0]))
            self.camera_views = tuple(camera_views) if camera_views is not None else inferred
            missing = [record.get("episode_id", "<missing>") for record in self.records
                       if not set(self.camera_views).issubset(
                           _record_image_paths(record)
                       )]
            if missing:
                raise ValueError(
                    "ACT dataset records are missing configured image views: "
                    f"{missing[:5]}"
                )
        else:
            self.camera_views = tuple(camera_views or ())
        self.index = []
        for rec in self.records:
            with np.load(self.root / rec["arrays"], mmap_mode="r") as arrays:
                if arrays["action"].ndim != 2 or arrays["action"].shape[1] != 4:
                    raise ValueError(f"{rec['episode_id']} does not contain 4-D ACT actions")
                if (arrays["state"].ndim != 2
                        or arrays["state"].shape[1] != ACT_STATE_DIM):
                    raise ValueError(
                        f"{rec['episode_id']} does not contain "
                        f"{ACT_STATE_DIM}-D ACT states"
                    )
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
            for key in (f"observation.images.{view}" for view in self.camera_views)
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
                image_paths = _record_image_paths(record)
                for view in self.camera_views:
                    key = f"observation.images.{view}"
                    from PIL import Image

                    with Image.open(self.root / image_paths[view][int(frame_id)]) as pil_image:
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
        image_paths = _record_image_paths(rec)
        images = {}
        for view in self.camera_views:
            with Image.open(self.root / image_paths[view][i]) as image:
                images[f"observation.images.{view}"] = np.transpose(
                    np.asarray(image.convert("RGB"), dtype=np.uint8), (2, 0, 1)
                )
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
        sample = {
            # Keep uint8 at the dataset boundary, matching LeRobot's dataset
            # contract.  The official training loop converts camera values to
            # [0, 1] before the policy preprocessor applies dataset stats.
            "observation.state": state,
            "observation.environment_state": environment_state,
            "action": actions,
            "action_is_pad": ~valid,
        }
        sample.update(images)
        return sample
