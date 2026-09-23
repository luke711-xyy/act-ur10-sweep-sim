"""Small local dataset format for MuJoCo ACT demonstrations."""

from __future__ import annotations

import json
import os
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
        episode_dir = self.root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        overhead, wrist = [], []
        for i, obs in enumerate(observations):
            overhead_path = episode_dir / f"overhead_{i:05d}.png"
            wrist_path = episode_dir / f"wrist_{i:05d}.png"
            Image.fromarray(np.asarray(obs["overhead"], dtype=np.uint8)).save(overhead_path)
            Image.fromarray(np.asarray(obs["wrist"], dtype=np.uint8)).save(wrist_path)
            overhead.append(str(overhead_path.relative_to(self.root)))
            wrist.append(str(wrist_path.relative_to(self.root)))
        states = np.stack([o["state"] for o in observations]).astype(np.float32)
        env_states = np.stack([o["environment_state"] for o in observations]).astype(np.float32)
        np.savez_compressed(episode_dir / "arrays.npz", state=states,
                            environment_state=env_states,
                            action=np.asarray(actions, dtype=np.float32))
        record = {"episode_id": episode_id, "success": bool(success),
                  "overhead": overhead, "wrist": wrist,
                  "arrays": str((episode_dir / "arrays.npz").relative_to(self.root)),
                  **metadata}
        with self.manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return episode_dir


def compute_dataset_stats(root: str | Path, records: list[dict] | None = None) -> dict:
    """Compute mean/std statistics from the training episodes only."""
    root = Path(root)
    if records is None:
        records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    states, env_states, actions = [], [], []
    for rec in records:
        arrays = np.load(root / rec["arrays"])
        states.append(np.asarray(arrays["state"], dtype=np.float32))
        env_states.append(np.asarray(arrays["environment_state"], dtype=np.float32))
        actions.append(np.asarray(arrays["action"], dtype=np.float32))
    if not actions:
        raise ValueError("cannot compute ACT statistics from an empty dataset")

    def feature_stats(values):
        values = np.concatenate(values, axis=0)
        return {
            "mean": values.mean(axis=0).astype(np.float32).tolist(),
            "std": np.maximum(values.std(axis=0), 1e-6).astype(np.float32).tolist(),
        }

    return {
        "state": feature_stats(states),
        "environment_state": feature_stats(env_states),
        "action": feature_stats(actions),
        "version": 1,
    }


def load_dataset_stats(model_path: str | Path | None) -> dict | None:
    if model_path is None:
        return None
    path = Path(model_path)
    for candidate in (path / "dataset_stats.json", path.parent / "dataset_stats.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def normalize_feature(value: np.ndarray, stats: dict | None, name: str) -> np.ndarray:
    if not stats or name not in stats:
        return np.asarray(value, dtype=np.float32)
    item = stats[name]
    mean = np.asarray(item["mean"], dtype=np.float32)
    std = np.asarray(item["std"], dtype=np.float32)
    return (np.asarray(value, dtype=np.float32) - mean) / std


def unnormalize_action(value: np.ndarray, stats: dict | None) -> np.ndarray:
    if not stats or "action" not in stats:
        return np.asarray(value, dtype=np.float32)
    item = stats["action"]
    mean = np.asarray(item["mean"], dtype=np.float32)
    std = np.asarray(item["std"], dtype=np.float32)
    return np.asarray(value, dtype=np.float32) * std + mean


class ActDataset:
    """Returns one observation and a padded future action chunk."""

    def __init__(self, root: str, chunk_size: int = 20, stats: dict | None = None):
        self.root = Path(root)
        self.chunk_size = int(chunk_size)
        self.records = [json.loads(line) for line in (self.root / "manifest.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        self.stats = stats or compute_dataset_stats(self.root, self.records)
        self.index = []
        for rec in self.records:
            arrays = np.load(self.root / rec["arrays"], mmap_mode="r")
            self.index.extend((len(self.index), rec, i) for i in range(len(arrays["action"])))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        from PIL import Image

        _, rec, i = self.index[index]
        arrays = np.load(self.root / rec["arrays"])
        episode_dir = self.root / rec["episode_id"]
        overhead = np.asarray(Image.open(self.root / rec["overhead"][i]).convert("RGB"), dtype=np.float32)
        wrist = np.asarray(Image.open(self.root / rec["wrist"][i]).convert("RGB"), dtype=np.float32)
        actions = np.asarray(arrays["action"][i:i + self.chunk_size], dtype=np.float32)
        pad = self.chunk_size - len(actions)
        if pad > 0:
            actions = np.concatenate((actions, np.repeat(actions[-1:], pad, axis=0)), axis=0)
        return {
            "observation.images.overhead": np.transpose(overhead, (2, 0, 1)) / 255.0,
            "observation.images.wrist": np.transpose(wrist, (2, 0, 1)) / 255.0,
            "observation.state": normalize_feature(arrays["state"][i], self.stats, "state"),
            "observation.environment_state": normalize_feature(
                arrays["environment_state"][i], self.stats, "environment_state"
            ),
            "action": normalize_feature(actions, self.stats, "action"),
            "action_is_pad": np.array([False] * (self.chunk_size - pad) + [True] * pad),
        }
