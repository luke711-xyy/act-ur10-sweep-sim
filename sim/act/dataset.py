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


class ActDataset:
    """Returns one observation and a padded future action chunk."""

    def __init__(self, root: str, chunk_size: int = 20):
        self.root = Path(root)
        self.chunk_size = int(chunk_size)
        self.records = [json.loads(line) for line in (self.root / "manifest.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
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
            "observation.state": np.asarray(arrays["state"][i], dtype=np.float32),
            "observation.environment_state": np.asarray(arrays["environment_state"][i], dtype=np.float32),
            "action": actions,
            "action_is_pad": np.array([False] * (self.chunk_size - pad) + [True] * pad),
        }
