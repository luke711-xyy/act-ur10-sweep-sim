"""Local state and episode inspection helpers for the MuJoCo workbench.

The workbench deliberately reads the existing ACT manifest format.  Preview
episodes live in a separate directory and therefore cannot accidentally enter
the training dataset.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from ..act.dataset import ActDatasetWriter
from ..act.rollout import run_expert_episode


class EpisodeNotFound(LookupError):
    """Requested episode does not exist in the selected local store."""


class FrameNotFound(IndexError):
    """Requested episode frame is outside the recorded range."""


def _read_manifest(root: Path) -> list[dict]:
    path = root / "manifest.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


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
        self._preview_records: dict[str, dict] = {}

    def _records(self) -> list[tuple[Path, dict, bool]]:
        records = [(self.dataset_root, rec, False)
                   for rec in _read_manifest(self.dataset_root)]
        preview_records = _read_manifest(self.preview_root)
        self._preview_records = {rec["episode_id"]: rec for rec in preview_records}
        records.extend((self.preview_root, rec, True) for rec in preview_records)
        return records

    def list_episodes(self) -> list[dict]:
        output = []
        for root, rec, preview in self._records():
            output.append({
                "episode_id": rec["episode_id"],
                "success": bool(rec.get("success", False)),
                "seed": rec.get("seed"),
                "count": rec.get("count"),
                "length": _frame_count(root, rec),
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
        root, rec, preview = self._find(episode_id)
        return {**rec, "length": _frame_count(root, rec), "preview": preview}

    def load_episode_frame(self, episode_id: str, frame_index: int) -> dict:
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
        root, rec, _ = self._find(episode_id)
        path = root / rec["episode_id"] / "signals.json"
        if not path.is_file():
            return {"episode_id": rec["episode_id"], "t": [], "fz": [],
                    "contact": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "episode_id": rec["episode_id"],
            "t": [float(value) for value in data.get("t", [])],
            "fz": [float(value) for value in data.get("fz", [])],
            "contact": [bool(value) for value in data.get("contact", [])],
        }

    def build_preview(self, seed: int, count: int | None = None) -> dict:
        local_cfg = self.cfg.copy()
        if count is not None:
            local_cfg.set_path("components.count", int(count))
        result = run_expert_episode(local_cfg, seed=int(seed), collect_observations=True)
        self.preview_root.mkdir(parents=True, exist_ok=True)
        episode_id = f"preview_{uuid.uuid4().hex[:10]}"
        metadata = {
            "preview": True,
            "seed": int(seed),
            "count": int(count) if count is not None else int(local_cfg.components.count),
            "failure_reason": result.failure_reason,
        }
        ActDatasetWriter(str(self.preview_root)).add_episode(
            episode_id, result.observations, result.actions, result.success, metadata)
        from PIL import Image

        record = _read_manifest(self.preview_root)[-1]
        inspection_paths = []
        for index, observation in enumerate(result.observations):
            image = observation.get("inspection")
            if image is None:
                inspection_paths.append("")
                continue
            path = self.preview_root / episode_id / f"inspection_{index:05d}.png"
            Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
            inspection_paths.append(str(path.relative_to(self.preview_root)))
        if inspection_paths:
            record["inspection"] = inspection_paths
            lines = _read_manifest(self.preview_root)
            lines[-1] = record
            (self.preview_root / "manifest.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in lines),
                encoding="utf-8")
        signals = {
            "t": [float(row["t"]) for row in result.trace],
            "fz": [float(row["normal_force"]) for row in result.trace],
            "contact": [bool(row["contact"]) for row in result.trace],
        }
        (self.preview_root / episode_id / "signals.json").write_text(
            json.dumps(signals, ensure_ascii=False), encoding="utf-8")
        return self.episode_metadata(episode_id)
