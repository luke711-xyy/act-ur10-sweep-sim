"""Stable names for human-facing expert demonstrations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


_DEMO_NAME = re.compile(r"^preview_goal_(?P<target>[1-6])_(?P<serial>[0-9]+)$")


def demo_name(target_count: int, serial: int) -> str:
    """Return the canonical target-aware, per-target serialised episode name."""
    target_count = int(target_count)
    serial = int(serial)
    if not 1 <= target_count <= 6:
        raise ValueError("target_count must be between 1 and 6")
    if serial < 1:
        raise ValueError("serial must be positive")
    return f"preview_goal_{target_count}_{serial:04d}"


def _manifest_names(root: Path) -> Iterable[str]:
    manifest = Path(root) / "manifest.jsonl"
    if not manifest.is_file():
        return ()
    names = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            episode_id = str(json.loads(line).get("episode_id", ""))
        except json.JSONDecodeError:
            continue
        if episode_id:
            names.append(episode_id)
    return names


def _episode_directory_names(root: Path) -> Iterable[str]:
    """Return canonical episode directory names, including orphaned ones."""
    root = Path(root)
    if not root.is_dir():
        return ()
    return tuple(
        child.name for child in root.iterdir()
        if child.is_dir() and _DEMO_NAME.match(child.name)
    )


def next_demo_serial(roots: Iterable[Path], target_count: int) -> int:
    """Find the first unused suffix for one target count across manifests.

    Filling the first hole keeps a repaired or deleted batch contiguous.  The
    serial is still scoped to ``target_count`` and is checked across both the
    preview and formal dataset manifests.
    """
    target_count = int(target_count)
    if not 1 <= target_count <= 6:
        raise ValueError("target_count must be between 1 and 6")
    used = set()
    for root in roots:
        root = Path(root)
        for name in (*_manifest_names(root), *_episode_directory_names(root)):
            match = _DEMO_NAME.match(name)
            if match and int(match.group("target")) == target_count:
                used.add(int(match.group("serial")))
    serial = 1
    while serial in used:
        serial += 1
    return serial


def next_demo_name(roots: Iterable[Path], target_count: int) -> str:
    return demo_name(target_count, next_demo_serial(roots, target_count))
