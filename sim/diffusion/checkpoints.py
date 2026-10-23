"""Resolve published inference weights without importing Torch or loading tensors."""
import json
from pathlib import Path


def resolve_checkpoint(path):
    if not path:
        raise ValueError("DP requires an explicit trained checkpoint")
    p = Path(path).resolve()
    if p.is_dir():
        # A run's stop-time export may predate resumed training. The full-stage
        # atomic publication pointer always wins; never auto-select pilot.
        if (p / "full").is_dir():
            p = p / "full"
        pointer = p / "latest.json"
        if pointer.is_file():
            try:
                name = json.loads(pointer.read_text())["checkpoint"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("invalid DP checkpoint pointer") from exc
            if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".pt":
                raise ValueError("invalid DP checkpoint pointer")
            candidate = (p / name).resolve()
            if candidate.parent != p:
                raise ValueError("invalid DP checkpoint pointer")
            p = candidate
        else:
            p = p / "final_ema.pt"
    if p.suffix != ".pt" or not p.is_file() or p.stat().st_size == 0:
        raise ValueError("diffusion checkpoint required; ACT checkpoints are incompatible")
    return str(p)
