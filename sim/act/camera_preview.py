"""Render four candidate overhead corner views without changing task cameras.

The generated cameras are added to a copy of the in-memory configuration and
exist only for this preview run. They are not written into the dataset or ACT
observation contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ..config import load_config
from ..environments.sweep_env import SweepEnv


def candidate_camera_specs(cfg, *, margin: float = 0.28, height: float = 0.92,
                           fovy_deg: float = 56.0) -> dict[str, dict]:
    """Return four symmetric cameras outside the table corners, aimed at center."""
    if margin <= 0.0:
        raise ValueError("camera margin must be positive")
    if height <= 0.0:
        raise ValueError("camera height must be positive")
    if not 0.0 < fovy_deg < 180.0:
        raise ValueError("camera vertical field of view must be between 0 and 180 degrees")

    half_x, half_y, _ = (float(v) for v in cfg.table.half_size)
    table_z = float(cfg.table.top_z)
    x = half_x + float(margin)
    y = half_y + float(margin)
    lookat = [0.0, 0.0, table_z]
    up = [0.0, 0.0, 1.0]

    specs: dict[str, dict] = {}
    for x_sign, x_label in ((-1.0, "xminus"), (1.0, "xplus")):
        for y_sign, y_label in ((-1.0, "yminus"), (1.0, "yplus")):
            name = f"preview_{x_label}_{y_label}"
            specs[name] = {
                "pos": [x_sign * x, y_sign * y, table_z + float(height)],
                "lookat": lookat.copy(),
                "up": up.copy(),
                "fovy_deg": float(fovy_deg),
            }
    return specs


def compose_contact_sheet(frames: Mapping[str, np.ndarray], *, columns: int = 2,
                          label_height: int = 32, gap: int = 12):
    """Build a labeled PIL contact sheet from equal-sized RGB camera frames."""
    if not frames:
        raise ValueError("at least one camera frame is required")
    if columns < 1 or label_height < 0 or gap < 0:
        raise ValueError("columns must be positive and spacing values non-negative")

    from PIL import Image, ImageDraw

    ordered = list(frames.items())
    first = np.asarray(ordered[0][1])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("camera frames must be HxWx3 RGB arrays")
    height, width, _ = first.shape
    if any(np.asarray(frame).shape != first.shape for _, frame in ordered):
        raise ValueError("all camera frames must have the same dimensions")

    rows = math.ceil(len(ordered) / columns)
    cell_height = height + label_height
    sheet = Image.new(
        "RGB",
        (columns * width + (columns + 1) * gap,
         rows * cell_height + (rows + 1) * gap),
        (24, 32, 40),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, frame) in enumerate(ordered):
        row, column = divmod(index, columns)
        left = gap + column * (width + gap)
        top = gap + row * (cell_height + gap)
        sheet.paste(Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB"),
                    (left, top))
        draw.text((left + 8, top + height + 8), name.replace("preview_", "").replace("_", " / "),
                  fill=(235, 243, 247))
    return sheet


def preview_render_size(cfg, width: int | None = None,
                        height: int | None = None) -> tuple[int, int]:
    """Use ACT's frame dimensions for preview exports unless overridden."""
    policy_width, policy_height = (int(value) for value in cfg.act.image_size)
    resolved = (policy_width if width is None else int(width),
                policy_height if height is None else int(height))
    if any(value < 1 for value in resolved):
        raise ValueError("render dimensions must be positive")
    return resolved


def render_candidate_views(*, output_dir: str | Path = "runs/five_view_camera_preview",
                           seed: int = 2026, width: int | None = None,
                           height: int | None = None,
                           config_path: str | Path | None = None) -> dict[str, Path]:
    """Render four views of one fixed randomized scene and a contact sheet."""
    from PIL import Image

    cfg = load_config(str(config_path) if config_path is not None else None)
    width, height = preview_render_size(cfg, width, height)
    specs = candidate_camera_specs(cfg)
    extra = cfg.video.extra_cameras.to_dict()
    extra.update(specs)
    cfg.set_path("video.extra_cameras", extra)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = SweepEnv(cfg, seed=int(seed))
    frames: dict[str, np.ndarray] = {}
    try:
        env.reset(seed=int(seed))
        for name in specs:
            frame = env.render_rgb(name, size=(int(width), int(height)))
            frames[name] = np.asarray(frame, dtype=np.uint8)
    finally:
        env.close()

    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / f"{name}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        paths[name] = path

    sheet_path = output / "contact_sheet.png"
    compose_contact_sheet(frames).save(sheet_path)
    paths["contact_sheet"] = sheet_path

    metadata_path = output / "preview_metadata.json"
    metadata_path.write_text(json.dumps({
        "seed": int(seed),
        "robot_model": str(cfg.end_effector.type),
        "part_count": len(env.layout),
        "render_size": [int(width), int(height)],
        "views": specs,
        "note": "Preview-only cameras; no policy, dataset, or formal camera config changed.",
    }, indent=2) + "\n", encoding="utf-8")
    paths["metadata"] = metadata_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="runs/five_view_camera_preview")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    paths = render_candidate_views(output_dir=args.output, seed=args.seed,
                                   width=args.width, height=args.height,
                                   config_path=args.config)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
