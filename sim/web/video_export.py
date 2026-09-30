"""Synchronized side-by-side MP4 export for saved workbench episodes."""

from __future__ import annotations

import re
from pathlib import Path


CAMERA_ROLES = ("overhead", "wrist", "inspection")
CAMERA_LABELS = ("OVERHEAD", "WRIST", "INSPECTION")


def safe_video_filename(episode_id: str) -> str:
    """Create a portable download name from a public episode identifier."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(episode_id)).strip("._-")
    return f"{stem or 'episode'}.mp4"


def export_triptych_mp4(source: dict, destination: str | Path) -> Path:
    """Encode each saved time step as one overhead/wrist/inspection frame."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    frames = source.get("frames") or []
    fps = float(source.get("fps", 0))
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("episode has no frames or an invalid frame rate")

    panel_width = panel_height = 0
    for frame in frames:
        for role in CAMERA_ROLES:
            path = frame.get(role)
            if path is None:
                continue
            with Image.open(path) as image:
                width, height = image.size
            panel_width = max(panel_width, width)
            panel_height = max(panel_height, height)
    if panel_width <= 0 or panel_height <= 0:
        raise FileNotFoundError("episode has no readable camera frames")

    # H.264's common browser-compatible pixel format requires even dimensions.
    panel_width += panel_width % 2
    header_height = max(22, min(36, panel_height // 12))
    header_height += (panel_height + header_height) % 2
    canvas_size = (panel_width * len(CAMERA_ROLES), panel_height + header_height)
    font_size = max(10, min(18, header_height - 7))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:  # pragma: no cover - platform font availability
        font = ImageFont.load_default()

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output), fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            canvas = Image.new("RGB", canvas_size, (14, 20, 25))
            draw = ImageDraw.Draw(canvas)
            for index, (role, label) in enumerate(zip(CAMERA_ROLES, CAMERA_LABELS)):
                x0 = index * panel_width
                draw.text((x0 + 7, 5), label, fill=(211, 229, 234), font=font)
                path = frame.get(role)
                if path is None:
                    draw.text((x0 + 7, header_height + 8), "NOT RECORDED",
                              fill=(148, 170, 180), font=font)
                    continue
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
                        x = x0 + (panel_width - image.width) // 2
                        y = header_height + (panel_height - image.height) // 2
                        canvas.paste(image, (x, y))
                except FileNotFoundError:
                    draw.text((x0 + 7, header_height + 8), "FRAME MISSING",
                              fill=(240, 160, 160), font=font)
            writer.append_data(np.asarray(canvas, dtype=np.uint8))
    return output
