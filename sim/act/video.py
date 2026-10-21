"""Stream saved inference frames into a five-camera MP4 (no policy execution)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from threading import Event, Timer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ENCODING_TIMEOUT_SECONDS = 300.0


def inference_video_root(cfg, project_root: str | Path) -> Path:
    root = Path(str(cfg.get_path("web.inference_video_dir", "runs/inference_videos")))
    return (root if root.is_absolute() else Path(project_root) / root).resolve()


def _font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def export_inference_video(episode_dir: str | Path, output_root: str | Path,
                           resource_check=None) -> Path:
    """Export a published replay, preserving old videos and partial-error cleanup.

    Image paths, frame rate, target and checkpoint come from the replay manifest.
    Only one composite image is held at a time; encoding is limited to two CPU
    threads and never loads a policy or starts a simulator.
    """
    import imageio_ffmpeg

    episode = Path(episode_dir).resolve()
    record = None
    with (episode.parent / "manifest.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            if line.strip():
                candidate = json.loads(line)
                if candidate.get("episode_id") == episode.name:
                    record = candidate
                    break
    if record is None:
        raise ValueError(f"no published replay record for {episode.name}")
    images = record.get("images") or {name: record[name]
        for name in ("overhead", "inspection", "wrist") if record.get(name)}
    views = (
        ("corner_xminus_yminus", "Corner X- Y-", (0, 0)),
        ("corner_xminus_yplus", "Corner X- Y+", (320, 0)),
        ("corner_xplus_yminus", "Corner X+ Y-", (0, 320)),
        ("corner_xplus_yplus", "Corner X+ Y+", (320, 320)),
        ("wrist", "Wrist camera", (640, 0)),
    )
    view_suffix = "five_view"
    if not any(name.startswith("corner_") for name in images) and images.get("overhead") and images.get("wrist"):
        # Older inference replays predate the dynamic five-camera map.
        # Keep only their recorded cameras; do not manufacture missing views.
        views = tuple((name, name.capitalize() + " camera", (x, 0))
                      for name, x in (("overhead", 0), ("inspection", 320), ("wrist", 640))
                      if name in images)
        view_suffix = "three_view" if "inspection" in images else "two_view"
    fps = float(record.get("fps", 25))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("replay frame rate must be positive and finite")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    font, small, caption_font = _font(16), _font(14), _font(12)
    target = int(record.get("target_count", 0))
    model = Path(str(record.get("model") or ""))
    checkpoint = model.name or "untrained policy"
    model_name = (model.parent.parent.name if model.parent.name == "checkpoints"
                  else model.parent.name)
    model_caption = textwrap.wrap(f"Model: {model_name or 'not recorded'}", width=42)
    desired = record.get("desired_force")
    desired_label = "Desired force not recorded" if desired is None else f"Desired force {float(desired):.2f} N"

    with np.load(episode / "arrays.npz", allow_pickle=False) as data:
        count = len(data["t"])
        if count == 0:
            raise ValueError("replay contains no frames")
        # Validate before encoding; do not publish misleading placeholder views.
        frame_paths = {}
        for name, _, _ in views:
            paths = images.get(name, [])
            if len(paths) != count:
                raise ValueError(f"replay camera {name} has {len(paths)} of {count} frames")
            frame_paths[name] = []
            for relative in paths:
                path = (episode.parent / relative).resolve()
                if path.parent != episode or not path.is_file():
                    raise FileNotFoundError(f"missing or unsafe replay frame: {relative}")
                frame_paths[name].append(path)
        raw = data["raw_normal_force"] if "raw_normal_force" in data else None
        filtered = data["normal_force"]
        ceiling = max(10.0, float(np.max(raw)) if raw is not None else 0.0,
                      float(np.max(filtered))) * 1.1
        fd, temporary_name = tempfile.mkstemp(prefix=".inference-video-", suffix=".mp4", dir=output_root)
        os.close(fd)
        temporary = Path(temporary_name)
        process = None
        watchdog = None
        deadline_expired = Event()
        try:
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen([
                    imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-hide_banner",
                    "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "960x640", "-framerate", str(fps), "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(temporary),
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)

                def stop_stalled_encoder():
                    if process.poll() is None:
                        deadline_expired.set()
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass

                # A deadline around wait() alone cannot interrupt a blocked pipe
                # write or close. Killing its reader releases both operations.
                watchdog = Timer(ENCODING_TIMEOUT_SECONDS, stop_stalled_encoder)
                watchdog.daemon = True
                watchdog.start()
                try:
                    for i in range(count):
                        if resource_check is not None:
                            resource_check()
                        frame = Image.new("RGB", (960, 640), "#111923")
                        draw = ImageDraw.Draw(frame)
                        for name, label, (x, y) in views:
                            with Image.open(frame_paths[name][i]) as source:
                                tile = source.convert("RGB").resize((320, 320))
                                frame.paste(tile, (x, y))
                            draw.rectangle((x, y, x + 319, y + 28), fill="#101820")
                            draw.text((x + 9, y + 5), label, font=font, fill="white")
                        lines = (
                            (332, f"{'DP' if record.get('policy_type')=='diffusion' else 'ACT'} {checkpoint} | target {target}", "white"),
                            (386, f"Frame {i + 1}/{count} | t {data['t'][i]:.2f} s", "#dae1e9"),
                            (408, (f"Raw Fz       {raw[i]:7.2f} N" if raw is not None
                                   else "Raw Fz       not recorded"), "#65b7ff"),
                            (430, f"Filtered Fz  {filtered[i]:7.2f} N", "#ffd16b"),
                        )
                        for y, label, color in lines:
                            draw.text((654, y), label, font=font, fill=color)
                        for line_index, label in enumerate(model_caption[:2]):
                            draw.text((654, 356 + line_index * 15), label, font=caption_font, fill="#a9b9c8")
                        draw.text((654, 452), desired_label, font=small, fill="#a9b9c8")
                        draw.text((654, 474), f"TCP Z {data['tcp_pose'][i, 2] * 1000:.2f} mm | cmd {data['reference'][i, 2] * 1000:.2f}", font=small, fill="#a9b9c8")
                        draw.text((654, 496), f"Collected {data['fully_collected'][i]}/{target} | {data['z_owner'][i]}", font=small, fill="#a9b9c8")
                        x0, x1, y0, y1 = 654, 945, 516, 604
                        draw.rectangle((x0, y0, x1, y1), outline="#364757")
                        window = max(1, int(round(5 * fps)))
                        start = max(0, i - window + 1)
                        for values, color in ((raw, "#65b7ff"), (filtered, "#ffd16b")):
                            if values is None:
                                continue
                            points = [(x0 + (j - start) / max(1, window - 1) * (x1 - x0),
                                       y1 - min(ceiling, max(0.0, float(values[j]))) / ceiling * (y1 - y0))
                                      for j in range(start, i + 1)]
                            if len(points) > 1:
                                draw.line(points, fill=color, width=2)
                        draw.text((654, 612), "Force history: last 5 seconds", font=small, fill="#a9b9c8")
                        process.stdin.write(frame.tobytes())
                except OSError as exc:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
                    errors.seek(0)
                    detail = errors.read(8192).decode("utf-8", errors="replace")
                    if deadline_expired.is_set():
                        raise TimeoutError("video encoding exceeded its deadline") from exc
                    raise RuntimeError(f"video encoding failed: {detail or str(exc)}") from exc
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        # Preserve the useful diagnostic from writing, or let
                        # the process exit status below report a close failure.
                        pass
                code = process.wait(timeout=30)
                if deadline_expired.is_set():
                    raise TimeoutError("video encoding exceeded its deadline")
                if code or temporary.stat().st_size == 0:
                    errors.seek(0)
                    detail = errors.read(8192).decode("utf-8", errors="replace")
                    raise RuntimeError(f"video encoding failed ({code}): {detail}")
            # Publish with an exclusive hard link: even concurrent exports cannot
            # overwrite an existing video, and unfinished MP4s are never served.
            ordinal = 1
            while True:
                suffix = "" if ordinal == 1 else f"_{ordinal:02d}"
                output = output_root / f"{episode.name}_{view_suffix}{suffix}.mp4"
                try:
                    os.link(temporary, output)
                    return output
                except FileExistsError:
                    ordinal += 1
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            temporary.unlink(missing_ok=True)
