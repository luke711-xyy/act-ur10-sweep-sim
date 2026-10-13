"""Read-only export of published episode media; no simulator or policy execution."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event, Thread
import time
import uuid

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from ..act.video import _font

VIEWS = (
    ("corner_xminus_yminus", "Corner X- / Y-", 0, 0),
    ("corner_xminus_yplus", "Corner X- / Y+", 320, 0),
    ("wrist", "Wrist camera", 640, 0),
    ("corner_xplus_yminus", "Corner X+ / Y-", 0, 344),
    ("corner_xplus_yplus", "Corner X+ / Y+", 320, 344),
)
ENCODING_TIMEOUT_SECONDS = 300.


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _episode_file(root, episode, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("episode media must use relative saved paths")
    raw = root / relative
    path = raw.resolve()
    if raw.is_symlink() or path.parent != episode or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe episode media: {relative}")
    return path


def _run_media(command, *, cancel_event, timeout):
    """Bound the whole decoder process, including header/first-frame reads."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if cancel_event.wait(.05):
                    raise RuntimeError("video export cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("video validation exceeded its deadline")
            output.seek(0)
            errors.seek(0)
            stdout, stderr = output.read().decode(errors="replace"), errors.read().decode(errors="replace")
            if process.returncode:
                raise RuntimeError("video validation failed: " + stderr[:1000])
            return stdout, stderr
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def _verify(video, frames, fps, size, cancel_event):
    # showinfo reports the actual decoded first-frame geometry and stream rate,
    # avoiding an unbounded generator read or a new ffprobe dependency.
    _, metadata = _run_media([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "info", "-xerror",
        "-threads", "1", "-i", str(video), "-map", "0:v:0", "-an",
        "-filter_threads", "1", "-vf", "showinfo", "-frames:v", "1", "-f", "null", "-",
    ], cancel_event=cancel_event, timeout=30)
    dimensions = re.search(r"\bs:(\d+)x(\d+)", metadata)
    rate = re.search(r"config in time_base:.*?frame_rate:\s*(\d+)/(\d+)", metadata)
    if (not dimensions or not rate or not int(rate[2])
            or tuple(map(int, dimensions.groups())) != tuple(size)
            or abs(int(rate[1]) / int(rate[2]) - fps) > .01):
        raise RuntimeError("video dimensions or FPS do not match saved media")
    stdout, stderr = _run_media([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
        "-threads", "1", "-i", str(video), "-map", "0:v:0", "-an",
        "-progress", "pipe:1", "-threads", "1", "-f", "null", "-",
    ], cancel_event=cancel_event, timeout=120)
    counts = [int(line.split("=", 1)[1]) for line in stdout.splitlines()
              if line.startswith("frame=")]
    if stderr.strip() or not counts or counts[-1] != frames:
        raise RuntimeError(f"video validation failed: expected {frames} frames; "
                           f"decoded {counts[-1] if counts else 0}: {stderr[:500]}")


def export_saved_episode(root, record, output_root, *, video_root=None,
                         progress=None, cancel_event=None):
    """Encode every recorded frame, or copy a verified cleaned inference MP4.

    Output is a separate export plus provenance sidecar. No manifest, image,
    telemetry, simulation, control, policy or training state is modified.
    """
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    name = str(record.get("episode_id", ""))
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("invalid published episode ID")
    raw_episode = root / name
    episode = raw_episode.resolve()
    if episode.parent != root or raw_episode.is_symlink() or not episode.is_dir():
        raise ValueError("episode directory is outside its store or missing")
    # A collector may append other records. Only the captured record must stay
    # identical; do not freeze or rewrite the live manifest.
    def check_record():
        rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
                if line.strip()]
        if not any(row == record for row in rows):
            raise ValueError("selected published episode changed during export")
    check_record()
    cancelled = cancel_event or Event()
    notify = progress or (lambda _p: None)

    def check_cancel():
        if cancelled.is_set():
            raise RuntimeError("video export cancelled")

    check_cancel()
    images = record.get("images") or {}
    if set(images) != {v[0] for v in VIEWS}:
        raise ValueError("record is not a complete five-camera episode; missing views are not invented")
    fps = float(record.get("fps", 25))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("saved frame rate must be positive and finite")
    arrays_path = _episode_file(root, episode, record["arrays"])
    arrays_sha = _sha(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ("t", "phase", "fully_collected", "normal_force")}
    count = len(arrays["t"])
    if (not count or any(len(a) != count for a in arrays.values())
            or not np.isfinite(arrays["t"]).all() or not np.isfinite(arrays["normal_force"]).all()):
        raise ValueError("saved telemetry has invalid frame lengths or non-finite values")
    layout = {v: [x, y + 24, 320, 320] for v, _, x, y in VIEWS}
    result = {"episode_id": name, "episode_kind": record.get("episode_kind", "unknown"),
              "seed": record.get("seed"), "target_count": record.get("target_count"),
              "success": bool(record.get("success")), "frames": count, "fps": fps,
              "source_root": str(root), "arrays_sha256": arrays_sha,
              "reused": bool(record.get("frame_images_deleted"))}
    source_hashes, paths = {}, {}
    original_video = None
    size = (960, 688)
    notify({"stage": "validating", "frames": 0, "total_frames": count})
    if result["reused"]:
        original_video = Path(str(record.get("video_path", "")))
        resolved = original_video.resolve()
        verification = record.get("image_cleanup", {}).get("video_verification", {})
        expected_layout = {v: [x, y, 320, 320] for v, (x, y) in zip(
            ("corner_xminus_yminus", "corner_xminus_yplus", "corner_xplus_yminus",
             "corner_xplus_yplus", "wrist"), ((0, 0), (320, 0), (0, 320), (320, 320), (640, 0)))}
        if (record.get("episode_kind") != "inference"
                or not re.fullmatch(re.escape(name) + r"_five_view(?:_\d{2,})?\.mp4", original_video.name)
                or record.get("video_layout") != expected_layout
                or video_root is None or original_video.is_symlink()
                or not resolved.is_relative_to(Path(video_root).resolve())
                or not resolved.is_file() or resolved.suffix != ".mp4"
                or verification.get("sha256") != _sha(resolved)
                or verification.get("frames") != count
                or abs(float(verification.get("fps", 0)) - fps) > .01
                or set(record.get("video_layout", {})) != set(images)):
            raise ValueError("cleaned inference video identity/path verification failed")
        original_video = resolved
        layout = record["video_layout"]
        size = tuple(verification.get("size", ()))
        if size != (960, 640):
            raise ValueError("unsupported saved inference video layout")
        source_hashes[str(original_video)] = verification["sha256"]
    else:
        for view, _, _, _ in VIEWS:
            if len(images[view]) != count:
                raise ValueError(f"saved {view} camera has {len(images[view])} of {count} frames")
            paths[view] = [_episode_file(root, episode, relative) for relative in images[view]]

    # Temp files are never available through the HTTP route. Exclusive publish
    # and unique names preserve existing exports even across server restarts.
    output_root.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".episode-export-", suffix=".mp4", dir=output_root)
    os.close(fd)
    temporary = Path(temporary_name)
    process = watchdog = None
    finished = Event()
    timed_out = Event()
    try:
        if original_video:
            check_cancel()
            notify({"stage": "copying", "frames": count, "total_frames": count})
            shutil.copyfile(original_video, temporary)
        else:
            font, small = _font(17), _font(15)
            kind = "ACT inference" if record.get("episode_kind") == "inference" else "Expert demonstration"
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen([
                    imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "960x688",
                    "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-threads", "1", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(temporary),
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)

                def deadline():
                    expires = time.monotonic() + ENCODING_TIMEOUT_SECONDS
                    while not finished.wait(.05) and process.poll() is None:
                        if cancelled.is_set() or time.monotonic() >= expires:
                            if not cancelled.is_set():
                                timed_out.set()
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                            return
                watchdog = Thread(target=deadline, daemon=True)
                watchdog.start()
                try:
                    for i in range(count):
                        check_cancel()
                        frame = Image.new("RGB", size, "#111923")
                        draw = ImageDraw.Draw(frame)
                        for view, label, x, y in VIEWS:
                            raw = paths[view][i].read_bytes()
                            source_hashes[str(paths[view][i])] = hashlib.sha256(raw).hexdigest()
                            with Image.open(io.BytesIO(raw)) as tile:
                                frame.paste(tile.convert("RGB").resize((320, 320)), (x, y + 24))
                            draw.text((x + 8, y + 3), label, font=font, fill="#e5edf5")
                        lines = [kind, "Saved frames - no new simulation", name,
                                 f"Seed: {record.get('seed', 'not recorded')}",
                                 f"Goal: {record.get('target_count', '?')} of 6 parts",
                                 f"Collected: {int(arrays['fully_collected'][i])}",
                                 f"Phase: {arrays['phase'][i]}",
                                 f"Frame: {i + 1}/{count} | {fps:g} fps",
                                 f"Elapsed: {i / fps:.2f} s",
                                 f"Simulation t: {arrays['t'][i]:.2f} s",
                                 f"Measured Fz: {arrays['normal_force'][i]:.2f} N",
                                 "Full episode | original speed",
                                 "Saved result: " + ("SUCCESS" if result['success'] else "FAILURE")]
                        for j, label in enumerate(lines):
                            draw.text((650, 352 + 24 * j), label, font=font if j == 0 else small,
                                      fill="#e5edf5" if j != 12 else ("#8ddfc1" if result['success'] else "#ffaaaa"))
                        process.stdin.write(frame.tobytes())
                        if i == 0 or (i + 1) % 25 == 0 or i + 1 == count:
                            notify({"stage": "encoding", "frames": i + 1, "total_frames": count})
                except OSError as exc:
                    check_cancel()
                    if timed_out.is_set():
                        raise TimeoutError("video encoding exceeded its deadline") from exc
                    raise RuntimeError("video encoder pipe failed") from exc
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                code = process.wait(timeout=30)
                if code or timed_out.is_set():
                    errors.seek(0)
                    raise RuntimeError("video encoding failed or timed out: " + errors.read(8192).decode(errors="replace"))
        check_cancel()
        notify({"stage": "verifying", "frames": count, "total_frames": count})
        _verify(temporary, count, fps, size, cancelled)
        if _sha(arrays_path) != arrays_sha or any(_sha(p) != h for p, h in source_hashes.items()):
            raise ValueError("source media changed during video export")
        check_record()
        check_cancel()
        output = output_root / f"episode_{name}_{uuid.uuid4().hex[:12]}.mp4"
        result.update(video_path=str(output), video_layout=layout, size=list(size),
                      sha256=_sha(temporary), bytes=temporary.stat().st_size,
                      record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest())
        os.link(temporary, output)
        try:
            with output.with_suffix(".json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
        except Exception:
            output.unlink()  # Only our unpublished new export, never a source.
            raise
        notify({**result, "stage": "complete", "total_frames": count})
        return result
    finally:
        if watchdog is not None:
            finished.set()
            watchdog.join(timeout=1)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        temporary.unlink(missing_ok=True)
