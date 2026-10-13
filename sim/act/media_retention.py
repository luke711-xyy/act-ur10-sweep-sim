"""Keep verified MP4s and telemetry, not disposable inference frame images."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np


def _records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _image_map(record: dict) -> dict:
    return record.get("images") or {name: record[name]
        for name in ("overhead", "inspection", "wrist") if record.get(name)}


def _verify_causal_trace(root: Path, episode: Path, record: dict) -> None:
    # Historical records may not have an original trace. Never fabricate it.
    if not record.get("causal_trace_required"):
        return
    try:
        raw = root / record["causal_trace"]
        trace = raw.resolve()
        if trace.parent != episode or raw.is_symlink() or not trace.is_file():
            raise ValueError("causal trace is missing or outside the episode")
        if hashlib.sha256(trace.read_bytes()).hexdigest() != record["causal_trace_sha256"]:
            raise ValueError("causal trace checksum mismatch")
        with gzip.open(trace, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)  # Reads to EOF, including gzip CRC.
        if not isinstance(rows, list) or not rows or len(rows) != record["causal_trace_rows"]:
            raise ValueError("causal trace row count mismatch")
    except (KeyError, ValueError, OSError, EOFError) as exc:
        raise RuntimeError(f"causal trace validation failed: {exc}") from exc


def verify_inference_video(video: Path, *, frames: int, fps: float) -> dict:
    """Decode the complete video before granting permission to remove images."""
    import imageio_ffmpeg

    video = Path(video).resolve()
    if video.suffix != ".mp4" or not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError("video is missing, empty or not an MP4")
    reader = imageio_ffmpeg.read_frames(str(video), output_params=["-threads", "1"])
    try:
        meta = next(reader)
        next(reader)  # The first frame must actually decode, not just its header.
    finally:
        reader.close()
    if tuple(meta["size"]) != (960, 640) or abs(float(meta["fps"]) - fps) > 0.01:
        raise RuntimeError("video dimensions or frame rate do not match the replay")
    decoded = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-threads", "1",
        "-i", str(video), "-map", "0:v:0", "-an", "-progress", "pipe:1",
        "-threads", "1", "-f", "null", "-",
    ], capture_output=True, text=True, timeout=120)
    counts = [int(line.split("=", 1)[1].strip())
              for line in decoded.stdout.splitlines() if line.startswith("frame=")]
    if decoded.returncode or decoded.stderr.strip() or not counts or counts[-1] != frames:
        raise RuntimeError(f"video failed full-frame validation: {decoded.stderr.strip()} "
                           f"decoded={counts[-1] if counts else 0}, expected={frames}")
    return {"frames": frames, "fps": fps, "size": [960, 640],
            "bytes": video.stat().st_size,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}


def finalize_inference_video(episode_dir: str | Path, video_path: str | Path) -> dict:
    """Validate media, publish playback metadata, then remove only own images.

    The dataset writer's lock serializes manifest changes against a collector
    or another inference. Every deletion target is validated before any file
    is touched. Expert images, arrays, control traces and videos are retained.
    """
    episode = Path(episode_dir).resolve()
    root = episode.parent
    video = Path(video_path).resolve()
    record = next((r for r in _records(root) if r["episode_id"] == episode.name), None)
    if record is None or record.get("episode_kind") != "inference":
        raise ValueError("image cleanup is permitted only for a published inference")
    arrays = (root / record["arrays"]).resolve()
    if arrays.parent != episode or not arrays.is_file():
        raise ValueError("inference telemetry is missing or outside the episode")
    with np.load(arrays, allow_pickle=False) as values:
        frames = len(values["t"])
    _verify_causal_trace(root, episode, record)
    verification = verify_inference_video(video, frames=frames,
                                         fps=float(record.get("fps", 25)))
    targets = set()
    for paths in _image_map(record).values():
        for relative in paths:
            raw = root / relative
            path = raw.resolve()
            if (path.parent != episode or path.suffix.lower() not in {".png", ".webp"}
                    or raw.is_symlink()):
                raise ValueError(f"image cleanup target is outside its episode: {relative}")
            targets.add(path)
    existing = [path for path in targets if path.is_file()]
    removed_bytes = sum(path.stat().st_size for path in existing)
    names = set(_image_map(record))
    corners = ("corner_xminus_yminus", "corner_xminus_yplus",
               "corner_xplus_yminus", "corner_xplus_yplus", "wrist")
    layout = {name: [x, y, 320, 320] for name, (x, y) in zip(
        corners, ((0, 0), (320, 0), (0, 320), (320, 320), (640, 0))) if name in names}
    for name, x in (("overhead", 0), ("inspection", 320), ("wrist", 640)):
        if name in names:
            layout[name] = [x, 0, 320, 320]
    report = {"deleted_files": len(existing), "freed_bytes": removed_bytes,
              "video_verification": verification, "completed_at": time.time()}
    manifest = root / "manifest.jsonl"
    with (root / ".manifest.jsonl.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = _records(root)
        fresh = next((r for r in records if r["episode_id"] == episode.name), None)
        if fresh is None or fresh.get("episode_kind") != "inference":
            raise ValueError("inference record changed before image cleanup")
        _verify_causal_trace(root, episode, fresh)
        previous = fresh.get("image_cleanup", {})
        stored_report = {**report,
            "deleted_files": len(existing) + int(previous.get("deleted_files", 0)),
            "freed_bytes": removed_bytes + int(previous.get("freed_bytes", 0))}
        fresh.update({"video_path": str(video), "video_layout": layout,
                      "frame_images_deleted": True, "image_cleanup": stored_report})
        temporary = root / f".manifest-retention-{uuid.uuid4().hex}.tmp"
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in records), encoding="utf-8")
        os.replace(temporary, manifest)
        # Playback has switched to the verified MP4 before images disappear.
        for path in existing:
            path.unlink(missing_ok=True)
    return report


def main(argv=None) -> int:
    from .video import export_inference_video
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--export-missing", action="store_true")
    parser.add_argument("--skip-seed", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    root, video_root = Path(args.root).resolve(), Path(args.video_root).resolve()
    report = {"root": str(root), "episodes": [], "skipped": [], "errors": []}
    for record in _records(root):
        if record.get("episode_kind") != "inference":
            continue
        if record.get("seed") in args.skip_seed:
            report["skipped"].append(record["episode_id"])
            continue
        episode = root / record["episode_id"]
        candidates = ([Path(record["video_path"])] if record.get("video_path") else
                      sorted(video_root.glob(record["episode_id"] + "_*view*.mp4")))
        try:
            if not candidates and args.export_missing:
                candidates = [export_inference_video(episode, video_root)]
            if not candidates:
                report["skipped"].append(record["episode_id"])
                continue
            cleanup = finalize_inference_video(episode, candidates[0])
            report["episodes"].append({"episode_id": episode.name,
                                       "video_path": str(candidates[0]), **cleanup})
            print(json.dumps(report["episodes"][-1], ensure_ascii=False), flush=True)
        except Exception as exc:
            report["errors"].append({"episode_id": episode.name,
                                      "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(report["errors"][-1]), flush=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
