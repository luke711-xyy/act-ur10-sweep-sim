"""Incremental, read-only training telemetry. Never imports a policy or Torch."""

from collections import deque
import json
import math
import os
from pathlib import Path
from threading import RLock
import time


ACTIVE = {"starting", "running", "stopping"}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _option(argv, key, default=None):
    for i, value in enumerate(argv):
        if value == key and i + 1 < len(argv):
            return argv[i + 1]
        if value.startswith(key + "="):
            return value[len(key) + 1:]
    return default


def _metadata(path):
    try:
        if path.stat().st_size > 256 * 1024:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class _Series:
    def __init__(self):
        self.points = []
        self.recent = deque(maxlen=101)
        self.smooth = deque(maxlen=50)

    def add(self, step, loss, seconds):
        if self.points and step == self.points[-1]["step"]:
            return
        if self.points and step < self.points[-1]["step"]:
            self.__init__()  # a restarted run must not connect to its old future
        self.smooth.append(loss)
        self.points.append({"step": step, "loss": loss,
                            "smooth": sum(self.smooth) / len(self.smooth)})
        self.recent.append((step, loss, seconds))

    def plot(self, limit=1200):
        if len(self.points) <= limit:
            return list(self.points)
        # Min/max buckets preserve spikes, endpoints and real optimizer steps.
        middle = self.points[1:-1]
        width = math.ceil(len(middle) / ((limit - 2) // 2))
        result = [self.points[0]]
        for i in range(0, len(middle), width):
            bucket = middle[i:i + width]
            low = min(bucket, key=lambda p: p["loss"])
            high = max(bucket, key=lambda p: p["loss"])
            result.extend(sorted([low] if low is high else [low, high],
                                 key=lambda p: p["step"]))
        return result + [self.points[-1]]


class _Tail:
    def __init__(self, path):
        self.path = path
        self.inode = None
        self.offset = 0
        self.stages = {}
        self.invalid_rows = 0
        self.updated_at = None

    def read(self):
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.inode or stat.st_size < self.offset:
                    self.__init__(self.path)
                    self.inode = identity
                stream.seek(self.offset)
                for line in stream:
                    if not line.endswith(b"\n"):
                        break  # retry this exact byte offset after the writer finishes
                    self.offset += len(line)
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(row, dict) or "loss" not in row:
                        continue
                    step, loss = _number(row.get("step")), _number(row.get("loss"))
                    if step is None or step < 0 or int(step) != step or loss is None:
                        self.invalid_rows += 1
                        continue
                    seconds = _number(row.get("budget_seconds", row.get("elapsed_s")))
                    stage = row.get("stage", "train")
                    if stage not in {"full", "pilot", "train"}:
                        continue
                    self.stages.setdefault(stage, _Series()).add(int(step), loss, seconds)
                    self.updated_at = stat.st_mtime
        except FileNotFoundError:
            pass


class TrainingProgress:
    def __init__(self, project_root, *, stale_after=30):
        self.root = Path(project_root).resolve()
        self.stale_after = float(stale_after)
        if not math.isfinite(self.stale_after) or self.stale_after <= 0:
            raise ValueError("training freshness window must be positive and finite")
        self._tails = {}
        self._lock = RLock()

    def _local(self, path):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("training telemetry path is outside this workspace")
        return path

    def snapshot(self, jobs, *, now=None):
        now = time.time() if now is None else now
        candidates = [job for job in jobs if job.get("kind") == "train"]
        if not candidates:
            return {"state": "idle", "freshness": "empty", "points": []}
        job = max(candidates, key=lambda j: (j.get("state") in ACTIVE,
                                             j.get("started_at", 0)))
        argv = job.get("argv", [])
        diffusion = any(item.startswith("sim.diffusion.") for item in argv)
        policy = "diffusion" if diffusion else "act"
        out = self._local(_option(argv, "--out", "runs/diffusion_policy_mvp" if diffusion
                                 else "runs/act_model"))
        path = self._local(out / "curves.jsonl" if diffusion else
                           job.get("log_path") or f"runs/.workbench_jobs/{job['job_id']}.log")
        state = job.get("state", "unknown")
        pid = job.get("pid")
        if state in ACTIVE and pid:
            try:
                os.kill(int(pid), 0)
            except (OSError, ValueError, OverflowError):
                state = "exited"
        with self._lock:
            key = (job["job_id"], str(path))
            if key not in self._tails:
                if len(self._tails) >= 4:
                    self._tails.pop(next(iter(self._tails)))
                self._tails[key] = _Tail(path)
            tail = self._tails[key]
            tail.read()
            status = _metadata(self._local(out / "status.json")) if diffusion else {}
            config = _metadata(self._local(out / "training_config.json")) if diffusion else {}
            stage = status.get("stage")
            if stage not in tail.stages:
                stage = "full" if "full" in tail.stages else next(iter(tail.stages), "train")
            series = tail.stages.get(stage)
            age = max(0, now - tail.updated_at) if tail.updated_at is not None else None
            finished = status.get("state") in {"training_complete", "failed", "stopped", "pilot_stopped"}
            freshness = ("stopped" if state not in ACTIVE or finished else
                         "waiting" if age is None else "stale" if age > self.stale_after else "live")
            max_steps = _number(config.get("max_steps"))
            try:
                max_steps = int(_option(argv, "--steps", max_steps))
            except (TypeError, ValueError):
                max_steps = None
            result = dict(job_id=job["job_id"], pid=pid, policy_type=policy, state=state,
                          backbone=config.get("backbone","unet") if diffusion else None,
                          stage=stage, training_state=status.get("state"),
                          out_dir=str(out.relative_to(self.root)), max_steps=max_steps,
                          freshness=freshness, updated_at=tail.updated_at, age_seconds=age,
                          step=None, loss=None, mean_loss=None, mean_samples=0,
                          seconds_per_update=None, updates_per_minute=None,
                          rate_updates=0, invalid_rows=tail.invalid_rows, points=[],
                          smooth_window=50, rate_window=100)
            if series and series.points:
                recent = list(series.recent)
                last = recent[-1]
                losses = [row[1] for row in recent[-100:]]
                result.update(step=last[0], loss=last[1], mean_loss=sum(losses) / len(losses),
                              mean_samples=len(losses), points=series.plot())
                timed = [row for row in recent if row[2] is not None]
                if freshness == "live" and len(timed) > 1:
                    steps = timed[-1][0] - timed[0][0]
                    seconds = timed[-1][2] - timed[0][2]
                    if steps > 0 and seconds > 0:
                        result.update(seconds_per_update=seconds / steps,
                                      updates_per_minute=60 * steps / seconds,
                                      rate_updates=steps)
            return result
