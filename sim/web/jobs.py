"""Bounded local subprocess jobs used by the training and inference panels."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def build_train_argv(project_root, config, dataset, out, steps=None,
                     preview_training=False, resume=None):
    argv = [sys.executable, "-m", "sim.act.train", "--config", str(config),
            "--dataset", str(dataset), "--out", str(out)]
    if steps is not None:
        argv.extend(["--steps", str(int(steps))])
    if preview_training:
        argv.append("--preview-training")
    if resume:
        argv.extend(["--resume", str(resume)])
    return argv


def build_inference_argv(project_root, config, seed, model=None, target_count=None,
                         replay_root="runs/workbench_previews", preview=False):
    argv = [sys.executable, "-m", "sim.act.evaluate", "--config", str(config),
            "--seed", str(int(seed))]
    if model:
        argv.extend(["--model", str(model)])
    if target_count is not None:
        argv.extend(["--target-count", str(int(target_count))])
    if preview:
        argv.append("--preview")
    argv.extend(["--record-replay", "--replay-root", str(replay_root)])
    return argv


@dataclass
class JobRecord:
    job_id: str
    kind: str
    argv: list[str]
    state: str = "starting"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    returncode: int | None = None
    output: list[str] = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    pid: int | None = None
    log_path: str | None = None
    log_offset: int = 0
    process: object | None = field(default=None, repr=False)
    log_file: object | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False)

    def snapshot(self) -> dict:
        return {"job_id": self.job_id, "kind": self.kind, "argv": self.argv,
                "state": self.state, "started_at": self.started_at,
                "ended_at": self.ended_at, "returncode": self.returncode,
                "output": self.output[-200:], "progress": dict(self.progress),
                "pid": self.pid, "log_path": self.log_path,
                "log_offset": self.log_offset}


class JobRegistry:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._jobs_root = self.project_root / "runs" / ".workbench_jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _state_path(self, job_id: str) -> Path:
        return self._jobs_root / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self._jobs_root / f"{job_id}.log"

    def _persist_locked(self, job: JobRecord) -> None:
        """Persist a small job record atomically, not its live process object."""
        target = self._state_path(job.job_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(job.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _restore(self) -> None:
        """Restore visible jobs after a workbench restart."""
        records = []
        for path in sorted(self._jobs_root.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda item: float(item.get("started_at", 0.0)))
        for data in records:
            try:
                job = JobRecord(
                    str(data["job_id"]), str(data["kind"]),
                    [str(item) for item in data.get("argv", [])],
                    state=str(data.get("state", "failed")),
                    started_at=float(data.get("started_at", time.time())),
                    ended_at=data.get("ended_at"),
                    returncode=data.get("returncode"),
                    output=[str(item) for item in data.get("output", [])],
                    progress=dict(data.get("progress", {})),
                    pid=data.get("pid"),
                    log_path=data.get("log_path"),
                    log_offset=int(data.get("log_offset", 0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if job.state in {"starting", "running", "stopping"}:
                if self._pid_alive(job.pid):
                    job.state = "running"
                    self._jobs[job.job_id] = job
                    threading.Thread(
                        target=self._watch_detached, args=(job,), daemon=True,
                        name=f"workbench-job-recovered-{job.job_id}",
                    ).start()
                    continue
                job.state = "failed"
                job.ended_at = time.time()
                job.output.append(
                    "workbench restarted after the job process exited; "
                    "see the persisted log for the last output"
                )
            self._jobs[job.job_id] = job
            with self._lock:
                self._persist_locked(job)

    def _active(self, kind: str) -> bool:
        return any(job.kind == kind and job.state in {
            "starting", "running", "stopping"
        }
                   for job in self._jobs.values())

    def start(self, kind: str, argv: list[str]) -> dict:
        with self._lock:
            if self._active(kind):
                raise RuntimeError(f"a {kind} job is already running")
            job = JobRecord(uuid.uuid4().hex[:12], kind, [str(item) for item in argv])
            log_path = self._log_path(job.job_id)
            log_file = log_path.open("a", encoding="utf-8", buffering=1)
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            job.process = subprocess.Popen(
                job.argv, cwd=str(self.project_root), env=environment,
                stdout=log_file, stderr=subprocess.STDOUT,
                start_new_session=True)
            job.pid = int(job.process.pid)
            job.log_path = str(log_path)
            job.log_file = log_file
            job.state = "running"
            self._jobs[job.job_id] = job
            self._persist_locked(job)
        threading.Thread(target=self._watch, args=(job,), daemon=True,
                         name=f"workbench-job-{job.job_id}").start()
        return job.snapshot()

    def start_callable(self, kind: str, runner) -> dict:
        """Run a local Python callable without occupying an HTTP request.

        ``runner`` receives the :class:`JobRecord`, which lets a long-running
        workbench operation publish progress and observe the cancellation
        event.  The existing subprocess path remains unchanged for training
        and inference jobs.
        """
        with self._lock:
            if self._active(kind):
                raise RuntimeError(f"a {kind} job is already running")
            job = JobRecord(uuid.uuid4().hex[:12], kind,
                            ["<background callable>"])
            job.state = "running"
            self._jobs[job.job_id] = job
            self._persist_locked(job)
        threading.Thread(target=self._watch_callable, args=(job, runner),
                         daemon=True,
                         name=f"workbench-job-{job.job_id}").start()
        return job.snapshot()

    def update(self, job_id: str, *, message: str | None = None,
               progress: dict | None = None) -> dict:
        """Publish bounded output/progress from a background callable."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if message:
                job.output.append(str(message))
                del job.output[:-200]
            if progress is not None:
                job.progress = dict(progress)
            self._persist_locked(job)
            return job.snapshot()

    def _watch(self, job: JobRecord):
        self._tail_log(job, wait_for_process=True)
        code = job.process.wait()
        with self._lock:
            job.returncode = int(code)
            job.ended_at = time.time()
            if job.state not in {"stopped", "stopping", "failed"}:
                job.state = "done" if code == 0 else "failed"
            elif job.state == "stopping":
                job.state = "stopped"
            self._persist_locked(job)
        if job.log_file is not None:
            job.log_file.close()

    def _tail_log(self, job: JobRecord, *, wait_for_process: bool) -> None:
        if not job.log_path:
            return
        try:
            stream = open(job.log_path, "r", encoding="utf-8")
        except OSError:
            return
        with stream:
            stream.seek(max(0, int(job.log_offset)))
            while True:
                line = stream.readline()
                if line:
                    with self._lock:
                        job.output.append(line.rstrip())
                        del job.output[:-200]
                        job.log_offset = stream.tell()
                        self._persist_locked(job)
                    continue
                if not wait_for_process:
                    break
                if job.process is not None and job.process.poll() is not None:
                    break
                time.sleep(0.05)

    def _watch_detached(self, job: JobRecord):
        while self._pid_alive(job.pid):
            self._tail_log(job, wait_for_process=False)
            time.sleep(0.05)
        self._tail_log(job, wait_for_process=False)
        with self._lock:
            job.ended_at = time.time()
            job.state = "failed" if any(
                "Traceback" in line or line.startswith(("ERROR", "Error"))
                for line in job.output
            ) else "done"
            self._persist_locked(job)

    def _watch_callable(self, job: JobRecord, runner):
        error = None
        try:
            runner(job)
        except Exception as exc:  # the job endpoint reports the error later
            error = exc
            with self._lock:
                job.output.append(f"{type(exc).__name__}: {exc}")
                del job.output[:-200]
        with self._lock:
            job.returncode = 0 if error is None else 1
            job.ended_at = time.time()
            if job.state == "stopping":
                job.state = "stopped"
            elif job.state != "stopped":
                job.state = "done" if error is None else "failed"
            self._persist_locked(job)

    def list(self) -> list[dict]:
        with self._lock:
            return [job.snapshot() for job in reversed(list(self._jobs.values()))]

    def stop(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state in {"done", "failed", "stopped"}:
                return job.snapshot()
            # Keep the kind occupied until the worker has actually returned;
            # otherwise a second preview can start while MuJoCo is still
            # finishing the cancelled episode.
            job.state = "stopping"
            job.cancel_event.set()
            if job.process is not None:
                job.process.terminate()
            elif self._pid_alive(job.pid):
                os.kill(job.pid, signal.SIGTERM)
            self._persist_locked(job)
        return job.snapshot()
