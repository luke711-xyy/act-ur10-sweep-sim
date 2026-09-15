"""Bounded local subprocess jobs used by the training and inference panels."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def build_train_argv(project_root, config, dataset, out, steps=None):
    argv = [sys.executable, "-m", "sim.act.train", "--config", str(config),
            "--dataset", str(dataset), "--out", str(out)]
    if steps is not None:
        argv.extend(["--steps", str(int(steps))])
    return argv


def build_inference_argv(project_root, config, seed, model=None):
    argv = [sys.executable, "-m", "sim.act.evaluate", "--config", str(config),
            "--seed", str(int(seed))]
    if model:
        argv.extend(["--model", str(model)])
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
    process: object | None = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {"job_id": self.job_id, "kind": self.kind, "argv": self.argv,
                "state": self.state, "started_at": self.started_at,
                "ended_at": self.ended_at, "returncode": self.returncode,
                "output": self.output[-200:]}


class JobRegistry:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _active(self, kind: str) -> bool:
        return any(job.kind == kind and job.state in {"starting", "running"}
                   for job in self._jobs.values())

    def start(self, kind: str, argv: list[str]) -> dict:
        with self._lock:
            if self._active(kind):
                raise RuntimeError(f"a {kind} job is already running")
            job = JobRecord(uuid.uuid4().hex[:12], kind, [str(item) for item in argv])
            job.process = subprocess.Popen(
                job.argv, cwd=str(self.project_root), env=os.environ.copy(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            job.state = "running"
            self._jobs[job.job_id] = job
        threading.Thread(target=self._watch, args=(job,), daemon=True,
                         name=f"workbench-job-{job.job_id}").start()
        return job.snapshot()

    def _watch(self, job: JobRecord):
        stream = job.process.stdout
        if stream is not None:
            for line in iter(stream.readline, ""):
                with self._lock:
                    job.output.append(line.rstrip())
                    del job.output[:-200]
        code = job.process.wait()
        with self._lock:
            job.returncode = int(code)
            job.ended_at = time.time()
            if job.state not in {"stopped", "failed"}:
                job.state = "done" if code == 0 else "failed"

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
            job.state = "stopped"
            job.process.terminate()
        return job.snapshot()
