"""Owned storage, effective wall-clock accounting and recoverable atomic writes.

Callers must use one RunLease for a run and pass an explicit native_lock when
sharing frozen ACT inference. No PID probing, process signals or cleanup occurs.
Interrupted temporary files and crash reservations remain charged, for inspection.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import uuid

from .config import BudgetConfig

GIB = 2**30


class BudgetExceeded(RuntimeError):
    pass


class LeaseBusy(RuntimeError):
    pass


def safe_path(path) -> Path:
    """Reject symlinks at every existing ancestor, including dangling links."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal is forbidden")
    for item in (*reversed(path.parents), path):
        if item.is_symlink():
            raise ValueError(f"symlink path is forbidden: {item}")
    return path


def owned_size(roots) -> int:
    """Conservative logical/allocated size; nested roots and inodes count once."""
    seen = set()
    total = 0
    for root in roots:
        root = safe_path(root)
        paths = [root]
        while paths:
            path = paths.pop()
            if not path.exists() and not path.is_symlink():
                continue
            info = path.lstat()
            inode = (info.st_dev, info.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            if stat.S_ISDIR(info.st_mode):
                paths.extend(path.iterdir())
            elif stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                # Inventory a venv's interpreter links without traversing or
                # charging shared uv targets. Write paths still reject links.
                total += max(info.st_size, info.st_blocks * 512)
            else:
                raise ValueError(f"unsupported owned file: {path}")
    return total


class RunLease:
    """Nonblocking flock, compatible with the frozen native inference flock.

    Lock files are retained. Their existence/content is never liveness evidence.
    The native runtime must receive exactly native_lock, never its input default.
    """
    def __init__(self, path, *, native_lock=None):
        self.paths = [safe_path(path)]
        if native_lock is not None:
            self.paths.append(safe_path(native_lock))
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("run and native lock must be distinct")
        self.handles = []

    def __enter__(self):
        if self.handles:
            raise LeaseBusy("lease is already held")
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(safe_path(path), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    os.close(fd)
                    raise LeaseBusy(f"active lease: {path}") from exc
                self.handles.append(fd)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for fd in reversed(self.handles):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        self.handles.clear()


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _replace_bytes(path, data):
    """Small ledger metadata only; partials intentionally retained on errors."""
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    owned_bytes: int
    free_bytes: int
    reserved_bytes: int
    effective_seconds: float
    pilot_seconds: float

    def require(self):
        if not self.allowed:
            raise BudgetExceeded("; ".join(self.reasons))
        return self


class BudgetLedger:
    """Persistent single-run ledger. Start/pause encloses ALL expensive work.

    Resuming an unpaused crash conservatively charges elapsed time since its
    last durable tick. Paused idle time is excluded. A backwards clock fails
    closed. Reserve remaining mandatory work using expected_bytes/seconds.
    """
    def __init__(self, path, *, owned_roots, config: BudgetConfig,
                 disk_probe=shutil.disk_usage, clock=time.time):
        self.path = safe_path(path)
        candidates = sorted({safe_path(p) for p in owned_roots}, key=lambda p: (len(p.parts), str(p)))
        self.roots = tuple(p for p in candidates if not any(q != p and q in p.parents for q in candidates))
        if not self.roots:
            raise ValueError("owned roots required")
        self.config, self.disk_probe, self.clock = config, disk_probe, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._state() as state:
            pass

    @contextmanager
    def _state(self):
        with RunLease(self.path.with_name(self.path.name + ".lock")):
            policy = {"budget": asdict(self.config), "roots": [str(p) for p in self.roots]}
            if self.path.exists():
                state = json.loads(self.path.read_bytes())
                stored = state.get("policy", {})
                if state.get("schema") != 1 or stored.get("budget") != policy["budget"] or sorted(stored.get("roots", [])) != sorted(policy["roots"]):
                    raise ValueError("ledger policy/owned roots changed")
                state["policy"] = policy
            else:
                state = dict(schema=1, policy=policy, effective_seconds=0., pilot_seconds=0.,
                             active=None, reservations={}, events=[])
            now = _number(self.clock(), "clock")
            if state["active"] is not None:
                delta = now - state["active"]["tick"]
                if delta < 0:
                    raise ValueError("clock moved backwards")
                state["effective_seconds"] += delta
                if state["active"]["pilot"]:
                    state["pilot_seconds"] += delta
                state["active"]["tick"] = now
            try:
                yield state
            finally:
                _replace_bytes(self.path, json.dumps(state, sort_keys=True, allow_nan=False).encode())

    def ensure_owned(self, path):
        path = safe_path(path)
        if not any(path == root or root in path.parents for root in self.roots):
            raise ValueError("write destination is outside owned roots")
        return path

    def _decision(self, state, expected_bytes=0, seconds=0, pilot=False):
        _number(expected_bytes, "expected bytes")
        _number(seconds, "seconds")
        size = owned_size(self.roots)
        reserved = sum(r["bytes"] for r in state["reservations"].values())
        reserved_seconds = sum(r["seconds"] for r in state["reservations"].values())
        # Probe every owned volume. Other processes' usage is reflected live.
        free = min(self.disk_probe(next(p for p in (root, *root.parents) if p.exists())).free for root in self.roots)
        c = self.config
        peak = size + reserved + expected_bytes
        remaining = free - reserved - expected_bytes
        elapsed = state["effective_seconds"]
        pilot_elapsed = state["pilot_seconds"]
        pilot = pilot or (state["active"] is not None and state["active"]["pilot"])
        reasons, warnings = [], []
        if peak > c.artifact_limit_gib * GIB:
            reasons.append("45 GiB owned artifact peak exceeded")
        if remaining < c.disk_min_free_gib * GIB:
            reasons.append("25 GiB free disk floor breached")
        if elapsed >= c.total_hours * 3600 or elapsed + reserved_seconds + seconds > c.total_hours * 3600:
            reasons.append("72 hour effective compute budget exhausted")
        pilot_reserved = sum(r["seconds"] for r in state["reservations"].values() if r["pilot"])
        if pilot and (pilot_elapsed >= c.throughput_pilot_hours * 3600 or pilot_elapsed + pilot_reserved + seconds > c.throughput_pilot_hours * 3600):
            reasons.append("2 hour throughput pilot budget exhausted")
        if peak >= c.artifact_warning_gib * GIB:
            warnings.append("40 GiB artifact warning")
        if remaining < c.disk_warning_free_gib * GIB:
            warnings.append("below 30 GiB free disk warning")
        return Admission(not reasons, tuple(reasons), tuple(warnings), size, free, reserved, elapsed, pilot_elapsed)

    def admit(self, expected_bytes=0, *, seconds=0, pilot=False):
        with self._state() as state:
            return self._decision(state, expected_bytes, seconds, pilot)

    def poll(self):
        return self.admit()

    @contextmanager
    def reserve(self, expected_bytes, *, seconds=0, pilot=False):
        token = uuid.uuid4().hex
        with self._state() as state:
            self._decision(state, expected_bytes, seconds, pilot).require()
            state["reservations"][token] = dict(bytes=expected_bytes, seconds=seconds, pilot=pilot)
        try:
            yield token
        finally:
            with self._state() as state:
                state["reservations"].pop(token)

    def record(self, event, *, seconds=0, kind=None, pilot=False, reservation=None):
        _number(seconds, "seconds")
        with self._state() as state:
            if event == "start":
                if state["active"] is not None or not kind:
                    raise ValueError("activity already active or kind missing")
                self._decision(state, pilot=pilot).require()
                state["active"] = dict(tick=self.clock(), kind=kind, pilot=pilot)
            elif event == "pause":
                state["active"] = None
            elif event == "prior":
                if not kind:
                    raise ValueError("prior cost kind required")
                state["effective_seconds"] += seconds
                state["pilot_seconds"] += seconds if pilot else 0
            elif event == "release":
                # Explicit reconciliation after acquiring the run lease. Never
                # infer crash/liveness from lock text; retained files stay owned.
                if not kind or reservation not in state["reservations"]:
                    raise ValueError("known orphan reservation and reconciliation reason required")
                state["reservations"].pop(reservation)
            else:
                raise ValueError("unknown ledger event")
            state["events"].append(dict(event=event, seconds=seconds, kind=kind, pilot=pilot, reservation=reservation))

    @contextmanager
    def activity(self, kind, *, seconds=0, pilot=False):
        self.admit(seconds=seconds, pilot=pilot).require()
        self.record("start", kind=kind, pilot=pilot)
        try:
            yield
        finally:
            self.record("pause")


def atomic_checkpoint(path, payload, budget: BudgetLedger, *, expected_bytes=None):
    """Publish complete bytes only. Reserve old + new + temporary copy peak.

    Bytes or a byte-chunk iterator is accepted. Iterators require a bound before
    any iteration/write. Failure retains both old checkpoint and partial evidence.
    """
    path = budget.ensure_owned(path)
    if isinstance(payload, bytes):
        expected_bytes = len(payload) if expected_bytes is None else expected_bytes
        chunks = (payload,)
    else:
        chunks = payload
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError("expected_bytes bound required")
    with budget.reserve(2 * expected_bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
        with os.fdopen(fd, "wb") as stream:
            written = 0
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("checkpoint chunks must be bytes")
                written += len(chunk)
                if written > expected_bytes:
                    raise BudgetExceeded("checkpoint exceeds reserved size")
                # Include external disk changes before each real write.
                if budget.disk_probe(path.parent).free - len(chunk) < budget.config.disk_min_free_gib * GIB:
                    raise BudgetExceeded("disk floor changed during checkpoint")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, safe_path(path))
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return path
