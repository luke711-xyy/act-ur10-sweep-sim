"""Serialize manual and batch inference without locking the trainer."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def inference_lease(path=None, *, on_wait=None):
    path = Path(path) if path is not None else (
        Path(__file__).resolve().parents[2] / "runs" / ".act_inference.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait is not None:
                on_wait()
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
