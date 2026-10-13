"""Manual and batch CLI inference share a process-level lease, not training."""

import importlib.util
import subprocess
import sys
import time


def test_inference_lease_serializes_subprocesses_and_releases(tmp_path):
    assert importlib.util.find_spec("sim.act.inference_lock") is not None
    path = tmp_path / "inference.lock"
    script = ("from sim.act.inference_lock import inference_lease; import time; "
              f"\nwith inference_lease({str(path)!r}, on_wait=lambda: None):\n"
              " print('acquired', flush=True)\n time.sleep(0.4)\n")
    first = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    second = None
    try:
        assert first.stdout.readline().strip() == "acquired"
        second = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        time.sleep(0.08)
        assert second.poll() is None
        assert first.wait(timeout=5) == 0
        assert second.communicate(timeout=5)[0].strip() == "acquired"
        assert second.returncode == 0
    finally:
        for process in (first, second):
            if process and process.poll() is None:
                process.kill()
                process.wait()
