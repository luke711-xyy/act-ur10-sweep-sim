import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def isolate_workbench_job_records(request, monkeypatch):
    """Workbench tests must never recover or persist the user's live jobs."""
    if not request.node.path.name.startswith("test_workbench"):
        return
    from sim.web import jobs

    registry_type = jobs.JobRegistry
    project_root = Path(__file__).resolve().parents[1]
    temporary_root = request.getfixturevalue("tmp_path") / "app-job-registry"

    def isolated_registry(root):
        destination = temporary_root if Path(root).resolve() == project_root else root
        return registry_type(destination)

    monkeypatch.setattr(jobs, "JobRegistry", isolated_registry)
