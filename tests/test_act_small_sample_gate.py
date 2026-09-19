from types import SimpleNamespace
import json

import numpy as np
import pytest

from sim.act.small_sample_gate import summarize_small_sample_gate


def _records():
    return {
        target: {
            "target_count": target,
            "first_contact_position": [0.30 + target * 0.001, 0.01, 0.0],
        }
        for target in range(1, 7)
    }


def _results(successes=5, error=0.02, peak_force=10.0):
    return {
        target: SimpleNamespace(
            success=target <= successes,
            failure_reason="" if target <= successes else "exact count missed",
            sampled_frames=400,
            actions=np.zeros((400, 4), dtype=np.float32),
            peak_force=peak_force,
            first_contact_position=np.array([
                0.30 + target * 0.001 + error, 0.01, 0.0
            ]),
        )
        for target in range(1, 7)
    }


def test_small_sample_gate_accepts_five_of_six_with_bounded_contact_and_force():
    summary = summarize_small_sample_gate(_records(), _results())

    assert summary["passed"] is True
    assert summary["successes"] == 5
    assert summary["median_contact_xy_error_m"] == pytest.approx(0.02)


def test_small_sample_gate_rejects_force_or_contact_regression():
    force = summarize_small_sample_gate(
        _records(), _results(successes=6, peak_force=20.1)
    )
    contact = summarize_small_sample_gate(
        _records(), _results(successes=6, error=0.031)
    )

    assert force["passed"] is False
    assert force["force_ok"] is False
    assert contact["passed"] is False


def test_small_sample_gate_reads_the_v5_paired_layout_manifest(tmp_path):
    records = [
        {
            "target_count": target,
            "layout_id": "paired_000",
            "episode_kind": "expert",
            "split": "train",
            "success": True,
            "first_contact_position": [0.3, 0.0, 0.0],
        }
        for target in range(1, 7)
    ]
    (tmp_path / "manifest_v5.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    from sim.act.small_sample_gate import _paired_references

    assert set(_paired_references(tmp_path)) == set(range(1, 7))
