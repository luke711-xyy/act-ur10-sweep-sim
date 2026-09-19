import json

import pytest


def test_objectact_training_plan_has_the_approved_layout_composition():
    from sim.act.generate_objectact_dataset import objectact_training_episode_plan

    plan = objectact_training_episode_plan(4100)
    assert len(plan) == 120
    assert {item.target_count for item in plan} == set(range(1, 7))
    for target in range(1, 7):
        assert sum(item.target_count == target for item in plan) == 20
    paired = [item for item in plan if item.layout_kind == "paired"]
    independent = [item for item in plan if item.layout_kind == "independent"]
    assert len(paired) == 48
    assert len({item.layout_id for item in paired}) == 8
    assert len(independent) == 72
    assert all(
        sum(item.target_count == target for item in independent) == 12
        for target in range(1, 7)
    )


def test_validate_objectact_training_manifest_requires_complete_layout_slots(tmp_path):
    from sim.act.generate_objectact_dataset import validate_objectact_training_manifest

    records = []
    for target in range(1, 7):
        for index in range(20):
            paired = index < 8
            layout_id = f"paired_{index:03d}" if paired else f"independent_n{target}_{index - 8:03d}"
            records.append({
                "episode_id": f"episode_{target}_{index}",
                "schema_version": 5,
                "episode_kind": "expert",
                "split": "train",
                "success": True,
                "action_dim": 4,
                "robot_state_dim": 36,
                "target_count": target,
                "layout_id": layout_id,
                "layout_kind": "paired" if paired else "independent",
                "seed": 4100 + index,
            })
    (tmp_path / "manifest_v5.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = validate_objectact_training_manifest(tmp_path)
    assert summary["total"] == 120
    assert summary["per_target"] == {str(target): 20 for target in range(1, 7)}

    records[-1]["target_indices"] = [0]
    (tmp_path / "manifest_v5.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_indices"):
        validate_objectact_training_manifest(tmp_path)


def test_detector_device_auto_prefers_mps_when_available(monkeypatch):
    import torch

    from sim.act.generate_objectact_dataset import resolve_detector_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_detector_device("auto") == "mps"
    assert resolve_detector_device("cpu") == "cpu"


def test_projection_rejection_filter_only_accepts_expected_geometry_failures():
    from sim.act.object_interface import is_expected_projection_rejection

    assert is_expected_projection_rejection(
        ValueError("instance mask projects outside the fixed table BEV")
    )
    assert is_expected_projection_rejection(
        ValueError("instance mask has no valid intersection with the table plane")
    )
    assert not is_expected_projection_rejection(ValueError("bad class probabilities"))
