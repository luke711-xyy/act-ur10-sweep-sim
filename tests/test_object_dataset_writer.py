import json

import numpy as np


def _observation(frame: int):
    token = np.zeros((6, 29), dtype=np.float32)
    token[0, 3:5] = [0.30 + 0.001 * frame, 0.0]
    masks = np.zeros((6, 128, 160), dtype=bool)
    masks[0, 64, 80 + frame] = True
    return {
        "overhead": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "inspection": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot_state": np.zeros(36, dtype=np.float32),
        "task_state": np.array([1 / 6, 2 / 6, 0.0, 1 / 6, 2 / 6, 0.0], dtype=np.float32),
        "object_tokens": token,
        "object_valid": np.array([True, False, False, False, False, False]),
        "instance_bev": masks,
        "action_delta": np.zeros(4, dtype=np.float32),
        "policy_mask": True,
        "phase": "sweep",
        "t": frame / 25.0,
    }


def test_objectact_writer_round_trips_rgb_sidecar_and_arrays(tmp_path):
    from sim.act.object_dataset import ObjectActDataset, ObjectActDatasetWriter

    writer = ObjectActDatasetWriter(str(tmp_path))
    writer.add_episode(
        "expert_v5_n2_s1",
        [_observation(0), _observation(1)],
        np.array([[True, False, False, False, False, False]] * 2),
        True,
        {"split": "train", "target_count": 2},
    )
    record = json.loads((tmp_path / "manifest_v5.jsonl").read_text().splitlines()[0])
    assert record["schema_version"] == 5
    assert record["perception_source"] == "rgb_detector"
    dataset = ObjectActDataset(str(tmp_path))
    sample = dataset[0]
    assert sample["observation.robot_state"].shape == (36,)
    assert sample["observation.object_tokens"].shape == (6, 29)
    assert sample["observation.bev"].shape == (6, 128, 160)
    assert sample["selection_target"].shape == (6,)


def test_selection_targets_are_offline_labels_from_final_collected_parts():
    from types import SimpleNamespace

    from sim.act.generate_objectact_dataset import _selection_targets

    legacy = _observation(0)
    legacy["object_pose"] = np.array([
        [0.30, 0.0, 0.01, 0, 0, 0, 1],
        [0.36, 0.0, 0.01, 0, 0, 0, 1],
    ], dtype=np.float32)
    v5 = _observation(0)
    v5["object_tokens"][0, 3:5] = [0.30, 0.0]
    v5["object_tokens"][1, 3:5] = [0.36, 0.0]
    v5["object_valid"][:2] = True
    result = SimpleNamespace(
        observations=[legacy],
        object_observations=[v5],
        final_collected_mask=np.array([True, False]),
    )
    labels = _selection_targets(result)
    assert labels.shape == (1, 6)
    assert labels[0, 0]
    assert not labels[0, 1]
