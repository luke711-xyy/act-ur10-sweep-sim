import numpy as np
import pytest


def test_v5_contract_exposes_fixed_observation_shapes():
    from sim.act.v5 import (
        BEV_CHANNELS,
        BEV_HEIGHT,
        BEV_WIDTH,
        OBJECT_TOKEN_DIM,
        ROBOT_STATE_DIM,
        TASK_STATE_DIM,
    )

    assert (ROBOT_STATE_DIM, TASK_STATE_DIM, OBJECT_TOKEN_DIM) == (36, 6, 29)
    assert (BEV_CHANNELS, BEV_HEIGHT, BEV_WIDTH) == (6, 128, 160)


def test_instance_bev_pack_round_trips_without_changing_slots():
    from sim.act.v5 import pack_instance_bev, unpack_instance_bev

    masks = np.zeros((6, 128, 160), dtype=bool)
    masks[0, 3:10, 20:31] = True
    masks[5, 100:127, 150:160] = True

    packed = pack_instance_bev(masks)
    restored = unpack_instance_bev(packed)

    assert packed.shape == (6, 128, 20)
    np.testing.assert_array_equal(restored, masks)


def test_validate_v5_observation_rejects_truth_only_and_wrong_shapes():
    from sim.act.v5 import validate_v5_observation

    observation = {
        "observation.robot_state": np.zeros(36, dtype=np.float32),
        "observation.task_state": np.zeros(6, dtype=np.float32),
        "observation.object_tokens": np.zeros((6, 29), dtype=np.float32),
        "observation.object_valid": np.ones(6, dtype=bool),
        "observation.instance_bev": np.zeros((6, 128, 160), dtype=bool),
    }
    validate_v5_observation(observation)

    invalid = dict(observation)
    invalid["observation.state"] = np.zeros(42, dtype=np.float32)
    with pytest.raises(ValueError, match="schema v5"):
        validate_v5_observation(invalid)

    invalid = dict(observation)
    invalid["observation.object_tokens"] = np.zeros((6, 28), dtype=np.float32)
    with pytest.raises(ValueError, match="object_tokens"):
        validate_v5_observation(invalid)

    invalid = dict(observation)
    invalid["observation.mujoco_truth"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="truth"):
        validate_v5_observation(invalid)


def test_v5_sidecar_round_trips_predicted_fields_and_optional_labels(tmp_path):
    from sim.act.v5 import read_v5_sidecar, write_v5_sidecar

    tokens = np.zeros((3, 6, 29), dtype=np.float32)
    valid = np.ones((3, 6), dtype=bool)
    masks = np.zeros((3, 6, 128, 160), dtype=bool)
    masks[1, 2, 50:55, 60:67] = True
    task = np.zeros((3, 6), dtype=np.float32)
    labels = np.zeros((3, 6), dtype=bool)
    labels[:, 2] = True
    counts = np.array([0, 1, 1], dtype=np.int16)

    path = tmp_path / "perception_v5.npz"
    write_v5_sidecar(path, tokens, valid, masks, task, labels, counts)
    loaded = read_v5_sidecar(path)

    np.testing.assert_array_equal(loaded["object_tokens"], tokens)
    np.testing.assert_array_equal(loaded["object_valid"], valid)
    np.testing.assert_array_equal(loaded["instance_bev"], masks)
    np.testing.assert_array_equal(loaded["task_state"], task)
    np.testing.assert_array_equal(loaded["selection_target"], labels)
    np.testing.assert_array_equal(loaded["visual_count"], counts)


def test_objectact_dataset_reads_v5_manifest_and_sidecar(tmp_path):
    import json
    from PIL import Image

    from sim.act.object_dataset import ObjectActDataset
    from sim.act.v5 import write_v5_sidecar

    episode = tmp_path / "episode_0001"
    episode.mkdir()
    frame_count = 2
    for camera in ("overhead", "wrist"):
        for frame in range(frame_count):
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                episode / f"{camera}_{frame:05d}.png"
            )
    np.savez_compressed(
        episode / "arrays.npz",
        robot_state=np.zeros((frame_count, 36), dtype=np.float32),
        action=np.zeros((frame_count, 4), dtype=np.float32),
        action_valid=np.ones(frame_count, dtype=bool),
        phase=np.asarray(["approach", "sweep"]),
    )
    write_v5_sidecar(
        episode / "perception_v5.npz",
        np.zeros((frame_count, 6, 29), dtype=np.float32),
        np.ones((frame_count, 6), dtype=bool),
        np.zeros((frame_count, 6, 128, 160), dtype=bool),
        np.zeros((frame_count, 6), dtype=np.float32),
    )
    record = {
        "episode_id": "episode_0001",
        "schema_version": 5,
        "episode_kind": "expert",
        "split": "train",
        "success": True,
        "action_dim": 4,
        "robot_state_dim": 36,
        "overhead": [f"episode_0001/overhead_{i:05d}.png" for i in range(frame_count)],
        "wrist": [f"episode_0001/wrist_{i:05d}.png" for i in range(frame_count)],
        "arrays": "episode_0001/arrays.npz",
        "perception_sidecar": "episode_0001/perception_v5.npz",
        "target_count": 1,
    }
    (tmp_path / "manifest_v5.jsonl").write_text(json.dumps(record) + "\n")

    dataset = ObjectActDataset(str(tmp_path), manifest_name="manifest_v5.jsonl")
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["observation.robot_state"].shape == (36,)
    assert sample["observation.task_state"].shape == (6,)
    assert sample["observation.object_tokens"].shape == (6, 29)
    assert sample["observation.instance_bev"].shape == (6, 128, 160)
    assert sample["observation.bev"].shape == (6, 128, 160)
    assert sample["action"].shape == (25, 4)
    for value in sample.values():
        if isinstance(value, np.ndarray):
            assert value.flags.writeable


def test_validate_v5_manifest_enforces_exact_per_target_counts():
    from sim.act.object_dataset import validate_v5_manifest

    records = [
        {
            "episode_id": f"episode_{target}",
            "schema_version": 5,
            "episode_kind": "expert",
            "success": True,
            "action_dim": 4,
            "robot_state_dim": 36,
            "target_count": target,
        }
        for target in range(1, 7)
    ]
    validate_v5_manifest(records, expected_per_target=1)

    with pytest.raises(ValueError, match="target 3"):
        validate_v5_manifest(records[:-1] + [dict(records[-1], target_count=3)], expected_per_target=1)
