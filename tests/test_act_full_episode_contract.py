import json
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from sim.act.collect_dataset import (
    evaluation_layout_plan,
    promote_preview_batch,
    promote_success_preview_set,
    preview_batch_plan,
    training_episode_plan,
    validate_preview_training_manifest,
    validate_training_manifest,
)
from sim.act.dataset import ActDataset, ActDatasetWriter
from sim.act.evaluate import _policy_reference_substep
from sim.act.interface import ACTObservationBuilder, action_deltas_to_absolute
from sim.act.rollout import (
    brush_delivery_reached,
    collection_goal_status,
    collection_step_status,
    CollectionTerminationTracker,
    policy_action_delta,
)
from sim.config import load_config


class _FakeEndEffector:
    def joint_state(self):
        return np.arange(6, dtype=np.float32)

    def tcp_yaw(self):
        return 0.25


class _FakeEnv:
    ee = _FakeEndEffector()
    layout = [object()] * 6
    time = 1.25

    def tcp(self):
        return np.array([0.42, -0.03, 0.08], dtype=np.float32)

    def wrench(self):
        return np.arange(6, dtype=np.float32) + 10.0

    def collected_mask(self):
        return np.array([True, False, False, False, False, False])

    def render_rgb(self, _camera, size):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def render_wrist_rgb(self, size):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def _write_v4_episode(root, episode_id="expert_1", *, episode_kind="expert",
                      success=True, target_count=1, phases=None,
                      split="train", layout_id=""):
    phases = phases or ["approach", "descent", "contact_build", "sweep"]
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observations = []
    for index, phase in enumerate(phases):
        observations.append({
            "overhead": image,
            "wrist": image,
            "state": np.full(42, index, dtype=np.float32),
            "environment_state": np.array([6, target_count, 0], dtype=np.float32),
            "phase": phase,
            "policy_mask": phase in {"approach", "descent", "contact_build", "sweep"},
            "contact_latched": phase in {"contact_build", "sweep"},
            "policy_z": 0.08 - index * 0.01,
            "applied_z": 0.08 - index * 0.01,
            "z_owner": "admittance" if phase in {"contact_build", "sweep"} else "policy",
        })
    actions = np.zeros((len(observations), 4), dtype=np.float32)
    actions[:, 0] = -0.01
    actions[:2, 2] = -0.01
    ActDatasetWriter(str(root)).add_episode(
        episode_id,
        observations,
        actions,
        success,
        {
            "episode_kind": episode_kind,
            "split": split,
            "target_count": target_count,
            "total_count": 6,
            "layout_id": layout_id,
            "layout_kind": "paired" if layout_id.startswith("paired_") else "independent",
            "seed": 4100,
        },
    )


def test_full_episode_action_contract_accumulates_xyz_and_yaw():
    start = np.array([0.40, -0.10, 0.08, 0.20], dtype=np.float32)
    deltas = np.array([
        [0.01, 0.02, -0.03, 0.10],
        [-0.02, 0.00, -0.01, -0.20],
    ], dtype=np.float32)

    absolute = action_deltas_to_absolute(start, deltas)

    assert absolute.shape == (2, 4)
    np.testing.assert_allclose(absolute[0], [0.41, -0.08, 0.05, 0.30])
    np.testing.assert_allclose(absolute[1], [0.39, -0.08, 0.04, 0.10])


def test_observation_state_is_42d_and_keeps_current_and_previous_contact_latch():
    cfg = load_config(overrides=["act.image_size=[8,8]"])
    builder = ACTObservationBuilder(cfg)
    env = _FakeEnv()

    first = builder.observe(env, contact_latched=False)
    second = builder.observe(env, contact_latched=True)

    assert builder.spec.action_dim == 4
    assert builder.spec.state_dim == 42
    assert first["observation.state"].shape == (42,)
    assert first["observation.state"][20] == 0.0
    assert first["observation.state"][41] == 0.0
    assert second["observation.state"][20] == 1.0
    assert second["observation.state"][41] == 0.0

    batch = builder.torch_batch(second)
    assert "contact_latched" not in batch
    assert "t" not in batch
    assert batch["observation.state"].shape == (42,)


def test_policy_substep_executes_z_before_contact_and_ignores_it_after_contact():
    start = np.array([0.40, 0.0, 0.08, 0.0], dtype=np.float32)
    target = np.array([0.50, 0.10, -0.02, 1.0], dtype=np.float32)

    airborne = _policy_reference_substep(
        start, target, substep=4, substeps=4, dt=0.01,
        max_xy_speed=0.25, max_z_speed=0.05, max_yaw_rate=0.5,
        contact_latched=False,
    )
    contact = _policy_reference_substep(
        start, target, substep=4, substeps=4, dt=0.01,
        max_xy_speed=0.25, max_z_speed=0.05, max_yaw_rate=0.5,
        contact_latched=True, admittance_z=0.012,
    )

    assert np.linalg.norm(airborne[:2] - start[:2]) <= 0.0100001
    assert airborne[2] == pytest.approx(0.078)
    assert airborne[3] == pytest.approx(0.02)
    assert contact[2] == pytest.approx(0.012)


def test_expert_action_delta_zeros_z_only_after_contact_latch():
    previous = np.array([0.40, 0.0, 0.08, 0.10], dtype=np.float32)
    requested = np.array([0.39, 0.01, 0.04, 0.30], dtype=np.float32)

    airborne = policy_action_delta(requested, previous, contact_latched=False)
    contact = policy_action_delta(requested, previous, contact_latched=True)

    np.testing.assert_allclose(airborne, [-0.01, 0.01, -0.04, 0.20], atol=1e-7)
    np.testing.assert_allclose(contact, [-0.01, 0.01, 0.0, 0.20], atol=1e-7)


def test_exact_goal_uses_any_n_identities_but_rejects_partial_tray_overlap():
    collected = np.array([False, True, False, True, False, True])

    success, reason = collection_goal_status(
        collected, target_indices=np.array([0, 2, 4]), target_count=3,
        partial_overlap_mask=np.zeros(6, dtype=bool),
    )
    partial, partial_reason = collection_goal_status(
        collected, target_indices=np.array([0, 2, 4]), target_count=3,
        partial_overlap_mask=np.array([True, False, False, False, False, False]),
    )

    assert success is True and reason == ""
    assert partial is False
    assert partial_reason == "component projection only partially overlaps the collection region"


def test_shared_collection_termination_contract_matches_exact_and_partial_cases():
    exact, exact_failure, exact_reason = collection_step_status(
        np.array([True, False, True, False]),
        target_indices=None,
        target_count=2,
        partial_overlap_mask=np.zeros(4, dtype=bool),
    )
    partial, partial_failure, partial_reason = collection_step_status(
        np.array([True, False, True, False]),
        target_indices=None,
        target_count=2,
        partial_overlap_mask=np.array([False, True, False, False]),
    )

    assert exact is True and exact_failure == "" and exact_reason == ""
    assert partial is False
    assert partial_failure == partial_reason == (
        "component projection only partially overlaps the collection region"
    )


def test_delivery_depth_gate_requires_brush_inside_and_safe_tray_depth(monkeypatch):
    import sim.act.rollout as rollout

    class FakeEnv:
        def __init__(self, x, inside):
            self._x = x
            self._inside = inside

        def tcp(self):
            return np.array([self._x, 0.0, 0.02], dtype=np.float32)

        def brush_fully_inside_target(self):
            return self._inside

    monkeypatch.setattr(rollout, "_tray_exit_x", lambda _cfg: 0.20)
    assert brush_delivery_reached(FakeEnv(0.19, True), object()) is True
    assert brush_delivery_reached(FakeEnv(0.19, False), object()) is False
    assert brush_delivery_reached(FakeEnv(0.25, True), object()) is False


def test_inference_collection_tracker_keeps_act_running_during_target_hold():
    tracker = CollectionTerminationTracker(
        target_count=3, target_hold_seconds=2.0, stall_timeout_seconds=3.0,
    )
    empty = np.zeros(6, dtype=bool)
    one = np.array([True, False, False, False, False, False])
    target = np.array([True, True, True, False, False, False])

    assert tracker.update(empty, 0.0) == ""
    assert tracker.update(one, 1.0) == ""
    assert tracker.update(target, 2.0) == ""
    assert tracker.update(target, 3.9) == ""
    assert tracker.update(target, 4.0) == "target count hold elapsed"


def test_inference_collection_tracker_stops_after_three_seconds_without_new_entry():
    tracker = CollectionTerminationTracker(
        target_count=4, target_hold_seconds=2.0, stall_timeout_seconds=3.0,
    )
    one = np.array([True, False, False, False, False, False])

    assert tracker.update(one, 5.0) == ""
    assert tracker.update(one, 7.99) == ""
    assert tracker.update(one, 8.0) == "collection stalled for 3 seconds"


def test_expert_and_act_inference_reference_shared_termination_helpers():
    import inspect
    import sim.act.evaluate as evaluate
    import sim.act.rollout as rollout

    assert "collection_step_status" in inspect.getsource(rollout.run_action_path)
    assert "brush_delivery_reached" in inspect.getsource(rollout.run_action_path)
    assert "collection_step_status" in inspect.getsource(evaluate.run_act_episode)
    assert "CollectionTerminationTracker" in inspect.getsource(evaluate.run_act_episode)
    assert "brush_delivery_reached" not in inspect.getsource(evaluate.run_act_episode)


def test_act_inference_module_has_no_astar_or_layout_dependent_staging_dependency():
    import sim.act.evaluate as evaluate

    source = inspect.getsource(evaluate)
    assert "expert_staging_xy" not in source
    assert "_act_staging_xy" not in source
    assert "contact_home_xy" not in source


def test_training_reader_and_act_rollout_work_when_astar_entrypoints_raise(
        tmp_path, monkeypatch):
    import torch
    import sim.act.evaluate as evaluate
    import sim.act.expert as expert

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A* must not be called by training or ACT inference")

    for name in ("plan_expert_sweep", "expert_plan_candidates", "contact_home_xy"):
        monkeypatch.setattr(expert, name, forbidden)

    dataset_root = tmp_path / "dataset"
    _write_v4_episode(dataset_root)
    assert len(ActDataset(str(dataset_root), chunk_size=2)) > 0

    class FakePolicy:
        def eval(self):
            return self

        def reset(self):
            return None

        def predict_action_chunk(self, _batch):
            return torch.zeros((1, 25, 4), dtype=torch.float32)

    class FakePolicyConfig:
        output_features = {"action": SimpleNamespace(shape=(4,))}

    class FakeEndEffector(_FakeEndEffector):
        pass

    class FakeSweepEnv(_FakeEnv):
        def __init__(self, cfg, seed=0):
            self.cfg = cfg
            self.seed = seed
            self.time = 0.0
            self.layout = [object()] * 6
            self.ee = FakeEndEffector()
            self._tcp = np.array([0.42, 0.0, 0.10], dtype=float)
            self.control_dt = 0.01

        def reset(self, seed=None):
            return None

        def tcp(self):
            return self._tcp.copy()

        def normal_force(self):
            return 0.0

        def step_control(self, command):
            self._tcp[:] = [command.x, command.y, command.z]
            self.time += self.control_dt

        def brush_fully_inside_target(self):
            return False

        def partial_collection_overlap_mask(self):
            return np.zeros(6, dtype=bool)

        def unsafe_robot_collision(self):
            return False

        def lost_mask(self):
            return np.zeros(6, dtype=bool)

        def close(self):
            return None

    monkeypatch.setattr(evaluate, "SweepEnv", FakeSweepEnv)
    monkeypatch.setattr(
        evaluate, "build_act_policy",
        lambda cfg, pretrained_path=None: (FakePolicy(), FakePolicyConfig()),
    )
    monkeypatch.setattr(
        evaluate, "build_act_processors",
        lambda *args, **kwargs: (lambda value: value, lambda value: value),
    )
    cfg = load_config(overrides=[
        "sim.real_time=false", "episode.max_frames=1",
        f"act.dataset_dir={tmp_path / 'missing'}",
        "act.image_size=[8,8]",
    ])

    result = evaluate.run_act_episode(cfg, seed=7, model_path=None, preview=True)

    assert result.failure_reason == "episode frame limit reached"
    assert result.scheduler_queries >= 1
    assert result.first_contact_position is None


def test_checkpoint_inference_does_not_rescan_training_images(tmp_path, monkeypatch):
    import sim.act.evaluate as evaluate

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = load_config(overrides=[f"act.dataset_dir={dataset_root}"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("checkpoint inference must use saved processor statistics")

    monkeypatch.setattr(evaluate, "ActDataset", forbidden)

    assert evaluate._inference_dataset_stats(cfg, "checkpoint") is None


def test_schema_v4_training_reader_accepts_only_successful_4d_experts(tmp_path):
    _write_v4_episode(tmp_path, "expert_success")
    _write_v4_episode(tmp_path, "inference_success", episode_kind="inference")
    _write_v4_episode(tmp_path, "expert_failure", success=False)

    dataset = ActDataset(str(tmp_path), chunk_size=2)

    assert [record["episode_id"] for record in dataset.records] == ["expert_success"]
    assert dataset.records[0]["schema_version"] == 4
    assert dataset.records[0]["action_dim"] == 4
    assert dataset[0]["observation.state"].shape == (42,)
    assert dataset[0]["action"].shape == (2, 4)


def test_phase_sampling_weights_allocate_35_percent_to_approach_and_descent(tmp_path):
    _write_v4_episode(
        tmp_path,
        phases=["approach", "descent", "contact_build", "sweep", "sweep", "sweep"],
    )
    dataset = ActDataset(str(tmp_path), chunk_size=2)

    weights = dataset.phase_sampling_weights(approach_fraction=0.35)
    phase_weights = {}
    for weight, (_, record, frame_index) in zip(weights, dataset.index):
        arrays = np.load(tmp_path / record["arrays"], allow_pickle=False)
        phase = str(arrays["phase"][frame_index])
        phase_weights[phase] = phase_weights.get(phase, 0.0) + float(weight)

    assert phase_weights["approach"] + phase_weights["descent"] == pytest.approx(0.35)
    assert phase_weights["contact_build"] + phase_weights["sweep"] == pytest.approx(0.65)


def test_dataset_plans_have_exact_paired_independent_and_split_counts():
    preview = preview_batch_plan(4100)
    train = training_episode_plan(4100)
    validation = evaluation_layout_plan("val", 60000, layouts_per_target=10)
    test = evaluation_layout_plan("test", 90000, layouts_per_target=20)

    assert [(item.target_count, item.seed) for item in preview] == [
        (1, 4100), (2, 4100), (3, 4100),
        (4, 4100), (5, 4100), (6, 4100),
    ]
    assert len(train) == 120
    assert sum(item.layout_kind == "paired" for item in train) == 48
    assert sum(item.layout_kind == "independent" for item in train) == 72
    for target_count in range(1, 7):
        assert sum(item.target_count == target_count for item in train) == 20
    assert len(validation) == 60
    assert len(test) == 120
    assert {item.seed for item in train}.isdisjoint(item.seed for item in validation)
    assert {item.seed for item in train}.isdisjoint(item.seed for item in test)
    assert {item.seed for item in validation}.isdisjoint(item.seed for item in test)


def test_shared_layout_screen_records_ik_exception_as_rejected_seed(monkeypatch):
    from sim.act.collect_dataset import screen_shared_layout
    from sim.config import load_config

    def fail(*_args, **_kwargs):
        raise RuntimeError("UR10 IK target is unreachable")

    monkeypatch.setattr("sim.act.collect_dataset.run_expert_episode", fail)
    accepted, outcomes = screen_shared_layout(
        load_config("configs/default.yaml"), 4100, target_counts=[6]
    )

    assert accepted is False
    assert outcomes[0]["success"] is False
    assert "RuntimeError" in outcomes[0]["reason"]


def test_independent_seed_screen_skips_ik_exception_and_logs_lightweight_failure(
    monkeypatch, tmp_path
):
    from sim.act.generate_objectact_dataset import _find_independent_seed

    calls = {"count": 0}

    def fail_once(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("UR10 IK target is unreachable")
        return SimpleNamespace(success=True)

    monkeypatch.setattr("sim.act.generate_objectact_dataset.run_expert_episode", fail_once)
    seed, attempt = _find_independent_seed(
        load_config("configs/default.yaml"),
        target_count=6,
        requested_seed=49100,
        max_attempts=2,
        root=tmp_path,
        layout_id="independent_n6_test",
    )

    assert (seed, attempt) == (49101, 2)
    failures = [
        json.loads(line)
        for line in (tmp_path / "generation_failures.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(failures) == 1
    assert failures[0]["planner_status"] == "exception"
    assert "RuntimeError" in failures[0]["reason"]


def test_manifest_validator_requires_exactly_20_successful_experts_per_target(tmp_path):
    records = []
    for index, spec in enumerate(training_episode_plan(4100)):
        records.append({
                "episode_id": f"expert_{index:03d}",
                "schema_version": 4,
                "episode_kind": "expert",
                "action_dim": 4,
                "state_dim": 42,
                "frame_count": 100,
                "success": True,
                "split": "train",
                "target_count": spec.target_count,
                "layout_id": spec.layout_id,
                "layout_kind": spec.layout_kind,
                "seed": spec.seed,
            })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = validate_training_manifest(tmp_path)

    assert summary["total"] == 120
    assert summary["per_target"] == {str(index): 20 for index in range(1, 7)}
    assert summary["paired_records"] == 48
    assert summary["independent_records"] == 72


def test_approved_same_layout_previews_promote_losslessly_to_train_set(tmp_path):
    preview_root = tmp_path / "preview"
    train_root = tmp_path / "train"
    for target_count in range(1, 7):
        _write_v4_episode(
            preview_root,
            f"preview_n{target_count}",
            episode_kind="expert_preview",
            split="pilot",
            target_count=target_count,
            layout_id="paired_000",
        )

    promoted = promote_preview_batch(preview_root, train_root)

    assert len(promoted) == 6
    assert {record["target_count"] for record in promoted} == set(range(1, 7))
    assert {record["episode_kind"] for record in promoted} == {"expert"}
    assert {record["split"] for record in promoted} == {"train"}
    assert all("target_indices" not in record for record in promoted)
    for record in promoted:
        assert (train_root / record["arrays"]).is_file()


def test_all_successful_previews_can_be_promoted_to_a_pilot_train_set(tmp_path):
    preview_root = tmp_path / "preview"
    train_root = tmp_path / "train"
    for target_count in range(1, 7):
        _write_v4_episode(
            preview_root,
            f"preview_goal_{target_count}_0001",
            episode_kind="expert_preview",
            split="pilot",
            target_count=target_count,
        )
    manifest = preview_root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["target_indices"] = [0]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    promoted = promote_success_preview_set(
        preview_root, train_root, expected_per_target=1
    )

    assert len(promoted) == 6
    assert validate_preview_training_manifest(train_root, per_target=1)["total"] == 6
    assert {record["episode_kind"] for record in promoted} == {"expert"}
    assert {record["split"] for record in promoted} == {"train"}
    assert all("target_indices" not in record for record in promoted)


def test_training_manifest_rejects_planner_target_identity(tmp_path):
    records = []
    for index, spec in enumerate(training_episode_plan(4100)):
        records.append({
            "episode_id": f"expert_{index:03d}",
            "schema_version": 4,
            "episode_kind": "expert",
            "action_dim": 4,
            "state_dim": 42,
            "frame_count": 100,
            "success": True,
            "split": "train",
            "target_count": spec.target_count,
            "layout_id": spec.layout_id,
            "layout_kind": spec.layout_kind,
            "seed": spec.seed,
        })
    records[0]["target_indices"] = [0]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible records"):
        validate_training_manifest(tmp_path)
