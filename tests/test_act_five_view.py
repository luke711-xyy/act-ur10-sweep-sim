import json
from types import SimpleNamespace

import numpy as np

from sim.act.dataset import ActDataset, ActDatasetWriter
from sim.act.interface import ACTObservationBuilder
from sim.config import load_config
from sim.web.workbench import WorkbenchState


VIEWS = (
    "corner_xminus_yminus",
    "corner_xminus_yplus",
    "corner_xplus_yminus",
    "corner_xplus_yplus",
    "wrist",
)


class _FakeEE:
    def joint_state(self):
        return np.zeros(6, dtype=np.float32)

    def tcp_yaw(self):
        return 0.0


class _FakeEnv:
    def __init__(self):
        self.ee = _FakeEE()
        self.layout = [object()] * 6
        self.time = 0.0

    def tcp(self):
        return np.array([0.2, -0.1, 0.08], dtype=np.float32)

    def wrench(self):
        return np.zeros(6, dtype=np.float32)

    def collected_mask(self):
        return np.array([True, False, False, False, False, False])

    def render_rgb(self, camera, size):
        channel = VIEWS.index(camera) + 1
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 0] = channel
        return frame

    def render_wrist_rgb(self, size):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 0] = 5
        return frame


def test_observation_builder_emits_all_five_policy_images_and_only_goal_count():
    cfg = load_config(overrides=[
        "act.image_size=[16,16]", "task.target_count=4",
    ])
    observation = ACTObservationBuilder(cfg).observe(_FakeEnv())

    image_keys = {
        key.removeprefix("observation.images.")
        for key in observation if key.startswith("observation.images.")
    }
    assert image_keys == set(VIEWS)
    assert all(observation[f"observation.images.{view}"].shape == (3, 16, 16)
               for view in VIEWS)
    np.testing.assert_array_equal(observation["observation.environment_state"], [4])
    np.testing.assert_array_equal(observation["supervisor_state"], [6, 4, 1])


def test_default_collection_config_uses_five_views_and_rudra_force_tuning():
    cfg = load_config()

    assert tuple(cfg.act.camera_views) == VIEWS
    assert int(cfg.components.count) == 6
    assert int(cfg.act.training_episodes_per_target) == 50
    assert cfg.controller.force_sensor == "wrist_ft"
    assert float(cfg.controller.desired_force) == 0.1
    assert float(cfg.controller.force_filter_cutoff_hz) == 5.0
    assert float(cfg.controller.yaw_speed_limit) == 0.35


def test_five_view_episode_round_trips_through_dataset_and_workbench(tmp_path):
    cfg = load_config(overrides=[
        "act.image_size=[16,16]", f"act.dataset_dir={tmp_path / 'dataset'}",
    ])
    raw = ACTObservationBuilder(cfg).observe(_FakeEnv())
    images = {
        view: np.transpose(raw[f"observation.images.{view}"], (1, 2, 0)) * 255
        for view in VIEWS
    }
    images = {name: value.astype(np.uint8) for name, value in images.items()}
    observation = {
        "images": images,
        "inspection": np.full((16, 16, 3), 99, dtype=np.uint8),
        "state": raw["observation.state"],
        "environment_state": raw["observation.environment_state"],
        "supervisor_state": raw["supervisor_state"],
        "fully_collected": 1,
        "phase": "approach",
        "policy_mask": True,
    }
    root = tmp_path / "dataset"
    ActDatasetWriter(str(root), image_format="webp-lossless").add_episode(
        "expert_five_view", [observation], np.zeros((1, 4), dtype=np.float32),
        True, {
            "episode_kind": "expert", "split": "train", "target_count": 4,
            "total_count": 6,
        },
        control_trace=[
            {"t": 0.0, "tcp": [0.0, 0.0, 0.1], "normal_force": 0.0,
             "raw_normal_force_peak": 0.01, "raw_normal_force_peak_t": 0.001},
            {"t": 0.01, "tcp": [0.0, 0.0, 0.0], "normal_force": 0.08,
             "raw_normal_force_peak": 0.12, "raw_normal_force_peak_t": 0.008,
             "table_contact": True},
        ],
    )

    dataset = ActDataset(str(root), chunk_size=2)
    sample = dataset[0]
    assert dataset.camera_views == VIEWS
    assert all(sample[f"observation.images.{view}"].shape == (3, 16, 16)
               for view in VIEWS)
    manifest_record = json.loads(
        (root / "manifest.jsonl").read_text().splitlines()[0]
    )
    assert manifest_record["image_storage_format"] == "webp-lossless"
    assert all(path.endswith(".webp")
               for paths in manifest_record["images"].values()
               for path in paths)
    assert all(path.endswith(".webp") for path in manifest_record["inspection"])
    for view in VIEWS:
        np.testing.assert_array_equal(
            np.transpose(sample[f"observation.images.{view}"], (1, 2, 0)),
            images[view],
        )
    assert sample["observation.environment_state"].shape == (1,)
    assert set(dataset.stats) == {
        "observation.state", "observation.environment_state", "action",
        *(f"observation.images.{view}" for view in VIEWS),
    }

    state = WorkbenchState(cfg, dataset_root=root, preview_root=tmp_path / "empty")
    frame = state.load_episode_frame("expert_five_view", 0)
    assert set(frame["images"]) == set(VIEWS)
    assert frame["inspection"].shape == (16, 16, 3)
    for view in VIEWS:
        np.testing.assert_array_equal(frame["images"][view], images[view])
    np.testing.assert_array_equal(
        frame["inspection"], np.full((16, 16, 3), 99, dtype=np.uint8)
    )
    signals = state.load_episode_signals("expert_five_view")
    assert signals["environment_state"] == [[6.0, 3.0, 1.0]]
    assert signals["control_t"] == [0.0, 0.01]
    np.testing.assert_allclose(signals["control_fz"], [0.0, 0.08])
    np.testing.assert_allclose(signals["control_raw_fz_peak"], [0.01, 0.12])
    np.testing.assert_allclose(signals["control_raw_fz_peak_t"], [0.001, 0.008])


def test_policy_config_declares_five_visual_inputs_and_scalar_goal(monkeypatch):
    import sim.act.policy as policy_module

    class FeatureType:
        VISUAL = "visual"
        STATE = "state"
        ENV = "environment"
        ACTION = "action"

    class PolicyFeature:
        def __init__(self, type, shape):
            self.type = type
            self.shape = shape

    class FakeACTConfig:
        def __init__(self, **values):
            self.__dict__.update(values)

    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        policy_module, "require_act_dependencies",
        lambda: (FeatureType, PolicyFeature, FakeACTConfig, object),
    )
    cfg = load_config(overrides=["act.device=cpu"])
    policy_cfg = policy_module.build_act_config(cfg)

    assert {
        key.removeprefix("observation.images.")
        for key in policy_cfg.input_features
        if key.startswith("observation.images.")
    } == set(VIEWS)
    assert policy_cfg.input_features["observation.environment_state"].shape == (1,)
