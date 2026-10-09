import json
import threading
from pathlib import Path
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


def test_writer_encodes_views_concurrently_without_changing_pixels_or_paths(
    tmp_path, monkeypatch,
):
    from PIL import Image

    root = tmp_path / "parallel_writer"
    views = ("corner_a", "corner_b", "corner_c", "corner_d", "wrist")
    expected = {
        (frame, view): np.full((12, 12, 3), frame * 20 + index, dtype=np.uint8)
        for frame in range(2)
        for index, view in enumerate(views)
    }
    observations = [{
        "images": {view: expected[(frame, view)] for view in views},
        "state": np.zeros(36, dtype=np.float32),
        "environment_state": np.array([2.0], dtype=np.float32),
    } for frame in range(2)]

    original_save = Image.Image.save
    barrier = threading.Barrier(2)
    active_saves = 0
    max_active_saves = 0
    active_lock = threading.Lock()

    def observe_parallel_save(image, path, *args, **kwargs):
        nonlocal active_saves, max_active_saves
        with active_lock:
            active_saves += 1
            max_active_saves = max(max_active_saves, active_saves)
        try:
            try:
                barrier.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
            return original_save(image, path, *args, **kwargs)
        finally:
            with active_lock:
                active_saves -= 1

    monkeypatch.setattr(Image.Image, "save", observe_parallel_save)
    ActDatasetWriter(str(root), image_format="webp-lossless").add_episode(
        "parallel_episode", observations, np.zeros((2, 4), dtype=np.float32),
        True, {"episode_kind": "expert", "target_count": 2},
    )

    assert max_active_saves >= 2
    record = json.loads((root / "manifest.jsonl").read_text().splitlines()[0])
    assert record["frame_count"] == 2
    assert set(record["images"]) == set(views)
    for view in views:
        paths = record["images"][view]
        assert [Path(path).name for path in paths] == [
            f"image_{view}_00000.webp", f"image_{view}_00001.webp",
        ]
        for frame, relative_path in enumerate(paths):
            with Image.open(root / relative_path) as saved:
                np.testing.assert_array_equal(
                    np.asarray(saved.convert("RGB")), expected[(frame, view)]
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
    assert "inspection_cam" not in cfg.video.extra_cameras
    assert int(cfg.components.count) == 6
    assert int(cfg.act.training_episodes_per_target) == 50
    assert cfg.controller.force_sensor == "wrist_ft"
    assert float(cfg.controller.desired_force) == 0.1
    assert float(cfg.controller.force_filter_cutoff_hz) == 5.0
    assert float(cfg.controller.yaw_speed_limit) == 0.35


def test_act_model_config_receives_all_five_views_not_inspection(monkeypatch):
    import sys
    import sim.act.policy as policy

    class FeatureType:
        VISUAL = "visual"
        STATE = "state"
        ENV = "environment"
        ACTION = "action"

    class FakeACTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        policy, "require_act_dependencies",
        lambda: (FeatureType, lambda feature_type, shape: {
            "type": feature_type, "shape": shape,
        }, FakeACTConfig, object),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    ))
    cfg = load_config()

    model_config = policy.build_act_config(cfg)
    model_views = {
        key.removeprefix("observation.images.")
        for key, feature in model_config.input_features.items()
        if feature["type"] == FeatureType.VISUAL
    }

    assert model_views == set(VIEWS)
    assert "observation.images.inspection" not in model_config.input_features


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
             "raw_normal_force": 0.02,
             "compensated_normal_force_raw": 0.0,
             "raw_normal_force_peak": 0.01, "raw_normal_force_peak_t": 0.001},
            {"t": 0.01, "tcp": [0.0, 0.0, 0.0], "normal_force": 0.08,
             "raw_normal_force": 0.15,
             "compensated_normal_force_raw": 0.09,
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
    assert "inspection" not in manifest_record
    assert not list((root / "expert_five_view").glob("inspection_*"))
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
    assert set(frame) == {"images", "overhead", "wrist"}
    for view in VIEWS:
        np.testing.assert_array_equal(frame["images"][view], images[view])
    signals = state.load_episode_signals("expert_five_view")
    assert signals["environment_state"] == [[6.0, 3.0, 1.0]]
    assert signals["control_t"] == [0.0, 0.01]
    np.testing.assert_allclose(signals["control_fz"], [0.0, 0.08])
    np.testing.assert_allclose(signals["control_raw_fz"], [0.02, 0.15])
    np.testing.assert_allclose(signals["control_compensated_fz_raw"], [0.0, 0.09])
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
