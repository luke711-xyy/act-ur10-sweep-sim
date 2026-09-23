import numpy as np
from types import SimpleNamespace

from sim.act.interface import ACTObservationBuilder
from sim.act.realtime import ActionChunkScheduler
from sim.act.rollout import run_action_path
from sim.config import load_config
from sim.environments.sweep_env import SweepEnv
from sim.model.scene_builder import build_scene_xml
from sim.environments.layout import sample_layout


def test_ur10_scene_exposes_six_joints_brush_and_wrist_camera():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    for name in ("ur10_shoulder_joint", "ur10_upper_arm_joint", "ur10_forearm_joint",
                 "ur10_wrist_1_joint", "ur10_wrist_2_joint", "ur10_wrist_3_joint"):
        assert f'name="{name}"' in xml
    assert 'name="brush_head"' in xml
    assert 'name="wrist_cam"' in xml


def test_mujoco_reset_render_and_act_observation_contract():
    cfg = load_config()
    env = SweepEnv(cfg, seed=3)
    try:
        env.reset(seed=3)
        assert np.linalg.norm(env.tcp() - np.array([0.42, 0.0, cfg.end_effector.z_home])) < 1e-3
        assert env.render_rgb("overhead_cam", size=(64, 48)).shape == (48, 64, 3)
        assert env.render_wrist_rgb(size=(64, 48)).shape == (48, 64, 3)
        observation = ACTObservationBuilder(cfg).observe(env)
        assert observation["observation.images.overhead"].shape == (3, 320, 320)
        assert observation["observation.images.wrist"].shape == (3, 320, 320)
        assert observation["observation.state"].shape == (26,)
        assert observation["observation.environment_state"].shape == (2,)
    finally:
        env.close()


def test_scheduler_uses_future_aligned_action_index():
    cfg = load_config()
    scheduler = ActionChunkScheduler(cfg)
    scheduler.reset(0.0)
    values = np.arange(80, dtype=np.float32).reshape(20, 4)
    assert scheduler.accept(0.0, values, 0.199)
    assert scheduler.action_for(0.199) is None
    np.testing.assert_array_equal(scheduler.action_for(0.2), values[4])


def test_scheduler_aligns_late_preview_chunk_to_current_simulation_time():
    cfg = load_config()
    scheduler = ActionChunkScheduler(cfg)
    scheduler.reset(0.0)
    values = np.arange(80, dtype=np.float32).reshape(20, 4)
    assert scheduler.accept(0.0, values, 0.7, allow_late=True)
    np.testing.assert_array_equal(scheduler.action_for(0.7), values[14])
    assert not scheduler.accept(0.0, values, 1.01, allow_late=True)


def test_airborne_transfer_does_not_consume_contact_path_budget():
    cfg = load_config()

    class AirborneEnv:
        def __init__(self):
            self.pose = np.array([0.42, 0.0, cfg.end_effector.z_home], dtype=float)
            self.yaw = 0.0
            self.ee = SimpleNamespace(tcp_yaw=lambda: self.yaw)
            self.layout = [object()]
            self.table_top_z = float(cfg.table.top_z)
            self.time = 0.0

        def tcp(self):
            return self.pose.copy()

        def normal_force(self):
            return 0.0

        def step_control(self, command):
            value = np.asarray(command.as_array(), dtype=float)
            self.pose = value[:3]
            self.yaw = float(value[3])
            self.time += 1.0 / float(cfg.sim.control_hz)

        def collected_mask(self):
            return np.zeros(1, dtype=bool)

        def lost_mask(self):
            return np.zeros(1, dtype=bool)

    xy = np.array([
        [0.42, 0.0], [0.42, 0.30], [-0.40, 0.30], [-0.40, -0.30],
        [0.42, -0.30], [0.42, 0.0], [0.42, 0.30],
    ], dtype=np.float32)
    path = np.column_stack((
        xy,
        np.full(len(xy), float(cfg.end_effector.z_home), dtype=np.float32),
        np.zeros(len(xy), dtype=np.float32),
    ))
    result = run_action_path(AirborneEnv(), cfg, path, collect_observations=False)

    assert result.failure_reason == "not all components were collected"
