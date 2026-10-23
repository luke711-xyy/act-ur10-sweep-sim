"""Dynamic native-scene adapter retains the original task and sensor contracts."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_native_collection_adapter_preserves_task_and_camera_contracts():
    assert (ROOT/'sim/environments/rudra_sweep_env.py').exists(), 'missing dynamic native sweep adapter'
    from sim.environments.rudra_sweep_env import RudraSweepEnv, collection_config
    from sim.config import load_config
    cfg = collection_config()
    old = load_config(str(ROOT/'configs/five-view-fine-teached-300.yaml'))
    assert cfg.controller.to_dict() == old.controller.to_dict()
    assert list(cfg.act.camera_views) == ['top','wrist']
    env = RudraSweepEnv(cfg,seed=410000)
    try:
        env.reset()
        assert env.model.nu == 7
        assert env.model.ncam == 3
        assert env.model.nsensor == 4
        assert abs(env.time-float(cfg.episode.settle_time))<1e-9
        np.testing.assert_allclose(env.tcp(),[.42,0,.10],atol=.0005)
        assert abs(env.ee.tcp_yaw()) < .001
        assert np.isfinite(env.wrench()).all()
        for i in range(6):
            assert mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'joint_{i}_act') >= 0
        assert env.component_positions().shape == (6,3)
        assert not env.collected_mask().any()
        assert np.max(np.abs(env.ee.compensated_wrench_world())) < .00001
        for view in ('top','wrist'):
            pixels = env.render_rgb(view,size=(320,320))
            assert pixels.shape == (320,320,3) and pixels.std() > 8
    finally:
        env.close()


def test_native_arm_uses_servos_not_teleportation_for_control_steps():
    assert (ROOT/'sim/environments/rudra_sweep_env.py').exists(), 'missing dynamic native sweep adapter'
    from sim.environments.rudra_sweep_env import RudraSweepEnv, collection_config
    from sim.controllers.hybrid import Command
    env = RudraSweepEnv(collection_config(),seed=410000)
    try:
        env.reset()
        before = env.data.qpos.copy()
        env.ee.set_command(.419,0,.1,0)
        np.testing.assert_allclose(env.data.qpos,before,atol=1e-12)
        for _ in range(50):
            env.step_control(Command(.419,0,.1,0))
        assert np.isfinite(env.data.qpos).all()
        assert abs(env.tcp()[0]-.419) < .0005
        assert max(abs(env.compensated_wrench())) < .05
        assert not env.brush_table_contact()
    finally:
        env.close()


def test_expert_environment_selection_is_opt_in():
    from sim.act import rollout
    assert hasattr(rollout,'_episode_environment'), 'expert runner has no opt-in environment factory'
    from sim.environments.rudra_sweep_env import RudraSweepEnv, collection_config
    from sim.environments.sweep_env import SweepEnv
    from sim.config import load_config
    native = rollout._episode_environment(collection_config(),123)
    original = rollout._episode_environment(load_config(),123)
    assert isinstance(native,RudraSweepEnv)
    assert type(original) is SweepEnv
    native.close()
    original.close()


def test_native_force_control_keeps_micrometre_z_commands():
    from sim.environments.rudra_sweep_env import RudraSweepEnv, collection_config
    env = RudraSweepEnv(collection_config(),seed=410000)
    try:
        env.reset()
        ids = list(env.ee.act_ids.values())
        before = env.data.ctrl[ids].copy()
        env.ee.set_command(.42,0,.100001,0)
        assert np.linalg.norm(env.data.ctrl[ids]-before)>1e-7, 'IK discarded a 1 micrometre force-control correction'
    finally:
        env.close()


def test_sole_contact_adaptation_does_not_change_part_pushing_friction():
    from sim.environments.rudra_sweep_env import RudraSweepEnv, collection_config
    cfg = collection_config()
    cfg.set_path('end_effector.brush_sole_friction',[.1,.0001,.0001])
    env = RudraSweepEnv(cfg,410000)
    try:
        env.reset()
        gid = lambda n:mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,n)
        np.testing.assert_allclose(env.model.geom_friction[gid('brush_sole')],[.1,.0001,.0001])
        np.testing.assert_allclose(env.model.geom_friction[gid('brush_head')],cfg.end_effector.brush_friction)
    finally:
        env.close()
