import numpy as np
import mujoco
import xml.etree.ElementTree as ET

from sim.config import load_config
from sim.act.expert import ExpertPlan
from sim.act.rollout import _expert_execution_path, run_action_path
from sim.environments.layout import sample_layout
from sim.environments.sweep_env import SweepEnv
from sim.model.scene_builder import build_scene_xml, scene_assets


def _rudra_config():
    return load_config(overrides=[
        "end_effector.type=ur10_cb3_rudra",
        "sim.real_time=false",
    ])


def test_default_robot_profile_uses_the_integrated_rudra_cb3_arm():
    assert str(load_config().end_effector.type) == "ur10_cb3_rudra"


def test_rudra_ur10_mjcf_loads_with_existing_task_tool_and_sensor_contract():
    cfg = _rudra_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())

    for name in ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act"
        ) >= 0

    for kind, names in (
        (mujoco.mjtObj.mjOBJ_BODY, ("base", "wrist_3_link", "tool")),
        (mujoco.mjtObj.mjOBJ_SITE, ("tcp_site", "ft_site")),
        (mujoco.mjtObj.mjOBJ_GEOM,
         ("brush_head", "brush_sole", "tray_floor_visual")),
        (mujoco.mjtObj.mjOBJ_CAMERA, ("wrist_cam", "scene_cam")),
    ):
        for name in names:
            assert mujoco.mj_name2id(model, kind, name) >= 0

    assert model.nu == 6
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_2f85") >= 0


def test_rudra_tool_is_mounted_at_the_imported_flange_not_a_second_wrist_offset():
    cfg = _rudra_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())
    flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "flange_dh")
    tool_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")

    assert model.body_parentid[tool_id] == flange_id
    np.testing.assert_allclose(model.body_pos[tool_id], [0.0, 0.0, 0.0], atol=1e-8)


def test_rudra_wrist_camera_keeps_the_previous_camera_pose_relative_to_tcp():
    camera_poses = {}
    for profile in ("ur10e", "ur10_cb3_rudra"):
        cfg = load_config(overrides=[
            f"end_effector.type={profile}",
            "sim.real_time=false",
        ])
        env = SweepEnv(cfg, seed=0)
        try:
            env.reset(seed=0)
            camera_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
            )
            tcp_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site"
            )
            camera_poses[profile] = (
                env.data.cam_xpos[camera_id] - env.data.site_xpos[tcp_id],
                env.data.cam_xmat[camera_id].reshape(3, 3).copy(),
            )
        finally:
            env.close()

    # Switching MJCF sources must not move the eye-in-hand viewpoint relative
    # to the task tool. The upstream CB3 wrist frame is 10 cm higher than the
    # Menagerie frame at the same TCP pose, so the Rudra camera needs a
    # profile-specific mount offset.
    np.testing.assert_allclose(
        camera_poses["ur10_cb3_rudra"][0],
        camera_poses["ur10e"][0],
        atol=0.005,
    )
    np.testing.assert_allclose(
        camera_poses["ur10_cb3_rudra"][1],
        camera_poses["ur10e"][1],
        atol=0.005,
    )


def test_rudra_initial_posture_does_not_interpenetrate_upstream_arm_meshes():
    cfg = _rudra_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    root = ET.fromstring(xml)
    visual_geom = root.find("./default/default[@class='ur_vis']/geom")
    assert visual_geom is not None
    # The vendored arm is visual-only in production. Enable its mesh convex
    # hulls in this diagnostic model to catch self-interpenetrating poses.
    visual_geom.set("contype", "1")
    visual_geom.set("conaffinity", "1")
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=scene_assets()
    )
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot_bodies = set()
    for body_id in range(model.nbody):
        ancestor = body_id
        while ancestor > 0:
            if ancestor == base_id:
                robot_bodies.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    robot_mesh_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_bodies
        and int(model.geom_contype[geom_id]) == 1
    }
    for index, value in enumerate(cfg.end_effector.rudra_initial_joint_positions):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}"
        )
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
    mujoco.mj_forward(model, data)

    self_contacts = [
        (int(data.contact[i].geom1), int(data.contact[i].geom2))
        for i in range(int(data.ncon))
        if int(data.contact[i].geom1) in robot_mesh_geoms
        and int(data.contact[i].geom2) in robot_mesh_geoms
    ]
    assert not self_contacts, f"initial UR10 posture self-intersects: {self_contacts}"


def test_rudra_ur10_joint_limits_allow_the_existing_ik_solver_to_move_the_arm():
    cfg = _rudra_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())

    for index in range(6):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}"
        )
        expected_limit = np.pi if index == 2 else 2.0 * np.pi
        assert model.jnt_limited[joint_id]
        np.testing.assert_allclose(
            model.jnt_range[joint_id], [-expected_limit, expected_limit], atol=1e-5
        )


def test_rudra_ur10_servos_match_existing_ur10_position_control_contract():
    cfg = _rudra_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())
    expected_torque_limits = (330.0, 330.0, 150.0, 56.0, 56.0, 56.0)

    for index, torque_limit in enumerate(expected_torque_limits):
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{index}_act"
        )
        assert model.actuator_gainprm[actuator_id, 0] == 5000.0
        np.testing.assert_allclose(
            model.actuator_biasprm[actuator_id, 1:3], [-5000.0, -500.0]
        )
        assert model.actuator_forcelimited[actuator_id]
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator_id], [-torque_limit, torque_limit]
        )


def test_rudra_ur10_holds_brush_contact_across_nominal_sweep_lane():
    cfg = load_config(overrides=[
        "end_effector.type=ur10_cb3_rudra",
        "sim.real_time=false",
        "task.target_count=6",
    ])
    env = SweepEnv(cfg, seed=290926)
    try:
        env.reset(seed=290926)
        plan = ExpertPlan(
            target_indices=np.arange(6),
            waypoints=np.array([
                [0.24, -0.30], [0.14, -0.30],
                [0.04, -0.30], [-0.06, -0.30],
            ]),
            feasible=True,
            strategy="coverage_astar",
        )
        path, phases = _expert_execution_path(cfg, plan)
        result = run_action_path(
            env, cfg, path, collect_observations=False,
            phases=phases, target_indices=np.arange(6),
        )
        sweep = [row for row in result.trace if row["phase"] == "sweep"]

        assert len(sweep) > 400
        assert all(row["contact"] for row in sweep)
        assert result.failure_reason != "contact lost for 100 ms"
        assert result.peak_force <= 20.0
    finally:
        env.close()


def test_rudra_ur10_can_reset_and_reach_the_existing_task_start_pose():
    cfg = _rudra_config()
    env = SweepEnv(cfg, seed=12)
    try:
        env.reset(seed=12)
        np.testing.assert_allclose(env.tcp(), [0.42, 0.0, cfg.end_effector.z_home], atol=1e-3)
        assert env.unsafe_robot_collision() is False
    finally:
        env.close()
