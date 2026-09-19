import numpy as np
import mujoco
import pytest
from xml.etree import ElementTree as ET

from sim.act.interface import ACTObservationBuilder
from sim.act.realtime import ActionChunkScheduler
from sim.config import load_config
from sim.environments.sweep_env import SweepEnv
from sim.model.scene_builder import build_scene_xml, scene_assets
from sim.environments.layout import sample_layout
from sim.planners.geometry_utils import pusher_width
from sim.act.expert import (_grid_astar, _plan_for_order, _tray_exit_x,
                            ExpertPlan, expert_target_indices, expert_waypoints,
                            plan_expert_sweep, sample_polyline)
from sim.act.rollout import _expert_execution_path, _force_loop


def test_ur10e_scene_uses_menagerie_structure_and_custom_brush():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    for name in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
        assert f'name="{name}"' in xml
    assert 'name="upper_arm_link"' in xml
    assert 'mesh="base_0"' in xml
    assert 'name="attachment_site"' in xml
    assert 'name="brush_head"' in xml
    assert 'name="wrist_cam"' in xml


def test_brush_material_is_high_visibility_yellow():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    material = root.find("./asset/material[@name='mat_brush']")
    assert material is not None
    rgba = np.fromstring(material.get("rgba", ""), sep=" ")
    assert rgba.size == 4
    assert rgba[0] >= 0.80
    assert rgba[1] >= 0.45
    assert rgba[2] <= 0.25


def test_global_act_camera_is_front_oblique_not_vertical_overhead():
    cfg = load_config()
    camera = cfg.video.extra_cameras.overhead_cam
    pos = np.asarray(camera.pos, dtype=float)
    lookat = np.asarray(camera.lookat, dtype=float)
    # Keep the historical camera name/ACT tensor key, but make its geometry a
    # useful global view: a front-oblique camera sees the sweep lane instead of
    # letting the arm hide parts directly underneath it.
    assert np.linalg.norm((pos - lookat)[:2]) > 0.10
    assert pos[2] - lookat[2] > 0.20


def test_wrist_camera_is_pitched_towards_tool_contact_region():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    camera = root.find(".//body[@name='wrist_3_link']/camera[@name='wrist_cam']")
    assert camera is not None
    axes = np.fromstring(camera.get("xyaxes", ""), sep=" ")
    assert axes.size == 6
    # The camera is expressed in the final wrist-link frame.  The official
    # attachment and the new gripper/handle stack put the plate centre near
    # ``[0, -0.16, 0]`` in this local frame.
    forward = -np.cross(axes[:3], axes[3:])
    camera_pos = np.fromstring(camera.get("pos", ""), sep=" ")
    contact = np.array([0.0, -0.16, 0.0])
    to_contact = contact - camera_pos
    to_contact /= np.linalg.norm(to_contact)
    assert np.linalg.norm(camera_pos) < 0.30
    assert float(np.dot(forward, to_contact)) > 0.85


def test_wrist_camera_rotates_about_the_vertical_installation_axis():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    camera = root.find(".//body[@name='wrist_3_link']/camera[@name='wrist_cam']")
    assert camera is not None
    axes = np.fromstring(camera.get("xyaxes", ""), sep=" ")
    camera_pos = np.fromstring(camera.get("pos", ""), sep=" ")
    assert axes.size == 6 and camera_pos.size == 3

    # At the level-brush pose the wrist_3_link local +Y axis is the physical
    # vertical installation axis (it maps to world +Z).  Rebuild the prior
    # camera frame, then apply one rigid -25-degree rotation about that local
    # Y axis.  The camera must orbit to the opposite side of the installation
    # axis so the plate does not occlude the swept parts.  Rotating only the optical frame about local Z is the bug that
    # made the previous preview look like a yaw and exposed the broad plate.
    theta = np.deg2rad(-25.0)
    rotation = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    baseline_pos = np.array([0.0, -0.05, 0.19])
    baseline_x = np.array([1.0, 0.0, 0.0])
    baseline_y = np.array([0.0, 0.845489, -0.533993])
    np.testing.assert_allclose(camera_pos, rotation @ baseline_pos, atol=2e-5)
    np.testing.assert_allclose(axes[:3], rotation @ baseline_x, atol=2e-5)
    np.testing.assert_allclose(axes[3:], rotation @ baseline_y, atol=2e-5)

    # The camera remains aimed toward the contact region and the brush width
    # is still close to edge-on, so the turn reveals only a small plate face.
    forward = -np.cross(axes[:3], axes[3:])
    assert float(np.dot(forward, np.array([0.0, 0.0, 1.0]))) < -0.70
    assert float(np.dot(forward, np.array([1.0, 0.0, 0.0]))) > 0.25
    assert abs(float(np.dot(forward, np.array([0.0, 0.0, 1.0])))) < 0.90


def test_ur10e_mounts_a_fixed_closed_robotiq_gripper():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    gripper = root.find(".//body[@name='robotiq_2f85']")
    assert gripper is not None

    # The approved ACT setup uses the official Robotiq meshes at one frozen
    # closed pose.  It must not add a grasp action or passive finger dynamics.
    assert gripper.findall(".//joint") == []
    actuator_names = {node.get("name", "") for node in root.find("actuator")}
    assert "fingers_actuator" not in actuator_names
    mesh_names = {node.get("mesh", "") for node in gripper.findall(".//geom")}
    assert {"robotiq_base", "robotiq_driver", "robotiq_pad"} <= mesh_names


def test_brush_handle_is_centered_between_closed_gripper_pads():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    gripper = root.find(".//body[@name='robotiq_2f85']")
    assert gripper is not None
    handle = gripper.find(".//geom[@name='brush_handle']")
    left = gripper.find(".//geom[@name='robotiq_left_pad_contact']")
    right = gripper.find(".//geom[@name='robotiq_right_pad_contact']")
    assert handle is not None and left is not None and right is not None

    handle_pos = np.fromstring(handle.get("pos", ""), sep=" ")
    handle_size = np.fromstring(handle.get("size", ""), sep=" ")
    left_pos = np.fromstring(left.get("pos", ""), sep=" ")
    right_pos = np.fromstring(right.get("pos", ""), sep=" ")
    left_size = np.fromstring(left.get("size", ""), sep=" ")
    right_size = np.fromstring(right.get("size", ""), sep=" ")
    assert handle_pos[0] == pytest.approx(0.0)
    assert left_pos[0] < handle_pos[0] < right_pos[0]
    assert left_pos[0] + left_size[0] == pytest.approx(-handle_size[0], abs=1e-6)
    assert right_pos[0] - right_size[0] == pytest.approx(handle_size[0], abs=1e-6)
    assert abs(left_pos[2] - handle_pos[2]) < handle_size[2]
    assert abs(right_pos[2] - handle_pos[2]) < handle_size[2]


def test_global_act_camera_is_on_the_opposite_elevated_side():
    cfg = load_config()
    camera = cfg.video.extra_cameras.overhead_cam
    pos = np.asarray(camera.pos, dtype=float)
    # The previous view was on negative Y and let the forearm cover the stroke.
    assert pos[1] > 0.30
    assert pos[2] > 0.60


def test_global_act_camera_is_rotated_higher_and_slightly_closer():
    cfg = load_config()
    camera = cfg.video.extra_cameras.overhead_cam
    pos = np.asarray(camera.pos, dtype=float)
    lookat = np.asarray(camera.lookat, dtype=float)
    delta = pos - lookat
    horizontal = np.linalg.norm(delta[:2])
    elevation = np.arctan2(delta[2], horizontal)

    # Relative to the old [0.30, 0.62, 0.80] position, this is a roughly
    # 17-degree counter-clockwise turn around the table centre, with a tighter
    # horizontal radius and a slightly higher elevation.
    assert pos[0] < 0.15
    assert pos[1] > 0.60
    assert np.linalg.norm(delta) < 1.06
    assert elevation > np.deg2rad(50.0)


def test_inspection_camera_is_a_far_full_scene_view():
    cfg = load_config()
    camera = cfg.video.extra_cameras.inspection_cam
    pos = np.asarray(camera.pos, dtype=float)
    lookat = np.asarray(camera.lookat, dtype=float)

    # This camera is for human inspection, so it must frame the complete arm,
    # table and the -X collection tray rather than act as a second close-up.
    assert np.linalg.norm(pos - lookat) > 1.50
    assert pos[2] > 0.90
    assert float(camera.fovy_deg) >= 52.0
    assert -0.10 <= lookat[0] <= 0.15
    assert abs(lookat[1]) < 0.15


def test_render_lighting_uses_shadowless_multi_source_fill():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    lights = root.findall("./worldbody/light")
    assert len(lights) >= 4
    assert all(light.get("castshadow") == "false" for light in lights)
    headlight = root.find("./visual/headlight")
    assert headlight is not None
    assert headlight.get("ambient") is not None


def test_wrist_camera_is_mounted_beside_final_wrist_link():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    wrist3 = root.find(".//body[@name='wrist_3_link']")
    assert wrist3 is not None
    # The real-style adapter is fixed beside the final wrist joint, not on the
    # brush child body.  The latter made the simulated lens appear detached
    # from the end-effector in the preview.
    assert wrist3.find("./camera[@name='wrist_cam']") is not None
    assert root.find(".//body[@name='tool']/camera[@name='wrist_cam']") is None


def test_brush_plate_is_coaxial_with_the_terminal_wrist_joint():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    tool = root.find(".//body[@name='tool']")
    assert tool is not None
    mount = np.fromstring(tool.get("pos", ""), sep=" ")
    assert mount.size == 3
    np.testing.assert_allclose(mount, [0.0, 0.1, 0.0], atol=1e-6)


def test_ur10e_scene_loads_with_vendored_menagerie_assets():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brush_head") >= 0


def test_ur10e_brush_width_is_transverse_to_the_sweep_direction():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())
    brush_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brush_head")
    np.testing.assert_allclose(
        model.geom_size[brush_id],
        [cfg.end_effector.brush_depth / 2.0,
         cfg.end_effector.brush_width / 2.0,
         cfg.end_effector.brush_height / 2.0],
    )
    assert pusher_width(cfg) == pytest.approx(float(cfg.end_effector.brush_width))


def test_ur10e_brush_head_has_a_plate_like_height_and_width():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())
    brush_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brush_head")
    depth, width, height = 2.0 * model.geom_size[brush_id]
    # A plate needs a visible vertical face, not a thin lip.  Keep the
    # sweeping width dominant while making height materially larger than depth.
    assert width >= 0.14
    assert height >= 0.06
    assert depth <= 0.01
    assert height > depth


def test_brush_plate_collision_proxy_can_push_components():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    model = mujoco.MjModel.from_xml_string(xml, assets=scene_assets())
    brush_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brush_head")
    proxy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 "comp_0_g0_collision")
    assert model.geom_contype[brush_id] & model.geom_conaffinity[proxy_id]


def test_screw_is_stationary_at_the_start_of_an_episode():
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1", "components.geometry=screw",
        "components.spawn_mode=uniform",
    ])
    env = SweepEnv(cfg, seed=7)
    try:
        env.reset(seed=7)
        body_id = env.component_body_ids[0]
        joint_id = int(env.model.body_jntadr[body_id])
        dof_start = int(env.model.jnt_dofadr[joint_id])
        np.testing.assert_allclose(env.data.qvel[dof_start:dof_start + 6], 0.0, atol=1e-10)
    finally:
        env.close()


def test_component_contacts_enable_rolling_friction():
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1", "components.geometry=screw",
        "components.spawn_mode=uniform",
    ])
    env = SweepEnv(cfg, seed=7)
    try:
        env.reset(seed=7)
        assert all(int(env.model.geom_condim[gid]) >= 6
                   for gid in env.component_geom_ids[0])
    finally:
        env.close()


def test_screw_does_not_drift_without_external_force():
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1", "components.geometry=screw",
        "components.spawn_mode=uniform",
    ])
    env = SweepEnv(cfg, seed=7)
    try:
        env.reset(seed=7)
        start = env.component_positions()[0, :2].copy()
        for _ in range(3000):
            mujoco.mj_step(env.model, env.data)
        body_id = env.component_body_ids[0]
        joint_id = int(env.model.body_jntadr[body_id])
        dof_start = int(env.model.jnt_dofadr[joint_id])
        angular_speed = np.linalg.norm(env.data.qvel[dof_start + 3:dof_start + 6])
        drift = np.linalg.norm(env.component_positions()[0, :2] - start)
        assert angular_speed < 0.05
        assert drift < 0.01
    finally:
        env.close()


def test_ur10e_chain_has_explicit_gravity_compensation_for_position_control():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    base = root.find("./worldbody/body[@name='base']")
    assert base is not None
    assert all(body.get("gravcomp") == "1" for body in base.iter("body"))


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
        assert observation["observation.state"].shape == (42,)
        assert observation["observation.environment_state"].shape == (3,)
        assert observation["observation.environment_state"][0] == 6
        assert observation["observation.environment_state"][1] == cfg.task.target_count
    finally:
        env.close()


def test_ur10_rejects_unreachable_ik_and_reset_has_no_unsafe_collision():
    cfg = load_config(overrides=["sim.real_time=false"])
    env = SweepEnv(cfg, seed=0)
    try:
        env.reset(seed=0)
        assert env.unsafe_robot_collision() is False
        with pytest.raises(RuntimeError, match="IK target is unreachable"):
            env.ee.set_command(2.0, 2.0, 2.0, 0.0)
    finally:
        env.close()


def test_ur10e_initial_pose_keeps_visual_links_above_table():
    cfg = load_config()
    env = SweepEnv(cfg, seed=0)
    try:
        env.reset(seed=0)
        body_ids = [
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("shoulder_link", "upper_arm_link", "forearm_link",
                         "wrist_1_link", "wrist_2_link", "wrist_3_link", "tool")
        ]
        assert min(float(env.data.xpos[body_id][2]) for body_id in body_ids) > 0.0
    finally:
        env.close()


def test_ur10e_tcp_is_brush_bottom_and_contacts_at_search_height():
    from sim.controllers.hybrid import Command
    brush_id = None

    cfg = load_config(overrides=["sim.real_time=false"])
    env = SweepEnv(cfg, seed=0)
    try:
        env.reset(seed=0)
        assert np.isclose(env.tcp()[2], cfg.end_effector.z_home, atol=1e-3)
        brush_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "brush_head")
        brush_bottom = (env.data.geom_xpos[brush_id]
                        - env.data.geom_xmat[brush_id].reshape(3, 3)[:, 2]
                        * env.model.geom_size[brush_id, 2])
        np.testing.assert_allclose(brush_bottom, env.tcp(), atol=1e-5)
        # The longer gripper/handle stack needs slightly more actuator-settle
        # time than the former short rigid stem before the plate reaches the
        # table; the assertion remains about real measured contact.
        for _ in range(220):
            env.step_control(Command(0.42, 0.0, float(cfg.workspace.z_search_start), 0.0))
        assert env.tcp()[2] < 0.01
        assert env.normal_force() > 0.0
    finally:
        env.close()


def test_scheduler_uses_future_aligned_action_index():
    cfg = load_config()
    scheduler = ActionChunkScheduler(cfg)
    scheduler.reset(0.0)
    values = np.arange(75, dtype=np.float32).reshape(25, 3)
    assert scheduler.accept(0.0, values, 0.199)
    assert scheduler.action_for(0.199) is None
    np.testing.assert_array_equal(scheduler.action_for(0.2), values[5])


def test_preview_scheduler_accepts_late_chunk_and_aligns_to_current_position():
    cfg = load_config()
    scheduler = ActionChunkScheduler(cfg, allow_late=True)
    scheduler.reset(0.0)
    values = np.arange(75, dtype=np.float32).reshape(25, 3)

    assert scheduler.accept(0.0, values, 0.35)
    np.testing.assert_array_equal(scheduler.action_for(0.35), values[8])
    assert scheduler.late_results == 1


def test_act_contact_reference_is_interpolated_and_rate_limited():
    from sim.act.evaluate import _contact_reference_substep

    start = np.array([0.42, 0.0, -0.001, 0.0], dtype=np.float32)
    target = np.array([0.4044, -0.0151, -0.001, 0.2202], dtype=np.float32)
    dt = 1.0 / float(load_config().sim.control_hz)
    substeps = 4
    refs = [
        _contact_reference_substep(
            start, target, index, substeps, dt,
            max_speed=0.08, max_yaw_rate=0.7853981634,
        )
        for index in range(1, substeps + 1)
    ]
    previous = np.array([start[0], start[1], start[3]], dtype=np.float32)
    for reference in refs:
        delta_xy = np.linalg.norm(reference[:2] - previous[:2])
        delta_yaw = abs(float(np.arctan2(
            np.sin(reference[2] - previous[2]),
            np.cos(reference[2] - previous[2]),
        )))
        assert delta_xy <= 0.08 * dt + 1e-7
        assert delta_yaw <= 0.7853981634 * dt + 1e-7
        previous = reference
    assert np.linalg.norm(refs[-1][:2] - start[:2]) < np.linalg.norm(target[:2] - start[:2])
    assert abs(float(refs[-1][2])) < abs(float(target[3]))


def test_act_contact_z_command_includes_upward_overforce_relief_without_act_z():
    from sim.act.evaluate import _contact_z_command

    cfg = load_config()
    baseline = _force_loop(cfg)
    with_relief = _force_loop(cfg)
    z_nominal = -0.001
    measured = 10.0
    baseline_z = z_nominal + baseline.step(float(cfg.controller.desired_force), measured)
    relieved_z = _contact_z_command(cfg, with_relief, z_nominal, measured)

    assert relieved_z > baseline_z
    assert relieved_z >= float(cfg.workspace.z_search_min)


def test_symmetric_brush_does_not_flip_180_degrees_when_path_reverses():
    path = sample_polyline(
        np.array([[0.0, 0.0], [-0.10, 0.0], [0.0, 0.0]]),
        hz=25.0, speed=0.08, initial_yaw=0.0,
    )
    assert np.max(np.abs(path[:, 3])) < np.deg2rad(1.0)


def test_act_contact_latch_does_not_teleport_tcp_to_table_height():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=["sim.real_time=false", "components.count=1"])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    first_contact = next(row for row in result.trace if row["contact"])
    # The force loop must start from the measured TCP pose.  A large downward
    # jump at the latch is an impact, not a controlled contact search.
    assert first_contact["command"][2] >= first_contact["tcp"][2] - 0.005


def test_act_contact_latch_requires_measured_brush_contact():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=["sim.real_time=false", "components.count=1"])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    first_contact = next(row for row in result.trace if row["contact"])
    # Entering the search-height command band is not itself contact.  The
    # force loop must latch only after the brush has produced a real measured
    # load, otherwise the subsequent admittance hold is airborne.
    assert first_contact["normal_force"] >= float(cfg.controller.contact_threshold)


def test_act_episode_always_starts_from_the_same_fixed_reset_pose():
    """Target count must not select a planner-owned inference staging point."""
    poses = []
    for target_count in (1, 3, 6):
        cfg = load_config(overrides=[
            "sim.real_time=false", "components.count=6",
            "components.spawn_mode=cluster", f"task.target_count={target_count}",
        ])
        env = SweepEnv(cfg, seed=5203)
        try:
            env.reset(seed=5203)
            poses.append(np.array([*env.tcp(), env.ee.tcp_yaw()]))
        finally:
            env.close()
    for pose in poses[1:]:
        np.testing.assert_allclose(pose, poses[0], atol=1e-6)


def test_expert_waypoints_stop_inside_ur10e_command_workspace():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        points = expert_waypoints(env, cfg)
        assert float(points[:, 0].min()) >= float(cfg.workspace.x_min)
        plan = plan_expert_sweep(env, cfg)
        if plan.feasible:
            assert float(points[-1, 0]) < float(cfg.target.x_max)
        else:
            assert float(points[-1, 0]) > float(cfg.target.x_max)
    finally:
        env.close()


def test_expert_does_not_sweep_along_tray_side_walls():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        points = expert_waypoints(env, cfg)
        # Once the brush crosses the tray mouth, the centreline must remain
        # inside the usable opening.  A recovery lane at y=+/-tray edge makes
        # the brush itself hit a fixed side wall and is not a valid expert
        # action, even if it sometimes appears to help a lucky layout.
        inside = points[:, 0] <= float(cfg.target.x_max)
        half_width = float(cfg.end_effector.brush_width) / 2.0
        safe_y = (min(float(cfg.target.y_max), float(cfg.target.y_min) * -1.0)
                  - half_width - float(cfg.target.wall_thickness))
        assert np.all(np.abs(points[inside, 1]) <= safe_y + 1e-9)
    finally:
        env.close()


def test_expert_does_not_move_laterally_after_entering_the_tray():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        points = expert_waypoints(env, cfg)
        inside = points[:, 0] <= float(cfg.target.x_max)
        # A lateral consolidation move inside the open-fronted tray can pull
        # an already collected fastener back across a side/entrance boundary.
        if np.any(inside):
            assert np.ptp(points[inside, 1]) == pytest.approx(0.0, abs=1e-9)
        else:
            assert not plan_expert_sweep(env, cfg).feasible
    finally:
        env.close()


def test_expert_finishes_last_object_with_a_horizontal_sweep():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        points = expert_waypoints(env, cfg)
        plan = plan_expert_sweep(env, cfg)
        if not plan.feasible:
            assert float(points[-1, 0]) > float(cfg.target.x_max)
            return
        # A capture-lane route may taper back to the tray centreline.  The
        # invariant is that the brush enters the opening at a wall-safe y,
        # rather than preserving the last object's initial y at the boundary.
        inside = points[:, 0] <= float(cfg.target.x_max)
        half_width = float(cfg.end_effector.brush_width) / 2.0
        safe_y = (min(float(cfg.target.y_max), -float(cfg.target.y_min))
                  - half_width - float(cfg.target.wall_thickness))
        assert np.all(np.abs(points[inside, 1]) <= safe_y + 1e-9)
        assert points[-1, 0] == pytest.approx(
            float(np.clip(float(cfg.planner.stroke_end_x),
                          float(cfg.workspace.x_min), float(cfg.workspace.x_max))))
    finally:
        env.close()


def test_expert_delivery_endpoint_is_deep_inside_the_tray():
    cfg = load_config()
    exit_x = _tray_exit_x(cfg)
    assert exit_x <= float(cfg.target.x_max) - 0.08
    assert exit_x >= (
        float(cfg.target.x_min) + float(cfg.target.wall_thickness)
        + float(cfg.end_effector.brush_depth) / 2.0
    )


def test_expert_final_push_has_no_artificial_long_hold():
    cfg = load_config()
    plan = ExpertPlan(
        target_indices=np.array([0]),
        waypoints=np.array([
            [0.42, 0.0], [0.20, 0.0], [-0.32, 0.0], [-0.40, 0.0],
        ]),
        feasible=True,
    )
    path, phases = _expert_execution_path(cfg, plan)
    sweep = path[np.asarray(phases) == "sweep"]
    outside_entry = np.array([-0.32, 0.0])
    held = np.linalg.norm(sweep[:, :2] - outside_entry, axis=1) < 1e-6
    assert int(held.sum()) <= 3


def test_expert_final_push_remains_successful_without_a_fixed_settle_hold():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1",
        "controller.safe_max_force=100.0",
    ])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success


def test_contact_force_loop_bounds_virtual_penetration():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1",
        "controller.safe_max_force=100.0",
    ])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    contact_rows = [row for row in result.trace if row["contact"]]
    assert contact_rows
    # A position-controlled compliant contact needs a small virtual target
    # below the rigid table plane.  Bound the command by the configured
    # admittance correction and verify the physical TCP remains within the
    # one-millimetre equivalent brush-compliance envelope.
    command_floor = (float(cfg.workspace.z_search_start)
                     - float(cfg.controller.delta_z_limit))
    assert min(float(row["command"][2]) for row in contact_rows) >= command_floor - 1e-9
    assert min(float(row["tcp"][2]) for row in contact_rows) >= float(cfg.table.top_z) - 0.001


def test_default_official_ur10e_demo_uses_calibrated_force_ceiling():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=["sim.real_time=false", "components.count=1"])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success
    assert max(float(row["normal_force"]) for row in result.trace) < float(cfg.controller.safe_max_force)
    assert result.peak_force >= max(
        float(row["normal_force"]) for row in result.trace
    )


def test_expert_peak_force_covers_every_saved_policy_observation():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=["sim.real_time=false", "components.count=1"])
    result = run_expert_episode(cfg, seed=12345, collect_observations=True)

    assert result.success
    assert result.peak_force >= max(
        float(observation["normal_force"])
        for observation in result.observations
    )


def test_default_expert_reaches_contact_before_approach_dominates_episode():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=["sim.real_time=false", "components.count=1"])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    first_contact = next(row for row in result.trace if row["contact"])

    # The demonstration should spend most of its budget on the intentional
    # horizontal sweep, not on a long airborne descent.  Keep this as an
    # end-to-end assertion so changing either the home height or descent speed
    # cannot silently restore the old imbalanced timing.
    assert float(first_contact["t"]) <= 5.0
    assert float(first_contact["t"]) / float(result.elapsed) < 0.50


def test_ur10e_can_reach_expert_tray_lane_from_home():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        lane_x = float(expert_waypoints(env, cfg)[-1, 0])
        solution = env.ee._solve_ik(np.array([lane_x, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(env.ee.tcp_position() - np.array([lane_x, 0.0, 0.0])) < 0.01
        assert solution.shape == (6,)
    finally:
        env.close()


def test_ur10e_ik_tracks_a_continuous_home_to_tray_path():
    cfg = load_config()
    env = SweepEnv(cfg, seed=0)
    try:
        env.reset(seed=0)
        for x in np.linspace(0.42, -0.36, 80):
            env.ee._solve_ik(np.array([x, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(env.ee.tcp_position() - np.array([-0.36, 0.0, 0.0])) < 0.01
    finally:
        env.close()


def test_act_rollout_finishes_when_all_components_are_in_the_tray():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1",
    ])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success
    assert result.collected == result.total == 1


def test_expert_continues_to_tray_depth_after_exact_count_is_reached():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6", "task.target_count=6",
    ])
    result = run_expert_episode(cfg, seed=4104, collect_observations=False)
    assert result.success
    assert result.collected == 6
    trace_x = np.asarray([row["tcp"][0] for row in result.trace], dtype=float)
    assert trace_x.min() <= _tray_exit_x(cfg) + 0.01


def test_official_expert_does_not_recover_a_missed_component():
    """Success must come from one exact capture pass, never a recovery sweep."""
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6", "task.target_count=5",
    ])
    result = run_expert_episode(cfg, seed=2010, collect_observations=False)
    assert result.total == 6
    assert result.success
    assert result.collected == result.target_collected == result.target_count == 5
    assert result.unexpected_collected == 0
    assert not any(row["phase"] == "recovery" for row in result.trace)


def test_collection_goal_is_exact_and_rejects_an_extra_part():
    from sim.act.rollout import collection_goal_status

    # A three-part task is not successful when a fourth, non-target part also
    # crosses the tray boundary.  This is the regression that the former
    # ``collected >= target_count`` check missed.
    ok, reason = collection_goal_status(
        np.array([True, True, True, True, False, False]),
        target_indices=np.array([0, 1, 2]), target_count=3,
    )
    assert not ok
    assert reason == "unexpected component entered the target region"


def test_collection_goal_accepts_any_exact_count_without_expert_identity():
    """ACT evaluation accepts any exact N parts when no A* set is supplied."""
    from sim.act.rollout import collection_goal_status

    ok, reason = collection_goal_status(
        np.array([False, True, True, False, True, False]),
        target_indices=None, target_count=3,
    )
    assert ok
    assert reason == ""


def test_collection_goal_rejects_an_extra_part_carried_to_tray_edge():
    """Three full parts plus a still-carried fourth part is not success."""
    from sim.act.rollout import (
        collection_goal_status,
        unintended_component_mask,
    )

    carried = unintended_component_mask(
        collected_mask=np.array([True, True, True, False]),
        target_indices=None,
        brush_contact_mask=np.array([False, False, False, False]),
        partial_overlap_mask=np.array([False, False, False, True]),
    )
    ok, reason = collection_goal_status(
        np.array([True, True, True, False]),
        target_indices=None,
        target_count=3,
        unintended_component_mask=carried,
    )
    assert not ok
    assert reason == "component projection only partially overlaps the collection region"


def test_partial_overlap_validation_does_not_use_contact_history():
    from sim.act.rollout import unintended_component_mask

    # A part that was contacted in the past but is now away from the tray is
    # not a failure by itself; only current contact or current partial overlap
    # is relevant to the exact-count result.
    mask = unintended_component_mask(
        collected_mask=np.array([True, True, True, False]),
        target_indices=None,
        brush_contact_mask=np.array([False, False, False, False]),
        partial_overlap_mask=np.array([False, False, False, False]),
    )
    assert not np.any(mask)


def test_expert_uses_one_pass_astar_to_collect_exactly_three_of_six():
    from sim.act.rollout import run_expert_episode

    # This deliberately separated six-part layout gives A* room to route the
    # 14 cm plate around three distractors.  The old right-most-centres path
    # neither encoded the target set nor produced the required yaw turns.
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=uniform", "task.target_count=3",
    ])
    result = run_expert_episode(cfg, seed=3010, collect_observations=False)
    assert result.success
    assert result.total == 6
    assert result.collected == result.target_collected == result.target_count == 3
    assert result.unexpected_collected == 0
    assert not any(row["phase"] == "recovery" for row in result.trace)

    env = SweepEnv(cfg, seed=3010)
    try:
        env.reset(seed=3010)
        points = expert_waypoints(env, cfg)
        plan = plan_expert_sweep(env, cfg)
        assert plan.strategy == "capture_lane"
        assert 3 <= len(points) <= 5
        sampled = sample_polyline(
            points[1:], float(cfg.act.action_hz), float(cfg.controller.sweep_speed),
            initial_yaw=0.0, yaw_rate=float(cfg.controller.yaw_speed_limit), accel=0.8,
        )
        assert np.all(np.isfinite(sampled))
    finally:
        env.close()


def test_astar_shortcuts_open_grid_staircase_without_clearance_loss():
    """A clear route should not retain one waypoint per grid cell."""
    cfg = load_config(overrides=["sim.real_time=false"])
    path = _grid_astar(np.array([0.40, -0.20]), np.array([-0.30, 0.20]), [], cfg)
    assert path is not None
    assert len(path) == 2
    np.testing.assert_allclose(path[[0, -1]], [[0.40, -0.20], [-0.30, 0.20]])


def test_sample_polyline_keeps_moving_through_internal_corner():
    """Independent stop-start profiles create a near-zero step at every turn."""
    points = np.array([[0.0, 0.0], [0.20, 0.0], [0.20, 0.20]])
    sampled = sample_polyline(points, hz=25.0, speed=0.12,
                              initial_yaw=0.0, accel=0.8)
    steps = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1)
    # The final sample may be the only zero-velocity endpoint; the internal
    # corner must not be synthesized as a complete stop and restart.
    assert float(np.min(steps[:-1])) > 5e-4


def test_sample_polyline_accepts_a_single_point_probe():
    """A safe no-sweep probe must not abort a batch generation request."""
    from sim.act.expert import sample_polyline

    sampled = sample_polyline(np.array([[0.2, -0.1]]), hz=25.0, speed=0.12)
    assert sampled.shape == (1, 4)
    np.testing.assert_allclose(sampled[0, :2], [0.2, -0.1])


def test_infeasible_expert_plan_stops_safely_instead_of_following_fallback():
    """A* feasibility is honoured; only an infeasible case gets a safe probe."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=cluster", "task.target_count=1",
    ])
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        plan = plan_expert_sweep(env, cfg)
        if plan.feasible:
            assert plan.strategy == "capture_lane"
            assert len(plan.target_indices) == 1
            assert plan.waypoints.shape[0] >= 2
        else:
            assert plan.failure_reason == "astar_no_feasible_path"
            assert plan.waypoints.shape[0] == 2
            assert np.ptp(plan.waypoints[:, 1]) == pytest.approx(0.0, abs=1e-9)
            positions = np.asarray(env.component_positions(), dtype=float)[:, :2]
            assert np.min(np.linalg.norm(
                plan.waypoints[:, None, :] - positions[None, :, :], axis=2)) > 0.04
    finally:
        env.close()


def test_all_six_expert_plan_allows_contact_with_future_target_parts():
    """Future targets are not distractors when the requested set is all six."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=uniform", "task.target_count=6",
    ])
    env = SweepEnv(cfg, seed=4105)
    try:
        env.reset(seed=4105)
        planned = _plan_for_order(env, cfg, tuple(range(6)))
        assert planned is not None
        assert planned[1] == "point_visit"
        plan = plan_expert_sweep(env, cfg)
        assert not plan.feasible
        assert len(plan.target_indices) == 6
    finally:
        env.close()


def test_expert_prefers_one_capture_lane_for_a_clustered_target_set():
    """A selected object must stay under one continuous delivery stroke."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=uniform", "task.target_count=3",
    ])
    env = SweepEnv(cfg, seed=4102)
    try:
        env.reset(seed=4102)
        plan = plan_expert_sweep(env, cfg)
        assert plan.feasible
        assert plan.strategy == "capture_lane"
        assert plan.target_indices.tolist() == [5, 4, 1]
        # The delivery path should not leave the first target to go and visit
        # a different target centre; its longest connected stroke is the one
        # carrying all selected parts towards the tray.
        assert len(plan.waypoints) <= 5
    finally:
        env.close()


def test_expert_rejects_point_visit_route_as_a_success_demonstration():
    """Visiting centres without carrying them to the tray is not one-pass."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=uniform", "task.target_count=4",
    ])
    env = SweepEnv(cfg, seed=4103)
    try:
        env.reset(seed=4103)
        plan = plan_expert_sweep(env, cfg)
        if plan.feasible:
            assert plan.strategy == "capture_lane"
            assert len(plan.target_indices) == 4
        else:
            assert plan.strategy == "no_capture_lane"
            assert plan.failure_reason == "no_single_capture_lane"
            assert plan.waypoints.shape == (2, 2)
    finally:
        env.close()


def test_target_four_cluster_has_a_continuous_capture_lane():
    """The compact seed 5203 layout should not be rejected by early tray alignment."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=cluster", "task.target_count=4",
    ])
    env = SweepEnv(cfg, seed=5203)
    try:
        env.reset(seed=5203)
        plan = plan_expert_sweep(env, cfg)
        assert plan.feasible
        assert plan.strategy == "capture_lane"
        assert len(plan.target_indices) == 4
    finally:
        env.close()


def test_stall_failure_biases_the_tray_entry_without_driving_the_brush_into_wall():
    """Side-wall failures bias the final object lane, not the tool body."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6", "task.target_count=4",
    ])
    plan = ExpertPlan(
        target_indices=np.array([0, 1, 2, 3]),
            waypoints=np.array([
                [0.42, 0.0], [0.12, 0.04], [-0.02, 0.04],
                    [-0.32, 0.09], [-0.40, 0.09],
            ]),
        feasible=True,
    )
    path, phases = _expert_execution_path(
        cfg, plan, failure_mode="stall_outside_tray", failure_seed=2)
    sweep = path[np.asarray(phases) == "sweep"]
    inside_tray = sweep[sweep[:, 0] <= float(cfg.target.x_max)]
    assert len(inside_tray) > 0
    # The final part of the original capture stroke must still be present;
    # only the delivery into the tray is allowed to change lanes.
    capture_tail = np.asarray([-0.02, 0.04])
    assert np.min(np.linalg.norm(sweep[:, :2] - capture_tail, axis=1)) < 0.01
    # Every point crossing the real tray mouth is on the same virtual-tray
    # lane; there must not be a centre-lane push followed by a wall impact.
    assert np.all(np.abs(inside_tray[:, 1]) > 0.05)
    brush_half = float(cfg.end_effector.brush_width) / 2.0
    max_safe = (float(cfg.target.y_max) - float(cfg.target.wall_thickness)
                - brush_half)
    assert np.max(np.abs(inside_tray[:, 1])) < max_safe


def test_stall_planner_uses_a_virtual_lateral_tray_lane():
    """The wrong lane must be selected before execution, not patched after A*."""
    base = load_config(overrides=[
        "sim.real_time=false", "components.count=6", "task.target_count=4",
        "components.spawn_mode=cluster",
    ])
    normal_env = SweepEnv(base, seed=5203)
    try:
        normal_env.reset(seed=5203)
        normal = plan_expert_sweep(normal_env, base)
        assert normal.feasible
        assert abs(float(normal.waypoints[-1, 1])) < 1e-6
    finally:
        normal_env.close()

    virtual = base.copy()
    virtual.set_path("planner.virtual_tray_y", 0.09)
    virtual_env = SweepEnv(virtual, seed=5203)
    try:
        virtual_env.reset(seed=5203)
        shifted = plan_expert_sweep(virtual_env, virtual)
        assert shifted.feasible
        assert abs(float(shifted.waypoints[-1, 1]) - 0.09) < 0.015
        assert abs(float(shifted.waypoints[-2, 1]) - 0.09) < 0.015
    finally:
        virtual_env.close()


def test_all_six_cluster_can_use_an_oriented_capture_lane():
    """A diagonal brush lane should cover the compact six-part seed."""
    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=cluster", "task.target_count=6",
    ])
    env = SweepEnv(cfg, seed=5205)
    try:
        env.reset(seed=5205)
        plan = plan_expert_sweep(env, cfg)
        assert plan.feasible
        assert plan.strategy == "capture_lane"
        assert len(plan.target_indices) == 6
    finally:
        env.close()


def test_expert_retries_geometric_candidates_in_mujoco_before_labeling_failure():
    """A geometrically shorter lane must not veto a physically successful one."""
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=6",
        "components.spawn_mode=cluster", "task.target_count=1",
    ])
    # 5202 remains a real MuJoCo-in-the-loop retry case under the concentrated
    # centre distribution and the updated tray geometry.
    result = run_expert_episode(cfg, seed=5202, collect_observations=False)
    assert result.success
    assert result.collected == result.target_collected == 1
    assert result.unexpected_collected == 0
    assert result.planner_attempts >= 2
