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
from sim.act.expert import expert_waypoints


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


def test_wrist_camera_sees_the_brush_thin_edge_as_a_vertical_divider():
    cfg = load_config()
    xml = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(0)))
    root = ET.fromstring(xml)
    camera = root.find(".//body[@name='wrist_3_link']/camera[@name='wrist_cam']")
    assert camera is not None
    axes = np.fromstring(camera.get("xyaxes", ""), sep=" ")
    camera_x, camera_y = axes[:3], axes[3:]
    forward = -np.cross(camera_x, camera_y)

    # In the wrist_3_link frame, the plate width points along local Z and its
    # vertical height points along local Y.  Looking mostly along the width
    # collapses the broad face to a thin edge; aligning height with image Y
    # makes that edge divide the image into left and right halves.
    plate_width_axis = np.array([0.0, 0.0, 1.0])
    plate_height_axis = np.array([0.0, 1.0, 0.0])
    assert abs(float(np.dot(forward, plate_width_axis))) > 0.80
    assert abs(float(np.dot(camera_x, plate_height_axis))) < 0.10
    assert abs(float(np.dot(camera_y, plate_height_axis))) > 0.75


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
        assert observation["observation.state"].shape == (26,)
        assert observation["observation.environment_state"].shape == (2,)
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
    values = np.arange(80, dtype=np.float32).reshape(20, 4)
    assert scheduler.accept(0.0, values, 0.199)
    assert scheduler.action_for(0.199) is None
    np.testing.assert_array_equal(scheduler.action_for(0.2), values[4])


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


def test_expert_waypoints_stop_inside_ur10e_command_workspace():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        points = expert_waypoints(env, cfg)
        assert float(points[:, 0].min()) >= float(cfg.workspace.x_min)
        assert float(points[-1, 0]) < float(cfg.target.x_max)
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


def test_expert_finishes_last_object_with_a_horizontal_sweep():
    cfg = load_config()
    env = SweepEnv(cfg, seed=12345)
    try:
        env.reset(seed=12345)
        points = expert_waypoints(env, cfg)
        positions = np.asarray(env.component_positions())[:, :2]
        last = positions[np.argmin(positions[:, 0])]
        # The last object must stay under the brush until it reaches the tray
        # mouth; a diagonal move changes y at the same time and can peel the
        # brush off an object near the edge of its transverse footprint.
        tray_i = np.flatnonzero(points[:, 0] <= float(cfg.target.x_max))[0]
        assert points[tray_i, 1] == pytest.approx(last[1], abs=1e-9)
        assert points[tray_i, 0] == pytest.approx(
            float(np.clip(float(cfg.target.x_max)
                          - max(0.02, float(cfg.end_effector.brush_depth) / 2.0),
                          float(cfg.workspace.x_min), float(cfg.workspace.x_max))))
    finally:
        env.close()


def test_expert_allows_the_arm_to_settle_at_the_tray_mouth():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1",
        "controller.safe_max_force=100.0", "episode.final_push_time=1.0",
    ])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success


def test_contact_force_loop_never_commands_brush_through_table():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=[
        "sim.real_time=false", "components.count=1",
        "controller.safe_max_force=100.0",
    ])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    contact_rows = [row for row in result.trace if row["contact"]]
    assert contact_rows
    assert min(float(row["command"][2]) for row in contact_rows) >= float(cfg.table.top_z)


def test_default_official_ur10e_demo_uses_calibrated_force_ceiling():
    from sim.act.rollout import run_expert_episode

    cfg = load_config(overrides=["sim.real_time=false", "components.count=1"])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success
    assert max(float(row["normal_force"]) for row in result.trace) < float(cfg.controller.safe_max_force)


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
        "workspace.z_search_start=-0.001", "controller.safe_max_force=100.0",
    ])
    result = run_expert_episode(cfg, seed=12345, collect_observations=False)
    assert result.success
    assert result.collected == result.total == 1
