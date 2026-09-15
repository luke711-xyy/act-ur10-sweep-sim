"""Tests for MJCF generation and configuration handling."""

import numpy as np
import pytest
from xml.etree import ElementTree as ET

from sim.config import Config, load_config, save_config
from sim.environments.layout import sample_layout
from sim.model.geometries import GEOMETRY_NAMES, make_component_spec
from sim.model.scene_builder import build_scene_xml, yaw_to_quat


@pytest.fixture
def cfg():
    return load_config()


def build(cfg, count=3, geometry="hex_nut", seed=0):
    cfg.set_path("components.count", count)
    cfg.set_path("components.geometry", geometry)
    layout = sample_layout(cfg, np.random.default_rng(seed))
    return ET.fromstring(build_scene_xml(cfg, layout)), layout


def test_scene_is_well_formed_xml(cfg):
    root, _ = build(cfg)
    assert root.tag == "mujoco"
    assert root.find("worldbody") is not None
    assert root.find("actuator") is not None
    assert root.find("sensor") is not None


def test_scene_has_the_expected_actuators_and_sensors(cfg):
    root, _ = build(cfg)
    actuators = {a.get("name") for a in root.find("actuator")}
    assert {"act_x", "act_y", "act_z", "act_yaw"} <= actuators
    sensors = {s.get("name") for s in root.find("sensor")}
    assert {"ft_force", "ft_torque", "tcp_pos", "tcp_linvel"} <= sensors


def test_end_effector_has_three_slides_and_a_yaw_hinge(cfg):
    root, _ = build(cfg)
    ee = root.find(".//body[@name='ee']")
    joints = {(j.get("name"), j.get("type")) for j in ee.findall("joint")}
    assert ("ee_x", "slide") in joints
    assert ("ee_y", "slide") in joints
    assert ("ee_z", "slide") in joints
    assert ("ee_yaw", "hinge") in joints


def test_two_closed_gripper_tips_with_the_configured_gap(cfg):
    root, _ = build(cfg)
    tool = root.find(".//body[@name='tool']")
    tips = [g for g in tool.findall("geom") if g.get("name", "").startswith("tip_")]
    assert len(tips) == 2
    ys = sorted(float(g.get("pos").split()[1]) for g in tips)
    gap = ys[1] - ys[0] - 2 * float(cfg.end_effector.tip_half[1])
    assert gap == pytest.approx(float(cfg.end_effector.tip_gap), abs=1e-9)
    # the TCP (body origin) sits at the BOTTOM of the tips
    z = float(tips[0].get("pos").split()[2])
    assert z == pytest.approx(float(cfg.end_effector.tip_half[2]))


def test_end_effector_joints_are_in_world_coordinates(cfg):
    """Slide-joint values must equal world coordinates even if the table moves.

    Otherwise every task-space command would be silently offset by the table
    height perturbation.
    """
    cfg.set_path("table.top_z", 0.004)
    root, _ = build(cfg)
    ee = root.find(".//body[@name='ee']")
    assert [float(v) for v in ee.get("pos").split()] == [0.0, 0.0, 0.0]
    lo, hi = (float(v) for v in ee.find("joint[@name='ee_z']").get("range").split())
    assert lo < 0.004 < hi
    table_z = float(root.find(".//geom[@name='table']").get("pos").split()[2])
    half_z = float(cfg.table.half_size[2])
    assert table_z + half_z == pytest.approx(0.004)     # table TOP at top_z


def test_no_grasping_degree_of_freedom_exists(cfg):
    """The gripper is permanently closed: no finger joint or actuator anywhere."""
    root, _ = build(cfg)
    tool = root.find(".//body[@name='tool']")
    assert tool.findall("joint") == []
    names = {a.get("name") for a in root.find("actuator")}
    assert not any("finger" in n or "grip" in n for n in names)


def test_target_tray_is_open_towards_plus_x(cfg):
    root, _ = build(cfg)
    tray = root.find(".//body[@name='target_tray']")
    walls = {g.get("name") for g in tray.findall("geom")}
    assert {"tray_wall_x", "tray_wall_yp", "tray_wall_yn"} <= walls
    assert not any("tray_wall_xp" in name for name in walls)   # no wall on the open side


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
def test_component_count_is_configurable(cfg, count):
    root, layout = build(cfg, count=count)
    bodies = [b for b in root.find("worldbody").findall("body")
              if (b.get("name") or "").startswith("comp_")]
    assert len(bodies) == count == len(layout)
    for body in bodies:
        assert body.find("freejoint") is not None


@pytest.mark.parametrize("geometry", list(GEOMETRY_NAMES) + ["mixed"])
def test_every_geometry_builds(cfg, geometry):
    root, _ = build(cfg, count=4, geometry=geometry)
    geoms = root.findall(".//body[@name='comp_0']/geom")
    assert geoms
    for geom in geoms:
        assert geom.get("mass") is not None
        assert geom.get("friction") is not None


def test_hex_prism_mesh_has_twelve_vertices(cfg):
    spec = make_component_spec("hex_nut", 1.0)
    verts = spec.meshes["mesh_hex_nut"]
    assert verts.shape == (12, 3)
    radii = np.linalg.norm(verts[:, :2], axis=1)
    assert np.allclose(radii, radii[0])
    root, _ = build(cfg, geometry="hex_nut")
    mesh = root.find(".//mesh[@name='mesh_hex_nut']")
    assert mesh is not None and len(mesh.get("vertex").split()) == 36


def test_scene_is_deterministic_for_a_seed(cfg):
    a = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    b = build_scene_xml(cfg, sample_layout(cfg, np.random.default_rng(5)))
    assert a == b


def test_physics_timestep_is_smaller_than_the_control_period(cfg):
    assert float(cfg.sim.physics_dt) < 1.0 / float(cfg.sim.control_hz)
    root, _ = build(cfg)
    assert float(root.find("option").get("timestep")) == pytest.approx(cfg.sim.physics_dt)


def test_yaw_to_quat_is_unit_and_correct():
    q = yaw_to_quat(np.pi / 2)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert q[0] == pytest.approx(np.cos(np.pi / 4))
    assert q[3] == pytest.approx(np.sin(np.pi / 4))


# ------------------------------------------------------------------- config
def test_config_attribute_and_item_access(cfg):
    assert cfg.controller.desired_force == cfg["controller"]["desired_force"]
    assert cfg.get_path("controller.admittance.k") == cfg.controller.admittance.k
    assert cfg.get_path("does.not.exist", 42) == 42


def test_config_overrides_are_typed():
    cfg = load_config(overrides=["controller.desired_force=5.5",
                                 "components.count=7",
                                 "planner.consolidation=false",
                                 "planner.name=visual_greedy"])
    assert cfg.controller.desired_force == 5.5
    assert cfg.components.count == 7
    assert cfg.planner.consolidation is False
    assert cfg.planner.name == "visual_greedy"


def test_config_copy_is_deep(cfg):
    other = cfg.copy()
    other.set_path("controller.desired_force", 99.0)
    assert cfg.controller.desired_force != 99.0


def test_config_round_trips_through_yaml(cfg, tmp_path):
    path = str(tmp_path / "cfg.yaml")
    save_config(cfg, path)
    reloaded = load_config(path)
    a, b = cfg.to_dict(), reloaded.to_dict()
    a.pop("_source_config", None)
    b.pop("_source_config", None)
    assert a == b


def test_invalid_override_is_rejected():
    with pytest.raises(ValueError):
        load_config(overrides=["not-an-assignment"])
