"""Programmatic MJCF generation for the sweeping scene.

The scene is emitted as an XML string so that every episode can be built from a
*sampled layout* (component count, geometry, mass, friction, poses) while the
rest of the pipeline keeps a stable, model-agnostic interface.

Scene contents
--------------
* a flat table with configurable friction, top surface at ``table.top_z``
* a fixed shallow open-fronted tray (the target region) at the left (-X) edge
* by default, the vendored MuJoCo Menagerie six-joint UR10e with its official
  visual/collision geometry, plus a task-specific brush attached at the
  source model's ``attachment_site``
* a legacy 3-DOF Cartesian end-effector (x/y/z slides, optional yaw hinge)
  whose body origin coincides with the TCP; it remains available for isolated
  controller diagnostics
* N free-floating components
* one fixed high-oblique camera used for the conventional-vision backend

Nothing in this module imports MuJoCo; it only produces text.  That keeps the
model definition testable in environments where MuJoCo is not installed.  The
Menagerie asset bytes are supplied separately by :func:`scene_assets` when the
XML is loaded from a string.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from .geometries import ComponentSpec, make_component_spec
from .ur10_mjcf import add_ur10_actuators, add_ur10_arm, add_ur10_sensors


MENAGERIE_ROOT = Path(__file__).resolve().parent / "assets" / "universal_robots_ur10e"
MENAGERIE_XML = MENAGERIE_ROOT / "ur10e.xml"
UR10_MODEL_TYPES = {"ur10_cb3", "ur10e", "ur10e_menagerie"}


def scene_assets() -> Dict[str, bytes]:
    """Return the vendored MuJoCo Menagerie assets for ``from_xml_string``.

    MuJoCo's string loader has no filesystem-relative asset directory, so the
    generated scene passes the mesh files as an in-memory virtual file system.
    The returned keys intentionally retain the ``assets/`` prefix used by the
    Menagerie MJCF's ``meshdir`` declaration.
    """
    if not MENAGERIE_ROOT.is_dir():
        return {}
    return {
        str(path.relative_to(MENAGERIE_ROOT)): path.read_bytes()
        for path in MENAGERIE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".obj", ".stl", ".dae", ".png", ".jpg", ".jpeg"}
    }


def _uses_menagerie(cfg) -> bool:
    return str(cfg.end_effector.type) in UR10_MODEL_TYPES


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    clean = {}
    for key, value in attrs.items():
        if value is None:
            continue
        key = key.rstrip("_")
        if isinstance(value, (list, tuple, np.ndarray)):
            clean[key] = _fmt(np.asarray(value).ravel())
        elif isinstance(value, bool):
            clean[key] = "true" if value else "false"
        elif isinstance(value, float):
            clean[key] = f"{value:.6g}"
        else:
            clean[key] = str(value)
    return ET.SubElement(parent, tag, clean)


def camera_xyaxes(pos: Sequence[float], lookat: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """MuJoCo camera ``xyaxes`` (6 numbers) for a camera at ``pos`` facing ``lookat``.

    A MuJoCo camera looks along its local -Z with local +Y up.
    """
    pos = np.asarray(pos, dtype=float)
    lookat = np.asarray(lookat, dtype=float)
    up = np.asarray(up, dtype=float)
    forward = lookat - pos
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("camera position coincides with look-at point")
    forward /= norm
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.concatenate([x_axis, y_axis])


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


# ---------------------------------------------------------------------------
# scene builder
# ---------------------------------------------------------------------------
def build_scene_xml(cfg, layout: List[dict]) -> str:
    """Return the MJCF XML string for one episode.

    Parameters
    ----------
    cfg
        The full :class:`sim.config.Config`.
    layout
        One dict per component with keys ``geometry``, ``x``, ``y``, ``yaw``,
        ``mass``, ``friction``.  Produced by :mod:`sim.environments.layout`.
    """
    table_half = np.asarray(cfg.table.half_size, dtype=float)
    top_z = float(cfg.table.top_z)
    ee_cfg = cfg.end_effector
    # Legacy Cartesian scenes still use the two-tip pusher.  The ACT scene uses
    # the fixed brush branch below and therefore does not require these fields.
    tip_half = np.asarray(ee_cfg.get("tip_half", [0.008, 0.012, 0.030]), dtype=float)
    tip_gap = float(ee_cfg.get("tip_gap", 0.004))

    root = ET.Element("mujoco", {"model": "tabletop_sweep"})
    # The Menagerie meshes are declared with meshdir="assets".  The actual
    # bytes are supplied by ``scene_assets`` when the model is loaded.
    _sub(root, "compiler", angle="radian", autolimits="true",
         meshdir="assets" if _uses_menagerie(cfg) else None)
    option = _sub(
        root,
        "option",
        timestep=float(cfg.sim.physics_dt),
        gravity=cfg.sim.gravity,
        integrator="implicitfast",
        cone="elliptic",
        impratio=10.0,
        iterations=int(cfg.sim.get("solver_iterations", 50)),
    )
    _sub(option, "flag", multiccd="enable")

    visual = _sub(root, "visual")
    off_w = max(int(cfg.perception.camera.width), int(cfg.get_path("video.width", 640)))
    off_h = max(int(cfg.perception.camera.height), int(cfg.get_path("video.height", 480)))
    _sub(visual, "global", offwidth=off_w, offheight=off_h)
    _sub(visual, "quality", shadowsize=2048)

    # ---------------- assets ----------------
    asset = _sub(root, "asset")
    _sub(asset, "texture", name="tex_sky", type="skybox", builtin="gradient",
         rgb1=(0.5, 0.6, 0.7), rgb2=(0.05, 0.07, 0.1), width=256, height=256)
    _sub(asset, "texture", name="tex_table", type="2d", builtin="checker",
         rgb1=(0.28, 0.30, 0.33), rgb2=(0.33, 0.35, 0.38), width=300, height=300)
    _sub(asset, "material", name="mat_table", texture="tex_table", texrepeat=(6, 5),
         specular=0.1, shininess=0.1, reflectance=0.0)
    _sub(asset, "material", name="mat_tray", rgba=(0.20, 0.45, 0.75, 1.0))
    _sub(asset, "material", name="mat_tip", rgba=(0.15, 0.15, 0.18, 1.0))
    _sub(asset, "material", name="mat_brush", rgba=(0.10, 0.20, 0.12, 1.0))

    # Component specs (one per distinct geometry present in the layout).
    specs: Dict[str, ComponentSpec] = {}
    scale = float(cfg.components.size_scale)
    for item in layout:
        name = item["geometry"]
        if name not in specs:
            specs[name] = make_component_spec(name, scale)
    declared_meshes = set()
    for spec in specs.values():
        for mesh_name, verts in spec.meshes.items():
            if mesh_name in declared_meshes:
                continue
            declared_meshes.add(mesh_name)
            _sub(asset, "mesh", name=mesh_name, vertex=np.asarray(verts).ravel())

    # ---------------- defaults ----------------
    default = _sub(root, "default")
    comp_default = ET.SubElement(default, "default", {"class": "component"})
    _sub(comp_default, "geom", condim=4, solref=(0.004, 1.0), solimp=(0.95, 0.99, 0.001),
         margin=0.0, group=0)
    tip_default = ET.SubElement(default, "default", {"class": "tip"})
    _sub(tip_default, "geom", type="box", material="mat_tip", condim=4,
         friction=(0.5, 0.01, 0.0002), solref=(0.004, 1.0), group=0)
    if _uses_menagerie(cfg):
        _add_menagerie_defaults_and_assets(root, default, asset)

    # ---------------- worldbody ----------------
    world = _sub(root, "worldbody")
    _sub(world, "light", name="light0", pos=(0.0, 0.0, 1.6), dir=(0, 0, -1),
         diffuse=(0.8, 0.8, 0.8), specular=(0.2, 0.2, 0.2), castshadow="true")
    _sub(world, "light", name="light1", pos=(0.6, 0.6, 1.0), dir=(-0.5, -0.5, -1),
         diffuse=(0.35, 0.35, 0.35), castshadow="false")

    # The perception camera. Fixed for the whole episode, identical for training
    # and testing -- nothing may move it.
    cam = cfg.perception.camera
    _sub(world, "camera", name="scene_cam", mode="fixed", pos=cam.pos,
         xyaxes=camera_xyaxes(cam.pos, cam.lookat, cam.up), fovy=float(cam.fovy_deg))
    _add_render_cameras(world, cfg)

    # Table: top face at ``top_z``.
    _sub(world, "geom", name="table", type="box",
         size=table_half, pos=(0.0, 0.0, top_z - table_half[2]),
         material="mat_table", friction=cfg.table.friction, condim=4,
         solref=(0.004, 1.0), solimp=(0.95, 0.99, 0.001), group=0)

    _add_target_tray(world, cfg, top_z)
    if _uses_menagerie(cfg):
        _add_menagerie_ur10e(world, cfg)
    elif str(cfg.end_effector.type) == "ur10_cb3":
        # Kept as a compatibility path for old explicitly-selected configs.
        add_ur10_arm(world, cfg)
    else:
        _add_end_effector(world, cfg, tip_half, tip_gap, top_z)

    for index, item in enumerate(layout):
        _add_component(world, index, item, specs[item["geometry"]], top_z)

    # ---------------- actuators ----------------
    actuator = _sub(root, "actuator")
    if _uses_menagerie(cfg):
        _add_menagerie_actuators(actuator)
    elif str(ee_cfg.type) == "ur10_cb3":
        # Kept as a compatibility path for old explicitly-selected configs.
        add_ur10_actuators(actuator, cfg)
    else:
        kp = float(ee_cfg.kp)
        _sub(actuator, "position", name="act_x", joint="ee_x", kp=kp, forcerange=(-300, 300))
        _sub(actuator, "position", name="act_y", joint="ee_y", kp=kp, forcerange=(-300, 300))
        _sub(actuator, "position", name="act_z", joint="ee_z", kp=kp, forcerange=(-300, 300))
        if bool(ee_cfg.yaw_enabled):
            _sub(actuator, "position", name="act_yaw", joint="ee_yaw",
                 kp=float(ee_cfg.yaw_kp), forcerange=(-20, 20))

    # ---------------- sensors ----------------
    sensor = _sub(root, "sensor")
    if _uses_menagerie(cfg):
        _add_ur10_sensors(sensor)
    elif str(ee_cfg.type) == "ur10_cb3":
        # Kept as a compatibility path for old explicitly-selected configs.
        add_ur10_sensors(sensor)
    else:
        _sub(sensor, "force", name="ft_force", site="ft_site")
        _sub(sensor, "torque", name="ft_torque", site="ft_site")
        _sub(sensor, "framepos", name="tcp_pos", objtype="site", objname="tcp_site")
        _sub(sensor, "framelinvel", name="tcp_linvel", objtype="site", objname="tcp_site")

    return ET.tostring(root, encoding="unicode")


def _menagerie_root() -> ET.Element:
    if not MENAGERIE_XML.is_file():
        raise FileNotFoundError(f"vendored MuJoCo Menagerie model is missing: {MENAGERIE_XML}")
    return ET.parse(MENAGERIE_XML).getroot()


def _add_menagerie_defaults_and_assets(root: ET.Element, default: ET.Element,
                                       asset: ET.Element) -> None:
    """Merge the official Menagerie UR10e visual/dynamics declarations."""
    source = _menagerie_root()
    source_default = source.find("default")
    if source_default is not None:
        for child in source_default:
            default.append(deepcopy(child))
    source_asset = source.find("asset")
    if source_asset is not None:
        for child in source_asset:
            asset.append(deepcopy(child))


def _add_menagerie_ur10e(world: ET.Element, cfg) -> None:
    """Append the vendored Google DeepMind Menagerie UR10e body.

    The source robot ends at ``attachment_site``.  The task-specific brush is
    attached in that exact frame, which preserves the source robot kinematics
    and keeps the task geometry independent from the vendor model.
    """
    source = _menagerie_root()
    base_source = source.find("./worldbody/body[@name='base']")
    if base_source is None:
        raise ValueError("Menagerie UR10e model has no base body")
    base = deepcopy(base_source)
    base_pos = cfg.end_effector.get("base_pos", (0.70, 0.0, 0.04))
    base.set("pos", _fmt(base_pos))

    wrist3 = base.find(".//body[@name='wrist_3_link']")
    if wrist3 is None:
        raise ValueError("Menagerie UR10e model has no wrist_3_link body")

    ee = cfg.end_effector
    brush_height = float(ee.brush_height)
    brush_width = float(ee.brush_width)
    brush_depth = float(ee.brush_depth)
    stem_length = float(ee.get("brush_stem_length", 0.15))
    if stem_length < 0.0:
        raise ValueError("brush_stem_length must be non-negative")
    brush_mount_pos = tuple(float(v) for v in ee.get("brush_mount_pos", (0.0, 0.1, 0.0)))
    brush = ET.Element("body", {"name": "tool", "pos": _fmt(brush_mount_pos),
                                  "quat": "-1 1 0 0"})
    _sub(brush, "inertial", pos=(0.0, 0.0, -stem_length / 2.0), mass=float(ee.brush_mass),
         diaginertia=(0.001, 0.001, 0.0004))
    if stem_length > 0.0:
        _sub(brush, "geom", name="brush_stem", type="cylinder",
             size=(0.009, stem_length / 2.0), pos=(0.0, 0.0, -stem_length / 2.0),
             material="mat_tip", friction=(0.65, 0.01, 0.0004), condim=4)
    _sub(brush, "geom", name="brush_head", type="box",
         # The stroke runs along -X, so the long brush dimension must span Y.
         # Keeping width on X would turn the tool into a narrow scraper and
         # make components slide sideways at the brush edge.
         size=(brush_depth / 2.0, brush_width / 2.0, brush_height / 2.0),
         pos=(0.0, 0.0, -(stem_length + brush_height / 2.0)), material="mat_brush",
         friction=(0.65, 0.01, 0.0004), condim=4, margin=0.00005)
    _sub(brush, "site", name="ft_site", pos=(0.0, 0.0, 0.0), size=(0.004,),
         rgba=(1.0, 0.2, 0.2, 0.4))
    # The task TCP is the brush's bottom-centre contact point.  At z=0 the
    # force controller therefore places the brush on the table, while z_home
    # remains a clearance pose above it.
    _sub(brush, "site", name="tcp_site", pos=(0.0, 0.0, -(stem_length + brush_height)), size=(0.003,),
         rgba=(0.1, 1.0, 0.1, 0.4))
    wrist3.append(brush)
    # Eye-in-hand camera: mount it on the final wrist link beside the tool,
    # like a short side adapter on a real UR10.  Keeping it outside the brush
    # body makes the lens follow the wrist joint without inheriting the brush's
    # downward tool rotation and appearing detached from the end effector.
    wrist_camera = ee.get("wrist_camera", {})
    _sub(wrist3, "camera", name="wrist_cam",
         pos=wrist_camera.get("pos", (0.15, -0.04, 0.10)),
         xyaxes=wrist_camera.get("xyaxes", (0.554700, 0.0, -0.832050,
                                               -0.118785, 0.989821, -0.079190)),
         fovy=float(wrist_camera.get("fovy_deg", 58.0)))
    if bool(ee.get("gravity_compensation", True)):
        # The vendor MJCF describes the physical links but deliberately leaves
        # gravity compensation to the robot controller.  The task adapter is
        # position-controlled, so make that controller property explicit for
        # every dynamic link; otherwise the real arm's pose sags while moving
        # to the far lanes and the brush impacts the table.
        for body in base.iter("body"):
            body.set("gravcomp", "1")
    # The real UR10e base is raised above the tabletop in this side-mounted
    # layout.  A fixed pedestal makes that transform explicit instead of
    # leaving the official base mesh visually floating in space.
    base_z = float(base_pos[2])
    table_z = float(cfg.table.top_z)
    if base_z > table_z + 1e-6:
        mount = _sub(world, "body", name="robot_mount")
        _sub(mount, "geom", name="robot_mount_pedestal", type="cylinder",
             size=(0.13, (base_z - table_z) / 2.0),
             pos=(float(base_pos[0]), float(base_pos[1]), (base_z + table_z) / 2.0),
             material="mat_tip", friction=(0.6, 0.01, 0.0002), condim=4)
    world.append(base)


def _add_menagerie_actuators(actuator: ET.Element) -> None:
    source = _menagerie_root()
    source_actuator = source.find("actuator")
    if source_actuator is not None:
        for child in source_actuator:
            actuator.append(deepcopy(child))


def _add_ur10_sensors(sensor: ET.Element) -> None:
    _sub(sensor, "force", name="ft_force", site="ft_site")
    _sub(sensor, "torque", name="ft_torque", site="ft_site")
    _sub(sensor, "framepos", name="tcp_pos", objtype="site", objname="tcp_site")
    _sub(sensor, "framelinvel", name="tcp_linvel", objtype="site", objname="tcp_site")


def _add_render_cameras(world: ET.Element, cfg) -> None:
    """Extra cameras used **only** for recording video.

    They are never read by any perception backend -- `sim.perception` looks up
    ``scene_cam`` by name and nothing else -- so adding view angles cannot leak
    information into the planner or into a dataset.
    """
    extra = cfg.get_path("video.extra_cameras", None)
    if not extra:
        return
    for name, spec in extra.items():
        if str(spec.get("mode", "fixed")) == "targetbody":
            _sub(world, "camera", name=name, mode="targetbody",
                 target=str(spec.get("target", "tool")), pos=spec.pos,
                 fovy=float(spec.fovy_deg))
        else:
            _sub(world, "camera", name=name, mode="fixed", pos=spec.pos,
                 xyaxes=camera_xyaxes(spec.pos, spec.lookat,
                                      spec.get("up", (0.0, 0.0, 1.0))),
                 fovy=float(spec.fovy_deg))


def _add_target_tray(world: ET.Element, cfg, top_z: float) -> None:
    """Three-walled shallow tray, open towards +X so components slide straight in."""
    tgt = cfg.target
    t = float(tgt.wall_thickness)
    h = float(tgt.wall_height)
    x_min, x_max = float(tgt.x_min), float(tgt.x_max)
    y_min, y_max = float(tgt.y_min), float(tgt.y_max)

    body = _sub(world, "body", name="target_tray", pos=(0.0, 0.0, 0.0))
    # Back wall (-X side)
    _sub(body, "geom", name="tray_wall_x", type="box",
         size=(t / 2.0, (y_max - y_min) / 2.0 + t, h / 2.0),
         pos=(x_min - t / 2.0, 0.5 * (y_min + y_max), top_z + h / 2.0),
         material="mat_tray", condim=3, group=0)
    # Side walls
    for tag, y_wall in (("yp", y_max), ("yn", y_min)):
        sign = 1.0 if y_wall > 0 else -1.0
        _sub(body, "geom", name=f"tray_wall_{tag}", type="box",
             size=((x_max - x_min) / 2.0 + t / 2.0, t / 2.0, h / 2.0),
             pos=(0.5 * (x_min + x_max) - t / 2.0, y_wall + sign * t / 2.0, top_z + h / 2.0),
             material="mat_tray", condim=3, group=0)
    # Visual-only floor patch marking the target footprint (no collisions).
    _sub(body, "geom", name="tray_floor_visual", type="box",
         size=((x_max - x_min) / 2.0, (y_max - y_min) / 2.0, 0.0005),
         pos=(0.5 * (x_min + x_max), 0.5 * (y_min + y_max), top_z + 0.0005),
         rgba=(0.20, 0.45, 0.75, 0.35), contype=0, conaffinity=0, group=1)


def _add_end_effector(world: ET.Element, cfg, tip_half, tip_gap: float, top_z: float) -> None:
    """Cartesian 3-DOF end-effector whose body origin IS the TCP."""
    ee_cfg = cfg.end_effector
    ws = cfg.workspace
    kd = float(ee_cfg.kd)
    z_home = float(ee_cfg.z_home)

    gravcomp = 1.0 if bool(ee_cfg.get("gravity_compensation", True)) else 0.0
    # The EE body sits at the world origin (NOT at the table top) so that the slide
    # joint values are world coordinates.  Task-space commands, the measured TCP and
    # the workspace limits are then all in the same frame even when the table height
    # is perturbed by domain randomisation.
    ee = _sub(world, "body", name="ee", pos=(0.0, 0.0, 0.0), gravcomp=gravcomp)
    _sub(ee, "inertial", pos=(0.0, 0.0, 0.10), mass=float(ee_cfg.mass),
         diaginertia=(0.01, 0.01, 0.005))
    _sub(ee, "joint", name="ee_x", type="slide", axis=(1, 0, 0),
         range=(float(ws.safe_x_min), float(ws.safe_x_max)), damping=kd, armature=0.05)
    _sub(ee, "joint", name="ee_y", type="slide", axis=(0, 1, 0),
         range=(float(ws.safe_y_min), float(ws.safe_y_max)), damping=kd, armature=0.05)
    _sub(ee, "joint", name="ee_z", type="slide", axis=(0, 0, 1),
         range=(top_z - 0.03, top_z + max(0.30, z_home + 0.05)), damping=kd, armature=0.05)
    if bool(ee_cfg.yaw_enabled):
        _sub(ee, "joint", name="ee_yaw", type="hinge", axis=(0, 0, 1),
             range=(-1.5708, 1.5708), damping=float(ee_cfg.yaw_kd), armature=0.005)

    # Welded tool body: the force/torque sensor measures the wrench transmitted
    # through this weld, i.e. a simulated wrist F/T sensor at the fingertip frame.
    tool = _sub(ee, "body", name="tool", pos=(0.0, 0.0, 0.0), gravcomp=gravcomp)
    _sub(tool, "inertial", pos=(0.0, 0.0, 0.03), mass=0.25,
         diaginertia=(0.0008, 0.0008, 0.0004))
    _sub(tool, "site", name="ft_site", pos=(0.0, 0.0, 0.0), size=(0.004,),
         rgba=(1.0, 0.2, 0.2, 0.4))
    _sub(tool, "site", name="tcp_site", pos=(0.0, 0.0, 0.0), size=(0.003,),
         rgba=(0.1, 1.0, 0.1, 0.4))
    offset_y = tip_gap / 2.0 + tip_half[1]
    for tag, sign in (("left", +1.0), ("right", -1.0)):
        ET.SubElement(
            tool,
            "geom",
            {
                "name": f"tip_{tag}",
                "class": "tip",
                "size": _fmt(tip_half),
                "pos": _fmt((0.0, sign * offset_y, tip_half[2])),
            },
        )
    # Visual-only shank so renders look like a closed two-finger gripper.
    _sub(tool, "geom", name="gripper_shank", type="cylinder", size=(0.014, 0.045),
         pos=(0.0, 0.0, 2.0 * tip_half[2] + 0.045), rgba=(0.25, 0.25, 0.30, 1.0),
         contype=0, conaffinity=0, group=1)


def _add_component(world: ET.Element, index: int, item: dict,
                   spec: ComponentSpec, top_z: float) -> None:
    name = f"comp_{index}"
    z = top_z + spec.half_height + 0.0008
    body = _sub(world, "body", name=name, pos=(item["x"], item["y"], z),
                quat=yaw_to_quat(float(item.get("yaw", 0.0))))
    _sub(body, "freejoint", name=f"{name}_free")
    mass_share = float(item["mass"]) / max(1, len(spec.geoms))
    mu = float(item["friction"])
    for gi, geom in enumerate(spec.geoms):
        attrs = {
            "name": f"{name}_g{gi}",
            "class": "component",
            "type": geom.type,
            "pos": _fmt(geom.pos),
            "quat": _fmt(geom.quat),
            "mass": f"{mass_share:.6g}",
            "friction": _fmt((mu, 0.008, 0.0004)),
            "rgba": _fmt(geom.rgba),
        }
        if geom.mesh is not None:
            attrs["mesh"] = geom.mesh
            # Keep the polygon mesh for appearance, but use a simple convex
            # collision proxy for stable dynamic pushing.  This avoids the
            # poor contact response of tiny freejoint mesh prisms while
            # retaining the nut/bolt silhouette in the cameras.
            attrs.update({"mass": "0", "contype": "0", "conaffinity": "0"})
            ET.SubElement(body, "geom", attrs)
            vertices = np.asarray(spec.meshes[geom.mesh], dtype=float)
            radius = float(np.max(np.linalg.norm(vertices[:, :2], axis=1)))
            proxy = {
                "name": f"{name}_g{gi}_collision",
                "type": "cylinder",
                "pos": _fmt(geom.pos),
                "quat": _fmt(geom.quat),
                "size": _fmt((radius, float(spec.half_height))),
                "mass": f"{mass_share:.6g}",
                "friction": _fmt((mu, 0.008, 0.0004)),
                "rgba": _fmt(geom.rgba),
            }
            ET.SubElement(body, "geom", proxy)
            continue
        else:
            attrs["size"] = _fmt(geom.size)
        ET.SubElement(body, "geom", attrs)
