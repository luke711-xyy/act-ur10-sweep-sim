"""A self-contained MuJoCo MJCF description of a UR10/CB3-like arm.

The official CB3 robot is represented here by its six revolute joints and the
published UR10 link dimensions.  The visual meshes are intentionally replaced
by capsules and boxes so the simulator stays portable and does not depend on a
ROS workspace.  This is a dynamics and kinematics model for ACT feasibility,
not a claim of calibrated hardware dynamics.
"""

from __future__ import annotations

from math import pi
from typing import Sequence
from xml.etree import ElementTree as ET

import numpy as np


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.7g}" for v in values)


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    clean = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (tuple, list, np.ndarray)):
            clean[key] = _fmt(np.asarray(value).ravel())
        elif isinstance(value, bool):
            clean[key] = "true" if value else "false"
        else:
            clean[key] = str(value)
    return ET.SubElement(parent, tag, clean)


def _link_body(parent: ET.Element, name: str, pos, joint_axis, length: float,
               qrange=(-2.0 * pi, 2.0 * pi), kp=120.0, kd=12.0) -> ET.Element:
    body = _sub(parent, "body", name=name, pos=pos)
    _sub(body, "joint", name=f"{name}_joint", type="hinge", axis=joint_axis,
         range=qrange, damping=kd, armature=0.08)
    # Lightened visual link inertias keep this portable feasibility model
    # responsive to 100 Hz joint-position targets; the brush mass remains the
    # task-relevant contact inertia.
    _sub(body, "inertial", pos=(0.0, 0.0, length / 2.0), mass=0.25,
         diaginertia=(0.003, 0.003, 0.001))
    _sub(body, "geom", name=f"{name}_visual", type="capsule",
         fromto=(0.0, 0.0, 0.0, 0.0, 0.0, length), size=(0.045,),
         rgba=(0.42, 0.44, 0.48, 1.0), contype=0, conaffinity=0)
    return body


def add_ur10_arm(world: ET.Element, cfg) -> None:
    """Append the UR10 arm, fixed brush, TCP and wrist sensor sites."""
    ee = cfg.end_effector
    base = _sub(world, "body", name="ur10_base", pos=(0.22, 0.0, 0.04))
    _sub(base, "geom", name="ur10_base_geom", type="cylinder", size=(0.11, 0.04),
         rgba=(0.25, 0.27, 0.30, 1.0), contype=0, conaffinity=0)

    # The nested chain uses the standard UR10 dimensional envelope.  Joint axes
    # alternate to expose all six task-space degrees of freedom to IK.
    shoulder = _link_body(base, "ur10_shoulder", (0.0, 0.0, 0.10), (0, 0, 1), 0.1273)
    upper = _link_body(shoulder, "ur10_upper_arm", (0.0, 0.0, 0.1273), (0, 1, 0), 0.612)
    fore = _link_body(upper, "ur10_forearm", (0.0, 0.0, 0.612), (0, 1, 0), 0.5723)
    wrist1 = _link_body(fore, "ur10_wrist_1", (0.0, 0.0, 0.5723), (1, 0, 0), 0.1639)
    wrist2 = _link_body(wrist1, "ur10_wrist_2", (0.0, 0.0, 0.1639), (0, 1, 0), 0.1157)
    wrist3 = _link_body(wrist2, "ur10_wrist_3", (0.0, 0.0, 0.1157), (1, 0, 0), 0.0922)

    tool = _sub(wrist3, "body", name="tool", pos=(0.0, 0.0, 0.0922),
                gravcomp=1.0 if bool(ee.gravity_compensation) else 0.0)
    _sub(tool, "inertial", pos=(0.0, 0.0, -0.01), mass=float(ee.brush_mass),
         diaginertia=(0.001, 0.001, 0.0004))
    # The task sweeps along -X.  The brush's broad face must therefore span Y
    # while its thin dimension points along X; the early V4 implementation had
    # these two dimensions reversed, so the pusher contacted only a narrow
    # 2.4-cm strip and could not carry the sampled parts into the tray.
    brush_size = (float(ee.brush_depth) / 2.0, float(ee.brush_width) / 2.0,
                  float(ee.brush_height) / 2.0)
    _sub(tool, "geom", name="brush_head", type="box", size=brush_size,
         pos=(0.0, 0.0, -brush_size[2]), material="mat_brush",
         friction=(0.65, 0.01, 0.0004), condim=4)
    _sub(tool, "site", name="ft_site", pos=(0.0, 0.0, -brush_size[2]),
         size=(0.004,), rgba=(1.0, 0.2, 0.2, 0.4))
    _sub(tool, "site", name="tcp_site", pos=(0.0, 0.0, -2.0 * brush_size[2]),
         size=(0.003,), rgba=(0.1, 1.0, 0.1, 0.4))
    # Body-attached camera used by ACT.  Its local frame is fixed to the tool,
    # so the wrist view follows the brush during the sweep.
    _sub(tool, "camera", name="wrist_cam", pos=(0.0, -0.16, 0.09),
         xyaxes=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0), fovy=58.0)


def add_ur10_actuators(actuator: ET.Element, cfg) -> None:
    kp = float(cfg.end_effector.joint_kp)
    joints = ["ur10_shoulder_joint", "ur10_upper_arm_joint", "ur10_forearm_joint",
              "ur10_wrist_1_joint", "ur10_wrist_2_joint", "ur10_wrist_3_joint"]
    for i, joint in enumerate(joints, 1):
        _sub(actuator, "position", name=f"ur10_act_{i}", joint=joint, kp=kp,
             kv=float(cfg.end_effector.joint_kd), ctrlrange=(-2.0 * pi, 2.0 * pi),
             forcerange=(-220.0, 220.0))


def add_ur10_sensors(sensor: ET.Element) -> None:
    _sub(sensor, "force", name="ft_force", site="ft_site")
    _sub(sensor, "torque", name="ft_torque", site="ft_site")
    _sub(sensor, "framepos", name="tcp_pos", objtype="site", objname="tcp_site")
    _sub(sensor, "framelinvel", name="tcp_linvel", objtype="site", objname="tcp_site")
