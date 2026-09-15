"""End-effector abstraction.

The planner and the hybrid controller only ever talk to an
:class:`EndEffectorInterface`.  Everything they need is expressed in *task
space*: a Cartesian TCP pose command, the measured TCP pose/velocity, and the
contact wrench at the fingertip frame.

Two implementations are foreseen:

* :class:`CartesianEndEffector` -- the simplified 3-DOF (x/y/z slide) +
  optional yaw prototype used in this first version.  Position actuators track
  the commanded TCP directly.
* :class:`UR10eEndEffector`     -- six-joint UR10e adapter for the vendored
  MuJoCo Menagerie model.  It converts the same task-space command into joint
  commands and reads the same wrench.  No planner or controller code changes
  when the model is swapped.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class EndEffectorInterface:
    """Task-space interface shared by every end-effector implementation."""

    #: number of actuated task-space DOF exposed to the controller
    n_task_dof: int = 4

    def reset(self, tcp_target: Sequence[float]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def set_command(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        raise NotImplementedError

    def tcp_position(self) -> np.ndarray:
        raise NotImplementedError

    def tcp_velocity(self) -> np.ndarray:
        raise NotImplementedError

    def tcp_yaw(self) -> float:
        raise NotImplementedError

    def wrench(self) -> np.ndarray:
        """6-vector ``[fx, fy, fz, tx, ty, tz]`` acting on the tool, world frame.

        Every implementation MUST provide this: the full wrench is the project's
        primary sensing channel, not an optional extra.  A zero wrench is a
        legitimate reading (no contact); a *missing* wrench is a bug, and the
        episode runner refuses to run without one.
        """
        raise NotImplementedError

    def contact_breakdown(self):
        """``(wrench_parts, n_part_contacts)`` -- simulator ground truth.

        The share of the tool wrench that comes from tip-**component** contacts,
        separated from tip-table contacts.  This decomposition does not exist on
        hardware; it is here so that a contact-phase or material classifier
        trained on the *total* wrench can be checked against what the parts
        actually contributed.  Never feed it to a policy.
        """
        return np.zeros(6), 0

    def normal_force(self) -> float:
        """Contact normal force in the *pressing-positive* convention [N]."""
        raise NotImplementedError

    def joint_state(self) -> np.ndarray:
        raise NotImplementedError

    def gripper_state(self) -> float:
        """0 = fully open, 1 = fully closed.  This prototype is always closed."""
        return 1.0


class CartesianEndEffector(EndEffectorInterface):
    """3-DOF Cartesian prototype backed by MuJoCo slide joints."""

    def __init__(self, model, data, cfg):
        import mujoco  # local import: MuJoCo is only needed for the simulator

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.cfg = cfg
        self.yaw_enabled = bool(cfg.end_effector.yaw_enabled)

        mj = mujoco
        self.jnt_ids = {
            name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            for name in ("ee_x", "ee_y", "ee_z")
        }
        if self.yaw_enabled:
            self.jnt_ids["ee_yaw"] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ee_yaw")
        self.qpos_adr = {k: model.jnt_qposadr[v] for k, v in self.jnt_ids.items()}
        self.qvel_adr = {k: model.jnt_dofadr[v] for k, v in self.jnt_ids.items()}

        self.act_ids = {
            name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (("act_x", "act_y", "act_z") + (("act_yaw",) if self.yaw_enabled else ()))
        }
        self.tool_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "tool")
        self.tcp_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "tcp_site")
        self.ft_sensor_adr = self._sensor_adr("ft_force")
        self.torque_sensor_adr = self._sensor_adr("ft_torque")
        self.tip_geom_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in ("tip_left", "tip_right")
        ]
        self.force_backend = str(cfg.controller.force_sensor)
        self.wrist_sign = float(cfg.controller.wrist_ft_sign)
        self.component_geom_ids: set = set()
        self._contact_sign: Optional[float] = None
        self._contact_buffer = np.zeros(6, dtype=float)

    def set_component_geoms(self, geom_ids) -> None:
        """Tell the sensor which geoms are components (for the force breakdown)."""
        self.component_geom_ids = {int(g) for g in geom_ids}

    def _sensor_adr(self, name: str) -> Optional[int]:
        mj = self._mujoco
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
        return None if sid < 0 else int(self.model.sensor_adr[sid])

    # -- commands -----------------------------------------------------------
    def reset(self, tcp_target: Sequence[float]) -> None:
        x, y, z = (float(v) for v in tcp_target[:3])
        self.data.qpos[self.qpos_adr["ee_x"]] = x
        self.data.qpos[self.qpos_adr["ee_y"]] = y
        self.data.qpos[self.qpos_adr["ee_z"]] = z
        for key in self.qvel_adr:
            self.data.qvel[self.qvel_adr[key]] = 0.0
        if self.yaw_enabled:
            self.data.qpos[self.qpos_adr["ee_yaw"]] = 0.0
        self.set_command(x, y, z, 0.0)

    def set_command(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        self.data.ctrl[self.act_ids["act_x"]] = float(x)
        self.data.ctrl[self.act_ids["act_y"]] = float(y)
        self.data.ctrl[self.act_ids["act_z"]] = float(z)
        if self.yaw_enabled:
            self.data.ctrl[self.act_ids["act_yaw"]] = float(yaw)

    # -- measurements -------------------------------------------------------
    def tcp_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.tcp_site_id], dtype=float)

    def tcp_velocity(self) -> np.ndarray:
        return np.array(
            [self.data.qvel[self.qvel_adr["ee_x"]],
             self.data.qvel[self.qvel_adr["ee_y"]],
             self.data.qvel[self.qvel_adr["ee_z"]]],
            dtype=float,
        )

    def tcp_yaw(self) -> float:
        if not self.yaw_enabled:
            return 0.0
        return float(self.data.qpos[self.qpos_adr["ee_yaw"]])

    def wrench(self) -> np.ndarray:
        """External wrench on the tool body, world frame (from ``cfrc_ext``)."""
        cf = np.array(self.data.cfrc_ext[self.tool_body_id], dtype=float)
        # MuJoCo stores [torque(3), force(3)] in the com-based frame.
        return np.concatenate([cf[3:6], cf[0:3]])

    def wrist_ft(self) -> np.ndarray:
        """Raw wrist force/torque sensor reading, site frame."""
        if self.ft_sensor_adr is None:
            return np.zeros(6)
        f = np.array(self.data.sensordata[self.ft_sensor_adr:self.ft_sensor_adr + 3], dtype=float)
        if self.torque_sensor_adr is None:
            return np.concatenate([f, np.zeros(3)])
        tq = np.array(
            self.data.sensordata[self.torque_sensor_adr:self.torque_sensor_adr + 3], dtype=float
        )
        return np.concatenate([f, tq])

    def normal_force(self) -> float:
        if self.force_backend == "wrist_ft":
            value = self.wrist_sign * float(self.wrist_ft()[2])
        else:
            # Total external (contact) force on the tool body; pressing on the
            # table produces an upward reaction, hence +Z is "pressing".
            value = float(self.wrench()[2])
        return max(0.0, value)

    def contact_breakdown(self):
        """Component-contact share of the tool wrench, plus the contact count.

        Walks the active contacts, keeps the ones that involve a gripper tip, and
        sums those whose other geom belongs to a component.

        The sign MuJoCo uses for ``mj_contactForce`` (which of the two bodies the
        returned force acts on) has varied between versions, so it is not assumed:
        it is calibrated once, the first time there is meaningful contact, by
        checking which sign makes the per-contact sum agree with ``cfrc_ext`` --
        which is unambiguous.  After that the breakdown is exact, not approximate.
        """
        mj = self._mujoco
        parts = np.zeros(6)
        table = np.zeros(6)
        n_part_contacts = 0
        tips = set(self.tip_geom_ids)

        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            tip_is_2 = g2 in tips
            if not tip_is_2 and g1 not in tips:
                continue
            mj.mj_contactForce(self.model, self.data, i, self._contact_buffer)
            frame = np.array(contact.frame, dtype=float).reshape(3, 3)
            wrench = np.concatenate([frame.T @ self._contact_buffer[:3],
                                     frame.T @ self._contact_buffer[3:]])
            # Reported force acts on one of the two bodies; flip when the tip is
            # the other one.  The absolute convention is resolved by _calibrate.
            wrench = wrench if tip_is_2 else -wrench
            other = g1 if tip_is_2 else g2
            if other in self.component_geom_ids:
                parts += wrench
                n_part_contacts += 1
            else:
                table += wrench

        sign = self._calibrate_contact_sign(parts[:3] + table[:3])
        return sign * parts, int(n_part_contacts)

    def _calibrate_contact_sign(self, summed_force: np.ndarray) -> float:
        """Resolve mj_contactForce's sign convention against ``cfrc_ext`` once."""
        if self._contact_sign is not None:
            return self._contact_sign
        total = self.wrench()[:3]
        if np.linalg.norm(total) < 0.5 or np.linalg.norm(summed_force) < 0.5:
            return 1.0                       # not enough contact to decide yet
        positive = np.linalg.norm(summed_force - total)
        negative = np.linalg.norm(-summed_force - total)
        self._contact_sign = -1.0 if negative < positive else 1.0
        return self._contact_sign

    def contact_breakdown_residual(self) -> float:
        """||(parts + table) - cfrc_ext|| -- should be ~0; used by the smoke test."""
        parts, _ = self.contact_breakdown()
        total = self.wrench()[:3]
        if np.linalg.norm(total) < 1e-9:
            return 0.0
        sign = self._contact_sign if self._contact_sign is not None else 1.0
        mj = self._mujoco
        summed = np.zeros(3)
        tips = set(self.tip_geom_ids)
        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            tip_is_2 = g2 in tips
            if not tip_is_2 and g1 not in tips:
                continue
            mj.mj_contactForce(self.model, self.data, i, self._contact_buffer)
            frame = np.array(contact.frame, dtype=float).reshape(3, 3)
            force = frame.T @ self._contact_buffer[:3]
            summed += (force if tip_is_2 else -force)
        return float(np.linalg.norm(sign * summed - total) / max(np.linalg.norm(total), 1e-9))

    def joint_state(self) -> np.ndarray:
        keys = ["ee_x", "ee_y", "ee_z"] + (["ee_yaw"] if self.yaw_enabled else [])
        return np.array([self.data.qpos[self.qpos_adr[k]] for k in keys], dtype=float)


class UR10eEndEffector(EndEffectorInterface):  # pragma: no cover - compatibility shim
    """Lazy compatibility wrapper for the concrete UR10e MuJoCo adapter."""

    def __new__(cls, *args, **kwargs):
        # Import lazily to avoid a cycle when ``sim.environments.ur10_interface``
        # itself imports the base interface module.
        from .ur10_interface import UR10eEndEffector as implementation

        return implementation(*args, **kwargs)


def build_end_effector(model, data, cfg) -> EndEffectorInterface:
    kind = str(cfg.end_effector.type)
    if kind in ("ur10_cb3", "ur10e", "ur10e_menagerie"):
        from .ur10_interface import UR10CB3EndEffector

        return UR10CB3EndEffector(model, data, cfg)
    if kind == "cartesian3dof":
        return CartesianEndEffector(model, data, cfg)
    raise ValueError(f"unknown end_effector.type {kind!r}")
