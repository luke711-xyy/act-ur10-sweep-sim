"""MuJoCo end-effector adapters for the vendored UR10 model variants."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .ee_interface import CartesianEndEffector


class UR10CB3EndEffector(CartesianEndEffector):
    """Convert absolute TCP targets to six official UR10e actuator targets.

    The class name is retained for configuration/API compatibility with the
    earlier CB3 prototype; the active MJCF is the Menagerie UR10e model.
    """

    JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    ACTUATOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

    def __init__(self, model, data, cfg):
        import mujoco

        self._mujoco = mujoco
        self.model, self.data, self.cfg = model, data, cfg
        self.yaw_enabled = True
        if str(cfg.end_effector.type) == "ur10_cb3_rudra":
            self.joint_names = tuple(f"joint_{index}" for index in range(6))
            self.actuator_names = tuple(f"joint_{index}_act" for index in range(6))
        else:
            self.joint_names = self.JOINT_NAMES
            self.actuator_names = self.ACTUATOR_NAMES
        self.jnt_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                        for n in self.joint_names}
        if any(v < 0 for v in self.jnt_ids.values()):
            raise RuntimeError("MuJoCo model is missing one or more UR10 joints")
        self.qpos_adr = {n: int(model.jnt_qposadr[j]) for n, j in self.jnt_ids.items()}
        self.qvel_adr = {n: int(model.jnt_dofadr[j]) for n, j in self.jnt_ids.items()}
        self.act_ids = {name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names}
        if any(v < 0 for v in self.act_ids.values()):
            raise RuntimeError("MuJoCo model is missing one or more UR10 position actuators")
        self.tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        self.ft_sensor_adr = self._sensor_adr("ft_force")
        self.torque_sensor_adr = self._sensor_adr("ft_torque")
        self.ft_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
        if self.ft_sensor_adr is None or self.ft_site_id < 0:
            raise RuntimeError("MuJoCo model is missing the wrist force sensor or its site")
        self.wrist_sign = float(cfg.controller.wrist_ft_sign)
        self._wrist_ft_bias_world_z = 0.0
        self._normal_force_raw = 0.0
        self._normal_force_filtered = 0.0
        self.tip_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                             for name in ("brush_head", "brush_sole")]
        self.normal_geom_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "brush_sole")]
        self.component_geom_ids: set[int] = set()
        self._contact_sign: Optional[float] = None
        self._contact_buffer = np.zeros(6, dtype=float)
        self._normal_force_filtered = 0.0

    def reset(self, tcp_target: Sequence[float]) -> None:
        if str(self.cfg.end_effector.type) == "ur10_cb3_rudra":
            initial = self.cfg.end_effector.get(
                "rudra_initial_joint_positions",
                self.cfg.end_effector.initial_joint_positions,
            )
        else:
            initial = self.cfg.end_effector.initial_joint_positions
        q0 = np.asarray(initial, dtype=float)
        if q0.shape != (6,):
            raise ValueError("initial_joint_positions must contain six values")
        for name, value in zip(self.joint_names, q0):
            self.data.qpos[self.qpos_adr[name]] = value
            self.data.qvel[self.qvel_adr[name]] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self.set_command(float(tcp_target[0]), float(tcp_target[1]), float(tcp_target[2]), 0.0)
        self._mujoco.mj_forward(self.model, self.data)
        self._mujoco.mj_rnePostConstraint(self.model, self.data)
        self.calibrate_wrist_ft()

    def set_command(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        target = np.array([x, y, z, yaw], dtype=float)
        q_current = self.joint_state()
        q_solution = self._solve_ik(target)
        # IK iterations use qpos as a scratch space.  Do not leave the
        # converged solution teleported into the physical state: the actuators
        # must move the arm there.  The step limit also prevents a near-singular
        # least-squares solve from selecting a visually discontinuous branch.
        max_step = float(self.cfg.end_effector.get("ik_max_joint_step", 0.04))
        q = q_current + np.clip(q_solution - q_current, -max_step, max_step)
        for name, value in zip(self.joint_names, q_current):
            self.data.qpos[self.qpos_adr[name]] = value
        self._mujoco.mj_forward(self.model, self.data)
        for name, value in zip(self.actuator_names, q):
            self.data.ctrl[self.act_ids[name]] = float(value)

    def _solve_ik(self, target: np.ndarray) -> np.ndarray:
        mj = self._mujoco
        q = np.array([self.data.qpos[self.qpos_adr[n]] for n in self.joint_names], dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        damping = float(self.cfg.end_effector.ik_damping)
        for _ in range(int(self.cfg.end_effector.ik_iterations)):
            mj.mj_forward(self.model, self.data)
            pos = np.asarray(self.data.site_xpos[self.tcp_site_id], dtype=float)
            mat = np.asarray(self.data.site_xmat[self.tcp_site_id], dtype=float).reshape(3, 3)
            # Keep the fixed brush face level with the table while exposing
            # yaw as the fourth task command.  Position+yaw-only IK leaves
            # roll/pitch unconstrained and can make the brush edge hit the
            # table before the TCP reaches the search height.
            cy, sy = np.cos(target[3]), np.sin(target[3])
            desired = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
            # World-frame rotation vector from current tool orientation to the
            # desired level orientation.  Unlike a summed axis cross-product,
            # this remains well behaved for the large initial tilt of the
            # simplified arm.
            rot_error = Rotation.from_matrix(desired @ mat.T).as_rotvec()
            error = np.concatenate((target[:3] - pos, rot_error))
            if np.linalg.norm(error) < 1e-4:
                break
            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_site_id)
            cols = [self.qvel_adr[n] for n in self.joint_names]
            jac = np.vstack((jacp[:, cols], jacr[:, cols]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), error)
            q += np.clip(dq, -0.12, 0.12)
            for i, name in enumerate(self.joint_names):
                lo, hi = self.model.jnt_range[self.jnt_ids[name]]
                q[i] = float(np.clip(q[i], lo, hi))
            for i, name in enumerate(self.joint_names):
                self.data.qpos[self.qpos_adr[name]] = q[i]
        mj.mj_forward(self.model, self.data)
        final_pos = np.asarray(
            self.data.site_xpos[self.tcp_site_id], dtype=float
        )
        final_mat = np.asarray(
            self.data.site_xmat[self.tcp_site_id], dtype=float
        ).reshape(3, 3)
        cy, sy = np.cos(target[3]), np.sin(target[3])
        desired = np.array(
            ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)),
            dtype=float,
        )
        position_error = float(np.linalg.norm(target[:3] - final_pos))
        orientation_error = float(np.linalg.norm(
            Rotation.from_matrix(desired @ final_mat.T).as_rotvec()
        ))
        if (not np.isfinite(position_error)
                or not np.isfinite(orientation_error)
                or position_error > float(self.cfg.end_effector.get(
                    "ik_position_tolerance", 0.005))
                or orientation_error > float(self.cfg.end_effector.get(
                    "ik_orientation_tolerance", 0.02))):
            raise RuntimeError(
                "UR10 IK target is unreachable: "
                f"position residual={position_error:.6f} m, "
                f"orientation residual={orientation_error:.6f} rad"
            )
        return q

    def tcp_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.tcp_site_id], dtype=float)

    def tcp_velocity(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_site_id)
        return jacp @ np.asarray(self.data.qvel, dtype=float)

    def tcp_yaw(self) -> float:
        mat = np.asarray(self.data.site_xmat[self.tcp_site_id], dtype=float).reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def wrench(self) -> np.ndarray:
        """Return the raw 6D wrist sensor wrench, rotated into world axes."""
        return self.wrist_ft_world()

    def _external_wrench(self) -> np.ndarray:
        """External contact wrench on the tool, expressed in world axes.

        MuJoCo's site force sensor also contains inertial reactions during
        aggressive joint tracking.  The control channel is the simulated
        single-axis wrist normal-force readout, so it uses the external
        contact wrench; the raw 6D sensor remains available through
        :meth:`wrench` for diagnostics.
        """
        cf = np.asarray(self.data.cfrc_ext[self.tool_body_id], dtype=float)
        # MuJoCo stores cfrc_ext as [force_xyz, torque_xyz].
        return np.concatenate((cf[:3], cf[3:6]))

    def wrist_ft(self) -> np.ndarray:
        if self.ft_sensor_adr is None:
            return np.zeros(6, dtype=float)
        f = np.array(self.data.sensordata[self.ft_sensor_adr:self.ft_sensor_adr + 3], dtype=float)
        t = (np.array(self.data.sensordata[self.torque_sensor_adr:self.torque_sensor_adr + 3], dtype=float)
             if self.torque_sensor_adr is not None else np.zeros(3))
        return np.concatenate((f, t))

    def wrist_ft_world(self) -> np.ndarray:
        raw = self.wrist_ft()
        rotation = np.asarray(
            self.data.site_xmat[self.ft_site_id], dtype=float
        ).reshape(3, 3)
        return np.concatenate((rotation @ raw[:3], rotation @ raw[3:]))

    def calibrate_wrist_ft(self) -> None:
        """Tare world-Z at the reset pose; later Z force stays sensor-derived."""
        self._wrist_ft_bias_world_z = float(self.wrist_ft_world()[2])
        self._normal_force_raw = 0.0
        self._normal_force_filtered = 0.0

    def raw_normal_force(self) -> float:
        """Unfiltered, reset-tared wrist F/T world-Z; pressing is positive."""
        delta = float(self.wrist_ft_world()[2]) - self._wrist_ft_bias_world_z
        return max(0.0, self.wrist_sign * delta)

    def update_normal_force(self, dt: float) -> float:
        """Sample and low-pass wrist force once per control tick."""
        raw = float(self.raw_normal_force())
        self._normal_force_raw = raw
        cutoff = max(0.0, float(self.cfg.controller.force_filter_cutoff_hz))
        alpha = 1.0 - float(np.exp(-2.0 * np.pi * cutoff * max(float(dt), 0.0)))
        self._normal_force_filtered += alpha * (raw - self._normal_force_filtered)
        return float(self._normal_force_filtered)

    def normal_force(self) -> float:
        """Filtered, reset-tared wrist-sensor world-Z load; pressing is positive."""
        return float(self._normal_force_filtered)

    def normal_force_observation(self) -> float:
        """Return the controller's latest filtered sample without updating it."""
        return float(self._normal_force_filtered)

    def joint_state(self) -> np.ndarray:
        return np.array([self.data.qpos[self.qpos_adr[n]] for n in self.joint_names], dtype=float)


class UR10eEndEffector(UR10CB3EndEffector):
    """Compatibility alias for the same MuJoCo task-space adapter."""
