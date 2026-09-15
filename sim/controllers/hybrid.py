"""Hybrid force/position controller.

Axis assignment
---------------
* **XY** -- position controlled.  The command comes from the planner via a
  smooth Cartesian trajectory (:mod:`sim.planners.trajectory`).
* **Z**  -- force regulated.  The command comes from the outer admittance loop
  (:mod:`sim.controllers.admittance`) as ``z_cmd = z_nominal + delta_z``.
* **yaw** -- position controlled, held at the stroke's yaw.

The controller owns the whole per-stroke sequence (APPROACH ... RETRACT); the
environment only advances physics and reports measurements.  This separation is
what allows the simplified Cartesian end-effector to be swapped for a UR10e
later without touching either the controller or the planners.

Safety features implemented here
--------------------------------
* normal-force low-pass filtering and optional sensor noise
* force-target ramping (up and down) with a configurable slew rate
* force-target saturation and a hard measured-force safety abort
* Z position-correction saturation and Z-correction velocity limit
* contact detection threshold with dwell, and contact-loss accounting
* admittance state reset at the start of every sweep
* **release guard**: as soon as the TCP enters the guard band in front of the
  tray entrance / table edge, the desired force is ramped to zero and the tool
  retracts.  The Z loop never keeps searching for contact past the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..planners.base import SweepStroke
from ..planners.trajectory import Trajectory, make_linear_trajectory
from .admittance import AdmittanceController1D, AdmittanceParams
from .filters import DelayBuffer, LowPassFilter, RateLimiter
from .state_machine import CONTACT_PHASES, ContactStateMachine, Phase, Signals


@dataclass
class Command:
    x: float
    y: float
    z: float
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw], dtype=float)


@dataclass
class ControlRecord:
    """One control-period sample of everything worth logging.

    The full 6-axis wrench and the TCP velocity are recorded on *every* step,
    not just the scalar normal force that closes the Z loop: the tangential
    components are what an offline contact-phase or material classifier needs,
    and they are invisible from ``force_filtered`` alone.
    """

    t: float
    stroke: int
    phase: str
    tcp_x: float
    tcp_y: float
    tcp_z: float
    cmd_x: float
    cmd_y: float
    cmd_z: float
    cmd_yaw: float
    z_nominal: float
    delta_z: float
    force_desired: float
    force_raw: float
    force_filtered: float
    in_contact: bool
    # --- full state, added for offline force analysis ---
    tcp_yaw: float = 0.0
    tcp_vx: float = 0.0
    tcp_vy: float = 0.0
    tcp_vz: float = 0.0
    fx: float = 0.0          # external wrench on the tool, world frame
    fy: float = 0.0
    fz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    # --- simulator ground truth: the component-contact share of the wrench ---
    fx_p: float = 0.0
    fy_p: float = 0.0
    fz_p: float = 0.0
    tx_p: float = 0.0
    ty_p: float = 0.0
    tz_p: float = 0.0
    n_part_contacts: int = 0

    @property
    def tangential_force(self) -> float:
        """In-plane force magnitude -- the channel that carries contact events."""
        return float(np.hypot(self.fx, self.fy))

    @property
    def tangential_force_parts(self) -> float:
        """In-plane force from component contacts only (simulator ground truth)."""
        return float(np.hypot(self.fx_p, self.fy_p))


class HybridForcePositionController:
    def __init__(self, cfg, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        c = cfg.controller
        self.dt = 1.0 / float(cfg.sim.control_hz)
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.admittance = AdmittanceController1D(
            AdmittanceParams(float(c.admittance.m), float(c.admittance.b), float(c.admittance.k)),
            dt=self.dt,
            delta_limit=float(c.delta_z_limit),
            rate_limit=float(c.delta_z_rate_limit),
            direction=-1.0,   # inward contact normal points along -Z
        )
        self.force_filter = LowPassFilter(float(c.force_filter_cutoff_hz), self.dt)
        self.force_ramp = RateLimiter(float(c.force_ramp_rate), self.dt)
        self.delay = DelayBuffer(int(c.control_delay_steps))
        self.fsm = ContactStateMachine(contact_dwell_steps=2)

        self.desired_force_target = float(np.clip(c.desired_force, 0.0, float(c.max_force)))
        self.trace: List[ControlRecord] = []
        #: When True, ``step`` refuses to run without a wrench.  The episode
        #: runner sets this; unit tests that drive the controller from a scalar
        #: analytic plant leave it False.
        self.require_full_state = False

        # per-stroke state
        self._stroke: Optional[SweepStroke] = None
        self._approach_traj: Optional[Trajectory] = None
        self._sweep_traj: Optional[Trajectory] = None
        self._retract_traj: Optional[Trajectory] = None
        self._phase_t0 = 0.0
        self._z_nominal = 0.0
        self._z_search = 0.0
        self._hold_xy: Optional[np.ndarray] = None
        self.last_transfer_waypoints: List[np.ndarray] = []
        self._last_cmd = Command(0.0, 0.0, float(cfg.end_effector.z_home), 0.0)
        self._contact_lost_time = 0.0
        self._contact_time_total = 0.0
        self._abort_reason = ""
        self.stroke_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ setup
    def reset(self, tcp: np.ndarray) -> None:
        self.force_filter.reset(0.0)
        self.force_ramp.reset(0.0)
        self.admittance.reset()
        self.delay.reset(np.array([tcp[0], tcp[1], tcp[2], 0.0]))
        self.trace.clear()
        self._last_cmd = Command(float(tcp[0]), float(tcp[1]), float(tcp[2]), 0.0)
        self._contact_lost_time = 0.0
        self._contact_time_total = 0.0

    @property
    def phase(self) -> Phase:
        return self.fsm.phase

    @property
    def stroke_done(self) -> bool:
        return self.fsm.stroke_finished

    # -------------------------------------------------------------- per stroke
    def start_stroke(self, stroke: SweepStroke, tcp: np.ndarray, t: float,
                     obstacles=None) -> None:
        """Begin a stroke.

        ``obstacles`` are planar points (from the *perception* observation, never
        from ground truth) that the contact-free APPROACH should route around
        when ``planner.transfer.mode == "rrt"``.
        """
        ws = self.cfg.workspace
        p = self.cfg.planner
        c = self.cfg.controller
        kind = str(p.interpolation)
        accel = float(p.get("accel", 0.8))

        self._stroke = stroke
        z_travel = float(ws.z_travel)
        self._z_search = float(ws.z_search_start)

        start_xy = stroke.start
        # APPROACH: lift -> contact-free transfer above the stroke start -> descend.
        # Only the middle leg is planned; the vertical legs are always straight.
        from ..planners.rrt import plan_transfer
        from ..planners.trajectory import make_segment

        lift = [float(tcp[0]), float(tcp[1]), z_travel]
        arrive = [float(start_xy[0]), float(start_xy[1]), z_travel]
        transfer = plan_transfer(self.cfg, lift, arrive, self.rng, obstacles)
        self.last_transfer_waypoints = [np.asarray(w, dtype=float) for w in transfer]

        waypoints = [[float(tcp[0]), float(tcp[1]), float(tcp[2])]]
        waypoints += [list(map(float, w)) for w in transfer]
        waypoints += [[float(start_xy[0]), float(start_xy[1]), self._z_search]]
        speeds = ([float(c.z_speed)]
                  + [float(c.travel_speed)] * (len(transfer) - 1)
                  + [float(c.z_speed)])
        segments = [make_segment(waypoints[i], waypoints[i + 1], speeds[i], kind, accel)
                    for i in range(len(waypoints) - 1)]
        self._approach_traj = Trajectory(segments)

        # SWEEP: straight planar stroke at the sweeping speed.
        self._sweep_traj = make_linear_trajectory(
            [stroke.start, stroke.end], float(c.sweep_speed), kind, accel
        )
        self._retract_traj = None
        self._hold_xy = None
        self._abort_reason = ""
        self._contact_lost_time = 0.0
        self._contact_time_total = 0.0
        self.force_ramp.reset(0.0)
        self.admittance.reset()
        self.fsm.begin_stroke(t)
        self._phase_t0 = float(t)
        self.stroke_stats = {
            "contact_established": False,
            "contact_time": 0.0,
            "lost_contact_time": 0.0,
            "peak_force": 0.0,
            "aborted": False,
            "abort_reason": "",
            "path_length": 0.0,
        }

    # --------------------------------------------------------------- main step
    def step(
        self,
        t: float,
        tcp: np.ndarray,
        force_normal_raw: float,
        wrench: Optional[np.ndarray] = None,
        tcp_velocity: Optional[np.ndarray] = None,
        tcp_yaw: float = 0.0,
        wrench_parts: Optional[np.ndarray] = None,
        n_part_contacts: int = 0,
    ) -> Command:
        """Advance one control period and return the joint-space Cartesian command.

        ``wrench`` (6-vector ``[fx, fy, fz, tx, ty, tz]``, world frame),
        ``tcp_velocity`` and ``wrench_parts`` do not enter the control law -- it
        uses ``force_normal_raw`` -- but they are the project's primary sensing
        record, so when ``require_full_state`` is set (as the episode runner
        does) a missing wrench is an error rather than a silent row of zeros.
        """
        if self.require_full_state and wrench is None:
            raise RuntimeError(
                "HybridForcePositionController.step() was called without a wrench while "
                "require_full_state is set. The 6-axis wrench is the primary sensing "
                "channel of this project; recording zeros instead would silently corrupt "
                "every exported demonstration. Give the environment a wrench() method."
            )
        c = self.cfg.controller
        ws = self.cfg.workspace

        # ---- force conditioning ----
        noise_std = float(c.force_noise_std)
        measured = float(force_normal_raw)
        if noise_std > 0.0:
            measured += float(self.rng.normal(0.0, noise_std))
        f_filt = self.force_filter.step(measured)
        self.stroke_stats["peak_force"] = max(self.stroke_stats.get("peak_force", 0.0), f_filt)

        phase_before = self.fsm.phase
        phase_elapsed = float(t) - self._phase_t0

        # ---- guards ----
        release_zone = self._in_release_zone(tcp)
        abort = False
        reason = ""
        if f_filt > float(c.safe_max_force):
            abort, reason = True, f"over-force {f_filt:.1f} N > {float(c.safe_max_force):.1f} N"
        elif not self._inside_safe_box(tcp):
            abort, reason = True, "TCP left the safe workspace"

        signals = Signals(
            t=float(t),
            approach_done=(self._approach_traj is not None
                           and self._approach_traj.is_done(phase_elapsed)),
            search_exhausted=(self._last_cmd.z <= float(ws.get("z_search_min", -0.012))),
            contact=f_filt > float(c.contact_threshold),
            contact_lost=f_filt < float(c.release_threshold),
            ramp_up_done=self.force_ramp.at(self.desired_force_target, tol=1e-2),
            ramp_down_done=(self.force_ramp.at(0.0, tol=1e-2)
                            and f_filt < float(c.release_threshold)
                            and self._sweep_finished(phase_before, phase_elapsed)),
            trajectory_done=(self._sweep_traj is not None
                             and self._sweep_traj.is_done(phase_elapsed)),
            release_zone=release_zone,
            retract_done=(self._retract_traj is not None
                          and self._retract_traj.is_done(phase_elapsed)),
            abort=abort,
            abort_reason=reason,
        )
        phase = self.fsm.update(signals)
        if abort and not self.stroke_stats.get("aborted"):
            self.stroke_stats["aborted"] = True
            self.stroke_stats["abort_reason"] = reason
            self._abort_reason = reason
            self._hold_xy = np.array([tcp[0], tcp[1]], dtype=float)

        if phase is not phase_before:
            self._on_phase_enter(phase, t, tcp)
            phase_elapsed = 0.0

        # ---- desired force schedule ----
        if phase in (Phase.FORCE_RAMP, Phase.SWEEP) and not release_zone:
            f_target = self.desired_force_target
        else:
            f_target = 0.0
        f_des = self.force_ramp.step(float(np.clip(f_target, 0.0, float(c.max_force))))

        # ---- axis commands ----
        cmd = self._axis_commands(phase, phase_elapsed, tcp, f_des, f_filt)

        # ---- contact bookkeeping ----
        in_contact = f_filt > float(c.contact_threshold)
        if phase in CONTACT_PHASES:
            self._contact_time_total += self.dt
            if not in_contact:
                self._contact_lost_time += self.dt
        self.stroke_stats["contact_time"] = self._contact_time_total
        self.stroke_stats["lost_contact_time"] = self._contact_lost_time
        if phase is Phase.SWEEP and in_contact:
            self.stroke_stats["contact_established"] = True

        # ---- workspace clamp, delay, emit ----
        cmd.x = float(np.clip(cmd.x, ws.safe_x_min, ws.safe_x_max))
        cmd.y = float(np.clip(cmd.y, ws.safe_y_min, ws.safe_y_max))
        delayed = self.delay.step(cmd.as_array())
        out = Command(float(delayed[0]), float(delayed[1]), float(delayed[2]), float(delayed[3]))
        self._last_cmd = cmd

        w = np.zeros(6) if wrench is None else np.asarray(wrench, dtype=float).ravel()
        v = np.zeros(3) if tcp_velocity is None else np.asarray(tcp_velocity, dtype=float).ravel()
        wp = np.zeros(6) if wrench_parts is None else np.asarray(wrench_parts, dtype=float).ravel()
        self.trace.append(
            ControlRecord(
                t=float(t), stroke=self.fsm.stroke_index, phase=phase.value,
                tcp_x=float(tcp[0]), tcp_y=float(tcp[1]), tcp_z=float(tcp[2]),
                cmd_x=cmd.x, cmd_y=cmd.y, cmd_z=cmd.z, cmd_yaw=cmd.yaw,
                z_nominal=self._z_nominal, delta_z=self.admittance.delta_z,
                force_desired=float(f_des), force_raw=float(measured),
                force_filtered=float(f_filt), in_contact=bool(in_contact),
                tcp_yaw=float(tcp_yaw),
                tcp_vx=float(v[0]), tcp_vy=float(v[1]), tcp_vz=float(v[2]),
                fx=float(w[0]), fy=float(w[1]), fz=float(w[2]),
                tx=float(w[3]), ty=float(w[4]), tz=float(w[5]),
                fx_p=float(wp[0]), fy_p=float(wp[1]), fz_p=float(wp[2]),
                tx_p=float(wp[3]), ty_p=float(wp[4]), tz_p=float(wp[5]),
                n_part_contacts=int(n_part_contacts),
            )
        )
        return out

    # ------------------------------------------------------------- transitions
    def _on_phase_enter(self, phase: Phase, t: float, tcp: np.ndarray) -> None:
        self._phase_t0 = float(t)
        c = self.cfg.controller
        ws = self.cfg.workspace
        if phase is Phase.SEARCH_CONTACT:
            self._z_search = float(self._last_cmd.z)
        elif phase is Phase.CONTACT_DETECTED:
            # Latch the nominal contact height and clear the admittance state so
            # every sweep starts from a known, zero-correction condition.
            self._z_nominal = float(self._last_cmd.z)
            self.admittance.reset(delta_z=0.0, velocity=0.0)
        elif phase is Phase.RETRACT:
            kind = str(self.cfg.planner.interpolation)
            accel = float(self.cfg.planner.get("accel", 0.8))
            start = [float(tcp[0]), float(tcp[1]), float(self._last_cmd.z)]
            end = [float(tcp[0]), float(tcp[1]), float(ws.z_travel)]
            self._retract_traj = make_linear_trajectory([start, end], float(c.z_speed),
                                                        kind, accel)

    # --------------------------------------------------------------- axis logic
    def _axis_commands(self, phase: Phase, elapsed: float, tcp: np.ndarray,
                       f_des: float, f_filt: float) -> Command:
        c = self.cfg.controller
        ws = self.cfg.workspace
        yaw = float(self._stroke.yaw) if self._stroke is not None else 0.0
        prev = self._last_cmd

        if phase is Phase.APPROACH and self._approach_traj is not None:
            p = self._approach_traj.point(elapsed)
            return Command(float(p[0]), float(p[1]), float(p[2]), yaw)

        if phase is Phase.SEARCH_CONTACT:
            z = prev.z - float(c.search_velocity) * self.dt
            z = max(z, float(ws.get("z_search_min", -0.012)))
            return Command(prev.x, prev.y, z, yaw)

        if phase is Phase.CONTACT_DETECTED:
            return Command(prev.x, prev.y, self._z_nominal, yaw)

        if phase is Phase.FORCE_RAMP:
            delta = self.admittance.step(f_des, f_filt)
            return Command(prev.x, prev.y, self._z_nominal + delta, yaw)

        if phase is Phase.SWEEP and self._sweep_traj is not None:
            delta = self.admittance.step(f_des, f_filt)
            xy = self._sweep_traj.point(elapsed) if self._hold_xy is None else self._hold_xy
            return Command(float(xy[0]), float(xy[1]), self._z_nominal + delta, yaw)

        if phase is Phase.FORCE_RELEASE:
            delta = self.admittance.step(f_des, f_filt)
            if self._hold_xy is not None:
                xy = self._hold_xy
            elif self._sweep_traj is not None:
                # keep finishing the stroke while the normal force ramps out
                xy = self._sweep_traj.point(self._sweep_elapsed())
            else:
                xy = np.array([prev.x, prev.y])
            return Command(float(xy[0]), float(xy[1]), self._z_nominal + delta, yaw)

        if phase is Phase.RETRACT and self._retract_traj is not None:
            p = self._retract_traj.point(elapsed)
            return Command(float(p[0]), float(p[1]), float(p[2]), yaw)

        return Command(prev.x, prev.y, prev.z, yaw)

    # ------------------------------------------------------------------ guards
    def _in_release_zone(self, tcp: np.ndarray) -> bool:
        """True once the TCP enters the guard band in front of the tray/table edge.

        The band starts ``release_margin`` before ``x_release_line`` so the force
        target has time to slew to zero *before* the guard line is reached.
        """
        c = self.cfg.controller
        guard_start = float(c.x_release_line) + float(c.release_margin)
        return float(tcp[0]) <= guard_start

    def _inside_safe_box(self, tcp: np.ndarray) -> bool:
        ws = self.cfg.workspace
        return (float(ws.safe_x_min) - 1e-3 <= tcp[0] <= float(ws.safe_x_max) + 1e-3
                and float(ws.safe_y_min) - 1e-3 <= tcp[1] <= float(ws.safe_y_max) + 1e-3)

    def _sweep_elapsed(self) -> float:
        """Elapsed time along the sweep trajectory, continued through FORCE_RELEASE."""
        if not self.trace:
            return 0.0
        sweep_start = None
        for event in self.fsm.events:
            if event.stroke == self.fsm.stroke_index and event.phase is Phase.SWEEP:
                sweep_start = event.time
        if sweep_start is None:
            return 0.0
        return float(self.trace[-1].t) - float(sweep_start)

    def _sweep_finished(self, phase: Phase, elapsed: float) -> bool:
        if self._sweep_traj is None:
            return True
        if phase is Phase.FORCE_RELEASE:
            return self._sweep_traj.is_done(self._sweep_elapsed())
        return self._sweep_traj.is_done(elapsed)
