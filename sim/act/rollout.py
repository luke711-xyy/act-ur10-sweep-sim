"""MuJoCo rollout loop shared by automatic demonstrations and ACT inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..controllers.admittance import AdmittanceController1D, AdmittanceParams
from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .expert import expert_waypoints, sample_polyline
from .interface import ACTObservationBuilder


@dataclass
class ActRolloutResult:
    success: bool
    failure_reason: str
    observations: list[dict] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    trace: list[dict] = field(default_factory=list)
    collected: int = 0
    total: int = 0
    elapsed: float = 0.0


def _force_loop(cfg):
    c = cfg.controller
    return AdmittanceController1D(
        AdmittanceParams(float(c.admittance.m), float(c.admittance.b), float(c.admittance.k)),
        dt=1.0 / float(cfg.sim.control_hz),
        delta_limit=float(c.delta_z_limit),
        rate_limit=float(c.delta_z_rate_limit),
        direction=-1.0,
    )


def run_action_path(env: SweepEnv, cfg, path: np.ndarray,
                    collect_observations: bool = False) -> ActRolloutResult:
    """Execute absolute xyz/yaw targets at the ACT action rate.

    The policy supplies z until the wrist sensor detects contact.  From that
    sample onward the z command is replaced by the reusable one-dimensional
    admittance loop; x, y and yaw remain policy controlled.
    """
    builder = ACTObservationBuilder(cfg)
    builder.reset()
    admittance = _force_loop(cfg)
    observations, targets, trace = [], [], []
    contact = False
    z_nominal = None
    peak_force = 0.0
    overforce_steps = 0
    path_length = 0.0
    previous_xy = env.tcp()[:2].copy()
    failure = ""
    completed_now = False
    for action in np.asarray(path, dtype=np.float32).reshape(-1, 4):
        if collect_observations:
            obs = builder.observe(env)
            observations.append({
                "overhead": np.clip(np.transpose(obs["observation.images.overhead"], (1, 2, 0)) * 255,
                                     0, 255).astype(np.uint8),
                "wrist": np.clip(np.transpose(obs["observation.images.wrist"], (1, 2, 0)) * 255,
                                  0, 255).astype(np.uint8),
                "state": obs["observation.state"],
                "environment_state": obs["observation.environment_state"],
                # Observation-only camera: stored beside demonstrations for
                # human inspection, but absent from the ACT input contract.
                "inspection": np.asarray(env.render_rgb("inspection_cam",
                                                         size=builder.spec.image_size),
                                           dtype=np.uint8),
            })
            targets.append(action.copy())

        for _ in range(max(1, int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz))))):
            measured = float(env.normal_force())
            if contact:
                z = float(z_nominal + admittance.step(float(cfg.controller.desired_force), measured))
            else:
                z = float(action[2])
            # The brush TCP is the bottom face of the tool.  A negative task
            # Z would therefore put the brush through the table, which is not
            # a valid unilateral-contact state.  Keep the force controller
            # active, but project its position command onto the non-penetrating
            # half-space; the measured contact force still determines the
            # admittance response above the surface.
            z = max(z, float(env.table_top_z))
            cmd = Command(float(action[0]), float(action[1]), z, float(action[3]))
            env.step_control(cmd)
            tcp = env.tcp()
            path_length += float(np.linalg.norm(tcp[:2] - previous_xy))
            previous_xy = tcp[:2].copy()
            post_force = float(env.normal_force())
            # The wrist sensor also sees transient inertial reactions from the
            # position-controlled arm.  Latch only after the *post-step*
            # measured brush load is above threshold while the TCP is near the
            # table.  Entering the action's height band alone is not contact.
            if not contact:
                near_table = float(tcp[2]) <= float(cfg.workspace.z_search_start) + 0.025
                if near_table and post_force >= float(cfg.controller.contact_threshold):
                    contact = True
                    # The policy action is a target, not a teleport.  Starting
                    # the admittance loop from the measured pose avoids an
                    # artificial centimetre-scale impact on the next command.
                    z_nominal = float(tcp[2])
                    admittance.reset()
            peak_force = max(peak_force, post_force)
            trace.append({"t": env.time, "tcp": tcp.copy(), "command": cmd.as_array(),
                          "normal_force": post_force, "contact": contact,
                          "collected": int(env.collected_mask().sum())})
            # Collection is the task terminal event.  Do not continue into the
            # optional recovery lanes after every component is already inside
            # the tray; those lanes can create a new collision after success.
            if contact and len(env.layout) > 0 and int(env.collected_mask().sum()) == len(env.layout):
                completed_now = True
                break
            # Before contact, the sensor includes arm acceleration.  The
            # safety bound is a contact-load bound and is meaningful only
            # after the contact latch above.
            if contact and post_force > float(cfg.controller.safe_max_force):
                overforce_steps += 1
            else:
                overforce_steps = 0
            if contact and overforce_steps >= int(cfg.controller.get("safe_force_dwell_steps", 3)):
                failure = "normal force exceeded safety threshold"
                break
            if path_length > float(cfg.episode.max_contact_path):
                failure = "contact path exceeded limit"
                break
            if np.any(env.lost_mask()):
                failure = "component left the safe workspace"
                break
        if failure or completed_now:
            break

    completed = (not failure and contact and len(env.layout) > 0
                 and int(env.collected_mask().sum()) == len(env.layout))
    if not failure and contact and not completed:
        # Hold the final pose for the explicit stability interval without lifting
        # the brush.  This is part of the task result, not a second sweep.
        final = env.tcp().copy()
        stable_steps = int(round(float(cfg.episode.stable_time) * float(cfg.sim.control_hz)))
        for _ in range(stable_steps):
            force = float(env.normal_force())
            z = float(z_nominal + admittance.step(float(cfg.controller.desired_force), force))
            env.step_control(Command(float(final[0]), float(final[1]), z, float(env.ee.tcp_yaw())))
            if float(env.normal_force()) < float(cfg.controller.release_threshold):
                failure = "contact lost during stability hold"
                break

    total = len(env.layout)
    collected = int(env.collected_mask().sum())
    success = not failure and total > 0 and collected == total and contact
    if not success and not failure:
        failure = "not all components were collected"
    return ActRolloutResult(success=success, failure_reason=failure,
                            observations=observations,
                            actions=np.asarray(targets, dtype=np.float32),
                            trace=trace, collected=collected, total=total,
                            elapsed=float(env.time))


def run_expert_episode(cfg, seed: int = 0, collect_observations: bool = True) -> ActRolloutResult:
    env = SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    try:
        points = expert_waypoints(env, cfg)
        # Keep the brush at the reset height during the lateral transfer, then
        # descend vertically at the first sweep lane.  Building this as three
        # explicit phases prevents the contact search from sliding past the
        # parts before the force loop has latched.
        hz = float(cfg.act.action_hz)
        approach = sample_polyline(points[:2], hz, float(cfg.controller.sweep_speed))
        approach[:, 2] = float(cfg.end_effector.z_home)
        n_descent = max(1, int(np.ceil(
            (float(cfg.end_effector.z_home) - float(cfg.workspace.z_search_start)) /
            max(float(cfg.controller.z_speed), 1e-6) * hz)))
        descent = np.zeros((n_descent, 4), dtype=np.float32)
        descent[:, 0:2] = points[1]
        descent[:, 2] = np.linspace(float(cfg.end_effector.z_home),
                                     float(cfg.workspace.z_search_start), n_descent)
        if len(points) >= 4:
            # The final horizontal segment is the actual push into the tray.
            # Hold its endpoint briefly so the position-controlled official
            # arm can catch the Cartesian target before we move the brush
            # centreline, instead of peeling off the last object due to
            # actuator lag.
            sweep_prefix = sample_polyline(points[1:-1], hz,
                                            float(cfg.controller.sweep_speed))
            final_push = sample_polyline(points[-2:], hz,
                                         float(cfg.controller.sweep_speed))
            push_hold_steps = int(round(float(cfg.episode.get("final_push_time", 0.0)) * hz))
            push_hold = np.repeat(
                np.array([[points[-2, 0], points[-2, 1],
                           float(cfg.workspace.z_search_start), 0.0]], dtype=np.float32),
                max(0, push_hold_steps), axis=0)
            sweep = np.concatenate((sweep_prefix, push_hold, final_push), axis=0)
        else:
            sweep = sample_polyline(points[1:], hz, float(cfg.controller.sweep_speed))
        sweep[:, 2] = float(cfg.workspace.z_search_start)
        path = np.concatenate((approach, descent, sweep), axis=0)
        return run_action_path(env, cfg, path, collect_observations=collect_observations)
    finally:
        env.close()
