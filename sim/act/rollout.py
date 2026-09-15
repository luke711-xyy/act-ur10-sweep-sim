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
    path_length = 0.0
    previous_xy = env.tcp()[:2].copy()
    failure = ""
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
            })
            targets.append(action.copy())

        for _ in range(max(1, int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz))))):
            measured = float(env.normal_force())
            # The wrist sensor also sees transient inertial reactions from the
            # position-controlled arm.  A contact latch is therefore valid
            # only while the brush is physically near the table search height;
            # this keeps an airborne acceleration spike from handing Z control
            # to the admittance loop.
            near_table = float(env.tcp()[2]) <= float(cfg.workspace.z_search_start) + 0.025
            approach_contact = float(action[2]) <= float(cfg.workspace.z_search_start) + 0.02
            if (not contact and near_table
                    and (measured >= float(cfg.controller.contact_threshold) or approach_contact)):
                contact = True
                z_nominal = float(env.table_top_z) + 0.001
                admittance.reset()
            if contact:
                z = float(z_nominal + admittance.step(float(cfg.controller.desired_force), measured))
            else:
                z = float(action[2])
            cmd = Command(float(action[0]), float(action[1]), z, float(action[3]))
            env.step_control(cmd)
            tcp = env.tcp()
            path_length += float(np.linalg.norm(tcp[:2] - previous_xy))
            previous_xy = tcp[:2].copy()
            post_force = float(env.normal_force())
            peak_force = max(peak_force, post_force)
            trace.append({"t": env.time, "tcp": tcp.copy(), "command": cmd.as_array(),
                          "normal_force": post_force, "contact": contact,
                          "collected": int(env.collected_mask().sum())})
            # Before contact, the sensor includes arm acceleration.  The
            # safety bound is a contact-load bound and is meaningful only
            # after the contact latch above.
            if contact and peak_force > float(cfg.controller.safe_max_force):
                failure = "normal force exceeded safety threshold"
                break
            if path_length > float(cfg.episode.max_contact_path):
                failure = "contact path exceeded limit"
                break
            if np.any(env.lost_mask()):
                failure = "component left the safe workspace"
                break
        if failure:
            break

    if not failure and contact:
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
        sweep = sample_polyline(points[1:], hz, float(cfg.controller.sweep_speed))
        sweep[:, 2] = float(cfg.workspace.z_search_start)
        path = np.concatenate((approach, descent, sweep), axis=0)
        return run_action_path(env, cfg, path, collect_observations=collect_observations)
    finally:
        env.close()
