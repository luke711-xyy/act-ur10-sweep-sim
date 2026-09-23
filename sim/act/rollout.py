"""MuJoCo rollout loop shared by automatic demonstrations and ACT inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..controllers.admittance import AdmittanceController1D, AdmittanceParams
from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .expert import expert_path
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
    inference_queries: int = 0
    inference_timeouts: int = 0
    simulation_preview: bool = False


def _force_loop(cfg):
    c = cfg.controller
    return AdmittanceController1D(
        AdmittanceParams(float(c.admittance.m), float(c.admittance.b), float(c.admittance.k)),
        dt=1.0 / float(cfg.sim.control_hz),
        delta_limit=float(c.delta_z_limit),
        rate_limit=float(c.delta_z_rate_limit),
        direction=-1.0,
    )


def _rate_limit_action(cfg, desired, env, contact_latched: bool) -> np.ndarray:
    """Turn an absolute ACT target into a bounded, executable target.

    ACT predicts absolute table-frame poses, while the UR10 controller runs at
    a finer rate.  This limiter is the interpolation/safety boundary between
    those interfaces; it prevents a stale or early ACT prediction from
    teleporting Z through the table or sweeping faster than the demonstrated
    contact dynamics.
    """
    target = np.asarray(desired, dtype=np.float32).reshape(4).copy()
    target[0] = np.clip(target[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
    target[1] = np.clip(target[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
    z_lower = (float(cfg.workspace.z_search_min) if contact_latched
               else float(cfg.workspace.z_search_start))
    target[2] = np.clip(target[2], z_lower,
                        float(cfg.end_effector.z_home))
    target[3] = np.arctan2(np.sin(target[3]), np.cos(target[3]))

    current = np.asarray([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    dt = 1.0 / float(cfg.act.action_hz)
    xy_speed = (float(cfg.controller.sweep_speed) if contact_latched
                else float(cfg.controller.travel_speed))
    limits = np.asarray([
        xy_speed * dt,
        xy_speed * dt,
        float(cfg.controller.z_speed) * dt,
        float(cfg.controller.yaw_speed) * dt,
    ], dtype=np.float32)
    delta = target - current
    delta[3] = np.arctan2(np.sin(delta[3]), np.cos(delta[3]))
    limited = current + np.clip(delta, -limits, limits)
    limited[0] = np.clip(limited[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
    limited[1] = np.clip(limited[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
    limited[2] = np.clip(limited[2], z_lower,
                         float(cfg.end_effector.z_home))
    limited[3] = np.arctan2(np.sin(limited[3]), np.cos(limited[3]))
    return limited.astype(np.float32)


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
    stable_since = None
    stable_confirmed = False
    total = len(env.layout)
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
            release_request = float(action[2]) > (
                float(cfg.workspace.z_search_start)
                + 0.5 * float(cfg.workspace.z_travel)
            )
            if contact and release_request:
                # Multi-lane expert sweeps lift before transferring to the
                # next lane.  Re-arm contact so the next descent gets a fresh
                # Z reference instead of dragging the brush through the table.
                contact = False
                z_nominal = None
                admittance.reset()
            # The wrist sensor also sees transient inertial reactions from the
            # position-controlled arm.  A contact latch is therefore valid
            # only while the brush is physically near the table search height;
            # this keeps an airborne acceleration spike from handing Z control
            # to the admittance loop.
            near_table = float(env.tcp()[2]) <= float(cfg.workspace.z_search_start) + 0.025
            # Reaching the nominal search height is not itself contact.  The
            # early V4 loop used an ``action z`` shortcut here, which handed
            # the Z axis to admittance before the brush had a measured load;
            # on the joint-controlled UR10 that produced a deep transient and
            # could pin thin parts under the brush.  Contact must be latched
            # from the filtered wrist/contact measurement only.
            if (not contact and not release_request and near_table
                    and measured >= float(cfg.controller.contact_threshold)):
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
            # The configured budget is a contact-path limit.  Airborne
            # approach and lifted lane transfers must not consume it.
            if contact:
                path_length += float(np.linalg.norm(tcp[:2] - previous_xy))
            previous_xy = tcp[:2].copy()
            post_force = float(env.normal_force())
            peak_force = max(peak_force, post_force)
            trace.append({"t": env.time, "tcp": tcp.copy(), "command": cmd.as_array(),
                          "normal_force": post_force, "contact": contact,
                          "collected": int(env.collected_mask().sum())})
            # Stop the expert at the first continuously stable exact result.
            # Continuing to push after the last part has entered the tray can
            # create an artificial wall impact and contaminate an otherwise
            # valid demonstration.
            if total > 0 and int(env.collected_mask().sum()) == total:
                if stable_since is None:
                    stable_since = float(env.time)
                elif float(env.time) - stable_since >= float(cfg.episode.stable_time):
                    stable_confirmed = True
                    break
            else:
                stable_since = None
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
        if failure or stable_confirmed:
            break

    if not failure and contact and not stable_confirmed:
        # Hold the final pose for the explicit stability interval without lifting
        # the brush.  This is part of the task result, not a second sweep.
        final = env.tcp().copy()
        stable_steps = int(round(float(cfg.episode.stable_time) * float(cfg.sim.control_hz)))
        for _ in range(stable_steps):
            force = float(env.normal_force())
            z = float(z_nominal + admittance.step(float(cfg.controller.desired_force), force))
            env.step_control(Command(float(final[0]), float(final[1]), z, float(env.ee.tcp_yaw())))
            # Once the object is in the tray, the brush may legitimately leave
            # the table/contact manifold while the object settles.  Stability
            # is a task-result condition on the collected objects, not another
            # requirement to keep pressing on the tray lip.

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
        path = expert_path(env, cfg)
        return run_action_path(env, cfg, path, collect_observations=collect_observations)
    finally:
        env.close()
