"""MuJoCo rollout loop shared by automatic demonstrations and ACT inference."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import multiprocessing

import numpy as np

from ..controllers.admittance import (
    AdmittanceController1D,
    AdmittanceParams,
    OverforceReliefController1D,
    OverforceReliefParams,
)
from ..controllers.hybrid import Command
from ..config import Config
from ..environments.sweep_env import SweepEnv
from .expert import (ExpertPlan, expert_plan_candidates,
                      _tray_exit_x, contact_home_xy, expert_recovery_waypoints,
                      plan_expert_sweep, sample_polyline)
from .interface import ACTObservationBuilder


@dataclass
class ActRolloutResult:
    success: bool
    failure_reason: str
    observations: list[dict] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    action_valid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    trace: list[dict] = field(default_factory=list)
    collected: int = 0
    total: int = 0
    target_count: int = 0
    target_indices: list[int] = field(default_factory=list)
    target_collected: int = 0
    unexpected_collected: int = 0
    initial_object_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    final_object_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    final_collected_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool))
    elapsed: float = 0.0
    scheduler_queries: int = 0
    scheduler_timeouts: int = 0
    scheduler_late_results: int = 0
    peak_force: float = 0.0
    peak_raw_force: float = 0.0
    sampled_frames: int = 0
    first_contact_position: np.ndarray | None = None
    planner_status: str = "unknown"
    planner_failure_reason: str = ""
    planner_strategy: str = "unknown"
    planner_turn_count: int = 0
    planner_score: float = float("inf")
    planner_attempts: int = 0
    failure_mode: str = ""
    termination_reason: str = ""


@dataclass
class CollectionTerminationTracker:
    """Track inference-only collection timers from full-entry transitions.

    The tracker never inspects brush depth.  It observes only the complete
    object-in-tray mask, so ACT keeps producing normal actions while either
    timer is running.  The target timer takes precedence over the no-new-entry
    timer when both become due on the same control step.
    """

    target_count: int
    target_hold_seconds: float = 2.0
    stall_timeout_seconds: float = 3.0
    previous_mask: np.ndarray | None = field(default=None, repr=False)
    last_entry_time: float | None = None
    target_reached_time: float | None = None

    def update(self, collected_mask: np.ndarray, now: float) -> str:
        mask = np.asarray(collected_mask, dtype=bool).reshape(-1)
        if self.previous_mask is None:
            newly_entered = mask
        else:
            if self.previous_mask.shape != mask.shape:
                raise ValueError("collection mask shape changed during an episode")
            newly_entered = mask & ~self.previous_mask

        if np.any(newly_entered):
            self.last_entry_time = float(now)
        collected = int(mask.sum())
        if (collected >= int(self.target_count)
                and self.target_reached_time is None):
            self.target_reached_time = float(now)
        self.previous_mask = mask.copy()

        if (self.target_reached_time is not None
                and float(now) - self.target_reached_time
                >= float(self.target_hold_seconds)):
            return "target count hold elapsed"
        if (self.last_entry_time is not None
                and float(now) - self.last_entry_time
                >= float(self.stall_timeout_seconds)):
            return "collection stalled for 3 seconds"
        return ""


def collection_goal_status(collected_mask: np.ndarray,
                           target_indices: np.ndarray | None,
                           target_count: int,
                           partial_overlap_mask: np.ndarray | None = None,
                           unintended_component_mask: np.ndarray | None = None,
                           ) -> tuple[bool, str]:
    """Return whether a collection state satisfies the exact task contract.

    ``target_count`` is an exact cardinality, not a lower bound.  The generic
    contract is therefore satisfied by *any* ``target_count`` fully collected
    parts.  ``target_indices`` is accepted only as planner provenance and does
    not affect the result: both expert generation and learned inference use
    the same "any exact N" success contract.  A component whose XY projection
    only partly overlaps the tray invalidates an otherwise exact count.
    """
    mask = np.asarray(collected_mask, dtype=bool).reshape(-1)
    target_count = int(target_count)
    collected = int(mask.sum())
    if collected > target_count:
        return False, "unexpected component entered the target region"
    if collected < target_count:
        return False, "target count was not collected"
    # ``unintended_component_mask`` was the schema-v3 name.  Preserve keyword
    # compatibility for old callers while the new contract intentionally
    # treats it only as a geometric partial-overlap mask, never brush-carry
    # history or planner identity.
    partial = (partial_overlap_mask if partial_overlap_mask is not None
               else unintended_component_mask)
    if partial is not None and np.any(np.asarray(partial, dtype=bool)):
        return False, "component projection only partially overlaps the collection region"
    return True, ""


_COLLECTION_HARD_FAILURE_REASONS = frozenset({
    "unexpected component entered the target region",
    "component projection only partially overlaps the collection region",
})


def collection_step_status(
        collected_mask: np.ndarray,
        target_indices: np.ndarray | None,
        target_count: int,
        partial_overlap_mask: np.ndarray | None = None,
        completion_target_count: int | None = None,
        failure_mode: str = "",
        stop_on_contract_violation: bool = True,
        ) -> tuple[bool, str, str]:
    """Evaluate one physical collection state for both expert and ACT runs.

    The first return value is the requested completion state, the second is an
    immediate hard failure, and the third preserves the detailed contract
    reason.  Deliberate wrong-count demonstrations are the only exception to
    the immediate hard-failure stop; they need to finish their planned wrong
    route so the saved failure remains meaningful.
    """
    requested = (int(target_count) if completion_target_count is None
                 else int(completion_target_count))
    goal_reached, reason = collection_goal_status(
        collected_mask, target_indices, requested,
        partial_overlap_mask=partial_overlap_mask,
    )
    collected = int(np.asarray(collected_mask, dtype=bool).sum())
    failure = ""
    if (stop_on_contract_violation
            and reason in _COLLECTION_HARD_FAILURE_REASONS
            and collected >= int(target_count)
            and failure_mode not in {"wrong_count_over", "wrong_count_under"}):
        failure = reason
    return goal_reached, failure, reason


def brush_delivery_reached(env, cfg, tcp: np.ndarray | None = None) -> bool:
    """Return the shared gate required before one-second goal confirmation."""
    pose = np.asarray(env.tcp() if tcp is None else tcp, dtype=float).reshape(-1)
    return bool(
        pose[0] <= _tray_exit_x(cfg) + 0.005
        and env.brush_fully_inside_target()
    )


def unintended_component_mask(
        collected_mask: np.ndarray,
        target_indices: np.ndarray | None,
        brush_contact_mask: np.ndarray,
        partial_overlap_mask: np.ndarray) -> np.ndarray:
    """Compatibility helper returning only geometric partial tray entries.

    Brush contact/carry is deliberately not a failure branch.  Planner target
    identity is audit metadata only, so neither input changes the mask.
    """
    full = np.asarray(collected_mask, dtype=bool).reshape(-1)
    current = np.asarray(brush_contact_mask, dtype=bool).reshape(-1)
    partial = np.asarray(partial_overlap_mask, dtype=bool).reshape(-1)
    if not (len(full) == len(current) == len(partial)):
        raise ValueError("component carry masks must have the same length")
    return (~full) & partial


def stall_failure_sidewall_ok(result: ActRolloutResult, cfg) -> bool:
    """Check that a stall happened at the real tray wall, not mid-table."""
    target_count = int(result.target_count)
    if target_count == 1:
        return int(result.collected) == 0
    if not (0 < int(result.collected) < target_count):
        return False
    mask = np.asarray(result.final_collected_mask, dtype=bool)
    positions = np.asarray(result.final_object_positions, dtype=float)
    selected = np.asarray(result.target_indices, dtype=int)
    if (mask.ndim != 1 or positions.ndim != 2 or positions.shape[1] < 2
            or len(selected) != target_count
            or np.any(selected < 0) or np.any(selected >= len(mask))):
        return False
    missed = selected[~mask[selected]]
    if len(missed) == 0:
        return False
    # A valid stall leaves the missed selected parts at the actual mouth and
    # near a real side wall.  Parts left in the middle of the table fail this
    # gate even if the total count happens to be partial.
    mouth_limit = float(cfg.target.x_max) + 0.10
    wall_band = max(abs(float(cfg.target.y_min)),
                    abs(float(cfg.target.y_max))) - 0.05
    missed_positions = positions[missed, :2]
    return bool(np.all(missed_positions[:, 0] <= mouth_limit)
                and np.all(np.abs(missed_positions[:, 1]) >= wall_band))


def _force_loop(cfg):
    c = cfg.controller
    return AdmittanceController1D(
        AdmittanceParams(float(c.admittance.m), float(c.admittance.b), float(c.admittance.k)),
        dt=1.0 / float(cfg.sim.control_hz),
        delta_limit=float(c.delta_z_limit),
        rate_limit=float(c.delta_z_rate_limit),
        direction=-1.0,
        upward_rate_limit=float(c.get(
            "delta_z_upward_rate_limit", c.delta_z_rate_limit)),
        downward_rate_limit=float(c.get(
            "delta_z_downward_rate_limit", c.delta_z_rate_limit)),
    )


def _overforce_relief_loop(cfg):
    c = cfg.controller
    p = c.get("overforce_relief", {})
    return OverforceReliefController1D(
        OverforceReliefParams(
            activation_force=float(p.get("activation_force", 10.0)),
            release_force=float(p.get("release_force", 7.0)),
            upward_rate=float(p.get("upward_rate", 0.02)),
            return_rate=float(p.get("return_rate", 0.10)),
            max_offset=float(p.get("max_offset", 0.00025)),
        ),
        dt=1.0 / float(cfg.sim.control_hz),
    )


def _contact_z_command(cfg, admittance, relief, z_nominal: float,
                       measured_force: float) -> float:
    correction = (
        float(admittance.step(
            float(cfg.controller.desired_force), float(measured_force)))
        + float(relief.step(float(measured_force)))
    )
    correction_limit = abs(float(cfg.controller.delta_z_limit))
    z = float(z_nominal) + float(np.clip(
        correction, -correction_limit, correction_limit))
    return max(float(cfg.workspace.z_search_min), z)


def _angle_delta(value: float, previous: float) -> float:
    return float(np.arctan2(np.sin(value - previous), np.cos(value - previous)))


def policy_action_delta(reference: np.ndarray, previous_reference: np.ndarray,
                        contact_latched: bool) -> np.ndarray:
    """Return schema-v4 ``[dx, dy, dz, dyaw]`` supervision.

    After contact, physical Z is not policy-owned.  The zero ``dz`` label is a
    deliberate no-op mask within the fixed four-dimensional action tensor.
    """
    reference = np.asarray(reference, dtype=np.float32).reshape(4)
    previous = np.asarray(previous_reference, dtype=np.float32).reshape(4)
    return np.asarray([
        float(reference[0] - previous[0]),
        float(reference[1] - previous[1]),
        0.0 if contact_latched else float(reference[2] - previous[2]),
        _angle_delta(float(reference[3]), float(previous[3])),
    ], dtype=np.float32)


def _contact_loss_ready(contact_sweep_time: float,
                        contact_sweep_path: float,
                        planned_sweep_time: float,
                        planned_sweep_path: float,
                        min_time: float = 1.5,
                        min_path: float = 0.10,
                        progress: float = 0.30) -> bool:
    """Return whether a deliberate loss has enough real sweep context.

    A deliberate contact-loss failure is useful training data only after the
    brush has actually travelled over the table.  The absolute floors cover
    unusually short plans; the relative thresholds keep the event from being
    injected at touchdown on longer or shorter layouts.
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    required_time = max(float(min_time), progress * float(planned_sweep_time))
    required_path = max(float(min_path), progress * float(planned_sweep_path))
    return (float(contact_sweep_time) >= required_time
            and float(contact_sweep_path) >= required_path)


def run_action_path(env: SweepEnv, cfg, path: np.ndarray,
                    collect_observations: bool = False,
                    phases: list[str] | None = None,
                    target_indices: np.ndarray | None = None,
                    recovery_planner=None,
                    max_recovery_rounds: int = 0,
                    failure_mode: str = "",
                    contact_z_mode: str = "admittance") -> ActRolloutResult:
    """Execute expert absolute targets and record full-episode ACT labels.

    Approach, descent, contact build and sweep are all behavior-cloning
    phases.  Labels are consecutive fixed-frame ``[dx, dy, dz, dyaw]``
    increments.  Once measured force latches contact, ``dz`` becomes a logged
    no-op while the admittance controller owns the applied Z command.
    """
    builder = ACTObservationBuilder(cfg)
    builder.reset()
    admittance = _force_loop(cfg)
    overforce_relief = _overforce_relief_loop(cfg)
    if contact_z_mode not in {"geometric", "admittance", "expert_admittance"}:
        raise ValueError(
            "contact_z_mode must be 'geometric', 'admittance', or 'expert_admittance'")
    uses_admittance = contact_z_mode in {"admittance", "expert_admittance"}
    geometry_latches_contact = contact_z_mode in {"geometric", "expert_admittance"}
    geometric_contact_z = float(getattr(env, "table_top_z", cfg.table.top_z))
    geometric_contact_tolerance = max(0.0, float(cfg.controller.get(
        "expert_contact_geometry_tolerance", 0.0005)))
    observations, targets, target_valid, trace = [], [], [], []
    contact = False
    admittance_active = False
    z_nominal = None
    peak_force = 0.0
    peak_raw_force = 0.0
    overforce_steps = 0
    contact_loss_steps = 0
    path_length = 0.0
    previous_xy = env.tcp()[:2].copy()
    failure = ""
    goal_reached = False
    first_contact_position = None
    path = np.asarray(path, dtype=np.float32).reshape(-1, 4)
    failure_mode = str(failure_mode or "")
    phases = list(phases or ["sweep"] * len(path))
    if len(phases) != len(path):
        raise ValueError("one phase label is required per execution reference")
    target_count = int(np.clip(int(cfg.get_path("task.target_count", len(env.layout))),
                               1, max(1, len(env.layout))))
    selected_indices = ([int(value) for value in np.asarray(target_indices, dtype=int).ravel()]
                        if target_indices is not None else [])
    # Deliberate wrong-count demonstrations execute a physically valid plan
    # for the wrong cardinality, while the episode metadata still retains the
    # requested target count.  This lets the quality gate describe the exact
    # mismatch instead of stopping when the nominal count is first reached.
    completion_target_count = target_count
    if failure_mode in {"wrong_count_over", "wrong_count_under"}:
        completion_target_count = len(selected_indices)
    initial_object_positions = np.asarray(env.component_positions(), dtype=np.float32).copy()
    max_frames = max(1, int(cfg.episode.get("max_frames", 500)))
    sampled_frames = 0
    brush_inside_time = 0.0
    brush_inside_timeout = max(0.0, float(cfg.episode.get(
        "brush_inside_target_timeout", 1.0)))
    delivery_reached = False
    previous_reference = np.array([*env.tcp(), float(env.ee.tcp_yaw())], dtype=float)
    executed_reference = np.array([*env.tcp(), float(env.ee.tcp_yaw())], dtype=float)
    deliberate_lift_started = False
    deliberate_lift_end = 0.0
    contact_sweep_time = 0.0
    contact_sweep_path = 0.0
    sweep_rows = [row for row, phase in zip(path, phases) if phase == "sweep"]
    if len(sweep_rows) >= 2:
        planned_sweep_path = float(np.linalg.norm(
            np.diff(np.asarray(sweep_rows, dtype=float)[:, :2], axis=0), axis=1).sum())
    else:
        planned_sweep_path = 0.0
    planned_sweep_time = planned_sweep_path / max(
        float(cfg.controller.sweep_speed), 1e-6)
    do_not_stop_on_goal = (
        failure_mode in {"stall_outside_tray", "misroute"}
        or (failure_mode == "wrong_count_under"
            and completion_target_count == 0))

    def capture(action, phase: str, valid: bool):
        nonlocal peak_force, peak_raw_force
        reference = np.asarray(action, dtype=np.float32)
        delta = policy_action_delta(
            reference, previous_reference, contact_latched=contact)
        captured_index = None
        if collect_observations:
            obs = builder.observe(env, contact_latched=contact)
            images = {
                view: np.clip(
                    np.transpose(obs[f"observation.images.{view}"], (1, 2, 0)) * 255,
                    0, 255,
                ).astype(np.uint8)
                for view in builder.camera_views
            }
            wrench = np.asarray(env.wrench(), dtype=np.float32).reshape(6)
            tcp = np.asarray(env.tcp(), dtype=np.float32).reshape(3)
            joints = np.asarray(env.ee.joint_state(), dtype=np.float32).reshape(6)
            # The UR10 force channel has a stateful low-pass filter.  Reading
            # telemetry must not advance that filter, or recording an episode
            # changes the physical control loop relative to the screening pass.
            sampled_force = float(env.normal_force_observation())
            peak_force = max(peak_force, sampled_force)
            raw_force_peak = float(
                getattr(env, "last_raw_force_peak", None)
                or getattr(env, "raw_normal_force", env.normal_force)()
            )
            peak_raw_force = max(peak_raw_force, raw_force_peak)
            frame_observation = {
                "images": images,
                "state": obs["observation.state"],
                "environment_state": obs["observation.environment_state"],
                "supervisor_state": obs["supervisor_state"],
                "t": float(env.time),
                "phase": str(phase),
                "policy_mask": bool(valid),
                "joint_position": joints,
                "tcp_pose": np.array([*tcp, float(env.ee.tcp_yaw())], dtype=np.float32),
                "wrench": wrench,
                "normal_force": sampled_force,
                "raw_normal_force": float(
                    getattr(env, "raw_normal_force", env.normal_force)()),
                "compensated_normal_force_raw": float(
                    getattr(env, "compensated_normal_force_raw", env.normal_force)()),
                "raw_normal_force_peak": raw_force_peak,
                "raw_normal_force_peak_t": float(
                    getattr(env, "last_raw_force_peak_time", None) or env.time),
                "contact": bool(contact),
                "contact_latched": bool(contact),
                "fully_collected": int(env.collected_mask().sum()),
                # Simulator truth is logged for audit/visualization only.  It
                # is deliberately not appended to observation.state and never
                # reaches ACT.
                "object_pose": np.concatenate((
                    np.asarray(env.component_positions(), dtype=np.float32),
                    np.asarray(env.component_quats(), dtype=np.float32),
                ), axis=1),
                "object_collected": np.asarray(env.collected_mask(), dtype=bool),
                "reference": executed_reference.copy(),
                "policy_reference": reference.copy(),
                "applied_reference": executed_reference.copy(),
                "policy_z": float(reference[2]),
                "applied_z": float(executed_reference[2]),
                "z_owner": "admittance" if contact else "policy",
                "action_delta": delta,
            }
            # Retain the historical names for callers/tests that consume a
            # legacy two-view observation.  The complete dynamic map is the
            # authoritative five-view training input.
            frame_observation.update({
                name: images[name] for name in ("overhead", "wrist")
                if name in images
            })
            observations.append(frame_observation)
            captured_index = len(observations) - 1
            targets.append(delta)
            target_valid.append(bool(valid))
        return captured_index

    def update_brush_inside_time() -> bool:
        """Accumulate full-brush-in-tray time at the MuJoCo control rate."""
        nonlocal brush_inside_time
        if env.brush_fully_inside_target():
            brush_inside_time += float(env.control_dt)
        else:
            brush_inside_time = 0.0
        return brush_inside_timeout > 0.0 and brush_inside_time >= brush_inside_timeout

    execution = [(row, phase) for row, phase in zip(path, phases)]
    execution_index = 0
    recovery_round = 0
    while execution_index < len(execution):
        if sampled_frames >= max_frames:
            failure = "episode frame limit reached"
            break
        action, phase = execution[execution_index]
        execution_index += 1
        policy_phase = phase in {"approach", "descent", "contact_build", "sweep"}
        if contact and uses_admittance and not admittance_active:
            z_nominal = float(env.tcp()[2])
            admittance.reset()
            overforce_relief.reset()
            admittance_active = True
        valid = bool(policy_phase)
        frame_contact_at_start = bool(contact)
        captured_index = capture(action, phase, valid)
        sampled_frames += 1
        previous_reference[:] = action

        substeps = max(1, int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz))))
        command_start = executed_reference.copy()
        applied_z_command = float(command_start[2])
        yaw_step = _angle_delta(float(action[3]), float(command_start[3]))
        for substep in range(substeps):
            # The 25 Hz reference is a sampled trajectory, not a sequence of
            # Cartesian set-point jumps.  Interpolate it at the 100 Hz control
            # rate so coupled UR10 IK motion does not turn each XY/yaw update
            # into an artificial vertical impact on the force-control sole.
            alpha = float(substep + 1) / float(substeps)
            x = float(command_start[0] + alpha * (float(action[0]) - command_start[0]))
            y = float(command_start[1] + alpha * (float(action[1]) - command_start[1]))
            yaw = float(command_start[3] + alpha * yaw_step)
            measured = float(env.normal_force())
            peak_force = max(peak_force, measured)
            if contact_z_mode == "geometric" and contact:
                z = geometric_contact_z
            elif contact and admittance_active:
                z = _contact_z_command(
                    cfg, admittance, overforce_relief, z_nominal, measured)
            else:
                z = float(command_start[2] + alpha * (float(action[2]) - command_start[2]))
            if contact and phase == "sweep":
                contact_sweep_time += 1.0 / float(cfg.sim.control_hz)
            # Deliberate failure data remains a real MuJoCo rollout.  Wait
            # until a meaningful portion of the planned contact pass has been
            # executed, then lift the compliant brush for a short interval so
            # the trace contains an observable mid-sweep contact-loss event.
            if (failure_mode == "lose_contact" and contact and phase == "sweep"
                    and not deliberate_lift_started
                    and _contact_loss_ready(
                        contact_sweep_time, contact_sweep_path,
                        planned_sweep_time, planned_sweep_path,
                        min_time=float(cfg.episode.get(
                            "failure_contact_min_time", 1.5)),
                        min_path=float(cfg.episode.get(
                            "failure_contact_min_path", 0.10)),
                        progress=float(cfg.episode.get(
                            "failure_contact_progress", 0.30)))):
                deliberate_lift_started = True
                deliberate_lift_end = float(env.time) + 0.60
            if failure_mode == "lose_contact" and deliberate_lift_started:
                if float(env.time) < deliberate_lift_end:
                    z = max(z, float(cfg.workspace.z_search_start)
                            + max(float(cfg.workspace.z_travel), 0.04))
            # A position-controlled MuJoCo contact needs a small virtual
            # penetration target to create sustained normal load.  The rigid
            # contact constraint keeps the *measured TCP* at the surface; only
            # the servo reference may go below it, bounded by z_search_min.
            z = max(z, float(cfg.workspace.z_search_min))
            applied_z_command = float(z)
            cmd = Command(x, y, z, yaw)
            env.step_control(cmd)
            raw_force_peak = float(getattr(env, "last_raw_force_peak", None) or 0.0)
            peak_raw_force = max(peak_raw_force, raw_force_peak)
            tcp = env.tcp()
            if contact:
                travelled = float(np.linalg.norm(tcp[:2] - previous_xy))
                path_length += travelled
                if phase == "sweep":
                    contact_sweep_path += travelled
            previous_xy = tcp[:2].copy()
            post_force = float(env.normal_force())
            if not failure_mode and not do_not_stop_on_goal:
                delivery_reached = delivery_reached or brush_delivery_reached(
                    env, cfg, tcp
                )
                # Contact latching uses the post-step wrist Fz after the modeled
                # distal-tool inertial-load delta is removed. Entering the action's
                # height band alone is not contact.
            if not contact:
                contact_band_z = (geometric_contact_z if geometry_latches_contact
                                  else float(cfg.workspace.z_search_start))
                near_table = float(tcp[2]) <= contact_band_z + 0.025
                geometric_touch = (
                    geometry_latches_contact
                    and float(tcp[2]) <= geometric_contact_z
                    + geometric_contact_tolerance)
                force_touch = (
                    contact_z_mode == "admittance"
                    and post_force >= float(cfg.controller.contact_threshold))
                if near_table and (geometric_touch or force_touch):
                    contact = True
                    if uses_admittance:
                        z_nominal = float(tcp[2])
                        admittance.reset()
                        overforce_relief.reset()
                        admittance_active = True
                    if first_contact_position is None:
                        first_contact_position = np.asarray(
                            tcp, dtype=np.float32
                        ).copy()
            brush_inside_timeout_reached = update_brush_inside_time()
            peak_force = max(peak_force, post_force)
            trace.append({"t": env.time, "tcp": tcp.copy(), "command": cmd.as_array(),
                          "policy_reference": np.asarray(action, dtype=np.float32).copy(),
                          "policy_z": float(action[2]), "applied_z": float(z),
                          "z_owner": ("admittance" if frame_contact_at_start
                                      else ("transition" if contact else "policy")),
                          "phase": str(phase), "wrench": np.asarray(env.wrench()).copy(),
                          "normal_force": post_force,
                          "raw_normal_force": float(
                              getattr(env, "raw_normal_force", env.normal_force)()),
                          "compensated_normal_force_raw": float(
                              getattr(env, "compensated_normal_force_raw",
                                      env.normal_force)()),
                          "contact": contact,
                          "raw_normal_force_peak": raw_force_peak,
                          "raw_normal_force_peak_t": float(
                              getattr(env, "last_raw_force_peak_time", None) or env.time),
                          "table_contact": bool(env.brush_table_contact()),
                          "geometric_contact": (
                              float(tcp[2]) <= geometric_contact_z
                              + geometric_contact_tolerance),
                          "collected": int(env.collected_mask().sum())})
            if contact:
                partial_mask = (None if failure_mode else
                                env.partial_collection_overlap_mask())
                goal_reached, hard_failure, goal_reason = collection_step_status(
                    env.collected_mask(), target_indices, target_count,
                    partial_overlap_mask=partial_mask,
                    completion_target_count=completion_target_count,
                    failure_mode=failure_mode,
                )
                if hard_failure:
                    failure = hard_failure
                    break
                if goal_reached and failure_mode and not do_not_stop_on_goal:
                    break
            if brush_inside_timeout_reached and (not goal_reached or failure_mode):
                failure = "brush remained fully inside collection region for 1 s without exact goal"
                break
            # The safety bound is a contact-load bound and is meaningful only
            # after the contact latch above.
            if contact and post_force > float(cfg.controller.safe_max_force):
                overforce_steps += 1
            else:
                overforce_steps = 0
            if contact and overforce_steps >= int(cfg.controller.get("safe_force_dwell_steps", 3)):
                failure = "normal force exceeded safety threshold"
                break
            if geometry_latches_contact:
                contact_hold_tolerance = geometric_contact_tolerance
                if contact_z_mode == "expert_admittance":
                    contact_hold_tolerance += abs(float(cfg.controller.delta_z_limit))
                contact_loss_steps = (
                    contact_loss_steps + 1
                    if (contact and float(tcp[2]) > geometric_contact_z
                        + contact_hold_tolerance) else 0)
            elif admittance_active and post_force < float(cfg.controller.release_threshold):
                contact_loss_steps += 1
            else:
                contact_loss_steps = 0
            if contact_loss_steps >= max(1, int(round(0.1 * float(cfg.sim.control_hz)))):
                # If the exact requested count and the configured tray-depth
                # gate become true on this same control substep, delivery has
                # completed.  Do not let the contact-loss timer preempt that
                # terminal condition; otherwise a late sensor/geometry release
                # can turn an already-delivered episode into a false failure.
                if not (goal_reached and delivery_reached and not failure_mode):
                    failure = "contact lost for 100 ms"
                break
            if path_length > float(cfg.episode.max_contact_path):
                failure = "contact path exceeded limit"
                break
            if env.unsafe_robot_collision():
                failure = "unsafe robot collision"
                break
            if np.any(env.lost_mask()):
                failure = "component left the safe workspace"
                break
            if env.time >= float(cfg.episode.max_time):
                failure = "episode timeout"
                break
            # Once an exact count is reached and the brush has delivered it to
            # the configured tray depth, stop the expert path and enter the
            # existing one-second object-stability confirmation.  Continuing
            # the remaining preplanned curve can lift the brush or disturb a
            # correctly delivered set, turning a completed sweep into a false
            # contact-loss failure.
            if goal_reached and delivery_reached and not failure_mode:
                break
        executed_reference[:] = [
            float(action[0]), float(action[1]), applied_z_command,
            float(action[3]),
        ]
        if captured_index is not None:
            owner = ("transition" if (not frame_contact_at_start and contact)
                     else ("admittance" if frame_contact_at_start else "policy"))
            observations[captured_index]["reference"] = executed_reference.copy()
            observations[captured_index]["applied_reference"] = executed_reference.copy()
            observations[captured_index]["applied_z"] = float(executed_reference[2])
            observations[captured_index]["z_owner"] = owner
        if failure or (goal_reached and failure_mode and not do_not_stop_on_goal):
            break
        if goal_reached and delivery_reached and not failure_mode:
            break
        if (execution_index == len(execution) and recovery_planner is not None
                and recovery_round < int(max_recovery_rounds)):
            extension, extension_phases = recovery_planner(env, recovery_round)
            extension = np.asarray(extension, dtype=np.float32).reshape(-1, 4)
            if len(extension):
                if len(extension_phases) != len(extension):
                    raise ValueError("recovery planner returned mismatched phases")
                execution.extend(zip(extension, extension_phases))
                recovery_round += 1

    if not failure and goal_reached and not failure_mode and not delivery_reached:
        failure = "exact count reached before configured tray depth"

    stable_complete = False
    if not failure and goal_reached and not failure_mode:
        # Confirm the requested count for a full second.  These frames stay in
        # the episode viewer but are masked out of ACT behavior-cloning loss.
        final = env.tcp().copy()
        final_yaw = float(env.ee.tcp_yaw())
        hold_action = np.array([final[0], final[1], final[2], final_yaw], dtype=np.float32)
        stable_frames = int(round(float(cfg.episode.stable_time) * float(cfg.act.action_hz)))
        stable_complete = True
        for _ in range(stable_frames):
            if sampled_frames >= max_frames:
                failure = "episode frame limit reached"
                stable_complete = False
                break
            capture(hold_action, "stability", False)
            sampled_frames += 1
            previous_reference[:] = hold_action
            for _ in range(max(1, int(round(float(cfg.sim.control_hz) /
                                            float(cfg.act.action_hz))))):
                force = float(env.normal_force())
                peak_force = max(peak_force, force)
                z = (geometric_contact_z if contact_z_mode == "geometric" else
                     _contact_z_command(
                         cfg, admittance, overforce_relief, z_nominal, force))
                env.step_control(Command(float(final[0]), float(final[1]), z, final_yaw))
                raw_force_peak = float(getattr(env, "last_raw_force_peak", None) or 0.0)
                peak_raw_force = max(peak_raw_force, raw_force_peak)
                post_force = float(env.normal_force())
                peak_force = max(peak_force, post_force)
                brush_contacts = env.brush_component_contact_mask()
                trace.append({"t": env.time, "tcp": env.tcp().copy(),
                              "command": np.array([final[0], final[1], z, final_yaw]),
                              "phase": "stability", "wrench": np.asarray(env.wrench()).copy(),
                              "z_owner": "admittance",
                              "normal_force": post_force,
                              "raw_normal_force": float(
                                  getattr(env, "raw_normal_force", env.normal_force)()),
                              "compensated_normal_force_raw": float(
                                  getattr(env, "compensated_normal_force_raw",
                                          env.normal_force)()),
                              "contact": contact,
                              "raw_normal_force_peak": raw_force_peak,
                              "raw_normal_force_peak_t": float(
                                  getattr(env, "last_raw_force_peak_time", None) or env.time),
                              "table_contact": bool(env.brush_table_contact()),
                              "geometric_contact": (
                                  float(env.tcp()[2]) <= geometric_contact_z
                                  + geometric_contact_tolerance),
                              "collected": int(env.collected_mask().sum())})
                if post_force > float(cfg.controller.safe_max_force):
                    overforce_steps += 1
                else:
                    overforce_steps = 0
                if overforce_steps >= int(cfg.controller.get(
                        "safe_force_dwell_steps", 3)):
                    failure = "normal force exceeded safety threshold"
                    stable_complete = False
                    break
                stable_ok, stable_reason = collection_goal_status(
                    env.collected_mask(), target_indices, target_count,
                    partial_overlap_mask=env.partial_collection_overlap_mask())
                if not stable_ok:
                    failure = ("target count did not remain stable for 1 s"
                               if stable_reason == "target count was not collected"
                               else stable_reason)
                    stable_complete = False
                    break
                # Once the exact object set is inside the tray, the brush is
                # allowed to unload while the one-second object hold is
                # verified.  Contact loss is a sweep-phase failure; requiring
                # the tool to remain pressed during this post-collection hold
                # can pull a boundary object back out and is not part of the
                # task's success contract.
            if failure:
                break

    total = len(env.layout)
    final_mask = np.asarray(env.collected_mask(), dtype=bool)
    collected = int(final_mask.sum())
    exact_goal, exact_reason = collection_goal_status(
        final_mask, target_indices, completion_target_count,
        partial_overlap_mask=(None if failure_mode else
                              env.partial_collection_overlap_mask()))
    target_collected = (int(final_mask[np.asarray(selected_indices, dtype=int)].sum())
                        if selected_indices else collected)
    unexpected_collected = (int((final_mask & ~np.isin(
        np.arange(len(final_mask)), np.asarray(selected_indices, dtype=int))).sum())
                            if selected_indices else 0)
    success = (not failure and total > 0 and exact_goal and contact and stable_complete)
    if failure_mode and not failure:
        if failure_mode == "wrong_count_over":
            failure = (f"wrong count: collected {collected} instead of "
                       f"target {target_count} (over)")
        elif failure_mode == "wrong_count_under":
            failure = (f"wrong count: collected {collected} instead of "
                       f"target {target_count} (under)")
        else:
            failure = f"deliberate failure: {failure_mode}"
        success = False
    if not success and not failure:
        failure = exact_reason or "target count was not collected"
    return ActRolloutResult(success=success, failure_reason=failure,
                            observations=observations,
                            actions=np.asarray(targets, dtype=np.float32),
                            action_valid=np.asarray(target_valid, dtype=bool),
                            trace=trace, collected=collected, total=total,
                            target_count=target_count,
                            target_indices=selected_indices,
                            target_collected=target_collected,
                            unexpected_collected=unexpected_collected,
                            initial_object_positions=initial_object_positions,
                            final_object_positions=np.asarray(
                                env.component_positions(), dtype=np.float32).copy(),
                            final_collected_mask=final_mask.copy(),
                            elapsed=float(env.time), failure_mode=failure_mode,
                            peak_force=float(peak_force),
                            peak_raw_force=float(peak_raw_force),
                            sampled_frames=int(sampled_frames),
                            first_contact_position=(
                                np.asarray(first_contact_position, dtype=np.float32)
                                if first_contact_position is not None
                                else None
                            ))


def _expert_execution_path(cfg, plan: ExpertPlan,
                           failure_mode: str = "",
                           failure_seed: int = 0,
                           table_top_z: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Build the supervisor phases for one geometric expert candidate."""
    home = contact_home_xy(cfg).astype(float)
    waypoints = np.asarray(plan.waypoints, dtype=float)
    if (failure_mode in {"stall_outside_tray", "wrong_count_under"}
            and not len(plan.target_indices) and len(waypoints)):
        # The n=1 boundary has no positive partial target set.  Preserve a
        # real approach/contact sweep, but stop before the tray crossing so
        # the recorded result is an honest zero-collected under-count.
        crossing = np.flatnonzero(
            waypoints[:, 0] <= float(cfg.target.x_max))
        if len(crossing):
            waypoints = waypoints[:max(2, int(crossing[0]))]
    if failure_mode == "misroute" and len(waypoints) >= 2:
        # Keep the initial contact approach, then move the whole contact lane
        # toward one of the two wrong lateral sides.  The side is a
        # deterministic function of the seed, so repeated manual generation
        # remains reproducible while producing both route variants.
        side = -1.0 if int(failure_seed) % 2 else 1.0
        waypoints = waypoints.copy()
        waypoints[1:, 1] = np.clip(
            waypoints[1:, 1] + side * 0.14,
            float(cfg.workspace.y_min) + 0.03,
            float(cfg.workspace.y_max) - 0.03,
        )
    points = (waypoints if len(waypoints) and np.allclose(waypoints[0], home)
              else np.vstack((home, waypoints)))
    # Keep the brush at the reset height during the lateral transfer, then
    # descend vertically at the first sweep lane.  Building this as three
    # explicit phases prevents the contact search from sliding past the parts
    # before the force loop has latched.
    hz = float(cfg.act.action_hz)
    first_yaw = float(plan.yaw_offset)
    yaw_rate = float(cfg.get_path("controller.yaw_speed_limit", np.deg2rad(45.0)))
    accel = float(cfg.get_path("planner.accel", 0.8))
    # Global coverage-A* waypoints are already selectively filleted and
    # capture-validated.  A second uniform pass here can cut the critical
    # corner that touches one of the targets, so follow that curve verbatim.
    curve_radius = (0.0 if plan.strategy == "coverage_astar" else
                    float(cfg.get_path("planner.expert_corner_radius", 0.025)))
    # The airborne transfer is supervisor-owned and should use the travel
    # limit, not the deliberately slower contact sweep speed.  Reusing the
    # sweep speed made the approach dominate short demonstrations and left
    # less time for the motion ACT is meant to learn.
    approach = sample_polyline(points[:2], hz,
                               float(cfg.controller.get("travel_speed",
                                                         cfg.controller.sweep_speed)),
                               yaw=first_yaw, yaw_rate=yaw_rate, accel=accel,
                               curve_radius=curve_radius)
    approach[:, 2] = float(cfg.end_effector.z_home)
    z_home = float(cfg.end_effector.z_home)
    contact_z = float(cfg.table.top_z if table_top_z is None else table_top_z)
    fast_speed = max(float(cfg.controller.z_speed), 1e-6)
    slow_speed = max(float(cfg.controller.get("expert_contact_z_speed", 0.01)), 1e-6)
    slow_band = max(0.0, float(cfg.controller.get("expert_contact_slow_band", 0.008)))
    slow_start_z = contact_z + min(slow_band, max(0.0, z_home - contact_z))
    n_fast = int(np.ceil(max(0.0, z_home - slow_start_z) / fast_speed * hz))
    n_slow = int(np.ceil(max(0.0, slow_start_z - contact_z) / slow_speed * hz))
    fast_descent = (np.linspace(z_home, slow_start_z, n_fast + 1)
                    if n_fast else np.asarray([z_home], dtype=float))
    slow_descent = (np.linspace(slow_start_z, contact_z, n_slow + 1)[1:]
                    if n_slow else np.zeros(0, dtype=float))
    descent_z = np.concatenate((fast_descent, slow_descent))
    descent = np.zeros((len(descent_z), 4), dtype=np.float32)
    descent[:, 0:2] = points[1]
    descent[:, 2] = descent_z
    descent[:, 3] = first_yaw
    final_push_distance = (
        float(np.linalg.norm(points[-1] - points[-2]))
        if len(points) >= 4 else 0.0
    )
    sweep = sample_polyline(
        points[1:], hz, float(cfg.controller.sweep_speed),
        yaw_rate=yaw_rate, initial_yaw=first_yaw, accel=accel,
        curve_radius=curve_radius,
        terminal_speed=(
            float(cfg.controller.get("tray_entry_speed", cfg.controller.sweep_speed))
            if final_push_distance > 0.0 else None
        ),
        terminal_distance=final_push_distance,
    )
    if failure_mode == "stall_outside_tray" and len(plan.target_indices):
        # The plan was generated against a virtual, laterally shifted tray in
        # ``run_expert_episode``.  Preserve that complete A* route here; the
        # physical scene still contains the real tray, so the mismatch is
        # resolved by actual contact with its side wall rather than by a
        # post-hoc waypoint edit.
        pass
    sweep[:, 2] = contact_z
    build_steps = int(round(float(cfg.episode.get("contact_build_time", 0.5)) * hz))
    contact_build = np.repeat(
        np.array([[points[1, 0], points[1, 1],
                   contact_z, first_yaw]], dtype=np.float32),
        max(1, build_steps), axis=0)
    path = np.concatenate((approach, descent, contact_build, sweep), axis=0)
    phases = (["approach"] * len(approach) + ["descent"] * len(descent)
              + ["contact_build"] * len(contact_build)
              + ["sweep"] * len(sweep))
    return path, phases


def _annotate_expert_result(result: ActRolloutResult, plan: ExpertPlan,
                            attempts: int) -> ActRolloutResult:
    """Attach planner provenance without changing the physical result."""
    result.planner_status = "feasible" if plan.feasible else "infeasible"
    result.planner_failure_reason = str(plan.failure_reason)
    result.planner_strategy = str(plan.strategy)
    result.planner_turn_count = int(plan.turn_count)
    result.planner_score = float(plan.score)
    result.planner_attempts = int(attempts)
    if not plan.feasible:
        # A safe probe must never inherit a generic count failure and look like
        # an executed expert sweep.
        result.success = False
        result.failure_reason = str(plan.failure_reason)
    return result


def _screen_expert_candidate_process(payload):
    """Run one render-free candidate rollout in an isolated worker process."""
    cfg_data, seed, total, failure_mode, plan, attempt = payload
    cfg = Config(cfg_data)
    env = SweepEnv(cfg, seed=seed)
    try:
        env.reset(seed=seed)
        path, phases = _expert_execution_path(
            cfg, plan, failure_mode, failure_seed=seed,
            table_top_z=env.table_top_z)
        result = run_action_path(
            env, cfg, path, collect_observations=False,
            phases=phases,
            target_indices=np.asarray(plan.target_indices, dtype=int),
            recovery_planner=None, max_recovery_rounds=0,
            failure_mode=failure_mode,
            contact_z_mode="expert_admittance")
        result = _annotate_expert_result(result, plan, attempt)
        if failure_mode and not result.failure_reason:
            result.failure_reason = f"deliberate failure: {failure_mode}"
        result.failure_mode = failure_mode
        return plan, attempt, result
    except RuntimeError as exc:
        # A 2-D-feasible plan can still be outside the arm's orientation-
        # constrained workspace. Treat that as a candidate rejection, while
        # allowing unrelated runtime faults to reach the parent process.
        if "UR10 IK target is unreachable" not in str(exc):
            raise
        result = ActRolloutResult(
            success=False,
            failure_reason=f"candidate rejected by IK: {exc}",
            total=int(total),
            target_count=int(cfg.get_path(
                "task.target_count", cfg.get_path("components.count", total))),
            target_indices=[int(value) for value in plan.target_indices],
        )
        return plan, attempt, _annotate_expert_result(result, plan, attempt)
    finally:
        env.close()


def run_expert_episode(cfg, seed: int = 0, collect_observations: bool = True,
                       failure_mode: str = "") -> ActRolloutResult:
    """Execute and physically vet a shortlist of deterministic expert plans.

    Geometry is still the first filter, but one open-loop lane can lose a
    fastener because of solver/contact details that a 2-D clearance test does
    not model.  Replaying the same layout with the next candidate makes this
    a general MuJoCo-in-the-loop expert generator rather than a seed-specific
    patch.  Only the first physically successful exact-count rollout is
    returned; failed attempts are never mixed into a successful demonstration.
    """
    failure_mode = str(failure_mode or "")
    valid_failure_modes = {
        "stall_outside_tray", "wrong_count_over", "wrong_count_under",
        "misroute",
    }
    if failure_mode and failure_mode not in valid_failure_modes:
        raise ValueError(f"unknown expert failure mode: {failure_mode}")
    planning_cfg = cfg
    requested_target_count = int(cfg.get_path(
        "task.target_count", cfg.get_path("components.count", 6)))
    if failure_mode == "wrong_count_over" and requested_target_count >= 6:
        raise ValueError(
            "wrong_count_over is physically impossible with six components "
            "when target_count is 6")
    if failure_mode == "wrong_count_over":
        planning_cfg = cfg.copy()
        planning_cfg.set_path("task.target_count", min(6, requested_target_count + 1))
    elif failure_mode == "wrong_count_under":
        planning_cfg = cfg.copy()
        planning_cfg.set_path("task.target_count", max(1, requested_target_count - 1))
    elif failure_mode == "stall_outside_tray" and requested_target_count > 1:
        planning_cfg = cfg.copy()
        # Plan the full requested group against a deliberately false tray
        # lane.  The real MuJoCo tray is not moved: when this route is
        # executed, the virtual lane crosses the real mouth off-centre and
        # its side wall can retain only part of the carried group.
        planning_cfg.set_path("task.target_count", requested_target_count)
        side = -1.0 if int(seed) % 2 else 1.0
        virtual_offset = float(planning_cfg.get_path(
            "planner.stall_virtual_tray_offset", 0.14))
        planning_cfg.set_path(
            "planner.virtual_tray_y", side * abs(virtual_offset))
    elif failure_mode == "misroute":
        planning_cfg = cfg.copy()
        planning_cfg.set_path("task.target_count", 6)
    probe = SweepEnv(planning_cfg, seed=seed)
    probe.reset(seed=seed)
    try:
        candidates = expert_plan_candidates(probe, planning_cfg)
        if not candidates:
            candidates = [plan_expert_sweep(probe, planning_cfg)]
    finally:
        probe.close()

    best_failure = None
    best_key = None
    validation_workers = max(1, int(planning_cfg.get_path(
        "planner.expert_validation_workers", 1)))
    # Deliberate failures use physically valid candidates, but unlike the
    # success path they are accepted only when their measured outcome matches
    # the requested failure family.  This prevents an ``over`` example that
    # actually collected too few parts from being mislabeled.
    if (failure_mode in {"stall_outside_tray", "wrong_count_under"}
            and requested_target_count == 1):
        # The boundary case has no positive partial target.  Do not spend
        # time physically vetting unrelated candidate lanes for a guaranteed
        # zero-collected record.
        candidates = candidates[:1]
    def execute(candidate: ExpertPlan, attempt: int,
                capture_observations: bool) -> ActRolloutResult:
        env = SweepEnv(cfg, seed=seed)
        env.reset(seed=seed)
        try:
            path, phases = _expert_execution_path(
                cfg, candidate, failure_mode, failure_seed=seed,
                table_top_z=env.table_top_z)
            out = run_action_path(
                env, cfg, path, collect_observations=capture_observations,
                phases=phases,
                target_indices=np.asarray(candidate.target_indices, dtype=int),
                recovery_planner=None, max_recovery_rounds=0,
                failure_mode=failure_mode,
                contact_z_mode="expert_admittance")
            out = _annotate_expert_result(out, candidate, attempt)
            if failure_mode and not out.failure_reason:
                out.failure_reason = f"deliberate failure: {failure_mode}"
            out.failure_mode = failure_mode
            return out
        finally:
            env.close()

    def screen_one(item):
        plan, attempt = item
        try:
            result = execute(plan, attempt, False)
        except RuntimeError as exc:
            # A candidate can pass the 2-D planner yet contain a Cartesian
            # waypoint outside the orientation-constrained workspace.
            if "UR10 IK target is unreachable" not in str(exc):
                raise
            result = ActRolloutResult(
                success=False,
                failure_reason=f"candidate rejected by IK: {exc}",
                total=len(probe.layout),
                target_count=requested_target_count,
                target_indices=[int(value) for value in plan.target_indices],
            )
            result = _annotate_expert_result(result, plan, attempt)
        return plan, attempt, result

    def screened_batches(plans_with_attempts):
        """Screen small parallel batches, yielding results in planner order."""
        items = list(plans_with_attempts)
        for start_index in range(0, len(items), validation_workers):
            batch = items[start_index:start_index + validation_workers]
            if validation_workers == 1 or len(batch) == 1:
                yield [screen_one(item) for item in batch]
            else:
                payloads = [
                    (cfg.to_dict(), int(seed), len(probe.layout), failure_mode,
                     plan, attempt)
                    for plan, attempt in batch
                ]
                with ProcessPoolExecutor(
                        max_workers=len(batch),
                        mp_context=multiprocessing.get_context("spawn")) as pool:
                    # map preserves candidate rank even when rollout runtimes
                    # differ, so selection remains deterministic.
                    screened = list(pool.map(
                        _screen_expert_candidate_process, payloads))
                yield screened

    def outcome_matches(result):
        return (
            (not failure_mode and result.success)
            or (failure_mode == "wrong_count_over"
                and result.collected == requested_target_count + 1)
            or (failure_mode == "wrong_count_under"
                and result.collected == max(0, requested_target_count - 1))
            or (failure_mode == "stall_outside_tray"
                and stall_failure_sidewall_ok(result, cfg))
            or (failure_mode == "misroute" and len(result.trace) >= 100)
        )

    prepared = []
    for attempt, original_plan in enumerate(candidates, start=1):
        plan = original_plan
        if (failure_mode in {"wrong_count_under", "stall_outside_tray"}
                and requested_target_count == 1):
            # There is no positive partial target at goal 1; zero collected is
            # the only physically meaningful boundary case.  Use a normal
            # one-target plan but stop before its tray crossing.
            plan = ExpertPlan(
                target_indices=np.zeros(0, dtype=int),
                waypoints=plan.waypoints,
                feasible=plan.feasible,
                failure_reason=plan.failure_reason,
                score=plan.score,
                turn_count=plan.turn_count,
                strategy=plan.strategy,
                yaw_offset=plan.yaw_offset,
            )
        prepared.append((plan, attempt))

    for batch in screened_batches(prepared):
        for plan, attempt, result in batch:
            matched = outcome_matches(result)
            if matched:
                if collect_observations:
                    result = execute(plan, attempt, True)
                    matched = outcome_matches(result)
                if matched:
                    return result
            # Keep the closest exact-count attempt as the diagnostic return if
            # all candidates fail, while preferring no unexpected parts.
            key = (int(result.unexpected_collected > 0),
                   abs(int(result.target_count) - int(result.target_collected)),
                   -int(result.target_collected), float(plan.score))
            if best_key is None or key < best_key:
                best_key, best_failure = key, result

    # The longer pre-alignment corridor prevents a last-moment yaw hook at the
    # tray mouth, but it can be physically worse for some layouts. Preserve the
    # old mouth-adjacent entry as a fallback, and let the same MuJoCo exact-count
    # check choose between them instead of imposing one geometry on every task.
    alignment_lead = max(0.0, float(planning_cfg.get_path(
        "planner.expert_tray_alignment_lead", 0.16)))
    if best_failure is not None and not failure_mode and alignment_lead > 1e-6:
        fallback_cfg = planning_cfg.copy()
        fallback_cfg.set_path("planner.expert_tray_alignment_lead", 0.0)
        fallback_probe = SweepEnv(fallback_cfg, seed=seed)
        fallback_probe.reset(seed=seed)
        try:
            fallback_candidates = expert_plan_candidates(
                fallback_probe, fallback_cfg)
            if not fallback_candidates:
                fallback_candidates = [
                    plan_expert_sweep(fallback_probe, fallback_cfg)]
        finally:
            fallback_probe.close()

        attempt_offset = len(candidates)
        fallback_items = list(enumerate(
            fallback_candidates, start=attempt_offset + 1))
        for batch in screened_batches(
                [(plan, attempt) for attempt, plan in fallback_items]):
            for plan, fallback_index, result in batch:
                if result.success and collect_observations:
                    result = execute(plan, fallback_index, True)
                if result.success:
                    return result
                key = (int(result.unexpected_collected > 0),
                       abs(int(result.target_count) - int(result.target_collected)),
                       -int(result.target_collected), float(plan.score))
                if best_key is None or key < best_key:
                    best_key, best_failure = key, result

    if best_failure is not None:
        return best_failure
    raise RuntimeError("expert candidate search returned no executable plan")


def _expert_recovery_extension(env: SweepEnv, cfg, round_index: int):
    """Convert one live-state recovery plan into 25 Hz execution references."""
    points = expert_recovery_waypoints(env, cfg)
    if len(points) < 2:
        return np.zeros((0, 4), dtype=np.float32), []
    hz = float(cfg.act.action_hz)
    recovery = sample_polyline(
        points, hz, float(cfg.controller.get(
            "recovery_speed", cfg.controller.sweep_speed)),
        yaw_rate=float(cfg.get_path("controller.yaw_speed_limit", np.deg2rad(45.0))),
        initial_yaw=float(env.ee.tcp_yaw()),
        accel=float(cfg.get_path("planner.accel", 0.8)),
    )
    recovery[:, 2] = float(cfg.workspace.z_search_start)
    return recovery, ["recovery"] * len(recovery)
