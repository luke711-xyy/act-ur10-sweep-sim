"""Schema-v5 ObjectACT runtime loop.

The ordinary ACT evaluator remains the reference implementation.  This module
keeps the ObjectACT path separate so it cannot accidentally import the expert
planner or feed simulator object truth into the policy observation.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .object_interface import ObjectACTObservationBuilder, RGBObjectPerceptionFrontend
from .object_training import ObjectACTPreprocessor
from .realtime import ActionChunkScheduler
from .rollout import (
    ActRolloutResult,
    CollectionTerminationTracker,
    collection_goal_status,
    collection_step_status,
    _force_loop,
)


def run_objectact_episode(
    cfg,
    seed: int = 0,
    model_path: str | None = None,
    preview: bool = False,
) -> ActRolloutResult:
    """Run one v5 RGB-perception/ObjectACT episode.

    The policy starts from the normal reset pose and owns ``[dx, dy, dz,
    dyaw]`` from the first action.  No A* route, target object IDs, or object
    positions are consulted.  Once force contact is latched, only the
    applied Z is replaced by the admittance controller; the policy keeps
    producing XY/yaw actions while collection timers are active.
    """
    import torch

    from .evaluate import (
        _contact_z_command,
        _policy_reference_substep,
    )
    from .policy import build_objectact_policy

    env = SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    total = len(env.layout)
    if total <= 0:
        env.close()
        return ActRolloutResult(
            success=False,
            failure_reason="no components in layout",
            target_count=0,
            total=0,
            elapsed=float(env.time),
        )

    target_count = int(np.clip(
        int(cfg.get_path("task.target_count", total)), 1, total
    ))
    policy, object_config, _checkpoint = build_objectact_policy(cfg, model_path)
    device = str(policy.config.device)
    checkpoint = _checkpoint
    normalizer_path = (
        checkpoint / "normalizer.json" if checkpoint is not None else None
    )
    if normalizer_path is None or not normalizer_path.exists():
        env.close()
        raise FileNotFoundError(
            "ObjectACT inference requires a schema-v5 checkpoint with normalizer.json"
        )
    preprocessor = ObjectACTPreprocessor.load(normalizer_path, device=device)
    detector_checkpoint = cfg.act.get("objectact_detector_checkpoint", None)
    frontend = RGBObjectPerceptionFrontend(
        cfg,
        device=device,
        detector_checkpoint=detector_checkpoint,
    )
    builder = ObjectACTObservationBuilder(cfg, frontend=frontend)
    builder.reset()
    scheduler = ActionChunkScheduler(cfg, allow_late=preview)
    scheduler.reset(env.time)
    admittance = _force_loop(cfg)
    contact = False
    z_nominal = float(env.tcp()[2])
    last_action = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    observations: list[dict] = []
    actions: list[np.ndarray] = []
    trace: list[dict] = []
    pending: Future | None = None
    issued_at = 0.0
    scheduler_queries = 0
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="objectact-inference")
    failure = ""
    peak_force = 0.0
    overforce_steps = 0
    path_length = 0.0
    previous_xy = env.tcp()[:2].copy()
    max_frames = max(1, int(cfg.episode.get("max_frames", 500)))
    sampled_frames = 0
    collection_tracker = CollectionTerminationTracker(
        target_count=target_count,
        target_hold_seconds=float(cfg.episode.get("inference_target_hold_time", 2.0)),
        stall_timeout_seconds=float(cfg.episode.get("inference_collection_stall_timeout", 3.0)),
    )
    termination_reason = ""
    contact_time: float | None = None
    contact_loss_steps = 0
    first_contact_position = None

    def predict(observation: dict, start_reference: np.ndarray):
        batch = builder.torch_batch(observation, device=device)
        batch = preprocessor(batch)
        with torch.no_grad():
            normalized = policy.predict_action_chunk(batch)
            deltas = preprocessor.unnormalize("action", normalized)
        values = deltas.detach().cpu().numpy()
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != 4:
            raise ValueError(f"ObjectACT prediction must be (1, chunk, 4), got {values.shape}")
        from .interface import action_deltas_to_absolute

        return action_deltas_to_absolute(start_reference, values[0])

    try:
        while env.time < float(cfg.episode.max_time):
            if sampled_frames >= max_frames:
                failure = "episode frame limit reached"
                break
            if pending is None and scheduler.query_due(env.time):
                observation = builder.observe(env, contact_latched=contact)
                if len(observations) < 2000:
                    observations.append(observation)
                issued_at = float(env.time)
                scheduler.mark_query(env.time)
                scheduler_queries += 1
                pending = executor.submit(predict, observation, last_action.copy())
            if (preview and pending is not None and not pending.done()
                    and env.time >= issued_at + scheduler.budget):
                while not pending.done():
                    time.sleep(0.001)
            if pending is not None and pending.done():
                try:
                    values = pending.result()
                    first_preview_alignment = (
                        preview
                        and not scheduler.has_active_action(env.time)
                        and float(env.time) > issued_at + scheduler.budget + 1e-9
                    )
                    if first_preview_alignment:
                        measured_reference = np.array(
                            [*env.tcp(), float(env.ee.tcp_yaw())], dtype=np.float32
                        )
                        accepted = scheduler.accept_preview_aligned(
                            issued_at,
                            values,
                            env.time,
                            current_reference=measured_reference,
                        )
                        if accepted:
                            last_action = measured_reference.copy()
                    else:
                        accepted = scheduler.accept(issued_at, values, env.time)
                    if not accepted:
                        failure = (
                            "ObjectACT inference missed 200 ms deadline"
                            if not preview
                            else "ObjectACT inference chunk expired before preview result arrived"
                        )
                        break
                except Exception as exc:
                    failure = f"ObjectACT inference failed: {exc}"
                    break
                pending = None

            action = scheduler.action_for(env.time)
            if action is None:
                action = last_action.copy()
            action = np.asarray(action, dtype=np.float32).reshape(4)
            action[0] = np.clip(action[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
            action[1] = np.clip(action[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
            action[2] = np.clip(action[2], float(cfg.workspace.z_search_min), float(cfg.end_effector.z_home))
            action[3] = np.arctan2(np.sin(action[3]), np.cos(action[3]))
            actions.append(action.copy())
            substeps = max(1, int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz))))
            command_start = last_action.copy()
            final_reference = command_start.copy()
            frame_contact_at_start = bool(contact)
            phase = (
                "contact_build" if contact and contact_time is not None
                and env.time - contact_time < float(cfg.episode.contact_build_time)
                else ("sweep" if contact else (
                    "descent" if action[2] < float(cfg.end_effector.z_home) - 0.005
                    or command_start[2] < float(cfg.end_effector.z_home) - 0.005
                    else "approach"))
            )
            for substep in range(1, substeps + 1):
                measured = float(env.normal_force())
                peak_force = max(peak_force, measured)
                admittance_z = _contact_z_command(cfg, admittance, z_nominal, measured) if contact else None
                reference = _policy_reference_substep(
                    command_start,
                    action,
                    substep,
                    substeps,
                    1.0 / float(cfg.sim.control_hz),
                    max_xy_speed=float(
                        cfg.controller.sweep_speed if contact
                        else cfg.controller.get("travel_speed", cfg.controller.sweep_speed)
                    ),
                    max_z_speed=float(cfg.controller.z_speed),
                    max_yaw_rate=float(cfg.controller.yaw_speed_limit),
                    contact_latched=contact,
                    admittance_z=admittance_z,
                )
                reference[0] = np.clip(reference[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
                reference[1] = np.clip(reference[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
                reference[2] = np.clip(reference[2], float(cfg.workspace.z_search_min), float(cfg.end_effector.z_home))
                x, y, z, yaw = (float(value) for value in reference)
                try:
                    env.step_control(Command(x, y, z, yaw))
                except (RuntimeError, ValueError) as exc:
                    failure = f"ObjectACT reference failed IK or collision validation: {exc}"
                    break
                post_force = float(env.normal_force())
                peak_force = max(peak_force, post_force)
                tcp = env.tcp()
                if not contact:
                    near_table = float(tcp[2]) <= float(cfg.workspace.z_search_start) + 0.025
                    if near_table and post_force >= float(cfg.controller.contact_threshold):
                        contact = True
                        z_nominal = float(tcp[2])
                        admittance.reset()
                        contact_time = float(env.time)
                        first_contact_position = np.asarray(tcp, dtype=np.float32).copy()
                if contact:
                    path_length += float(np.linalg.norm(tcp[:2] - previous_xy))
                previous_xy = tcp[:2].copy()
                final_reference = reference.copy()
                collected_mask = np.asarray(env.collected_mask(), dtype=bool).copy()
                trace.append({
                    "t": env.time,
                    "tcp": tcp.copy(),
                    "command": np.array([x, y, z, yaw], dtype=np.float32),
                    "policy_reference": action.copy(),
                    "policy_z": float(action[2]),
                    "applied_z": float(z),
                    "z_owner": "admittance" if frame_contact_at_start else ("transition" if contact else "policy"),
                    "normal_force": post_force,
                    "contact": contact,
                    "collected": int(collected_mask.sum()),
                    "phase": phase,
                })
                if contact:
                    collection_step_status(
                        collected_mask,
                        None,
                        target_count,
                        partial_overlap_mask=env.partial_collection_overlap_mask(),
                        stop_on_contract_violation=False,
                    )
                pending_stop = collection_tracker.update(collected_mask, env.time)
                if contact and post_force > float(cfg.controller.safe_max_force):
                    overforce_steps += 1
                else:
                    overforce_steps = 0
                if contact and overforce_steps >= int(cfg.controller.get("safe_force_dwell_steps", 3)):
                    failure = "normal force exceeded safety threshold"
                    break
                if contact and post_force < float(cfg.controller.release_threshold):
                    contact_loss_steps += 1
                else:
                    contact_loss_steps = 0
                if contact_loss_steps >= max(1, int(round(0.1 * float(cfg.sim.control_hz)))):
                    failure = "contact lost for 100 ms"
                    break
                if path_length > float(cfg.episode.max_contact_path):
                    failure = "contact path exceeded limit"
                    break
                if callable(getattr(env, "unsafe_robot_collision", None)) and env.unsafe_robot_collision():
                    failure = "unsafe robot collision"
                    break
                if np.any(env.lost_mask()):
                    failure = "component left the safe workspace"
                    break
                if pending_stop:
                    termination_reason = pending_stop
                    break
            last_action = final_reference.astype(np.float32, copy=True)
            sampled_frames += 1
            if failure or termination_reason:
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    collected = int(env.collected_mask().sum())
    exact_goal, exact_reason = collection_goal_status(
        env.collected_mask(),
        None,
        target_count,
        partial_overlap_mask=env.partial_collection_overlap_mask(),
    )
    if not failure and not contact:
        failure = "contact was not established"
    if not failure and not exact_goal:
        failure = (f"{termination_reason}: {exact_reason}" if termination_reason and exact_reason else exact_reason or "target count was not collected")
    if not failure and not termination_reason:
        failure = "inference termination condition not reached"
    result = ActRolloutResult(
        success=not failure and exact_goal and bool(termination_reason) and total > 0,
        failure_reason=failure,
        observations=observations,
        actions=np.asarray(actions, dtype=np.float32),
        trace=trace,
        collected=collected,
        total=total,
        target_count=target_count,
        target_collected=collected,
        unexpected_collected=max(0, collected - target_count),
        final_collected_mask=np.asarray(env.collected_mask(), dtype=bool).copy(),
        elapsed=float(env.time),
        scheduler_queries=scheduler_queries,
        scheduler_timeouts=int(scheduler.timeouts),
        scheduler_late_results=int(scheduler.late_results),
        peak_force=float(peak_force),
        sampled_frames=int(sampled_frames),
        termination_reason=termination_reason,
        first_contact_position=(
            np.asarray(first_contact_position, dtype=np.float32)
            if first_contact_position is not None else None
        ),
    )
    env.close()
    return result
