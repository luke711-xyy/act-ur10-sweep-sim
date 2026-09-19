"""ACT policy inference against the MuJoCo sweep environment."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .interface import ACTObservationBuilder
from .interface import action_deltas_to_absolute
from .dataset import ActDataset
from .policy import build_act_policy, build_act_processors
from .realtime import ActionChunkScheduler
from .rollout import (
    ActRolloutResult,
    CollectionTerminationTracker,
    _force_loop,
    collection_goal_status,
    collection_step_status,
)


def _resolve_model_path(model_path: str | None) -> str | None:
    """Resolve a model directory or ``latest_checkpoint.txt`` pointer."""
    if not model_path:
        return None
    path = Path(str(model_path))
    if (path / "model.safetensors").exists():
        return str(path)
    pointer = path / "latest_checkpoint.txt"
    if pointer.exists():
        relative = pointer.read_text(encoding="utf-8").strip()
        candidate = path / relative
        if (candidate / "model.safetensors").exists():
            return str(candidate)
    raise FileNotFoundError(
        f"ACT checkpoint not found under {path}; expected model.safetensors "
        "or a valid latest_checkpoint.txt"
    )


def _policy_reference_substep(start: np.ndarray, target: np.ndarray,
                              substep: int, substeps: int, dt: float,
                              max_xy_speed: float, max_z_speed: float,
                              max_yaw_rate: float, contact_latched: bool,
                              admittance_z: float | None = None) -> np.ndarray:
    """Interpolate a 25 Hz ACT target at the 100 Hz control rate.

    The whole 40 ms interval is slew-limited before interpolation.  Airborne
    execution follows ACT in XYZ+yaw.  Once contact is latched, the policy's Z
    target is intentionally ignored and the caller-provided admittance Z is
    used without blending, which makes the ownership transition explicit.
    """
    start = np.asarray(start, dtype=np.float64).reshape(4)
    target = np.asarray(target, dtype=np.float64).reshape(4)
    n_substeps = max(1, int(substeps))
    interval = n_substeps * float(dt)

    delta_xy = target[:2] - start[:2]
    distance = float(np.linalg.norm(delta_xy))
    max_distance = max(0.0, float(max_xy_speed)) * interval
    if distance > max_distance and distance > 1e-12:
        delta_xy *= max_distance / distance

    delta_yaw = float(np.arctan2(
        np.sin(target[3] - start[3]), np.cos(target[3] - start[3])))
    max_yaw = max(0.0, float(max_yaw_rate)) * interval
    delta_yaw = float(np.clip(delta_yaw, -max_yaw, max_yaw))

    delta_z = float(target[2] - start[2])
    max_z = max(0.0, float(max_z_speed)) * interval
    delta_z = float(np.clip(delta_z, -max_z, max_z))

    alpha = float(np.clip(int(substep), 0, n_substeps)) / n_substeps
    if contact_latched:
        if admittance_z is None:
            raise ValueError("admittance_z is required after contact is latched")
        z = float(admittance_z)
    else:
        z = float(start[2] + alpha * delta_z)
    return np.array([
        start[0] + alpha * delta_xy[0],
        start[1] + alpha * delta_xy[1],
        z,
        start[3] + alpha * delta_yaw,
    ], dtype=np.float32)


def _contact_reference_substep(start: np.ndarray, target: np.ndarray,
                               substep: int, substeps: int, dt: float,
                               max_speed: float, max_yaw_rate: float) -> np.ndarray:
    """Backward-compatible XY/yaw view of the full policy interpolator."""
    values = _policy_reference_substep(
        start, target, substep, substeps, dt,
        max_xy_speed=max_speed, max_z_speed=0.0,
        max_yaw_rate=max_yaw_rate, contact_latched=True,
        admittance_z=float(start[2]),
    )
    return values[[0, 1, 3]]


def _inference_dataset_stats(cfg, resolved_model_path: str | None):
    """Return fallback dataset stats only when no checkpoint has saved stats.

    A LeRobot ACT checkpoint carries the image/state normalization processors
    used during training. Re-opening the training manifest during inference is
    unnecessary and can exhaust macOS file/image resources while training is
    active.
    """
    if resolved_model_path:
        return None
    dataset_root = Path(str(cfg.act.dataset_dir))
    if not (dataset_root / "manifest.jsonl").exists():
        return None
    return ActDataset(
        str(dataset_root), chunk_size=int(cfg.act.chunk_size)
    ).stats


def _contact_z_command(cfg, admittance, z_nominal: float,
                       measured_force: float) -> float:
    """Compute contact Z from admittance plus the upward over-force reflex."""
    z = float(z_nominal) + float(admittance.step(
        float(cfg.controller.desired_force), float(measured_force)))
    excess = max(0.0, float(measured_force)
                 - float(cfg.controller.desired_force))
    relief = min(float(cfg.controller.get("overforce_relief_limit", 0.001)),
                 float(cfg.controller.get("overforce_relief_gain", 0.0)) * excess)
    return max(float(cfg.workspace.z_search_min), z + relief)


def run_act_episode(cfg, seed: int = 0, model_path: str | None = None,
                    preview: bool = False) -> ActRolloutResult:
    """Run one causal, full-episode ACT policy from the fixed reset pose.

    No geometric planner, object position, target identity, or supervisor
    staging point is consulted.  ACT owns XYZ+yaw until force contact is
    measured.  At that instant Z ownership transfers continuously to the 1 N
    admittance loop while ACT continues to own X, Y and yaw.
    """
    if str(cfg.act.get("policy_variant", "ordinary")).lower() == "objectact":
        from .object_runtime import run_objectact_episode

        return run_objectact_episode(
            cfg, seed=seed, model_path=model_path, preview=preview
        )
    env = SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    total = len(env.layout)
    if total <= 0:
        env.close()
        return ActRolloutResult(
            success=False, failure_reason="no components in layout",
            target_count=0, total=0, elapsed=float(env.time),
        )
    # The workbench always creates six components, while the selected target
    # is an exact cardinality within that fixed layout.
    target_count = int(np.clip(
        int(cfg.get_path("task.target_count", total)), 1, total
    ))
    resolved_model_path = _resolve_model_path(model_path)
    policy, policy_cfg = build_act_policy(cfg, pretrained_path=resolved_model_path)
    dataset_stats = _inference_dataset_stats(cfg, resolved_model_path)
    preprocessor, postprocessor = build_act_processors(
        policy_cfg, dataset_stats=dataset_stats, pretrained_path=resolved_model_path
    )
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    builder = ACTObservationBuilder(cfg)
    scheduler = ActionChunkScheduler(cfg, allow_late=preview)
    scheduler.reset(env.time)
    admittance = _force_loop(cfg)
    contact = False
    z_nominal = float(env.tcp()[2])
    last_action = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    observations, actions, trace = [], [], []
    pending: Future | None = None
    issued_at = 0.0
    scheduler_queries = 0
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-inference")
    failure = ""
    peak_force = 0.0
    overforce_steps = 0
    path_length = 0.0
    previous_xy = env.tcp()[:2].copy()
    max_frames = max(1, int(cfg.episode.get("max_frames", 500)))
    sampled_frames = 0
    collection_tracker = CollectionTerminationTracker(
        target_count=target_count,
        target_hold_seconds=float(cfg.episode.get(
            "inference_target_hold_time", 2.0)),
        stall_timeout_seconds=float(cfg.episode.get(
            "inference_collection_stall_timeout", 3.0)),
    )
    termination_reason = ""
    contact_time: float | None = None
    contact_loss_steps = 0
    first_contact_position = None

    def predict(observation, start_reference):
        import torch

        batch = builder.torch_batch(observation)
        batch = preprocessor(batch)
        with torch.no_grad():
            normalized = policy.predict_action_chunk(batch)
            deltas = postprocessor(normalized)
        deltas = deltas.detach().cpu().numpy()
        if deltas.ndim != 3 or deltas.shape[-1] != int(policy_cfg.output_features["action"].shape[0]):
            raise ValueError(
                "ACT prediction must have shape (batch, chunk, 4), "
                f"got {deltas.shape}"
            )
        if deltas.shape[0] != 1 or deltas.shape[-1] != 4:
            raise ValueError(f"unsupported ACT action shape {deltas.shape}")
        # Capture the reference at query submission.  Closing over mutable
        # ``last_action`` rebases a slow prediction on a later pose and can
        # create an apparent jump in the replay trace.
        return action_deltas_to_absolute(start_reference, deltas[0])

    try:
        while env.time < float(cfg.episode.max_time):
            if sampled_frames >= max_frames:
                failure = "episode frame limit reached"
                break
            if pending is None and scheduler.query_due(env.time):
                obs = builder.observe(env, contact_latched=contact)
                if len(observations) < 2000:
                    observations.append(obs)
                issued_at = float(env.time)
                scheduler.mark_query(env.time)
                scheduler_queries += 1
                pending = executor.submit(predict, obs, last_action.copy())
            if (preview and pending is not None and not pending.done()
                    and env.time >= issued_at + scheduler.budget):
                # MuJoCo can advance much faster than wall time.  If preview
                # kept stepping while a slow first MPS forward was pending,
                # the whole one-second chunk could expire before it returned.
                # Freeze simulated time at the deadline, then align the late
                # result to the corresponding action index.
                while not pending.done():
                    time.sleep(0.001)
            if pending is not None and pending.done():
                try:
                    values = pending.result()
                    accepted = scheduler.accept(issued_at, values, env.time)
                    if not accepted:
                        failure = ("ACT inference missed 200 ms deadline"
                                   if not preview else
                                   "ACT inference chunk expired before preview result arrived")
                        break
                except Exception as exc:  # surface a concise rollout failure
                    failure = f"ACT inference failed: {exc}"
                    break
                pending = None

            action = scheduler.action_for(env.time)
            if action is None:
                # A missed chunk is held at the latest safe absolute pose.
                action = last_action.copy()
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (4,):
                raise ValueError(f"executor action must have shape (4,), got {action.shape}")
            action[0] = np.clip(action[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
            action[1] = np.clip(action[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
            action[2] = np.clip(action[2], float(cfg.workspace.z_search_min),
                                float(cfg.end_effector.z_home))
            action[3] = np.arctan2(np.sin(action[3]), np.cos(action[3]))
            actions.append(action.copy())
            substeps = max(1, int(round(
                float(cfg.sim.control_hz) / float(cfg.act.action_hz))))
            command_start = last_action.copy()
            final_reference = command_start.copy()
            frame_contact_at_start = bool(contact)
            if contact:
                phase = ("contact_build" if contact_time is not None and
                         env.time - contact_time < float(cfg.episode.contact_build_time)
                         else "sweep")
            elif (float(action[2]) < float(cfg.end_effector.z_home) - 0.005
                  or float(command_start[2]) < float(cfg.end_effector.z_home) - 0.005):
                phase = "descent"
            else:
                phase = "approach"
            for substep in range(1, substeps + 1):
                measured = float(env.normal_force())
                peak_force = max(peak_force, measured)
                admittance_z = (_contact_z_command(
                    cfg, admittance, z_nominal, measured) if contact else None)
                reference = _policy_reference_substep(
                    command_start, action, substep, substeps,
                    1.0 / float(cfg.sim.control_hz),
                    max_xy_speed=float(
                        cfg.controller.sweep_speed if contact
                        else cfg.controller.get("travel_speed", cfg.controller.sweep_speed)),
                    max_z_speed=float(cfg.controller.z_speed),
                    max_yaw_rate=float(cfg.controller.yaw_speed_limit),
                    contact_latched=contact,
                    admittance_z=admittance_z,
                )
                reference[0] = np.clip(
                    reference[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
                reference[1] = np.clip(
                    reference[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
                reference[2] = np.clip(
                    reference[2], float(cfg.workspace.z_search_min),
                    float(cfg.end_effector.z_home))
                x, y, z, yaw = (float(value) for value in reference)
                try:
                    env.step_control(Command(x, y, z, yaw))
                except (RuntimeError, ValueError) as exc:
                    failure = f"ACT reference failed IK or collision validation: {exc}"
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
                trace.append({"t": env.time, "tcp": tcp.copy(),
                              "command": np.array([x, y, z, yaw],
                                                   dtype=np.float32),
                              "policy_reference": action.copy(),
                              "policy_z": float(action[2]),
                              "applied_z": float(z),
                              "z_owner": ("admittance" if frame_contact_at_start
                                          else ("transition" if contact else "policy")),
                              "normal_force": post_force, "contact": contact,
                              "collected": int(collected_mask.sum()),
                              "phase": phase})
                if contact:
                    collection_step_status(
                        collected_mask, None, target_count,
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
                if contact_loss_steps >= max(1, int(round(
                        0.1 * float(cfg.sim.control_hz)))):
                    failure = "contact lost for 100 ms"
                    break
                if path_length > float(cfg.episode.max_contact_path):
                    failure = "contact path exceeded limit"
                    break
                unsafe_collision = getattr(env, "unsafe_robot_collision", None)
                if callable(unsafe_collision) and unsafe_collision():
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
        env.collected_mask(), None, target_count,
        partial_overlap_mask=env.partial_collection_overlap_mask()
    )
    if not failure and not contact:
        failure = "contact was not established"
    if not failure and not exact_goal:
        failure = (f"{termination_reason}: {exact_reason}"
                   if termination_reason and exact_reason
                   else exact_reason or "target count was not collected")
    if not failure and not termination_reason:
        failure = "inference termination condition not reached"
    result = ActRolloutResult(success=not failure and exact_goal
                              and bool(termination_reason) and total > 0,
                              failure_reason=failure,
                              observations=observations,
                              actions=np.asarray(actions, dtype=np.float32),
                              trace=trace, collected=collected, total=total,
                              target_count=target_count,
                              target_collected=collected,
                              unexpected_collected=max(0, collected - target_count),
                              final_collected_mask=np.asarray(
                                  env.collected_mask(), dtype=bool).copy(),
                              elapsed=float(env.time),
                              scheduler_queries=scheduler_queries,
                              scheduler_timeouts=int(scheduler.timeouts),
                              scheduler_late_results=int(scheduler.late_results),
                              peak_force=float(peak_force),
                              sampled_frames=int(sampled_frames),
                              termination_reason=termination_reason,
                              first_contact_position=(
                                  np.asarray(first_contact_position, dtype=np.float32)
                                  if first_contact_position is not None
                                  else None))
    env.close()
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ACT policy episode")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--target-count", type=int, default=None,
                        help="exact number of the six components to collect")
    parser.add_argument("--record-replay", action="store_true",
                        help="save a completed rollout for workbench playback")
    parser.add_argument("--replay-root", default="runs/workbench_previews")
    parser.add_argument("--preview", action="store_true",
                        help="allow late inference and align chunks for simulation preview")
    return parser


def main(argv=None) -> int:
    from ..config import load_config

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.target_count is not None and not 1 <= args.target_count <= 6:
        parser.error("--target-count must be between 1 and 6")
    cfg = load_config(args.config)
    if args.target_count is not None:
        cfg.set_path("task.target_count", args.target_count)
    result = run_act_episode(cfg, seed=args.seed, model_path=args.model,
                             preview=args.preview)
    replay_episode_id = None
    replay_error = ""
    if args.record_replay:
        try:
            from .replay import save_inference_replay

            replay_episode_id = save_inference_replay(
                cfg, args.seed, result, args.replay_root, model=args.model
            )
        except Exception as exc:  # keep the physical result visible if replay fails
            replay_error = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"success": result.success, "reason": result.failure_reason,
                      "collected": result.collected, "target_count": result.target_count,
                      "total": result.total, "elapsed": result.elapsed,
                      "scheduler_actions": len(result.actions),
                      "scheduler_queries": result.scheduler_queries,
                      "scheduler_timeouts": result.scheduler_timeouts,
                      "scheduler_late_results": result.scheduler_late_results,
                      "peak_force": result.peak_force,
                      "termination_reason": result.termination_reason,
                      "preview": bool(args.preview),
                      "replay_episode_id": replay_episode_id,
                      "replay_error": replay_error}, ensure_ascii=False))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
