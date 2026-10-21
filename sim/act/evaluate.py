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
    _overforce_relief_loop,
    collection_goal_status,
)


INFERENCE_MAX_FRAMES = 2000


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
                       measured_force: float, relief=None) -> float:
    """Compute contact Z from admittance plus the upward over-force reflex."""
    if relief is None:
        relief = _overforce_relief_loop(cfg)
    correction = (
        float(admittance.step(
            float(cfg.controller.desired_force), float(measured_force)))
        + float(relief.step(float(measured_force)))
    )
    correction_limit = abs(float(cfg.controller.delta_z_limit))
    z = float(z_nominal) + float(np.clip(
        correction, -correction_limit, correction_limit))
    return max(float(cfg.workspace.z_search_min), z)


def run_act_episode(cfg, seed: int = 0, model_path: str | None = None,
                    preview: bool = False, capture_replay: bool = False,
                    policy_type: str = "act", wall_seconds: float = 1200,
                    resource_check=None) -> ActRolloutResult:
    """Run one causal, full-episode ACT policy from the fixed reset pose.

    No geometric planner, object position, target identity, or supervisor
    staging point is consulted.  ACT owns XYZ+yaw until force contact is
    measured. At that instant Z ownership transfers continuously to the
    configured wrist-force admittance loop while ACT continues to own X, Y,
    and yaw.
    """
    if policy_type not in {"act", "diffusion"}:
        raise ValueError("policy_type must be act or diffusion")
    is_diffusion = policy_type == "diffusion"
    wall_started = time.perf_counter()
    runtime = None
    if is_diffusion:
        from sim.diffusion.runtime import DiffusionRuntime
        runtime = DiffusionRuntime(model_path, cfg, seed=seed)
    env = SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    if capture_replay:
        import mujoco
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state_size = mujoco.mj_stateSize(env.model, state_spec)
        # Separate history: audit capture must not advance the policy builder.
        replay_builder = ACTObservationBuilder(cfg)
        initial_object_poses = np.concatenate((
            env.component_positions(), env.component_quats()), axis=1).astype("<f4")
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
    if not is_diffusion:
        resolved_model_path = _resolve_model_path(model_path)
        policy, policy_cfg = build_act_policy(cfg, pretrained_path=resolved_model_path)
        dataset_stats = _inference_dataset_stats(cfg, resolved_model_path)
        preprocessor, postprocessor = build_act_processors(
            policy_cfg, dataset_stats=dataset_stats, pretrained_path=resolved_model_path
        )
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
    policy_inputs = None if is_diffusion else getattr(policy_cfg, "input_features", None)
    if policy_inputs:
        policy_views = [
            key.removeprefix("observation.images.")
            for key in policy_inputs
            if key.startswith("observation.images.")
        ]
        policy_env_dim = int(
            policy_inputs["observation.environment_state"].shape[0]
        )
    else:
        policy_views = list(cfg.act.get(
            "camera_views", ["overhead", "wrist"]
        ))
        policy_env_dim = (1 if bool(cfg.act.get(
            "ordinary_target_count_only", False)) else 3)
    builder = ACTObservationBuilder(
        cfg, camera_views=policy_views,
        environment_state_dim=policy_env_dim,
    )
    if is_diffusion:
        from sim.diffusion.runtime import RollingChunk
        scheduler = RollingChunk()
    else:
        scheduler = ActionChunkScheduler(cfg, allow_late=preview)
    scheduler.reset(env.time)
    admittance = _force_loop(cfg)
    overforce_relief = _overforce_relief_loop(cfg)
    contact = False
    z_nominal = float(env.tcp()[2])
    last_action = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    observations, actions, trace = [], [], []
    pending: Future | None = None
    issued_at = 0.0
    scheduler_queries = 0
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-inference")
    peak_force = 0.0
    max_frames = INFERENCE_MAX_FRAMES
    sampled_frames = 0
    termination_reason = ""
    collection_timer = CollectionTerminationTracker(target_count=target_count)
    contact_time: float | None = None
    first_contact_position = None

    def capture_state(row):
        if not capture_replay:
            return
        state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, state, state_spec)
        row.update({
            "replay_state": {"spec": int(state_spec), "values": state},
            "object_collected": np.asarray(env.collected_mask(), dtype=bool).copy(),
            "wrench": np.asarray(env.wrench(), dtype=np.float64).copy(),
            "raw_normal_force": float(env.raw_normal_force()),
            "compensated_normal_force_raw": float(env.compensated_normal_force_raw()),
            "tcp_pose": np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float64),
            "object_pose": np.concatenate((env.component_positions(),
                                           env.component_quats()), axis=1),
            "replay_observation": replay_builder.observe(
                env, contact_latched=contact, include_images=False),
        })

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
        while sampled_frames < max_frames:
            if is_diffusion:
                if resource_check is not None:
                    resource_check()
                if time.perf_counter()-wall_started >= wall_seconds:
                    termination_reason = "wall-clock limit"
                    break
                runtime.history.append(builder.observe(env,contact_latched=contact))
                if scheduler.query_due(env.time):
                    issued_at = float(env.time)
                    values = action_deltas_to_absolute(last_action.copy(),runtime.predict())
                    if time.perf_counter()-wall_started >= wall_seconds:
                        termination_reason = "wall-clock limit"
                        break
                    scheduler.accept(issued_at,values,env.time)
                    scheduler_queries += 1
            elif pending is None and scheduler.query_due(env.time):
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
                        # Expired chunks are discarded; the last valid absolute
                        # pose remains the command until a later prediction lands.
                        pass
                except Exception:
                    # A policy/runtime prediction error is non-terminal. Clear
                    # this future and keep the last valid command while the
                    # next scheduled query is attempted.
                    pass
                pending = None

            try:
                action = scheduler.action_for(env.time)
            except Exception:
                if is_diffusion:
                    raise
                action = None
            if action is None:
                # Missing, late, or invalid policy output holds the last valid
                # reference rather than ending the rollout.
                action = last_action.copy()
            try:
                action = np.asarray(action, dtype=np.float32)
            except Exception:
                action = last_action.copy()
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                if is_diffusion:
                    raise ValueError("invalid DP action; inference terminated")
                action = last_action.copy()
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
                    cfg, admittance, z_nominal, measured,
                    relief=overforce_relief) if contact else None)
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
                except Exception:
                    if is_diffusion:
                        raise
                    # Rejecting a policy reference (for example, unreachable
                    # IK or collision validation) is also non-terminal. Hold
                    # the measured pose; if even that command is rejected,
                    # count the outer inference frame and try again next frame.
                    try:
                        tcp = np.asarray(env.tcp(), dtype=np.float32).reshape(3)
                        hold_yaw = float(env.ee.tcp_yaw())
                        env.step_control(Command(
                            float(tcp[0]), float(tcp[1]), float(tcp[2]), hold_yaw
                        ))
                    except Exception as rejected:
                        if capture_replay:
                            row = {"t": env.time, "tcp": env.tcp().copy(),
                                   "command": np.array([x, y, z, yaw], dtype=np.float32),
                                   "policy_reference": action.copy(), "policy_z": float(action[2]),
                                   "applied_z": float(z), "phase": phase,
                                   "z_owner": "admittance" if frame_contact_at_start else "policy",
                                   "normal_force": float(env.normal_force()), "contact": contact,
                                   "collected": int(env.collected_mask().sum()),
                                   "control_step_advanced": False,
                                   "rejection_error": f"{type(rejected).__name__}: {rejected}"}
                            capture_state(row)
                            trace.append(row)
                        final_reference = last_action.copy()
                        continue
                    x, y, z, yaw = (float(tcp[0]), float(tcp[1]),
                                    float(tcp[2]), hold_yaw)
                    reference = np.array([x, y, z, yaw], dtype=np.float32)
                post_force = float(env.normal_force())
                peak_force = max(peak_force, post_force)
                tcp = env.tcp()
                if not contact:
                    near_table = float(tcp[2]) <= float(cfg.workspace.z_search_start) + 0.025
                    if near_table and post_force >= float(cfg.controller.contact_threshold):
                        contact = True
                        z_nominal = float(tcp[2])
                        admittance.reset()
                        overforce_relief.reset()
                        contact_time = float(env.time)
                        first_contact_position = np.asarray(tcp, dtype=np.float32).copy()
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
                capture_state(trace[-1])
                # Observe every physical control step, not only 25 Hz image
                # frames: a transient full entry must start the deadline and
                # contribute to the achievement count even if it later exits.
                termination_reason = collection_timer.update(
                    collected_mask, float(env.time),
                    partial_overlap_mask=env.partial_collection_overlap_mask(),
                )
                if termination_reason:
                    break
            # Keep the same 25 Hz state-history cadence used by expert
            # demonstrations even though ACT is queried only at 5 Hz.
            builder.advance_state_history(env, contact_latched=contact)
            last_action = final_reference.astype(np.float32, copy=True)
            sampled_frames += 1

            if termination_reason:
                break
    except BaseException:
        env.close()
        raise
    finally:
        # A terminal scene can still have a running prediction. Drain it before
        # the CLI releases its process lease; cancellation only stops queued work.
        executor.shutdown(wait=True, cancel_futures=True)

    final_mask = np.asarray(env.collected_mask(), dtype=bool).copy()
    final_collected = int(final_mask.sum())
    collected = int(collection_timer.peak_collected)
    exact_goal, goal_reason = collection_goal_status(
        final_mask, None, target_count,
        partial_overlap_mask=env.partial_collection_overlap_mask()
    )
    success = bool(exact_goal and termination_reason == "exact target count reached"
                   and total > 0)
    failure_reason = ("" if success else
                      f"{termination_reason}: {goal_reason}" if termination_reason else
                      f"{max_frames}-frame timeout before task completion")
    result = ActRolloutResult(success=success,
                              failure_reason=failure_reason,
                              observations=observations,
                              actions=np.asarray(actions, dtype=np.float32),
                              trace=trace, collected=collected, total=total,
                              target_count=target_count,
                              target_collected=collected,
                              unexpected_collected=max(0, collected - target_count),
                              final_collected_mask=final_mask,
                              peak_collected=collected,
                              final_collected=final_collected,
                              first_entry_time=collection_timer.first_entry_time,
                              collection_window_seconds=collection_timer.window_seconds,
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
    if is_diffusion:
        result.policy_type = "diffusion"
        result.inference_latencies = list(runtime.latencies)
        result.wall_seconds = time.perf_counter()-wall_started
    if capture_replay:
        result.initial_object_poses = initial_object_poses
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ACT policy episode")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--policy-type", choices=("act", "diffusion"), default="act")
    parser.add_argument("--wall-seconds", type=float, default=1200)
    parser.add_argument("--target-count", type=int, default=None,
                        help="exact number of the six components to collect")
    parser.add_argument("--record-replay", action="store_true",
                        help="save a completed rollout for workbench playback")
    parser.add_argument("--replay-root", default="runs/workbench_previews")
    parser.add_argument("--replay-image-format", choices=("png", "webp-lossless"),
                        default="png", help="saved replay encoding, not policy input encoding")
    parser.add_argument("--video-root", default=None,
                        help="automatic MP4 directory (defaults to workbench configuration)")
    parser.add_argument("--defer-video", action="store_true",
                        help="publish replay for a batch media worker; retain its frame images")
    parser.add_argument("--preview", action="store_true",
                        help="allow late inference and align chunks for simulation preview")
    return parser


def _run_main(argv=None) -> int:
    from ..config import load_config

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.target_count is not None and not 1 <= args.target_count <= 6:
        parser.error("--target-count must be between 1 and 6")
    cfg = load_config(args.config)
    resource_check = None
    if args.policy_type=="diffusion":
        from sim.diffusion.runtime import inference_resource_guard
        workspace = Path(__file__).resolve().parents[2]
        args.video_root = args.video_root or str(workspace/"runs/diffusion_policy_mvp/videos")
        resource_check = inference_resource_guard(workspace,args.replay_root,args.video_root)
    if args.target_count is not None:
        cfg.set_path("task.target_count", args.target_count)
    # Snapshot latest once, before loading; a concurrent trainer can update the
    # pointer while inference or video export is running.
    if args.policy_type == "diffusion":
        from sim.diffusion.runtime import resolve_checkpoint
        resolved_model = resolve_checkpoint(args.model)
    else:
        resolved_model = _resolve_model_path(args.model)
    if resolved_model:
        resolved_model = str(Path(resolved_model).resolve())
    extra = ({"policy_type": "diffusion", "wall_seconds": args.wall_seconds,
              "resource_check": resource_check}
             if args.policy_type == "diffusion" else {})
    result = run_act_episode(cfg, seed=args.seed, model_path=resolved_model,
                             preview=args.preview, capture_replay=args.record_replay, **extra)
    physical_summary = {"success": result.success, "reason": result.failure_reason,
                        "collected": result.collected, "target_count": result.target_count,
                        "peak_collected": (result.collected if getattr(
                            result, "peak_collected", None) is None else result.peak_collected),
                        "final_collected": (result.collected if getattr(
                            result, "final_collected", None) is None else result.final_collected),
                        "first_entry_time": getattr(result, "first_entry_time", None),
                        "collection_window_seconds": getattr(
                            result, "collection_window_seconds", None),
                        "total": result.total, "elapsed": result.elapsed,
                        "scheduler_actions": len(result.actions),
                        "scheduler_queries": result.scheduler_queries,
                        "scheduler_timeouts": result.scheduler_timeouts,
                        "scheduler_late_results": result.scheduler_late_results,
                        "peak_force": result.peak_force,
                        "termination_reason": result.termination_reason,
                        "preview": bool(args.preview),
                        "model_checkpoint": resolved_model}
    if args.policy_type == "diffusion":
        physical_summary.update(policy_type="diffusion",
            inference_latencies=getattr(result,"inference_latencies",[]),
            wall_seconds=getattr(result,"wall_seconds",None),
            real_time_200ms_claim=False)
    replay_episode_id = None
    replay_error = ""
    video_path = None
    video_error = ""
    frame_images_deleted = False
    image_cleanup_error = ""
    replay_input_path = None
    if args.record_replay:
        if resource_check is not None:
            resource_check()
        # Replay uses the causal command trace, not the cached query images.
        # Release those after inference, before rendering another five views.
        if getattr(result, "observations", None) is not None:
            result.observations.clear()
        try:
            from .replay import persist_replay_input, save_inference_replay

            replay_input_path = str(persist_replay_input(
                cfg, args.seed, result, args.replay_root, model=resolved_model))
            print(json.dumps({**physical_summary, "event": "physical_result_saved",
                              "replay_input_path": replay_input_path}, ensure_ascii=False), flush=True)

            replay_episode_id = save_inference_replay(
                cfg, args.seed, result, args.replay_root, model=resolved_model,
                image_format=args.replay_image_format,
                **({"resource_check":resource_check} if resource_check is not None else {}),
            )
        except Exception as exc:  # keep the physical result visible if replay fails
            replay_error = f"{type(exc).__name__}: {exc}"
        if replay_episode_id:
            # JobRegistry persists stdout as it arrives. Make the physical
            # outcome visible even while a later encoder is running/failing.
            print(json.dumps({**physical_summary, "event": "inference_result",
                              "replay_episode_id": replay_episode_id,
                              "replay_error": replay_error}, ensure_ascii=False), flush=True)
        if replay_episode_id and not args.defer_video:
            try:
                from .video import export_inference_video, inference_video_root

                output_root = args.video_root or inference_video_root(cfg, Path.cwd())
                video_path = str(export_inference_video(
                    Path(args.replay_root) / replay_episode_id, output_root,
                    **({"resource_check":resource_check} if resource_check is not None else {})))
                from .media_retention import finalize_inference_video
                try:
                    finalize_inference_video(
                        Path(args.replay_root) / replay_episode_id, video_path)
                    frame_images_deleted = True
                except Exception as exc:
                    image_cleanup_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # video errors must not replace rollout status
                video_error = f"{type(exc).__name__}: {exc}"
    print(json.dumps({**physical_summary,
                      "replay_episode_id": replay_episode_id,
                      "replay_input_path": replay_input_path,
                      "replay_error": replay_error,
                      "video_path": video_path,
                      "video_error": video_error,
                      "video_deferred": bool(args.defer_video and replay_episode_id),
                      "frame_images_deleted": frame_images_deleted,
                      "image_cleanup_error": image_cleanup_error}, ensure_ascii=False), flush=True)
    if args.policy_type=="diffusion" and (replay_error or video_error):
        raise RuntimeError(f"DP media infrastructure failed: {replay_error or video_error}")
    return 0 if result.success else 1


def main(argv=None) -> int:
    from .inference_lock import inference_lease
    with inference_lease(on_wait=lambda: print(json.dumps({
            "event": "waiting_for_inference_slot"}), flush=True)):
        return _run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
