"""Render a completed ACT rollout into the workbench episode format."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .dataset import ActDatasetWriter
from .interface import ACTObservationBuilder
from .rollout import policy_action_delta


def _angle_delta(value: float, previous: float) -> float:
    return float(np.arctan2(np.sin(value - previous), np.cos(value - previous)))


def _episode_id(root: Path, target_count: int, seed: int) -> str:
    """Return a collision-free id that is visibly an inference replay."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"inference_{stamp}_n{int(target_count)}_s{int(seed)}"
    candidate = base
    ordinal = 2
    while (root / candidate).exists():
        candidate = f"{base}_{ordinal:02d}"
        ordinal += 1
    return candidate


def save_inference_replay(cfg, seed: int, result, root: str | Path,
                          model: str | None = None) -> str | None:
    """Replay the recorded commands and save three cameras plus telemetry.

    The inference process itself does not render frames, so its 5 Hz/25 Hz
    timing budget is not distorted by image encoding.  After the episode, the
    exact command trace is run through a fresh, same-seed MuJoCo scene and
    sampled at the configured action rate.  The resulting record is consumed
    by the existing workbench episode browser and therefore supports the same
    frame slider, Play button and Fz/telemetry plots as demonstrations.
    """
    trace = list(getattr(result, "trace", ()))
    if not trace:
        return None
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target_count = int(getattr(result, "target_count", 0)
                       or cfg.get_path("task.target_count", 0))
    replay_cfg = cfg.copy()
    replay_cfg.set_path("task.target_count", target_count)
    env = SweepEnv(replay_cfg, seed=int(seed))
    env.reset(seed=int(seed))
    builder = ACTObservationBuilder(replay_cfg)
    period = max(1, int(round(
        float(replay_cfg.sim.control_hz) / float(replay_cfg.act.action_hz)
    )))
    observations: list[dict] = []
    actions: list[np.ndarray] = []
    previous_policy_reference = np.array(
        [*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32
    )

    try:
        for index, row in enumerate(trace):
            command = np.asarray(row.get("command"), dtype=np.float32).reshape(4)
            env.step_control(Command(
                float(command[0]), float(command[1]),
                float(command[2]), float(command[3])
            ))
            if index % period != period - 1 and index != len(trace) - 1:
                continue

            contact_latched = bool(row.get("contact", False))
            obs = builder.observe(env, contact_latched=contact_latched)
            image_size = tuple(int(value) for value in replay_cfg.act.image_size)
            overhead = np.clip(
                np.transpose(obs["observation.images.overhead"], (1, 2, 0)) * 255.0,
                0, 255,
            ).astype(np.uint8)
            wrist = np.clip(
                np.transpose(obs["observation.images.wrist"], (1, 2, 0)) * 255.0,
                0, 255,
            ).astype(np.uint8)
            applied_reference = command.copy()
            policy_reference = np.asarray(
                row.get("policy_reference", applied_reference), dtype=np.float32
            ).reshape(4)
            delta = policy_action_delta(
                policy_reference, previous_policy_reference,
                contact_latched=contact_latched,
            )
            previous_policy_reference = policy_reference.copy()
            observations.append({
                "overhead": overhead,
                "wrist": wrist,
                "inspection": np.asarray(env.render_rgb(
                    "inspection_cam", size=image_size), dtype=np.uint8),
                "state": obs["observation.state"],
                "environment_state": obs["observation.environment_state"],
                "t": float(env.time),
                "phase": str(row.get("phase", "unknown")),
                "policy_mask": str(row.get("phase", "")) in {
                    "approach", "descent", "contact_build", "sweep"
                },
                "joint_position": np.asarray(env.ee.joint_state(), dtype=np.float32),
                "tcp_pose": np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32),
                "wrench": np.asarray(env.wrench(), dtype=np.float32),
                "normal_force": float(env.normal_force()),
                "contact": contact_latched,
                "contact_latched": contact_latched,
                "fully_collected": int(env.collected_mask().sum()),
                "object_pose": np.concatenate((
                    np.asarray(env.component_positions(), dtype=np.float32),
                    np.asarray(env.component_quats(), dtype=np.float32),
                ), axis=1),
                "object_collected": np.asarray(env.collected_mask(), dtype=bool),
                "reference": applied_reference,
                "applied_reference": applied_reference,
                "policy_reference": policy_reference,
                "policy_z": float(policy_reference[2]),
                "applied_z": float(applied_reference[2]),
                "z_owner": str(row.get(
                    "z_owner", "admittance" if contact_latched else "policy"
                )),
            })
            actions.append(delta)
    finally:
        env.close()

    if not observations:
        return None
    episode_id = _episode_id(root, target_count, int(seed))
    metadata = {
        "schema_version": 4,
        "episode_kind": "inference",
        "split": "inference",
        "seed": int(seed),
        "count": 6,
        "total_count": int(getattr(result, "total", 6)),
        "target_count": target_count,
        "collected": int(getattr(result, "collected", 0)),
        "fps": float(replay_cfg.act.action_hz),
        "planner": "act_inference",
        "planner_status": "completed",
        "planner_strategy": "contact_act_admittance",
        "target_mode": "exact",
        "spawn_mode": str(replay_cfg.components.get("spawn_mode", "cluster")),
        "generation_outcome": "inference",
        "inference_replay": True,
        "model": str(model or ""),
        "scheduler_queries": int(getattr(result, "scheduler_queries", 0)),
        "scheduler_timeouts": int(getattr(result, "scheduler_timeouts", 0)),
        "scheduler_late_results": int(getattr(result, "scheduler_late_results", 0)),
        "peak_force": float(getattr(result, "peak_force", 0.0)),
        "termination_reason": str(getattr(result, "termination_reason", "")),
        "first_contact_position": (
            np.asarray(getattr(result, "first_contact_position"), dtype=float).tolist()
            if getattr(result, "first_contact_position", None) is not None else None
        ),
        "failure_reason": str(getattr(result, "failure_reason", "")),
    }
    ActDatasetWriter(str(root)).add_episode(
        episode_id, observations, np.asarray(actions, dtype=np.float32),
        bool(getattr(result, "success", False)), metadata,
    )
    return episode_id


def save_objectact_inference_replay(
    cfg, seed: int, result, root: str | Path, model: str | None = None
) -> str | None:
    """Replay an ObjectACT trace into an isolated schema-v5 workbench record."""
    from .object_dataset import ObjectActDatasetWriter
    from .object_interface import ObjectACTObservationBuilder, RGBObjectPerceptionFrontend
    from ..perception.detector_training import verified_detector_checkpoint

    trace = list(getattr(result, "trace", ()))
    if not trace:
        return None
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target_count = int(getattr(result, "target_count", 0) or cfg.get_path("task.target_count", 0))
    replay_cfg = cfg.copy()
    replay_cfg.set_path("task.target_count", target_count)
    env = SweepEnv(replay_cfg, seed=int(seed))
    env.reset(seed=int(seed))
    detector_checkpoint = replay_cfg.act.get("objectact_detector_checkpoint", None)
    if not detector_checkpoint:
        detector_checkpoint = replay_cfg.act.get(
            "objectact_detector_model_dir", "runs/objectact_detector"
        )
    detector_checkpoint = verified_detector_checkpoint(detector_checkpoint)
    frontend = RGBObjectPerceptionFrontend(
        replay_cfg, device="cpu", detector_checkpoint=detector_checkpoint
    )
    builder = ObjectACTObservationBuilder(replay_cfg, frontend=frontend)
    builder.reset()
    period = max(1, int(round(float(replay_cfg.sim.control_hz) / float(replay_cfg.act.action_hz))))
    observations: list[dict] = []
    previous_policy_reference = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    try:
        for index, row in enumerate(trace):
            command = np.asarray(row.get("command"), dtype=np.float32).reshape(4)
            env.step_control(Command(float(command[0]), float(command[1]), float(command[2]), float(command[3])))
            if index % period != period - 1 and index != len(trace) - 1:
                continue
            contact = bool(row.get("contact", False))
            v5 = builder.observe(env, contact_latched=contact)
            policy_reference = np.asarray(row.get("policy_reference", command), dtype=np.float32).reshape(4)
            delta = policy_action_delta(policy_reference, previous_policy_reference, contact_latched=contact)
            previous_policy_reference = policy_reference.copy()
            observations.append({
                "overhead": np.clip(np.transpose(v5["observation.images.overhead"], (1, 2, 0)) * 255, 0, 255).astype(np.uint8),
                "wrist": np.clip(np.transpose(v5["observation.images.wrist"], (1, 2, 0)) * 255, 0, 255).astype(np.uint8),
                "inspection": np.asarray(env.render_rgb("inspection_cam", size=tuple(int(v) for v in replay_cfg.act.image_size)), dtype=np.uint8),
                "robot_state": np.asarray(v5["observation.robot_state"], dtype=np.float32),
                "task_state": np.asarray(v5["observation.task_state"], dtype=np.float32),
                "object_tokens": np.asarray(v5["observation.object_tokens"], dtype=np.float32),
                "object_valid": np.asarray(v5["observation.object_valid"], dtype=bool),
                "instance_bev": np.asarray(v5["observation.instance_bev"], dtype=bool),
                "action_delta": delta,
                "policy_mask": True,
                "phase": str(row.get("phase", "unknown")),
                "t": float(env.time),
                "joint_position": np.asarray(env.ee.joint_state(), dtype=np.float32),
                "tcp_pose": np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32),
                "wrench": np.asarray(env.wrench(), dtype=np.float32),
                "normal_force": float(env.normal_force()),
                "contact": contact,
                "contact_latched": contact,
                "fully_collected": int(env.collected_mask().sum()),
                "reference": command.copy(),
                "policy_reference": policy_reference.copy(),
                "policy_z": float(policy_reference[2]),
                "applied_z": float(command[2]),
                "z_owner": str(row.get("z_owner", "admittance" if contact else "policy")),
            })
    finally:
        env.close()
    if not observations:
        return None
    episode_id = _episode_id(root, target_count, int(seed))
    metadata = {
        "episode_kind": "inference",
        "split": "inference",
        "seed": int(seed),
        "total_count": int(getattr(result, "total", 6)),
        "target_count": target_count,
        "collected": int(getattr(result, "collected", 0)),
        "fps": float(replay_cfg.act.action_hz),
        "perception_source": "rgb_detector",
        "detector_checkpoint": str(detector_checkpoint or ""),
        "inference_replay": True,
        "model": str(model or ""),
        "planner": "objectact_inference",
        "planner_strategy": "rgb_object_tokens_bev_act_admittance",
        "termination_reason": str(getattr(result, "termination_reason", "")),
        "failure_reason": str(getattr(result, "failure_reason", "")),
        "peak_force": float(getattr(result, "peak_force", 0.0)),
    }
    ObjectActDatasetWriter(str(root)).add_episode(
        episode_id,
        observations,
        np.zeros((len(observations), 6), dtype=bool),
        bool(getattr(result, "success", False)),
        metadata,
    )
    return episode_id
