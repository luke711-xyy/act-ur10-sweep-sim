"""Render a completed ACT rollout into the workbench episode format."""

from __future__ import annotations

import time
import hashlib
import gzip
import json
import os
import uuid
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .dataset import ActDatasetWriter
from .interface import ACTObservationBuilder, ACT_SCHEMA_VERSION
from .rollout import policy_action_delta


def _angle_delta(value: float, previous: float) -> float:
    return float(np.arctan2(np.sin(value - previous), np.cos(value - previous)))


def _episode_id(root: Path, target_count: int, model: str | None) -> str:
    """Return a collision-free id that is visibly an inference replay."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(str(model)).name).strip("_") if model else ""
    label = label[:80] or "no_checkpoint"
    base = f"inference_{stamp}_n{int(target_count)}_{label}"
    candidate = base
    ordinal = 2
    while (root / candidate).exists():
        candidate = f"{base}_{ordinal:02d}"
        ordinal += 1
    return candidate


def persist_replay_input(cfg, seed: int, result, root: str | Path,
                         model: str | None = None) -> Path:
    """Persist the measured outcome before any fallible rendering starts."""
    root = Path(root)
    directory = root / "replay_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    keys = ("success", "failure_reason", "collected", "total", "target_count",
            "scheduler_queries", "scheduler_timeouts", "scheduler_late_results",
            "peak_force", "termination_reason", "first_contact_position",
            "peak_collected", "final_collected", "first_entry_time",
            "collection_window_seconds", "initial_object_poses",
            "policy_type", "inference_latencies", "wall_seconds")
    values = {key: getattr(result, key) for key in keys if hasattr(result, key)}
    values["trace"] = list(result.trace)
    document = {"schema_version": 1, "config": cfg.to_dict(),
                "seed": int(seed), "model": str(model or ""), "result": values}
    path = directory / f"n{int(result.target_count)}_s{int(seed)}_{uuid.uuid4().hex}.json.gz"
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
                compressed.write(json.dumps(document, allow_nan=False,
                    default=lambda value: np.asarray(value).tolist()).encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_replay_input(path: str | Path, cfg, *, seed: int, model: str | None = None):
    """Never relabel an earlier measurement as another seed/config/checkpoint."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if (document.get("schema_version") != 1 or document.get("seed") != int(seed)
            or document.get("model") != str(model or "")
            or document.get("config") != cfg.to_dict()
            or document.get("result", {}).get("target_count") != int(cfg.task.target_count)):
        raise ValueError("saved replay input identity does not match this evaluation")
    return SimpleNamespace(**document["result"])


def recover_replay_input(path: str | Path, cfg, *, seed: int,
                         model: str | None = None, image_format="webp-lossless"):
    path = Path(path)
    result = load_replay_input(path, cfg, seed=seed, model=model)
    return save_inference_replay(cfg, seed, result, path.parent.parent,
                                 model=model, image_format=image_format)


def save_inference_replay(cfg, seed: int, result, root: str | Path,
                          model: str | None = None,
                          image_format: str = "png", resource_check=None) -> str | None:
    """Replay the recorded commands and save five policy cameras plus telemetry.

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
    recorded_states = all("replay_state" in row for row in trace)
    if not recorded_states and any("replay_state" in row for row in trace):
        raise ValueError("recorded replay state is missing from part of the trace")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target_count = int(getattr(result, "target_count", 0)
                       or cfg.get_path("task.target_count", 0))
    replay_cfg = cfg.copy()
    replay_cfg.set_path("task.target_count", target_count)
    if replay_cfg.get_path("scene.backend",None)=="rudra_native":
        from .rollout import _episode_environment
        env = _episode_environment(replay_cfg,int(seed))
    else:
        env = SweepEnv(replay_cfg, seed=int(seed))
    env.reset(seed=int(seed))
    initial_poses = np.concatenate((env.component_positions(), env.component_quats()), axis=1).astype("<f4")
    if getattr(result, "initial_object_poses", None) is not None:
        measured_initial = np.asarray(result.initial_object_poses, dtype="<f4")
        if not np.array_equal(measured_initial, initial_poses):
            env.close()
            raise ValueError("saved replay initial layout identity does not match reset")
    layout_fingerprint = hashlib.sha256(np.ascontiguousarray(initial_poses).tobytes()).hexdigest()
    builder = ACTObservationBuilder(replay_cfg)
    period = max(1, int(round(
        float(replay_cfg.sim.control_hz) / float(replay_cfg.act.action_hz)
    )))
    previous_policy_reference = np.array(
        [*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32
    )

    def frames():
        nonlocal previous_policy_reference
        for index, row in enumerate(trace):
            if resource_check is not None:
                resource_check()
            command = np.asarray(row.get("command"), dtype=np.float32).reshape(4)
            if recorded_states:
                import mujoco
                saved = row["replay_state"]
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                values = np.asarray(saved["values"], dtype=np.float64)
                if (saved["spec"] != int(spec)
                        or values.shape != (mujoco.mj_stateSize(env.model, spec),)
                        or not np.all(np.isfinite(values))):
                    raise ValueError("invalid recorded replay physics state")
                # Render the original state, without re-solving IK or advancing
                # physics. Rejected commands can have side effects not present
                # in the old accepted-command-only trace.
                mujoco.mj_setState(env.model, env.data, values, spec)
                mujoco.mj_forward(env.model, env.data)
                mujoco.mj_rnePostConstraint(env.model, env.data)
            else:
                env.step_control(Command(
                    float(command[0]), float(command[1]),
                    float(command[2]), float(command[3])
                ))
            if index % period != period - 1 and index != len(trace) - 1:
                continue

            contact_latched = bool(row.get("contact", False))
            obs = builder.observe(env, contact_latched=contact_latched)
            if recorded_states and "replay_observation" in row:
                # Preserve numeric state/history/supervisor measurements too;
                # mj_forward refreshes derived geometry for rendering only.
                obs.update(row["replay_observation"])
            images = {
                view: np.clip(
                    np.transpose(obs[f"observation.images.{view}"], (1, 2, 0)) * 255.0,
                    0, 255,
                ).astype(np.uint8)
                for view in builder.camera_views
            }
            applied_reference = command.copy()
            policy_reference = np.asarray(
                row.get("policy_reference", applied_reference), dtype=np.float32
            ).reshape(4)
            delta = policy_action_delta(
                policy_reference, previous_policy_reference,
                contact_latched=contact_latched,
            )
            previous_policy_reference = policy_reference.copy()
            frame_observation = {
                "images": images,
                "state": obs["observation.state"],
                "environment_state": obs["observation.environment_state"],
                "supervisor_state": obs["supervisor_state"],
                "t": float(env.time),
                "phase": str(row.get("phase", "unknown")),
                "policy_mask": str(row.get("phase", "")) in {
                    "approach", "descent", "contact_build", "sweep"
                },
                "joint_position": np.asarray(env.ee.joint_state(), dtype=np.float32),
                "tcp_pose": np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32),
                "wrench": np.asarray(env.wrench(), dtype=np.float32),
                "normal_force": float(env.normal_force()),
                "raw_normal_force": float(env.raw_normal_force()),
                "compensated_normal_force_raw": float(
                    env.compensated_normal_force_raw()),
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
            }
            frame_observation.update({
                name: images[name] for name in ("overhead", "wrist")
                if name in images
            })
            if recorded_states:
                # These signals are measurements, not freshly simulated
                # estimates. The rendered geometry comes from native state.
                for name in ("normal_force", "raw_normal_force",
                             "compensated_normal_force_raw", "wrench", "tcp_pose", "object_pose"):
                    if name in row:
                        frame_observation[name] = row[name]
                frame_observation["fully_collected"] = int(row["collected"])
                frame_observation["object_collected"] = row["object_collected"]
            yield frame_observation, delta
            del obs, images, frame_observation

    episode_id = _episode_id(root, target_count, model)
    policy_type = str(getattr(result,"policy_type","act"))
    metadata = {
        "schema_version": ACT_SCHEMA_VERSION,
        "episode_kind": "inference",
        "split": "inference",
        "seed": int(seed),
        "initial_object_poses": initial_poses.tolist(),
        "layout_fingerprint": layout_fingerprint,
        "count": 6,
        "total_count": int(getattr(result, "total", 6)),
        "target_count": target_count,
        "collected": int(getattr(result, "collected", 0)),
        "fps": float(replay_cfg.act.action_hz),
        "planner": f"{policy_type}_inference",
        "planner_status": "completed",
        "planner_strategy": f"contact_{policy_type}_admittance",
        "target_mode": "exact",
        "spawn_mode": str(replay_cfg.components.get("spawn_mode", "cluster")),
        "generation_outcome": "inference",
        "inference_replay": True,
        "replay_render_mode": ("recorded_physics_state" if recorded_states
                               else "legacy_command_resimulation"),
        "model": str(model or ""),
        "desired_force": float(replay_cfg.controller.desired_force),
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
    if policy_type == "diffusion":
        metadata.update(policy_type="diffusion",
                        inference_latencies=list(getattr(result,"inference_latencies",[])),
                        wall_seconds=getattr(result,"wall_seconds",None),
                        real_time_200ms_claim=False)
    if getattr(result, "peak_collected", None) is not None:
        # These counts belong to the causal inference, not the fresh scene
        # used to render images. Keep both achievement and terminal retention.
        metadata.update({
            "counting_mode": "peak_simultaneous_full_entry",
            "peak_collected": int(result.peak_collected),
            "final_collected": int(result.final_collected),
            "first_entry_time": result.first_entry_time,
            "collection_window_seconds": float(result.collection_window_seconds),
        })
    try:
        ActDatasetWriter(str(root), image_format=image_format).add_episode_stream(
            episode_id, frames(), bool(getattr(result, "success", False)),
            metadata, causal_trace=trace,
        )
    finally:
        env.close()
    return episode_id
