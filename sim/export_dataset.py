"""Export demonstrations for later ACT training.

**No training happens here.**  The exporter records what an imitation-learning
policy would need and nothing more.

Layout on disk::

    <out_dir>/
      dataset.json                  # global metadata, action spec, config
      index.jsonl                   # one line per exported episode
      episode_000000/
        meta.json                   # seed, planner, geometry, masses, frictions, success
        episode.npz                 # dense low-level trace + high-level actions
        obs_000_rgb.png             # fixed-camera frame at each re-planning step
        obs_000_mask.png            # component occupancy mask (image space)
      ...

Per episode the following are stored (all required by the task specification):

* RGB image and occupancy mask at every observation (one per stroke)
* TCP position and velocity, Cartesian/joint state, gripper state
* measured force/torque, desired force, executed action (the command sent to
  the robot at each control step)
* timestamps, component count and geometry, material/friction parameters,
  random seed, planner name, success label

High-level action definition::

    [action_x_start, action_y_start, action_x_end, action_y_end, action_yaw]

Z force is **not** part of the action: it stays inside the hybrid controller.

    python -m sim.export_dataset --planner visual_greedy --episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

from .config import load_config, save_config
from .logging_utils import trace_to_arrays
from .perception.base import GridSpec
from .planners.base import ACTION_LABELS

#: number of sparse waypoints per stroke in the alternative action parameterisation
N_WAYPOINTS = 16
WAYPOINT_LABELS = ("dx", "dy", "dpsi")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export sweeping demonstrations.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--planner", default="visual_greedy",
                        choices=["fixed", "global_sweep", "visual_greedy"])
    parser.add_argument("--perception", default="vision",
                        choices=["ground_truth", "vision"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--spawn-mode", default=None, choices=["uniform", "cluster"],
                        help="demonstration data normally uses the clustered main task")
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--all-episodes", action="store_true",
                        help="export failed episodes too (default: successful only)")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    return parser


def _history_horizon(cfg_dict) -> int:
    return int(((cfg_dict or {}).get("dataset") or {}).get("wrench_history_steps", 10))


def goal_region_grid_from_config(cfg_dict) -> np.ndarray:
    from .config import Config

    return goal_region_grid(Config(cfg_dict))


def _save_png(array: np.ndarray, path: str) -> None:
    from PIL import Image

    arr = np.asarray(array)
    if arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    Image.fromarray(arr).save(path)


def _resample_xy_yaw(points: np.ndarray, yaws: np.ndarray, n: int) -> np.ndarray:
    """Resample a planar path to ``n`` points equally spaced in arc length."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    yaws = np.asarray(yaws, dtype=float).ravel()
    if points.shape[0] == 0:
        return np.zeros((n, 3))
    if points.shape[0] == 1:
        return np.repeat(np.array([[points[0, 0], points[0, 1], yaws[0]]]), n, axis=0)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] < 1e-9:
        return np.repeat(np.array([[points[0, 0], points[0, 1], yaws[0]]]), n, axis=0)
    targets = np.linspace(0.0, arc[-1], n)
    return np.stack([
        np.interp(targets, arc, points[:, 0]),
        np.interp(targets, arc, points[:, 1]),
        np.interp(targets, arc, yaws),
    ], axis=1)


def stroke_waypoints(trace, stroke_index: int, capture_pose, fallback,
                     n: int = N_WAYPOINTS) -> np.ndarray:
    """``(n, 3)`` sparse waypoints ``[dx, dy, dpsi]`` relative to the capture pose.

    The path is the **executed** in-contact TCP trajectory (FORCE_RAMP through
    FORCE_RELEASE), resampled by arc length.  This is the alternative action
    parameterisation: identical information to the 5-D stroke for a straight
    segment, but able to represent a curved stroke, and it matches the
    sparse-waypoint + interpolation structure used by comparable systems.

    ``fallback`` is the commanded ``(start, end, yaw)`` used when the stroke
    never established contact.
    """
    phases = ("FORCE_RAMP", "SWEEP", "FORCE_RELEASE")
    records = [r for r in trace if r.stroke == stroke_index and r.phase in phases]
    capture = np.asarray(capture_pose, dtype=float).ravel()
    if len(records) >= 2:
        xy = np.array([[r.tcp_x, r.tcp_y] for r in records], dtype=float)
        yaw = np.array([r.cmd_yaw for r in records], dtype=float)
    else:
        start, end, yaw_value = fallback
        xy = np.stack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)])
        yaw = np.array([yaw_value, yaw_value], dtype=float)
    path = _resample_xy_yaw(xy, yaw, n)
    path[:, 0] -= capture[0]
    path[:, 1] -= capture[1]
    path[:, 2] -= capture[2]
    return path.astype(np.float32)


def goal_region_grid(cfg) -> np.ndarray:
    """Binary goal-region channel on the same table-plane grid as the occupancy map."""
    grid = GridSpec.from_config(cfg, res=float(cfg.perception.get("grid_res", 0.01)))
    iy, ix = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
    centres = grid.to_xy(iy, ix)
    tgt = cfg.target
    return ((centres[..., 0] >= float(tgt.x_min)) & (centres[..., 0] <= float(tgt.x_max))
            & (centres[..., 1] >= float(tgt.y_min))
            & (centres[..., 1] <= float(tgt.y_max)))


def _wrench_history(arrays: dict, t_obs: float, horizon: int) -> np.ndarray:
    """The last ``horizon`` control-step wrenches at or before ``t_obs``."""
    if not arrays or "t" not in arrays:
        return np.zeros((horizon, 6), dtype=np.float32)
    times = arrays["t"]
    wrench = arrays["wrench"]
    end = int(np.searchsorted(times, float(t_obs), side="right"))
    start = max(0, end - horizon)
    window = wrench[start:end]
    if window.shape[0] < horizon:
        pad = np.zeros((horizon - window.shape[0], 6), dtype=window.dtype)
        window = np.concatenate([pad, window], axis=0)
    return window.astype(np.float32)


def _episode_payload(result, keep_images: bool) -> dict:
    arrays = trace_to_arrays(result.trace)
    strokes = result.strokes
    actions = np.array([s.action for s in strokes], dtype=np.float32) if strokes \
        else np.zeros((0, 5), dtype=np.float32)
    # Per control step: which high-level action was being executed.
    stroke_index = arrays.get("stroke_index", np.zeros(0, dtype=np.int16))
    capture_poses = np.array([s.capture_pose for s in strokes], dtype=np.float32) \
        if strokes else np.zeros((0, 3), dtype=np.float32)

    waypoints = np.stack([
        stroke_waypoints(result.trace, s.index, s.capture_pose,
                         fallback=(s.action[0:2], s.action[2:4], float(s.action[4])))
        for s in strokes
    ]) if strokes else np.zeros((0, N_WAYPOINTS, 3), dtype=np.float32)

    # Per control step, the TCP pose RELATIVE to the capture pose of the stroke it
    # belongs to.  Absolute z is deliberately excluded from this policy-facing
    # array (it encodes the table height, the nuisance variable the force loop
    # exists to absorb); it stays available in `tcp_position` for debugging.
    tcp_abs = arrays.get("tcp_position", np.zeros((0, 3), dtype=np.float32))
    yaw_abs = arrays.get("tcp_yaw", np.zeros(tcp_abs.shape[0], dtype=np.float32))
    tcp_rel = np.full((tcp_abs.shape[0], 3), np.nan, dtype=np.float32)
    for s in strokes:
        sel = stroke_index == s.index
        if np.any(sel):
            tcp_rel[sel, 0] = tcp_abs[sel, 0] - s.capture_pose[0]
            tcp_rel[sel, 1] = tcp_abs[sel, 1] - s.capture_pose[1]
            tcp_rel[sel, 2] = yaw_abs[sel] - s.capture_pose[2]

    payload = dict(arrays)
    if "wrench" in arrays and "wrench_parts" in arrays:
        # table share = total - component share (exact, both come from the same sum)
        payload["wrench_table"] = (arrays["wrench"] - arrays["wrench_parts"]).astype(np.float32)
    payload.update({
        "actions": actions,
        "action_labels": np.array(ACTION_LABELS, dtype="U18"),
        "actions_waypoints": np.asarray(waypoints, dtype=np.float32),
        "actions_waypoint_labels": np.array(WAYPOINT_LABELS, dtype="U8"),
        "capture_poses": capture_poses,
        "tcp_pose_relative": tcp_rel,
        "goal_region": goal_region_grid_from_config(result.config),
        "stroke_t_start": np.array([s.t_start for s in strokes], dtype=np.float32),
        "stroke_t_end": np.array([s.t_end for s in strokes], dtype=np.float32),
        "stroke_collected_before": np.array([s.collected_before for s in strokes], dtype=np.int16),
        "stroke_collected_after": np.array([s.collected_after for s in strokes], dtype=np.int16),
        "stroke_observation_index": np.array([s.observation_index for s in strokes],
                                             dtype=np.int16),
        "stroke_ride_over": np.array([s.n_ride_over for s in strokes], dtype=np.int16),
        "stroke_jam_events": np.array([s.n_jam_events for s in strokes], dtype=np.int16),
        "stroke_touchdown_overshoot": np.array([s.touchdown_overshoot for s in strokes],
                                               dtype=np.float32),
        "active_action_index": stroke_index,
        "initial_positions": np.asarray(result.initial_positions, dtype=np.float32),
        "final_positions": np.asarray(result.final_positions, dtype=np.float32),
    })
    horizon = int(_history_horizon(result.config))
    for i, obs in enumerate(result.observations):
        payload[f"obs_{i:03d}_points"] = np.asarray(obs["points"], dtype=np.float32)
        payload[f"obs_{i:03d}_counts"] = np.asarray(obs["counts"], dtype=np.float32)
        payload[f"obs_{i:03d}_occupancy"] = np.asarray(obs["occupancy"], dtype=bool)
        payload[f"obs_{i:03d}_t"] = np.float32(obs["t"])
        payload[f"obs_{i:03d}_capture_pose"] = np.asarray(
            obs.get("capture_pose", np.zeros(3)), dtype=np.float32)
        payload[f"obs_{i:03d}_wrench_history"] = _wrench_history(arrays, obs["t"], horizon)
        if keep_images and obs.get("mask") is not None:
            payload[f"obs_{i:03d}_mask"] = np.asarray(obs["mask"], dtype=bool)
    return payload


class WrenchMissingError(RuntimeError):
    """Raised when an episode would be exported with an empty wrench channel."""


def validate_wrench(result) -> None:
    """Refuse to export a demonstration whose wrench channel is dead.

    A wrench of exactly zero *while sweeping in contact* cannot happen
    physically, so it means the sensing path was not wired up.  Catching it here
    costs one check; catching it after a 200-episode collection run costs the
    run.
    """
    sweeping = [r for r in result.trace if r.phase == "SWEEP" and r.in_contact]
    if not sweeping:
        return                      # nothing was swept; nothing to validate
    if max(abs(r.fx) + abs(r.fy) + abs(r.fz) for r in sweeping) < 1e-9:
        raise WrenchMissingError(
            f"episode (seed {result.metrics.seed}, planner {result.metrics.planner}) has "
            f"{len(sweeping)} in-contact SWEEP samples but an identically zero wrench. "
            "The environment is not reporting the tool wrench -- fix that before "
            "exporting demonstrations."
        )


def export_episode(result, episode_dir: str, keep_images: bool) -> dict:
    validate_wrench(result)
    os.makedirs(episode_dir, exist_ok=True)
    metrics = result.metrics
    cfg = result.config

    meta = {
        "seed": metrics.seed,
        "planner": metrics.planner,
        "perception": metrics.perception,
        "success": bool(metrics.success),
        "component_count": metrics.n_components,
        "component_geometry": metrics.geometry,
        "collection_rate": metrics.collection_rate,
        "n_strokes": metrics.n_strokes,
        "completion_time": metrics.completion_time,
        "control_hz": cfg["sim"]["control_hz"],
        "physics_dt": cfg["sim"]["physics_dt"],
        "desired_force": cfg["controller"]["desired_force"],
        "gripper_state": 1.0,
        "components": [
            {
                "index": item["index"],
                "geometry": item["geometry"],
                "mass": item["mass"],
                "friction": item["friction"],
                "x0": item["x"], "y0": item["y"], "yaw0": item["yaw"],
            }
            for item in result.layout
        ],
        "table_friction": cfg["table"]["friction"],
        "camera": cfg["perception"]["camera"],
        "action_labels": list(ACTION_LABELS),
        "action_space": "planar sweep stroke; Z force handled by the hybrid controller",
        "action_waypoint_labels": list(WAYPOINT_LABELS),
        "n_waypoints": N_WAYPOINTS,
        "wrench_history_steps": _history_horizon(cfg),
        "policy_facing_arrays": [
            "tcp_pose_relative", "actions", "actions_waypoints",
            "obs_*_mask", "obs_*_occupancy", "goal_region", "obs_*_wrench_history",
            "wrench", "tangential_force",
        ],
        "debug_only_arrays": [
            "tcp_position (absolute, includes table height)", "z_nominal", "delta_z",
            "command", "obs_*_points",
        ],
        "simulator_ground_truth_arrays": {
            "arrays": ["wrench_parts", "wrench_table", "tangential_force_parts",
                       "n_part_contacts", "phase"],
            "note": (
                "Decomposition of the measured wrench into the component-contact and "
                "table-contact shares, plus the exact contact-phase label. None of this "
                "exists on hardware. Use it to supervise and validate a classifier that "
                "runs on the total wrench -- never as a policy input."
            ),
        },
    }
    with open(os.path.join(episode_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    np.savez_compressed(os.path.join(episode_dir, "episode.npz"),
                        **_episode_payload(result, keep_images))

    if keep_images:
        for i, obs in enumerate(result.observations):
            if obs.get("rgb") is not None:
                _save_png(obs["rgb"], os.path.join(episode_dir, f"obs_{i:03d}_rgb.png"))
            if obs.get("mask") is not None:
                _save_png(obs["mask"], os.path.join(episode_dir, f"obs_{i:03d}_mask.png"))
    return meta


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg.set_path("planner.name", args.planner)
    cfg.set_path("perception.backend", args.perception)
    if args.geometry:
        cfg.set_path("components.geometry", args.geometry)
    if args.spawn_mode:
        cfg.set_path("components.spawn_mode", args.spawn_mode)

    from .environments.episode import run_episode  # lazy MuJoCo import

    out_dir = args.out or str(cfg.dataset.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    keep_images = (not args.no_images) and bool(cfg.dataset.save_images)

    index_path = os.path.join(out_dir, "index.jsonl")
    exported = 0
    attempted = 0
    t0 = time.time()
    counts = list(args.counts)

    with open(index_path, "w", encoding="utf-8") as index:
        while exported < int(args.episodes):
            count = counts[attempted % len(counts)]
            seed = int(args.seed_base) + attempted
            cfg.set_path("components.count", int(count))
            result = run_episode(cfg, seed=seed, planner_name=args.planner,
                                 perception_name=args.perception,
                                 keep_images=keep_images, verbose=False)
            attempted += 1
            keep = bool(result.metrics.success) or bool(args.all_episodes) \
                or not bool(cfg.dataset.only_successful)
            if keep:
                episode_dir = os.path.join(out_dir, f"episode_{exported:06d}")
                meta = export_episode(result, episode_dir, keep_images)
                meta["path"] = os.path.relpath(episode_dir, out_dir)
                index.write(json.dumps(meta) + "\n")
                index.flush()
                exported += 1
            print(f"  [{exported:4d}/{args.episodes}] attempted={attempted} n={count} "
                  f"seed={seed} rate={result.metrics.collection_rate:.2f} "
                  f"{'KEPT' if keep else 'skipped'} "
                  f"| {(time.time() - t0)/60:.1f} min", flush=True)

    with open(os.path.join(out_dir, "dataset.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "episodes": exported,
            "attempted": attempted,
            "planner": args.planner,
            "perception": args.perception,
            "counts": counts,
            "only_successful": bool(cfg.dataset.only_successful) and not args.all_episodes,
            "action_labels": list(ACTION_LABELS),
            "action_dim": len(ACTION_LABELS),
            "note": "Z normal force is regulated by the hybrid controller and is NOT an action.",
            "config": cfg.to_dict(),
        }, handle, indent=2)
    print(f"  exported {exported} episodes to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
