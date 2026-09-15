"""Single-episode runner: perception -> planner -> hybrid controller -> physics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..controllers.hybrid import Command, ControlRecord, HybridForcePositionController
from ..controllers.state_machine import Phase
from ..metrics import (EpisodeMetrics, compute_episode_metrics, count_jam_events,
                       touchdown_overshoot)
from ..perception.base import Perception, SceneObservation, build_perception
from ..planners.base import Planner, SweepStroke, build_planner
from ..planners.geometry_utils import point_segment_frames, pusher_width
from ..planners.rrt import plan_transfer
from ..planners.trajectory import make_linear_trajectory
from .sweep_env import SweepEnv


@dataclass
class StrokeRecord:
    index: int
    action: np.ndarray
    meta: Dict[str, Any]
    observation_index: int
    collected_before: int
    collected_after: int
    contact_established: bool
    peak_force: float
    aborted: bool
    abort_reason: str
    t_start: float
    t_end: float
    capture_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_ride_over: int = 0
    n_jam_events: int = 0
    touchdown_overshoot: float = 0.0


@dataclass
class EpisodeResult:
    metrics: EpisodeMetrics
    trace: List[ControlRecord] = field(default_factory=list)
    strokes: List[StrokeRecord] = field(default_factory=list)
    observations: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    layout: List[dict] = field(default_factory=list)
    config: Optional[dict] = None
    component_tracks: Optional[np.ndarray] = None   # (T, N, 2) sampled positions
    track_times: Optional[np.ndarray] = None        # (T,) simulated seconds
    initial_positions: Optional[np.ndarray] = None
    final_positions: Optional[np.ndarray] = None


def require_wrench(env) -> None:
    """Fail fast when an environment cannot report the tool wrench.

    The 6-axis wrench is this project's primary sensing channel: the contact-phase
    and material analyses are built on it, and a run that silently logged zeros
    would only be found out after the demonstrations were collected.  So a
    missing wrench stops the episode instead of degrading it.
    """
    if not hasattr(env, "wrench"):
        raise RuntimeError(
            f"{type(env).__name__} does not implement wrench(). The 6-axis wrench is "
            "required for every episode -- see EndEffectorInterface.wrench()."
        )


def _full_state(env):
    """Wrench, component-contact share, TCP velocity and yaw."""
    wrench = np.asarray(env.wrench(), dtype=float).ravel()
    velocity = np.asarray(env.tcp_velocity(), dtype=float).ravel() \
        if hasattr(env, "tcp_velocity") else np.zeros(3)
    ee = getattr(env, "ee", None)
    yaw = float(ee.tcp_yaw()) if ee is not None and hasattr(ee, "tcp_yaw") else 0.0
    if hasattr(env, "contact_breakdown"):
        parts, n_contacts = env.contact_breakdown()
        parts = np.asarray(parts, dtype=float).ravel()
    else:
        parts, n_contacts = np.zeros(6), 0
    return wrench, velocity, yaw, parts, int(n_contacts)


def _ride_over_count(stroke: SweepStroke, before: np.ndarray, after: np.ndarray,
                     eligible: np.ndarray, cfg) -> int:
    """Components the pusher swept over without moving.

    This is the thin-washer failure mode: with the normal force too low (or the
    part too flat) the tip rides over the component instead of pushing it.
    """
    if before.size == 0:
        return 0
    half = pusher_width(cfg) / 2.0 + float(cfg.get_path("metrics.ride_over_pad", 0.002))
    along, lateral, _ = point_segment_frames(before[:, :2], stroke.start, stroke.end)
    swept = (along >= 0.0) & (along <= 1.0) & (np.abs(lateral) <= half)
    moved = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
    threshold = float(cfg.get_path("metrics.ride_over_move_threshold", 0.004))
    return int(np.sum(swept & eligible & (moved < threshold)))


def _transfer_to(env: SweepEnv, target_xyz, speed: float, trace: List[ControlRecord],
                 kind: str, yaw: float = 0.0, max_time: float = 8.0,
                 recorder=None, planner_name: str = "", rng=None, obstacles=None,
                 tracker=None) -> None:
    """Lifted, contact-free Cartesian move (used to reach the observation pose)."""
    cfg = env.cfg
    start = env.tcp().copy()
    z_travel = float(cfg.workspace.z_travel)
    cruise = max(float(start[2]), z_travel)
    transfer = plan_transfer(
        cfg,
        [float(start[0]), float(start[1]), cruise],
        [float(target_xyz[0]), float(target_xyz[1]), cruise],
        rng if rng is not None else np.random.default_rng(0),
        obstacles,
    )
    waypoints = ([[float(start[0]), float(start[1]), cruise]]
                 + [list(map(float, w)) for w in transfer[1:]]
                 + [[float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])]])
    traj = make_linear_trajectory(waypoints, speed, str(cfg.planner.interpolation),
                                  float(cfg.planner.get("accel", 0.8)))
    t0 = env.time
    elapsed = 0.0
    while elapsed <= traj.duration + 0.05 and (env.time - t0) < max_time:
        p = traj.point(elapsed)
        cmd = Command(float(p[0]), float(p[1]), float(p[2]), float(yaw))
        env.step_control(cmd)
        tcp = env.tcp()
        w, v, tcp_yaw, wp, n_parts = _full_state(env)
        trace.append(ControlRecord(
            t=env.time, stroke=-1, phase=kind,
            tcp_x=float(tcp[0]), tcp_y=float(tcp[1]), tcp_z=float(tcp[2]),
            cmd_x=cmd.x, cmd_y=cmd.y, cmd_z=cmd.z, cmd_yaw=cmd.yaw,
            z_nominal=float("nan"), delta_z=0.0, force_desired=0.0,
            force_raw=env.normal_force(), force_filtered=env.normal_force(),
            in_contact=False,
            tcp_yaw=tcp_yaw, tcp_vx=float(v[0]), tcp_vy=float(v[1]), tcp_vz=float(v[2]),
            fx=float(w[0]), fy=float(w[1]), fz=float(w[2]),
            tx=float(w[3]), ty=float(w[4]), tz=float(w[5]),
            fx_p=float(wp[0]), fy_p=float(wp[1]), fz_p=float(wp[2]),
            tx_p=float(wp[3]), ty_p=float(wp[4]), tz_p=float(wp[5]),
            n_part_contacts=n_parts,
        ))
        if tracker is not None:
            tracker.append((env.time, env.component_positions()[:, :2].copy()))
        if recorder is not None:
            recorder.capture(env, {"t": env.time, "phase": kind, "stroke": -1,
                                   "force_desired": 0.0, "force_measured": env.normal_force(),
                                   "in_contact": False,
                                   "collected": int(env.collected_mask().sum()),
                                   "total": len(env.layout), "planner": planner_name})
        elapsed += env.control_dt


def _snapshot(obs: SceneObservation, keep_images: bool, capture_yaw: float = 0.0) -> dict:
    """One observation, plus the *capture pose* it was taken from.

    Policy observations must be expressed relative to the capture pose (never in
    absolute table coordinates, and never with absolute z), so the pose is
    stored alongside every frame.
    """
    tcp = np.asarray(obs.tcp, dtype=np.float32)
    return {
        "t": float(obs.t),
        "capture_pose": np.array([tcp[0], tcp[1], capture_yaw], dtype=np.float32),
        "points": np.asarray(obs.points, dtype=np.float32),
        "counts": np.asarray(obs.counts, dtype=np.float32),
        "areas": np.asarray(obs.areas, dtype=np.float32),
        "occupancy": np.asarray(obs.occupancy, dtype=bool),
        "tcp": np.asarray(obs.tcp, dtype=np.float32),
        "backend": obs.backend,
        "n_detected": obs.n_detected,
        "n_remaining_true": obs.n_remaining_true,
        "rgb": (np.asarray(obs.rgb, dtype=np.uint8)
                if (keep_images and obs.rgb is not None) else None),
        "mask": (np.asarray(obs.mask, dtype=bool)
                 if (keep_images and obs.mask is not None) else None),
    }


def run_episode(
    cfg,
    seed: int = 0,
    planner_name: Optional[str] = None,
    perception_name: Optional[str] = None,
    keep_images: bool = False,
    env: Optional[SweepEnv] = None,
    verbose: bool = False,
    recorder=None,
    track_every: int = 0,
) -> EpisodeResult:
    wall_t0 = time.time()
    planner_name = planner_name or str(cfg.planner.name)
    perception_name = perception_name or str(cfg.perception.backend)

    env = env or SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    require_wrench(env)

    rng = np.random.default_rng(seed + 10_000)
    controller = HybridForcePositionController(cfg, rng=rng)
    planner: Planner = build_planner(cfg, planner_name)
    planner.reset(rng)
    perception: Perception = build_perception(cfg, perception_name)

    controller.require_full_state = True
    controller.reset(env.tcp())
    trace = controller.trace

    max_time = float(cfg.sim.max_episode_time)
    success_rate = float(cfg.episode.success_collection_rate)
    needs_image = perception.name == "vision" or keep_images
    observe_pose = list(cfg.workspace.get("observe_pose", [0.48, -0.36, 0.26]))

    strokes: List[StrokeRecord] = []
    observations: List[dict] = []
    last_detected_points = None
    _tracker: List = []
    _track_counter = 0
    failure_reason = ""
    aborts = 0
    strokes_with_contact = 0
    total_ride_over = 0
    total_jams = 0
    overshoots: List[float] = []

    while True:
        if env.time >= max_time:
            failure_reason = failure_reason or "episode time budget exhausted"
            break
        if env.collection_rate() >= success_rate:
            break
        if planner.is_exhausted():
            failure_reason = failure_reason or "stroke budget exhausted"
            break

        if needs_image:
            _transfer_to(env, observe_pose, float(cfg.controller.travel_speed),
                         trace, kind="OBSERVE_MOVE", recorder=recorder,
                         planner_name=planner_name, rng=rng,
                         obstacles=last_detected_points,
                         tracker=_tracker if track_every else None)
        obs = perception.observe(env, rng)
        _, _, capture_yaw, _, _ = _full_state(env)
        observations.append(_snapshot(obs, keep_images, capture_yaw))
        obs_index = len(observations) - 1
        capture_pose = np.array([obs.tcp[0], obs.tcp[1], capture_yaw], dtype=float)

        stroke: Optional[SweepStroke] = planner.plan(obs)
        if stroke is None:
            failure_reason = failure_reason or (
                "planner returned no stroke (nothing detected or plan exhausted)"
            )
            break

        collected_mask_before = env.collected_mask()
        collected_before = int(collected_mask_before.sum())
        positions_before = env.component_positions().copy()
        trace_start = len(trace)
        t_start = env.time
        last_detected_points = np.asarray(obs.points, dtype=float).reshape(-1, 2) \
            if getattr(obs, "points", None) is not None else None
        controller.start_stroke(stroke, env.tcp(), env.time, obstacles=last_detected_points)
        while not controller.stroke_done:
            wrench, velocity, tcp_yaw, wrench_parts, n_parts = _full_state(env)
            cmd = controller.step(env.time, env.tcp(), env.normal_force(),
                                  wrench=wrench, tcp_velocity=velocity, tcp_yaw=tcp_yaw,
                                  wrench_parts=wrench_parts, n_part_contacts=n_parts)
            env.step_control(cmd)
            if track_every:
                _track_counter += 1
                if _track_counter % int(track_every) == 0:
                    _tracker.append((env.time, env.component_positions()[:, :2].copy()))
            if recorder is not None:
                from ..video import recorder_state

                recorder.capture(env, recorder_state(env, controller, planner_name))
            if env.time - t_start > float(cfg.episode.get("stroke_timeout", 60.0)):
                failure_reason = failure_reason or "stroke timeout"
                break
            if env.time >= max_time:
                break
        collected_after = int(env.collected_mask().sum())
        positions_after = env.component_positions().copy()
        stats = controller.stroke_stats
        if stats.get("aborted"):
            aborts += 1
        if stats.get("contact_established"):
            strokes_with_contact += 1

        stroke_trace = trace[trace_start:]
        n_ride_over = _ride_over_count(stroke, positions_before, positions_after,
                                       ~collected_mask_before, cfg)
        n_jams = count_jam_events(stroke_trace, cfg)
        overshoot = touchdown_overshoot(stroke_trace, float(cfg.controller.desired_force))
        total_ride_over += n_ride_over
        total_jams += n_jams
        overshoots.append(overshoot)

        strokes.append(StrokeRecord(
            index=len(strokes),
            action=stroke.to_action(),
            meta=dict(stroke.meta),
            observation_index=obs_index,
            collected_before=collected_before,
            collected_after=collected_after,
            contact_established=bool(stats.get("contact_established", False)),
            peak_force=float(stats.get("peak_force", 0.0)),
            aborted=bool(stats.get("aborted", False)),
            abort_reason=str(stats.get("abort_reason", "")),
            t_start=t_start,
            t_end=env.time,
            capture_pose=capture_pose,
            n_ride_over=n_ride_over,
            n_jam_events=n_jams,
            touchdown_overshoot=overshoot,
        ))
        planner.notify_stroke_done(stroke, {
            "collected_delta": collected_after - collected_before,
            "collected": collected_after,
        })
        if verbose:
            print(f"  stroke {len(strokes) - 1:2d} {planner_name:13s} "
                  f"({stroke.x_start:+.3f},{stroke.y_start:+.3f})->"
                  f"({stroke.x_end:+.3f},{stroke.y_end:+.3f})  "
                  f"collected {collected_before}->{collected_after}  "
                  f"peakF={stats.get('peak_force', 0.0):.2f}N  t={env.time:.1f}s")

    collected = int(env.collected_mask().sum())
    lost = int(env.lost_mask().sum())
    n = len(env.layout)
    success = (collected / n if n else 0.0) >= success_rate
    controller.fsm.finish(success, env.time, note=failure_reason)
    if not success and not failure_reason:
        failure_reason = "collection rate below threshold"

    metrics = compute_episode_metrics(
        trace=trace,
        n_components=n,
        n_collected=collected,
        n_pushed_out=lost,
        n_strokes=len(strokes),
        sim_time=env.time,
        success=success,
        seed=seed,
        planner=planner_name,
        perception=perception_name,
        geometry=str(cfg.components.geometry),
        failure_reason="" if success else failure_reason,
        wall_time=time.time() - wall_t0,
        aborts=aborts,
        strokes_with_contact=strokes_with_contact,
        n_ride_over=total_ride_over,
        n_jam_events=total_jams,
        touchdown_overshoot_max=float(max(overshoots)) if overshoots else 0.0,
        touchdown_overshoot_mean=float(np.mean(overshoots)) if overshoots else 0.0,
        extra={"table_top_z": env.table_top_z,
               "planned_strokes": getattr(planner, "n_planned_strokes", None),
               "spawn_mode": str(cfg.components.get("spawn_mode", "uniform"))},
    )

    return EpisodeResult(
        metrics=metrics,
        trace=list(trace),
        strokes=strokes,
        observations=observations,
        events=controller.fsm.event_table(),
        layout=list(env.layout),
        config=cfg.to_dict(),
        initial_positions=np.asarray(env.initial_positions, dtype=float),
        final_positions=env.component_positions().copy(),
        component_tracks=(np.stack([p for _, p in _tracker]) if _tracker else None),
        track_times=(np.array([t for t, _ in _tracker], dtype=float) if _tracker else None),
    )
