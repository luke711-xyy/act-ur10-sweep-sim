"""ACT policy inference against the MuJoCo sweep environment."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .dataset import load_dataset_stats, unnormalize_action
from .interface import ACTObservationBuilder
from .policy import build_act_policy
from .realtime import ActionChunkScheduler
from .rollout import ActRolloutResult, _force_loop, _rate_limit_action


def run_act_episode(cfg, seed: int = 0, model_path: str | None = None,
                    simulation_preview: bool | None = None) -> ActRolloutResult:
    """Run one learned episode with timestamped asynchronous ACT chunks.

    The policy owns absolute ``x, y, z, yaw`` targets.  Once the brush is near
    the table, the same one-dimensional admittance loop used by demonstrations
    owns only Z; XY and yaw remain policy outputs.  In simulation-preview mode,
    MuJoCo time pauses while ACT is queried, so slow inference does not make the
    predicted chunk stale.  Strict asynchronous timing remains available for
    real-time evaluation.
    """
    if simulation_preview is None:
        simulation_preview = bool(cfg.act.get("simulation_preview", True))
    env = SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    total = len(env.layout)
    policy, _ = build_act_policy(cfg, pretrained_path=model_path)
    policy.eval()
    stats = load_dataset_stats(model_path)
    builder = ACTObservationBuilder(cfg, stats=stats)
    scheduler = ActionChunkScheduler(cfg)
    scheduler.reset(env.time)
    admittance = _force_loop(cfg)
    contact = False
    z_nominal = float(cfg.table.top_z) + 0.001
    last_action = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
    observations, actions, trace = [], [], []
    pending: Future | None = None
    issued_at = 0.0
    executor = (None if simulation_preview else
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-inference"))
    failure = ""
    peak_force = 0.0
    inference_queries = 0
    inference_timeouts = 0
    path_length = 0.0
    previous_xy = env.tcp()[:2].copy()
    stable_since = None
    stable_confirmed = False

    def predict(observation):
        import torch

        batch = builder.torch_batch(observation, str(policy.config.device))
        with torch.no_grad():
            values = policy.predict_action_chunk(batch).detach().cpu().numpy()[0]
        return unnormalize_action(values, stats)

    try:
        while env.time < float(cfg.episode.max_time):
            if pending is None and scheduler.query_due(env.time):
                obs = builder.observe(env)
                if len(observations) < 2000:
                    observations.append(obs)
                issued_at = float(env.time)
                scheduler.mark_query(env.time)
                inference_queries += 1
                if simulation_preview:
                    try:
                        # Do not advance MuJoCo while waiting.  The chunk is
                        # then aligned to this live pose even when wall-clock
                        # inference exceeds the real-time budget.
                        values = predict(obs)
                        if not scheduler.accept(issued_at, values, env.time,
                                                allow_late=True):
                            inference_timeouts += 1
                    except Exception as exc:
                        failure = f"ACT inference failed: {exc}"
                        break
                else:
                    pending = executor.submit(predict, obs)
            if pending is not None and pending.done():
                try:
                    values = pending.result()
                    if not scheduler.accept(issued_at, values, env.time):
                        inference_timeouts += 1
                except Exception as exc:  # surface a concise rollout failure
                    failure = f"ACT inference failed: {exc}"
                    break
                pending = None

            action = scheduler.action_for(env.time)
            if action is None:
                # A missed chunk is held at the latest safe absolute pose.
                action = last_action.copy()
            action = np.asarray(action, dtype=np.float32).reshape(4)
            action[0] = np.clip(action[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
            action[1] = np.clip(action[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
            action[2] = np.clip(action[2], float(cfg.workspace.z_search_min),
                                float(cfg.end_effector.z_home))
            action[3] = np.arctan2(np.sin(action[3]), np.cos(action[3]))
            action = _rate_limit_action(cfg, action, env, contact)
            last_action = action.copy()
            actions.append(action.copy())

            for _ in range(max(1, int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz))))):
                measured = float(env.normal_force())
                release_request = float(action[2]) > (
                    float(cfg.workspace.z_search_start)
                    + 0.5 * float(cfg.workspace.z_travel)
                )
                if contact and release_request:
                    contact = False
                    z_nominal = float(cfg.table.top_z) + 0.001
                    admittance.reset()
                near_table = float(env.tcp()[2]) <= float(cfg.workspace.z_search_start) + 0.025
                if (not contact and not release_request and near_table
                        and measured >= float(cfg.controller.contact_threshold)):
                    contact = True
                    z_nominal = float(env.table_top_z) + 0.001
                    admittance.reset()
                if contact:
                    z = z_nominal + admittance.step(float(cfg.controller.desired_force), measured)
                else:
                    z = float(action[2])
                env.step_control(Command(float(action[0]), float(action[1]), float(z), float(action[3])))
                post_force = float(env.normal_force())
                peak_force = max(peak_force, post_force)
                tcp = env.tcp()
                if contact:
                    path_length += float(np.linalg.norm(tcp[:2] - previous_xy))
                previous_xy = tcp[:2].copy()
                current_count = int(env.collected_mask().sum())
                trace.append({"t": env.time, "tcp": tcp.copy(), "command": action.copy(),
                              "normal_force": post_force, "contact": contact,
                              "collected": current_count})
                if current_count == total:
                    if stable_since is None:
                        stable_since = float(env.time)
                    elif float(env.time) - stable_since >= float(cfg.episode.stable_time):
                        stable_confirmed = True
                        break
                else:
                    stable_since = None
                if contact and peak_force > float(cfg.controller.safe_max_force):
                    failure = "normal force exceeded safety threshold"
                    break
                if path_length > float(cfg.episode.max_contact_path):
                    failure = "contact path exceeded limit"
                    break
            if stable_confirmed:
                break
            if failure:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    collected = int(env.collected_mask().sum())
    total = len(env.layout)
    if not failure and (not contact or collected != total or not stable_confirmed):
        failure = "not all components were collected"
    result = ActRolloutResult(success=not failure and total > 0,
                              failure_reason=failure,
                              observations=observations,
                              actions=np.asarray(actions, dtype=np.float32),
                              trace=trace, collected=collected, total=total,
                              elapsed=float(env.time),
                              inference_queries=inference_queries,
                              inference_timeouts=inference_timeouts,
                              simulation_preview=bool(simulation_preview))
    env.close()
    return result


if __name__ == "__main__":
    import argparse

    from ..config import load_config

    parser = argparse.ArgumentParser(description="Run one ACT policy episode")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--strict-realtime", action="store_true",
                        help="enforce the configured inference deadline instead of pausing simulation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_act_episode(cfg, seed=args.seed, model_path=args.model,
                             simulation_preview=not args.strict_realtime)
    print({"success": result.success, "reason": result.failure_reason,
           "collected": result.collected, "total": result.total,
           "elapsed": result.elapsed, "scheduler_actions": len(result.actions),
           "inference_queries": result.inference_queries,
           "inference_timeouts": result.inference_timeouts,
           "simulation_preview": result.simulation_preview})
