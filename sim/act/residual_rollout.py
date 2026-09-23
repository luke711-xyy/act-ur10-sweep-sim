"""MuJoCo rollout for a frozen V4 ACT plus a residual policy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..controllers.hybrid import Command
from ..environments.sweep_env import SweepEnv
from .dataset import load_dataset_stats, unnormalize_action
from .interface import ACTObservationBuilder
from .policy import build_act_policy
from .residual import (ResidualAgent, residual_observation, residual_reward,
                       clamp_residual, spec_from_config)
from .rollout import _force_loop, _rate_limit_action


@dataclass
class ResidualTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass
class ResidualRolloutResult:
    success: bool
    failure_reason: str
    transitions: list[ResidualTransition] = field(default_factory=list)
    base_actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    residual_actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    trace: list[dict] = field(default_factory=list)
    collected: int = 0
    total: int = 0
    elapsed: float = 0.0
    peak_force: float = 0.0
    contact_steps: int = 0
    first_contact_time: float | None = None
    contact_path_length: float = 0.0
    object_goal_progress: float = 0.0
    part_contact_fraction: float = 0.0
    mean_contact_force: float = 0.0
    mean_abs_residual_action: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    max_abs_residual_action: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


class V4ActionProvider:
    """Synchronous chunk provider matching the early V4 ACT execution contract."""

    def __init__(self, cfg, model_path: str):
        import torch

        self.cfg = cfg
        self.policy, _ = build_act_policy(cfg, pretrained_path=model_path)
        self.policy.eval()
        self.stats = load_dataset_stats(model_path)
        self.builder = ACTObservationBuilder(cfg, stats=self.stats)
        self.execute_steps = int(cfg.act.execute_steps)
        self.chunk = None
        self.chunk_index = self.execute_steps
        self._vision_context = None
        self._encoder_hook = self.policy.model.encoder.register_forward_hook(
            self._capture_encoder_context
        )
        self.torch = torch

    def _capture_encoder_context(self, _module, _inputs, output):
        # ACT's encoder receives [sequence, batch, dim]; token 0 is the fused
        # latent token after attending to V4's camera, robot, and count inputs.
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim != 3 or output.shape[0] < 1:
            raise RuntimeError(f"unexpected ACT encoder output shape: {tuple(output.shape)}")
        self._vision_context = output[0, 0].detach().float().cpu().numpy().copy()

    def reset(self):
        self.builder.reset()
        self.chunk = None
        self.chunk_index = self.execute_steps
        self._vision_context = None

    def close(self):
        self._encoder_hook.remove()

    def step(self, env, contact_latched: bool) -> tuple[np.ndarray, np.ndarray]:
        needs_chunk = self.chunk is None or self.chunk_index >= self.execute_steps
        observation = self.builder.observe(env, include_images=needs_chunk)
        if needs_chunk:
            batch = self.builder.torch_batch(observation, str(self.policy.config.device))
            with self.torch.no_grad():
                self.chunk = self.policy.predict_action_chunk(batch).detach().cpu().numpy()[0]
            self.chunk = unnormalize_action(self.chunk, self.stats)
            self.chunk_index = 0
            if self._vision_context is None:
                raise RuntimeError("V4 ACT query did not produce its fused encoder context")
        action_index = self.chunk_index
        action = np.asarray(self.chunk[action_index], dtype=np.float32).reshape(4)
        # The residual policy must observe the command that the controller will
        # actually execute, not an out-of-workspace raw network prediction.
        action = action.copy()
        action[0] = np.clip(action[0], float(self.cfg.workspace.x_min),
                            float(self.cfg.workspace.x_max))
        action[1] = np.clip(action[1], float(self.cfg.workspace.y_min),
                            float(self.cfg.workspace.y_max))
        action[2] = np.clip(action[2], float(self.cfg.workspace.z_search_min),
                            float(self.cfg.end_effector.z_home))
        action[3] = np.arctan2(np.sin(action[3]), np.cos(action[3]))
        plan = np.asarray(self.chunk[action_index:], dtype=np.float32)
        horizon = int(self.cfg.act.chunk_size)
        if len(plan) < horizon:
            plan = np.concatenate((plan, np.repeat(plan[-1:], horizon - len(plan), axis=0)), axis=0)
        self.chunk_index += 1
        residual_obs = residual_observation(observation["observation.state"],
                                            plan[:horizon], self._vision_context,
                                            contact_latched)
        return residual_obs, action


def _apply_action(cfg, base_action, residual, contact_latched):
    """Apply one bounded residual directly to the current absolute ACT target."""
    correction = np.asarray(residual, dtype=np.float32).reshape(4).copy()
    limits = np.asarray(cfg.residual_rl.residual_limit, dtype=np.float32)
    correction = np.clip(correction, -limits, limits)
    if contact_latched:
        correction[2] = 0.0
    action = np.asarray(base_action, dtype=np.float32).reshape(4).copy()
    action += correction
    action[0] = np.clip(action[0], float(cfg.workspace.x_min), float(cfg.workspace.x_max))
    action[1] = np.clip(action[1], float(cfg.workspace.y_min), float(cfg.workspace.y_max))
    action[2] = np.clip(action[2], float(cfg.workspace.z_search_min),
                        float(cfg.end_effector.z_home))
    action[3] = np.arctan2(np.sin(action[3]), np.cos(action[3]))
    return action.astype(np.float32), correction.astype(np.float32)


def _remaining_object_distance(env) -> float:
    """Sum distances of uncollected objects to the target rectangle.

    This uses simulator truth for reward shaping only; it is not part of the
    residual policy observation.
    """
    positions = np.asarray(env.component_positions(), dtype=np.float64)[:, :2]
    if not len(positions):
        return 0.0
    active = ~(np.asarray(env.collected_mask(), dtype=bool)
               | np.asarray(env.lost_mask(), dtype=bool))
    positions = positions[active]
    if not len(positions):
        return 0.0
    target = env.cfg.target
    dx = np.maximum(np.maximum(float(target.x_min) - positions[:, 0], 0.0),
                    positions[:, 0] - float(target.x_max))
    dy = np.maximum(np.maximum(float(target.y_min) - positions[:, 1], 0.0),
                    positions[:, 1] - float(target.y_max))
    return float(np.hypot(dx, dy).sum())


def run_residual_episode(cfg, base_model: str, agent: ResidualAgent,
                         seed: int = 0, explore: bool = True,
                         base_provider=None) -> ResidualRolloutResult:
    """Run one episode and return transitions for the residual replay buffer."""
    env = SweepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    owns_provider = base_provider is None
    provider = base_provider or V4ActionProvider(cfg, base_model)
    provider.reset()
    spec = spec_from_config(cfg)
    admittance = _force_loop(cfg)
    contact = False
    z_nominal = float(cfg.table.top_z) + 0.001
    transitions: list[ResidualTransition] = []
    base_actions, residual_actions, trace = [], [], []
    previous_count = int(env.collected_mask().sum())
    previous_xy = env.tcp()[:2].copy()
    previous_z = float(env.tcp()[2])
    previous_object_distance = _remaining_object_distance(env)
    initial_object_distance = previous_object_distance
    max_abs_residual_action = np.zeros(4, dtype=np.float32)
    path_length = 0.0
    peak_force = 0.0
    failure = ""
    target_count = (len(env.layout) if cfg.residual_rl.target_count is None
                    else int(cfg.residual_rl.target_count))
    reached_at = None
    current_obs, base_action = provider.step(env, contact)
    max_action_steps = int(np.ceil(float(cfg.episode.max_time) * float(cfg.act.action_hz)))
    for _ in range(max_action_steps):
        residual = agent.act(current_obs, explore=explore,
                             exploration_std=float(cfg.residual_rl.exploration_std))
        residual = clamp_residual(residual, spec)
        action, residual = _apply_action(cfg, base_action, residual, contact)
        max_abs_residual_action = np.maximum(max_abs_residual_action,
                                             np.abs(residual))
        action = _rate_limit_action(cfg, action, env, contact)
        base_actions.append(base_action.copy())
        residual_actions.append(residual.copy())
        step_distance = 0.0
        descent_progress = 0.0
        contact_started = False
        contact_held = False
        contact_lost = False
        step_peak_force = 0.0
        part_contact_samples = 0
        control_samples = 0
        for _ in range(max(1, int(round(float(cfg.sim.control_hz) / float(cfg.act.action_hz))))):
            measured = float(env.normal_force())
            release_request = float(action[2]) > (
                float(cfg.workspace.z_search_start)
                + 0.5 * float(cfg.workspace.z_travel)
            )
            if contact and release_request:
                # Match the V4 expert/controller contract: a lifted command is
                # an explicit contact release during a lane transfer.  Clear
                # the latch and the admittance state before the next descent.
                contact = False
                contact_lost = True
                z_nominal = float(env.table_top_z) + 0.001
                admittance.reset()
            near_table = float(env.tcp()[2]) <= float(cfg.workspace.z_search_start) + 0.025
            # The search-height command is not evidence of contact.  Keep the
            # residual rollout aligned with the corrected V4 expert loop and
            # latch Z admittance only after a measured load is present.
            if (not contact and not release_request and near_table
                    and measured >= float(cfg.controller.contact_threshold)):
                contact = True
                contact_started = True
                # Z is now owned by admittance; residual Z is masked from here.
                z_nominal = float(env.table_top_z) + 0.001
                admittance.reset()
            z = (float(z_nominal + admittance.step(float(cfg.controller.desired_force), measured))
                 if contact else float(action[2]))
            env.step_control(Command(float(action[0]), float(action[1]), z, float(action[3])))
            tcp = env.tcp()
            _, n_part_contacts = env.contact_breakdown()
            control_samples += 1
            part_contact_samples += int(n_part_contacts > 0)
            if not contact:
                # Keep the shaping signed: an actor must not learn to avoid
                # the terminal failure by commanding an upward escape.
                descent_progress += previous_z - float(tcp[2])
            previous_z = float(tcp[2])
            contact_held = contact_held or contact
            distance = float(np.linalg.norm(tcp[:2] - previous_xy))
            # ``max_contact_path`` is a task-limit on the swept/contact leg,
            # not on airborne approach or lifted lane transfers. Counting
            # those motions made an exploratory residual fail before it had a
            # chance to establish contact.
            if contact:
                path_length += distance
                step_distance += distance
            previous_xy = tcp[:2].copy()
            post_force = float(env.normal_force())
            peak_force = max(peak_force, post_force)
            step_peak_force = max(step_peak_force, post_force)
            trace.append({"t": env.time, "tcp": tcp.copy(), "command": action.copy(),
                          "base_action": base_action.copy(), "residual": residual.copy(),
                          "normal_force": post_force, "contact": contact,
                          "part_contact_count": int(n_part_contacts),
                          "component_xy": np.asarray(env.component_positions()[:, :2],
                                                     dtype=np.float32).copy(),
                          "collected": int(env.collected_mask().sum())})
            if contact and peak_force > float(cfg.controller.safe_max_force):
                failure = "normal force exceeded safety threshold"
                break
            if path_length > float(cfg.episode.max_contact_path):
                failure = "contact path exceeded limit"
                break
            if np.any(env.lost_mask()):
                failure = "component left the safe workspace"
                break
        current_count = int(env.collected_mask().sum())
        current_object_distance = _remaining_object_distance(env)
        object_progress = previous_object_distance - current_object_distance
        previous_object_distance = current_object_distance
        # Early V4 sweeps every component. Disable this shaping for exact
        # subset-count experiments so it cannot reward over-collection.
        if target_count != len(env.layout):
            object_progress = 0.0
        part_contact_fraction = (part_contact_samples / control_samples
                                 if control_samples else 0.0)
        if target_count != len(env.layout):
            part_contact_fraction = 0.0
        if current_count > target_count:
            failure = "more components entered the target region than requested"
        if current_count != target_count:
            # A temporarily correct count is not enough.  Stability must be
            # continuous; if a part leaves or an extra part enters, restart
            # the confirmation clock.
            reached_at = None
        elif reached_at is None:
            reached_at = float(env.time)
        success = bool(
            not failure and contact and current_count == target_count
            and reached_at is not None
            and float(env.time) - reached_at >= float(cfg.episode.stable_time)
        )
        done = bool(failure or success or env.time >= float(cfg.episode.max_time))
        if done and not success and not failure:
            failure = "episode time limit reached before exact collection"
        reward = residual_reward(
            previous_count, current_count, target_count, step_distance, step_peak_force,
            success, bool(failure), float(cfg.controller.desired_force),
            float(cfg.controller.safe_max_force),
            contact_started=contact_started, contact_held=contact_held,
            contact_lost=contact_lost, descent_progress=descent_progress,
            collection_progress_weight=float(cfg.residual_rl.reward_collection_progress),
            contact_start_reward=float(cfg.residual_rl.reward_contact_start),
            contact_hold_reward=float(cfg.residual_rl.reward_contact_hold),
            contact_loss_penalty=float(cfg.residual_rl.reward_contact_loss),
            descent_progress_weight=float(cfg.residual_rl.reward_descent_progress),
            object_progress=object_progress,
            object_progress_weight=float(cfg.residual_rl.reward_object_progress),
            part_contact_fraction=part_contact_fraction,
            part_contact_weight=float(cfg.residual_rl.reward_part_contact),
        )
        if done:
            next_obs = current_obs.copy()
        else:
            next_obs, next_base = provider.step(env, contact)
            # The next base action is part of the next residual observation.
            # The provider already returned the full concatenated observation.
            next_obs = next_obs
            base_action = next_base
        transitions.append(ResidualTransition(
            current_obs.copy(), residual.copy(), reward, next_obs.copy(), done
        ))
        previous_count = current_count
        current_obs = next_obs
        if done:
            break

    collected = int(env.collected_mask().sum())
    total = len(env.layout)
    if not failure and not (
            contact and collected == target_count and reached_at is not None
            and float(env.time) - reached_at >= float(cfg.episode.stable_time)):
        failure = "exact target was not held for the stability interval"
    contact_samples = [item for item in trace if item["contact"]]
    contact_times = [float(item["t"]) for item in contact_samples]
    mean_contact_force = (float(np.mean([item["normal_force"] for item in contact_samples]))
                          if contact_samples else 0.0)
    result = ResidualRolloutResult(
        success=not failure,
        failure_reason=failure,
        transitions=transitions,
        base_actions=np.asarray(base_actions, dtype=np.float32),
        residual_actions=np.asarray(residual_actions, dtype=np.float32),
        trace=trace,
        collected=collected,
        total=total,
        elapsed=float(env.time),
        peak_force=peak_force,
        contact_steps=len(contact_times),
        first_contact_time=contact_times[0] if contact_times else None,
        contact_path_length=path_length,
        object_goal_progress=initial_object_distance - previous_object_distance,
        part_contact_fraction=(
            sum(int(item.get("part_contact_count", 0) > 0) for item in trace) / len(trace)
            if trace else 0.0
        ),
        mean_contact_force=mean_contact_force,
        mean_abs_residual_action=(
            np.abs(np.asarray(residual_actions, dtype=np.float32)).mean(axis=0)
            if residual_actions else np.zeros(4, dtype=np.float32)
        ),
        max_abs_residual_action=max_abs_residual_action.copy(),
    )
    env.close()
    if owns_provider and hasattr(provider, "close"):
        provider.close()
    return result
