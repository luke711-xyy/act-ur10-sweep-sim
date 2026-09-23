"""Small, bounded residual actor-critic for the early V4 ACT policy.

The original ACT policy is intentionally treated as a frozen base policy.  The
residual actor sees the V4 low-dimensional state, the current absolute ACT
command, and the contact latch, and emits a bounded *per-step additive*
correction.  TD3 supplies twin critics, target-action smoothing, and delayed
actor updates while keeping the correction policy small and inspectable.

This module does not import MuJoCo or LeRobot at import time.  The simulator
and the V4 checkpoint remain optional for unit tests and for the legacy planner
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional training extra
        raise RuntimeError(
            "Residual RL requires PyTorch. Install the ACT extra first: "
            "uv pip install -e '.[act]'"
        ) from exc
    return torch, nn, F


@dataclass(frozen=True)
class ResidualSpec:
    observation_dim: int = 619
    action_dim: int = 4
    hidden_dims: tuple[int, ...] = (256, 256)
    residual_limit: tuple[float, ...] = (0.08, 0.08, 0.06, 0.50)
    plan_horizon: int = 20
    visual_context_dim: int = 512

    def __post_init__(self):
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("residual dimensions must be positive")
        if len(self.residual_limit) != self.action_dim:
            raise ValueError("residual_limit must have one value per action dimension")
        if any(float(value) <= 0.0 for value in self.residual_limit):
            raise ValueError("residual limits must be positive")
        expected = (26 + int(self.plan_horizon) * self.action_dim
                    + int(self.visual_context_dim) + 1)
        if self.observation_dim != expected:
            raise ValueError(
                f"residual observation_dim must be {expected} for plan horizon "
                f"{self.plan_horizon}, got {self.observation_dim}"
            )


def residual_observation(state: np.ndarray, base_action_chunk: np.ndarray,
                         visual_context: np.ndarray,
                         contact_latched: bool) -> np.ndarray:
    """Build residual input from V4 state, plan, and its fused ACT encoder token."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    base_action_chunk = np.asarray(base_action_chunk, dtype=np.float32)
    visual_context = np.asarray(visual_context, dtype=np.float32).reshape(-1)
    if base_action_chunk.ndim == 1:
        base_action_chunk = base_action_chunk.reshape(1, -1)
    if state.size != 26 or base_action_chunk.ndim != 2 or base_action_chunk.shape[1] != 4:
        raise ValueError(
            f"V4 residual input expects state/chunk (26,Hx4), got "
            f"{state.size}/{base_action_chunk.shape}"
        )
    if visual_context.size != 512:
        raise ValueError(f"ACT residual visual context must have 512 values, got {visual_context.size}")
    return np.concatenate((state, base_action_chunk.reshape(-1), visual_context,
                           np.array([float(bool(contact_latched))], dtype=np.float32)
                           )).astype(np.float32)


def clamp_residual(action: np.ndarray, spec: ResidualSpec) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(spec.action_dim)
    return np.clip(values, -np.asarray(spec.residual_limit, dtype=np.float32),
                   np.asarray(spec.residual_limit, dtype=np.float32))


class ResidualActor:
    """Tanh actor whose output is expressed in physical residual units."""

    def __init__(self, spec: ResidualSpec):
        torch, nn, _ = _torch()
        layers: list[object] = []
        width = int(spec.observation_dim)
        for hidden in spec.hidden_dims:
            layers += [nn.Linear(width, int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU()]
            width = int(hidden)
        output = nn.Linear(width, int(spec.action_dim))
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.module = nn.Sequential(*layers)
        self._limit = torch.tensor(spec.residual_limit, dtype=torch.float32)

    def to(self, device):
        self.module.to(device)
        self._limit = self._limit.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, observation):
        _, _, _ = _torch()
        return torch_tanh(observation, self.module, self._limit)

    def state_dict(self):
        return {"module": self.module.state_dict(), "limit": self._limit.detach().cpu()}

    def load_state_dict(self, state):
        self.module.load_state_dict(state["module"])


def torch_tanh(observation, module, limit):
    torch, _, _ = _torch()
    return torch.tanh(module(observation)) * limit


class ResidualCritic:
    """Q(s, residual) critic."""

    def __init__(self, spec: ResidualSpec):
        _, nn, _ = _torch()
        width = int(spec.observation_dim + spec.action_dim)
        layers: list[object] = []
        for hidden in spec.hidden_dims:
            layers += [nn.Linear(width, int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU()]
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.module = nn.Sequential(*layers)

    def to(self, device):
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, observation, action):
        _, _, _ = _torch()
        return self.module(torch_cat(observation, action))

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def torch_cat(observation, action):
    torch, _, _ = _torch()
    return torch.cat((observation, action), dim=-1)


class ResidualReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_observation = np.zeros_like(self.observation)
        self.done = np.zeros(capacity, dtype=np.float32)
        self._next = 0
        self._size = 0

    def __len__(self):
        return self._size

    def add(self, observation, action, reward, next_observation, done):
        i = self._next
        self.observation[i] = np.asarray(observation, dtype=np.float32)
        self.action[i] = np.asarray(action, dtype=np.float32)
        self.reward[i] = float(reward)
        self.next_observation[i] = np.asarray(next_observation, dtype=np.float32)
        self.done[i] = float(done)
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        if len(self) < int(batch_size):
            raise ValueError("not enough residual transitions")
        ids = rng.integers(0, len(self), size=int(batch_size))
        return tuple(array[ids] for array in (
            self.observation, self.action, self.reward,
            self.next_observation, self.done,
        ))


class ResidualAgent:
    """Bounded TD3 residual actor with a zero-correction prior."""

    def __init__(self, spec: ResidualSpec, device: str = "cpu", gamma: float = 0.99,
                 tau: float = 0.005, actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4, prior_weight: float = 0.02,
                 seed: int = 0, policy_delay: int = 2,
                 target_policy_noise: float = 0.2, noise_clip: float = 0.5):
        torch, _, _ = _torch()
        self.torch = torch
        self.spec = spec
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.prior_weight = float(prior_weight)
        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = float(target_policy_noise)
        self.noise_clip = float(noise_clip)
        self.action_scale = torch.as_tensor(
            spec.residual_limit, dtype=torch.float32, device=self.device
        )
        self.rng = np.random.default_rng(int(seed))
        self.actor = ResidualActor(spec).to(self.device)
        self.actor_target = ResidualActor(spec).to(self.device)
        self.critic1 = ResidualCritic(spec).to(self.device)
        self.critic2 = ResidualCritic(spec).to(self.device)
        self.critic1_target = ResidualCritic(spec).to(self.device)
        self.critic2_target = ResidualCritic(spec).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=float(critic_lr))
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=float(critic_lr))
        self.updates = 0

    def _tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device)

    def act(self, observation: np.ndarray, explore: bool = False,
            exploration_std: float = 0.0) -> np.ndarray:
        self.actor.eval()
        with self.torch.no_grad():
            action = self.actor(self._tensor(np.asarray(observation)[None]))[0].cpu().numpy()
        if explore and exploration_std > 0.0:
            scale = np.asarray(self.spec.residual_limit, dtype=np.float32)
            action = action + self.rng.normal(
                0.0, float(exploration_std), size=self.spec.action_dim
            ).astype(np.float32) * scale
        return clamp_residual(action, self.spec)

    def update(self, batch) -> dict[str, float]:
        observation, action, reward, next_observation, done = batch
        observation = self._tensor(observation)
        action = self._tensor(action)
        normalized_action = action / self.action_scale
        reward = self._tensor(reward).reshape(-1, 1)
        next_observation = self._tensor(next_observation)
        done = self._tensor(done).reshape(-1, 1)
        with self.torch.no_grad():
            next_action = self.actor_target(next_observation) / self.action_scale
            noise = self.torch.randn_like(next_action) * self.target_policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q = self.torch.minimum(
                self.critic1_target(next_observation, next_action),
                self.critic2_target(next_observation, next_action),
            )
            target_q = reward + (1.0 - done) * self.gamma * target_q
        q1 = self.critic1(observation, normalized_action)
        q2 = self.critic2(observation, normalized_action)
        q1_loss = self.torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = self.torch.nn.functional.mse_loss(q2, target_q)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = self.torch.zeros((), device=self.device)
        predicted = self.actor(observation)
        normalized_predicted = predicted / self.action_scale
        actor_updated = self.updates % self.policy_delay == 0
        if actor_updated:
            actor_loss = -self.critic1(observation, normalized_predicted).mean()
            actor_loss = actor_loss + self.prior_weight * (normalized_predicted ** 2).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
        self.updates += 1
        return {"critic_loss": float((q1_loss + q2_loss).detach().cpu()),
                "q1_mean": float(q1.detach().mean().cpu()),
                "q2_mean": float(q2.detach().mean().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "mean_residual": float(predicted.detach().abs().mean().cpu()),
                "mean_residual_fraction": float(
                    normalized_predicted.detach().abs().mean().cpu()),
                "actor_updated": float(actor_updated)}

    def _soft_update(self, target, source):
        with self.torch.no_grad():
            for target_param, source_param in zip(
                    target.module.parameters(), source.module.parameters()):
                target_param.mul_(1.0 - self.tau).add_(source_param, alpha=self.tau)

    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spec": self.spec.__dict__,
            "device": str(self.device),
            "gamma": self.gamma,
            "tau": self.tau,
            "prior_weight": self.prior_weight,
            "policy_delay": self.policy_delay,
            "target_policy_noise": self.target_policy_noise,
            "noise_clip": self.noise_clip,
            "updates": self.updates,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "extra": extra or {},
        }
        self.torch.save(payload, path)
        return path

    def load(self, path: str | Path) -> dict:
        payload = self.torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic1_target.load_state_dict(payload["critic1_target"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(payload["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(payload["critic2_optimizer"])
        self.updates = int(payload.get("updates", 0))
        self.policy_delay = int(payload.get("policy_delay", self.policy_delay))
        self.target_policy_noise = float(payload.get(
            "target_policy_noise", self.target_policy_noise
        ))
        self.noise_clip = float(payload.get("noise_clip", self.noise_clip))
        return dict(payload.get("extra", {}))


def residual_reward(previous_count: int, current_count: int, goal_count: int,
                    path_distance: float, normal_force: float, success: bool,
                    failure: bool, desired_force: float, safe_max_force: float,
                    *, contact_started: bool = False, contact_held: bool = False,
                    contact_lost: bool = False, descent_progress: float = 0.0,
                    collection_progress_weight: float = 2.0,
                    contact_start_reward: float = 1.5,
                    contact_hold_reward: float = 0.002,
                    contact_loss_penalty: float = 1.0,
                    descent_progress_weight: float = 5.0,
                    object_progress: float = 0.0,
                    object_progress_weight: float = 0.0,
                    part_contact_fraction: float = 0.0,
                    part_contact_weight: float = 0.0) -> float:
    """Exact-count shaped reward used by both training and evaluation.

    It never rewards ``count >= goal``: crossing above the requested count is
    explicitly penalised.  The simulator may use object truth for this reward,
    but the residual observation does not contain object positions.
    """
    before = -abs(int(previous_count) - int(goal_count))
    after = -abs(int(current_count) - int(goal_count))
    reward = float(collection_progress_weight) * float(after - before)
    if int(current_count) > int(goal_count):
        reward -= 2.0 * float(int(current_count) - int(goal_count))
    reward -= 0.002 * float(max(path_distance, 0.0))
    if contact_started or contact_held:
        reward -= 0.01 * float(max(normal_force - desired_force, 0.0))
    if contact_started:
        reward += float(contact_start_reward)
    if contact_held:
        reward += float(contact_hold_reward)
    if contact_lost:
        reward -= float(contact_loss_penalty)
    reward += float(descent_progress_weight) * float(descent_progress)
    reward += float(object_progress_weight) * float(object_progress)
    reward += (float(part_contact_weight)
               * float(np.clip(part_contact_fraction, 0.0, 1.0)))
    if success:
        reward += 10.0
    if failure:
        reward -= 5.0
    if normal_force > safe_max_force:
        reward -= 10.0
    return float(reward)


def spec_from_config(cfg) -> ResidualSpec:
    section = cfg.residual_rl
    horizon = int(cfg.act.chunk_size)
    vision_dim = int(section.visual_context_dim)
    expected_observation_dim = 26 + horizon * int(section.action_dim) + vision_dim + 1
    if int(section.observation_dim) != expected_observation_dim:
        raise ValueError(
            "residual_rl.observation_dim must match V4 state + full ACT chunk + "
            f"ACT visual context + contact latch ({expected_observation_dim})"
        )
    return ResidualSpec(
        observation_dim=int(section.observation_dim),
        action_dim=int(section.action_dim),
        hidden_dims=tuple(int(value) for value in section.hidden_dims),
        residual_limit=tuple(float(value) for value in section.residual_limit),
        plan_horizon=horizon,
        visual_context_dim=vision_dim,
    )
