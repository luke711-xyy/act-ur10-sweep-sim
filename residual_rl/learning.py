"""CPU recurrent SAC and whole-episode replay, independent of ACT and physics.

Replay actions are the actor's *raw normalized* tanh samples. Applied physical
residuals belong to the controller and must never replace those actions here.
The caller owns episode hidden states, collection, and the update schedule.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import math
from dataclasses import asdict, dataclass, fields
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DEFAULT, STABILITY
from .policy_guard import ProbeBank, ProbeHistory, PolicyUpdateGuard, zero_residual_prior


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value, name, boolean=False):
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if boolean:
        if value.dtype != np.bool_:
            raise ValueError(f"{name} must have boolean dtype")
        result = value.copy()
    else:
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite real numbers")
        with np.errstate(over="ignore"):
            result = value.astype(np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError(f"{name} is outside the float32 range")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Episode:
    episode_id: str
    target_count: int
    actor_obs: np.ndarray
    critic_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    learn_mask: np.ndarray

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        goal = _integer(self.target_count, "target_count")
        if goal > 6:
            raise ValueError("target_count must be in 1..6")
        object.__setattr__(self, "target_count", goal)
        for name in ("actor_obs", "critic_obs", "actions", "rewards", "terminated", "learn_mask"):
            object.__setattr__(self, name, _array(getattr(self, name), name,
                                                boolean=name in {"terminated", "learn_mask"}))
        steps = len(self.rewards) if self.rewards.ndim == 1 else 0
        if steps < 1 or self.actions.shape != (steps, 3):
            raise ValueError("expected rewards[T] and raw normalized actions[T,3], T > 0")
        if np.any(np.abs(self.actions) > 1):
            raise ValueError("actions must be raw normalized tanh samples in [-1,1]")
        for name in ("actor_obs", "critic_obs"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[0] != steps + 1 or array.shape[1] < 1:
                raise ValueError(f"{name} must have shape [T+1, positive dimension]")
        if self.terminated.shape != (steps,) or self.learn_mask.shape != (steps,):
            raise ValueError("terminated and learn_mask must have shape [T]")
        if self.terminated[:-1].any():
            raise ValueError("an episode cannot continue after termination")


def _episode_state(episode):
    return {field.name: (getattr(episode, field.name).copy()
                         if isinstance(getattr(episode, field.name), np.ndarray)
                         else getattr(episode, field.name)) for field in fields(Episode)}


def _copy_episode(episode):
    if not isinstance(episode, Episode):
        raise ValueError("expected an Episode")
    return Episode(**_episode_state(episode))


@dataclass(frozen=True)
class SequenceSample:
    """A detached episode plus requested window; entries after T are padding.

    The learner evaluates only real entries, equivalent to zero loss on padding.
    Observations before start supply burn-in context with no gradient/loss.
    """

    episode: Episode
    start: int
    length: int

    def __post_init__(self):
        if not isinstance(self.episode, Episode):
            raise ValueError("expected an Episode")
        start = _integer(self.start, "start", 0)
        _integer(self.length, "length")
        if start >= len(self.episode.rewards):
            raise ValueError("start must identify a real transition")
        if not self.episode.learn_mask[start:start + self.length].any():
            raise ValueError("window must contain a learnable transition")


class EpisodicReplay:
    """Capacity counts all T steps (including prefix), evicts whole oldest episodes.

    Goals rotate 1..6 across calls, even when batch_size is not divisible by six.
    Sampling requires at least one eligible episode for *every* goal. It does not
    impose the learner's warmup threshold. Returned episodes never alias storage.
    """

    def __init__(self, capacity=100_000, seed=0):
        self.capacity = _integer(capacity, "capacity")
        self._rng = np.random.default_rng(_integer(seed, "seed", 0))
        self._episodes = []
        self._seen_ids = set()
        self._dimensions = None
        self._goal_cursor = 0

    @property
    def episodes(self):
        return tuple(_copy_episode(episode) for episode in self._episodes)

    @property
    def steps(self):
        return sum(len(episode.rewards) for episode in self._episodes)

    @property
    def learned_steps(self):
        return sum(int(episode.learn_mask.sum()) for episode in self._episodes)

    def add(self, episode):
        episode = _copy_episode(episode)
        dimensions = (episode.actor_obs.shape[1], episode.critic_obs.shape[1])
        if episode.episode_id in self._seen_ids:
            raise ValueError("duplicate episode_id (including previously evicted IDs)")
        if len(episode.rewards) > self.capacity:
            raise ValueError("episode exceeds replay capacity")
        if self._dimensions is not None and dimensions != self._dimensions:
            raise ValueError("inconsistent replay observation dimensions")
        self._dimensions = dimensions
        self._episodes.append(episode)
        self._seen_ids.add(episode.episode_id)
        while self.steps > self.capacity:
            self._episodes.pop(0)

    def ready(self, min_steps=256):
        min_steps = _integer(min_steps, "min_steps", 0)
        goals = {episode.target_count for episode in self._episodes if episode.learn_mask.any()}
        return self.learned_steps >= min_steps and goals == set(range(1, 7))

    def training_ready(self):
        """Stronger production gate; generic replay sampling keeps its small contract."""
        counts = {goal: [0, 0] for goal in range(1, 7)}
        for episode in self._episodes:
            steps = int(episode.learn_mask.sum())
            if episode.terminated[-1]:
                counts[episode.target_count][0] += int(steps > 0)
                counts[episode.target_count][1] += steps
        return (sum(steps for _, steps in counts.values()) >= STABILITY.warmup_steps and
                all(episodes >= STABILITY.episodes_per_goal and steps >= STABILITY.steps_per_goal
                    for episodes, steps in counts.values()))

    def sample(self, batch_size=4, length=32):
        batch_size = _integer(batch_size, "batch_size")
        length = _integer(length, "length")
        eligible = {goal: [] for goal in range(1, 7)}
        for episode in self._episodes:
            starts = np.arange(len(episode.rewards))
            cumulative = np.concatenate(([0], np.cumsum(episode.learn_mask)))
            counts = cumulative[np.minimum(starts + length, len(starts))] - cumulative[starts]
            starts = starts[counts > 0]
            if len(starts):
                eligible[episode.target_count].append((episode, starts))
        if not all(eligible.values()):
            raise ValueError("sampling requires learnable transitions for every goal 1..6")
        result = []
        for _ in range(batch_size):
            goal = self._goal_cursor + 1
            choices = eligible[goal]
            episode, starts = choices[int(self._rng.integers(len(choices)))]
            start = int(starts[int(self._rng.integers(len(starts)))])
            result.append(SequenceSample(_copy_episode(episode), start, length))
            self._goal_cursor = (self._goal_cursor + 1) % 6
        return result

    def state_dict(self):
        return {"schema_version": 2, "action_contract":DEFAULT.action_contract,
                "reward_contract":DEFAULT.reward_contract,"capacity": self.capacity,
                "episodes": [_episode_state(episode) for episode in self._episodes],
                "seen_ids": sorted(self._seen_ids), "dimensions": self._dimensions,
                "goal_cursor": self._goal_cursor,
                "rng_state": copy.deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state):
        """Validate an entire detached snapshot before replacing live state."""
        try:
            if (not isinstance(state, dict) or _integer(state["schema_version"], "schema_version") != 2
                    or state['action_contract']!=DEFAULT.action_contract
                    or state['reward_contract']!=DEFAULT.reward_contract):
                raise ValueError("unsupported replay schema")
            candidate = EpisodicReplay(state["capacity"])
            if not isinstance(state["episodes"], list):
                raise ValueError("episodes must be a list")
            for payload in state["episodes"]:
                candidate.add(Episode(**payload))
            if len(candidate._episodes) != len(state["episodes"]):
                raise ValueError("snapshot exceeds replay capacity")
            dimensions = state["dimensions"]
            if dimensions is not None:
                dimensions = tuple(_integer(value, "dimension") for value in dimensions)
                if len(dimensions) != 2:
                    raise ValueError("invalid observation dimensions")
            if candidate._dimensions is not None and candidate._dimensions != dimensions:
                raise ValueError("snapshot dimensions disagree with episodes")
            candidate._dimensions = dimensions
            ids = state["seen_ids"]
            if (not isinstance(ids, list) or any(not isinstance(i, str) or not i.strip() for i in ids)
                    or len(set(ids)) != len(ids) or not candidate._seen_ids.issubset(set(ids))):
                raise ValueError("invalid seen episode IDs")
            candidate._seen_ids = set(ids)
            cursor = _integer(state["goal_cursor"], "goal_cursor", 0)
            if cursor > 5:
                raise ValueError("invalid goal cursor")
            candidate._goal_cursor = cursor
            candidate._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid replay snapshot") from exc
        self.__dict__.update(candidate.__dict__)


class _Actor(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        self.gru = nn.GRU(observation_dim, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.mean = nn.Linear(128, 3)
        self.log_std = nn.Linear(128, 3)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -4)

    def distribution(self, features):
        features = self.head(features)
        return self.mean(features), self.log_std(features).clamp(-5, 0)

    @staticmethod
    def sample(mean, log_std, generator):
        """Reparameterized raw tanh action and its density in normalized space."""
        noise = torch.randn(mean.shape, dtype=mean.dtype, device="cpu", generator=generator)
        pre_tanh = mean + log_std.exp() * noise
        gaussian = -0.5 * noise.square() - log_std - 0.5 * math.log(2 * math.pi)
        # Stable log(1 - tanh(x)^2), including saturated float32 actions.
        log_jacobian = 2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))
        return pre_tanh.tanh(), (gaussian - log_jacobian).sum(-1)


class _Critic(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        # History at state t contains replay a[t-1]. Candidate a[t] goes only
        # into the Q head, so changing the candidate cannot rewrite history.
        self.gru = nn.GRU(observation_dim + 3, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128 + 3, 128), nn.ReLU(), nn.Linear(128, 1))

    def value(self, features, action):
        return self.head(torch.cat((features, action), dim=-1)).squeeze(-1)


def _unroll(gru, inputs, start, stop):
    hidden = None
    if start:
        with torch.no_grad():
            _, hidden = gru(inputs[:, :start])
    output, _ = gru(inputs[:, start:stop], hidden)
    return output


class RecurrentSAC:
    """Independent CPU residual policy; updates only when explicitly requested.

    Hidden state is always caller-owned: pass None at an episode boundary and
    retain the returned tensor throughout the whole episode, including prefix.
    Checkpoints contain this learner's local torch RNG; replay owns its numpy RNG
    and must be saved separately. Neither construction nor loading auto-resumes.
    """

    _CONFIG = {"hidden_dim": 128, "head_dim": 128, "gru_layers": 1,
               "action_dim": 3, "actor_lr": STABILITY.actor_lr,
               "critic_lr": STABILITY.critic_lr, "tau": 0.005, "gamma": 1.0,
               "alpha": 0.001, "grad_clip": 5.0, "batch_size": 4,
               "length": 32, "min_steps": 256, "log_std_min": -5.0,
               "log_std_max": 0.0, "device": "cpu", "prior_weight": STABILITY.prior_weight,
               "action_contract":DEFAULT.action_contract,"reward_contract":DEFAULT.reward_contract}

    def __init__(self, actor_dim, critic_dim, seed=0):
        self.actor_dim = _integer(actor_dim, "actor_dim")
        self.critic_dim = _integer(critic_dim, "critic_dim")
        seed = _integer(seed, "seed", 0)
        self.config = self._CONFIG.copy()
        # Network initialization is deterministic without consuming global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = _Actor(self.actor_dim).cpu()
            self.critics = nn.ModuleList([_Critic(self.critic_dim), _Critic(self.critic_dim)]).cpu()
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.config["critic_lr"])
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self.updates = 0
        self.actor_updates = 0
        self.rolled_back_actor_updates = 0
        self.failed_actor_updates = 0
        self.guard_failed = False
        self._probe_bank = None
        self._probe_spec = None
        self._round_guard = None
        self._round_critic_updates = 0
        self._round_actor_updates = 0

    def act(self, observation, hidden=None, deterministic=False):
        if self.guard_failed:
            raise RuntimeError('rejected residual policy is stopped; no automatic fallback/resume')
        observation = _array(observation, "observation")
        if observation.shape != (self.actor_dim,):
            raise ValueError(f"actor observation must have shape ({self.actor_dim},)")
        if hidden is not None:
            if (not isinstance(hidden, torch.Tensor) or hidden.shape != (1, 1, 128)
                    or hidden.device.type != "cpu" or hidden.dtype != torch.float32
                    or not torch.isfinite(hidden).all()):
                raise ValueError("hidden must be a finite CPU float32 tensor [1,1,128]")
            hidden = hidden.detach()
        if not isinstance(deterministic, (bool, np.bool_)):
            raise ValueError("deterministic must be boolean")
        with torch.no_grad():
            features, next_hidden = self.actor.gru(torch.tensor(observation).reshape(1, 1, -1), hidden)
            mean, log_std = self.actor.distribution(features[0, 0])
            action = mean.tanh() if deterministic else self.actor.sample(mean, log_std, self._rng)[0]
        return action.numpy().copy(), next_hidden.detach()

    @contextmanager
    def update_round(self, replay, latest_episode=None):
        """Atomic Actor transaction across a capped between-episode update burst.

        Critics may learn, but no candidate Actor may be deployed unless the
        WHOLE round passes. Rejection rolls back Actor+Adam+counter+local RNG
        and latches a fatal state. No zero-action fallback is introduced.
        """
        if self.guard_failed or self._round_guard is not None:
            raise RuntimeError('policy round is stopped or already active')
        # Setup errors must also latch the stop. Take a restoration snapshot
        # BEFORE resolving probes or evaluating the first recurrent history.
        actor_before = copy.deepcopy(self.actor.state_dict())
        optimizer_before = copy.deepcopy(self.actor_optimizer.state_dict())
        actor_updates, rng = self.actor_updates, self._rng.get_state().clone()
        guard = None
        self._round_critic_updates = self._round_actor_updates = 0
        try:
            episodes = replay.episodes
            if self._probe_bank is None:
                self._probe_bank = (ProbeBank.from_spec(self._probe_spec, episodes)
                                    if self._probe_spec is not None else ProbeBank.from_episodes(episodes))
                self._probe_spec = self._probe_bank.spec()
            latest = latest_episode if latest_episode is not None else episodes[-1]
            histories = list(self._probe_bank.histories)
            if latest.learn_mask.any():
                histories.append(ProbeHistory.from_episode(latest))
            guard = PolicyUpdateGuard(self.actor, self.actor_optimizer, histories)
            self._round_guard = guard
            guard.check()  # Reject an already-unsafe policy before any learning.
            yield guard
            guard.check()
        except BaseException as failure:
            if guard is not None:
                guard.rollback()
            else:
                self.actor.load_state_dict(actor_before)
                self.actor_optimizer.load_state_dict(optimizer_before)
                self.actor_optimizer.zero_grad(set_to_none=True)
            self.rolled_back_actor_updates += self.actor_updates-actor_updates
            self.actor_updates = actor_updates
            # A completed Critic step may be followed by an Actor operation
            # that raises before Adam completes. Account for that scheduled
            # slot explicitly, instead of hiding the Critic's real step count.
            slots = max(0, self.updates-STABILITY.critic_only_updates)//STABILITY.critics_per_actor
            self.failed_actor_updates = slots-self.actor_updates-self.rolled_back_actor_updates
            self._rng.set_state(rng)
            from .safety import BudgetStop
            # A normal collection cutoff still needs its last committed Actor
            # for final deterministic validation; it is not policy rejection.
            self.guard_failed = not (isinstance(failure, BudgetStop) and
                                     str(failure) == 'stop_collection')
            raise
        finally:
            self._round_guard = None

    def _losses(self, sample, update_actor=True):
        episode, start = sample.episode, sample.start
        if (episode.actor_obs.shape[1], episode.critic_obs.shape[1]) != (self.actor_dim, self.critic_dim):
            raise ValueError("replay dimensions do not match this learner")
        stop = min(start + sample.length, len(episode.rewards))
        actor_obs = torch.tensor(episode.actor_obs).unsqueeze(0)
        critic_obs = torch.tensor(episode.critic_obs).unsqueeze(0)
        actions = torch.tensor(episode.actions).unsqueeze(0)
        previous_actions = torch.cat((torch.zeros(1, 1, 3), actions), dim=1)
        critic_inputs = torch.cat((critic_obs, previous_actions), dim=-1)
        mask = torch.tensor(episode.learn_mask[start:stop])
        # No fabricated padded transitions: evaluate exactly the real n+1 states.
        with torch.set_grad_enabled(update_actor):
            features = _unroll(self.actor.gru, actor_obs, start, stop + 1)
            mean, log_std = self.actor.distribution(features)
        critic_features = [_unroll(critic.gru, critic_inputs, start, stop + 1)
                           for critic in self.critics]
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(mean[:, 1:], log_std[:, 1:], self._rng)
            target_values = [critic.value(
                _unroll(critic.gru, critic_inputs, start, stop + 1)[:, 1:], next_action)
                for critic in self.target_critics]
            next_value = torch.minimum(*target_values) - self.config["alpha"] * next_logp
            rewards = torch.tensor(episode.rewards[start:stop])
            terminated = torch.tensor(episode.terminated[start:stop])
            targets = rewards + self.config["gamma"] * torch.where(terminated, 0.0, next_value[0])
        values = [critic.value(history[:, :-1], actions[:, start:stop])[0]
                  for critic, history in zip(self.critics, critic_features)]
        critic_loss = sum((value[mask] - targets[mask]).square().sum() for value in values)
        if not update_actor:
            return None, critic_loss, targets[mask].sum(), int(mask.sum())
        policy_action, logp = self.actor.sample(mean[:, :-1], log_std[:, :-1], self._rng)
        # Actor gets dQ/da but never changes critic parameters or replay history.
        self.critics.requires_grad_(False)
        try:
            policy_values = [critic.value(history[:, :-1].detach(), policy_action)[0]
                             for critic, history in zip(self.critics, critic_features)]
        finally:
            self.critics.requires_grad_(True)
        actor_loss = (self.config["alpha"] * logp[0] - torch.minimum(*policy_values) +
                      self.config['prior_weight'] * zero_residual_prior(policy_action)[0])[mask].sum()
        return actor_loss, critic_loss, targets[mask].sum(), int(mask.sum())

    def update(self, replay):
        if self.guard_failed:
            raise RuntimeError('policy check previously failed; learning cannot resume')
        if not replay.ready(min_steps=self.config["min_steps"]):
            return {"skipped": True, "reason": "replay_not_ready", "updates": self.updates}
        actor_due = (self.updates + 1 > STABILITY.critic_only_updates and
                     (self.updates + 1 - STABILITY.critic_only_updates) % STABILITY.critics_per_actor == 0)
        if actor_due and self._round_guard is None:
            with self.update_round(replay):
                return self.update(replay)
        if self._round_guard is not None:
            if (self._round_critic_updates >= STABILITY.critics_per_episode or
                    actor_due and self._round_actor_updates >= STABILITY.actors_per_episode):
                raise RuntimeError('between-episode learning burst exceeded its cap')
        samples = replay.sample(batch_size=self.config["batch_size"], length=self.config["length"])
        if not samples:
            raise ValueError("ready replay returned no sequences")
        losses = [self._losses(sample, update_actor=actor_due) for sample in samples]
        count = sum(loss[3] for loss in losses)
        if count < 1:
            raise ValueError("sample contains no learnable real transitions")
        actor_loss = sum(loss[0] for loss in losses) / count if actor_due else None
        critic_loss = sum(loss[1] for loss in losses) / count
        target_mean = sum(loss[2] for loss in losses) / count
        checked_losses = (critic_loss, target_mean) + ((actor_loss,) if actor_due else ())
        if not all(torch.isfinite(loss).item() for loss in checked_losses):
            raise ValueError("nonfinite SAC loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        actor_norm = None
        if actor_due:
            actor_loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config["grad_clip"],
                                                  error_if_nonfinite=True)
        critic_norm = nn.utils.clip_grad_norm_(self.critics.parameters(), self.config["grad_clip"],
                                               error_if_nonfinite=True)
        # Critic updates are independent of sequence length and Actor cadence.
        self.critic_optimizer.step()
        self.updates += 1
        if self._round_guard is not None:
            self._round_critic_updates += 1
        with torch.no_grad():
            for target, current in zip(self.target_critics.parameters(), self.critics.parameters()):
                target.mul_(1 - self.config["tau"]).add_(current, alpha=self.config["tau"])
        if actor_due:
            self.actor_optimizer.step()
            self.actor_updates += 1
            self._round_actor_updates += 1
            self._round_guard.check()
        return {"skipped": False, "updates": self.updates,
                "actor_updates": self.actor_updates, "actor_updated": actor_due,
                "rolled_back_actor_updates": self.rolled_back_actor_updates,
                "failed_actor_updates": self.failed_actor_updates,
                "actor_loss": actor_loss.item() if actor_loss is not None else None,
                "critic_loss": critic_loss.item(), "target_q_mean": target_mean.item(),
                "learned_transitions": count,
                "actor_grad_norm": actor_norm.item() if actor_norm is not None else None,
                "critic_grad_norm": critic_norm.item()}

    def checkpoint(self):
        if self._round_guard is not None:
            raise RuntimeError('cannot publish a checkpoint from an uncommitted policy round')
        return copy.deepcopy({"schema_version": 3, "actor_dim": self.actor_dim,
                              "critic_dim": self.critic_dim, "config": self.config,
                              "stability_contract": asdict(STABILITY),
                              "actor": self.actor.state_dict(),
                              "critics": [critic.state_dict() for critic in self.critics],
                              "target_critics": [critic.state_dict() for critic in self.target_critics],
                              "actor_optimizer": self.actor_optimizer.state_dict(),
                              "critic_optimizer": self.critic_optimizer.state_dict(),
                              "rng_state": self._rng.get_state(), "updates": self.updates,
                              "actor_updates": self.actor_updates, "guard_failed": self.guard_failed,
                              "rolled_back_actor_updates": self.rolled_back_actor_updates,
                              "failed_actor_updates": self.failed_actor_updates,
                              "probe_spec": self._probe_spec})

    @staticmethod
    def _load_model(model, state):
        expected = model.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("model checkpoint schema mismatch")
        for key, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"invalid model tensor: {key}")
        model.load_state_dict(state)

    @staticmethod
    def _load_optimizer(optimizer, state, updates):
        expected = optimizer.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("optimizer checkpoint schema mismatch")
        groups = state["param_groups"]
        if not isinstance(groups, list) or len(groups) != len(expected["param_groups"]):
            raise ValueError("invalid optimizer groups")
        for actual, wanted in zip(groups, expected["param_groups"]):
            if actual != wanted:
                raise ValueError("optimizer configuration or parameter IDs mismatch")
        if not isinstance(state["state"], dict):
            raise ValueError("invalid optimizer state")
        expected_ids = {key for group in groups for key in group["params"]}
        if set(state["state"]) != (expected_ids if updates else set()):
            raise ValueError("missing or unexpected optimizer state")
        parameters = {key: parameter
                      for saved, live in zip(groups, optimizer.param_groups)
                      for key, parameter in zip(saved["params"], live["params"])}
        # Adam's loader casts moment dtypes. Validate the original payload first
        # so malformed snapshots cannot be silently repaired during loading.
        for key, values in state["state"].items():
            parameter = parameters[key]
            if not isinstance(values, dict) or values.keys() != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam state")
            for name, tensor in values.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                        or not torch.isfinite(tensor).all()):
                    raise ValueError("invalid optimizer tensor")
                if name == "step":
                    if tensor.numel() != 1 or tensor.item() != updates:
                        raise ValueError("optimizer step disagrees with update count")
                elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError("optimizer moment shape/dtype mismatch")
                elif name == "exp_avg_sq" and (tensor < 0).any():
                    raise ValueError("Adam squared moments must be nonnegative")
        optimizer.load_state_dict(copy.deepcopy(state))

    def load_checkpoint(self, payload):
        """Strict, atomic load of models, targets, Adam states and local RNG."""
        try:
            if not isinstance(payload, dict) or _integer(payload["schema_version"], "schema_version") != 3:
                raise ValueError("unsupported learner checkpoint schema")
            actor_dim = _integer(payload["actor_dim"], "actor_dim")
            critic_dim = _integer(payload["critic_dim"], "critic_dim")
            if (actor_dim, critic_dim) != (self.actor_dim, self.critic_dim):
                raise ValueError("checkpoint dimensions do not match this learner")
            if payload["config"] != self._CONFIG:
                raise ValueError("checkpoint configuration does not match the fixed SAC contract")
            if payload['stability_contract'] != asdict(STABILITY):
                raise ValueError('checkpoint policy stability contract differs')
            candidate = RecurrentSAC(actor_dim, critic_dim)
            candidate.updates = _integer(payload["updates"], "updates", 0)
            candidate.actor_updates = _integer(payload['actor_updates'], 'actor_updates', 0)
            candidate.rolled_back_actor_updates = _integer(payload['rolled_back_actor_updates'],
                                                          'rolled_back_actor_updates', 0)
            candidate.failed_actor_updates = _integer(payload['failed_actor_updates'], 'failed_actor_updates', 0)
            allowed_actors = max(0, candidate.updates-STABILITY.critic_only_updates)//STABILITY.critics_per_actor
            if candidate.actor_updates > allowed_actors or type(payload['guard_failed']) is not bool:
                raise ValueError('invalid independent Actor count or guard state')
            candidate.guard_failed = payload['guard_failed']
            if (candidate.actor_updates + candidate.rolled_back_actor_updates +
                    candidate.failed_actor_updates) != allowed_actors:
                raise ValueError('Actor count disagrees with the frozen update cadence')
            from .policy_guard import validate_probe_spec
            candidate._probe_spec = validate_probe_spec(payload['probe_spec'])
            if candidate.actor_updates and candidate._probe_spec is None:
                raise ValueError('trained Actor lacks its fixed probe identity')
            self._load_model(candidate.actor, payload["actor"])
            for key in ("critics", "target_critics"):
                if not isinstance(payload[key], list) or len(payload[key]) != 2:
                    raise ValueError("checkpoint requires two independent critics and targets")
                for model, state in zip(getattr(candidate, key), payload[key]):
                    self._load_model(model, state)
            self._load_optimizer(candidate.actor_optimizer, payload["actor_optimizer"], candidate.actor_updates)
            self._load_optimizer(candidate.critic_optimizer, payload["critic_optimizer"], candidate.updates)
            rng = payload["rng_state"]
            if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.device.type != "cpu":
                raise ValueError("invalid CPU torch RNG state")
            candidate._rng.set_state(rng.clone())
        except (KeyError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError("invalid learner checkpoint") from exc
        self.__dict__.update(candidate.__dict__)
