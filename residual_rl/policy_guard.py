"""CPU recurrent-policy checks, independent of simulator, ACT and rewards."""
import copy
from dataclasses import dataclass

import numpy as np
import torch

from .config import DEFAULT, STABILITY


class PolicyUpdateRejected(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('residual policy update rejected: ' + ', '.join(report['violations']))


@dataclass(frozen=True)
class ProbeHistory:
    episode_id: str
    goal: int
    observations: np.ndarray
    indices: np.ndarray

    def __post_init__(self):
        observations = np.asarray(self.observations, dtype=np.float32).copy()
        indices = np.asarray(self.indices)
        if (type(self.goal) is not int or not 1 <= self.goal <= 6 or not self.episode_id
                or observations.ndim != 2 or not np.isfinite(observations).all()
                or indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
                or indices[0] < 0 or np.any(np.diff(indices.astype(np.int64)) <= 0)
                or indices[-1] >= len(observations)):
            raise ValueError('invalid recurrent probe history')
        observations = observations[:int(indices[-1])+1].copy()
        indices = indices.astype(np.int64, copy=True)
        observations.setflags(write=False); indices.setflags(write=False)
        object.__setattr__(self, 'observations', observations)
        object.__setattr__(self, 'indices', indices)

    @classmethod
    def from_episode(cls, episode, indices=None):
        if indices is None:
            indices = np.flatnonzero(episode.learn_mask)
        indices = np.asarray(indices)
        if (indices.dtype.kind not in 'iu' or np.any(indices < 0)
                or np.any(indices >= len(episode.rewards))
                or not episode.learn_mask[indices].all()):
            raise ValueError('probes may only select actually learnable states')
        return cls(episode.episode_id, episode.target_count, episode.actor_obs, indices)


class ProbeBank:
    def __init__(self, histories):
        self.histories = tuple(histories)
        counts = {goal: 0 for goal in range(1, 7)}
        for history in self.histories:
            counts[history.goal] += len(history.indices)
        if any(n != STABILITY.probe_states_per_goal for n in counts.values()):
            raise ValueError('fixed bank requires exactly 64 learnable states per goal')

    @classmethod
    def from_episodes(cls, episodes):
        histories = []
        for goal in range(1, 7):
            choices = [(episode, np.flatnonzero(episode.learn_mask))
                       for episode in episodes if episode.target_count == goal]
            total = sum(len(indices) for _, indices in choices)
            if total < STABILITY.probe_states_per_goal:
                raise ValueError('not enough real states for fixed policy probes')
            selected = np.linspace(0, total-1, STABILITY.probe_states_per_goal, dtype=int)
            offset = 0
            for episode, indices in choices:
                local = selected[(selected >= offset) & (selected < offset+len(indices))]-offset
                if len(local):
                    histories.append(ProbeHistory.from_episode(episode, indices[local]))
                offset += len(indices)
        return cls(histories)

    def spec(self):
        # Compact checkpoint references, not duplicated multi-megabyte histories.
        return [dict(episode_id=h.episode_id, goal=h.goal, indices=h.indices.tolist())
                for h in self.histories]

    @classmethod
    def from_spec(cls, spec, episodes):
        spec = validate_probe_spec(spec)
        by_id = {episode.episode_id: episode for episode in episodes}
        histories = []
        for item in spec:
            episode = by_id[item['episode_id']]
            if episode.target_count != item['goal']:
                raise ValueError('fixed probe goal changed')
            histories.append(ProbeHistory.from_episode(episode, np.asarray(item['indices'])))
        return cls(histories)


def validate_probe_spec(spec):
    if spec is None:
        return None
    if not isinstance(spec, list) or not spec:
        raise ValueError('invalid fixed probe descriptor')
    counts = {goal: 0 for goal in range(1, 7)}
    seen = set()
    for item in spec:
        if (not isinstance(item, dict) or item.keys() != {'episode_id', 'goal', 'indices'}
                or not isinstance(item['episode_id'], str) or not item['episode_id']
                or item['episode_id'] in seen or type(item['goal']) is not int
                or item['goal'] not in counts or not isinstance(item['indices'], list)
                or not item['indices'] or any(type(i) is not int or i < 0 for i in item['indices'])
                or any(a >= b for a, b in zip(item['indices'], item['indices'][1:]))):
            raise ValueError('malformed fixed recurrent probe descriptor')
        seen.add(item['episode_id'])
        counts[item['goal']] += len(item['indices'])
    if any(count != STABILITY.probe_states_per_goal for count in counts.values()):
        raise ValueError('unbalanced fixed probe descriptor')
    return copy.deepcopy(spec)


def requested_residual(action):
    """Same direct square-to-disk map as the relative target generator."""
    xy = action[..., :2]
    mapped=torch.stack((xy[...,0]*torch.sqrt(1.-xy[...,1].square()/2.),
                        xy[...,1]*torch.sqrt(1.-xy[...,0].square()/2.)),dim=-1)
    return torch.cat((mapped, action[..., 2:3]), dim=-1)


def zero_residual_prior(action):
    return requested_residual(action).square().sum(-1)


def _distribution(actor, histories):
    means, stds = [], []
    with torch.no_grad():
        for history in histories:
            observations = torch.tensor(history.observations).unsqueeze(0)
            # Full real prefix, reset only at the episode boundary. Never splice
            # selected observations into an artificial recurrent sequence.
            features, _ = actor.gru(observations)
            selected = features[0, torch.tensor(history.indices)]
            mean, log_std = actor.distribution(selected)
            means.append(mean); stds.append(log_std)
    return torch.cat(means), torch.cat(stds)


class PolicyUpdateGuard:
    """One immutable reference for the ENTIRE between-episode update round."""
    def __init__(self, actor, optimizer, histories):
        self.actor, self.optimizer = actor, optimizer
        self.histories = tuple(histories)
        if not self.histories:
            raise ValueError('policy checks require actual histories')
        self.actor_before = copy.deepcopy(actor.state_dict())
        self.optimizer_before = copy.deepcopy(optimizer.state_dict())
        self.old_mean, self.old_log_std = _distribution(actor, self.histories)
        generator = torch.Generator(device='cpu').manual_seed(STABILITY.probe_seed)
        self.noise = torch.randn((len(self.old_mean), STABILITY.probe_noise_samples, 3),
                                 generator=generator)
        self.failed = False
        self.last_report = None

    def rollback(self):
        self.actor.load_state_dict(self.actor_before)
        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_before))
        self.optimizer.zero_grad(set_to_none=True)
        self.failed = True

    def check(self):
        if self.failed:
            raise PolicyUpdateRejected(self.last_report)
        mean, log_std = _distribution(self.actor, self.histories)
        finite = all(torch.isfinite(value).all().item()
                     for value in (mean, log_std, self.old_mean, self.old_log_std))
        report = dict(accepted=False, checked_states=len(mean), mean_kl=None,
            max_xy_change_m=None, max_xy_change_fraction=None,max_yaw_change_rad=None,
            deterministic_saturation_fraction=None, sampled_saturation_fraction=None,
            violations=[])
        if not finite:
            report['violations'].append('nonfinite_policy')
        else:
            old, new = self.old_mean.double(), mean.double()
            old_log, new_log = self.old_log_std.double(), log_std.double()
            # KL(old || candidate) in pre-tanh Gaussian space. Tanh is bijective;
            # do not approximate this using rounded/saturated tanh samples.
            kl = (new_log-old_log + ((2*old_log).exp()+(old-new).square()) /
                  (2*(2*new_log).exp()) - .5).sum(-1).clamp_min(0.)
            previous = requested_residual(old.tanh())
            current = requested_residual(new.tanh())
            delta = current-previous
            raw_deterministic=mean.tanh()
            raw_stochastic=(mean[:,None,:]+log_std.exp()[:,None,:]*self.noise).tanh()
            deterministic = requested_residual(raw_deterministic)
            stochastic = requested_residual(raw_stochastic)

            def saturation(values):
                return ((torch.linalg.vector_norm(values[..., :2], dim=-1) >= STABILITY.saturation_level) |
                        (values[..., 2].abs() >= STABILITY.saturation_level)).float().mean().item()

            report.update(mean_kl=kl.mean().item(),
                max_xy_change_fraction=(torch.linalg.vector_norm(delta[:,:2],dim=-1).max()*DEFAULT.residual_xy_fraction).item(),
                max_yaw_change_rad=(delta[:, 2].abs().max()*DEFAULT.relative_yaw_limit).item(),
                deterministic_saturation_fraction=max(saturation(deterministic),saturation(raw_deterministic)),
                sampled_saturation_fraction=max(saturation(stochastic),saturation(raw_stochastic)))
            from .observations import ACTOR_DIM
            if all(h.observations.shape[1]==ACTOR_DIM for h in self.histories):
                lengths=np.concatenate([np.linalg.norm(h.observations[h.indices,-6:-4],axis=1) for h in self.histories])
                report['max_xy_change_m']=(torch.linalg.vector_norm(delta[:,:2],dim=-1)*
                    DEFAULT.residual_xy_fraction*torch.tensor(lengths)).max().item()
            for key, bound in [('mean_kl', STABILITY.max_mean_kl),
                    ('max_xy_change_fraction',STABILITY.max_xy_fraction_change),
                    ('max_yaw_change_rad', STABILITY.max_yaw_change),
                    ('deterministic_saturation_fraction', STABILITY.max_saturation_fraction),
                    ('sampled_saturation_fraction', STABILITY.max_saturation_fraction)]:
                if report[key] > bound:
                    report['violations'].append(key)
            if report['max_xy_change_m'] is not None and report['max_xy_change_m']>STABILITY.max_xy_change:
                report['violations'].append('max_xy_change_m')
        report['accepted'] = not report['violations']
        self.last_report = report
        if not report['accepted']:
            self.rollback()
            raise PolicyUpdateRejected(report)
        return report
