"""Geometry-event rewards. No simulator force truth is read here."""
import numpy as np


class ContactReward:
    def __init__(self, target_count, components=6, dwell=.020):
        if type(target_count) is not int or not 1 <= target_count <= components:
            raise ValueError('target must be an integer in 1..6')
        if components != 6 or not np.isfinite(dwell) or dwell <= 0:
            raise ValueError('invalid six-component contact contract')
        self.target_count, self.dwell = target_count, float(dwell)
        self.seen = np.zeros(components, dtype=bool)
        self.durations = np.zeros(components, dtype=np.float64)
        self.finished = False
        self.event_return = self.rl_contact_return = 0.
        self.short_contact_breaks = 0

    @property
    def unique_count(self):
        return int(self.seen.sum())

    def observe(self, mask, dt, *, active):
        if self.finished:
            raise RuntimeError('episode reward already finished')
        mask = np.asarray(mask)
        if mask.shape != (6,) or mask.dtype != bool or not np.isfinite(dt) or dt <= 0:
            raise ValueError('invalid contact mask or physical time increment')
        self.short_contact_breaks += int(np.sum((~mask) & (self.durations > 0) & (~self.seen)))
        self.durations = np.where(mask, self.durations + float(dt), 0.)
        new = (~self.seen) & (self.durations >= self.dwell - 1e-10)
        count = int(new.sum())
        quota = min(count, max(0, self.target_count - self.unique_count))
        excess = count - quota
        event_reward = (60. * quota - 120. * excess) / self.target_count
        self.seen |= new
        credited = event_reward if active else 0.
        self.event_return += event_reward
        self.rl_contact_return += credited
        return dict(reward=credited, event_reward=event_reward,
                    quota_events=quota, excess_events=excess,
                    new_mask=new.copy(), unique_count=self.unique_count)

    def finish(self, outcome):
        if self.finished:
            raise RuntimeError('terminal reward cannot be stacked or repeated')
        if outcome not in {'success', 'failure', 'safety'}:
            raise ValueError('infrastructure/budget interruptions have no terminal task reward')
        self.finished = True
        return {'success': 60., 'failure': -20., 'safety': -60.}[outcome]


def classify_failure(error):
    """Narrow known physical errors only; unknown/runtime faults stop the run."""
    text = str(error).lower()
    if any(term in text for term in ('unsafe robot collision', 'force ceiling',
                                    'safety termination', 'workspace violation')):
        return 'safety'
    if ('ik' in text and any(term in text for term in
                            ('residual=', 'unreachable', 'failed', 'no solution', 'solve'))):
        return 'failure'
    return None
