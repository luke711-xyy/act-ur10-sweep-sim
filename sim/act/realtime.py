"""Timestamped asynchronous action chunk scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ScheduledChunk:
    issued_at: float
    valid_from: float
    valid_until: float
    actions: np.ndarray


class ActionChunkScheduler:
    """Keep MuJoCo running while ACT inference happens in another thread."""

    def __init__(self, cfg):
        self.query_period = 1.0 / float(cfg.act.query_hz)
        self.action_period = 1.0 / float(cfg.act.action_hz)
        self.execute_steps = int(cfg.act.execute_steps)
        self.budget = float(cfg.act.realtime_budget_seconds)
        self.chunk: Optional[ScheduledChunk] = None
        self.next_query = 0.0
        self.timeouts = 0

    def reset(self, now: float = 0.0) -> None:
        self.chunk = None
        self.next_query = float(now)
        self.timeouts = 0

    def query_due(self, now: float) -> bool:
        return float(now) + 1e-9 >= self.next_query

    def mark_query(self, now: float) -> None:
        self.next_query = float(now) + self.query_period

    def accept(self, issued_at: float, actions, now: float,
               *, allow_late: bool = False) -> bool:
        """Accept an action chunk under real-time or simulation-preview timing.

        Strict real-time mode keeps the fixed query deadline.  Preview mode
        may accept a late chunk, but timestamps it at ``now`` and selects the
        matching point within the chunk so it is never replayed from a stale
        simulation pose.
        """
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < self.execute_steps:
            raise ValueError("ACT action chunk must be (chunk, action_dim) and contain execute_steps")
        age = max(0.0, float(now) - float(issued_at))
        if allow_late and age >= values.shape[0] * self.action_period:
            self.timeouts += 1
            return False
        valid_from = (float(now) if allow_late
                      else float(issued_at) + self.budget)
        if not allow_late and float(now) > valid_from + 1e-9:
            self.timeouts += 1
            return False
        self.chunk = ScheduledChunk(
            issued_at=float(issued_at),
            valid_from=valid_from,
            # The chunk is timestamped from the query instant.  If inference
            # takes the full budget, the action used at valid_from is the
            # action predicted for that future timestamp (usually index 4 at
            # 5 Hz -> 20 Hz).  This preserves temporal alignment instead of
            # replaying action[0] late.
            valid_until=float(issued_at) + values.shape[0] * self.action_period,
            actions=values,
        )
        return True

    def action_for(self, now: float) -> Optional[np.ndarray]:
        if self.chunk is None:
            return None
        if float(now) < self.chunk.valid_from:
            return None
        index = int(np.floor((float(now) - self.chunk.issued_at) / self.action_period + 1e-6))
        if index < 0 or index >= self.chunk.actions.shape[0] or float(now) >= self.chunk.valid_until:
            self.chunk = None
            return None
        return self.chunk.actions[index].copy()
