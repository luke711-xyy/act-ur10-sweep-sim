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


class TemporalChunkEnsembler:
    """Fuse overlapping absolute action chunks for a sparse query schedule.

    LeRobot's built-in ACT ensembler intentionally requires ``n_action_steps=1``
    and a policy query every control step.  This simulator queries at 5 Hz and
    executes at 25 Hz, so the same idea is applied to the accepted, timestamped
    chunks here instead of changing the trained policy configuration.
    """

    def __init__(self, coeff: float, max_chunks: int = 8):
        self.coeff = float(coeff)
        self.max_chunks = int(max_chunks)
        self.chunks: list[ScheduledChunk] = []

    def reset(self) -> None:
        self.chunks.clear()

    def add(self, chunk: ScheduledChunk) -> None:
        self.chunks.append(chunk)
        if len(self.chunks) > self.max_chunks:
            del self.chunks[:-self.max_chunks]

    def action_for(self, now: float, action_period: float) -> Optional[np.ndarray]:
        candidates = []
        active = []
        for chunk in self.chunks:
            age = float(now) - chunk.issued_at
            if age < 0.0 or age >= len(chunk.actions) * action_period:
                continue
            active.append(chunk)
            if float(now) < chunk.valid_from:
                continue
            index = int(np.floor(age / action_period + 1e-6))
            if 0 <= index < len(chunk.actions):
                # LeRobot's ACT ensembler weights the oldest prediction for a
                # shared execution instant most strongly, then applies
                # exp(-coeff * rank) as newer chunks are added.  ``index`` is
                # the age inside this chunk, not that ensemble rank; using it
                # here would reverse the official weighting semantics whenever
                # sparse 5 Hz queries overlap.
                candidates.append((chunk.issued_at, chunk.actions[index]))
        self.chunks = active
        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item[0]))
        values = np.asarray([item[1] for item in candidates], dtype=np.float64)
        weights = np.exp(
            -self.coeff * np.arange(len(candidates), dtype=np.float64)
        )
        weights /= max(float(weights.sum()), 1e-12)
        result = np.sum(values * weights[:, None], axis=0).astype(np.float32)
        if values.shape[1] >= 4:
            # Yaw is circular; averaging its raw radians can jump across +/-pi.
            yaw = np.arctan2(
                np.sum(weights * np.sin(values[:, 3])),
                np.sum(weights * np.cos(values[:, 3])),
            )
            result[3] = float(yaw)
        return result


def rebase_action_chunk_to_reference(
    actions: np.ndarray,
    *,
    current_reference: np.ndarray,
) -> np.ndarray:
    """Rebase a preview chunk to the current command-reference origin.

    ObjectACT predicts action deltas and the runtime integrates them into
    absolute targets in the policy command-reference frame. When preview
    mode lets a query finish late, restart the chunk at its first predicted
    point instead of replaying a stale temporal prefix. This deliberately
    does not reinterpret the target using measured TCP error.
    """
    values = np.asarray(actions, dtype=np.float32)
    current = np.asarray(current_reference, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 3:
        raise ValueError("actions must have shape (chunk, at least 3)")
    if current.shape[0] != values.shape[1] or current.shape[0] < 3:
        raise ValueError("current_reference must match the action dimension")
    anchor = values[0]
    offset = current - anchor
    aligned = values.copy()
    aligned[:, :3] += offset[:3]
    if aligned.shape[1] >= 4:
        offset_yaw = np.arctan2(
            np.sin(float(current[3] - anchor[3])),
            np.cos(float(current[3] - anchor[3])),
        )
        aligned[:, 3] = np.arctan2(
            np.sin(aligned[:, 3] + offset_yaw),
            np.cos(aligned[:, 3] + offset_yaw),
        )
    return aligned


class ActionChunkScheduler:
    """Keep MuJoCo running while ACT inference happens in another thread."""

    def __init__(self, cfg, allow_late: bool = False):
        self.query_period = 1.0 / float(cfg.act.query_hz)
        self.action_period = 1.0 / float(cfg.act.action_hz)
        self.execute_steps = int(cfg.act.execute_steps)
        self.budget = float(cfg.act.realtime_budget_seconds)
        self.allow_late = bool(allow_late)
        coeff = cfg.act.get("temporal_ensemble_coeff", None)
        self.temporal_ensemble = (
            TemporalChunkEnsembler(float(coeff))
            if coeff is not None and float(coeff) > 0.0 else None
        )
        self.chunk: Optional[ScheduledChunk] = None
        self.next_query = 0.0
        self.timeouts = 0
        self.late_results = 0

    def reset(self, now: float = 0.0) -> None:
        self.chunk = None
        if self.temporal_ensemble is not None:
            self.temporal_ensemble.reset()
        self.next_query = float(now)
        self.timeouts = 0
        self.late_results = 0

    def query_due(self, now: float) -> bool:
        return float(now) + 1e-9 >= self.next_query

    def mark_query(self, now: float) -> None:
        self.next_query = float(now) + self.query_period

    def has_active_action(self, now: float) -> bool:
        """Return whether a previously accepted chunk can still execute."""
        current = float(now)
        if self.temporal_ensemble is not None:
            return any(
                current < chunk.valid_until
                for chunk in self.temporal_ensemble.chunks
            )
        return self.chunk is not None and current < self.chunk.valid_until

    def accept(self, issued_at: float, actions, now: float) -> bool:
        """Accept a result, optionally retaining late results for preview."""
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < self.execute_steps:
            raise ValueError("ACT action chunk must be (chunk, action_dim) and contain execute_steps")
        valid_from = float(issued_at) + self.budget
        if float(now) > valid_from + 1e-9 and not self.allow_late:
            self.timeouts += 1
            return False
        if float(now) > valid_from + 1e-9:
            self.late_results += 1
        if float(now) >= float(issued_at) + values.shape[0] * self.action_period:
            # Once the whole chunk's horizon has elapsed there is no action
            # index left to align to.  Even preview mode must not replay an
            # obsolete chunk from index zero.
            self.timeouts += 1
            return False
        accepted = ScheduledChunk(
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
        if self.temporal_ensemble is not None:
            self.temporal_ensemble.add(accepted)
        else:
            self.chunk = accepted
        return True

    def accept_preview_aligned(
        self,
        issued_at: float,
        actions,
        now: float,
        *,
        current_reference: np.ndarray,
    ) -> bool:
        """Accept a late preview chunk with a fresh command/time origin."""
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < self.execute_steps:
            raise ValueError("ACT action chunk must be (chunk, action_dim) and contain execute_steps")
        if float(now) <= float(issued_at) + self.budget + 1e-9:
            return self.accept(issued_at, values, now)
        values = rebase_action_chunk_to_reference(
            values, current_reference=current_reference
        )
        if self.temporal_ensemble is not None:
            # A rebased chunk is on a new coordinate/time anchor. Old chunks
            # would otherwise be blended in a different command frame.
            self.temporal_ensemble.reset()
        accepted = ScheduledChunk(
            issued_at=float(now),
            valid_from=float(now),
            valid_until=float(now) + values.shape[0] * self.action_period,
            actions=values,
        )
        if self.temporal_ensemble is not None:
            self.temporal_ensemble.add(accepted)
        else:
            self.chunk = accepted
        self.late_results += 1
        return True

    def action_for(self, now: float) -> Optional[np.ndarray]:
        if self.temporal_ensemble is not None:
            return self.temporal_ensemble.action_for(now, self.action_period)
        if self.chunk is None:
            return None
        if float(now) < self.chunk.valid_from:
            return None
        index = int(np.floor((float(now) - self.chunk.issued_at) / self.action_period + 1e-6))
        if index < 0 or index >= self.chunk.actions.shape[0] or float(now) >= self.chunk.valid_until:
            self.chunk = None
            return None
        return self.chunk.actions[index].copy()
