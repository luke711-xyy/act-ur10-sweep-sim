"""Permutation-equivariant Top-N object selection for ObjectACT-BEV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PermutationInvariantSelectionHead(nn.Module):
    """Set-attention selector whose output permutes with input object slots."""

    def __init__(self, token_dim: int = 29, hidden_dim: int = 256, heads: int = 8):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.token_projection = nn.Linear(int(token_dim), int(hidden_dim))
        self.target_projection = nn.Sequential(
            nn.Linear(1, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.attention = nn.MultiheadAttention(
            int(hidden_dim), int(heads), batch_first=True, dropout=0.0
        )
        self.attention_norm = nn.LayerNorm(int(hidden_dim))
        self.feed_forward = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.score = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        object_tokens: torch.Tensor,
        object_valid: torch.Tensor,
        target_count: torch.Tensor,
    ) -> torch.Tensor:
        if object_tokens.ndim != 3:
            raise ValueError("object_tokens must have shape (B, S, D)")
        if object_valid.shape != object_tokens.shape[:2] or object_valid.dtype != torch.bool:
            raise ValueError("object_valid must be boolean with shape (B, S)")
        if target_count.ndim == 2 and target_count.shape[1] == 1:
            target_count = target_count[:, 0]
        if target_count.ndim != 1 or target_count.shape[0] != object_tokens.shape[0]:
            raise ValueError("target_count must have shape (B,)")
        # A causal RGB detector can temporarily miss every part (or some of
        # the parts behind the brush).  Keep the attention numerically valid
        # with a dummy unmasked key, but leave the returned logits invalid so
        # downstream selection never treats that key as a real object.
        attention_valid = object_valid.clone()
        empty_rows = ~attention_valid.any(dim=1)
        if torch.any(empty_rows):
            attention_valid[empty_rows, 0] = True
        h = self.token_projection(object_tokens)
        normalized_target = target_count.to(dtype=h.dtype).reshape(-1, 1, 1) / 6.0
        h = h + self.target_projection(normalized_target)
        attended, _ = self.attention(
            h, h, h, key_padding_mask=~attention_valid
        )
        h = self.attention_norm(h + attended)
        h = self.output_norm(h + self.feed_forward(h))
        logits = self.score(h).squeeze(-1)
        return logits.masked_fill(~object_valid, float("-inf"))


def _target_counts(target_count: torch.Tensor, batch: int) -> torch.Tensor:
    if target_count.ndim == 2 and target_count.shape[1] == 1:
        target_count = target_count[:, 0]
    if target_count.ndim != 1 or target_count.shape[0] != batch:
        raise ValueError("target_count must have shape (B,)")
    if not torch.isfinite(target_count).all():
        raise ValueError("target_count must be finite")
    rounded = target_count.to(dtype=torch.long)
    if not torch.equal(target_count, rounded.to(dtype=target_count.dtype)):
        raise ValueError("target_count must contain integer values")
    return rounded


def straight_through_top_n(
    logits: torch.Tensor,
    *,
    target_count: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select exactly N valid slots with a straight-through soft gradient."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, S)")
    if valid_mask.shape != logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with the same shape as logits")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    counts = _target_counts(target_count, logits.shape[0])
    outputs = []
    for batch_index, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        row = logits[batch_index]
        valid = valid_mask[batch_index]
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            # No visible instances is a valid transient RGB observation.  A
            # zero selection keeps the BEV branch finite and lets the next
            # causal frame recover instead of aborting the rollout.
            outputs.append(row * 0.0)
            continue
        # Occlusion can temporarily make valid_count smaller than the task
        # cardinality.  Select every currently visible instance; the policy
        # still receives the requested count through task_state and can
        # recover when the missing track reappears.
        count = min(count, valid_count)
        if count < 1:
            outputs.append(row * 0.0)
            continue
        valid_logits = row.masked_fill(~valid, torch.finfo(row.dtype).min)
        indices = torch.topk(valid_logits, k=count, dim=0, largest=True, sorted=True).indices
        hard = torch.zeros_like(row)
        hard.scatter_(0, indices, 1.0)
        soft = torch.softmax(valid_logits / float(temperature), dim=0)
        soft = soft * float(count)
        outputs.append(hard + soft - soft.detach())
    return torch.stack(outputs, dim=0)


def selection_bce_loss(
    logits: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """BCE on valid slots only; invalid detector slots never affect training."""

    if logits.shape != target_mask.shape or logits.shape != valid_mask.shape:
        raise ValueError("selection BCE tensors must have the same shape")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid_mask], target_mask.to(dtype=logits.dtype)[valid_mask]
    )


@dataclass(frozen=True)
class SelectionTeacherSchedule:
    """Teacher-forcing schedule for expert identity labels."""

    teacher_until: int = 5_000
    transition_end: int = 20_000

    def __post_init__(self) -> None:
        if self.teacher_until < 0 or self.transition_end <= self.teacher_until:
            raise ValueError("transition_end must be greater than teacher_until >= 0")

    def teacher_probability(self, step: int) -> float:
        step = int(step)
        if step <= int(self.teacher_until):
            return 1.0
        if step >= int(self.transition_end):
            return 0.0
        span = float(self.transition_end - self.teacher_until)
        return float((self.transition_end - step) / span)

    def use_teacher(
        self, step: int, *, generator: torch.Generator | None = None, device=None
    ) -> bool:
        probability = self.teacher_probability(step)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(torch.rand((), generator=generator, device=device) < probability)


def scheduled_selection(
    logits: torch.Tensor,
    *,
    target_mask: torch.Tensor,
    target_count: torch.Tensor,
    valid_mask: torch.Tensor,
    step: int,
    schedule: SelectionTeacherSchedule | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return hard expert selection early, then predicted straight-through Top-N."""

    if target_mask.shape != logits.shape or target_mask.dtype not in (torch.bool, torch.float16, torch.float32, torch.float64):
        raise ValueError("target_mask must match logits and be boolean or floating point")
    schedule = schedule or SelectionTeacherSchedule()
    if schedule.use_teacher(step, generator=generator, device=logits.device):
        return target_mask.to(dtype=logits.dtype) * valid_mask.to(dtype=logits.dtype)
    return straight_through_top_n(
        logits,
        target_count=target_count,
        valid_mask=valid_mask,
    )


def final_collection_selection_target(
    packed_track_ids: Sequence[int],
    final_collected_track_ids: Iterable[int],
    *,
    target_count: int,
) -> np.ndarray:
    """Label slots from actual final collected identities, never planner IDs."""

    packed = [int(value) for value in packed_track_ids]
    collected = [int(value) for value in final_collected_track_ids]
    count = int(target_count)
    if count < 1:
        raise ValueError("target_count must be positive")
    if len(set(packed)) != len(packed):
        raise ValueError("packed_track_ids must be unique")
    if len(set(collected)) != len(collected) or len(collected) != count:
        raise ValueError("final collected identities must contain exactly target_count unique objects")
    if not set(collected).issubset(set(packed)):
        raise ValueError("final collected identities must be present in packed tracks")
    return np.asarray([track_id in set(collected) for track_id in packed], dtype=bool)
