"""Shared relation-aware history encoding; no slot/track identity embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..actions import ActionCodec
from ..contracts import CandidateProgram, HistoryBatch, ReferencePreview


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def validate_history(history):
    if not isinstance(history, HistoryBatch):
        raise TypeError('HistoryBatch required; private records are not model inputs')
    history.__post_init__()  # Public tensors remain mutable after construction.
    return history.objects.shape[0]


def rows(value, batch, kind, name):
    if isinstance(value, kind):
        if batch != 1:
            raise ValueError(f'{name} requires one independent value per batch row')
        return [value]
    if not isinstance(value, (list, tuple)) or len(value) != batch or not all(isinstance(v, kind) for v in value):
        raise TypeError(f'{name} requires {batch} safe per-row values')
    return list(value)


def references(value, batch, device):
    result = rows(value, batch, ReferencePreview, 'reference')
    # Reconstruct to validate mutated arrays, including timing/provenance.
    result = [ReferencePreview(r.targets, r.times, r.valid, r.issued_at, r.source_hash) for r in result]
    target = torch.tensor(np.stack([r.targets for r in result]), dtype=torch.float32, device=device)
    valid = torch.tensor(np.stack([r.valid for r in result]), dtype=torch.float32, device=device)
    features = torch.cat((target[..., :3], target[..., 3:4].sin(), target[..., 3:4].cos()), -1)
    # The root-held pose is known even when inactive; later invalid rows are
    # unknown future placeholders. Pose availability must not activate ACT.
    pose_known = valid.clone()
    pose_known[:, 0] = 1
    features = torch.cat((features * pose_known[..., None], valid[..., None]), -1)
    return result, features


def candidates(value, batch, device):
    if isinstance(value, torch.Tensor):
        if (value.dtype != torch.float32 or value.device != device or tuple(value.shape) != (batch, 5, 3)
                or not torch.isfinite(value).all() or (value.abs() > 1).any()):
            raise ValueError('candidate must be finite normalized FP32 [B,5,3] on the history device')
        return value
    programs = rows(value, batch, CandidateProgram, 'candidate')
    return torch.tensor(np.stack([CandidateProgram(p.knots).knots for p in programs]), dtype=torch.float32, device=device)


def codecs(value, batch):
    states = rows(value, batch, dict, 'codec_state')
    result = []
    for state in states:
        codec = ActionCodec(bounds=state.get('bounds'))
        codec.restore(state)  # EXACT whitelist, protocol, pinned helper, workspace and values.
        result.append(codec)
    return result


def goals(value, batch, device):
    if type(value) is int and batch == 1:
        value = torch.tensor([value], device=device)
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.device != device
            or tuple(value.shape) != (batch,) or ((value < 1) | (value > 6)).any()):
        raise ValueError('goal must be int64 [B] in 1..6 (a Python int is allowed only for B=1)')
    return value - 1


def last_known(history):
    ticks = torch.arange(11, device=history.objects.device).expand(history.objects.shape[0], -1)
    index = torch.where(history.history_valid.bool(), ticks, -1).max(1).values
    batch = torch.arange(len(index), device=index.device)
    take = index.clamp_min(0)
    valid = index >= 0
    return (history.objects[batch, take], history.brush[batch, take],
            history.object_valid[batch, take].bool() & valid[:, None], valid)


def pool(objects, brush, mask):
    objects = objects * mask[..., None]
    mean = objects.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    maximum = objects.masked_fill(~mask[..., None], torch.finfo(objects.dtype).min).max(1).values
    maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum, brush), -1)


class RelationGraph(nn.Module):
    """Three shared pairwise message/GRU layers, hidden width128.

    Relative XYZ augments node features. Masked neighbors and self edges are
    omitted; the six object nodes and the distinct brush node move together.
    """
    def __init__(self):
        super().__init__()
        self.messages = nn.ModuleList(mlp([259, 128, 128]) for _ in range(3))
        self.updates = nn.ModuleList(nn.GRUCell(128, 128) for _ in range(3))

    def forward(self, nodes, positions, valid):
        b, n, _ = nodes.shape
        edges = valid[:, :, None] & valid[:, None, :]
        edges = edges & ~torch.eye(n, device=nodes.device, dtype=torch.bool)[None]
        relative = positions[:, None] - positions[:, :, None]
        for message, update in zip(self.messages, self.updates):
            pair = torch.cat((nodes[:, :, None].expand(-1, -1, n, -1),
                              nodes[:, None].expand(-1, n, -1, -1), relative), -1)
            incoming = (message(pair) * edges[..., None]).sum(2) / edges.sum(2, keepdim=True).clamp_min(1)
            nodes = update(incoming.reshape(-1, 128), nodes.reshape(-1, 128)).reshape(b, n, 128)
            nodes = nodes * valid[..., None]
        return nodes


@dataclass(frozen=True)
class EncodedHistory:
    objects: torch.Tensor
    brush: torch.Tensor
    scene: torch.Tensor
    object_valid: torch.Tensor
    valid: torch.Tensor


class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_input = nn.Linear(34, 128)
        self.brush_input = nn.Linear(24, 128)
        self.graph = RelationGraph()
        self.temporal = nn.GRUCell(128, 128)

    def forward(self, history):
        b = validate_history(history)
        hidden = history.objects.new_zeros(b, 7, 128)
        for t in range(11):
            mask = torch.cat((history.object_valid[:, t].bool(), history.history_valid[:, t:t+1].bool()), 1)
            node = torch.cat((self.object_input(history.objects[:, t]),
                              self.brush_input(torch.cat((history.brush[:, t], history.context[:, t]), -1))[:, None]), 1)
            position = torch.cat((torch.nn.functional.pad(history.objects[:, t, :, :2], (0, 1)),
                                  history.brush[:, t, None, :3]), 1)
            node = self.graph(node, position, mask)
            updated = self.temporal(node.reshape(-1, 128), hidden.reshape(-1, 128)).reshape(b, 7, 128)
            hidden = torch.where(mask[..., None], updated, hidden)
        _, _, object_valid, valid = last_known(history)
        objects = hidden[:, :6] * object_valid[..., None]
        brush = hidden[:, 6] * valid[:, None]
        return EncodedHistory(objects, brush, pool(objects, brush, object_valid), object_valid, valid)


def freeze(module):
    module.requires_grad_(False)
    module.eval()
    return module
