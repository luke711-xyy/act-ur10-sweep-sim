"""Learned one-second physical rollout; NumPy codec conditioning is exogenous."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import weakref

import numpy as np
import torch
from torch import nn

from ..contracts import CandidateProgram, WorldPrediction
from .graph import HistoryEncoder, RelationGraph, candidates, codecs, last_known, references, validate_history


def direction(pair):
    norm = pair.norm(dim=-1, keepdim=True)
    default = torch.zeros_like(pair); default[..., 1] = 1
    return torch.where(norm > 1e-6, pair / norm.clamp_min(1e-6), default)


def rotate(pair, angle):
    sine, cosine = pair.unbind(-1)
    return torch.stack((sine*angle.cos()+cosine*angle.sin(), cosine*angle.cos()-sine*angle.sin()), -1)


def signature(history, refs, knots, codec_list):
    """Offline provenance only: never a neural feature or target identity."""
    digest = hashlib.sha256()
    for key in history.__dataclass_fields__:
        digest.update(getattr(history, key).detach().cpu().contiguous().numpy().tobytes())
    digest.update(knots.detach().cpu().contiguous().numpy().tobytes())
    for ref, codec in zip(refs, codec_list):
        for array in (ref.targets, ref.times, ref.valid):
            digest.update(array.tobytes())
        digest.update(json.dumps([ref.issued_at, ref.source_hash, codec.snapshot()], sort_keys=True).encode())
    return digest.digest()


def tensor_stamp(value):
    # Inference tensors deliberately have no version counter. Fingerprint them
    # so inference_mode remains supported without accepting mutated futures.
    if value.is_inference():
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).digest()
    return value._version


class GraphGRU(nn.Module):
    """No goal/GT/future-control argument. Each row has an independent codec.

    The decoder sees OWN predicted TCP and already-known normalized reference.
    Only the NumPy conditioning path (including candidate) is detached. Object,
    brush and hidden-state recurrences retain their complete25-step graph.
    """
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = HistoryEncoder() if encoder is None else encoder
        self.graph = RelationGraph()
        self.object_step = nn.GRUCell(18, 128)
        self.brush_step = nn.GRUCell(18, 128)
        self.object_readout = nn.Linear(128, 19)
        self.brush_readout = nn.Linear(128, 18)
        self._predictions = deque(maxlen=32)

    def _weights(self):
        return tuple((id(p), tensor_stamp(p)) for p in self.parameters())

    def owns_prediction(self, prediction, history, reference, candidate, codec_state):
        b = validate_history(history)
        refs, _ = references(reference, b, history.objects.device)
        knots = candidates(candidate, b, history.objects.device)
        key = signature(history, refs, knots, codecs(codec_state, b))
        if not isinstance(prediction, WorldPrediction):
            return False
        for fields, versions, source, weights in self._predictions:
            if source == key and weights == self._weights() and all(
                    pointer() is getattr(prediction, name) and tensor_stamp(getattr(prediction, name)) == version
                    for (name, pointer), version in zip(fields, versions)):
                return True
        return False

    def rollout(self, history, reference, candidate, *, codec_state):
        b = validate_history(history)
        refs, _ = references(reference, b, history.objects.device)
        knots = candidates(candidate, b, history.objects.device)
        clone = codecs(codec_state, b)
        source = signature(history, refs, knots, clone)
        programs = [CandidateProgram(row) for row in knots.detach().cpu().numpy()]
        anchors = [codec.begin_program() for codec in clone]
        encoded = self.encoder(history)
        initial, brush, object_valid, valid = last_known(history)
        objects = torch.cat((initial[..., :6], direction(initial[..., 6:8]), initial[..., 8:9]), -1)
        brush = torch.cat((brush[..., :3], direction(brush[..., 3:5]), brush[..., 5:]), -1)
        hidden = torch.cat((encoded.objects, encoded.brush[:, None]), 1)
        mask = torch.cat((object_valid, valid[:, None]), 1)
        # First invalid row at root contains the known root-held reference. Once
        # active commands execute, their future held command is unknown: use the
        # model's predicted TCP as its hold estimate, never later placeholders.
        held = [r.targets[0].copy() for r in refs]
        output = {name: [] for name in WorldPrediction.__dataclass_fields__}
        for j in range(25):
            conditions = []
            own = brush.detach().cpu().numpy()
            for row, (codec, ref, program, anchor) in enumerate(zip(clone, refs, programs, anchors)):
                tcp = np.r_[own[row, :3], np.arctan2(own[row, 3], own[row, 4])]
                active = bool(ref.valid[j])
                nominal = ref.targets[j] if active else held[row]
                decoded = codec.decode(tcp, nominal, codec.sample(program, j*.04, anchor), .04, active)
                q = decoded.target
                conditions.append(np.r_[q[:3], np.sin(q[3]), np.cos(q[3]), float(active), decoded.filtered])
            control = torch.tensor(np.stack(conditions), dtype=torch.float32, device=brush.device)
            positions = torch.cat((torch.nn.functional.pad(objects[..., :2], (0, 1)), brush[:, None, :3]), 1)
            hidden = self.graph(hidden, positions, mask)
            oh = self.object_step(torch.cat((objects, control[:, None].expand(-1, 6, -1)), -1).reshape(-1, 18),
                                  hidden[:, :6].reshape(-1, 128)).reshape(b, 6, 128)
            bh = self.brush_step(torch.cat((brush, control), -1), hidden[:, 6])
            oh = oh * object_valid[..., None]; bh = bh * valid[:, None]
            hidden = torch.cat((oh, bh[:, None]), 1)
            value, bv = self.object_readout(oh), self.brush_readout(bh)
            delta = value[..., :9]
            velocity = objects[..., 2:4] + .04*delta[..., 2:4]
            omega = objects[..., 8:9] + .04*delta[..., 8:9]
            objects = torch.cat((objects[..., :2]+.04*(velocity+delta[..., :2]), velocity,
                (objects[..., 4:6].clamp_min(1e-6).log()+.04*delta[..., 4:6].tanh()).clamp(-14, 4).exp(),
                rotate(objects[..., 6:8], 2*.04*omega[..., 0]), omega), -1)
            brush = torch.cat((brush[..., :3]+.04*(brush[..., 5:8]+bv[..., :3]),
                rotate(brush[..., 3:5], .04*(brush[..., 8]+bv[..., 3])),
                brush[..., 5:8]+.04*bv[..., 5:8], brush[..., 8:9]+.04*bv[..., 8:9]), -1)
            for row, ref in enumerate(refs):
                if ref.valid[j]:
                    row_brush = brush[row].detach().cpu().numpy()
                    held[row] = np.r_[row_brush[:3], np.arctan2(row_brush[3], row_brush[4])]
            for name, tensor in dict(object_mean=objects, object_log_std=value[..., 9:13].clamp(-7, 3),
                    state_logits=value[..., 13:16], shape_logits=value[..., 16:18],
                    visibility_logits=value[..., 18:19], brush=brush, brush_log_std=bv[..., 9:18].clamp(-7, 3),
                    valid=valid.to(torch.float32)).items():
                output[name].append(tensor)
        prediction = WorldPrediction(**{name: torch.stack(values, 1) for name, values in output.items()})
        fields = [(name, weakref.ref(getattr(prediction, name))) for name in output]
        self._predictions.append((fields, [tensor_stamp(getattr(prediction, name)) for name in output], source, self._weights()))
        return prediction
