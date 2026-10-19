"""Matched terminal heads. Calibration fitting/version qualification is Task10."""
import torch
from torch import nn

from ..contracts import TaskScore
from .graph import RelationGraph, candidates, codecs, freeze, goals, mlp, pool, references, validate_history


class _Quality(nn.Module):
    def __init__(self, encoder, width, *, calibration_hash=None, allow_uncalibrated=False):
        super().__init__()
        self.encoder = freeze(encoder)
        if calibration_hash is not None and (not isinstance(calibration_hash, str) or not calibration_hash.strip()):
            raise ValueError('calibration_hash must be nonempty')
        self.calibration_hash = calibration_hash
        self.allow_uncalibrated = allow_uncalibrated
        self.goal_embedding = nn.Embedding(6, 16)
        self.control_encoder = mlp([171, 128, 64])
        self.mlp = mlp([width, 256, 128])
        self.success = nn.Linear(128, 1)
        self.count = nn.Linear(128, 7)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def known(self, history, goal, candidate, reference, codec_state):
        b = validate_history(history)
        device = history.objects.device
        # Terminal supervision never updates the actor through its proposals.
        knots = candidates(candidate, b, device).detach()
        _, ref = references(reference, b, device)
        clone = codecs(codec_state, b)
        # New program starts at latest executed raw, not an old program anchor.
        memory = torch.tensor([c.snapshot()['current_raw']+c.snapshot()['filtered'] for c in clone],
                              dtype=torch.float32, device=device)
        with torch.no_grad():
            encoded = self.encoder(history)
        control = self.control_encoder(torch.cat((knots.flatten(1), memory, ref.flatten(1)), -1))
        return torch.cat((encoded.scene, self.goal_embedding(goals(goal, b, device)), control), -1), encoded

    def score(self, value):
        if self.calibration_hash is None and not self.allow_uncalibrated:
            raise RuntimeError('calibration required; explicitly allow_uncalibrated for module fixtures')
        hidden = self.mlp(value)
        return TaskScore(self.success(hidden), self.count(hidden), self.calibration_hash or 'uncalibrated-fixture')


class TaskQualityHead(_Quality):
    """Supplied prediction must belong to this frozen world/current safe inputs.

    Pass prediction=None to compute it here. External truth-shaped predictions,
    changed fields/conditions/weights are rejected. The ephemeral provenance
    cache is bounded; serialized training data must recompute world predictions.
    """
    def __init__(self, dynamics, *, calibration_hash=None, allow_uncalibrated=False):
        super().__init__(dynamics.encoder, 592, calibration_hash=calibration_hash, allow_uncalibrated=allow_uncalibrated)
        self.dynamics = freeze(dynamics)
        self.future_objects = nn.Linear(19, 128)
        self.future_brush = nn.Linear(18, 128)
        self.future_graph = RelationGraph()
        self.future_gru = nn.GRU(384, 128, batch_first=True)

    def train(self, mode=True):
        super().train(mode)
        self.dynamics.eval()
        return self

    def evaluate_task(self, history, goal, candidate, prediction, *, reference, codec_state):
        known, encoded = self.known(history, goal, candidate, reference, codec_state)
        if prediction is None:
            with torch.no_grad():
                prediction = self.dynamics.rollout(history, reference, candidate, codec_state=codec_state)
        if not self.dynamics.owns_prediction(prediction, history, reference, candidate, codec_state):
            raise ValueError('task head requires its own frozen world prediction with matching provenance')
        prediction.__post_init__()
        hidden = known.new_zeros(1, known.shape[0], 128)
        for j in range(25):
            obj = torch.cat(tuple(getattr(prediction, name)[:, j].detach() for name in
                                  ('object_mean', 'object_log_std', 'state_logits', 'shape_logits', 'visibility_logits')), -1)
            brush = torch.cat((prediction.brush[:, j].detach(), prediction.brush_log_std[:, j].detach()), -1)
            valid = prediction.valid[:, j].bool() & encoded.valid
            omask = encoded.object_valid & valid[:, None]
            mask = torch.cat((omask, valid[:, None]), 1)
            nodes = torch.cat((self.future_objects(obj), self.future_brush(brush)[:, None]), 1)
            position = torch.cat((torch.nn.functional.pad(obj[..., :2], (0, 1)), brush[:, None, :3]), 1)
            nodes = self.future_graph(nodes, position, mask)
            scene = pool(nodes[:, :6], nodes[:, 6], omask)
            next_hidden = self.future_gru(scene[:, None], hidden)[1]
            hidden = torch.where(valid[None, :, None], next_hidden, hidden)
        return self.score(torch.cat((known, hidden[0]), -1))


class HistoryOnlyQualityHead(_Quality):
    """Same safe history/reference/filter/candidate and1+7 outputs; no world."""
    def __init__(self, encoder, *, calibration_hash=None, allow_uncalibrated=False):
        super().__init__(encoder, 464, calibration_hash=calibration_hash, allow_uncalibrated=allow_uncalibrated)

    def evaluate_task(self, history, goal, candidate, *, reference, codec_state):
        known, _ = self.known(history, goal, candidate, reference, codec_state)
        return self.score(known)
