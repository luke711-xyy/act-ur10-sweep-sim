"""Four distinct mode embeddings with one shared successful-program MLP."""
import torch
from torch import nn

from .graph import freeze, goals, mlp, references, validate_history


class ProposalHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = freeze(encoder)
        self.goal_embedding = nn.Embedding(6, 16)
        self.mode_embedding = nn.Embedding(4, 16)
        self.reference_gru = nn.GRU(6, 64, batch_first=True)
        self.mlp = mlp([480, 256, 256, 15])

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def propose(self, history, goal, reference):
        b = validate_history(history)
        device = history.objects.device
        _, ref = references(reference, b, device)
        with torch.no_grad():
            scene = self.encoder(history).scene
        reference_code = self.reference_gru(ref)[1][-1]
        shared = torch.cat((scene, self.goal_embedding(goals(goal, b, device)), reference_code), -1)
        modes = self.mode_embedding(torch.arange(4, device=device))[None].expand(b, -1, -1)
        return self.mlp(torch.cat((shared[:, None].expand(-1, 4, -1), modes), -1)).tanh().reshape(b, 4, 5, 3)
