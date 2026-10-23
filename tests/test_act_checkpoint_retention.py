"""Milestone saves survive rolling retention and carry resumable optimizer state."""
from pathlib import Path

import pytest

from sim.act import train


def test_permanent_milestones_survive_beyond_two_rolling_saves(tmp_path):
    checkpoints = tmp_path / 'checkpoints'
    for step in (40000, 45000, 50000, 55000, 100000, 185000, 190000, 195000, 200000):
        (checkpoints / f'step_{step:06d}').mkdir(parents=True)
    train._prune_checkpoints(
        tmp_path, keep_checkpoints=2, permanent_steps=(50000, 100000, 200000),
    )
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        'step_050000', 'step_100000', 'step_190000', 'step_195000', 'step_200000',
    ]


def test_permanent_step_is_saved_without_periodic_schedule():
    assert train.checkpoint_due(50000, 1, 0, permanent_steps=(50000,))
    assert not train.checkpoint_due(50001, 1, 0, permanent_steps=(50000,))


def test_saving_later_checkpoints_keeps_milestone_weights_and_adam_state(tmp_path):
    torch = pytest.importorskip('torch')

    class SavedPolicy:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def save_pretrained(self, root, **kwargs):
            torch.save(self.weight.detach(), Path(root) / 'weights.pt')

    class SavedProcessor:
        def save_pretrained(self, root, **kwargs):
            (Path(root) / kwargs['config_filename']).write_text('{}')

    policy = SavedPolicy()
    optimizer = torch.optim.AdamW([policy.weight], lr=1e-5)
    processor = SavedProcessor()
    for step in (50000, 55000, 60000, 65000):
        optimizer.zero_grad()
        policy.weight.square().sum().backward()
        optimizer.step()
        checkpoint = train.save_checkpoint(
            policy, processor, processor, optimizer, tmp_path,
            step=step, dataset_root='unchanged_dataset', device='cpu',
            keep_checkpoints=2, permanent_steps=(50000, 100000, 200000),
            elapsed_seconds=123.5,
        )
    root = tmp_path / 'checkpoints'
    assert sorted(path.name for path in root.iterdir()) == [
        'step_050000', 'step_060000', 'step_065000',
    ]
    assert (root / 'step_050000/weights.pt').is_file()
    state = torch.load(checkpoint / 'training_state.pt', weights_only=False)
    assert state['step'] == 65000
    assert state['elapsed_seconds'] == 123.5
    assert state['dataset'] == 'unchanged_dataset'
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([0.0]))], lr=0.1)
    restored.load_state_dict(state['optimizer'])
    assert restored.param_groups[0]['lr'] == 1e-5
    adam_state = next(iter(restored.state.values()))
    assert adam_state['step'].item() == 4
    assert torch.isfinite(adam_state['exp_avg']).all()
    assert adam_state['exp_avg'].abs().sum() > 0
    assert (tmp_path / 'latest_checkpoint.txt').read_text() == 'checkpoints/step_065000'
