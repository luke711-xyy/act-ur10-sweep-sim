import pytest
import torch

from test_pac_ppo import TinyPolicy, batch


def test_resource_stop_inside_update_restores_weights_optimizer_and_rng():
    from pac_act.ppo import ChunkPPO
    from pac_act.safety import StopExperiment
    policy = TinyPolicy()
    polls = []
    def poll():
        polls.append(True)
        if len(polls) == 12:
            raise StopExperiment('resource reserve')
    trainer = ChunkPPO(policy, poll=poll)
    data = batch(policy)
    before = trainer.snapshot()
    with pytest.raises(StopExperiment, match='resource reserve'):
        trainer.train(data)
    after = trainer.snapshot()
    for key, value in before['model'].items():
        torch.testing.assert_close(after['model'][key], value, atol=0, rtol=0)
    assert after['optimizer'] == before['optimizer']
    torch.testing.assert_close(after['rng']['torch'], before['rng']['torch'], atol=0, rtol=0)
    assert trainer.last_audit['rolled_back'] is True
    assert trainer._n_updates == 0
    assert data.consumed
