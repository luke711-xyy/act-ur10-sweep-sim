from dataclasses import replace

import pytest
import torch

from test_pac_ppo import TinyPolicy,batch


def test_finite_large_KL_keeps_all_epochs_and_accepts_the_update():
    from pac_act.config import DEFAULT
    from pac_act.ppo import ChunkPPO
    policy=TinyPolicy();trainer=ChunkPPO(policy,replace(DEFAULT,epochs=3))
    data=batch(policy);initial=policy.actor.bias.detach().clone()
    ordinary_step=trainer.update_minibatch
    def moving_step(*args):
        result=ordinary_step(*args)
        with torch.no_grad():policy.actor.bias.add_(.01)
        return result
    trainer.update_minibatch=moving_step
    audit=trainer.train(data)
    assert audit['accepted'] is True and audit['epochs']==3
    assert audit['exact_kl']>.02 and torch.isfinite(torch.tensor(audit['exact_kl']))
    assert trainer._n_updates==1 and trainer._audit_step_returns==3
    assert not torch.equal(policy.actor.bias,initial)


def test_KL_penalty_still_has_gradient_when_stopping_is_disabled():
    from pac_act.ppo import ChunkPPO
    policy=TinyPolicy();trainer=ChunkPPO(policy);data=batch(policy)
    with torch.no_grad():policy.actor.bias.add_(.01)
    parts=trainer.objective(data,torch.arange(32))
    parts['kl'].backward()
    assert float(policy.actor.bias.grad.abs().sum())>0
    expected=parts['actor_loss']+.5*parts['value_loss']-.001*parts['entropy']+3.*parts['kl']
    torch.testing.assert_close(parts['loss'],expected)


def test_nonfinite_proposal_is_still_rejected_and_rolled_back():
    from pac_act.ppo import ChunkPPO,UpdateRejected
    from pac_act.update_audit import state_hashes
    policy=TinyPolicy();trainer=ChunkPPO(policy);data=batch(policy);before=trainer.snapshot()
    def invalid(*args):
        with torch.no_grad():policy.actor.bias.fill_(float('nan'))
        return {}
    trainer.update_minibatch=invalid
    with pytest.raises(UpdateRejected,match='nonfinite'):trainer.train(data)
    assert trainer.last_audit['rollback_verified'] is True
    assert state_hashes(before)==state_hashes(trainer.snapshot())


def test_nonfinite_aggregate_KL_after_an_optimizer_step_is_rejected_and_restored():
    from pac_act.config import DEFAULT
    from pac_act.ppo import ChunkPPO,UpdateRejected
    from pac_act.update_audit import state_hashes
    policy=TinyPolicy();trainer=ChunkPPO(policy,replace(DEFAULT,epochs=1));data=batch(policy)
    before=trainer.snapshot();exact=trainer.exact_kl;calls=[]
    def overflow_after_update(data):
        value=exact(data);calls.append(value)
        if len(calls)>1:
            finite=torch.full((4,),1e38)
            assert torch.isfinite(finite).all()
            return float(finite.sum())  # Each value finite, aggregate overflows float32.
        return value
    trainer.exact_kl=overflow_after_update
    with pytest.raises(UpdateRejected,match='nonfinite'):trainer.train(data)
    assert trainer.last_audit['optimizer_steps']==1
    assert trainer.last_audit['rollback_verified'] is True
    assert state_hashes(before)==state_hashes(trainer.snapshot())


@pytest.mark.parametrize('device',['mps','cuda'])
def test_fresh_acceptance_is_CPU_only_before_resource_claim_or_model_load(tmp_path,monkeypatch,device):
    import pac_act.acceptance as module
    def forbidden():raise AssertionError('must reject device before claiming resources')
    monkeypatch.setattr(module,'require_local_slot',forbidden)
    with pytest.raises(ValueError,match='CPU'):module.preflight(tmp_path/'bundle',tmp_path/'out',device,start_index=120)
    assert not (tmp_path/'out').exists()


def test_fresh_preflight_offset_never_falls_back_to_old_or_training_seeds(tmp_path):
    from pac_act.lifecycle import next_spec
    from pac_act.storage import RunStore
    store=RunStore(tmp_path)
    first=next_spec(store,1,'preflight',start_index=120)
    assert first['seed']==8000120
    store.claim(first)
    assert next_spec(store,1,'preflight',start_index=120)['seed']==8000126
    assert next_spec(store,6,'preflight',start_index=120)['seed']==8000125
    assert next_spec(store,1)['seed']==6000000
    with pytest.raises(ValueError):next_spec(store,1,'train',start_index=120)
    with pytest.raises(ValueError):next_spec(store,1,'preflight',start_index=121)


def test_preflight_offset_cli_cannot_change_another_execution_mode():
    from pac_act.cli import main,parser
    args=parser().parse_args(['--mode','preflight','--preflight-start-index','120'])
    assert args.preflight_start_index==120
    with pytest.raises(SystemExit):main(['--mode','inspect','--preflight-start-index','120'])
