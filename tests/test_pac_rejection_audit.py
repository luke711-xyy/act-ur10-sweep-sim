from dataclasses import replace
import json

import pytest
import torch

from test_pac_ppo import TinyPolicy,batch


def force_rejection(trainer,policy):
    from pac_act.config import DEFAULT
    from pac_act.ppo import UpdateRejected
    trainer.update_minibatch(batch(policy),torch.arange(32))  # Nonempty Adam moments to restore.
    trainer.config=replace(DEFAULT,actor_lr=.1,critic_lr=.1,epochs=1)
    trainer._update_learning_rate([trainer.optimizer])
    before=trainer.snapshot()
    exact=trainer.exact_kl;passes=[]
    def injected_failure(data):
        value=exact(data);passes.append(value)
        if len(passes)>1:raise UpdateRejected('injected post-update exception; finite KL alone is not rejected')
        return value
    trainer.exact_kl=injected_failure
    with pytest.raises(UpdateRejected):trainer.train(batch(policy))
    return before


def test_rejection_is_durable_with_proposed_distribution_and_exact_rollback(tmp_path):
    from pac_act.ppo import audited_trainer
    from pac_act.update_audit import state_hashes
    policy=TinyPolicy();trainer=audited_trainer(policy,lambda:None,tmp_path,'test-identity')
    before=force_rejection(trainer,policy)
    paths=list((tmp_path/'update_audits').glob('rejected_*.json'))
    assert len(paths)==1
    audit=json.loads(paths[0].read_text())
    assert audit['accepted'] is False and audit['rollback_verified'] is True
    assert audit['state_hashes']['before']==audit['state_hashes']['restored']==state_hashes(before)
    assert audit['state_hashes']['proposed']['actor']!=audit['state_hashes']['before']['actor']
    assert audit['optimizer_steps']==1 and audit['completed_epochs']==1
    step=audit['optimizer_step_audits'][0]
    assert step['post_clip_gradient_norm']<=.500001
    assert [g['lr'] for g in step['learning_rates']]==[.1,.1]
    assert audit['distribution_after']['exact_kl']>.02
    assert len(audit['distribution_after']['proposed_means'])==32
    assert 'UpdateRejected' in audit['traceback']
    assert trainer._n_updates==0
    assert len((tmp_path/'rejected_updates.jsonl').read_text().splitlines())==1
    assert not (tmp_path/'updates.jsonl').exists()


def test_audit_write_failure_preserves_original_exception_and_restoration():
    from pac_act.ppo import ChunkPPO,UpdateRejected
    from pac_act.update_audit import state_hashes
    def broken_sink(audit):raise OSError('audit disk failure')
    policy=TinyPolicy();trainer=ChunkPPO(policy,audit_sink=broken_sink)
    before=force_rejection(trainer,policy)
    assert state_hashes(before)==state_hashes(trainer.snapshot())
    assert trainer.last_audit['rollback_verified'] is True
    assert any('audit disk failure' in e for e in trainer.last_audit['audit_errors'])


def test_nonfinite_loss_has_json_safe_rejection_audit(tmp_path):
    from pac_act.ppo import audited_trainer,UpdateRejected
    policy=TinyPolicy();trainer=audited_trainer(policy,lambda:None,tmp_path,'id')
    data=batch(policy);data.returns[0]=float('nan')
    with pytest.raises(UpdateRejected,match='nonfinite'):trainer.train(data)
    text=next((tmp_path/'update_audits').glob('*.json')).read_text()
    audit=json.loads(text)
    assert audit['optimizer_steps']==0 and audit['rollback_verified'] is True
    assert audit['distribution_before']['exact_kl']==0.
    assert audit['batch']['nonfinite_returns']==1
    assert 'NaN' not in text


def test_distribution_audit_has_physical_units_and_masks_suffix():
    from pac_act.ppo import ChunkPPO
    policy=TinyPolicy();trainer=ChunkPPO(policy);data=batch(policy)
    data.lengths.fill_(1)
    with torch.no_grad():policy.actor.bias.add_(.01)
    kl=trainer.exact_kl(data)
    audit=trainer.last_distribution_audit
    # Three active coordinates at one position, not all fifteen coordinates.
    assert audit['active_scalar_dimensions']==96
    assert kl==pytest.approx(.011641403,rel=2e-5)
    assert audit['physical_mean_shift_rms']['dx_mm']==pytest.approx(.01,rel=2e-5)
    assert audit['physical_mean_shift_rms']['dyaw_deg']==pytest.approx(.005729578,rel=2e-5)


def test_exception_after_adam_changes_weights_is_counted_as_uncertain_step_and_restored(tmp_path):
    from pac_act.ppo import audited_trainer,UpdateRejected
    policy=TinyPolicy();trainer=audited_trainer(policy,lambda:None,tmp_path,'id')
    step=trainer.optimizer.step
    def broken_step(*args,**kwargs):
        step(*args,**kwargs)
        raise RuntimeError('after Adam applied')
    trainer.optimizer.step=broken_step
    with pytest.raises(UpdateRejected,match='after Adam applied'):trainer.train(batch(policy))
    audit=trainer.last_audit
    assert audit['optimizer_step_calls']==1 and audit['optimizer_steps']==0
    assert audit['optimizer_step_may_have_partially_applied'] is True
    assert audit['distribution_after'] is None
    assert audit['rollback_verified'] is True


def test_failure_reports_attempt_all_writes_without_replacing_original_error(tmp_path,monkeypatch):
    import pac_act.update_audit as module
    error=RuntimeError('original KL rejection');paths=[]
    def broken(path,value):paths.append(path);raise OSError('full disk')
    monkeypatch.setattr(module,'write_json',broken)
    failures=module.write_failure_reports(error,{tmp_path/'acceptance.json':{},tmp_path/'status.json':{}})
    assert len(paths)==2 and len(failures)==2
    assert str(error)=='original KL rejection'
    assert len(error.__notes__)==2


@pytest.mark.parametrize('fault',['traceback','record'])
def test_diagnostic_construction_failure_cannot_bypass_rollback_or_replace_original(monkeypatch,fault):
    import pac_act.ppo as module
    import pac_act.update_audit as audit_module
    policy=TinyPolicy();trainer=module.ChunkPPO(policy)
    original=module.UpdateRejected('original physical-policy rejection')
    def corrupt(*args):
        with torch.no_grad():policy.actor.bias.add_(1.)
        raise original
    trainer.update_minibatch=corrupt
    def broken(*args,**kwargs):raise OSError('diagnostic construction failed')
    monkeypatch.setattr(module.traceback,'format_exc',broken) if fault=='traceback' else monkeypatch.setattr(audit_module,'json_safe',broken)
    before=trainer.snapshot()
    with pytest.raises(module.UpdateRejected) as caught:trainer.train(batch(policy))
    assert caught.value is original
    assert audit_module.state_hashes(before)==audit_module.state_hashes(trainer.snapshot())
    assert trainer.last_audit['accepted'] is False
    assert trainer.last_audit['rollback_verified'] is True
    assert any('diagnostic construction failed' in e for e in trainer.last_audit['audit_errors'])


def test_nonfinite_KL_retains_partial_raw_proposal_without_another_forward(tmp_path):
    from pac_act.ppo import audited_trainer,UpdateRejected
    policy=TinyPolicy();trainer=audited_trainer(policy,lambda:None,tmp_path,'id')
    calls=[];means=policy.means
    def counted(obs):calls.append(len(obs['x']));return means(obs)
    policy.means=counted
    data=batch(policy);calls.clear()
    def corrupt(*args):
        with torch.no_grad():policy.actor.bias[0]=float('nan')
        return {}
    trainer.update_minibatch=corrupt
    with pytest.raises(UpdateRejected,match='nonfinite'):trainer.train(data)
    audit=json.loads(next((tmp_path/'update_audits').glob('*.json')).read_text())
    after=audit['distribution_after']
    assert after['complete'] is False and after['blocks_evaluated']==4 and after['expected_blocks']==32
    assert after['proposed_means'][0][0][0]=={'nonfinite':'nan'}
    assert after['nonfinite_KL'] is True
    assert len(calls)==9  # Eight initial microbatches, one offending microbatch; no recovery inference.
    assert audit['rollback_verified'] is True


def test_failure_report_selects_only_the_rejection_causing_this_error():
    from types import SimpleNamespace
    from pac_act.update_audit import failure_audit_fields
    error=RuntimeError('media failure');accepted=dict(accepted=True,update=1)
    trainer=SimpleNamespace(last_audit=accepted,last_accepted_audit=accepted,_rejected_exception=None)
    fields=failure_audit_fields(trainer,error)
    assert fields['rejected_update_audit'] is None and fields['last_accepted_update_audit']==accepted
    original=RuntimeError('rejected');rejected=dict(accepted=False,attempt_id='a')
    trainer.last_audit=rejected;trainer._rejected_exception=original
    assert failure_audit_fields(trainer,error)['rejected_update_audit'] is None
    assert failure_audit_fields(trainer,original)['rejected_update_audit']==rejected
    wrapper=RuntimeError('wrapped');wrapper.__cause__=original
    assert failure_audit_fields(trainer,wrapper)['rejected_update_audit']==rejected
