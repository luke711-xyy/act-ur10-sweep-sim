"""Masked physical likelihoods, successful-set matching and terminal labels."""
from dataclasses import replace
import importlib.util

import numpy as np
import pytest
import torch

from wam_sweep.contracts import WorldPrediction, TaskScore
from wam_sweep.data.datasets import AlignedTargets


def api():
    assert importlib.util.find_spec('wam_sweep.models'), 'Task9 neural losses are absent'
    from wam_sweep.models import losses
    return losses


def physical():
    mean = torch.zeros(1, 25, 6, 9)
    mean[..., 4:6] = .02; mean[..., 7] = 1
    brush = torch.zeros(1, 25, 9); brush[..., 4] = 1
    pred = WorldPrediction(mean.requires_grad_(), torch.zeros(1,25,6,4,requires_grad=True),
        torch.zeros(1,25,6,3,requires_grad=True), torch.zeros(1,25,6,2,requires_grad=True),
        torch.zeros(1,25,6,1,requires_grad=True), brush.requires_grad_(),
        torch.zeros(1,25,9,requires_grad=True), torch.ones(1,25))
    mask = np.ones((25,6), bool)
    target = AlignedTargets(mean.detach()[0].numpy().copy(), mask.copy(), mask.copy(), mask.copy(),
        mask.copy(), np.zeros((25,6), np.int64), mask.copy(), np.ones((25,6),np.float32),
        mask.copy(), mask.copy(), np.ones(25,bool), np.ones(6,bool))
    return pred, target, brush.detach()[0].numpy().copy(), np.ones(25,bool)


def test_physical_loss_is_real_likelihood_with_gradients():
    p, t, b, bv = physical()
    t.object_mean[..., :4] = .3
    loss = api().prediction_loss(p, t, b, bv)
    loss.backward()
    assert torch.isfinite(loss)
    for value in (p.object_mean, p.object_log_std, p.state_logits, p.shape_logits, p.visibility_logits,
                  p.brush_log_std):
        assert value.grad is not None and value.grad.abs().sum() > 0


@pytest.mark.parametrize('mask', ['mapping_valid', 'valid_steps', 'object_valid'])
def test_mapping_steps_and_field_masks_hide_unknown_targets(mask):
    p, t, b, bv = physical()
    getattr(t, mask)[:] = False
    base = api().prediction_loss(p, t, b, bv)
    t.object_mean[:] = np.nan; t.state[:] = -99; t.visibility[:] = np.nan
    torch.testing.assert_close(api().prediction_loss(p, t, b, bv), base)


def test_per_field_masks_circle_yaw_and_terminal_padding():
    p, t, b, bv = physical()
    t.velocity_valid[:] = False; t.yaw_valid[:] = False; t.angular_valid[:] = False
    t.state_valid[:] = False; t.visibility_valid[:] = False; t.rectangle[:] = False
    t.valid_steps[10:] = False; bv[10:] = False
    base = api().prediction_loss(p,t,b,bv)
    t.object_mean[..., 2:4] = np.nan; t.object_mean[..., 6:9] = np.nan
    t.state[:] = -999; t.visibility[:] = np.nan
    t.object_mean[10:] = np.nan; b[10:] = np.nan
    torch.testing.assert_close(api().prediction_loss(p,t,b,bv), base)
    # Rectangle is a shape target: all circles still train shape logits.
    loss = api().prediction_loss(p,t,b,bv); loss.backward()
    assert p.shape_logits.grad.abs().sum() > 0


def test_periodic_yaw_losses_distinguish_half_turn_for_brush():
    p,t,b,bv = physical()
    base = api().prediction_loss(p,t,b,bv)
    # Rectangle yaw+pi is the SAME double-angle vector; brush+pi flips its vector.
    t.object_mean[..., 6] = np.sin(2*np.pi); t.object_mean[..., 7] = np.cos(2*np.pi)
    torch.testing.assert_close(api().prediction_loss(p,t,b,bv), base)
    b[..., 4] = -1
    assert api().prediction_loss(p,t,b,bv) > base


def test_all_unknown_losses_are_zero_and_batched_targets_are_independent():
    p,t,b,bv = physical()
    t.valid_steps[:] = False; bv[:] = False
    loss = api().prediction_loss(p,t,b,bv)
    assert loss.item() == 0
    loss.backward()
    assert p.object_mean.grad.abs().sum() == 0
    p,t,b,bv = physical()
    batched = WorldPrediction(*(getattr(p,k).repeat((2,)+(1,)*(getattr(p,k).ndim-1)) for k in p.__dataclass_fields__))
    t2 = replace(t, object_mean=t.object_mean.copy()); t2.object_mean[...,0] += 1
    expected = (api().prediction_loss(p,t,b,bv)+api().prediction_loss(p,t2,b,bv))/2
    torch.testing.assert_close(api().prediction_loss(batched,[t,t2],np.stack([b,b]),np.stack([bv,bv])), expected)
    with pytest.raises((ValueError,TypeError)):
        api().prediction_loss(batched,t,np.stack([b,b]),np.stack([bv,bv]))


def test_success_set_matching_preserves_left_and_right_instead_of_average():
    programs = torch.zeros(1,4,5,3,requires_grad=True)
    with torch.no_grad():
        programs[:,0,:,1] = -1; programs[:,1,:,1] = 1
    targets = torch.zeros(1,2,5,3); targets[:,0,:,1] = -1; targets[:,1,:,1] = 1
    mask = torch.ones(1,2,dtype=torch.bool)
    loss = api().proposal_loss(programs,targets,mask)
    assert loss.item() == 0
    average = torch.zeros_like(programs,requires_grad=True)
    assert api().proposal_loss(average,targets,mask).item() == pytest.approx(1/3)
    torch.testing.assert_close(loss,api().proposal_loss(programs,targets.flip(1),mask))
    loss.backward()
    assert programs.grad[:,2:].abs().sum() == 0


def test_matching_cannot_assign_one_mode_to_two_successes_and_masks_all_fail():
    programs = torch.ones(2,4,5,3,requires_grad=True)
    with torch.no_grad():
        programs[0,0] = 0
    targets = torch.zeros(2,3,5,3)
    mask = torch.tensor([[True,True,False],[False,False,False]])
    targets[~mask] = float('nan')
    loss = api().proposal_loss(programs,targets,mask)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert programs.grad[1].abs().sum() == 0
    assert (programs.grad[0].abs().sum(dim=(1,2)) > 0).sum() == 1
    empty = api().proposal_loss(programs,targets,torch.zeros_like(mask))
    assert empty.item() == 0


def test_terminal_loss_has_independent_unknown_count_mask_and_label_gradients():
    score = TaskScore(torch.zeros(2,1,requires_grad=True),torch.zeros(2,7,requires_grad=True),'uncalibrated-fixture')
    loss = api().terminal_loss(score,torch.tensor([[1.],[0.]]),torch.tensor([6,-999]),torch.tensor([True,False]))
    loss.backward()
    assert score.success_logit.grad[:,0].tolist() == [-.25,.25]
    assert score.final_count_logits.grad[0,6] < 0 and score.final_count_logits.grad[1].abs().sum() == 0
    changed = api().terminal_loss(score,torch.tensor([[0.],[0.]]),torch.tensor([6,100]),torch.tensor([True,False]))
    assert torch.isfinite(changed)
    changed_grad = torch.autograd.grad(changed,score.success_logit)[0]
    assert changed_grad[0] > 0
