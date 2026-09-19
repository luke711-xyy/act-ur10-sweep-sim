import numpy as np
import pytest
import torch

from sim.act.selection import (
    PermutationInvariantSelectionHead,
    SelectionTeacherSchedule,
    final_collection_selection_target,
    selection_bce_loss,
    straight_through_top_n,
)


def test_selection_head_is_permutation_equivariant():
    torch.manual_seed(4)
    head = PermutationInvariantSelectionHead(token_dim=29, hidden_dim=32, heads=4)
    head.eval()
    tokens = torch.randn(2, 6, 29)
    valid = torch.ones(2, 6, dtype=torch.bool)
    target = torch.tensor([2.0, 4.0])
    with torch.no_grad():
        logits = head(tokens, valid, target)
        permutation = torch.tensor([2, 5, 0, 3, 1, 4])
        permuted = head(tokens[:, permutation], valid[:, permutation], target)
    torch.testing.assert_close(permuted, logits[:, permutation])


def test_straight_through_top_n_selects_exactly_n_and_masks_invalid_slots():
    logits = torch.tensor([[0.1, 4.0, 3.0, 2.0, 9.0, 8.0]])
    valid = torch.tensor([[True, True, True, True, False, False]])
    selected = straight_through_top_n(logits, target_count=torch.tensor([2]), valid_mask=valid)
    assert selected.shape == logits.shape
    assert selected.detach().bool().tolist() == [[False, True, True, False, False, False]]
    assert torch.equal(selected[:, 4:], torch.zeros((1, 2)))


def test_straight_through_top_n_keeps_gradient_path():
    logits = torch.tensor([[0.1, 4.0, 3.0]], requires_grad=True)
    selected = straight_through_top_n(
        logits, target_count=torch.tensor([1]), valid_mask=torch.ones_like(logits, dtype=torch.bool)
    )
    selected.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_selection_bce_ignores_invalid_slots():
    logits = torch.tensor([[0.0, 0.0, 100.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    loss = selection_bce_loss(logits, targets, valid)
    assert np.isclose(float(loss), float(torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2], targets[:, :2]
    )))


def test_teacher_schedule_has_5k_transition_and_post_20k_prediction():
    schedule = SelectionTeacherSchedule(teacher_until=5000, transition_end=20000)
    assert schedule.teacher_probability(0) == 1.0
    assert schedule.teacher_probability(5000) == 1.0
    assert np.isclose(schedule.teacher_probability(12500), 0.5)
    assert schedule.teacher_probability(20000) == 0.0
    assert schedule.teacher_probability(80000) == 0.0


def test_final_selection_target_uses_actual_collected_track_ids():
    packed_ids = [17, 4, 9, 12, 22, 31]
    target = final_collection_selection_target(
        packed_track_ids=packed_ids,
        final_collected_track_ids=[12, 4],
        target_count=2,
    )
    assert target.tolist() == [False, True, False, True, False, False]
    with pytest.raises(ValueError, match="exactly target_count"):
        final_collection_selection_target(packed_ids, [4], target_count=2)
