"""Real neural computations on safe, deliberately different two-row inputs."""
import copy
import importlib.util
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from wam_sweep.actions import ActionCodec
from wam_sweep.contracts import CandidateProgram, HistoryBatch, ReferencePreview


def api():
    assert importlib.util.find_spec('wam_sweep.models'), 'Task9 neural modules are absent'
    from wam_sweep import models
    return models


def fixture(batch=2):
    torch.manual_seed(91)
    objects = torch.randn(batch, 11, 6, 34) * .03
    objects[..., 4:6] = .02
    objects[..., 6] = 0
    objects[..., 7] = 1
    brush = torch.zeros(batch, 11, 9)
    brush[..., 2] = .2
    brush[..., 4] = 1
    context = torch.zeros(batch, 11, 15)
    context[..., 2] = 1
    context[..., 11:13] = 1
    history = HistoryBatch(objects, brush, context, torch.ones(batch, 11, 6), torch.ones(batch, 11))
    references, programs, states = [], [], []
    for row in range(batch):
        q = np.tile([.1 + .03*row, .02*row, .2, .2*row], (25, 1))
        references.append(ReferencePreview(q, np.arange(25)*.04, np.ones(25), 0., f'reference-{row}'))
        programs.append(CandidateProgram(np.full((5, 3), .5 - row)))
        codec = ActionCodec(bounds=[[-1., 1.], [-1., 1.]])
        codec.decode([0., 0., .2, 0.], q[0], [.2*row, -.3*row, .1*row], .04, True)
        states.append(codec.snapshot())
    return history, references, programs, states


def single(history, row):
    return HistoryBatch(*(getattr(history, key)[row:row+1] for key in history.__dataclass_fields__))


def test_prediction_is_object_permutation_equivariant():
    h, r, c, s = fixture()
    model = api().GraphGRU().eval()
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    original = model.rollout(h, r, c, codec_state=s)
    ph = replace(h, objects=h.objects[:, :, permutation], object_valid=h.object_valid[:, :, permutation])
    permuted = model.rollout(ph, r, c, codec_state=s)
    torch.testing.assert_close(permuted.object_mean, original.object_mean[:, :, permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(permuted.brush, original.brush, rtol=1e-5, atol=1e-6)
    for key in ('object_log_std', 'state_logits', 'shape_logits', 'visibility_logits'):
        torch.testing.assert_close(getattr(permuted, key), getattr(original, key)[:, :, permutation], rtol=1e-5, atol=1e-6)


def test_proposal_has_four_modes():
    h, r, _, _ = fixture()
    actor = api().ProposalHead(api().HistoryEncoder())
    programs = actor.propose(h, torch.tensor([1, 6]), r)
    assert tuple(programs.shape) == (2, 4, 5, 3)
    assert (programs.abs() <= 1).all()
    assert not torch.equal(programs[:, 0], programs[:, 1])
    permutation = [3, 1, 5, 0, 2, 4]
    ph = replace(h, objects=h.objects[:, :, permutation], object_valid=h.object_valid[:, :, permutation])
    torch.testing.assert_close(actor.propose(ph, torch.tensor([1, 6]), r), programs, rtol=1e-5, atol=1e-6)
    assert actor.mlp[0].in_features == 480
    assert [actor.mlp[i].out_features for i in (0, 2, 4)] == [256, 256, 15]


def test_batched_conditioning_equals_independent_rows():
    h, r, c, s = fixture()
    world = api().GraphGRU().eval()
    actor = api().ProposalHead(world.encoder).eval()
    full = api().TaskQualityHead(world, allow_uncalibrated=True).eval()
    ablated = api().HistoryOnlyQualityHead(world.encoder, allow_uncalibrated=True).eval()
    pred = world.rollout(h, r, c, codec_state=s)
    scores = full.evaluate_task(h, torch.tensor([1, 6]), c, pred, reference=r, codec_state=s)
    no = ablated.evaluate_task(h, torch.tensor([1, 6]), c, reference=r, codec_state=s)
    proposed = actor.propose(h, torch.tensor([1, 6]), r)
    for row, goal in enumerate((1, 6)):
        sh = single(h, row)
        sp = world.rollout(sh, r[row], c[row], codec_state=s[row])
        torch.testing.assert_close(pred.object_mean[row:row+1], sp.object_mean, rtol=1e-5, atol=1e-6)
        ss = full.evaluate_task(sh, goal, c[row], sp, reference=r[row], codec_state=s[row])
        ns = ablated.evaluate_task(sh, goal, c[row], reference=r[row], codec_state=s[row])
        torch.testing.assert_close(scores.success_logit[row:row+1], ss.success_logit)
        torch.testing.assert_close(no.final_count_logits[row:row+1], ns.final_count_logits)
        torch.testing.assert_close(proposed[row:row+1], actor.propose(sh, goal, r[row]))
    with pytest.raises((ValueError, TypeError)):
        world.rollout(h, r[0], c[0], codec_state=s[0])
    tensor_candidates = torch.tensor(np.stack([p.knots for p in c]), dtype=torch.float32)
    torch.testing.assert_close(pred.brush, world.rollout(h, r, tensor_candidates, codec_state=s).brush)


def test_rollout_predicts_actual_response_and_has_no_goal_or_target_masks():
    h, r, c, s = fixture(1)
    model = api().GraphGRU()
    pred = model.rollout(h, r[0], c[0], codec_state=s[0])
    changed = model.rollout(h, r[0], CandidateProgram(-c[0].knots), codec_state=s[0])
    assert not torch.equal(pred.brush, changed.brush)
    assert not torch.equal(pred.object_mean, changed.object_mean)
    assert not torch.allclose(pred.brush[0, :, :3], torch.tensor(r[0].targets[:, :3], dtype=torch.float32))
    assert 'goal' not in inspect.signature(model.rollout).parameters
    assert 'supervision' not in inspect.signature(model.rollout).parameters
    assert pred.valid.tolist() == [[1.]*25]
    assert (pred.object_mean[..., 4:6] > 0).all()
    torch.testing.assert_close(pred.object_mean[..., 6:8].norm(dim=-1), torch.ones(1, 25, 6))
    torch.testing.assert_close(pred.brush[..., 3:5].norm(dim=-1), torch.ones(1, 25))
    for field in pred.__dataclass_fields__:
        assert torch.isfinite(getattr(pred, field)).all()


def test_world_gradient_crosses_25_steps_but_not_external_candidate():
    h, r, _, s = fixture(1)
    h.objects.requires_grad_()
    h.brush.requires_grad_()
    candidate = torch.full((1, 5, 3), .4, requires_grad=True)
    model = api().GraphGRU()
    pred = model.rollout(h, r[0], candidate, codec_state=s[0])
    grad = torch.autograd.grad(pred.object_mean[:, -1, :, :2].sum()+pred.brush[:, -1, :3].sum(),
                               (pred.object_mean, pred.brush, h.objects, h.brush, candidate), allow_unused=True,
                               retain_graph=True)
    assert grad[2].abs().sum() > 0 and grad[3].abs().sum() > 0
    assert grad[4] is None
    # Observe the real recursive state, rather than slices of the final stack.
    states = []
    handle = model.object_step.register_forward_hook(lambda module, args, out: (out.retain_grad(), states.append(out)) and None)
    pred = model.rollout(h, r[0], candidate, codec_state=s[0])
    pred.object_mean[:, -1, :, :2].sum().backward()
    handle.remove()
    assert len(states) == 25 and states[0].grad.abs().sum() > 0
    assert candidate.grad is None
    assert model.object_readout.weight.grad.abs().sum() > 0


def test_decoder_uses_own_tcp_fixed_anchor_and_real_mapping(monkeypatch):
    h, r, c, s = fixture(1)
    root = copy.deepcopy(s[0])
    root['current_raw'] = [.6, -.3, .2]
    root['program_anchor'] = [-.8, .8, -.8]  # begin_program must adopt current_raw once.
    calls = []
    original = ActionCodec.decode
    def record(codec, actual, nominal, raw, dt, active):
        result = original(codec, actual, nominal, raw, dt, active)
        calls.append((np.array(actual), np.array(nominal), np.array(raw), result, codec.snapshot()))
        return result
    monkeypatch.setattr(ActionCodec, 'decode', record)
    model = api().GraphGRU()
    pred = model.rollout(h, r[0], c[0], codec_state=root)
    assert len(calls) == 25 and root['frames'] == s[0]['frames']
    execution = ActionCodec(bounds=root['bounds'])
    execution.restore(root)
    anchor = execution.begin_program()
    for j, (actual, nominal, raw, decoded, after) in enumerate(calls.copy()):
        expected_tcp = h.brush[0, -1] if j == 0 else pred.brush[0, j-1]
        np.testing.assert_allclose(actual, np.r_[expected_tcp[:3].detach().numpy(),
                                    np.arctan2(*expected_tcp[3:5].detach().numpy())], atol=1e-7)
        wanted_raw = execution.sample(c[0], j*.04, anchor)
        np.testing.assert_allclose(raw, wanted_raw, atol=1e-15)
        expected = original(execution, actual, r[0].targets[j], wanted_raw, .04, True)
        np.testing.assert_array_equal(decoded.target, expected.target)
        assert after['program_anchor'] == [.6, -.3, .2]
        assert decoded.target[2] == .2
    # At t=0: .6 raw, filter=0 -> .108761548...; radius=.08.
    assert calls[0][3].filtered[0] == pytest.approx(.10876154815321091)
    assert calls[0][3].audit['correction_radius'] == pytest.approx(.08)
    np.testing.assert_allclose(calls[0][3].target,
        [.10869448873037765, -.00433757736191349, .2, .028473706722598323], atol=1e-12)
    assert 'residual_rl' not in __import__('sys').modules


def test_hold_does_not_use_future_placeholder_as_actual_command():
    h, r, c, s = fixture(1)
    valid = np.r_[np.ones(5), np.zeros(20)]
    first = replace(r[0], valid=valid)
    poisoned = first.targets.copy()
    poisoned[5:] = [100., 100., 100., 2.]
    second = replace(first, targets=poisoned)
    world = api().GraphGRU()
    a = world.rollout(h, first, c[0], codec_state=s[0])
    b = world.rollout(h, second, c[0], codec_state=s[0])
    torch.testing.assert_close(a.brush, b.brush)
    assert (a.valid == 1).all()  # hold is a physical prediction, not an invalid future.


def inactive_root_fixture(device):
    h, references, programs, states = fixture(1)
    # Identical history/goal/program/codec: ONLY the known root-held pose differs.
    h = HistoryBatch(*(getattr(h, key).repeat((2,)+(1,)*(getattr(h, key).ndim-1)).to(device)
                       for key in h.__dataclass_fields__))
    refs = []
    for pose in ([.125, -.25, .375, np.pi/2], [-.375, .5, .625, -np.pi/2]):
        targets = np.tile([.9, -.8, .2, .3], (25, 1))
        targets[0] = pose
        refs.append(replace(references[0], targets=targets, valid=np.zeros(25)))
    return h, refs, programs*2, [copy.deepcopy(states[0]) for _ in range(2)]


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_inactive_root_reference_features_preserve_only_known_pose(device):
    from wam_sweep.models.graph import references
    h, refs, _, _ = inactive_root_fixture(device)
    _, features = references(refs, 2, h.objects.device)
    expected = torch.tensor([[.125, -.25, .375, 1., 0., 0.],
                             [-.375, .5, .625, -1., 0., 0.]], device=device)
    assert features.shape == (2, 25, 6) and features.dtype == torch.float32
    torch.testing.assert_close(features[:, 0], expected, rtol=0, atol=1e-6)
    assert (features[:, 1:] == 0).all()
    assert (features[..., 5] == 0).all()  # A known hold is NOT active ACT.
    assert all(not ref.valid.any() for ref in refs)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_inactive_root_scoring_parity_without_future_placeholder_leak(device, monkeypatch):
    h, refs, programs, states = inactive_root_fixture(device)
    before = copy.deepcopy(states)
    world = api().GraphGRU().to(device)
    full = api().TaskQualityHead(world, allow_uncalibrated=True).to(device).eval()
    no = api().HistoryOnlyQualityHead(world.encoder, allow_uncalibrated=True).to(device).eval()
    goal = torch.tensor([3, 3], device=device)
    seen = {'full': [], 'no': []}
    handles = [head.control_encoder.register_forward_pre_hook(
        lambda module, args, name=name: seen[name].append(args[0].detach().clone()))
        for name, head in [('full', full), ('no', no)]]
    poisoned = []
    for ref in refs:
        targets = ref.targets.copy()
        targets[1:] = [50., -50., 30., 2.]
        poisoned.append(replace(ref, targets=targets))
    def forbidden(*args, **kwargs):
        raise AssertionError('inactive-root no-prediction scorer invoked GraphGRU')
    try:
        with torch.no_grad():
            # Full scorer uses its own frozen future, never a fabricated future.
            original = world.rollout(h, refs, programs, codec_state=states)
            full_score = full.evaluate_task(h, goal, programs, original, reference=refs, codec_state=states)
            with monkeypatch.context() as patch:
                patch.setattr(world, 'rollout', forbidden)
                no_score = no.evaluate_task(h, goal, programs, reference=refs, codec_state=states)
                # This must distinguish root-held poses with everything else fixed.
                assert (no_score.success_logit[0]-no_score.success_logit[1]).abs().max() > 1e-6
                no_poisoned = no.evaluate_task(h, goal, programs, reference=poisoned, codec_state=states)
                torch.testing.assert_close(no_score.success_logit, no_poisoned.success_logit)
                torch.testing.assert_close(no_score.final_count_logits, no_poisoned.final_count_logits)
            changed = world.rollout(h, poisoned, programs, codec_state=states)
            full_poisoned = full.evaluate_task(h, goal, programs, changed, reference=poisoned, codec_state=states)
            torch.testing.assert_close(original.object_mean, changed.object_mean)
            torch.testing.assert_close(original.brush, changed.brush)
            torch.testing.assert_close(full_score.success_logit, full_poisoned.success_logit)
            torch.testing.assert_close(full_score.final_count_logits, full_poisoned.final_count_logits)
        assert seen['full'][0].shape == seen['no'][0].shape == (2, 171)
        torch.testing.assert_close(seen['full'][0], seen['no'][0])
        expected = torch.tensor([[.125, -.25, .375, 1., 0., 0.],
                                 [-.375, .5, .625, -1., 0., 0.]], device=device)
        torch.testing.assert_close(seen['full'][0][:, 21:27], expected, rtol=0, atol=1e-6)
        assert (seen['full'][0][:, 27:] == 0).all()
        assert states == before
        print(f'INACTIVE_ROOT {device}: real full/no forward; shared171 root pose/inactive0; '
              'no world call; later-placeholder immunity; independent two-row root conditioning')
    finally:
        for handle in handles:
            handle.remove()


def test_quality_permutation_freeze_gradients_and_matched_ablation(monkeypatch):
    h, r, c, s = fixture()
    world = api().GraphGRU()
    actor = api().ProposalHead(world.encoder)
    full = api().TaskQualityHead(world, allow_uncalibrated=True)
    no = api().HistoryOnlyQualityHead(world.encoder, allow_uncalibrated=True)
    full.train(); no.train(); actor.train()
    assert not world.training and not world.encoder.training
    assert full.goal_embedding is not actor.goal_embedding and no.goal_embedding is not full.goal_embedding
    assert full.mlp[0].in_features == 592 and no.mlp[0].in_features == 464
    assert [full.mlp[i].out_features for i in (0, 2)] == [256, 128]
    assert full.control_encoder[0].in_features == no.control_encoder[0].in_features == 171
    goal = torch.tensor([2, 5])
    prediction = world.rollout(h, r, c, codec_state=s)
    a = full.evaluate_task(h, goal, c, prediction, reference=r, codec_state=s)
    permutation = [5, 4, 3, 2, 1, 0]
    ph = replace(h, objects=h.objects[:, :, permutation], object_valid=h.object_valid[:, :, permutation])
    pp = world.rollout(ph, r, c, codec_state=s)
    b = full.evaluate_task(ph, goal, c, pp, reference=r, codec_state=s)
    torch.testing.assert_close(a.success_logit, b.success_logit, rtol=1e-5, atol=1e-6)
    a.success_logit.sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in world.parameters())
    assert all(p.grad is None for p in actor.parameters())
    assert full.future_gru.weight_ih_l0.grad.abs().sum() > 0
    assert full.goal_embedding.weight.grad.abs().sum() > 0
    def forbidden(*args, **kwargs):
        raise AssertionError('no-prediction scorer called GraphGRU')
    monkeypatch.setattr(world, 'rollout', forbidden)
    base = no.evaluate_task(h, goal, c, reference=r, codec_state=s)
    torch.testing.assert_close(base.success_logit, no.evaluate_task(ph, goal, c, reference=r, codec_state=s).success_logit)
    nr = [replace(ref, targets=ref.targets + [.1, .2, 0., .4]) for ref in r]
    ns = copy.deepcopy(s); ns[0]['filtered'] = [.8, -.7, .5]
    for ref, state in ((nr,s),(r,ns)):
        changed = no.evaluate_task(h, goal, c, reference=ref, codec_state=state)
        assert not torch.equal(base.success_logit, changed.success_logit)
    base.success_logit.sum().backward()
    assert no.control_encoder[0].weight.grad.abs().sum() > 0


def test_full_quality_rejects_truth_future_and_changed_conditioning():
    h, r, c, s = fixture(1)
    world = api().GraphGRU()
    head = api().TaskQualityHead(world, allow_uncalibrated=True)
    p = world.rollout(h, r[0], c[0], codec_state=s[0])
    score = head.evaluate_task(h, 3, c[0], p, reference=r[0], codec_state=s[0])
    audit_record = {'truth_future': np.ones((25, 6, 9)), 'actual_future_controls': np.zeros((25, 4))}
    audit_record['truth_future'][:] = 1e6
    audit_record['actual_future_controls'][:] = -1e6
    torch.testing.assert_close(score.success_logit,
        head.evaluate_task(h, 3, c[0], p, reference=r[0], codec_state=s[0]).success_logit)
    with pytest.raises(ValueError, match='own|prediction|provenance'):
        head.evaluate_task(h, 3, c[0], replace(p, object_mean=p.object_mean+1), reference=r[0], codec_state=s[0])
    with pytest.raises(ValueError):
        head.evaluate_task(h, 3, CandidateProgram(-c[0].knots), p, reference=r[0], codec_state=s[0])
    torch.testing.assert_close(score.success_logit,
        head.evaluate_task(h, 3, c[0], None, reference=r[0], codec_state=s[0]).success_logit)


def test_calibration_and_mutable_safe_input_validation():
    h, r, c, s = fixture(1)
    with pytest.raises(RuntimeError, match='calibrat'):
        api().HistoryOnlyQualityHead(api().HistoryEncoder()).evaluate_task(h, 1, c[0], reference=r[0], codec_state=s[0])
    bad = copy.deepcopy(s[0]); bad['native_truth'] = [1.]*6
    with pytest.raises(ValueError):
        api().GraphGRU().rollout(h, r[0], c[0], codec_state=bad)
    h.objects[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        api().HistoryEncoder()(h)


def test_relation_encoder_uses_geometry_and_masked_history():
    h, _, _, _ = fixture(1)
    encoder = api().HistoryEncoder()
    original = encoder(h).scene
    changed = h.objects.clone(); changed[:, :, 0, 0] += .2
    moved = encoder(replace(h, objects=changed))
    assert not torch.equal(original, moved.scene)
    # A neighbor's geometry must affect an untouched object's relation encoding.
    assert not torch.equal(encoder(h).objects[:, 1], moved.objects[:, 1])
    empty = HistoryBatch(*(torch.zeros_like(getattr(h, key)) for key in h.__dataclass_fields__))
    encoded = encoder(empty)
    assert torch.isfinite(encoded.scene).all() and (encoded.scene == 0).all()


def test_quality_loss_cannot_backpropagate_into_proposal_candidates():
    h, r, _, s = fixture(1)
    world = api().GraphGRU()
    actor = api().ProposalHead(world.encoder)
    full = api().TaskQualityHead(world, allow_uncalibrated=True)
    no = api().HistoryOnlyQualityHead(world.encoder, allow_uncalibrated=True)
    proposed = actor.propose(h, 2, r[0])[:, 0]
    pred = world.rollout(h, r[0], proposed, codec_state=s[0])
    a = full.evaluate_task(h, 2, proposed, pred, reference=r[0], codec_state=s[0])
    b = no.evaluate_task(h, 2, proposed, reference=r[0], codec_state=s[0])
    (a.success_logit.sum()+b.success_logit.sum()).backward()
    assert all(p.grad is None for p in actor.parameters())
    assert full.success.weight.grad.abs().sum() > 0 and no.success.weight.grad.abs().sum() > 0


def test_inference_mode_supports_real_rollout_and_both_scorers():
    h, r, c, s = fixture(1)
    world = api().GraphGRU()
    full = api().TaskQualityHead(world, allow_uncalibrated=True)
    no = api().HistoryOnlyQualityHead(world.encoder, allow_uncalibrated=True)
    with torch.inference_mode():
        pred = world.rollout(h, r[0], c[0], codec_state=s[0])
        score = full.evaluate_task(h, 1, c[0], pred, reference=r[0], codec_state=s[0])
        ablated = no.evaluate_task(h, 1, c[0], reference=r[0], codec_state=s[0])
    assert torch.isfinite(score.final_count_logits).all() and torch.isfinite(ablated.success_logit).all()


def test_prediction_provenance_rejects_mutated_fields_and_changed_weights():
    h, r, c, s = fixture(1)
    world = api().GraphGRU()
    head = api().TaskQualityHead(world, allow_uncalibrated=True)
    pred = world.rollout(h, r[0], c[0], codec_state=s[0])
    with torch.no_grad():
        pred.brush.add_(1)
    with pytest.raises(ValueError):
        head.evaluate_task(h, 1, c[0], pred, reference=r[0], codec_state=s[0])
    pred = world.rollout(h, r[0], c[0], codec_state=s[0])
    with torch.no_grad():
        world.object_readout.weight.add_(.1)
    with pytest.raises(ValueError):
        head.evaluate_task(h, 1, c[0], pred, reference=r[0], codec_state=s[0])


def test_safe_dataset_row_poisoning_changes_only_terminal_supervision():
    from wam_sweep.data.collection import SafeInputs
    h,r,c,s = fixture(1)
    row = dict(inputs=SafeInputs(h,r[0],s[0]),goal_n=3,program=c[0],
               supervision=dict(success=True,final_count=3,count_valid=True),
               audit=dict(truth_future=np.ones((25,6,9)),actual_future_controls=np.zeros((25,4))))
    world = api().GraphGRU()
    full = api().TaskQualityHead(world,allow_uncalibrated=True)
    def evaluate():
        known = row['inputs']
        return full.evaluate_task(known.history,row['goal_n'],row['program'],None,
                                  reference=known.reference,codec_state=known.codec_state)
    a = evaluate()
    first = api().terminal_loss(a,torch.tensor([[1.]]),torch.tensor([3]),torch.tensor([True]))
    row['audit']['truth_future'][:] = 1e10
    row['audit']['actual_future_controls'][:] = -1e10
    row['supervision'].update(success=False,final_count=0)
    b = evaluate()
    torch.testing.assert_close(a.success_logit,b.success_logit)
    torch.testing.assert_close(a.final_count_logits,b.final_count_logits)
    second = api().terminal_loss(b,torch.tensor([[0.]]),torch.tensor([0]),torch.tensor([True]))
    assert first.item() != second.item()


def test_real_success_set_loss_trains_only_proposal_owned_parameters():
    h,r,_,_ = fixture(1)
    encoder = api().HistoryEncoder()
    actor = api().ProposalHead(encoder)
    programs = actor.propose(h,3,r[0])
    targets = torch.zeros(1,2,5,3); targets[:,0,:,1] = -.8; targets[:,1,:,1] = .8
    api().proposal_loss(programs,targets,torch.ones(1,2,dtype=torch.bool)).backward()
    for module in (actor.mlp,actor.reference_gru,actor.goal_embedding,actor.mode_embedding):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    world = api().GraphGRU()
    full = api().TaskQualityHead(world,allow_uncalibrated=True)
    no = api().HistoryOnlyQualityHead(world.encoder,allow_uncalibrated=True)
    print('PARAMETERS', {name:dict(total=sum(p.numel() for p in module.parameters()),
          trainable=sum(p.numel() for p in module.parameters() if p.requires_grad)) for name,module in
          [('history',encoder),('proposal',actor),('frozen_world',world),('full_quality',full),('history_only_quality',no)]})
    print('ARCHITECTURE', dict(graph_layers=len(world.graph.messages),hidden=world.object_step.hidden_size,
          future_gru=full.future_gru.hidden_size,reference_gru=actor.reference_gru.hidden_size,
          scene=encoder(h).scene.shape[-1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_mps_real_modules_losses_and_gradient_boundaries():
    from wam_sweep.data.datasets import AlignedTargets
    h,r,_,s = fixture(2)
    h = HistoryBatch(*(getattr(h,key).to('mps') for key in h.__dataclass_fields__))
    world = api().GraphGRU().to('mps')
    candidate = torch.full((2,5,3),.3,device='mps',requires_grad=True)
    pred = world.rollout(h,r,candidate,codec_state=s)
    def target(row):
        mask = np.ones((25,6),bool)
        mean = pred.object_mean[row].detach().cpu().numpy().copy(); mean[...,0] += .01
        return AlignedTargets(mean,mask,mask,mask,mask,np.zeros((25,6),np.int64),mask,
                              np.ones((25,6),np.float32),mask,mask,np.ones(25,bool),np.ones(6,bool))
    loss = api().prediction_loss(pred,[target(0),target(1)],pred.brush.detach(),torch.ones(2,25,device='mps',dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and world.object_readout.weight.grad.abs().sum()>0 and candidate.grad is None
    world.zero_grad(set_to_none=True)
    actor = api().ProposalHead(world.encoder).to('mps')
    goal = torch.tensor([2,5],device='mps')
    proposed = actor.propose(h,goal,r)
    assert proposed.shape == (2,4,5,3)
    full = api().TaskQualityHead(world,allow_uncalibrated=True).to('mps')
    no = api().HistoryOnlyQualityHead(world.encoder,allow_uncalibrated=True).to('mps')
    with torch.no_grad():
        prediction = world.rollout(h,r,proposed[:,0],codec_state=s)
    a = full.evaluate_task(h,goal,proposed[:,0],prediction,reference=r,codec_state=s)
    b = no.evaluate_task(h,goal,proposed[:,0],reference=r,codec_state=s)
    labels = torch.ones(2,1,device='mps')
    counts = torch.tensor([2,5],device='mps'); mask = torch.ones(2,device='mps',dtype=torch.bool)
    (api().terminal_loss(a,labels,counts,mask)+api().terminal_loss(b,labels,counts,mask)).backward()
    assert all(p.grad is None for p in actor.parameters())
    assert full.success.weight.grad.abs().sum()>0 and no.success.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in world.parameters())
