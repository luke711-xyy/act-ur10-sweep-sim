"""Approved stability regressions: CPU arithmetic/gradients, never ACT or physics."""
import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from residual_rl.learning import Episode, EpisodicReplay, RecurrentSAC
from residual_rl.pilot import UpdateSchedule


@pytest.fixture(scope='module', autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_episode(goal, serial=0, steps=420, prefix=3):
    rng = np.random.default_rng(goal * 100 + serial)
    return Episode(f'synthetic-{goal}-{serial}', goal,
        rng.normal(0, .1, (steps + 1, 4)).astype(np.float32),
        rng.normal(0, .1, (steps + 1, 7)).astype(np.float32),
        rng.normal(0, .018, (steps, 3)).astype(np.float32),
        np.zeros(steps, np.float32), np.array([False] * (steps - 1) + [True]),
        np.arange(steps) >= prefix)


def filled_replay(steps=420, rounds=2):
    store = EpisodicReplay(seed=1)
    for serial in range(rounds):
        for goal in range(1, 7):
            store.add(make_episode(goal, serial, steps))
    return store


def stability_api(name):
    try:
        module = importlib.import_module('residual_rl.policy_guard')
    except ModuleNotFoundError:
        module = None
    result = getattr(module, name, None)
    assert callable(result), f'production {name} must exist'
    return result


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_tree(left, right)
    else:
        assert a == b


def test_no_203_update_debt_when_sixth_goal_first_becomes_ready():
    schedule = UpdateSchedule()
    begin = getattr(schedule, 'begin_episode', None)
    assert callable(begin), 'schedule must delimit fresh-data episode permissions'
    for index, steps in enumerate([535, 417, 307, 550, 1458, 1831]):
        begin(steps, ready=index == 5)
        assert not schedule.due()
    begin(250, ready=True)
    for _ in range(10):
        assert schedule.due()
        schedule.completed()
    assert not schedule.due()


def test_episode_burst_cap_discards_integer_debt_but_keeps_only_small_remainder():
    schedule = UpdateSchedule()
    assert callable(getattr(schedule, 'begin_episode', None))
    schedule.begin_episode(5000, ready=True)  # readiness boundary is not backpay
    schedule.begin_episode(2001, ready=True)
    for _ in range(10):
        schedule.completed()
    assert not schedule.due()
    schedule.begin_episode(0, ready=True)
    assert not schedule.due()
    schedule.begin_episode(23, ready=True)
    assert not schedule.due()
    schedule.begin_episode(1, ready=True)
    assert schedule.due()
    schedule.completed()
    assert not schedule.due()
    with pytest.raises(RuntimeError):
        schedule.completed()


def test_training_readiness_needs_two_sample_bearing_episodes_each_goal_and_5000_steps():
    store = filled_replay(steps=400)
    ready = getattr(store, 'training_ready', None)
    assert callable(ready)
    assert not ready()  # 4764, not 5000 learnable steps
    store.add(make_episode(1, 9, 300))
    assert ready()
    single = filled_replay(steps=1000, rounds=1)
    assert not single.training_ready()
    poor_goal = EpisodicReplay()
    for serial in range(2):
        for goal in range(1, 7):
            poor_goal.add(make_episode(goal, serial, 600 if goal < 6 else 64))
    assert not poor_goal.training_ready()  # enough total, <256 for goal6


def test_first_100_critic_updates_leave_actor_and_adam_exactly_unchanged():
    store, model = filled_replay(steps=32), RecurrentSAC(4, 7, seed=2)
    actor = copy.deepcopy(model.actor.state_dict())
    adam = copy.deepcopy(model.actor_optimizer.state_dict())
    for index in range(100):
        result = model.update(store)
        assert result['updates'] == index + 1
        assert result.get('actor_updated') is False
    equal_tree(actor, model.actor.state_dict())
    equal_tree(adam, model.actor_optimizer.state_dict())
    assert model.actor_updates == 0


def test_fixed_probes_are_balanced_and_use_real_prefix_not_future_or_padding():
    bank = stability_api('ProbeBank').from_episodes(filled_replay().episodes)
    counts = {goal: 0 for goal in range(1, 7)}
    for history in bank.histories:
        counts[history.goal] += len(history.indices)
        assert history.indices.min() >= 3
        assert len(history.observations) == history.indices.max() + 1
    assert counts == {goal: 64 for goal in range(1, 7)}
    restored = stability_api('ProbeBank').from_spec(bank.spec(), filled_replay().episodes)
    equal_tree(bank.spec(), restored.spec())


def test_guard_rejects_saturated_candidate_and_restores_nonempty_adam_state():
    Guard, Rejected = stability_api('PolicyUpdateGuard'), stability_api('PolicyUpdateRejected')
    bank = stability_api('ProbeBank').from_episodes(filled_replay(steps=68).episodes)
    model = RecurrentSAC(4, 7, seed=2)
    # Seed genuine Adam moments without a rollout or learner training loop.
    loss = sum(p.square().sum() for p in model.actor.parameters())
    model.actor_optimizer.zero_grad(); loss.backward(); model.actor_optimizer.step()
    before_actor = copy.deepcopy(model.actor.state_dict())
    before_adam = copy.deepcopy(model.actor_optimizer.state_dict())
    guard = Guard(model.actor, model.actor_optimizer, bank.histories)
    with torch.no_grad():
        model.actor.mean.bias.fill_(5.)
        for state in model.actor_optimizer.state.values():
            state['exp_avg'].add_(1.)
    with pytest.raises(Rejected):
        guard.check()
    equal_tree(before_actor, model.actor.state_dict())
    equal_tree(before_adam, model.actor_optimizer.state_dict())


def test_guard_catches_stochastic_saturation_even_with_zero_deterministic_mean():
    Guard, Rejected = stability_api('PolicyUpdateGuard'), stability_api('PolicyUpdateRejected')
    bank = stability_api('ProbeBank').from_episodes(filled_replay(steps=68).episodes)
    model = RecurrentSAC(4, 7)
    guard = Guard(model.actor, model.actor_optimizer, bank.histories)
    with torch.no_grad():
        model.actor.log_std.bias.fill_(0.)
    with pytest.raises(Rejected) as caught:
        guard.check()
    assert caught.value.report['sampled_saturation_fraction'] > .01
    assert caught.value.report['deterministic_saturation_fraction'] == 0.


def test_guard_limits_entire_round_instead_of_reanchoring_each_actor_step():
    Guard, Rejected = stability_api('PolicyUpdateGuard'), stability_api('PolicyUpdateRejected')
    bank = stability_api('ProbeBank').from_episodes(filled_replay(steps=68).episodes)
    model = RecurrentSAC(4, 7)
    guard = Guard(model.actor, model.actor_optimizer, bank.histories)
    with torch.no_grad():
        model.actor.mean.bias[0] = .002
    assert guard.check()['accepted']
    with torch.no_grad():
        model.actor.mean.bias[0] = .004
    with pytest.raises(Rejected):
        guard.check()  # KL against zero is .02385, not against previous .002
    assert model.actor.mean.bias[0].item() == 0.


@pytest.mark.parametrize('seed', [3, 19, 41])
def test_200_real_synthetic_updates_do_not_rapidly_saturate_after_learning(seed, record_property):
    # Real GRUs, critics, losses and Adam; all observations are synthetic CPU data.
    store, model = filled_replay(steps=68), RecurrentSAC(4, 7, seed=seed)
    assert callable(getattr(model, 'update_round', None))
    reports = []
    for _ in range(20):
        with model.update_round(store) as guard:
            actor_count = 0
            for _ in range(10):
                metric = model.update(store)
                actor_count += int(metric['actor_updated'])
                assert np.isfinite(metric['critic_loss'])
            assert actor_count <= 2
        reports.append(guard.last_report)
    assert model.updates == 200 and model.actor_updates == 20
    assert not model.guard_failed
    for report in reports:
        assert report['accepted']
        assert report['mean_kl'] <= .01
        assert report['max_xy_change_m'] is None  # Synthetic obs have no physical ACT displacement.
        assert report['max_xy_change_fraction'] <= .8/32
        assert report['max_yaw_change_rad'] <= (np.pi/60)/28
        assert report['deterministic_saturation_fraction'] <= .01
        assert report['sampled_saturation_fraction'] <= .01
    for key in ['mean_kl', 'max_xy_change_fraction', 'max_yaw_change_rad',
                'deterministic_saturation_fraction', 'sampled_saturation_fraction']:
        record_property(key, max(report[key] for report in reports))
    record_property('critic_updates', model.updates)
    record_property('actor_updates', model.actor_updates)


def test_real_second_actor_update_rejection_rolls_back_whole_round_and_latches_stop(monkeypatch):
    model, store = RecurrentSAC(4, 7, seed=19), filled_replay(steps=68)
    assert callable(getattr(model, 'update_round', None))
    for _ in range(100):
        model.update(store)
    with model.update_round(store):
        for _ in range(5):
            model.update(store)
    assert model.updates == 105 and model.actor_updates == 1
    before = model.checkpoint()
    original_step = model.actor_optimizer.step
    calls = 0

    def overshoot_on_second(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 2:
            with torch.no_grad():
                model.actor.mean.bias.fill_(5.)
        return result

    monkeypatch.setattr(model.actor_optimizer, 'step', overshoot_on_second)
    with pytest.raises(stability_api('PolicyUpdateRejected')):
        with model.update_round(store):
            for _ in range(10):
                model.update(store)
    assert calls == 2
    equal_tree(before['actor'], model.actor.state_dict())
    equal_tree(before['actor_optimizer'], model.actor_optimizer.state_dict())
    assert model.actor_updates == 1 and model.guard_failed
    assert torch.equal(before['rng_state'], model.checkpoint()['rng_state'])
    with pytest.raises(RuntimeError):
        model.update(store)
    with pytest.raises(RuntimeError):
        model.act(np.zeros(4, np.float32))


def test_production_learning_round_uses_valid_mask_and_strong_readiness_only():
    from residual_rl import pilot
    run = getattr(pilot, 'learn_after_episode', None)
    assert callable(run), 'pilot must use bounded transactional update rounds'
    store, model, schedule = filled_replay(), RecurrentSAC(4, 7), UpdateSchedule()
    # First entry crosses readiness, but receives no backdated updates.
    first = make_episode(1, 40)
    assert run(model, store, schedule, first, lambda: None) == []
    assert model.updates == 0
    second = make_episode(2, 40, steps=2000, prefix=1800)
    metrics = run(model, store, schedule, second, lambda: None)
    assert len(metrics) == 8  # 200 actually valid, not 2000 context steps
    assert model.updates == 8 and model.actor_updates == 0
    assert not schedule.due()


def test_zero_residual_prior_penalizes_physical_normalized_request_not_environment_rewards():
    prior = stability_api('zero_residual_prior')
    actions = torch.tensor([[0., 0., 0.], [1., 1., 0.], [.3, .4, .5]], requires_grad=True)
    values = prior(actions)
    torch.testing.assert_close(values, torch.tensor([0., 1., .4856]))
    values.sum().backward()
    assert actions.grad[2, 0] > 0 and actions.grad[2, 2] > 0


def test_probe_distribution_rebuilds_entire_prefix_instead_of_splicing_selected_states():
    History = stability_api('ProbeHistory')
    model = RecurrentSAC(4, 7)
    with torch.no_grad():
        model.actor.mean.weight.fill_(.03)
    episode = make_episode(1, steps=68)
    indices = np.array([10, 50, 67])
    history = History.from_episode(episode, indices)
    guard = stability_api('PolicyUpdateGuard')(model.actor, model.actor_optimizer, [history])
    hidden = None; expected = []
    for index, observation in enumerate(episode.actor_obs[:68]):
        action, hidden = model.act(observation, hidden, deterministic=True)
        if index in indices:
            expected.append(action)
    np.testing.assert_allclose(guard.old_mean.tanh().numpy(), np.asarray(expected), atol=2e-7)


def test_nonfinite_candidate_rejects_and_clears_no_global_rng_or_actor_state():
    model, store = RecurrentSAC(4, 7), filled_replay(steps=68)
    bank = stability_api('ProbeBank').from_episodes(store.episodes)
    global_rng = torch.random.get_rng_state().clone()
    before = copy.deepcopy(model.actor.state_dict())
    guard = stability_api('PolicyUpdateGuard')(model.actor, model.actor_optimizer, bank.histories)
    with torch.no_grad():
        model.actor.mean.bias[0] = float('nan')
    with pytest.raises(stability_api('PolicyUpdateRejected')) as caught:
        guard.check()
    assert 'nonfinite_policy' in caught.value.report['violations']
    equal_tree(before, model.actor.state_dict())
    assert torch.equal(global_rng, torch.random.get_rng_state())


def test_checkpoint_binds_stability_contract_not_only_learning_rate():
    model = RecurrentSAC(4, 7)
    payload = model.checkpoint()
    assert 'stability_contract' in payload
    before = model.checkpoint()
    payload['stability_contract']['max_mean_kl'] = 99.
    with pytest.raises(ValueError):
        model.load_checkpoint(payload)
    equal_tree(before, model.checkpoint())


def test_collection_budget_interrupt_rolls_back_but_keeps_final_validation_policy_usable():
    from residual_rl.safety import BudgetStop
    model, store = RecurrentSAC(4, 7, seed=19), filled_replay(steps=68)
    for _ in range(100):
        model.update(store)
    with model.update_round(store):
        for _ in range(5):
            model.update(store)
    before = model.checkpoint()
    with pytest.raises(BudgetStop):
        with model.update_round(store):
            for _ in range(5):
                model.update(store)
            raise BudgetStop('stop_collection')
    equal_tree(before['actor'], model.actor.state_dict())
    equal_tree(before['actor_optimizer'], model.actor_optimizer.state_dict())
    assert model.actor_updates == 1
    assert not model.guard_failed  # normal collection cutoff is not policy failure
    restored = RecurrentSAC(4, 7)
    restored.load_checkpoint(model.checkpoint())
    assert np.isfinite(restored.act(np.zeros(4, np.float32), deterministic=True)[0]).all()


def test_administrative_nonterminal_episodes_never_qualify_as_complete_warmup():
    store = EpisodicReplay()
    for serial in range(2):
        for goal in range(1, 7):
            episode = make_episode(goal, serial)
            store.add(replace(episode, terminated=np.zeros(len(episode.rewards), bool)))
    assert not store.training_ready()


def test_administrative_partial_cannot_earn_fresh_update_permissions():
    from residual_rl.pilot import learn_after_episode
    store, model, schedule = filled_replay(), RecurrentSAC(4, 7), UpdateSchedule()
    learn_after_episode(model, store, schedule, make_episode(1, 80), lambda: None)
    partial = make_episode(2, 80)
    partial = replace(partial, terminated=np.zeros(len(partial.rewards), bool))
    assert learn_after_episode(model, store, schedule, partial, lambda: None) == []
    assert model.updates == 0 and not schedule.due()


def test_probe_setup_failure_latches_stop_and_prevents_policy_actions():
    store, model = filled_replay(steps=68), RecurrentSAC(4, 7)
    bank = stability_api('ProbeBank').from_episodes(store.episodes)
    model._probe_spec = bank.spec()
    different = EpisodicReplay()
    for goal in range(1, 7):
        different.add(make_episode(goal, serial=99, steps=68))
    before = model.checkpoint()
    with pytest.raises((KeyError, ValueError)):
        with model.update_round(different):
            pytest.fail('unresolved probe identities cannot enter an update round')
    assert model.guard_failed
    equal_tree(before['actor'], model.actor.state_dict())
    equal_tree(before['actor_optimizer'], model.actor_optimizer.state_dict())
    with pytest.raises(RuntimeError):
        model.act(np.zeros(4, np.float32))
    with pytest.raises(RuntimeError):
        model.update(store)


def test_failed_checkpoint_cannot_hide_impossible_rollback_counts():
    model = RecurrentSAC(4, 7)
    payload = model.checkpoint()
    payload['guard_failed'] = True
    payload['rolled_back_actor_updates'] = 999
    with pytest.raises(ValueError):
        model.load_checkpoint(payload)


@pytest.mark.parametrize('criterion', ['xy', 'yaw', 'variance'])
def test_each_policy_guard_criterion_can_reject_independently_of_kl(criterion):
    model = RecurrentSAC(4, 7)
    with torch.no_grad():
        model.actor.log_std.bias.fill_(-1.)
    bank = stability_api('ProbeBank').from_episodes(filled_replay(steps=68).episodes)
    guard = stability_api('PolicyUpdateGuard')(model.actor, model.actor_optimizer, bank.histories)
    assert guard.check()['accepted']
    with torch.no_grad():
        if criterion == 'variance':
            model.actor.log_std.bias.fill_(-.95)
        else:
            model.actor.mean.bias[0 if criterion == 'xy' else 2] = .04
    with pytest.raises(stability_api('PolicyUpdateRejected')) as caught:
        guard.check()
    expected = {'xy': 'max_xy_change_fraction', 'yaw': 'max_yaw_change_rad',
                'variance': 'sampled_saturation_fraction'}[criterion]
    assert caught.value.report['mean_kl'] < .01
    assert caught.value.report['violations'] == [expected]


def test_actor_optimizer_exception_keeps_completed_critic_count_and_checkpoint_consistent(monkeypatch):
    model, store = RecurrentSAC(4, 7, seed=19), filled_replay(steps=68)
    for _ in range(104):
        model.update(store)
    before = model.checkpoint()

    def broken_actor_step(*args, **kwargs):
        raise RuntimeError('injected Actor optimizer failure')

    monkeypatch.setattr(model.actor_optimizer, 'step', broken_actor_step)
    with pytest.raises(RuntimeError, match='injected Actor optimizer'):
        model.update(store)
    assert model.updates == 105
    assert model.actor_updates == 0 and model.guard_failed
    assert all(state['step'].item() == 105 for state in model.critic_optimizer.state.values())
    for old, target, current in zip(before['target_critics'], model.target_critics, model.critics):
        for key, actual in target.state_dict().items():
            torch.testing.assert_close(actual, old[key]*.995 + current.state_dict()[key]*.005)
    equal_tree(before['actor'], model.actor.state_dict())
    equal_tree(before['actor_optimizer'], model.actor_optimizer.state_dict())
    restored = RecurrentSAC(4, 7)
    restored.load_checkpoint(model.checkpoint())
    assert restored.updates == 105 and restored.guard_failed
