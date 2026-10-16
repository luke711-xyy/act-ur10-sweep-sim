"""Small CPU-only learner contracts; never load ACT or a physics runtime."""

import copy
import importlib
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    try:
        return importlib.import_module("residual_rl.learning")
    except ModuleNotFoundError as exc:
        if exc.name not in {"residual_rl", "residual_rl.learning"}:
            raise
        return SimpleNamespace()


def construct(name, *args, **kwargs):
    factory = getattr(api(), name, None)
    assert callable(factory), f"residual_rl.learning.{name} is not implemented"
    return factory(*args, **kwargs)


def episode(goal=1, steps=5, episode_id=None, mask=None, terminal=False):
    rng = np.random.default_rng(goal)
    return construct(
        "Episode",
        episode_id=episode_id or f"goal-{goal}",
        target_count=goal,
        actor_obs=rng.normal(size=(steps + 1, 4)).astype(np.float32),
        critic_obs=rng.normal(size=(steps + 1, 7)).astype(np.float32),
        actions=rng.uniform(-0.8, 0.8, (steps, 3)).astype(np.float32),
        rewards=np.arange(1, steps + 1, dtype=np.float32),
        terminated=np.array([False] * (steps - 1) + [terminal], dtype=bool),
        learn_mask=np.ones(steps, dtype=bool) if mask is None else np.array(mask, bool),
    )


def replay(steps=44, mask=None):
    result = construct("EpisodicReplay", seed=9)
    for goal in range(1, 7):
        result.add(episode(goal, steps, mask=mask))
    return result


def signature(samples):
    return [(s.episode.episode_id, s.start, s.length) for s in samples]


def test_replay_warmup_counts_only_learned_steps_and_requires_every_goal():
    store = construct("EpisodicReplay")
    for goal in range(1, 6):
        store.add(episode(goal, 60))
    assert not store.ready()
    store.add(episode(6, 60, mask=[False] * 60))
    assert not store.ready()
    store.add(episode(6, 4, episode_id="six-learnable"))
    assert store.ready() and not store.ready(min_steps=305)
    assert not replay(steps=42).ready()


def test_replay_evicts_whole_episodes_and_rejects_duplicate_and_oversize():
    store = construct("EpisodicReplay", capacity=8)
    first, second, third = episode(1, 3), episode(2, 3), episode(3, 3)
    store.add(first)
    store.add(second)
    store.add(third)
    assert store.steps == 6
    assert [e.episode_id for e in store.episodes] == ["goal-2", "goal-3"]
    with pytest.raises(ValueError):
        store.add(first)
    with pytest.raises(ValueError):
        store.add(episode(4, 9))
    assert store.steps == 6


def test_balanced_rotation_and_sparse_windows_never_cross_or_alias_episodes():
    store = replay(steps=7, mask=[False] * 6 + [True])
    first, second, third = [store.sample(batch_size=4, length=3) for _ in range(3)]
    samples = first + second + third
    assert [s.episode.target_count for s in samples] == list(range(1, 7)) * 2
    for sample in samples:
        assert 4 <= sample.start <= 6
        assert sample.length == 3
        assert sample.episode.learn_mask[sample.start:sample.start + sample.length].any()
        assert not np.shares_memory(sample.episode.actions, store.episodes[0].actions)
    assert not np.shares_memory(samples[0].episode.actions, samples[6].episode.actions)
    original_actions = store.state_dict()["episodes"][0]["actions"]
    samples[0].episode.actions.setflags(write=True)
    samples[0].episode.actions[:] = 0
    np.testing.assert_array_equal(store.state_dict()["episodes"][0]["actions"], original_actions)
    with pytest.raises(ValueError):
        construct("EpisodicReplay").sample()


def test_replay_state_restores_rng_rotation_and_owns_independent_data():
    original = replay(steps=7)
    original.sample()
    state = original.state_dict()
    restored = construct("EpisodicReplay", seed=999)
    restored.load_state_dict(state)
    assert signature(original.sample(12, 4)) == signature(restored.sample(12, 4))
    state["episodes"][0]["rewards"][0] = 999
    assert restored.episodes[0].rewards[0] == 1
    assert original.episodes[0].rewards[0] == 1


@pytest.mark.parametrize("field,value", [
    ("target_count", 0), ("target_count", True), ("episode_id", ""),
    ("actor_obs", np.zeros((5, 4))), ("critic_obs", np.zeros((6, 0))),
    ("actions", np.ones((5, 3)) * 1.1), ("rewards", np.full(5, np.nan)),
    ("terminated", np.zeros(5, np.float32)),
    ("terminated", np.array([True, False, False, False, False])),
    ("learn_mask", np.zeros(4, bool)),
])
def test_episode_schema_rejects_bad_shapes_flags_goals_and_nan(field, value):
    source = episode()
    payload = {key: getattr(source, key) for key in (
        "episode_id", "target_count", "actor_obs", "critic_obs", "actions",
        "rewards", "terminated", "learn_mask",
    )}
    payload[field] = value
    with pytest.raises(ValueError):
        construct("Episode", **payload)


def test_replay_load_rejects_bad_schema_nan_and_mismatched_dimensions_atomically():
    store = replay(steps=5)
    before = store.state_dict()
    for change in ("schema", "nan", "dimensions", "duplicate", "rng"):
        state = copy.deepcopy(before)
        if change == "schema":
            state["schema_version"] = 999
        elif change == "nan":
            state["episodes"][0]["actor_obs"][0, 0] = np.nan
        elif change == "dimensions":
            state["episodes"][0]["critic_obs"] = np.zeros((6, 8), np.float32)
        elif change == "duplicate":
            state["episodes"].append(copy.deepcopy(state["episodes"][0]))
        else:
            state["rng_state"] = {}
        with pytest.raises(ValueError):
            store.load_state_dict(state)
        assert store.steps == 30
        assert store.episodes[0].actor_obs.shape == (6, 4)


def learner(seed=3):
    return construct("RecurrentSAC", actor_dim=4, critic_dim=7, seed=seed)


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def test_act_starts_zero_has_small_raw_noise_and_explicit_episode_hidden_reset():
    model = learner()
    observation = np.array([1, -2, 0.5, 3], np.float32)
    deterministic, hidden = model.act(observation, deterministic=True)
    np.testing.assert_array_equal(deterministic, np.zeros(3, np.float32))
    assert hidden.shape == (1, 1, 128) and not hidden.requires_grad
    assert hidden.device.type == "cpu"
    sampled, _ = model.act(observation)
    assert np.max(np.abs(sampled)) < 0.1 and np.any(sampled != 0)
    # A nonzero trained readout is needed to observe memory through the action.
    with torch.no_grad():
        model.actor.mean.weight.fill_(0.02)
    state = None
    for _ in range(40):
        _, state = model.act(observation, state, deterministic=True)
    with_history, _ = model.act(-observation, state, deterministic=True)
    reset, _ = model.act(-observation, None, deterministic=True)
    assert not np.allclose(with_history, reset)
    np.testing.assert_array_equal(reset, model.act(-observation, None, deterministic=True)[0])


@pytest.mark.parametrize("bad", [np.zeros(7), np.full(4, np.nan), np.zeros((1, 4))])
def test_actor_rejects_privileged_dimensions_nan_and_non_vector_observations(bad):
    with pytest.raises(ValueError):
        learner().act(bad)


def test_skip_is_explicit_and_does_not_modify_models_optimizers_rng_or_hidden():
    model = learner()
    _, hidden = model.act(np.ones(4, np.float32))
    retained = hidden.clone()
    before = model.checkpoint()
    result = model.update(replay(steps=42))
    assert result["skipped"] is True and result["updates"] == 0
    assert result["reason"] == "replay_not_ready"
    assert_tree_equal(before, model.checkpoint())
    assert torch.equal(retained, hidden)


def test_initial_real_critic_update_freezes_actor_and_soft_updates_targets():
    model, store = learner(), replay()
    before_data = store.state_dict()["episodes"]
    actor_before = {key: value.clone() for key, value in model.actor.state_dict().items()}
    targets_before = [copy.deepcopy(q.state_dict()) for q in model.target_critics]
    _, hidden = model.act(np.ones(4, np.float32), deterministic=True)
    retained = hidden.clone()
    metrics = model.update(store)
    assert metrics["skipped"] is False and metrics["updates"] == 1
    assert metrics["learned_transitions"] > 0
    assert np.isfinite(metrics['critic_loss'])
    assert metrics['actor_loss'] is None and not metrics['actor_updated']
    assert_tree_equal(actor_before, model.actor.state_dict())
    assert torch.equal(hidden, retained)
    for old, target, current in zip(targets_before, model.target_critics, model.critics):
        for key, actual in target.state_dict().items():
            torch.testing.assert_close(actual, old[key] * 0.995 + current.state_dict()[key] * 0.005)
    assert_tree_equal(before_data, store.state_dict()["episodes"])


class FixedWindows:
    """Use real immutable episodes but choose exact windows for loss assertions."""

    def __init__(self, samples):
        self.samples = samples

    def ready(self, min_steps=256):
        return True

    def sample(self, batch_size=4, length=32):
        assert batch_size == 4 and length == 32
        return self.samples


def replace_episode(source, **changes):
    payload = {field: getattr(source, field) for field in (
        "episode_id", "target_count", "actor_obs", "critic_obs", "actions",
        "rewards", "terminated", "learn_mask",
    )}
    return construct("Episode", **(payload | changes))


def fixed_store(source, start):
    return FixedWindows([construct("SequenceSample", source, start, 32)])


def test_prefix_invalid_rewards_and_padding_have_no_loss_or_normalization_weight():
    source = episode(steps=6, mask=[False, False, False, True, False, True], terminal=True)
    altered_rewards = source.rewards.copy()
    altered_rewards[[0, 1, 2, 4]] = 1e8
    altered = replace_episode(source, rewards=altered_rewards)
    first, second = learner(), learner()
    result = first.update(fixed_store(source, 2))
    other = second.update(fixed_store(altered, 2))
    assert result["learned_transitions"] == 2
    assert result["actor_loss"] == other["actor_loss"]
    assert result["critic_loss"] == other["critic_loss"]
    assert_tree_equal(first.checkpoint(), second.checkpoint())


def test_full_prefix_burnin_matches_act_and_has_no_prefix_gradient():
    model = learner()
    source = episode(steps=40)
    seen = []
    handle = model.actor.gru.register_forward_hook(
        lambda module, args, output: seen.append((args[0].shape[1], torch.is_grad_enabled()))
    )
    model.update(fixed_store(source, 35))
    handle.remove()
    assert (35, False) in seen
    assert all(length <= 6 for length, gradients in seen if gradients)
    # The same whole prefix must reproduce online recurrent state before training.
    original = learner()
    observations = torch.tensor(source.actor_obs.copy()).unsqueeze(0)
    with torch.no_grad():
        sequence, _ = original.actor.gru(observations)
    state = None
    for index in range(41):
        _, state = original.act(source.actor_obs[index], state, deterministic=True)
    torch.testing.assert_close(state[0, 0], sequence[0, -1])


def test_target_critic_history_contains_previous_replay_action_at_aligned_next_state():
    model, source = learner(), episode(steps=5)
    seen = []
    handle = model.target_critics[0].gru.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    model.update(fixed_store(source, 2))
    handle.remove()
    # Prefix t=0,1 then train states t=2,3,4,5. At state t, history gets a[t-1].
    assert [part.shape[1] for part in seen] == [2, 4]
    history = torch.cat(seen, dim=1)[0].numpy()
    np.testing.assert_array_equal(history[:, :7], source.critic_obs)
    np.testing.assert_array_equal(history[0, 7:], np.zeros(3))
    np.testing.assert_array_equal(history[1:, 7:], source.actions)


@pytest.mark.parametrize("terminal,want_target", [(True, 2.0), (False, 12.0)])
def test_terminal_masks_bootstrap_but_ordinary_budget_truncation_does_not(terminal, want_target):
    model = learner()
    source = replace_episode(episode(steps=1, terminal=terminal), rewards=np.array([2], np.float32))
    # Constant real networks give a hand-computable Bellman target; cancel entropy
    # only for this numerical test (the production default remains fixed .001).
    model.config["alpha"] = 0.0
    with torch.no_grad():
        for critic in model.critics:
            for parameter in critic.parameters():
                parameter.zero_()
        for index, target in enumerate(model.target_critics):
            for parameter in target.parameters():
                parameter.zero_()
            target.head[-1].bias.fill_(10 + 10 * index)
    result = model.update(fixed_store(source, 0))
    assert result["target_q_mean"] == pytest.approx(want_target)
    assert result["critic_loss"] == pytest.approx(2 * want_target ** 2)


def test_squashed_log_probability_uses_raw_normalized_action_with_no_physical_scale():
    model = learner()
    mean = torch.tensor([[0.2, -0.4, 0.6]])
    log_std = torch.tensor([[-4.0, -4.0, -4.0]])
    sampled, log_probability = model.actor.sample(mean, log_std, model._rng)
    raw = sampled.detach().numpy()[0]
    pre_tanh = np.arctanh(raw)
    gaussian = -0.5 * ((pre_tanh - mean.numpy()[0]) / np.exp(-4)) ** 2 + 4 - 0.5 * math.log(2 * math.pi)
    expected = np.sum(gaussian - np.log(1 - raw ** 2))
    assert log_probability.item() == pytest.approx(float(expected), abs=2e-5)
    scaled = raw * np.array([.005, .005, math.pi/60])
    assert not np.allclose(raw, scaled)


def test_checkpoint_serialization_restores_predictions_rng_optimizers_and_next_update():
    model, store = learner(), replay()
    model.update(store)
    model.act(np.ones(4, np.float32))
    payload, replay_payload = model.checkpoint(), store.state_dict()
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored = learner(seed=900)
    restored.load_checkpoint(torch.load(buffer, weights_only=True, map_location="cpu"))
    restored_store = construct("EpisodicReplay")
    restored_store.load_state_dict(replay_payload)
    observation = np.array([0.2, -0.5, 0.7, 1], np.float32)
    assert_tree_equal(model.act(observation, deterministic=True), restored.act(observation, deterministic=True))
    for _ in range(3):
        assert_tree_equal(model.act(observation), restored.act(observation))
    assert model.update(store) == restored.update(restored_store)
    assert_tree_equal(model.checkpoint(), restored.checkpoint())
    payload["actor"]["mean.bias"].fill_(999)
    assert not torch.equal(payload["actor"]["mean.bias"], model.actor.mean.bias)


@pytest.mark.parametrize("corruption", ["schema", "dimension", "config", "nan", "target", "optimizer", "rng"])
def test_checkpoint_rejects_malformed_state_atomically(corruption):
    model = learner()
    model.update(replay())
    before = model.checkpoint()
    payload = copy.deepcopy(before)
    if corruption == "schema":
        payload["schema_version"] = 999
    elif corruption == "dimension":
        payload["actor_dim"] = 5
    elif corruption == "config":
        payload["config"]["gamma"] = 0.9
    elif corruption == "nan":
        payload["actor"]["mean.bias"][0] = float("nan")
    elif corruption == "target":
        payload["target_critics"][0]["gru.weight_ih_l0"] = torch.zeros(1)
    elif corruption == "optimizer":
        state = next(iter(payload["critic_optimizer"]["state"].values()))
        state["exp_avg"] = torch.zeros(1)
    else:
        payload["rng_state"] = torch.ones(2)
    with pytest.raises(ValueError):
        model.load_checkpoint(payload)
    assert_tree_equal(before, model.checkpoint())


@pytest.mark.parametrize("component", ["replay", "learner"])
def test_schema_version_rejects_booleans_instead_of_treating_true_as_version_one(component):
    model = replay(steps=5) if component == "replay" else learner()
    state = model.state_dict() if component == "replay" else model.checkpoint()
    state["schema_version"] = True
    with pytest.raises(ValueError):
        (model.load_state_dict if component == "replay" else model.load_checkpoint)(state)


def test_checkpoint_rejects_optimizer_moment_dtype_instead_of_silently_coercing_it():
    model = learner()
    model.update(replay())
    payload = model.checkpoint()
    state = next(iter(payload["critic_optimizer"]["state"].values()))
    state["exp_avg"] = state["exp_avg"].double()
    with pytest.raises(ValueError):
        model.load_checkpoint(payload)


def test_constructor_act_and_deterministic_prediction_do_not_consume_global_or_local_rng():
    global_rng = torch.random.get_rng_state().clone()
    model = learner()
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    local_rng = model.checkpoint()["rng_state"]
    model.act(np.ones(4, np.float32), deterministic=True)
    assert torch.equal(model.checkpoint()["rng_state"], local_rng)
    model.act(np.ones(4, np.float32))
    assert torch.equal(torch.random.get_rng_state(), global_rng)


def test_log_std_is_initially_minus_four_clamps_and_saturated_tanh_density_is_finite():
    model = learner()
    features = torch.zeros(1, 1, 128)
    _, log_std = model.actor.distribution(features)
    torch.testing.assert_close(log_std, torch.full((1, 1, 3), -4.0))
    with torch.no_grad():
        model.actor.log_std.bias.copy_(torch.tensor([-100.0, 100.0, -3.0]))
    _, log_std = model.actor.distribution(features)
    torch.testing.assert_close(log_std, torch.tensor([[[-5.0, 0.0, -3.0]]]))
    action, logp = model.actor.sample(torch.full((1, 3), 100.0), torch.zeros(1, 3), model._rng)
    assert torch.isfinite(logp).all() and (action == 1).all()


def test_checkpoint_without_optimizer_steps_and_evicted_replay_ids_roundtrip():
    model, restored = learner(), learner(100)
    restored.load_checkpoint(model.checkpoint())
    assert_tree_equal(model.checkpoint(), restored.checkpoint())
    original = construct("EpisodicReplay", capacity=5)
    original.add(episode(1))
    original.add(episode(2))
    store = construct("EpisodicReplay")
    store.load_state_dict(original.state_dict())
    with pytest.raises(ValueError):
        store.add(episode(1))
