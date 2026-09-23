"""Unit tests for the frozen-V4 residual RL layer."""

from pathlib import Path

import numpy as np

from sim.act.residual import (
    ResidualAgent,
    ResidualReplayBuffer,
    ResidualSpec,
    clamp_residual,
    residual_observation,
    residual_reward,
)
from sim.act.rollout import _rate_limit_action
from sim.act.residual_rollout import _apply_action, run_residual_episode
from sim.config import load_config


def test_residual_observation_and_limits_are_explicit():
    spec = ResidualSpec()
    value = residual_observation(np.zeros(26), np.zeros((20, 4)), np.ones(512), True)
    assert value.shape == (619,)
    assert value[-1] == 1.0
    assert value[-513] == 1.0
    assert spec.observation_dim == 619
    clipped = clamp_residual(np.ones(4), spec)
    np.testing.assert_allclose(clipped, np.asarray(spec.residual_limit, dtype=np.float32))


def test_exact_count_reward_penalises_overcollection():
    exact = residual_reward(3, 4, 4, 0.0, 1.0, True, False, 1.0, 20.0)
    over = residual_reward(4, 5, 4, 0.0, 1.0, False, True, 1.0, 20.0)
    assert exact > 0.0
    assert over < 0.0


def test_object_progress_shaping_rewards_toward_tray_not_policy_input():
    toward = residual_reward(0, 0, 2, 0.0, 1.0, False, False, 1.0, 20.0,
                             object_progress=0.01, object_progress_weight=10.0)
    away = residual_reward(0, 0, 2, 0.0, 1.0, False, False, 1.0, 20.0,
                           object_progress=-0.01, object_progress_weight=10.0)
    assert toward > away
    assert np.isclose(toward - away, 0.2)


def test_part_contact_reward_is_bounded_and_optional():
    no_contact = residual_reward(0, 0, 1, 0.0, 1.0, False, False, 1.0, 20.0,
                                 part_contact_fraction=0.0, part_contact_weight=0.1)
    contact = residual_reward(0, 0, 1, 0.0, 1.0, False, False, 1.0, 20.0,
                              part_contact_fraction=1.0, part_contact_weight=0.1)
    overshoot = residual_reward(0, 0, 1, 0.0, 1.0, False, False, 1.0, 20.0,
                                part_contact_fraction=2.0, part_contact_weight=0.1)
    assert np.isclose(contact - no_contact, 0.1)
    assert np.isclose(overshoot, contact)


def test_replay_buffer_wraps_and_samples():
    buffer = ResidualReplayBuffer(capacity=3, observation_dim=619, action_dim=4)
    for i in range(5):
        buffer.add(np.full(619, i), np.full(4, i), i, np.full(619, i + 1), False)
    assert len(buffer) == 3
    sample = buffer.sample(2, np.random.default_rng(0))
    assert sample[0].shape == (2, 619)
    assert sample[1].shape == (2, 4)


def test_residual_agent_updates_and_round_trips_checkpoint(tmp_path: Path):
    agent = ResidualAgent(ResidualSpec(), device="cpu", seed=0)
    rng = np.random.default_rng(0)
    batch = (
        rng.normal(size=(8, 619)).astype(np.float32),
        rng.normal(size=(8, 4)).astype(np.float32),
        rng.normal(size=8).astype(np.float32),
        rng.normal(size=(8, 619)).astype(np.float32),
        np.zeros(8, dtype=np.float32),
    )
    metrics = agent.update(batch)
    assert {"critic_loss", "q1_mean", "q2_mean", "actor_loss", "mean_residual",
            "mean_residual_fraction", "actor_updated"} == set(metrics)
    path = agent.save(tmp_path / "residual.pt", {"episode": 3})
    restored = ResidualAgent(ResidualSpec(), device="cpu", seed=1)
    assert restored.load(path)["episode"] == 3
    np.testing.assert_allclose(restored.act(np.zeros(619, dtype=np.float32)),
                               agent.act(np.zeros(619, dtype=np.float32)), atol=1e-6)


def test_default_config_keeps_residual_separate_from_v4():
    cfg = load_config()
    assert bool(cfg.residual_rl.enabled)
    assert str(cfg.residual_rl.out_dir) != str(cfg.act.model_dir)
    assert cfg.residual_rl.target_count is None


def test_absolute_act_target_is_rate_limited_before_execution():
    cfg = load_config()

    class _EE:
        @staticmethod
        def tcp_yaw():
            return 0.0

    class _Env:
        ee = _EE()

        @staticmethod
        def tcp():
            return np.asarray([0.20, 0.0, 0.18], dtype=np.float32)

    limited = _rate_limit_action(
        cfg, np.asarray([-0.40, 0.30, -0.012, np.pi], dtype=np.float32), _Env(), False
    )
    assert np.isclose(limited[0], 0.20 - cfg.controller.travel_speed / cfg.act.action_hz)
    assert np.isclose(limited[2], 0.18 - cfg.controller.z_speed / cfg.act.action_hz)
    assert abs(float(limited[3])) <= float(cfg.controller.yaw_speed / cfg.act.action_hz) + 1e-6
    assert limited[2] >= float(cfg.workspace.z_search_start)


def test_residual_is_direct_additive_and_z_is_masked_after_contact():
    cfg = load_config()
    base = np.asarray([0.20, 0.10, 0.12, 0.10], dtype=np.float32)
    correction = np.asarray([0.01, -0.01, -0.04, 0.05], dtype=np.float32)

    before_contact, applied = _apply_action(cfg, base, correction, False)
    np.testing.assert_allclose(before_contact, [0.21, 0.09, 0.08, 0.15], atol=1e-6)
    np.testing.assert_allclose(applied, correction, atol=1e-6)

    after_contact, applied = _apply_action(cfg, base, correction, True)
    assert after_contact[2] == base[2]
    assert applied[2] == 0.0


def test_residual_rollout_does_not_close_an_injected_provider():
    cfg = load_config()
    cfg.set_path("components.count", 1)
    cfg.set_path("episode.max_time", 0.05)

    class _Provider:
        closed = False

        def reset(self):
            pass

        def step(self, env, contact_latched):
            tcp = env.tcp()
            return np.zeros(619, dtype=np.float32), np.asarray(
                [tcp[0], tcp[1], tcp[2], env.ee.tcp_yaw()], dtype=np.float32
            )

        def close(self):
            self.closed = True

    class _ZeroAgent:
        @staticmethod
        def act(observation, explore=False, exploration_std=0.0):
            return np.zeros(4, dtype=np.float32)

    provider = _Provider()
    result = run_residual_episode(
        cfg, "unused", _ZeroAgent(), seed=7, explore=False, base_provider=provider
    )
    assert result.failure_reason
    assert not provider.closed
