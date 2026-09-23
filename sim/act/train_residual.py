"""Online residual-RL training on top of a frozen early-V4 ACT checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..config import load_config, save_config
from .residual import ResidualAgent, ResidualReplayBuffer, spec_from_config
from .residual_rollout import run_residual_episode


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None,
                        help="early-V4 LeRobot ACT checkpoint directory")
    parser.add_argument("--out", default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=40000)
    parser.add_argument("--counts", type=int, nargs="+", default=list(range(1, 7)),
                        help="component counts to cycle through during residual rollouts")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser


def _device(cfg, override):
    if override:
        return override
    requested = str(cfg.act.device)
    if requested == "mps":
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    base_model = args.base_model or str(cfg.act.model_dir)
    if not Path(base_model).exists():
        raise FileNotFoundError(
            f"V4 ACT checkpoint not found: {base_model}. Train V4 first with "
            "sim.act.train; residual RL never creates a replacement base actor."
        )
    out_root = Path(args.out or str(cfg.residual_rl.out_dir))
    out_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(out_root / "config.yaml"))
    spec = spec_from_config(cfg)
    agent = ResidualAgent(
        spec, device=_device(cfg, args.device), gamma=float(cfg.residual_rl.gamma),
        tau=float(cfg.residual_rl.tau), actor_lr=float(cfg.residual_rl.actor_lr),
        critic_lr=float(cfg.residual_rl.critic_lr),
        prior_weight=float(cfg.residual_rl.residual_prior_weight), seed=int(args.seed_base),
        policy_delay=int(cfg.residual_rl.policy_delay),
        target_policy_noise=float(cfg.residual_rl.target_policy_noise),
        noise_clip=float(cfg.residual_rl.target_noise_clip),
    )
    start_episode = 0
    if args.resume:
        metadata = agent.load(args.resume)
        start_episode = int(metadata.get("episode", 0))
    replay = ResidualReplayBuffer(
        int(cfg.residual_rl.replay_capacity), spec.observation_dim, spec.action_dim
    )
    rng = np.random.default_rng(int(args.seed_base))
    history = []
    started = time.monotonic()
    for episode in range(start_episode, start_episode + int(args.episodes)):
        episode_cfg = cfg.copy()
        episode_cfg.set_path("components.count", int(args.counts[episode % len(args.counts)]))
        result = run_residual_episode(
            episode_cfg, base_model, agent, seed=int(args.seed_base) + episode,
            explore=not args.eval_only,
        )
        for transition in result.transitions:
            replay.add(transition.observation, transition.action, transition.reward,
                       transition.next_observation, transition.done)
        metrics = {
            "episode": episode + 1,
            "count": int(episode_cfg.components.count),
            "success": bool(result.success),
            "failure_reason": result.failure_reason,
            "collected": result.collected,
            "total": result.total,
            "elapsed": result.elapsed,
            "peak_force": result.peak_force,
            "contact_steps": result.contact_steps,
            "first_contact_time": result.first_contact_time,
            "contact_path_length": result.contact_path_length,
            "object_goal_progress": result.object_goal_progress,
            "part_contact_fraction": result.part_contact_fraction,
            "mean_contact_force": result.mean_contact_force,
            "mean_abs_residual_action": result.mean_abs_residual_action.tolist(),
            "max_abs_residual_action": result.max_abs_residual_action.tolist(),
            "episode_reward": float(sum(item.reward for item in result.transitions)),
            "replay": len(replay),
        }
        if len(result.residual_actions):
            residual_limits = np.asarray(spec.residual_limit, dtype=np.float32)
            metrics["mean_residual_action"] = result.residual_actions.mean(axis=0).tolist()
            metrics["mean_residual_fraction_by_dim"] = (
                np.abs(result.residual_actions) / residual_limits
            ).mean(axis=0).tolist()
        if not args.eval_only and len(replay) >= int(cfg.residual_rl.warmup_transitions):
            updates = max(1, len(result.transitions) *
                          int(cfg.residual_rl.updates_per_transition))
            losses = []
            for _ in range(updates):
                losses.append(agent.update(replay.sample(int(cfg.residual_rl.batch_size), rng)))
            if losses:
                metrics.update({key: float(np.mean([item[key] for item in losses]))
                                for key in losses[0]})
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        every = int(cfg.residual_rl.checkpoint_every_episodes)
        if every > 0 and (episode + 1) % every == 0:
            agent.save(out_root / "checkpoints" / f"episode_{episode + 1:06d}.pt",
                       {"episode": episode + 1, "base_model": str(base_model)})
        if time.monotonic() - started >= float(cfg.act.train_hours) * 3600.0:
            break
    final = agent.save(out_root / "residual_latest.pt",
                       {"episode": history[-1]["episode"] if history else start_episode,
                        "base_model": str(base_model), "history": history})
    (out_root / "training_summary.json").write_text(
        json.dumps({"base_model": str(base_model), "checkpoint": str(final),
                    "episodes": len(history), "history": history}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
