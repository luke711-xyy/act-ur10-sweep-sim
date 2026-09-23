"""Run V4 ACT plus a saved residual checkpoint on MuJoCo layouts."""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..config import load_config
from .residual import ResidualAgent, spec_from_config
from .residual_rollout import run_residual_episode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=50000)
    parser.add_argument("--counts", type=int, nargs="+", default=list(range(1, 7)),
                        help="component counts to cycle through during evaluation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    requested = args.device or str(cfg.act.device)
    if requested == "mps":
        import torch
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    agent = ResidualAgent(
        spec_from_config(cfg), device=requested,
        gamma=float(cfg.residual_rl.gamma), tau=float(cfg.residual_rl.tau),
        actor_lr=float(cfg.residual_rl.actor_lr),
        critic_lr=float(cfg.residual_rl.critic_lr),
        prior_weight=float(cfg.residual_rl.residual_prior_weight),
        policy_delay=int(cfg.residual_rl.policy_delay),
        target_policy_noise=float(cfg.residual_rl.target_policy_noise),
        noise_clip=float(cfg.residual_rl.target_noise_clip),
    )
    metadata = agent.load(args.residual)
    print(json.dumps({"base_model": args.base_model, "residual": args.residual,
                      "checkpoint_metadata": metadata}, ensure_ascii=False))
    for index in range(int(args.episodes)):
        episode_cfg = cfg.copy()
        episode_cfg.set_path("components.count", int(args.counts[index % len(args.counts)]))
        result = run_residual_episode(
            episode_cfg, args.base_model, agent, seed=int(args.seed_base) + index, explore=False
        )
        print(json.dumps({"episode": index + 1, "seed": int(args.seed_base) + index,
                          "count": int(episode_cfg.components.count),
                          "success": result.success, "failure_reason": result.failure_reason,
                          "collected": result.collected, "total": result.total,
                          "elapsed": result.elapsed, "peak_force": result.peak_force,
                          "contact_steps": result.contact_steps,
                          "first_contact_time": result.first_contact_time,
                          "contact_path_length": result.contact_path_length,
                          "object_goal_progress": result.object_goal_progress,
                          "part_contact_fraction": result.part_contact_fraction,
                          "mean_contact_force": result.mean_contact_force,
                          "mean_abs_residual_action": result.mean_abs_residual_action.tolist(),
                          "max_abs_residual_action": result.max_abs_residual_action.tolist(),
                          "mean_residual_action": result.residual_actions.mean(axis=0).tolist(),
                          "mean_residual_fraction_by_dim": (
                              np.abs(result.residual_actions) /
                              np.asarray(spec_from_config(cfg).residual_limit, dtype=np.float32)
                          ).mean(axis=0).tolist(),
                          "episode_reward": float(sum(item.reward for item in result.transitions))},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
