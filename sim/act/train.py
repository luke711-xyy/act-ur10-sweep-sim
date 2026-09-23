"""Train the pinned LeRobot ACT policy on the local MuJoCo dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save a policy snapshot every N optimization steps")
    parser.add_argument("--resume", default=None,
                        help="load a previously saved ACT policy before continuing")
    args = parser.parse_args(argv)
    from torch.utils.data import DataLoader
    import torch

    from ..config import load_config
    from .dataset import ActDataset
    from .policy import build_act_policy

    cfg = load_config(args.config)
    dataset_root = args.dataset or str(cfg.act.dataset_dir)
    out_root = Path(args.out or str(cfg.act.model_dir))
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = ActDataset(dataset_root, chunk_size=int(cfg.act.chunk_size))
    stats_path = out_root / "dataset_stats.json"
    stats_path.write_text(json.dumps(dataset.stats, indent=2), encoding="utf-8")
    loader = DataLoader(dataset, batch_size=int(cfg.act.batch_size), shuffle=True,
                        num_workers=0, drop_last=True)
    policy, policy_cfg = build_act_policy(cfg, pretrained_path=args.resume)
    device = str(policy_cfg.device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    max_steps = int(args.steps or cfg.act.max_steps)
    iterator = iter(loader)
    started = time.monotonic()
    for step in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(batch)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), **metrics,
                              "elapsed_s": time.monotonic() - started}))
        if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
            checkpoint_dir = out_root / f"checkpoint-{step + 1:06d}"
            policy.save_pretrained(checkpoint_dir)
            shutil.copyfile(stats_path, checkpoint_dir / "dataset_stats.json")
            (checkpoint_dir / "training_state.json").write_text(
                json.dumps({"step": step + 1, "dataset": str(dataset_root),
                            "device": device}, indent=2),
                encoding="utf-8",
            )
        if time.monotonic() - started >= float(cfg.act.train_hours) * 3600.0:
            break
    policy.save_pretrained(out_root)
    (out_root / "training_summary.json").write_text(
        json.dumps({"steps": step + 1, "dataset": str(dataset_root), "device": device}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
