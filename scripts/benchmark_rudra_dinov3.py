"""Bounded disposable real-data updates to choose a shared-GPU ACT batch."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.utils.data import default_collate

from sim.config import load_config
from sim.act.collect_dataset import validate_training_manifest
from sim.act.dataset import ActDataset
from sim.act.policy import build_act_policy, build_act_processors
from sim.act.train import configure_cuda_memory_budget
from sim.diffusion.profiles import NATIVE, validate_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[4, 8, 12, 16])
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_task(cfg, NATIVE)
    configure_cuda_memory_budget(cfg)
    torch.manual_seed(int(cfg.seed))
    dataset = ActDataset(str(cfg.act.dataset_dir), chunk_size=25,
                         image_stat_samples=100, camera_views=list(cfg.act.camera_views))
    result = dict(scope='disposable real-data optimizer benchmark; no physical rollout or saved model',
        manifest_sha256=hashlib.sha256((Path(cfg.act.dataset_dir)/'manifest.jsonl').read_bytes()).hexdigest(),
        manifest=validate_training_manifest(cfg.act.dataset_dir, episodes_per_target=50,
                                           camera_views=list(cfg.act.camera_views)),
        valid_frames=len(dataset), image_size=list(cfg.act.image_size),
        camera_views=list(dataset.camera_views), amp=False,
        cuda_memory_fraction=float(cfg.act.cuda_memory_fraction), candidates=[])
    stats = dataset.stats
    policy, policy_cfg = build_act_policy(cfg)
    preprocessor, _ = build_act_processors(policy_cfg, stats)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=policy_cfg.optimizer_lr,
                                 weight_decay=policy_cfg.optimizer_weight_decay)
    policy.train()
    encoder = policy.model.backbone.encoder
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    opt_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not opt_ids.intersection(id(p) for p in encoder.parameters())
    result.update(trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                  frozen_parameters=sum(p.numel() for p in policy.parameters() if not p.requires_grad),
                  optimizer_lr=policy_cfg.optimizer_lr, optimizer_weight_decay=policy_cfg.optimizer_weight_decay)
    for size in args.batches:
        row = dict(batch_size=size)
        try:
            samples = [dataset[int((i+.5)*len(dataset)/size)] for i in range(size)]
            batch = default_collate(samples)
            del samples
            for key in batch:
                if key.startswith('observation.images.'):
                    batch[key] = batch[key].float().div_(255)
            batch = preprocessor(batch)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            durations = []
            for iteration in range(3):
                torch.cuda.synchronize(); started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = policy(batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite real-data benchmark loss')
                loss.backward(); optimizer.step(); torch.cuda.synchronize()
                if iteration:
                    durations.append(time.monotonic()-started)
            assert all(p.grad is None for p in encoder.parameters())
            row.update(passed=True, loss=float(loss.detach()),
                seconds_per_update=sum(durations)/len(durations),
                examples_per_second=size*len(durations)/sum(durations),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                frozen_encoder_verified=True)
            print(json.dumps(row), flush=True)
            del batch, loss, metrics
        except torch.OutOfMemoryError:
            row.update(passed=False, reason='current-process allocator budget exceeded')
            print(json.dumps(row), flush=True)
            result['candidates'].append(row)
            break
        result['candidates'].append(row)
        optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    passed=[r for r in result['candidates'] if r.get('passed')]
    if not passed:
        raise RuntimeError('batch 4 did not fit this shared-GPU budget')
    # Keep allocator headroom as well as the global reserve for the other job.
    safe=[r for r in passed if r['peak_reserved_mib'] < 0.90*8192*float(cfg.act.cuda_memory_fraction)]
    selected=max(safe or passed[:1], key=lambda r:r['examples_per_second'])
    result['selected_batch_size']=selected['batch_size']
    result['selection']='highest measured examples/s within 90 percent of this process budget; otherwise smallest passing batch'
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'selected_batch_size':result['selected_batch_size']}),flush=True)


if __name__ == '__main__':
    main()
