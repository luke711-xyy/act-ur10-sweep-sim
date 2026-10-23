"""Generate host-local two-view ACT/scene YAML; never launch jobs or download weights."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--config-out', type=Path,
                        default=ROOT/'configs/local-rudra-dinov3-act.yaml')
    parser.add_argument('--device', choices=('cuda','cpu','mps'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--cuda-memory-fraction', type=float)
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--check-data', action='store_true',
                        help='also run the existing read-only 300-demo manifest gate')
    args = parser.parse_args(argv)
    dataset, out, target = (p.expanduser().resolve() for p in
                            (args.dataset,args.out,args.config_out))
    scene_target = target.with_name(target.stem+'-scene.yaml')
    if not (dataset/'manifest.jsonl').is_file():
        parser.error('dataset manifest.jsonl is missing')
    if args.batch_size < 1 or not 1 <= args.port <= 65535:
        parser.error('batch-size must be positive and port must be in 1..65535')
    if args.cuda_memory_fraction is not None:
        if (args.device != 'cuda' or not math.isfinite(args.cuda_memory_fraction)
                or not 0 < args.cuda_memory_fraction <= 1):
            parser.error('cuda-memory-fraction requires CUDA and must be in (0,1]')
    if dataset == out or out.is_relative_to(dataset) or dataset.is_relative_to(out):
        parser.error('model output must be separate from the demonstration dataset')
    if target.is_relative_to(dataset) or target.is_relative_to(out):
        parser.error('configuration must be outside the dataset and model output directories')
    if target.exists() or scene_target.exists():
        parser.error('configuration already exists; choose a new --config-out')
    cfg = yaml.safe_load((ROOT/'configs/rudra-dinov3-act-cuda.yaml').read_text())
    scene_source = Path(cfg['scene']['profile'])
    if not scene_source.is_absolute():
        scene_source = ROOT/scene_source
    scene = yaml.safe_load(scene_source.read_text())
    cfg.pop('_source_config',None)
    cfg['act'].update(dataset_dir=str(dataset), model_dir=str(out),
                      device=args.device,batch_size=args.batch_size)
    cfg['act']['cuda_memory_fraction'] = (
        args.cuda_memory_fraction if args.cuda_memory_fraction is not None
        else cfg['act'].get('cuda_memory_fraction',.375)
    ) if args.device == 'cuda' else None
    cfg['web'].update(host='127.0.0.1',port=args.port,inference_model_root=str(out),
                      inference_video_dir=str(out.parent/(out.name+'_videos')),
                      inference_policy='act',readonly_dataset=True)
    cfg['scene']['profile'] = str(scene_target)
    scene['sweep_task']['source_config'] = str(target)
    manifest_summary = None
    if args.check_data:
        sys.path.insert(0,str(ROOT))
        from sim.act.collect_dataset import validate_training_manifest
        manifest_summary = validate_training_manifest(
            str(dataset),episodes_per_target=int(cfg['act']['training_episodes_per_target']),
            camera_views=list(cfg['act']['camera_views']))
    target.parent.mkdir(parents=True,exist_ok=True)
    with scene_target.open('x',encoding='utf-8') as handle:
        yaml.safe_dump(scene,handle,sort_keys=False,allow_unicode=True)
    try:
        with target.open('x',encoding='utf-8') as handle:
            yaml.safe_dump(cfg,handle,sort_keys=False,allow_unicode=True)
    except Exception:
        scene_target.unlink()
        raise
    train = [sys.executable,'-u','-B','-m','sim.act.train','--config',str(target),'--no-trackio']
    workbench = [sys.executable,'-B','-m','sim.web.run','--config',str(target)]
    print(json.dumps({'config':str(target),'scene_config':str(scene_target),
                      'dataset_validation':manifest_summary,
                      'train_argv':train,'workbench_argv':workbench,
                      'jobs_started':False},ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
