import argparse
import json
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description='Isolated Hybrid PAC-ACT; default inspect is read-only.')
    p.add_argument('--mode',choices=['inspect','preflight','train','resume','evaluate','infer'],default='inspect')
    p.add_argument('--bundle',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--run',type=Path,help='training run owning the verified checkpoint')
    p.add_argument('--acceptance-report',type=Path)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--acknowledge-gpu-training',action='store_true')
    p.add_argument('--seed',type=int)
    p.add_argument('--target',type=int,choices=range(1,7))
    return p


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if args.mode=='inspect':
        from .lifecycle import inspect
        print(json.dumps(inspect(args.bundle),indent=2)); return 0
    if args.bundle is None or args.out is None: p.error('execution requires --bundle and --out')
    from .lifecycle import require_runtime
    require_runtime()
    if args.mode in ('train','resume'):
        if not args.acknowledge_gpu_training or not args.acceptance_report:
            p.error('GPU execution needs separate explicit launch authorization and --acceptance-report')
        from .training import train
        train(args.bundle,args.out,args.acceptance_report,resume=args.mode=='resume')
    elif args.mode=='preflight':
        from .acceptance import preflight
        preflight(args.bundle,args.out,args.device)
    else:
        if args.run is None: p.error('evaluate/infer requires --run for checkpoint identity')
        if args.mode=='infer' and (args.seed is None or args.target is None): p.error('infer requires --seed and --target')
        from .inference import evaluate
        evaluate(args.bundle,args.run,args.out,args.device,args.seed,args.target)
    return 0
