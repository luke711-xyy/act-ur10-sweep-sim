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
    p.add_argument('--waive-cpu-acceptance',action='store_true',
        help='explicitly skip CPU acceptance for this new formal training run; the waiver is recorded')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--acknowledge-gpu-training',action='store_true')
    p.add_argument('--continue-authorized-from',type=Path,
        help='one explicit pre-update preflight continuation; path to unchanged original bundle')
    p.add_argument('--preflight-start-index',type=int,default=0,
        help='fresh six-aligned preflight substream; never changes train/validation seeds')
    p.add_argument('--seed',type=int)
    p.add_argument('--target',type=int,choices=range(1,7))
    return p


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if args.continue_authorized_from is not None and args.mode!='preflight':
        p.error('--continue-authorized-from is only valid for preflight')
    if args.preflight_start_index and args.mode!='preflight':p.error('--preflight-start-index is only valid for preflight')
    if args.waive_cpu_acceptance and args.mode!='train':
        p.error('--waive-cpu-acceptance is only valid for a new formal training run')
    if args.mode=='inspect':
        from .lifecycle import inspect
        print(json.dumps(inspect(args.bundle),indent=2)); return 0
    if args.bundle is None or args.out is None: p.error('execution requires --bundle and --out')
    from .lifecycle import require_runtime
    require_runtime()
    if args.mode in ('train','resume'):
        if not args.acknowledge_gpu_training:
            p.error('GPU execution needs separate explicit launch authorization')
        if args.mode=='resume' and args.waive_cpu_acceptance:
            p.error('CPU acceptance can only be waived for a new training run, not resume')
        if args.waive_cpu_acceptance and args.acceptance_report:
            p.error('--waive-cpu-acceptance cannot be combined with --acceptance-report')
        if not args.waive_cpu_acceptance and not args.acceptance_report:
            p.error('GPU execution needs --acceptance-report unless --waive-cpu-acceptance is explicit')
        from .training import train
        train(args.bundle,args.out,args.acceptance_report,resume=args.mode=='resume',
            waive_cpu_acceptance=args.waive_cpu_acceptance)
    elif args.mode=='preflight':
        from .acceptance import preflight
        preflight(args.bundle,args.out,args.device,continue_from=args.continue_authorized_from,start_index=args.preflight_start_index)
    else:
        if args.run is None: p.error('evaluate/infer requires --run for checkpoint identity')
        if args.mode=='infer' and (args.seed is None or args.target is None): p.error('infer requires --seed and --target')
        from .inference import evaluate
        evaluate(args.bundle,args.run,args.out,args.device,args.seed,args.target)
    return 0
