"""Start one explicitly authorized preflight after a verified paired evaluation."""
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--after-batch',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--queue-state',type=Path,required=True)
    parser.add_argument('--preflight-start-index',type=int,required=True)
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--dependency-profile',choices=('legacy_100k','dinov3_sync5_200k'),
        default='legacy_100k',help='explicit protocol used to validate the completed dependency batch')
    args=parser.parse_args()
    source=args.bundle.resolve()/'source'
    sys.path.insert(0,str(source))
    from pac_act.preflight_queue import run_queue
    run_queue(args.bundle,args.after_batch,args.out,args.queue_state,
        start_index=args.preflight_start_index,device=args.device,
        dependency_profile=args.dependency_profile)


if __name__=='__main__':main()
