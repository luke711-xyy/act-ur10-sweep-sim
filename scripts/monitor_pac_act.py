"""Read-only PAC-ACT dashboard, separate from the SAC workbench on 8896."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--port',type=int,default=8897)
    args=p.parse_args()
    from pac_act.dashboard import serve
    serve(args.out,port=args.port)
