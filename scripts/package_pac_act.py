"""Create a new portable model/control bundle; never train or modify inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--exclusions',type=Path,required=True)
    args=p.parse_args()
    from pac_act.identity import build_bundle
    result=build_bundle(args.out,args.model,json.loads(args.exclusions.read_text()))
    print(json.dumps(dict(identity=result['identity'],model=str(result['model']))))


if __name__=='__main__': main()
