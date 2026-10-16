"""Snapshot known dataset/evaluation/preflight identities without loading RGB."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old-plan','dataset-manifest','residual-schedule','residual-ledger','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--additional',type=Path,action='append',default=[])
    args=p.parse_args()
    from pac_act.exclusions import extract
    from pac_act.storage import write_json
    if args.out.exists(): raise FileExistsError('exclusion snapshot must be new')
    data=extract([args.old_plan,args.dataset_manifest,args.residual_schedule,args.residual_ledger,*args.additional])
    if not set(range(2120000,2120120)).issubset(data['seeds']):
        raise ValueError('old120 evidence incomplete')
    data['complete_known_inputs']=True
    write_json(args.out,data)
    print(f"excluded {len(data['seeds'])} known seeds and {len(data['fingerprints'])} fingerprints; entire residual5M interval reserved")
