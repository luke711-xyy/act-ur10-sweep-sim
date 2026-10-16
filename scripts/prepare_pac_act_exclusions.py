"""Snapshot known dataset/evaluation/preflight identities without loading RGB."""
import argparse
import json
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
    ledger=json.loads(args.residual_ledger.read_text())
    if isinstance(ledger,dict) and ledger.get('record_type')=='history_inventory':
        status=ledger.get('historical_status')
        reconstructed=ledger.get('historical_attempts_reconstructed')
        if not isinstance(status,str) or type(reconstructed) is not bool:
            raise ValueError('invalid residual history inventory status')
        data['residual_history']={key:ledger[key] for key in (
            'record_type','historical_status','historical_attempts_reconstructed',
            'missing_legacy_export_root','searched_roots') if key in ledger}
    data['complete_known_inputs']=True
    write_json(args.out,data)
    print(f"excluded {len(data['seeds'])} known seeds and {len(data['fingerprints'])} fingerprints; entire residual5M interval reserved")
