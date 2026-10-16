"""Read numeric logs and normalization only; never run a policy or physics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    from safetensors.numpy import load_file
    from pac_act.offline_diagnostics import numeric_diagnosis
    from pac_act.storage import write_json,sha256
    stats_path=next(args.model.glob('*unnormalizer*.safetensors'))
    stats=load_file(str(stats_path));scales=stats['action.std'][[0,1,3]]
    report=numeric_diagnosis(args.run,scales)
    paths=[Path(p) for p in report['source_numeric_files']]
    paths.extend(p for p in (args.run/'acceptance.json',args.run/'status.json',stats_path) if p.exists())
    report['source_hashes']={str(p.resolve()):sha256(p) for p in paths}
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'numeric_diagnosis.json',report)
    assert all(sha256(Path(p))==h for p,h in report['source_hashes'].items())
    print(json.dumps({k:report[k] for k in ('blocks','effective_actions','environment_reward','physical_std','sensitivity')},indent=2))


if __name__=='__main__':main()
