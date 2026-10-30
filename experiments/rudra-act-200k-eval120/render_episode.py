"""Explicit on-demand rendering of an existing measurement; never calls the policy."""
import argparse
import json
from pathlib import Path
import sys

from evaluate_batch import ROOT, PROJECT, sha256
sys.path.insert(0,str(PROJECT))


def main():
    import gzip
    from sim.config import Config
    from sim.act.replay import recover_replay_input
    from sim.act.video import export_inference_video
    from sim.act.media_retention import finalize_inference_video
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    rows=[json.loads(s) for s in (ROOT/'episodes.jsonl').read_text().splitlines() if s.strip()]
    matches=[r for r in rows if r['seed']==args.seed]
    if len(matches)!=1: raise ValueError('seed has no unique completed measurement')
    row=matches[0]; saved=Path(row['replay_input_path'])
    assert sha256(saved)==row['replay_input_sha256']
    with gzip.open(saved,'rt') as f: document=json.load(f)
    cfg=Config(document['config'])
    assert all('replay_state' in step for step in document['result']['trace'])
    replay_id=recover_replay_input(saved,cfg,seed=args.seed,model=row['model_path'])
    video=export_inference_video(saved.parent.parent/replay_id,ROOT/'videos')
    finalize_inference_video(saved.parent.parent/replay_id,video)
    print(json.dumps({'seed':args.seed,'video':str(video),'policy_reexecuted':False}),flush=True)


if __name__=='__main__': main()
