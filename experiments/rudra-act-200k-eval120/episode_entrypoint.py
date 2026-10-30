"""Native policy execution, then durable state/trace storage only. No media renderer."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_batch import PROJECT, save_measurement
sys.path.insert(0,str(PROJECT))


def main():
    from sim.act.evaluate import run_act_episode
    from sim.act.inference_lock import inference_lease
    from sim.config import load_config
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--spec',required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--target-count',type=int,required=True)
    p.add_argument('--model',required=True); args=p.parse_args()
    plan=json.loads((args.root/'episode_plan.json').read_text())
    spec=next(s for s in plan if s['key']==args.spec)
    assert (args.seed,args.target_count,args.model)==(spec['seed'],spec['target_count'],spec['model_path'])
    cfg=load_config(str(args.root/'evaluation_config.yaml'))
    cfg.set_path('task.target_count',args.target_count)
    with inference_lease(on_wait=lambda:print(json.dumps({'event':'waiting_for_inference_slot'}),flush=True)):
        result=run_act_episode(cfg,seed=args.seed,model_path=args.model,preview=True,capture_replay=True)
        physical=save_measurement(args.root,cfg,args.seed,args.model,result)
        assert physical['layout_fingerprint']==spec['layout_fingerprint']
        print(json.dumps({**physical,'event':'measurement_saved'},ensure_ascii=False,allow_nan=False),flush=True)
    return 0 if result.success else 1


if __name__=='__main__':
    raise SystemExit(main())
