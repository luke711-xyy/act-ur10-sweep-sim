"""Durable bounded full-training -> one-shot acceptance worker for this plan."""
import argparse
import json
from pathlib import Path
from .config import DEFAULT_DATASET,DEFAULT_OUT


def ready_for_acceptance(status):
    return (status.get("state")=="training_complete"
            and bool(status.get("pilot",{}).get("passed"))
            and int(status.get("full",{}).get("completed_updates",0))>0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--dataset",default=DEFAULT_DATASET)
    parser.add_argument("--out",default=DEFAULT_OUT)
    parser.add_argument("--resume")
    parser.add_argument("--steps",type=int)
    args=parser.parse_args(argv)
    from . import train,acceptance
    from .training import atomic_json
    train_argv=["--config",args.config,"--dataset",args.dataset,"--out",args.out]
    if args.resume: train_argv+=["--resume",args.resume]
    if args.steps is not None: train_argv+=["--steps",str(args.steps)]
    code=train.main(train_argv)
    status=json.loads((Path(args.out)/"status.json").read_text())
    if code or not ready_for_acceptance(status):
        print(json.dumps(dict(event="pipeline_stopped",training_status=status.get("state"),
                              acceptance="not_run")),flush=True)
        return code or 2
    print(json.dumps(dict(event="pipeline_acceptance_started",
                          completed_updates=status["full"]["completed_updates"])),flush=True)
    acceptance.main(["--config",args.config,"--dataset",args.dataset,"--run",args.out])
    attempts=json.loads((Path(args.out)/"acceptance/attempts.json").read_text())
    atomic_json(Path(args.out)/"pipeline_result.json",
                dict(state="completed",completed_updates=status["full"]["completed_updates"],
                     final_ema=status["final_ema"],cases=attempts,
                     generalization_claim=False,superiority_to_act_claim=False))
    print(json.dumps(dict(event="pipeline_complete",
                          successes=sum(r["state"]=="success" for r in attempts),
                          total=len(attempts))),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
