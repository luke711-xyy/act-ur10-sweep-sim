"""Submit to the authoritative local workbench JobRegistry; never create a second owner."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .config import read_launch_config


def launch_request(training_config,*,resume=None,pilot_only=False,task_config=None):
    path=Path(training_config).resolve()
    settings=read_launch_config(path)
    body=dict(policy_type="diffusion",dataset=settings["dataset"],out=settings["out"],
              training_config=str(path),resume=resume,pilot_only=pilot_only)
    if task_config:body["config"]=task_config
    return body


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config",required=True)
    parser.add_argument("--config")
    parser.add_argument("--resume")
    parser.add_argument("--pilot-only",action="store_true")
    parser.add_argument("--url",default="http://127.0.0.1:8773")
    args=parser.parse_args(argv)
    parsed=urlparse(args.url)
    if parsed.scheme!="http" or parsed.hostname not in ("127.0.0.1","localhost","::1"):
        raise ValueError("training launch is restricted to the local Linux workbench")
    body=launch_request(args.training_config,resume=args.resume,
                        pilot_only=args.pilot_only,task_config=args.config)
    request=Request(args.url.rstrip("/")+"/api/jobs/train",data=json.dumps(body).encode(),
                    headers={"Content-Type":"application/json"},method="POST")
    with urlopen(request,timeout=30) as response:
        job=json.load(response)
    print(json.dumps(dict(event="registered_training_job",**job)),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
