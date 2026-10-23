"""One-shot, preregistered native acceptance; no retries or checkpoint selection."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from sim.config import load_config
from .training import atomic_json, ResourceGuard, load_payload
from .config import DEFAULT_DATASET, DEFAULT_OUT


def acceptance_cases():
    return [dict(seed=81000000+n,target_count=n) for n in range(1,7)]


def check_stop(stop):
    if stop["requested"]:
        raise InterruptedError("acceptance stopped")


def require_new_layout(fingerprint, training_fingerprints):
    if fingerprint in training_fingerprints:
        raise RuntimeError("duplicate actual layout; stop, do not replace seed")


def require_training_identity(checkpoint_identity, run, audit):
    if not checkpoint_identity or not (
            checkpoint_identity == run.get("identity") == audit.get("identity")):
        raise ValueError("checkpoint/run/dataset identity mismatch")


def reserve_case(path, case):
    path = Path(path)
    rows = json.loads(path.read_text()) if path.exists() else []
    if any(row["seed"]==case["seed"] for row in rows):
        raise RuntimeError("seed already attempted; rerunning is forbidden")
    rows.append(dict(case,state="started"))
    atomic_json(path,rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--dataset",default=DEFAULT_DATASET)
    parser.add_argument("--run",default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    stop = {"requested":False}
    def request_stop(_signal,_frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    run = Path(args.run).resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not run.is_relative_to(workspace/"runs"):
        raise ValueError("acceptance artifacts must stay in independent worktree")
    status = json.loads((run/"status.json").read_text())
    if status["state"] != "training_complete" or not status.get("pilot",{}).get("passed"):
        raise RuntimeError("acceptance requires completed full training and a passed pilot")
    checkpoint = run/"final_ema.pt"
    payload = load_payload(checkpoint)
    if payload["stage"]!="full" or payload["step"]<1:
        raise RuntimeError("no completed full-training optimizer update")
    step = payload["step"]
    checkpoint_identity = payload["identity"]
    del payload
    saved_audit = json.loads((run/"data_audit.json").read_text())
    from .dataset import DiffusionDataset
    actual = DiffusionDataset(args.dataset)
    audit = actual.audit()
    require_training_identity(checkpoint_identity,status,audit)
    require_training_identity(checkpoint_identity,status,saved_audit)
    if saved_audit["layout_fingerprints"] != audit["layout_fingerprints"]:
        raise ValueError("training layout fingerprints changed")
    fingerprints = set(saved_audit["layout_fingerprints"])
    del actual
    root = run/"acceptance"
    root.mkdir(parents=True,exist_ok=True)
    with (root/".acceptance.lock").open("a+") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts_path = root/"attempts.json"
        if attempts_path.exists():
            raise RuntimeError("acceptance already started; no retries or replacement seeds")
        from sim.environments.sweep_env import SweepEnv
        layouts = []
        # Preflight all six before executing any policy. No model in preflight.
        for case in acceptance_cases():
            check_stop(stop)
            cfg = load_config(args.config,overrides=[f"task.target_count={case['target_count']}"])
            env = SweepEnv(cfg,seed=case["seed"])
            try:
                env.reset(seed=case["seed"])
                poses = np.concatenate((env.component_positions(),env.component_quats()),axis=1).astype("<f4")
                fingerprint = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
                require_new_layout(fingerprint,fingerprints)
                layouts.append(dict(case,layout_fingerprint=fingerprint))
            finally:
                env.close()
        atomic_json(root/"preregistration.json",dict(cases=layouts,model=str(checkpoint),
                    completed_updates=step,not_a_generalization_benchmark=True))
        guard = ResourceGuard(workspace)
        for case in layouts:
            check_stop(stop)
            guard.check(reserve=2*2**30)
            rows = reserve_case(attempts_path,case)
            log_path = root/f"N{case['target_count']}.log"
            command = [sys.executable,"-m","sim.act.evaluate","--config",args.config,
                       "--policy-type","diffusion","--model",str(checkpoint),
                       "--seed",str(case["seed"]),"--target-count",str(case["target_count"]),
                       "--wall-seconds","1200","--record-replay",
                       "--replay-root",str(root/"replays"),"--video-root",str(root/"videos")]
            started = time.perf_counter()
            with log_path.open("w") as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                           cwd=workspace,start_new_session=True)
                try:
                    while process.poll() is None:
                        check_stop(stop)
                        guard.check()
                        if time.perf_counter()-started >= 1200:
                            raise TimeoutError("20-minute wall-clock limit")
                        time.sleep(.5)
                except BaseException as exc:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
                    rows[-1].update(state="infrastructure_failure",error=str(exc),
                                    wall_seconds=time.perf_counter()-started)
                    atomic_json(attempts_path,rows)
                    raise
            summaries = []
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if "video_error" in row and "success" in row:
                        summaries.append(row)
                except (ValueError,TypeError):
                    pass
            result = summaries[-1] if summaries else None
            if (not result or result.get("replay_error") or result.get("video_error")
                    or not result.get("video_path") or process.returncode not in (0,1)):
                rows[-1].update(state="infrastructure_failure",result=result,
                                returncode=process.returncode)
                atomic_json(attempts_path,rows)
                raise RuntimeError(f"native acceptance infrastructure failure; see {log_path}")
            manifest = root/"replays"/"manifest.jsonl"
            replay_records = [json.loads(line) for line in manifest.read_text().splitlines()]
            replay = next(r for r in replay_records if r["episode_id"]==result["replay_episode_id"])
            if replay["layout_fingerprint"] != case["layout_fingerprint"]:
                raise RuntimeError("actual rollout layout differs from preregistration")
            rows[-1].update(state="success" if result["success"] else "task_failure",
                            result=result,returncode=process.returncode,
                            total_wall_seconds=time.perf_counter()-started)
            atomic_json(attempts_path,rows)
            print(json.dumps(rows[-1]),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
