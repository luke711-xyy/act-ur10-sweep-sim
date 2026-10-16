"""Explicit training-only 1M/72h runner. Default inspection never starts learning."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def main(argv=None):
    invoked_at=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('inspect','prepare','train','resume'),default='inspect')
    p.add_argument('--out',type=Path)
    p.add_argument('--acknowledge-training',action='store_true')
    p.add_argument('--owned-worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.mode in ('train','resume') and not args.acknowledge_training:p.error('training requires explicit acknowledgement')
    if args.mode!='inspect' and args.out is None:p.error('--out is required')
    from residual_rl.config import activate_long_training_profile
    config=activate_long_training_profile()
    if args.mode=='inspect':
        print(json.dumps(dict(execution_started=False,configuration=config.to_dict(),
            validation_layouts=0,preflight_executed=False,no_automatic_resume=True)))
        return 0
    from residual_rl.long_training import prepare,execute
    from residual_rl.pilot import verify_run
    from residual_rl.storage import RunStore,write_json
    resume=args.mode=='resume'
    marker_name='resume_started.json' if resume else 'started.json'
    if args.owned_worker:
        marker=json.loads((args.out/marker_name).read_text())
        if (marker['parent_pid']!=os.getppid() or os.environ.get('RESIDUAL_LONG_OWNER')!=marker['token']
                or marker['mode']!=('resume_long' if resume else 'train_long')):
            raise RuntimeError('worker lacks owning supervisor')
        from residual_rl.safety import worker_stop_signals
        with worker_stop_signals() as request:
            execute(RunStore(args.out),started_at=marker['monotonic_started'],stop_request=request,
                    resume=resume,attempt_token=marker['token'] if resume else None)
        return 0
    if resume and not args.out.is_dir():p.error('resume requires the existing run directory')
    store=RunStore(args.out) if args.out.exists() else prepare(args.out)
    verify_run(store)
    if args.mode=='prepare':
        print(json.dumps(dict(prepared=str(args.out),execution_started=False)));return 0
    token=secrets.token_hex(24)
    original_started=invoked_at
    if resume:
        from residual_rl.resume import verify_repair,load_recovery
        verify_repair(args.out)
        if json.loads((args.out/'status.json').read_text())['state']!='needs_attention':
            raise RuntimeError('resume requires the reviewed stopped boundary')
        original_started=float(json.loads((args.out/'started.json').read_text())['monotonic_started'])
        if not 0<=invoked_at-original_started<config.max_wall_seconds-60.:
            raise RuntimeError('original wall-clock budget expired or boot clock changed')
        # Validate weights, ledger, arrays, metrics and sampler BEFORE consuming
        # an exclusive resume marker or replacing launch/progress metadata.
        validated=load_recovery(args.out)
        del validated
    marker=dict(parent_pid=os.getpid(),monotonic_started=original_started,token=token,
                mode='resume_long' if resume else 'train_long')
    with (args.out/marker_name).open('x') as stream:json.dump(marker,stream)
    if not resume:write_json(args.out/'checkpoint_state.json',dict(initialized=False))
    env=dict(os.environ,RESIDUAL_LONG_OWNER=token)
    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--mode',args.mode,
        '--out',str(args.out.resolve()),'--acknowledge-training','--owned-worker'],env=env,start_new_session=True)
    write_json(args.out/'launch.json',dict(pid=os.getpid(),worker_pid=child.pid,started_at=time.time(),
        run=str(args.out.resolve()),command='scripts/train_residual_long.py',mode=marker['mode'],
        original_started_at=original_started))
    spec=importlib.util.spec_from_file_location('_long_supervisor',PROJECT/'scripts/train_residual_sac.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    supervisor.supervise(child,args.out,original_started,attempt_token=token if resume else None)
    return child.returncode or 0


if __name__=='__main__':raise SystemExit(main())
