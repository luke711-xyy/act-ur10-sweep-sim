"""Safe explicit entry for the residual pilot; default is read-only inspection."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def supervise(process,root,started_at):
    """A wall watchdog also covers a stuck resource monitor or evidence write."""
    from residual_rl.config import DEFAULT
    from residual_rl.safety import worker_stop_signals
    from residual_rl.storage import previous_checkpoint,write_json
    root=Path(root); began=time.monotonic()
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL)
    def hard_wall(_number,_frame):
        if process.poll() is not None: raise TimeoutError('hard six-hour deadline; worker already exited')
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError as error:
            raise TimeoutError(f'hard six-hour deadline; worker already exited: {error}') from error
        raise TimeoutError('hard six-hour deadline; owned worker group killed')
    signal.signal(signal.SIGALRM,hard_wall)
    signal.setitimer(signal.ITIMER_REAL,max(.001,started_at+DEFAULT.max_wall_seconds-began))
    try:
        with worker_stop_signals() as parent_request:
            return _supervise(process,root,started_at,parent_request=parent_request)
    except BaseException as error:
        # Also cover failure while writing a stop request or sending its signal.
        forced=process.poll() is None
        if forced:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError as gone:
                error.add_note(f'owned worker group already exited: {gone}')
            process.wait(timeout=5.)
        if not (root/'supervisor_shutdown.json').exists():
            evidence,rejected=previous_checkpoint(root)
            try: initialized=json.loads((root/'checkpoint_state.json').read_text())['initialized']
            except (OSError,ValueError,KeyError,TypeError): initialized=None
            save=dict(status='previous_reused' if evidence else 'not_applicable' if initialized is False else 'failed',
                checkpoint=evidence['checkpoint'] if evidence else None,evidence=evidence,
                error=f'supervisor cleanup: {type(error).__name__}: {error}',
                learner_initialized=initialized,rejected_checkpoints=rejected,no_automatic_resume=True)
            write_json(root/'save_result.json',save)
            result=dict(state='supervisor_stopped',reason='supervisor_exception',forced=forced,
                worker_returncode=process.returncode,worker_acknowledged=False,save=save,
                monitor_error=f'{type(error).__name__}: {error}',complete=False,no_automatic_resume=True)
            write_json(root/'supervisor_shutdown.json',result)
            write_json(root/'status.json',result)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0.:
            signal.setitimer(signal.ITIMER_REAL,max(.001,previous_timer[0]-(time.monotonic()-began)),previous_timer[1])


def _supervise(process,root,started_at,*,parent_request=None):
    """Own only the worker session created below; reserve cleanup within hard caps."""
    from residual_rl.config import DEFAULT
    from residual_rl.safety import BudgetGuard,STOP_RESERVE_SECONDS,resources
    from residual_rl.storage import checkpoint_evidence,previous_checkpoint,write_json
    root=Path(root); guard=BudgetGuard(started_at=started_at)
    reason=None; requested_at=None; forced=False; monitor_error=None
    deadline=started_at+DEFAULT.max_wall_seconds
    pending_error=None
    while process.poll() is None:
        try:
            sample=resources(root,pid=process.pid); now=time.monotonic()
            hard=guard.check(now=now,residual_steps=0,**sample,reserve=False)
            soft=guard.check(now=now,residual_steps=0,**sample)
            if parent_request is not None and parent_request.reason:
                soft=parent_request.reason
        except BaseException as error:
            # Monitoring failure cannot safely grant an unmonitored save window.
            monitor_error=f'{type(error).__name__}: {error}'
            pending_error=error; hard='supervisor_monitor_failed'; soft=hard
        if hard not in ('continue','stop_collection') or time.monotonic()>=deadline:
            reason=hard if hard not in ('continue','stop_collection') else 'stop_cleanup_deadline'
            forced=True
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError as error: monitor_error=f'worker group already exited: {error}'
            process.wait(timeout=5.)
            break
        if requested_at is None and soft not in ('continue','stop_collection'):
            reason=soft; requested_at=time.monotonic()
            deadline=min(deadline,requested_at+STOP_RESERVE_SECONDS-5.)
            write_json(root/'stop_request.json',dict(reason=reason,requested_at=requested_at,
                cleanup_deadline=deadline,hard_wall_deadline=started_at+DEFAULT.max_wall_seconds))
            try: os.kill(process.pid,signal.SIGTERM)
            except ProcessLookupError as error: monitor_error=f'worker already exited: {error}'
        try: time.sleep(min(.5,max(0.,deadline-time.monotonic())))
        except BaseException as error:
            # An interrupted supervisor still requests bounded cooperative cleanup.
            pending_error=error; reason='supervisor_interrupted'
            if requested_at is None:
                requested_at=time.monotonic(); deadline=min(deadline,requested_at+STOP_RESERVE_SECONDS-5.)
                write_json(root/'stop_request.json',dict(reason=reason,requested_at=requested_at,
                    cleanup_deadline=deadline,hard_wall_deadline=started_at+DEFAULT.max_wall_seconds))
                try: os.kill(process.pid,signal.SIGTERM)
                except ProcessLookupError as gone: monitor_error=f'worker already exited: {gone}'
    acknowledged=False; failure=None
    try:
        save=json.loads((root/'save_result.json').read_text())
        if save['status'] in ('succeeded','previous_reused'):
            if checkpoint_evidence(save['checkpoint'])!=save['evidence']:
                raise ValueError('worker save acknowledgement evidence mismatch')
        elif save['status'] not in ('failed','not_applicable'):
            raise ValueError('unknown worker save outcome')
        acknowledged=True
    except (OSError,ValueError,KeyError,TypeError) as error:
        failure=f'{type(error).__name__}: {error}'
        evidence,rejected=previous_checkpoint(root)
        try: initialized=json.loads((root/'checkpoint_state.json').read_text())['initialized']
        except (OSError,ValueError,KeyError,TypeError): initialized=None
        save=dict(status=('previous_reused' if evidence else
                          'not_applicable' if initialized is False else 'failed'),
            checkpoint=evidence['checkpoint'] if evidence else None,evidence=evidence,
            error='worker did not acknowledge shutdown save: '+failure,
            learner_initialized=initialized,rejected_checkpoints=rejected,no_automatic_resume=True)
        write_json(root/'save_result.json',save)
    result=dict(state='supervisor_stopped' if reason else 'worker_exited',reason=reason,
        forced=forced,worker_returncode=process.returncode,worker_acknowledged=acknowledged,
        save=save,monitor_error=monitor_error,no_automatic_resume=True)
    write_json(root/'supervisor_shutdown.json',result)
    if reason or not acknowledged:
        write_json(root/'status.json',dict(result,complete=False))
    if pending_error is not None: raise pending_error
    return result


def main(argv=None):
    invoked_at=time.monotonic()
    from residual_rl.config import DEFAULT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inspect','prepare','preflight','pilot'),default='inspect')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--acknowledge-pilot-training',action='store_true',
                        help='Explicitly authorize this single six-hour maximum pilot invocation')
    parser.add_argument('--acknowledge-preflight',action='store_true',
                        help='Explicitly authorize actual ACT preflight, without learner updates')
    parser.add_argument('--owned-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.mode=='inspect':
        from residual_rl.runtime import audit_frozen_sources
        print(json.dumps(dict(execution_started=False,configuration=DEFAULT.to_dict(),
            requires_explicit_execution_authorization=True,
            baseline_source_audit=audit_frozen_sources(),
            full_240_rollout_comparison_included=False),indent=2)); return 0
    if args.mode=='pilot' and not args.acknowledge_pilot_training:
        parser.error('pilot requires --acknowledge-pilot-training; nothing started')
    if args.mode=='preflight' and not args.acknowledge_preflight:
        parser.error('preflight requires --acknowledge-preflight; nothing started')
    if args.out is None: parser.error('--out must be a new isolated output directory')
    from residual_rl.pilot import prepare,execute,verify_run
    from residual_rl.storage import RunStore,write_json
    if args.owned_worker:
        marker=json.loads((args.out/'started.json').read_text())
        if marker['parent_pid']!=os.getppid() or os.environ.get('RESIDUAL_PILOT_OWNER')!=marker['token']:
            raise RuntimeError('worker lacks the owning supervisor')
        store=RunStore(args.out)
        from residual_rl.safety import worker_stop_signals
        with worker_stop_signals() as request:
            execute(store,preflight_only=args.mode=='preflight',started_at=marker['monotonic_started'],
                    stop_request=request)
        return 0
    store=RunStore(args.out) if args.out.exists() else prepare(args.out)
    verify_run(store)
    if args.mode=='prepare':
        print(json.dumps(dict(prepared=str(args.out),execution_started=False))); return 0
    import secrets
    started=invoked_at; token=secrets.token_hex(24)
    marker=dict(parent_pid=os.getpid(),monotonic_started=started,token=token,mode=args.mode)
    with (args.out/'started.json').open('x') as stream: json.dump(marker,stream)
    env=dict(os.environ,RESIDUAL_PILOT_OWNER=token)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode',args.mode,
             '--out',str(args.out.resolve()),'--owned-worker']
    command += ['--acknowledge-pilot-training'] if args.mode=='pilot' else ['--acknowledge-preflight']
    write_json(args.out/'checkpoint_state.json',dict(initialized=False))
    process=subprocess.Popen(command,env=env,start_new_session=True)
    supervise(process,args.out,started)
    return process.returncode or 0


if __name__=='__main__':
    raise SystemExit(main())
