"""Bounded parallel collection of the frozen native-camera expert plan."""
import argparse
import fcntl
import json
import multiprocessing as mp
import os
import re
from multiprocessing.connection import wait
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_rudra_seed_matched import (
    capture_one,event,load_source_plan,read_jsonl,recovered_successes,verify_frozen_files,
    validate_training_manifest,write_json,terminal_failure_status,
)
from sim.act.image_spool import DiskBackedDatasetWriter
from sim.config import Config,load_config
from scripts.rudra_collection_attempts import CollectionAttempts


def worker_count(out,default):
    path=out/'worker-count.json'
    value=json.loads(path.read_text())['workers'] if path.exists() else default
    if type(value) is not int or not 1<=value<=4:
        raise ValueError('workers must be an integer between 1 and 4')
    return value


def capture_worker(configuration,slot,out,source_sha,camera_sha,connection):
    start=time.monotonic();out=Path(out)
    try:
        cfg=Config(configuration)
        parent=out/'.image-spool';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{slot['episode_id']}-pid{os.getpid()}-",dir=parent) as scratch:
            cfg.set_path('act.image_spool_dir',scratch)
            result=capture_one(cfg,slot,out,DiskBackedDatasetWriter(str(out),image_format='webp-lossless'),
                               source_sha,camera_sha)
        connection.send(dict(result=result,elapsed_seconds=time.monotonic()-start,
                             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except BaseException:
        connection.send(dict(error=traceback.format_exc(),episode_id=slot['episode_id']))
        raise
    finally:
        connection.close()


def cleanup_spools(out,pid=None):
    parent=out/'.image-spool'
    if not parent.exists():return
    if parent.is_symlink():raise ValueError('image scratch root must not be a symlink')
    for path in parent.iterdir():
        match=re.fullmatch(r'preview_goal_[1-6]_[0-9]+-pid([0-9]+)-[a-z0-9_]+',path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            raise ValueError('unrecognized image scratch path; preserving it')
        owner=int(match.group(1))
        if pid is not None and owner!=pid:continue
        try:os.kill(owner,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'image scratch worker {owner} is still alive')
        shutil.rmtree(path)


def collect_parallel(cfg,plan,pinned,out,workers=2,worker_target=capture_worker,fill_missing=False):
    entries=plan['entries'];expected={r['episode_id']:r for r in entries}
    if len(expected)!=len(entries):raise ValueError('duplicate slots in fixed plan')
    source_sha=pinned['source_manifest_sha256'];camera_sha=pinned['scene_profile_sha256']
    events=read_jsonl(out/'events.jsonl')
    history=CollectionAttempts(entries,events) if fill_missing else None
    def recover():
        return recovered_successes(out,entries,source_sha,camera_sha,
                                   allowed_attempts=history.assignments if history else None)
    done=recover()
    failed={r['episode_id']:r for r in events if r['event']=='episode_failed'}
    failed={key:r for key,r in failed.items() if key not in done}
    for key,r in failed.items():
        slot=expected.get(key)
        if slot is None or (history is None and any(r.get(k)!=slot[k] for k in ['seed','target_count'])):
            raise ValueError('failure identity differs from frozen plan')
    queue=([r for r in entries if r['episode_id'] in failed] if fill_missing else [])
    queue+=[r for r in entries if r['episode_id'] not in done and r['episode_id'] not in failed]
    active=[];ctx=mp.get_context('spawn');start=time.monotonic()

    def progress(state='running'):
        status=dict(state=state,saved=len(done),processed=len(done)+len(failed),failed=len(failed),
                    total=len(entries),mode='parallel_native_640x480_fill' if fill_missing else 'parallel_native_640x480',
                    workers=worker_count(out,workers),active=[a['slot'] for a in active],
                    elapsed_seconds=time.monotonic()-start,
                    per_target={str(n):sum(r['target_count']==n for r in done.values()) for n in range(1,7)})
        if history:
            status.update(failed_attempts=len(history.failures),remaining=len(entries)-len(done))
        write_json(out/'progress.json',status)
        return status

    try:
        while queue or active:
            limit=worker_count(out,workers)
            waiting_for_space=False
            while queue and len(active)<limit:
                verify_frozen_files(out,pinned['files'])
                if shutil.disk_usage(out).free<12*1024**3:
                    waiting_for_space=True
                    break
                slot=queue.pop(0)
                if slot['episode_id'] in done:continue
                if history:
                    slot=history.next_attempt(slot['episode_id'])
                    if slot['seed'] not in history.assignments[slot['episode_id']]:
                        event(out,'attempt_allocated',**slot)
                        history.register(slot)
                receiving,sending=ctx.Pipe(duplex=False)
                process=ctx.Process(target=worker_target,args=(cfg.to_dict(),slot,str(out),source_sha,camera_sha,sending))
                process.start();sending.close()
                active.append(dict(slot=slot,process=process,connection=receiving))
                event(out,'episode_started',**slot,worker_pid=process.pid,
                      mode='parallel_native_640x480_fill' if fill_missing else 'parallel_native_640x480')
            progress('waiting_for_disk_space' if waiting_for_space else 'running')
            if not active:
                if waiting_for_space:time.sleep(10.)
                continue
            ready=wait([a['connection'] for a in active],timeout=1.)
            for item in list(active):
                process=item['process'];connection=item['connection'];slot=item['slot']
                if connection not in ready and process.is_alive():continue
                try:
                    message=connection.recv() if connection.poll() else None
                except EOFError:
                    message=None
                if not message or message.get('error'):
                    process.join(timeout=1.)
                    raise RuntimeError(f"worker failed for {slot['episode_id']} (exit={process.exitcode}): "
                                       +str((message or {}).get('error','no result received')))
                process.join(timeout=10.)
                if process.is_alive() or process.exitcode!=0:
                    raise RuntimeError(f"worker did not exit cleanly for {slot['episode_id']}")
                result=message['result']
                if any(result.get(k)!=slot[k] for k in ['episode_id','seed','target_count']):
                    raise RuntimeError('worker result identity differs from assigned slot')
                active.remove(item);connection.close()
                metrics={k:message[k] for k in ['elapsed_seconds','peak_rss_kib']}
                if result['state']=='saved':
                    done=recover()
                    if slot['episode_id'] not in done:
                        raise RuntimeError('worker reported saved before manifest commit')
                    event(out,'episode_saved',**result,**metrics)
                    failed={key:row for key,row in failed.items() if key not in done}
                elif result['state']=='failed':
                    failed[slot['episode_id']]=result
                    event(out,'episode_failed',**result,**metrics)
                    if history:
                        history.record_failure(result)
                        queue.insert(0,expected[slot['episode_id']])
                else:
                    raise RuntimeError('unknown worker outcome')
                progress()
    finally:
        for item in active:
            if item['process'].is_alive():item['process'].terminate()
        for item in active:
            process=item['process'];process.join(timeout=5.)
            if process.is_alive():process.kill();process.join(timeout=5.)
            item['connection'].close()
            if not process.is_alive():cleanup_spools(out,process.pid)
    state='needs_attention' if failed else 'complete'
    if fill_missing and len(done)!=len(entries):raise RuntimeError('successful slot quota is incomplete')
    if not failed:
        validate_training_manifest(out,episodes_per_target=50,camera_views=['top','wrist'],
                                   target_subset_mode='y_edge_contiguous')
    summary=progress(state);summary['failures']=list(failed.values())
    if history:summary['attempt_failures']=history.failures
    write_json(out/'summary.json',summary);write_json(out/'progress.json',summary)
    event(out,'collection_finished',**summary)
    return 2 if failed else 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--source-manifest',default='/home/luke/dataset-ur10/manifest.jsonl')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--fill-missing',action='store_true',help='replace known failed layout seeds until all 300 success slots are filled')
    args=parser.parse_args();out=Path(args.out).resolve()
    if not out.is_dir():raise ValueError('prepare the frozen output directory first')
    with (out/'.collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with terminal_failure_status(out):
            pinned=json.loads((out/'provenance.json').read_text());verify_frozen_files(out,pinned['files'])
            plan=json.loads((out/'plan.json').read_text())
            if load_source_plan(args.source_manifest)!=plan:raise ValueError('source plan changed')
            cfg=load_config(str(out/'collection-config.yaml'))
            if list(cfg.act.image_size)!=[640,480] or list(cfg.act.camera_views)!=['top','wrist']:
                raise ValueError('parallel native collection requires top+wrist 640x480')
            if int(cfg.planner.expert_validation_workers)!=1:
                raise ValueError('nested candidate workers must remain 1')
            cleanup_spools(out)
            write_json(out/'parallel-settings.json',dict(default_workers=args.workers,max_workers=4,
                      image_transport='pixel-exact uint8 disk memmap',image_size_wh=[640,480],
                      original_frozen_configuration_unchanged=True,fill_missing=args.fill_missing))
            if args.fill_missing and not (out/'completion-policy.json').exists():
                write_json(out/'completion-policy.json',dict(
                    authorized_by='user requested retry or seed replacement until targets 1-6 each have 50 successful demonstrations',
                    source_manifest_sha256=pinned['source_manifest_sha256'],
                    quotas={str(n):50 for n in range(1,7)},
                    known_failure_action='allocate next globally unused seed after original source seed',
                    success_stop_condition='300 successful unique slots and original training manifest validation',
                    unknown_execution_error_action='stop and preserve outputs',
                    original_plan_and_successes_unchanged=True))
            return collect_parallel(cfg,plan,pinned,out,args.workers,fill_missing=args.fill_missing)


if __name__=='__main__':
    raise SystemExit(main())
