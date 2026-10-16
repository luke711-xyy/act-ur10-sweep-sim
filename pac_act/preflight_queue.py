"""One authorized preflight after a verified paired batch; never a trainer."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .storage import RunStore,read_rows,sha256,write_json


def dependency_pids(root):
    import psutil
    root=str(Path(root).resolve());found=[]
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid():continue
        try:command=process.info['cmdline'] or []
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        execution=any(Path(arg).name in ('runner.py','evaluate_new_schedule_batch.py')
            or Path(arg).name.startswith('ffmpeg') for arg in command)
        if execution and any(arg==root or arg.startswith(root+os.sep) for arg in command):found.append(process.pid)
    return found


def local_slot_available():
    from .safety import local_inference_available
    return local_inference_available()[0]


def dependency_ready(root,*,active_pids):
    """Small status polling; media hashing happens only after process exit."""
    root=Path(root);status=json.loads((root/'status.json').read_text())
    if status.get('state')=='needs_attention':raise RuntimeError('dependency needs_attention; no automatic recovery')
    if status.get('state')!='complete':
        if not active_pids:raise RuntimeError('dependency exited before verified completion')
        return None
    if active_pids:return None
    metadata=json.loads((root/'metadata.json').read_text())
    if (metadata.get('label')!='new_schedule_no_noise_100k' or metadata.get('training') is not False
            or metadata.get('noise') is not False or metadata.get('checkpoint_step')!=100000):
        raise RuntimeError('wrong dependency protocol')
    plan_path=root/'episode_plan.json'
    if sha256(plan_path)!=metadata['plan_sha256']:raise RuntimeError('dependency plan changed')
    plan=json.loads(plan_path.read_text());planned={r['seed']:r for r in plan}
    rows=read_rows(root/'episodes.jsonl');seeds={r['seed'] for r in rows}
    if (len(plan)!=120 or len(planned)!=120 or seeds!=set(range(2120000,2120120)) or len(rows)!=120
            or Counter(r['target_count'] for r in rows)!=Counter({n:20 for n in range(1,7)})):
        raise RuntimeError('dependency has incomplete/duplicate/unpaired result ledger')
    summary=json.loads((root/'summary.json').read_text())
    if (summary.get('complete') is not True or summary.get('episodes')!=120
            or summary.get('media_pending') is not False or summary.get('verified_videos')!=120
            or summary.get('training_updates')!=0 or status.get('completed')!=120
            or summary.get('strict_successes')!=sum(r.get('success') is True for r in rows)):
        raise RuntimeError('dependency summary/status/media disagree')
    keys={RunStore.key(r) for r in rows}
    if {p.stem for p in (root/'attempts').glob('*.json')}!=keys:raise RuntimeError('dependency orphan/duplicate opportunity')
    for row in rows:
        expected=planned.get(row['seed']);key=RunStore.key(row);folder=root/'episodes'/key
        if (expected is None or row['role']!='comparison' or row['arm']!='new_schedule_zero'
                or row['target_count']!=expected['target_count'] or row['paired_original_key']!=expected['key']
                or row.get('layout_fingerprint')!=expected['layout_fingerprint']
                or row.get('outcome') not in ('success','failure','safety') or type(row.get('success')) is not bool
                or row.get('opportunities')!=1 or row.get('planning_calls') not in (0,1)
                or row.get('physical_attempts') not in (0,1)):
            raise RuntimeError('dependency pairing/one-shot result differs')
        if json.loads((folder/'result.json').read_text())!=row:raise RuntimeError('dependency result differs from ledger')
        marker=json.loads((root/'attempts'/f'{key}.json').read_text())
        if any(row.get(k)!=v for k,v in marker.items()):raise RuntimeError('dependency attempt marker differs')
        media=json.loads((folder/'media.json').read_text());video=folder/media['file'];verified=media.get('verification',{})
        if (video.resolve().parent!=folder.resolve() or video.is_symlink()
                or video.stat().st_size!=media['bytes'] or sha256(video)!=media['sha256']
                or verified.get('frames',0)<=0 or verified.get('sha256')!=media['sha256']
                or verified.get('bytes')!=media['bytes']):
            raise RuntimeError('dependency retained media is unverified or changed')
    return dict(episodes=120,verified_videos=120,ledger_sha256=sha256(root/'episodes.jsonl'),
        summary_sha256=sha256(root/'summary.json'),metadata_sha256=sha256(root/'metadata.json'),
        status_sha256=sha256(root/'status.json'),plan_sha256=sha256(plan_path),strict_successes=summary['strict_successes'])


def run_queue(bundle,after_batch,out,queue,*,start_index=120,device='cpu',poll_seconds=30.):
    bundle,after_batch,out,queue=map(lambda p:Path(p).resolve(),(bundle,after_batch,out,queue))
    if type(start_index) is not int or start_index<6 or start_index%6 or start_index>=1_000_000-5:
        raise ValueError('new preflight needs a fresh six-aligned registered start index')
    if device!='cpu':raise ValueError('authorized acceptance queue is CPU-only')
    def overlap(a,b):return a==b or a.is_relative_to(b) or b.is_relative_to(a)
    if overlap(out,queue) or any(overlap(p,protected) for p in (out,queue) for protected in (bundle,after_batch)):
        raise ValueError('preflight/queue destinations must not overlap each other, bundle or running batch')
    if out.exists():raise FileExistsError('preflight output must be new; no automatic resume')
    queue.mkdir(parents=True,exist_ok=False)
    def publish(state,**extra):
        value=dict(state=state,pid=os.getpid(),time=time.time(),training_started=False,**extra)
        write_json(queue/'status.json',value);return value
    try:
        started=time.time();deadline=started+24*3600
        protected={p:sha256(p) for p in (bundle/'manifest.json',after_batch/'metadata.json',after_batch/'episode_plan.json')}
        identity=dict(bundle=str(bundle),after_batch=str(after_batch),out=str(out),start_index=start_index,
            device=device,queued_at=started,queue_deadline=deadline,waiting_counts_against_preflight_budget=False,
            preflight_limits=dict(hours=2,opportunities=24,output_gib=5,free_gib=8),formal_training_authorized=False,
            inference_lock_paths=os.environ.get('PAC_ACT_INFERENCE_LOCKS','').split(os.pathsep),
            protected_sha256={str(p):h for p,h in protected.items()})
        write_json(queue/'authorization.json',identity)
        while True:
            if time.time()>=deadline:raise RuntimeError('queue deadline reached; no automatic extension')
            if any(sha256(p)!=h for p,h in protected.items()):raise RuntimeError('queued identity changed')
            active=dependency_pids(after_batch)
            status=json.loads((after_batch/'status.json').read_text())
            if status.get('state')=='complete' and not active and not local_slot_available():
                publish('waiting_for_local_slot');time.sleep(poll_seconds);continue
            ready=dependency_ready(after_batch,active_pids=active)
            if ready is None:
                publish('waiting_for_120_batch',dependency_pids=active);time.sleep(poll_seconds);continue
            # Every wait comes back through fresh status/results/media/process
            # validation. Never reuse a certificate from before a slot wait.
            if dependency_pids(after_batch) or not local_slot_available():
                publish('waiting_for_clear_handoff');time.sleep(poll_seconds);continue
            break
        write_json(queue/'dependency_verified.json',ready)
        if any(sha256(p)!=h for p,h in protected.items()):raise RuntimeError('queued identity changed before launch')
        stable={'episodes.jsonl':'ledger_sha256','summary.json':'summary_sha256','status.json':'status_sha256'}
        if dependency_pids(after_batch) or any(sha256(after_batch/name)!=ready[key] for name,key in stable.items()):
            raise RuntimeError('dependency changed immediately before preflight; no automatic launch')
        command=[sys.executable,'-B',str(bundle/'source/scripts/train_pac_act.py'),'--mode','preflight',
            '--bundle',str(bundle),'--out',str(out),'--device',device,'--preflight-start-index',str(start_index)]
        with (queue/'preflight_started.json').open('x') as stream:
            json.dump(dict(command=command,time=time.time(),authorized_once=True),stream,indent=2)
            stream.flush();os.fsync(stream.fileno())
        publish('preflight_running')
        with (queue/'preflight.log').open('xb') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:raise RuntimeError(f'preflight exited {completed.returncode}; no automatic retry')
        report=json.loads((out/'acceptance.json').read_text())
        if (report.get('engineering_passed') is not True or report.get('training_started') is not False
                or report.get('GPU_training_started') is not False):raise RuntimeError('preflight did not pass engineering checks')
        return publish('preflight_complete',engineering_passed=True,report=str(out/'acceptance.json'))
    except BaseException as error:
        try:
            write_json(queue/'status.json',dict(state='needs_attention',error=f'{type(error).__name__}: {error}',
                original_exception_repr=repr(error),
                automatic_retry=False,training_started=False,pid=os.getpid()))
        except BaseException as reporting:error.add_note('queue failure report: '+repr(reporting))
        raise
