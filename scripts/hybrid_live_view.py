"""External, read-only display adapter. Never imports or calls the evaluator.

An open read descriptor survives the evaluator's authorized unlink. Only after
the official ledger confirms full video verification is that descriptor copied
to the display directory. No controller, source video, or formal record is edited.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import shlex
import tempfile
import threading
import time


def atomic_json(path, data):
    if path.parent.is_symlink(): raise ValueError('display destination symlink')
    payload=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False).encode()
    fd,name=tempfile.mkstemp(prefix='.display-',dir=path.parent)
    temp=Path(name)
    with os.fdopen(fd,'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def read_rows(path):
    if not path.exists(): return []
    text=path.read_text()
    lines=text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'): lines.pop()
    try: return [json.loads(line) for line in lines if line.strip()]
    except json.JSONDecodeError as error: raise ValueError('malformed official ledger') from error


def count_matrix(rows):
    values=[[0]*6 for _ in range(7)]; totals=[0]*6
    for row in rows:
        target,actual=row['target_count'],row['final_collected']
        if type(target) is not int or type(actual) is not int or not 1<=target<=6 or not 0<=actual<=6:
            raise ValueError('unknown or out-of-range count; do not invent zero')
        values[actual][target-1]+=1; totals[target-1]+=1
    return values,totals


def render_charts(root,rows,planned):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    counts,totals=count_matrix(rows)
    probabilities=np.divide(np.array(counts,dtype=float),np.array(totals),
                            out=np.zeros((7,6)),where=np.array(totals)>0)
    fig,axis=plt.subplots(figsize=(8.5,7))
    picture=axis.imshow(probabilities,origin='lower',cmap='magma',vmin=0,vmax=1,
                        extent=(.5,6.5,-.5,6.5),aspect='auto')
    for target in range(1,7):
        for actual in range(7):
            value=probabilities[actual,target-1]
            label=f'{value:.0%}\n{counts[actual][target-1]}/{totals[target-1]}' if totals[target-1] else '--'
            axis.text(target,actual,label,ha='center',va='center',fontsize=10,
                      color='black' if value>.65 else 'white')
        axis.add_patch(Rectangle((target-.5,target-.5),1,1,fill=False,edgecolor='#ddf4fa',linewidth=2))
    success=sum(row['success'] for row in rows)
    exact=sum(row['final_collected']==row['target_count'] for row in rows)
    axis.set(xticks=range(1,7),yticks=range(7),xlabel='Target count',
             ylabel='Final retained fully collected count',
             title=f'Hybrid A* -> ACT | trial 4 | step 100K\nStrict success {success}/{len(rows)}; exact final count {exact}/{len(rows)}')
    label='FINAL' if len(rows)==planned else 'PROVISIONAL'
    fig.suptitle(f'{label} {len(rows)}/{planned} paired episodes',fontsize=16)
    fig.colorbar(picture,ax=axis,label='Frequency within each tested target')
    fig.text(.5,.012,'Final-count match is not strict success. Same paired layouts; seed 2120018 used in tuning.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ('png','svg'):
        path=root/f'final_count_density.{suffix}'
        fd,name=tempfile.mkstemp(prefix='.display-',dir=root); temp=Path(name)
        with os.fdopen(fd,'wb') as stream: fig.savefig(stream,format=suffix,dpi=180)
        temp.replace(path)
    plt.close(fig)
    # This conventional preview filename is explicitly FINAL count, not peak count.
    fd,name=tempfile.mkstemp(prefix='.display-',dir=root); temp=Path(name)
    with os.fdopen(fd,'wb') as stream:
        stream.write((root/'final_count_density.png').read_bytes())
    temp.replace(root/'count_density.png')
    path=root/'density_matrix.csv'; fd,name=tempfile.mkstemp(prefix='.display-',dir=root); temp=Path(name)
    with os.fdopen(fd,'w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream); writer.writerow(['target','actual_final','frequency','tested','percentage'])
        for target in range(1,7):
            for actual in range(7):
                count=counts[actual][target-1]; total=totals[target-1]
                writer.writerow([target,actual,count,total,count/total if total else ''])
    temp.replace(path)


def source_path(root,path):
    root=root.resolve(); path=Path(path)
    try: relative=path.relative_to(root)
    except ValueError as error: raise ValueError('video outside batch') from error
    parts=relative.parts
    if (len(parts)!=4 or parts[0]!='episodes' or not parts[1].startswith('seed_')
        or not parts[1][5:].isdigit() or parts[2]!='videos' or path.suffix!='.mp4'
        or any(p.is_symlink() for p in [path,*path.parents] if p!=root.parent)):
        raise ValueError('not a canonical generated episode video')
    if path.resolve()!=path: raise ValueError('noncanonical video path')
    return parts[1]


def capture_video(root,path):
    source_path(root,path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd); raise ValueError('video is not a regular file')
    return os.fdopen(fd,'rb')


def pending_capture(pending,path):
    return next((candidate for candidate in pending if candidate.parent==path.parent),None)


def retain_video(root,path,stream,verification):
    seed=source_path(root,path)
    if not isinstance(verification,dict) or not verification.get('sha256') or not verification.get('bytes'):
        raise ValueError('full source verification is required')
    destination=root/'videos'
    if destination.is_symlink(): raise ValueError('display directory is a symlink')
    destination.mkdir(exist_ok=True)
    target=destination/f'{seed}.mp4'
    if target.exists(): raise ValueError('do not overwrite a display video')
    fd,name=tempfile.mkstemp(prefix='.display-copy-',dir=destination);temp=Path(name)
    hasher=hashlib.sha256(); size=0; stream.seek(0)
    with os.fdopen(fd,'wb') as output:
        for block in iter(lambda:stream.read(1024*1024),b''):
            output.write(block); hasher.update(block); size+=len(block)
        output.flush(); os.fsync(output.fileno())
    if size!=verification['bytes'] or hasher.hexdigest()!=verification['sha256']:
        temp.unlink(); raise ValueError('copied bytes do not match validated source')
    identity=temp.stat()
    entry={'seed':int(seed[5:]),'path':str(target),'source':str(path),'bytes':size,
            'sha256':hasher.hexdigest(),'device':identity.st_dev,'inode':identity.st_ino,
            'source_full_decode_verified':True,'retained_at':time.time()}
    receipts=root/'live_view'/'copy_receipts'
    if receipts.is_symlink() or receipts.parent.is_symlink():raise ValueError('copy receipt directory symlink')
    receipts.mkdir(parents=True,exist_ok=True)
    atomic_json(receipts/f'{seed}.json',entry)
    # Exclusive publication cannot overwrite a concurrently created manual file.
    os.link(temp,target); temp.unlink()
    return entry


def validate_copies(root,entries):
    for entry in entries:
        path=Path(entry['path']); expected=root/'videos'/f"seed_{entry['seed']}.mp4"
        if path!=expected or path.is_symlink() or path.parent.is_symlink():
            raise ValueError('invalid display copy path')
        identity=path.stat()
        if (identity.st_nlink!=1 or identity.st_size!=entry['bytes'] or digest(path)!=entry['sha256']
            or identity.st_ino!=entry.get('inode',identity.st_ino)
            or identity.st_dev!=entry.get('device',identity.st_dev)):
            raise ValueError('display copy drift')


def reconcile_copies(root,state,rows):
    validate_copies(root,state['copies'])
    receipts=root/'live_view'/'copy_receipts'
    if not receipts.exists():return
    if receipts.is_symlink():raise ValueError('copy receipt directory symlink')
    lookup={row['seed']:row for row in rows};retained={entry['seed'] for entry in state['copies']}
    for path in receipts.iterdir():
        if path.is_symlink() or path.suffix!='.json':raise ValueError('invalid copy receipt')
        entry=json.loads(path.read_text())
        if entry['seed'] in retained or not Path(entry['path']).exists():continue
        row=lookup.get(entry['seed'])
        if row is None:raise ValueError('copy receipt without official result')
        verification=row['media_validation']['video_verification']
        if (entry['source']!=row['historical_video_path'] or entry['bytes']!=verification['bytes']
            or entry['sha256']!=verification['sha256']):raise ValueError('copy receipt identity drift')
        validate_copies(root,[entry]);state['copies'].append(entry);retained.add(entry['seed'])


def cleanup_copies(root,entries):
    validate_copies(root,entries)
    report={'deleted_files':len(entries),'freed_bytes':sum(e['bytes'] for e in entries),
            'permanent':True,'files':entries,'checked_at':time.time(),'complete':False}
    evidence=root/'live_view'; evidence.mkdir(exist_ok=True)
    atomic_json(evidence/'cleanup.json',report)
    if entries:
        directory=os.open(root/'videos',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            checked=[]
            for entry in entries:
                name=Path(entry['path']).name
                fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                with os.fdopen(fd,'rb') as stream:
                    identity=os.fstat(stream.fileno()); value=hashlib.sha256()
                    for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
                if identity.st_nlink!=1 or identity.st_size!=entry['bytes'] or value.hexdigest()!=entry['sha256']:
                    raise ValueError('display copy changed during cleanup')
                checked.append((name,identity.st_dev,identity.st_ino))
            for name,device,inode in checked:
                identity=os.stat(name,dir_fd=directory,follow_symlinks=False)
                if not stat.S_ISREG(identity.st_mode) or (identity.st_dev,identity.st_ino)!=(device,inode):
                    raise ValueError('display copy replaced during cleanup')
                os.unlink(name,dir_fd=directory)
        finally:os.close(directory)
    report.update(complete=True,completed_at=time.time())
    atomic_json(evidence/'cleanup.json',report)
    return report


def batch_finished(status,summary,count,parent_alive):
    return (count==120 and status.get('state')=='complete' and status.get('completed')==120
            and summary.get('complete') is True and summary.get('completed')==120 and not parent_alive)


def validate_rows(rows,plan):
    lookup={row['key']:row for row in plan}; seen=set()
    for row in rows:
        spec=lookup.get(row['key'])
        if spec is None or row['key'] in seen: raise ValueError('unpaired or duplicate official row')
        seen.add(row['key'])
        for field in ('seed','target_count','layout_fingerprint'):
            if row[field]!=spec[field]: raise ValueError('paired identity mismatch')
        if row.get('opportunities')!=1 or row.get('physical_attempts') not in (0,1) or row.get('media_complete') is not True:
            raise ValueError('invalid official opportunity or media state')
    count_matrix(rows)


def command_matches_batch(command,root):
    try: args=shlex.split(command)
    except ValueError: return True if 'evaluate_hybrid_paired_batch.py' in command else False
    if not any(Path(arg).name=='evaluate_hybrid_paired_batch.py' for arg in args): return False
    out=next((arg[6:] for arg in args if arg.startswith('--out=')),None)
    if out is None and '--out' in args:
        index=args.index('--out'); out=args[index+1] if index+1<len(args) else None
    # Unknown/relative cwd is conservatively live, never false completion.
    return out is None or not Path(out).is_absolute() or Path(out).resolve()==root.resolve()


def processes_alive(root):
    output=subprocess.check_output(['ps','-axo','command='],text=True)
    return any(command_matches_batch(line,root) for line in output.splitlines())


def capture_loop(root,seeds,pending,mutex,stop,errors):
    try:
        while not stop.is_set():
            status=json.loads((root/'status.json').read_text())
            seed=status.get('seed')
            if seed in seeds:
                directory=root/'episodes'/f'seed_{seed}'/'videos'
                if directory.is_symlink():raise ValueError('source video directory symlink')
                if directory.exists():
                    with mutex:
                        for path in directory.iterdir():
                            if path.suffix=='.mp4' and pending_capture(pending,path) is None:
                                try:pending[path]=capture_video(root,path)
                                except FileNotFoundError:pass
            stop.wait(.1)
    except BaseException as error:errors.append(error)


def run(root,resume=False):
    root=root.resolve(); evidence=root/'live_view'
    if root.is_symlink() or evidence.is_symlink(): raise ValueError('noncanonical display root')
    plan=json.loads((root/'episode_plan.json').read_text())
    if len(plan)!=120: raise ValueError('only this 120-row batch is in scope')
    metadata=json.loads((root/'metadata.json').read_text())
    seal=json.loads((root/'seal.json').read_text())
    if digest(root/'metadata.json')!=seal['metadata_sha256']: raise ValueError('metadata seal mismatch')
    formal={Path(path):value for path,value in metadata['frozen_sha256'].items() if Path(path).stat().st_size<2*1024*1024}
    project=Path(__file__).resolve().parents[1]
    formal.update({project/path:value for path,value in metadata['source_fingerprints'].items()})
    formal[root/'metadata.json']=seal['metadata_sha256']
    for path,value in formal.items():
        if digest(path)!=value: raise ValueError(f'frozen input mismatch: {path}')
    descriptor=os.open(evidence/'display.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=evidence/'manifest.json'
        rows=read_rows(root/'episodes.jsonl'); validate_rows(rows,plan)
        if journal.exists():
            if not resume:raise ValueError('display already started; explicit display-only resume required')
            state=json.loads(journal.read_text());reconcile_copies(root,state,rows)
            if state['state']=='complete_copies_cleaned':raise ValueError('display already complete')
            state.update(pid=os.getpid(),state='running',display_only_resumed_at=time.time())
        else:state={'kind':'external display only','pid':os.getpid(),'started_at':time.time(),
               'starting_completed':len(rows),'formal_inputs_unchanged':True,
               'retention':'new verified video copies until complete batch then clean own copies',
               'early_videos_already_deleted_not_regenerated':[r['seed'] for r in rows],
               'copies':[],'unavailable_later_videos':[],'state':'running'}
        pending={}; mutex=threading.RLock(); stop=threading.Event(); errors=[]
        capture=threading.Thread(target=capture_loop,args=(root,{p['seed'] for p in plan},pending,mutex,stop,errors),daemon=True)
        last=-1; last_checked=0
        atomic_json(journal,state)
        capture.start()
        try:
            while True:
                if errors:raise errors[0]
                status=json.loads((root/'status.json').read_text())
                now=time.time()
                if now-last_checked>=1:
                    rows=read_rows(root/'episodes.jsonl'); validate_rows(rows,plan)
                    retained={e['seed'] for e in state['copies']}
                    for row in rows:
                        if row['seed'] in retained or row['seed'] in state['early_videos_already_deleted_not_regenerated']: continue
                        if row.get('physical_attempts')==0 or not row.get('historical_video_path'): continue
                        path=Path(row['historical_video_path'])
                        with mutex:
                            captured=pending_capture(pending,path)
                            if captured is None and path.exists():
                                pending[path]=capture_video(root,path); captured=path
                            stream=pending[captured] if captured is not None else None
                        if captured is not None:
                            if shutil.disk_usage(root).free<6*2**30: raise RuntimeError('display copy suspended: free disk below 6 GiB')
                            entry=retain_video(root,path,stream,row['media_validation']['video_verification'])
                            state['copies'].append(entry);atomic_json(journal,state)
                            with mutex:pending.pop(captured).close()
                            print(json.dumps({'event':'video_retained','seed':row['seed'],'bytes':entry['bytes']}),flush=True)
                        elif row['seed'] not in state['unavailable_later_videos']:
                            state['unavailable_later_videos'].append(row['seed'])
                            print(json.dumps({'event':'video_unavailable_without_rerun','seed':row['seed']}),flush=True)
                    if len(rows)!=last:
                        render_charts(root,rows,120); last=len(rows)
                        state.update(completed=len(rows),updated_at=time.time())
                        atomic_json(journal,state)
                        print(json.dumps({'event':'heatmap_updated','completed':len(rows)}),flush=True)
                    last_checked=now
                    if status.get('state')=='needs_attention':
                        state.update(state='batch_needs_attention_copies_preserved',updated_at=time.time())
                        atomic_json(journal,state)
                    if status.get('state')=='complete':
                        summary=json.loads((root/'summary.json').read_text())
                        if batch_finished(status,summary,len(rows),processes_alive(root)):
                            cleanup_copies(root,state['copies'])
                            state.update(state='complete_copies_cleaned',updated_at=time.time())
                            atomic_json(journal,state); return
                time.sleep(.1)
        except BaseException as error:
            state.update(state='display_needs_attention_batch_untouched',error=repr(error),updated_at=time.time())
            atomic_json(journal,state); raise
        finally:
            stop.set(); capture.join(timeout=2)
            for stream in pending.values(): stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    parser.add_argument('--resume-display-only',action='store_true')
    args=parser.parse_args();run(Path(args.root),resume=args.resume_display_only)
