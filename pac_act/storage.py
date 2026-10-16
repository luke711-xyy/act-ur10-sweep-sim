"""Append-only results and bounded, verified, atomic learning snapshots."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def numeric(value):
    if hasattr(value,'tolist'): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def write_json(path,value):
    path=Path(path); payload=json.dumps(value,default=numeric,allow_nan=False,indent=2)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('x') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path); fsync_dir(path.parent)
    finally:
        if tmp.exists(): tmp.unlink()


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def append_json(path,value):
    payload=json.dumps(value,default=numeric,allow_nan=False)+'\n'
    path=Path(path)
    if path.exists(): preserve_incomplete_tail(path)
    with path.open('a') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())


def preserve_incomplete_tail(path):
    """Single writer under the experiment lease; retain crash evidence first."""
    with Path(path).open('r+b') as stream:
        stream.seek(0,2); end=stream.tell()
        if not end: return
        stream.seek(end-1)
        if stream.read(1)==b'\n': return
        offset=end; chunks=[]; boundary=0
        while offset:
            start=max(0,offset-65536); stream.seek(start); chunk=stream.read(offset-start)
            newline=chunk.rfind(b'\n')
            if newline>=0:
                boundary=start+newline+1; chunks.append(chunk[newline+1:]); break
            chunks.append(chunk); offset=start
        evidence=Path(path).with_name(Path(path).name+'.incomplete.'+uuid.uuid4().hex)
        with evidence.open('xb') as saved:
            for chunk in reversed(chunks): saved.write(chunk)
            saved.flush(); os.fsync(saved.fileno())
        fsync_dir(Path(path).parent)
        stream.truncate(boundary); stream.flush(); os.fsync(stream.fileno())


def iter_rows(path):
    path=Path(path)
    if not path.exists(): return
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'): break
            if line.strip(): yield json.loads(line)


def read_rows(path): return list(iter_rows(path))


def audit_identity(audit):
    if not audit: return None
    return audit.get('commit_id') or 'legacy_'+hashlib.sha256(
        json.dumps(audit,sort_keys=True,default=numeric).encode()).hexdigest()


def active_updates(root):
    root=Path(root); rows=read_rows(root/'updates.jsonl')
    pointer=root/'active_update.json'
    if not pointer.exists(): return rows
    cursor=json.loads(pointer.read_text())['commit_id']
    lookup={audit_identity(r):r for r in rows if r.get('accepted',True)}
    chain=[]; seen=set()
    while cursor is not None:
        if cursor in seen or cursor not in lookup: raise ValueError('missing/cyclic active update lineage')
        seen.add(cursor); row=lookup[cursor]; chain.append(row); cursor=row.get('parent_commit_id')
    return list(reversed(chain))


def last_row(path):
    """Bounded tail read for the million-action journal, including live writes."""
    path=Path(path)
    if not path.exists(): return None
    with path.open('rb') as stream:
        stream.seek(0,2); size=stream.tell()
        stream.seek(max(0,size-65536)); data=stream.read()
    complete=data[:data.rfind(b'\n')+1].splitlines()
    for line in reversed(complete):
        if line.strip(): return json.loads(line)
    return None


class Checkpoints:
    def __init__(self,root,identity,guard=None):
        self.root=Path(root); self.identity=identity
        self.guard=guard
        self.root.mkdir(parents=True,exist_ok=True)

    def index(self):
        path=self.root/'index.json'
        return json.loads(path.read_text()) if path.exists() else dict(identity=self.identity,slots=[],milestones=[])

    def save(self,state,actions,*,threshold=None,kind='rolling',writer=None):
        import torch
        index=self.index()
        if index['identity']!=self.identity: raise ValueError('checkpoint identity differs')
        name=f'{kind}_{actions:09d}_{uuid.uuid4().hex}.pt'
        path=self.root/name; tmp=path.with_suffix('.partial')
        try:
            # Direct file writer: no BytesIO, pickle.dumps, or whole-buffer copy.
            payload=dict(identity=self.identity,actions=int(actions),state=state)
            with tmp.open('xb') as stream:
                if writer is not None: writer(payload,stream)
                else:
                    import shutil
                    from residual_rl.storage import BoundedCheckpointWriter,save_checkpoint_payload
                    guard=self.guard
                    reserve=(guard.free_gib if guard else 8)*2**30
                    used=sum(p.stat().st_size for p in (guard.root if guard else self.root).rglob('*') if p.is_file() and not p.is_symlink())
                    cap=(guard.output_gib*2**30-used) if guard else 4*2**30
                    deadline=time.monotonic()+max(0.,guard.deadline-time.time()) if guard else time.monotonic()+300
                    def check():
                        if guard: guard.poll()
                        return max(0,int(shutil.disk_usage(self.root).free-reserve)),None
                    sink=BoundedCheckpointWriter(stream,max_bytes=int(cap),deadline=deadline,check=check)
                    save_checkpoint_payload(payload,sink)
                stream.flush(); os.fsync(stream.fileno())
            record=dict(file=name,bytes=tmp.stat().st_size,sha256=sha256(tmp),actions=int(actions),threshold=threshold,kind=kind)
            if record['bytes']<=0: raise OSError('empty checkpoint')
            # Verify deserialization too before publishing the index.
            loaded=torch.load(tmp,map_location='cpu',weights_only=False)
            if loaded['identity']!=self.identity or loaded['actions']!=actions: raise ValueError('checkpoint verification failed')
            del loaded
            os.replace(tmp,path); fsync_dir(self.root)
            old=[]
            if kind=='rolling':
                slots=index['slots']+[record]; old=slots[:-2]; index['slots']=slots[-2:]
            else: index['milestones'].append(record)
            write_json(self.root/'index.json',index)
            # Only retire this store's own no-longer-indexed rolling generations.
            for stale in old:
                candidate=self.root/stale['file']
                if candidate.parent==self.root and candidate.exists() and sha256(candidate)==stale['sha256']:
                    candidate.unlink()
            return record
        except BaseException as error:
            try: write_json(self.root/'checkpoint_error.json',dict(error=repr(error),temporary=tmp.name,
                actions=actions,original_index_preserved=True,time=time.time()))
            except Exception: pass  # original error is never replaced
            raise

    def latest(self):
        import torch
        index=self.index()
        if index['identity']!=self.identity: raise ValueError('checkpoint identity differs')
        failures=[]
        candidates=sorted(index['slots']+index['milestones'],key=lambda r:r['actions'],reverse=True)
        for record in candidates:
            path=self.root/record['file']
            if path.parent!=self.root: raise ValueError('invalid checkpoint path')
            try:
                if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
                    raise ValueError('checkpoint hash/size mismatch')
                value=torch.load(path,map_location='cpu',weights_only=False)
                if value['identity']!=self.identity: raise ValueError('checkpoint identity differs')
                return value
            except (OSError,ValueError,RuntimeError,EOFError) as error: failures.append(repr(error))
        raise RuntimeError('no verified complete checkpoint: '+repr(failures))


class RunStore:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'attempts').mkdir(exist_ok=True)
        (self.root/'episodes').mkdir(exist_ok=True)

    @staticmethod
    def key(spec):
        role=spec['role']; arm=spec.get('arm','actor')
        if role not in ('train','validation','preflight','comparison','infer') or not arm.replace('_','').isalnum():
            raise ValueError('invalid attempt key')
        return f"{role}_{int(spec['seed'])}_{arm}"

    def claim(self,spec):
        key=self.key(spec)
        marker=self.root/'attempts'/f'{key}.json'
        if spec['role']=='preflight' and not marker.exists() and len(list((self.root/'attempts').glob('preflight_*.json')))>=24:
            from .safety import StopExperiment
            raise StopExperiment('24 preflight opportunity cap reached')
        with marker.open('x') as f:
            json.dump(spec,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
        path=self.root/'episodes'/key; path.mkdir(exist_ok=False)
        return path

    def rows(self): return read_rows(self.root/'episodes.jsonl')

    def commit(self,row):
        key=self.key(row)
        if any(self.key(r)==key for r in self.rows()): raise ValueError('attempt already recorded')
        if not (self.root/'attempts'/f'{key}.json').is_file(): raise ValueError('result has no opportunity marker')
        if row.get('final_collected') is not None and not 0<=row['final_collected']<=6:
            raise ValueError('invalid measured count')
        write_json(self.root/'episodes'/key/'result.json',row)
        append_json(self.root/'episodes.jsonl',row)

    def next_seed(self,role):
        from .identity import RANGES
        start,end=RANGES[role]
        used=[json.loads(p.read_text())['seed'] for p in (self.root/'attempts').glob(f'{role}_*.json')]
        seed=max(used,default=start-1)+1
        if seed>=end: raise RuntimeError('predeclared seed range exhausted')
        return seed
