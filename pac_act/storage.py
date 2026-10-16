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
    with Path(path).open('a') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())


def read_rows(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


class Checkpoints:
    def __init__(self,root,identity):
        self.root=Path(root); self.identity=identity
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
                (writer or torch.save)(payload,stream); stream.flush(); os.fsync(stream.fileno())
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
        with (self.root/'attempts'/f'{key}.json').open('x') as f:
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
