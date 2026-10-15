"""New-run-only durable evidence and single-opportunity ledger."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np


def numeric(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def json_payload(value):
    return json.dumps(value,allow_nan=False,default=numeric)


def write_json(path,value):
    payload = json_payload(value)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(payload+'\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


class BoundedCheckpointWriter:
    """torch.save's file-like sink; reject overflow BEFORE writing that block."""
    def __init__(self,stream,*,max_bytes,deadline,check):
        self.stream=stream; self.max_bytes=max_bytes; self.deadline=deadline
        self.check=check; self.written=0; self.next_sample=0; self.remaining=0

    def write(self,value):
        if time.monotonic()>=self.deadline: raise RuntimeError('checkpoint deadline exhausted')
        if self.written+len(value)>self.max_bytes: raise RuntimeError('checkpoint byte allowance exhausted')
        view=memoryview(value)
        for offset in range(0,len(view),2**20):
            block=view[offset:offset+2**20]
            if time.monotonic()>=self.deadline: raise RuntimeError('checkpoint deadline exhausted')
            if self.written>=self.next_sample:
                self.remaining,_=self.check(); self.next_sample=self.written+2**20
            if len(block)>self.remaining: raise RuntimeError('checkpoint resource headroom exhausted')
            count=self.stream.write(block); self.written+=count; self.remaining-=count
        return len(value)

    def flush(self): self.stream.flush()

    def read(self,*args): return self.stream.read(*args)

    def tell(self): return self.written


def checkpoint_evidence(path):
    path=Path(path)
    evidence=json.loads(path.with_suffix('.json').read_text())
    if path.stat().st_size!=evidence['bytes'] or sha256(path)!=evidence['sha256']:
        raise ValueError('checkpoint evidence does not match complete file')
    return dict(evidence,checkpoint=str(path))


def previous_checkpoint(root,preferred=None):
    """Only verifiable completed files qualify. Partial/corrupt files are evidence too."""
    candidates=sorted((Path(root)/'checkpoints').glob('*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    if preferred is not None: candidates=[Path(preferred)]+[p for p in candidates if p!=Path(preferred)]
    failures=[]
    for path in candidates:
        try: return checkpoint_evidence(path),failures
        except (OSError,ValueError,KeyError,TypeError) as error:
            failures.append(dict(checkpoint=str(path),error=f'{type(error).__name__}: {error}'))
    return None,failures


def write_experience(path,arrays,guard,*,episode_id,target_count,complete,**metadata):
    """Publish numeric arrays only after a bounded, fsynced atomic write."""
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    allowed={'actor_obs','critic_obs','actions','rewards','terminated','learn_mask'}
    if not arrays or not set(arrays)<=allowed or (complete and set(arrays)!=allowed):
        raise ValueError('invalid experience array fields')
    for key,value in arrays.items():
        dtype=np.dtype(bool if key in {'terminated','learn_mask'} else np.float32)
        if not isinstance(value,np.ndarray) or value.dtype!=dtype:
            raise ValueError('experience must contain float32/bool numeric arrays')
        if complete and not np.isfinite(value).all(): raise ValueError('nonfinite completed experience')
    max_bytes,deadline=guard.checkpoint_allowance(compact=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp'); created=False
    try:
        with temporary.open('xb') as stream:
            created=True
            writer=BoundedCheckpointWriter(stream,max_bytes=max_bytes,deadline=deadline,
                check=lambda:guard.checkpoint_allowance(compact=True))
            np.savez_compressed(writer,**arrays)
            stream.flush(); os.fsync(stream.fileno())
        guard.checkpoint_allowance(compact=True)
        evidence=dict(path=str(path),sha256=sha256(temporary),bytes=temporary.stat().st_size,
            episode_id=episode_id,target_count=target_count,complete=bool(complete),
            valid_episode=bool(complete),arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype))
                                              for key,value in arrays.items()},**metadata)
        temporary.replace(path)
        write_json(path.with_suffix('.json'),evidence)
        return evidence
    finally:
        if created: temporary.unlink(missing_ok=True)


def experience_reference(root):
    root=Path(root); ledger=root/'episodes.jsonl'
    return dict(episodes_directory=str(root/'episodes'),ledger=str(ledger),
        ledger_bytes=ledger.stat().st_size if ledger.exists() else 0,
        ledger_prefix_sha256=sha256(ledger) if ledger.exists() else None,
        completed_pattern='episodes/*/experience.npz',partial_pattern='episodes/*/partial_experience.npz',
        partial_is_not_valid_replay=True)


class RunStore:
    def __init__(self,root,*,create=False):
        self.root=Path(root)
        if create:
            self.root.mkdir(parents=True,exist_ok=False)
            (self.root/'attempts').mkdir()
            write_json(self.root/'status.json',dict(state='prepared',training_started=False))
        elif not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def rows(self):
        path=self.root/'episodes.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def claim(self,spec):
        key=spec['episode_id']
        if (not re.fullmatch(r'[a-zA-Z0-9_-]+',key) or type(spec['seed']) is not int
                or type(spec['target_count']) is not int or not 1<=spec['target_count']<=6):
            raise ValueError('invalid opportunity identity')
        payload=json_payload(spec)
        with (self.root/'attempts'/f'{key}.json').open('x') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())

    def commit(self,row):
        key=row['episode_id']
        if any(value['episode_id']==key for value in self.rows()):
            raise RuntimeError('cannot replace/repeat a scored opportunity')
        marker=json.loads((self.root/'attempts'/f'{key}.json').read_text())
        if any(row.get(k)!=v for k,v in marker.items()):
            raise ValueError('result differs from consumed opportunity')
        count=row.get('final_collected')
        administrative=row.get('outcome')=='budget_truncated' and row.get('success') is None
        if ((type(row.get('success')) is not bool and not administrative) or row.get('opportunities')!=1
                or row.get('physical_attempts') not in (0,1)
                or (count is not None and (type(count) is not int or not 0<=count<=6))
                or (row['success'] and count!=row['target_count'])):
            raise ValueError('invalid physical result')
        payload=json_payload(row)
        with (self.root/'episodes.jsonl').open('a') as stream:
            stream.write(payload+'\n'); stream.flush(); os.fsync(stream.fileno())

    def commit_episode(self,row,episode,guard):
        destination=self.root/'episodes'/row['episode_id']
        if episode is None:
            evidence=dict(status='not_applicable',reason='no numeric Episode was produced')
        else:
            if episode.episode_id!=row['episode_id'] or episode.target_count!=row['target_count']:
                raise ValueError('experience identity differs from physical opportunity')
            arrays={key:getattr(episode,key) for key in
                    ('actor_obs','critic_obs','actions','rewards','terminated','learn_mask')}
            evidence=write_experience(destination/'experience.npz',arrays,guard,
                episode_id=episode.episode_id,target_count=episode.target_count,complete=True,
                administrative_truncation=row.get('outcome')=='budget_truncated')
        row=dict(row,experience=evidence)
        write_json(destination/'result.json',row)
        self.commit(row)
        return row

    def require_clean(self):
        rows=self.rows(); keys=[row['episode_id'] for row in rows]
        markers={p.stem for p in (self.root/'attempts').glob('*.json')}
        if len(set(keys))!=len(keys) or markers!=set(keys):
            raise RuntimeError('orphan attempt or duplicate ledger: explicit review required')


class TraceWriter:
    """Append numerical physical states, never RGB or live renderer handles."""
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=gzip.open(self.path,'xt',encoding='utf-8')
        self.rows=0

    def append(self,row):
        self.stream.write(json_payload(row)+'\n'); self.rows+=1
        if self.rows%100==0: self.stream.flush()

    def close(self):
        self.stream.close()
