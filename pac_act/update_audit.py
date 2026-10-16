"""Bounded numeric rejection evidence; never a learning checkpoint or resume."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .storage import append_json,sha256,write_json


def tree_digest(value):
    digest=hashlib.sha256()
    def token(raw):
        raw=raw.encode() if isinstance(raw,str) else raw
        digest.update(str(len(raw)).encode()+b':');digest.update(raw)
    def visit(item):
        if isinstance(item,torch.Tensor):
            token('tensor');token(str(item.dtype));token(str(tuple(item.shape)))
            data=item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            raw=memoryview(data).cast('B')
            for start in range(0,len(raw),1024*1024):token(raw[start:start+1024*1024])
        elif isinstance(item,np.ndarray):
            token('ndarray');token(item.dtype.str);token(str(item.shape))
            raw=memoryview(np.ascontiguousarray(item)).cast('B')
            for start in range(0,len(raw),1024*1024):token(raw[start:start+1024*1024])
        elif isinstance(item,dict):
            token('dict');token(str(len(item)))
            for key in sorted(item,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            token(type(item).__name__);token(str(len(item)))
            for child in item:visit(child)
        elif isinstance(item,(str,int,float,bool,type(None))):
            token(type(item).__name__);token(item.hex() if isinstance(item,float) else repr(item))
        else:raise TypeError('unsupported audit hash type: '+type(item).__name__)
    visit(value);return digest.hexdigest()


def state_hashes(state):
    result={key:tree_digest(state[key]) for key in ('model','optimizer','rng','updates')}
    for prefix in ('actor','critic','noise'):
        result[prefix]=tree_digest({k:v for k,v in state['model'].items() if k.startswith(prefix+'.')})
    return result


def json_safe(value):
    if isinstance(value,torch.Tensor):return json_safe(value.detach().cpu().tolist())
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return dict(nonfinite=repr(value))
    return value


def persist_rejection(root,audit,identity):
    if audit.get('accepted') is not False:raise ValueError('only rejected updates enter this store')
    root=Path(root);folder=root/'update_audits';folder.mkdir(exist_ok=True)
    path=folder/f'rejected_{audit["attempt_id"]}.json'
    if path.exists():raise FileExistsError('rejection audit already exists')
    write_json(path,json_safe(dict(audit,bundle_identity=identity)))
    ref=dict(audit_file=str(path.relative_to(root)),audit_sha256=sha256(path),
        attempt_id=audit['attempt_id'],accepted=False,rollback_verified=audit.get('rollback_verified'))
    append_json(root/'rejected_updates.jsonl',ref)
    return ref


def write_failure_reports(error,documents):
    failures=[]
    for path,document in documents.items():
        try:write_json(path,document)
        except BaseException as failure:
            failures.append(dict(path=str(path),error=repr(failure)))
            error.add_note('failure evidence write: '+str(path)+': '+repr(failure))
    return failures
