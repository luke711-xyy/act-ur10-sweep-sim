"""Import numeric layout identities only; never follow image or array paths."""
import hashlib
import json
from pathlib import Path


def extract(paths):
    seeds=set(); fingerprints=set(); evidence=[]
    def visit(value):
        if isinstance(value,dict):
            for key in ('seed','requested_seed'):
                if type(value.get(key)) is int: seeds.add(value[key])
            for key in ('layout_fingerprint','initial_layout_fingerprint'):
                if isinstance(value.get(key),str): fingerprints.add(value[key])
            if 'initial_object_poses' in value:
                import numpy as np
                poses=np.asarray(value['initial_object_poses'],dtype='<f4')
                if poses.shape==(6,7): fingerprints.add(hashlib.sha256(poses.tobytes()).hexdigest())
            for key,item in value.items():
                if key not in ('images','wrist','actions','trace','observations') and isinstance(item,(dict,list)):
                    visit(item)
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,(dict,list)): visit(item)
    for source in paths:
        path=Path(source); digest=hashlib.sha256(); size=0; incomplete=False
        if path.suffix=='.jsonl':
            with path.open('rb') as f:
                for line in f:
                    if not line.endswith(b'\n'):
                        incomplete=True; break
                    digest.update(line); size+=len(line)
                    if line.strip(): visit(json.loads(line))
        else:
            data=path.read_bytes(); digest.update(data); size=len(data); visit(json.loads(data))
        evidence.append(dict(name=path.name,sha256_of_consumed_prefix=digest.hexdigest(),bytes=size,
            ignored_incomplete_tail=incomplete))
    return dict(seeds=sorted(seeds),fingerprints=sorted(fingerprints),evidence=evidence,
        reserved_seed_ranges=[[2120000,2120120],[5000000,6000000]],complete_known_inputs=False)
