"""Explicit, single-use continuation of deterministic, pre-update acceptance.

This is not learner resume and cannot recover an interrupted physical attempt.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import time

from .identity import seed_spec,verify_bundle
from .storage import RunStore,fsync_dir,read_rows,sha256


def pending_comparisons(rows):
    completed={RunStore.key(row) for row in rows}
    return [spec for arm in ('old_hybrid','new_zero','new_exploration') for index in (0,5)
            if RunStore.key(spec:=seed_spec('preflight',index,arm)) not in completed]


def continuation_state(root,previous_identity):
    root=Path(root)
    report=json.loads((root/'acceptance.json').read_text())
    if report.get('bundle_identity')!=previous_identity or not report.get('incomplete'):
        raise RuntimeError('continuation requires the original incomplete acceptance')
    if (root/'authorized_preflight_continuation.json').exists():
        raise FileExistsError('authorized preflight continuation already consumed')
    if read_rows(root/'updates.jsonl') or list((root/'checkpoints').glob('*.pt')):
        raise RuntimeError('cannot reconstruct a learner after updates')
    rows=read_rows(root/'episodes.jsonl'); completed={RunStore.key(r):r for r in rows}
    markers={p.stem:json.loads(p.read_text()) for p in (root/'attempts').glob('*.json')}
    if not rows or len(completed)!=len(rows) or set(markers)!=set(completed):
        raise RuntimeError('unresolved or duplicate physical opportunity; never retry')
    allowed={RunStore.key(seed_spec('preflight',i,a)) for a in ('old_hybrid','new_zero') for i in (0,5)}
    protected={}
    for key,row in completed.items():
        if key not in allowed or row.get('outcome') not in ('success','failure','safety'):
            raise RuntimeError('only completed deterministic comparisons can continue')
        if row.get('final_collected') is None or not row.get('layout_fingerprint'):
            raise RuntimeError('missing measured outcome or layout')
        folder=root/'episodes'/key
        if json.loads((folder/'result.json').read_text())!=row:
            raise RuntimeError('canonical result differs from ledger')
        if any(row.get(k)!=v for k,v in markers[key].items()):
            raise RuntimeError('attempt marker differs from completed row')
        media=json.loads((folder/'media.json').read_text()); video=folder/media['file']
        if video.parent!=folder or video.stat().st_size!=media['bytes'] or sha256(video)!=media['sha256']:
            raise RuntimeError('completed media is not verified')
        for path in [root/'attempts'/f'{key}.json',*(folder/name for name in
                ('configuration.json','layout.json','result.json','physical_trace.jsonl.gz','media.json')),video]:
            protected[str(path.relative_to(root))]=sha256(path)
    for name in ('episodes.jsonl','acceptance.json','status.json'):
        protected[name]=sha256(root/name)
    return dict(rows=rows,budget=report['budget'],checks=report.get('checks',{}),
                previous_identity=previous_identity,protected=protected)


def verify_replacement(previous_bundle,replacement):
    previous_bundle=Path(previous_bundle)
    original=verify_bundle(previous_bundle,project=previous_bundle/'source')
    old=original['manifest']; new=replacement['manifest']
    if old['files']!=new['files'] or old['schedule']!=new['schedule']:
        raise RuntimeError('base model, control inputs or seed schedule changed')
    changed={name for name in old['sources'].keys()|new['sources'].keys()
             if old['sources'].get(name)!=new['sources'].get(name)}
    allowed={'pac_act/acceptance.py','pac_act/cli.py','pac_act/media.py',
             'pac_act/preflight_continuation.py','pac_act/hybrid.py'}
    if changed-allowed: raise RuntimeError('unauthorized executable revision: '+repr(changed-allowed))
    if 'pac_act/hybrid.py' in changed:
        from .identity import PROJECT
        def control_ast(path):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ClassDef):
                    node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name!='_save_row']
            return ast.dump(tree,include_attributes=False)
        if control_ast(previous_bundle/'source/pac_act/hybrid.py')!=control_ast(PROJECT/'pac_act/hybrid.py'):
            raise RuntimeError('Hybrid control changed beyond trace metadata')
    return original


def claim_continuation(root,state,replacement_identity):
    root=Path(root)
    # Exclusive owner first. Even an interrupted archive write consumes this
    # authorization; another restart needs a new human decision.
    evidence=dict(time=time.time(),authorization='human approved media-only repair and remaining unattempted acceptance',
        previous_bundle_identity=state['previous_identity'],bundle_identity=replacement_identity,
        budget=state['budget'],completed_keys=[RunStore.key(r) for r in state['rows']],
        protected=state['protected'],physical_retry_allowed=False,learner_updates_before_continue=0)
    with (root/'authorized_preflight_continuation.json').open('x') as stream:
        json.dump(evidence,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    fsync_dir(root)
    for name in ('acceptance','status'):
        with (root/f'{name}.json').open('rb') as source, (root/f'{name}_before_authorized_continue.json').open('xb') as target:
            shutil.copyfileobj(source,target); target.flush(); os.fsync(target.fileno())
    fsync_dir(root)
