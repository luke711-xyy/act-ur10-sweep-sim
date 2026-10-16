"""Deterministic saved Actor execution: critic and prior are not constructed."""
import json
from pathlib import Path

from .config import DEFAULT
from .identity import LayoutRegistry,seed_spec,verify_bundle,bundle_digest
from .lifecycle import experiment_lock,make_guard,require_local_slot
from .storage import Checkpoints,RunStore,write_json


def evaluate(bundle,run,out,device='cpu',seed=None,target=None):
    if device!='cuda': require_local_slot()
    from .policy import ChunkPolicy
    from .training import evaluate_set
    frozen=verify_bundle(bundle); run=Path(run)
    identity=bundle_digest(dict(bundle=frozen['identity'],config=DEFAULT.asdict()))
    saved=Checkpoints(run/'checkpoints',identity).latest()['state']
    policy=ChunkPolicy(frozen['model'],device,build_critic=False)
    policy.load_learned(saved['learner']['model'])
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    with experiment_lock(out,local=device!='cuda'):
        guard=make_guard(out,'mac' if device!='cuda' else 'gpu')
        registry=LayoutRegistry(json.loads((Path(bundle)/'exclusions.json').read_text()))
        specs=None
        if seed is not None:
            specs=[dict(role='infer',seed=seed,target_count=target,arm='deterministic')]
        store=RunStore(out)
        write_json(out/'status.json',dict(state='evaluating',training_started=False))
        try:
            result=evaluate_set(bundle,policy,store,registry,guard,arm='deterministic',media='all',specs=specs)
            write_json(out/'status.json',dict(state='complete',training_started=False,summary=result))
            return result
        except BaseException as error:
            write_json(out/'status.json',dict(state='needs_attention',error=repr(error),automatic_retry=False)); raise
