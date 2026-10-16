"""Explicit authorized persistence repair. Never runs physics or updates weights.

Original manifests remain immutable. A separate authorization binds the precise
checkpoint/ledger prefix and an allowlisted persistence-only source revision.
"""
import collections
import copy
import gzip
import json
from pathlib import Path
import random
import time
import os

import numpy as np

from .config import DEFAULT, STABILITY
from .storage import RunStore, checkpoint_evidence, sha256, write_json, unpack_replay


REPAIR_FILES = frozenset(('residual_rl/storage.py', 'residual_rl/pilot.py',
                         'residual_rl/long_training.py', 'residual_rl/resume.py',
                         'scripts/train_residual_long.py','scripts/train_residual_sac.py'))


def verify_repair(root):
    from .pilot import own_code_hashes
    root = Path(root)
    original = json.loads((root/'code_hashes.json').read_text())
    auth = json.loads((root/'resume_authorization.json').read_text())
    actual = own_code_hashes()
    changed = {k for k in original.keys() | actual.keys() if original.get(k) != actual.get(k)}
    if (auth.get('human_authorized') is not True or not changed <= REPAIR_FILES
            or auth['original_code_hashes'] != original or auth['repaired_code_hashes'] != actual
            or sha256(root/'seal.json') != auth['original_seal_sha256']):
        raise RuntimeError('unapproved source change outside checkpoint persistence repair')
    return auth


def reconstruct_sampling(records, *, capacity=100000, seed=20261015):
    """Replay only sampler RNG draws and update permissions, NOT gradients/actions.

    The four integer draws per sequence exactly match EpisodicReplay.sample.
    Eligibility uses the real learn masks, including windows that start in the
    prefix. Entire-episode eviction and zero-sample failures stay unchanged.
    """
    from .pilot import UpdateSchedule
    queue = collections.deque()
    contexts = 0
    rng = np.random.default_rng(seed)
    cursor = 0
    schedule = UpdateSchedule()
    seen = []
    for record in records:
        mask = record['mask']
        valid = 0
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            n, valid = len(mask), int(mask.sum())
            if not 0 < n <= capacity: raise ValueError('invalid durable episode size')
            cumulative = np.r_[0, np.cumsum(mask)]
            starts = int(np.sum(cumulative[np.minimum(np.arange(n)+32, n)] - cumulative[:-1] > 0))
            item = dict(record, contexts=n, valid=valid, starts=starts)
            queue.append(item)
            contexts += n
            seen.append(record['episode_id'])
            while contexts > capacity: contexts -= queue.popleft()['contexts']
        counts = {g: [0, 0] for g in range(1, 7)}
        for item in queue:
            if item['terminated']:
                counts[item['target_count']][0] += int(item['valid'] > 0)
                counts[item['target_count']][1] += item['valid']
        ready = (sum(v[1] for v in counts.values()) >= STABILITY.warmup_steps and
                 all(v[0] >= STABILITY.episodes_per_goal and v[1] >= STABILITY.steps_per_goal
                     for v in counts.values()))
        schedule.begin_episode(valid, ready=ready)
        while schedule.due():
            choices = {g: [v for v in queue if v['target_count'] == g and v['starts']]
                       for g in range(1, 7)}
            for _ in range(4):
                eligible = choices[cursor+1]
                item = eligible[int(rng.integers(len(eligible)))]
                rng.integers(item['starts'])
                cursor = (cursor+1) % 6
            schedule.completed()
    return dict(rng_state=copy.deepcopy(rng.bit_generator.state), goal_cursor=cursor,
                schedule=vars(schedule).copy(), retained_ids=[v['episode_id'] for v in queue],
                seen_ids=sorted(seen), contexts=contexts)


def read_episode(root, row):
    from .learning import Episode
    ref = row['experience']
    path = Path(ref['path']).resolve()
    expected = (Path(root)/'episodes'/row['episode_id']/'experience.npz').resolve()
    if (path != expected or not ref['complete'] or not ref['valid_episode']
            or path.stat().st_size != ref['bytes'] or sha256(path) != ref['sha256']):
        raise ValueError('durable experience evidence mismatch: '+row['episode_id'])
    with np.load(path, allow_pickle=False) as arrays:
        return Episode(row['episode_id'], row['target_count'], **{k: arrays[k] for k in
                       ('actor_obs', 'critic_obs', 'actions', 'rewards', 'terminated', 'learn_mask')})


def load_recovery(root, *, require_authorization=True):
    """Validate and reconstruct a complete learning boundary from saved evidence."""
    import torch
    from .learning import RecurrentSAC, EpisodicReplay
    from .pilot import UpdateSchedule, own_code_hashes
    from .runtime import identity
    from .policy_guard import ProbeBank
    root = Path(root)
    auth = verify_repair(root) if require_authorization else None
    seal = json.loads((root/'seal.json').read_text())
    if any(sha256(root/name) != digest for name, digest in seal.items()):
        raise ValueError('original sealed metadata changed')
    if json.loads((root/'frozen_identity.json').read_text()) != identity():
        raise ValueError('frozen Hybrid/ACT/Z baseline changed')
    if json.loads((root/'configuration.json').read_text()) != DEFAULT.to_dict():
        raise ValueError('approved training configuration changed')
    original = json.loads((root/'code_hashes.json').read_text())
    current = own_code_hashes()
    if not {k for k in original.keys() | current.keys() if original.get(k) != current.get(k)} <= REPAIR_FILES:
        raise ValueError('non-persistence source modification')
    evidence = checkpoint_evidence(root/'checkpoints/interrupted.pt')
    payload = torch.load(evidence['checkpoint'], map_location='cpu', weights_only=False)
    if payload['run_config'] != DEFAULT.to_dict() or payload['code_hashes'] != original:
        raise ValueError('interrupted checkpoint configuration/source identity mismatch')
    store = RunStore(root)
    store.require_clean()
    rows = store.rows()
    ref = payload['durable_experience']
    if (ref['ledger_bytes'] != (root/'episodes.jsonl').stat().st_size
            or ref['ledger_prefix_sha256'] != sha256(root/'episodes.jsonl')):
        raise ValueError('checkpoint does not describe the entire committed ledger')
    if auth and (auth['checkpoint_sha256'] != evidence['sha256']
                 or auth['ledger_sha256'] != ref['ledger_prefix_sha256']):
        raise ValueError('recovery evidence changed after authorization')
    learner = RecurrentSAC(payload['learner']['actor_dim'], payload['learner']['critic_dim'])
    learner.load_checkpoint(payload['learner'])
    records = []
    cumulative_steps = []
    total = 0
    for i, row in enumerate(rows):
        if (row['episode_id'] != f'train_{i:07d}' or row['seed'] != DEFAULT.training_seed_base+i
                or row['target_count'] != i % 6+1 or row['phase'] != 'train'
                or row.get('outcome') == 'budget_truncated'):
            raise ValueError('noncontiguous or administratively incomplete training history')
        episode = read_episode(root, row) if row['experience'].get('path') else None
        if episode is not None and not episode.terminated[-1]:
            raise ValueError('unfinished experience at resume boundary')
        mask = episode.learn_mask.copy() if episode is not None else None
        valid = int(mask.sum()) if mask is not None else 0
        if valid != row['residual_frames']: raise ValueError('recorded effective-step count differs')
        total += valid
        cumulative_steps.append(total)
        records.append(dict(episode_id=row['episode_id'], target_count=row['target_count'],
                            mask=mask, terminated=True))
    if total != payload['residual_decisions'] or total != evidence['residual_decisions']:
        raise ValueError('checkpoint/ledger effective-step mismatch')
    reconstructed = reconstruct_sampling(records, seed=DEFAULT.random_seed)
    if reconstructed['schedule']['updates'] != learner.updates:
        raise ValueError('reconstructed update cadence disagrees with checkpoint')
    # Legacy compact checkpoints omitted replay RNG. Verify reconstruction at an
    # independent old full snapshot before carrying its exact draws forward.
    witnesses = []
    for path in sorted((root/'checkpoints').glob('step_*_milestone_*.pt')):
        info = checkpoint_evidence(path)
        if info['residual_decisions'] >= total: continue
        old = torch.load(path, map_location='cpu', weights_only=False)
        state = unpack_replay(old['replay'],root=root)
        end = cumulative_steps.index(info['residual_decisions'])+1
        check = reconstruct_sampling(records[:end], capacity=state['capacity'], seed=DEFAULT.random_seed)
        if (check['rng_state'] != state['rng_state'] or check['goal_cursor'] != state['goal_cursor']
                or check['schedule']['updates'] != old['learner']['updates']
                or check['retained_ids'] != [e['episode_id'] for e in state['episodes']]):
            raise ValueError('sampler reconstruction failed independent full-checkpoint witness')
        witnesses.append(dict(path=str(path), sha256=info['sha256'], verified=True))
        del old, state
    if not witnesses: raise ValueError('legacy compact recovery requires a full sampler witness')
    with gzip.open(root/'learning_metrics.jsonl.gz', 'rt') as stream:
        metrics = [json.loads(line) for line in stream]
    if ([m['updates'] for m in metrics] != list(range(1, learner.updates+1))
            or metrics[-1]['actor_updates'] != learner.actor_updates):
        raise ValueError('committed learning metrics differ from checkpoint')
    replay = EpisodicReplay(seed=DEFAULT.random_seed)
    by_id = {r['episode_id']: r for r in rows}
    for key in reconstructed['retained_ids']: replay.add(read_episode(root, by_id[key]))
    replay._seen_ids = set(reconstructed['seen_ids'])
    replay._goal_cursor = reconstructed['goal_cursor']
    replay._rng.bit_generator.state = copy.deepcopy(reconstructed['rng_state'])
    probes = [read_episode(root, by_id[key]) for key in
              dict.fromkeys(v['episode_id'] for v in learner._probe_spec)]
    learner._probe_bank = ProbeBank.from_spec(learner._probe_spec, probes)
    schedule = UpdateSchedule()
    schedule.__dict__.update(reconstructed['schedule'])
    proof = dict(training_steps=total, updates=learner.updates, actor_updates=learner.actor_updates,
                 completed_episodes=len(rows), next_seed=DEFAULT.training_seed_base+len(rows),
                 replay_context_steps=replay.steps, replay_episodes=len(replay._episodes),
                 sampler_cursor=replay._goal_cursor, schedule=vars(schedule), witnesses=witnesses,
                 checkpoint_sha256=evidence['sha256'], ledger_sha256=ref['ledger_prefix_sha256'],
                 physical_execution=False, learning_updates_executed=0)
    return dict(learner=learner, replay=replay, schedule=schedule, payload=payload,
                proof=proof, next_index=len(rows), training_steps=total)


def authorize_recovery(root):
    """Called only after explicit human approval; archives, never rewrites seals."""
    from .pilot import own_code_hashes
    root = Path(root)
    if (root/'resume_authorization.json').exists(): raise FileExistsError('recovery already authorized')
    recovered = load_recovery(root, require_authorization=False)
    archive = root/'before_authorized_resume'
    archive.mkdir(exist_ok=False)
    for name in ('status.json','summary.json','exception.json','training_progress.json',
                 'launch.json','started.json','checkpoint_state.json','save_result.json',
                 'supervisor_shutdown.json'):
        path = root/name
        if path.is_file():
            with (archive/name).open('xb') as stream: stream.write(path.read_bytes())
    # Atomic checkpoint publication replaces inodes. Hard links retain every
    # pre-resume completed checkpoint without duplicating the large replay file.
    (archive/'checkpoints').mkdir()
    for path in (root/'checkpoints').iterdir():
        if path.suffix in ('.pt','.json') and path.is_file():
            os.link(path, archive/'checkpoints'/path.name)
    write_json(root/'resume_validation.json', recovered['proof'])
    write_json(root/'resume_authorization.json', dict(human_authorized=True,
        purpose='checkpoint persistence repair; continue without replaying any physical opportunity',
        original_code_hashes=json.loads((root/'code_hashes.json').read_text()),
        repaired_code_hashes=own_code_hashes(), original_seal_sha256=sha256(root/'seal.json'),
        checkpoint_sha256=recovered['proof']['checkpoint_sha256'],
        ledger_sha256=recovered['proof']['ledger_sha256'],
        training_steps=recovered['training_steps'], next_index=recovered['next_index'],
        created_at=time.time(), original_budgets_preserved=True))
    return recovered['proof']


def restore_global_rng(payload):
    import torch
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['mps_rng'] is not None:
        if not torch.backends.mps.is_available(): raise RuntimeError('saved ACT MPS device unavailable')
        torch.mps.set_rng_state(payload['mps_rng'])
