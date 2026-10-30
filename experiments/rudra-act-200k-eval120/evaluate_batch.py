"""This evaluation only: native Rudra ACT 200K, 120 fixed seeds, media on demand."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
LABEL = 'Rudra two-view DINOv3 ACT | step 200K | current native schedule'
MODEL_LABEL = 'rudra_dinov3_act_200k'
CHECKPOINT = PROJECT/'runs/rudra_dinov3_act_200k_20261023/model/checkpoints/step_200000'
SOURCE_PLAN = Path('/home/luke/.codex/worktrees/rudra-dinov3-act/UR１０SWEEP/runs/rudra_dinov3_act145k_eval120_seed2120000_20261024/episode_plan.json')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(temp, path)


def append_record(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def seed_target_pairs(rows):
    pairs = [(int(r['seed']), int(r['target_count'])) for r in rows]
    if (len(pairs) != 120 or {s for s, _ in pairs} != set(range(2120000, 2120120))
            or Counter(n for _, n in pairs) != Counter({n:20 for n in range(1, 7)})):
        raise ValueError('requires the original 120 unique seeds, twenty per target')
    return pairs


def failure_type(row):
    text = (str(row.get('reason', ''))+' '+str(row.get('termination_reason', ''))).lower()
    if row.get('success') is True:
        return None
    if 'ik' in text or 'inverse kinematics' in text:
        return 'ik_execution'
    if 'prediction' in text:
        return 'prediction'
    if 'collection window' in text:
        return 'collection_window'
    if 'timeout' in text or 'time limit' in text:
        return 'timeout'
    return 'other'


def summarize(rows, planned):
    known = [r for r in rows if r.get('final_collected') is not None]
    matrix = [[0]*6 for _ in range(7)]
    for r in known:
        actual = r['final_collected']
        if type(actual) is not int or not 0 <= actual <= 6:
            raise ValueError('invalid measured final collection count')
        matrix[actual][r['target_count']-1] += 1
    strict = sum(r.get('success') is True for r in rows)
    exact = sum(r['final_collected'] == r['target_count'] for r in known)
    errors = [r['final_collected']-r['target_count'] for r in known]
    by = {}
    for n in range(1,7):
        group = [r for r in rows if r['target_count']==n]
        successes = sum(r.get('success') is True for r in group)
        by[str(n)] = {'target':n, 'tested':len(group), 'planned':20,
                      'strict_successes':successes,
                      'strict_success_rate':successes/len(group) if group else None,
                      'unknown_counts':sum(r.get('final_collected') is None for r in group)}
    return {'label':LABEL, 'completed':len(rows), 'planned':planned,
            'complete':len(rows)==planned, 'strict_successes':strict,
            'strict_success_rate':strict/len(rows) if rows else None,
            'exact_final_matches':exact, 'count_match_rate':exact/len(rows) if rows else None,
            'known_counts':len(known), 'unknown_counts':len(rows)-len(known),
            'mean_absolute_count_error':sum(map(abs,errors))/len(errors) if errors else None,
            'mean_signed_count_error':sum(errors)/len(errors) if errors else None,
            'root_mean_square_count_error':(sum(e*e for e in errors)/len(errors))**.5 if errors else None,
            'failure_types':dict(Counter(v for r in rows if (v:=failure_type(r)) is not None)),
            'final_count_matrix':matrix, 'by_target':by, 'media_policy':'on_demand',
            'video_count':0, 'updated_at':time.time()}


def write_artifacts(root, rows, planned):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    summary = summarize(rows, planned)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    if rows:
        temp = root/'episodes.csv.tmp'
        with temp.open('w', newline='', encoding='utf-8-sig') as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        os.replace(temp, root/'episodes.csv')
    counts = np.asarray(summary['final_count_matrix'])
    totals = np.asarray([summary['by_target'][str(n)]['tested'] for n in range(1,7)])
    density = counts/np.maximum(totals,1)
    with (root/'density_matrix.csv.tmp').open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['target','actual','count','tested','frequency','unknown_counts'])
        for n in range(1,7):
            for actual in range(7):
                w.writerow([n,actual,int(counts[actual,n-1]),int(totals[n-1]),
                            float(density[actual,n-1]),summary['by_target'][str(n)]['unknown_counts']])
    os.replace(root/'density_matrix.csv.tmp', root/'density_matrix.csv')
    fig, ax=plt.subplots(figsize=(12,9), layout='constrained')
    im=ax.imshow(density, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='auto',
                 extent=(.5,6.5,-.5,6.5))
    for n in range(1,7):
        for actual in range(7):
            value=density[actual,n-1]; denom=int(totals[n-1])
            text=f'{value:.0%}\n{counts[actual,n-1]}/{denom}' if denom else '--\n0/0'
            ax.text(n,actual,text,ha='center',va='center',fontsize=13,
                    color='black' if value>.65 else 'white')
        ax.add_patch(Rectangle((n-.5,n-.5),1,1,fill=False,edgecolor='#ddf4fa',linewidth=2.5))
    ax.set(xticks=range(1,7),yticks=range(7),xlabel='Target count',
           ylabel='Final retained fully collected count',
           title=LABEL+'\n'+f"Strict success {summary['strict_successes']}/{len(rows)}; "
           +f"exact final count {summary['exact_final_matches']}/{len(rows)}")
    fig.colorbar(im,ax=ax,label='Frequency within each tested target')
    phase='FINAL' if len(rows)==planned else 'PROVISIONAL'
    fig.suptitle(f'{phase} {len(rows)}/{planned} independent episodes',fontsize=24)
    fig.supxlabel('Seeds 2120000-2120119; current Rudra scene. Final-count match differs from strict success.\n'
                  'Unknown counts excluded from cells; no videos or replay images generated.',fontsize=10)
    for suffix in ('png','svg'):
        temp=root/f'.success_density.tmp.{suffix}'
        fig.savefig(temp, format=suffix, dpi=150)
        os.replace(temp,root/f'success_density.{suffix}')
    plt.close(fig)
    atomic_json(root/'summary.json', summary)
    if summary['complete']:
        atomic_json(root/'evaluation_report.json', summary)


def episode_command(root, spec):
    return [sys.executable, '-u', str(root/'episode_entrypoint.py'), '--root',str(root),
            '--spec',spec['key'],'--seed',str(spec['seed']), '--target-count',str(spec['target_count']),
            '--model',spec['model_path']]


def save_measurement(root, cfg, seed, model, result):
    from sim.act.replay import persist_replay_input
    result.observations.clear()
    saved=persist_replay_input(cfg,seed,result,root/'replays',model=model)
    poses=result.initial_object_poses
    import numpy as np
    fingerprint=hashlib.sha256(np.asarray(poses,dtype='<f4').tobytes()).hexdigest()
    return {'success':bool(result.success),'reason':result.failure_reason,
            'target_count':int(result.target_count),'final_collected':result.final_collected,
            'peak_collected':result.peak_collected,'elapsed':float(result.elapsed),
            'scheduler_actions':len(result.actions),'sampled_frames':result.sampled_frames,
            'scheduler_queries':result.scheduler_queries,'scheduler_timeouts':result.scheduler_timeouts,
            'scheduler_late_results':result.scheduler_late_results,'peak_force':result.peak_force,
            'termination_reason':result.termination_reason,'first_entry_time':result.first_entry_time,
            'collection_window_seconds':result.collection_window_seconds,'model_checkpoint':model,
            'layout_fingerprint':fingerprint,'replay_input_path':str(saved),
            'replay_input_sha256':sha256(saved),'trace_rows':len(result.trace),
            'rejected_control_steps':sum(bool(r.get('rejection_error')) for r in result.trace),
            'media_policy':'on_demand','video_path':None}


def source_hashes():
    files=list((PROJECT/'sim').rglob('*.py'))+list(ROOT.glob('*.py'))
    return {str(p.relative_to(PROJECT)):sha256(p) for p in sorted(files)}


def prepare():
    import numpy as np
    from sim.config import load_config, save_config
    from sim.act.rollout import _episode_environment
    from sim.act.evaluate import INFERENCE_MAX_FRAMES
    if (ROOT/'metadata.json').exists():
        raise RuntimeError('prepared evaluation exists; use the frozen plan')
    cfg=load_config(str(PROJECT/'configs/rudra-dinov3-act-cuda.yaml'))
    source_profile=Path(cfg.scene.profile)
    shutil.copy2(source_profile,ROOT/'evaluation_scene.yaml')
    cfg.set_path('scene.profile',str(ROOT/'evaluation_scene.yaml'))
    assert cfg.scene.backend=='rudra_native'
    save_config(cfg,str(ROOT/'evaluation_config.yaml'))
    dataset=Path(cfg.act.dataset_dir)
    train=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines() if s.strip()]
    train_seeds={int(r['seed']) for r in train}
    train_hashes={r.get('layout_fingerprint') for r in train}
    pairs=seed_target_pairs(json.loads(SOURCE_PLAN.read_text()))
    assert not ({s for s,n in pairs} & train_seeds), 'evaluation seed occurs in training data'
    plan=[]; seen=set()
    for i,(seed,n) in enumerate(pairs):
        ec=cfg.copy(); ec.set_path('task.target_count',n)
        env=_episode_environment(ec,seed)
        try:
            env.reset(seed=seed)
            poses=np.concatenate((env.component_positions(),env.component_quats()),axis=1).astype('<f4')
        finally: env.close()
        fingerprint=hashlib.sha256(poses.tobytes()).hexdigest()
        assert fingerprint not in seen and fingerprint not in train_hashes
        seen.add(fingerprint)
        plan.append({'key':f'{MODEL_LABEL}:n{n}:l{i:03d}','seed':seed,'target_count':n,
                     'layout_index':i,'layout_fingerprint':fingerprint,'initial_object_poses':poses.tolist(),
                     'model_path':str(CHECKPOINT),'model_label':MODEL_LABEL,'checkpoint_step':200000})
    atomic_json(ROOT/'episode_plan.json',plan)
    hashes={p.name:sha256(p) for p in sorted(CHECKPOINT.iterdir()) if p.is_file()}
    assert all(k in hashes for k in ('model.safetensors','training_state.pt','config.json'))
    metadata={'label':LABEL,'created_at':time.time(),'project':str(PROJECT),'planned':120,
              'checkpoint':str(CHECKPOINT),'checkpoint_sha256':hashes,'seed_source':str(SOURCE_PLAN),
              'seed_source_sha256':sha256(SOURCE_PLAN),'episode_plan_sha256':sha256(ROOT/'episode_plan.json'),
              'config_sha256':sha256(ROOT/'evaluation_config.yaml'),'scene_sha256':sha256(ROOT/'evaluation_scene.yaml'),
              'dataset_manifest_sha256':sha256(dataset/'manifest.jsonl'),'source_hashes':source_hashes(),
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
              'max_frames_effective':INFERENCE_MAX_FRAMES,'media_policy':'on_demand',
              'schedule':'current native asynchronous ACT, external temporal ensemble coefficient 0.01',
              'scene':'rudra_native','camera_views':list(cfg.act.camera_views),'image_size':list(cfg.act.image_size),
              'training_continues':False,'one_physical_attempt_per_seed':True,
              'layout_note':'Original seed-target pairs regenerated in current Rudra scene; not claimed identical to legacy robot layouts.'}
    atomic_json(ROOT/'metadata.json',metadata)
    atomic_json(ROOT/'status.json',{'state':'prepared','completed':0,'planned':120,'updated_at':time.time()})
    write_artifacts(ROOT,[],120)
    print(json.dumps({'event':'batch_prepared','root':str(ROOT),'planned':120,'checkpoint':str(CHECKPOINT)}),flush=True)


def run():
    metadata=json.loads((ROOT/'metadata.json').read_text())
    assert sha256(ROOT/'episode_plan.json')==metadata['episode_plan_sha256']
    plan=json.loads((ROOT/'episode_plan.json').read_text()); seed_target_pairs(plan)
    ledger=ROOT/'episodes.jsonl'; attempts=ROOT/'attempts.jsonl'
    rows=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
    by_key={s['key']:s for s in plan}; completed={r['key'] for r in rows}
    assert len(completed)==len(rows) and completed<=by_key.keys()
    for r in rows:
        assert all(r[k]==by_key[r['key']][k] for k in ('seed','target_count','model_path','layout_fingerprint'))
        assert sha256(r['replay_input_path'])==r['replay_input_sha256']
    attempted={r['key'] for r in map(json.loads,attempts.read_text().splitlines())} if attempts.exists() else set()
    (ROOT/'logs').mkdir(exist_ok=True)
    child=None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise InterruptedError('batch stop requested; no automatic physical retry')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def status(**kw): atomic_json(ROOT/'status.json',dict(updated_at=time.time(),pid=os.getpid(),completed=len(rows),planned=120,**kw))
    with (ROOT/'batch.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for spec in plan:
                if spec['key'] in completed: continue
                assert source_hashes()==metadata['source_hashes'],'source changed during evaluation'
                assert sha256(ROOT/'evaluation_config.yaml')==metadata['config_sha256']
                assert sha256(ROOT/'evaluation_scene.yaml')==metadata['scene_sha256']
                assert {n:sha256(CHECKPOINT/n) for n in metadata['checkpoint_sha256']}==metadata['checkpoint_sha256']
                assert shutil.disk_usage(ROOT).free>4*1024**3
                log=ROOT/'logs'/f"seed_{spec['seed']}_n{spec['target_count']}.log"
                if spec['key'] not in attempted:
                    with log.open('x') as stream:
                        append_record(attempts,{'key':spec['key'],'seed':spec['seed'],'started_at':time.time()})
                        attempted.add(spec['key'])
                        child=subprocess.Popen(episode_command(ROOT,spec),cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
                        status(state='running',current=spec['key'],child_pid=child.pid,log_path=str(log))
                        print(json.dumps({'event':'episode_started','key':spec['key'],'seed':spec['seed'],
                                          'target_count':spec['target_count'],'completed':len(rows),'pid':child.pid}),flush=True)
                        code=child.wait()
                    child=None
                    if code not in (0,1): raise RuntimeError(f'episode process failed with code {code}; no retry')
                physical=None
                for line in reversed(log.read_text().splitlines()):
                    try: d=json.loads(line)
                    except json.JSONDecodeError: continue
                    if d.get('event')=='measurement_saved': physical=d; break
                if physical is None: raise RuntimeError('attempt lacks a durable physical result; no retry')
                assert physical['model_checkpoint']==spec['model_path'] and physical['target_count']==spec['target_count']
                assert physical['layout_fingerprint']==spec['layout_fingerprint']
                assert sha256(physical['replay_input_path'])==physical['replay_input_sha256']
                row={**spec,**physical}; row.pop('initial_object_poses'); row.pop('event')
                append_record(ledger,row); rows.append(row); completed.add(spec['key'])
                write_artifacts(ROOT,rows,120)
                status(state='running',current=None)
                print(json.dumps({'event':'episode_finished','completed':len(rows),'planned':120,
                    'seed':row['seed'],'target_count':row['target_count'],'success':row['success'],
                    'final_collected':row['final_collected'],'replay_input':row['replay_input_path']}),flush=True)
            write_artifacts(ROOT,rows,120); status(state='complete',current=None)
        except BaseException as exc:
            status(state='needs_attention',current=spec['key'] if 'spec' in locals() else None,
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.prepare_only: prepare()
    else: run()
