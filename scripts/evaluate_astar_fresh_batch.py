"""Fresh paired A*: one plan/execution per layout, measured failure counts.

Logging/batch adapter only. Original A* primitives, dynamics, IK thresholds,
admittance and safety are unchanged. No ACT policy or demonstration collector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from scripts import evaluate_astar_baseline as base

EXPORTS=Path('/Users/huazhi_luke/Documents/Codex/2026-09-08/zhe/exports')
LABEL='astar_single_attempt_fresh_20261015'
CONTRACT='expert any exact N, no partial overlap, delivery/contact and one-second stable hold'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,default=json_default,allow_nan=False,indent=2).encode()
    fd,name=tempfile.mkstemp(prefix='.astar-record-',dir=path.parent)
    with os.fdopen(fd,'wb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    Path(name).replace(path)


def layout_fingerprint(env):
    poses=np.concatenate((env.component_positions(),env.component_quats()),axis=1).astype('<f4')
    return hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()


def measure(env):
    """Read current component geometry; never advance physics to obtain a score."""
    positions=np.asarray(env.component_positions(),dtype=np.float64)
    quats=np.asarray(env.component_quats(),dtype=np.float64)
    mask=np.asarray(env.collected_mask())
    if (positions.shape!=(6,3) or quats.shape!=(6,4) or mask.shape!=(6,) or mask.dtype!=np.bool_
        or not np.isfinite(positions).all() or not np.isfinite(quats).all()
        or not np.isfinite(env.time)):
        raise ValueError('invalid live terminal measurement; never replace with zero')
    partial=np.asarray(env.partial_collection_overlap_mask())
    if partial.shape!=(6,) or partial.dtype!=np.bool_:raise ValueError('invalid partial-overlap measurement')
    values={'component_positions':positions,'component_quats':quats,
            'fully_collected_mask':mask.copy(),'partial_overlap_mask':partial.copy(),
            'simulated_time':np.asarray(env.time,dtype=np.float64)}
    for name in ('qpos','qvel','ctrl'):
        value=np.asarray(getattr(env.data,name),dtype=np.float64).copy()
        if not np.isfinite(value).all():raise ValueError('invalid physics measurement')
        values[name]=value
    if env.model is not None:
        import mujoco
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(env.model,spec),dtype=np.float64)
        mujoco.mj_getState(env.model,env.data,state,spec)
        if not np.isfinite(state).all():raise ValueError('invalid integration measurement')
        values.update(integration_state=state,integration_state_spec=np.asarray(int(spec)))
    return values


def run_trial(cfg,spec,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    local=base._expert_config(cfg,spec['target_count'])
    local.set_path('episode.max_frames',2000)
    local.set_path('episode.max_time',2000/float(local.act.action_hz))
    if local.get_path('planner.expert_planning_mode')!='coverage_astar_only':
        raise ValueError('baseline must use original coverage A*')
    env=base.SweepEnv(local,seed=spec['seed'])
    try:
        env.reset(seed=spec['seed'])
        if layout_fingerprint(env)!=spec['layout_fingerprint']:
            raise ValueError('initial layout does not match paired plan')
        started=float(env.time)
        write_json(destination/'initial_layout.json',{'key':spec['key'],'seed':spec['seed'],
            'fingerprint':spec['layout_fingerprint'],'positions':env.component_positions(),
            'quats':env.component_quats()})
        plans=base.expert_plan_candidates(env,local,max_candidates=1)
        no_plan=not plans or not plans[0].feasible
        result=None;error=None;fatal=None;physical=0;control_calls=0;control_completed=0
        last_command=None;control_error=None
        if not no_plan:
            plan=plans[0]
            path,phases=base._expert_execution_path(local,plan,table_top_z=env.table_top_z)
            original_step=env.step_control
            def observed_step(command):
                nonlocal control_calls,control_completed,last_command,control_error
                last_command=command;control_calls+=1
                try:original_step(command)
                except Exception as exc:control_error=exc;raise
                control_completed+=1
            env.step_control=observed_step
            physical=1
            try:
                result=base.run_action_path(env,local,path,collect_observations=False,phases=phases,
                    target_indices=np.asarray(plan.target_indices,dtype=int),
                    recovery_planner=None,max_recovery_rounds=0,contact_z_mode='expert_admittance')
            except Exception as exc:
                # This opportunity ends here; observe the STILL-LIVE state, no
                # recovery command, reset, alternative plan or second execution.
                error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
                match=re.fullmatch(r'UR10 IK target is unreachable: position residual=([0-9.eE+\-]+) m, orientation residual=([0-9.eE+\-]+) rad',str(exc))
                known=(exc is control_error and isinstance(exc,RuntimeError) and match is not None
                       and all(np.isfinite(float(value)) for value in match.groups()))
                error['recognized_control_execution_failure']=bool(known)
                if not known:fatal=exc
            finally:env.step_control=original_step
            if result is not None:result=base._annotate_expert_result(result,plan,1)
        state=measure(env)
        count=int(state['fully_collected_mask'].sum())
        with (destination/'final_state.npz').open('xb') as stream:
            np.savez_compressed(stream,**state);stream.flush();os.fsync(stream.fileno())
        if error is not None:write_json(destination/'execution_exception.json',error)
        if fatal is not None:raise fatal
        outcome='execution_exception' if error else ('no_plan' if no_plan else 'native_result')
        reason=(f"{error['type']}: {error['message']}" if error else
                ('no feasible A* plan' if no_plan else str(result.failure_reason)))
        command=None if last_command is None else np.asarray(last_command.as_array(),dtype=float).tolist()
        row={'key':spec['key'],'seed':spec['seed'],'target_count':spec['target_count'],
             'layout_fingerprint':spec['layout_fingerprint'],'model_label':LABEL,'model_path':None,
             'success':bool(not error and result is not None and result.success),
             'final_collected':count,'collected':count,'reason':reason,'outcome':outcome,
             'final_partial_overlaps':int(state['partial_overlap_mask'].sum()),
             'opportunities':1,'planning_calls':1,'physical_attempts':physical,'no_plan':bool(no_plan),
             'planner_strategy':'coverage_astar','frame_budget':2000,
             'elapsed':float(env.time)-started,'control_calls':control_calls,
             'control_completed':control_completed,'last_requested_command':command,
             'error':None if error is None else {k:error[k] for k in ('type','message')},
             'quantity_source':'live env.collected_mask at termination, including execution exceptions',
             'quantity_measured_before_env_close':True,
             'failure_snapshot_note':'post-exception environment state; no additional physics step or IK attempt',
             'native_success_contract':CONTRACT,'ACT_contract_differs':'ACT has first-entry two-second window',
             'media_retained':False,'final_state':str(destination/'final_state.npz'),
             'final_state_sha256':sha(destination/'final_state.npz')}
        write_json(destination/'result.json',row)
        return row
    finally:env.close()


def validate_row(row,spec):
    for key in ('key','seed','target_count','layout_fingerprint'):
        if row[key]!=spec[key]:raise ValueError('result pairing mismatch')
    if (type(row['final_collected']) is not int or not 0<=row['final_collected']<=6
        or type(row['success']) is not bool or row['opportunities']!=1
        or row['planning_calls']!=1 or row['physical_attempts'] not in (0,1)
        or row['outcome']=='execution_exception' and row['success']
        or row['success'] and row['final_collected']!=row['target_count']):
        raise ValueError('invalid opportunity result')


def summarize(rows):
    by_target={}
    for row in rows:
        if type(row['final_collected']) is not int or not 0<=row['final_collected']<=6:
            raise ValueError('unmeasured quantities cannot enter statistics')
    for target in range(1,7):
        group=[r for r in rows if r['target_count']==target]
        errors=[abs(r['final_collected']-target) for r in group]
        by_target[str(target)]={'episodes':len(group),'successes':sum(r['success'] for r in group),
            'exact_quantity':sum(e==0 for e in errors),'within_one':sum(e<=1 for e in errors),
            'absolute_error_counts':dict(Counter(map(str,errors))),
            'mean_absolute_error':sum(errors)/len(errors) if errors else None}
    return {'completed':len(rows),'planned':120,'complete':len(rows)==120,'opportunities':len(rows),
            'successes':sum(r['success'] for r in rows),
            'physical_attempts':sum(r['physical_attempts'] for r in rows),
            'no_plan_failures':sum(r['no_plan'] for r in rows),
            'execution_exceptions':sum(r['outcome']=='execution_exception' for r in rows),
            'all_final_quantities_measured':True,'by_target':by_target,
            'native_success_contract':CONTRACT,'quantity_match_is_not_strict_success':True,
            'updated_at':time.time()}


def claim_attempt(root,spec):
    markers=Path(root)/'attempts';markers.mkdir(exist_ok=True)
    with (markers/f"seed_{spec['seed']}.json").open('x') as stream:
        json.dump({k:spec[k] for k in ('key','seed','target_count','layout_fingerprint')}
                  |{'started_at':time.time(),'owner_pid':os.getpid()},stream)
        stream.flush();os.fsync(stream.fileno())


def render_artifacts(root,rows):
    summary=summarize(rows);write_json(root/'summary.json',summary)
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    counts=np.zeros((7,6),dtype=int)
    for row in rows:counts[row['final_collected'],row['target_count']-1]+=1
    totals=counts.sum(axis=0)
    probabilities=np.divide(counts,totals,out=np.zeros((7,6)),where=totals>0)
    fig,ax=plt.subplots(figsize=(8.5,7))
    picture=ax.imshow(probabilities,origin='lower',cmap='magma',vmin=0,vmax=1,
                      extent=(.5,6.5,-.5,6.5),aspect='auto')
    for target in range(1,7):
        for actual in range(7):
            value=probabilities[actual,target-1]
            text=f'{value:.0%}\n{counts[actual,target-1]}/{totals[target-1]}' if totals[target-1] else '--'
            ax.text(target,actual,text,ha='center',va='center',fontsize=10,color='black' if value>.65 else 'white')
        ax.add_patch(Rectangle((target-.5,target-.5),1,1,fill=False,edgecolor='#ddf4fa',linewidth=2))
    ax.set(xticks=range(1,7),yticks=range(7),xlabel='Target count',ylabel='Final fully collected count',
           title=f"Pure A* | one opportunity per layout | {len(rows)}/120\nStrict successes {summary['successes']}; execution exceptions {summary['execution_exceptions']}")
    fig.colorbar(picture,ax=ax,label='Frequency within each tested target')
    fig.text(.5,.015,'Failure quantities are measured, not replaced by zero. Native success differs from ACT.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    for suffix in ('png','svg'):
        fd,name=tempfile.mkstemp(prefix='.astar-chart-',dir=root)
        with os.fdopen(fd,'wb') as stream:fig.savefig(stream,format=suffix,dpi=180)
        Path(name).replace(root/f'final_count_density.{suffix}')
    plt.close(fig)
    fields=['key','seed','target_count','success','final_collected','outcome','reason','physical_attempts','elapsed']
    fd,name=tempfile.mkstemp(prefix='.astar-csv-',dir=root)
    with os.fdopen(fd,'w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    Path(name).replace(root/'episodes.csv')


def verify(root):
    metadata=json.loads((root/'metadata.json').read_text())
    if metadata.get('kind')!='fresh_astar_single_attempt_measured_errors_v2':
        raise ValueError('not a fresh single-attempt batch; historical batches are read-only')
    if sha(root/'metadata.json')!=json.loads((root/'seal.json').read_text())['metadata_sha256']:
        raise ValueError('metadata seal changed')
    if base._source_fingerprints(PROJECT)!=metadata['sources_sha256']:
        raise ValueError('simulation source changed')
    for name,expected in metadata['frozen_sha256'].items():
        if sha(name)!=expected:raise ValueError(f'frozen input changed: {name}')
    return metadata


def prepare(act_root,root):
    act_root=act_root.resolve();root=root.absolute()
    if root.parent!=EXPORTS or root.is_symlink() or root.exists():
        raise ValueError('fresh output must be a new independent exports directory')
    if not base.act_complete(act_root):raise ValueError('paired ACT batch is not complete')
    plan=json.loads((act_root/'episode_plan.json').read_text());base.validate_plan(plan)
    act_meta=json.loads((act_root/'metadata.json').read_text())
    if sha(act_root/'episode_plan.json')!=act_meta['episode_plan_sha256'] or sha(act_root/'evaluation_config.yaml')!=act_meta['config_sha256']:
        raise ValueError('paired frozen plan or config changed')
    if base._source_fingerprints(PROJECT)!=act_meta['source_fingerprints']:
        raise ValueError('paired simulation source changed')
    act_rows=[json.loads(line) for line in (act_root/'episodes.jsonl').read_text().splitlines()]
    lookup={p['key']:p for p in plan}
    if len(act_rows)!=120 or len({r['key'] for r in act_rows})!=120:raise ValueError('paired ACT ledger incomplete')
    for row in act_rows:
        for key in ('seed','target_count','layout_fingerprint'):
            if row[key]!=lookup[row['key']][key]:raise ValueError('ACT pairing changed')
    root.mkdir()
    for name in ('episode_plan.json','evaluation_config.yaml'):shutil.copyfile(act_root/name,root/name)
    frozen=[Path(__file__).resolve(),Path(base.__file__).resolve(),root/'episode_plan.json',root/'evaluation_config.yaml',
            act_root/'episode_plan.json',act_root/'evaluation_config.yaml',act_root/'metadata.json',act_root/'episodes.jsonl',
            act_root/'status.json',act_root/'summary.json']
    # Protect previous benchmark history without re-reading weights or media.
    for directory in ('batch_astar_single_attempt_120_paired_20261014','batch_hybrid_trial4_100k_120_paired_20261014'):
        for name in ('episodes.jsonl','status.json','summary.json','metadata.json'):
            path=EXPORTS/directory/name
            if path.exists():frozen.append(path)
    metadata={'kind':'fresh_astar_single_attempt_measured_errors_v2','paired_act_batch':str(act_root),
        'sources_sha256':act_meta['source_fingerprints'],'frozen_sha256':{str(p):sha(p) for p in frozen},
        'model_label':LABEL,'planned':120,'targets':list(range(1,7)),'episodes_per_target':20,
        'frame_budget':2000,'single_geometric_plan':True,'physical_attempt_limit':1,
        'no_physical_screening_or_recovery':True,'native_success_contract':CONTRACT,
        'all_failure_counts':'measured live before environment closes; never synthetic zero',
        'render_RGB':False,'ACT_loaded':False,'training_forbidden':True,'no_automatic_restart':True,
        'fresh_rerun_explicitly_authorized':True,'old_batches_are_history_not_imported_rows':True,
        'same_120_layouts_reused':True,'not_an_unseen_generalization_test':True,
        'tuning_layout_reused_seed':2120018,
        'created_at':time.time()}
    write_json(root/'metadata.json',metadata);write_json(root/'seal.json',{'metadata_sha256':sha(root/'metadata.json')})
    write_json(root/'status.json',{'state':'prepared','completed':0,'planned':120})
    render_artifacts(root,[])


def authorize_worker(root,index):
    if not 0<=index<120:raise ValueError('worker index outside plan')
    owner=json.loads((root/'started.json').read_text())
    if owner['pid']!=os.getppid():raise ValueError('worker must belong to the owning parent')
    with (root/'batch.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise ValueError('batch parent does not own its lock')
    plan=json.loads((root/'episode_plan.json').read_text());spec=plan[index]
    marker=json.loads((root/'attempts'/f"seed_{spec['seed']}.json").read_text())
    if marker['key']!=spec['key'] or marker['owner_pid']!=os.getppid():raise ValueError('worker opportunity identity mismatch')
    return spec


def run_batch(root):
    root=root.resolve()
    if root.parent!=EXPORTS:raise ValueError('batch outside exports')
    with (root/'batch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify(root)
        if (root/'started.json').exists() or (root/'episodes.jsonl').exists() or (root/'attempts').exists():
            raise ValueError('fresh batch already started; automatic restart forbidden')
        plan=json.loads((root/'episode_plan.json').read_text());base.validate_plan(plan)
        write_json(root/'started.json',{'pid':os.getpid(),'started_at':time.time()})
        (root/'logs').mkdir();rows=[]
        try:
            for index,spec in enumerate(plan):
                verify(root)
                if shutil.disk_usage(root).free<6*2**30:raise RuntimeError('free disk below 6 GiB')
                with base.inference_lease(on_wait=lambda:write_json(root/'status.json',{'state':'waiting_inference_slot',
                        'completed':len(rows),'planned':120,'current':spec['key'],'pid':os.getpid()})):
                    verify(root);claim_attempt(root,spec)
                    write_json(root/'status.json',{'state':'running','completed':len(rows),'planned':120,
                               'current':spec['key'],'seed':spec['seed'],'pid':os.getpid(),'updated_at':time.time()})
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--out',str(root),'--worker-index',str(index)]
                    with (root/'logs'/f"seed_{spec['seed']}.log").open('x') as log:
                        child=subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,
                            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                    if child.returncode:raise RuntimeError(f"worker exited {child.returncode}; seed {spec['seed']}; no retry")
                row=json.loads((root/'episodes'/f"seed_{spec['seed']}"/'result.json').read_text())
                validate_row(row,spec);verify(root)
                state=Path(row['final_state'])
                if state.parent!=root/'episodes'/f"seed_{spec['seed']}" or sha(state)!=row['final_state_sha256']:
                    raise ValueError('terminal evidence mismatch')
                with np.load(state,allow_pickle=False) as values:
                    if int(values['fully_collected_mask'].sum())!=row['final_collected']:
                        raise ValueError('terminal count differs from retained measurement')
                with (root/'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                rows.append(row);render_artifacts(root,rows)
                print(json.dumps({'event':'astar_episode_finished','completed':len(rows),'seed':spec['seed'],
                    'success':row['success'],'final_collected':row['final_collected'],'outcome':row['outcome']}),flush=True)
            verify(root)
            if len(rows)!=120 or Counter(r['target_count'] for r in rows)!=Counter({t:20 for t in range(1,7)}):
                raise ValueError('incomplete opportunity ledger')
            write_json(root/'status.json',{'state':'complete','completed':120,'planned':120,'pid':os.getpid(),'updated_at':time.time()})
        except BaseException as exc:
            write_json(root/'status.json',{'state':'needs_attention','completed':len(rows),'planned':120,
                'current':spec['key'] if 'spec' in locals() else None,'error':repr(exc),
                'automatic_retry_forbidden':True,'pid':os.getpid(),'updated_at':time.time()})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--act-batch');parser.add_argument('--out',required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker-index',type=int)
    args=parser.parse_args();root=Path(args.out)
    if args.prepare_only:
        if not args.act_batch:parser.error('--act-batch required for preparation')
        prepare(Path(args.act_batch),root)
    elif args.worker_index is not None:
        root=root.resolve();verify(root);spec=authorize_worker(root,args.worker_index)
        cfg=base.load_config(str(root/'evaluation_config.yaml'))
        run_trial(cfg,spec,root/'episodes'/f"seed_{spec['seed']}");verify(root)
    else:run_batch(root)


if __name__=='__main__':main()
