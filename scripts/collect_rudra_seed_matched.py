"""Collect the source dataset's exact 300 goal/seed pairs in the new scene."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_rudra_six_demos import checked_configuration, run_candidate, terminal_failure_status, write_json
from sim.act.collect_dataset import EpisodeSpec, _expert_metadata, validate_training_manifest
from sim.act.dataset import ActDatasetWriter
from sim.act.expert import expert_generation_protocol
from sim.config import load_config, save_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_source_plan(path):
    path=Path(path).resolve()
    records=read_jsonl(path)
    valid=lambda r:(r.get('schema_version')==5 and r.get('episode_kind')=='expert'
                    and r.get('split')=='train' and r.get('success') is True
                    and r.get('action_dim')==4 and r.get('state_dim')==36
                    and isinstance(r.get('frame_count'),int) and r['frame_count']>0
                    and isinstance(r.get('seed'),int) and 0<=r['seed']<=4294967295
                    and r.get('layout_kind')=='independent' and bool(r.get('layout_id')))
    if (len(records)!=300 or not all(valid(r) for r in records)
            or Counter(r.get('target_count') for r in records)!={n:50 for n in range(1,7)}
            or len({r['seed'] for r in records})!=300
            or len({r['episode_id'] for r in records})!=300):
        raise ValueError('source must contain 300 successful unique experts, 50 per target')
    entries=[]
    for ordinal,r in enumerate(records,1):
        episode_id=str(r['episode_id'])
        if not episode_id.startswith(f"preview_goal_{r['target_count']}_") or Path(episode_id).name!=episode_id:
            raise ValueError('invalid source episode identity')
        entries.append(dict(ordinal=ordinal,episode_id=episode_id,target_count=r['target_count'],
                            seed=r['seed'],layout_id=r['layout_id'],
                            source_requested_seed=r.get('requested_seed',r['seed']),
                            source_generation_attempt=r.get('generation_attempt',1),
                            source_record_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()))
    return dict(source_manifest=str(path),source_manifest_sha256=sha256(path),
                seed_policy='exact source actual seeds; no substitution',entries=entries)


def verify_frozen_files(out,pinned):
    for name,digest in pinned.items():
        if not (out/name).is_file() or sha256(out/name)!=digest:
            raise ValueError(f'frozen collection file changed: {name}')


def recovered_successes(out,entries,source_sha,camera_sha,allowed_attempts=None):
    expected={r['episode_id']:r for r in entries}
    done={}
    for record in read_jsonl(out/'manifest.jsonl'):
        key=record.get('episode_id');slot=expected.get(key)
        permitted=({slot['seed']:1} if slot is not None else {}) if allowed_attempts is None else allowed_attempts.get(key,{})
        replaced=slot is not None and record.get('seed')!=slot['seed']
        if (slot is None or key in done or record.get('seed') not in permitted
                or (replaced and (record.get('source_seed')!=slot['seed']
                                  or record.get('requested_seed')!=slot['seed']
                                  or record.get('generation_attempt')!=permitted.get(record.get('seed'))))
                or record.get('target_count')!=slot['target_count']
                or record.get('success') is not True or record.get('schema_version')!=5
                or record.get('frame_count',0)<=0 or record.get('camera_views')!=['top','wrist']
                or record.get('source_manifest_sha256')!=source_sha
                or record.get('scene_profile_sha256')!=camera_sha):
            raise ValueError(f'saved episode identity does not match frozen plan: {key}')
        done[key]=record
    return done


def capture_one(cfg,slot,out,writer,source_sha,camera_sha):
    local=cfg.copy();local.set_path('task.target_count',slot['target_count'])
    result,planning_failure=run_candidate(local,slot['seed'])
    identity={k:slot[k] for k in ['episode_id','target_count','seed']}
    source_seed=slot.get('source_seed',slot['seed']);attempt=slot.get('generation_attempt',1)
    identity.update(source_seed=source_seed,generation_attempt=attempt)
    if result is None or not result.success:
        failure=(planning_failure if result is None else
                 dict(reason=result.failure_reason,collected=result.collected,frames=result.sampled_frames))
        return dict(identity,state='failed',**failure)
    spec=EpisodeSpec('train',slot['target_count'],slot['seed'],slot['layout_id'],'independent')
    metadata=_expert_metadata(spec,result,source_seed,attempt,local)
    metadata.update(dataset_kind='rudra_native_front_seed_matched_300',scene_backend='rudra_native',
                    scene_profile=str(cfg.scene.profile),scene_profile_sha256=camera_sha,
                    source_manifest_sha256=source_sha,source_episode_id=slot['episode_id'],
                    source_requested_seed=slot.get('source_requested_seed',slot['seed']),
                    source_generation_attempt=slot.get('source_generation_attempt',1),
                    source_record_sha256=slot.get('source_record_sha256'),
                    source_seed=source_seed,seed_replaced=slot['seed']!=source_seed,
                    camera_views=['top','wrist'],contact_profile='trial4_1n',
                    brush_sole_friction=[.1,.0001,.0001],
                    admittance_seek_MBK=[10.,400.,10.],admittance_maintain_MBK=[10.,800.,10.],
                    coordinate_frame='original fixed sweep task frame',
                    task_to_native_world_translation_m=[.005,-.685,-.002])
    if slot['seed']!=source_seed:metadata['dataset_kind']='rudra_native_front_seed_replacement_300'
    writer.add_episode(slot['episode_id'],result.observations,result.actions,True,metadata,control_trace=result.trace)
    return dict(identity,state='saved',collected=result.collected,frames=result.sampled_frames,
                peak_force=result.peak_force,peak_raw_force=result.peak_raw_force)


def event(out,kind,**fields):
    row=dict(event=kind,time_utc=datetime.now(timezone.utc).isoformat(),**fields)
    with (out/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(row,ensure_ascii=False),flush=True)


def prepare(out,source,gate):
    plan=load_source_plan(source)
    cfg=checked_configuration(gate)
    records=read_jsonl(Path(source))
    if any(r.get('expert_generation_protocol')!=expert_generation_protocol(cfg) for r in records):
        raise ValueError('source layout/planning protocol differs from current task')
    shutil.copyfile(source,out/'source-manifest.jsonl')
    shutil.copyfile(gate,out/'contact-gate.json')
    profile=yaml.safe_load(Path(cfg.scene.profile).read_text())
    # Save the native rectangular camera output without policy-size resizing.
    cfg.set_path('act.image_size',[int(profile['render']['width']),int(profile['render']['height'])])
    task=Path(profile['sweep_task']['source_config'])
    if not task.is_absolute():task=Path(__file__).resolve().parents[1]/task
    shutil.copyfile(task,out/'task-source.yaml')
    # Native scene reset reloads this task independently of the collector config.
    profile['sweep_task']['source_config']=str(out/'task-source.yaml')
    (out/'scene-profile.yaml').write_text(yaml.safe_dump(profile,sort_keys=False))
    cfg.set_path('scene.profile',str(out/'scene-profile.yaml'))
    save_config(cfg,str(out/'collection-config.yaml'))
    write_json(out/'plan.json',plan)
    files=['source-manifest.jsonl','contact-gate.json','scene-profile.yaml','task-source.yaml','collection-config.yaml','plan.json']
    write_json(out/'provenance.json',dict(files={name:sha256(out/name) for name in files},
                                         source_manifest_sha256=plan['source_manifest_sha256'],
                                         scene_profile_sha256=sha256(out/'scene-profile.yaml')))
    write_json(out/'progress.json',dict(state='prepared',saved=0,processed=0,failed=0,total=300))
    return cfg,plan,json.loads((out/'provenance.json').read_text())


def collect(cfg,plan,pinned,out):
    entries=plan['entries'];source_sha=pinned['source_manifest_sha256'];camera_sha=pinned['scene_profile_sha256']
    done=recovered_successes(out,entries,source_sha,camera_sha)
    failed={r['episode_id']:r for r in read_jsonl(out/'events.jsonl') if r['event']=='episode_failed'}
    failed={key:value for key,value in failed.items() if key not in done}
    if any(key not in {x['episode_id'] for x in entries} for key in failed):
        raise ValueError('failure ledger contains an identity outside the frozen plan')
    writer=ActDatasetWriter(str(out),image_format='webp-lossless')
    for slot in entries:
        key=slot['episode_id']
        if key in done or key in failed:continue
        verify_frozen_files(out,pinned['files'])
        if shutil.disk_usage(out).free<12*1024**3:
            raise RuntimeError('less than 12 GiB free; collection stopped before starting another layout')
        status=dict(state='running',saved=len(done),processed=len(done)+len(failed),failed=len(failed),
                    total=300,current=slot,per_target={str(n):sum(r['target_count']==n for r in done.values()) for n in range(1,7)})
        write_json(out/'progress.json',status)
        event(out,'episode_started',**slot)
        result=capture_one(cfg,slot,out,writer,source_sha,camera_sha)
        if result['state']=='saved':
            done[key]=dict(target_count=slot['target_count'])
            event(out,'episode_saved',**result)
        else:
            failed[key]=result
            event(out,'episode_failed',**result)
        write_json(out/'progress.json',dict(state='running',saved=len(done),processed=len(done)+len(failed),
                                            failed=len(failed),total=300,last_result=result,
                                            per_target={str(n):sum(r['target_count']==n for r in done.values()) for n in range(1,7)}))
        gc.collect()
    state='needs_attention' if failed else 'complete'
    if not failed:
        validate_training_manifest(out,episodes_per_target=50,camera_views=['top','wrist'],
                                   target_subset_mode='y_edge_contiguous')
    summary=dict(state=state,saved=len(done),processed=len(done)+len(failed),failed=len(failed),total=300,
                 failures=list(failed.values()),per_target={str(n):sum(r['target_count']==n for r in done.values()) for n in range(1,7)})
    write_json(out/'summary.json',summary);write_json(out/'progress.json',summary)
    event(out,'collection_finished',**summary)
    return 2 if failed else 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-manifest',default='/home/luke/dataset-ur10/manifest.jsonl')
    parser.add_argument('--gate-report')
    parser.add_argument('--out',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=Path(args.out).resolve()
    if args.resume:
        if not out.is_dir():raise ValueError('resume directory does not exist')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('new seed-matched collection requires an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with terminal_failure_status(out):
            if args.resume:
                pinned=json.loads((out/'provenance.json').read_text());verify_frozen_files(out,pinned['files'])
                plan=json.loads((out/'plan.json').read_text())
                if load_source_plan(args.source_manifest)!=plan:
                    raise ValueError('source manifest changed since the collection plan was frozen')
                cfg=load_config(str(out/'collection-config.yaml'))
            else:
                if not args.gate_report:raise ValueError('--gate-report is required for a new collection')
                cfg,plan,pinned=prepare(out,args.source_manifest,args.gate_report)
            if args.prepare_only:
                print(json.dumps(dict(state='prepared',total=len(plan['entries']),source_sha256=plan['source_manifest_sha256'])),flush=True)
                return 0
            return collect(cfg,plan,pinned,out)


if __name__=='__main__':
    raise SystemExit(main())
