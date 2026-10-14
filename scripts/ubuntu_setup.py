"""Read-only USB dataset checks and explicit Ubuntu configuration generation.

No model is loaded, no simulation is stepped, and no training job is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def portable_config(cfg,*,dataset,model_dir,device):
    if device not in {'cpu','cuda'}:raise ValueError('choose cpu or cuda explicitly on Ubuntu')
    if not str(dataset).strip() or not str(model_dir).strip():raise ValueError('dataset/model paths required')
    result=cfg.copy()
    for key,value in {'act.device':device,'act.dataset_dir':str(dataset),
                      'act.model_dir':str(model_dir),'act.tracking_project':None,
                      'act.tracking_space':None,'web.inference_video_dir':'runs/inference_videos',
                      'web.inference_model_root':str(model_dir)}.items():result.set_path(key,value)
    return result


def contained_file(root,value):
    if not isinstance(value,str) or not value or Path(value).is_absolute():
        raise ValueError('dataset payload paths must be relative to the dataset root')
    path=(root/value).resolve()
    if not path.is_relative_to(root):raise ValueError('dataset payload path escapes its root')
    if not path.is_file():raise ValueError(f'missing imported payload: {value}')
    return path


def inspect_dataset(root,*,max_episodes=3,check_images=False):
    """Report bounded checks honestly; --all-episodes opts into full payload I/O."""
    root=Path(root).resolve()
    if max_episodes is not None and max_episodes<1:raise ValueError('positive sample limit required')
    records=[]
    with (root/'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():records.append(json.loads(line))
    if not records:raise ValueError('empty dataset manifest')
    identifiers=[r.get('episode_id') for r in records]
    if any(not isinstance(v,str) or not v for v in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('missing or duplicate episode identity')
    counts=Counter()
    for record in records:
        if (record.get('schema_version')!=5 or record.get('state_dim')!=36 or record.get('action_dim')!=4):
            raise ValueError('expected schema-v5/36-D state/4-D actions; legacy data require explicit conversion')
        target=record.get('target_count')
        if type(target) is not int or not 1<=target<=6:raise ValueError('unknown or invalid target count')
        if type(record.get('length')) is not int or record['length']<1:
            raise ValueError('positive integer episode length required')
        if record.get('episode_kind')=='expert' and record.get('split')=='train' and record.get('success') is True:
            counts[str(target)]+=1
    selected=records if max_episodes is None else records[:max_episodes]
    for record in selected:
        arrays=contained_file(root,record.get('arrays'))
        with np.load(arrays,allow_pickle=False) as payload:
            length=int(record['length'])
            if length<1:raise ValueError('empty episode')
            for name,width in [('state',36),('action',4)]:
                value=payload[name]
                if value.shape!=(length,width) or value.dtype!=np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'invalid {name} shape/type/finite values: {record["episode_id"]}')
            goal=payload['environment_state']
            if (goal.shape not in ((length,1),(length,3)) or goal.dtype!=np.float32
                or not np.isfinite(goal).all()):raise ValueError('invalid task-conditioning payload')
            target_column=0 if goal.shape[1]==1 else 1
            if not np.all(goal[:,target_column]==record['target_count']):raise ValueError('goal differs from manifest')
            time=payload['t']
            if time.shape!=(length,) or not np.isfinite(time).all() or (length>1 and not np.all(np.diff(time)>0)):
                raise ValueError('invalid episode time alignment')
            if ('action_valid' not in payload or payload['action_valid'].shape!=(length,)
                or payload['action_valid'].dtype!=np.bool_):
                raise ValueError('explicit boolean action_valid supervision mask required for imported data')
        images=record.get('images') or {k:record[k] for k in ('overhead','wrist') if k in record}
        if not isinstance(images,dict) or not images:raise ValueError('missing image view mapping')
        for view,paths in images.items():
            if not isinstance(paths,list) or len(paths)!=length:raise ValueError(f'image/array alignment: {view}')
            for relative in paths:
                path=contained_file(root,relative)
                if check_images:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.load()
                        if image.size!=(320,320) or image.mode!='RGB':raise ValueError('expected 320x320 RGB policy image')
    return {'records':len(records),'schema_version':5,
            'train_success_by_target':{str(i):counts[str(i)] for i in range(1,7)},
            'payload_episodes_checked':len(selected),'fully_checked':len(selected)==len(records) and check_images,
            'image_pixels_decoded':check_images,
            'scope':'paths/array alignment/finite values/masks/task widths; optional image decode',
            'not_a_split_leakage_or_trace_hash_audit':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    config=commands.add_parser('config',help='write a new full config, never edit the shared default')
    config.add_argument('--base',default=str(PROJECT/'configs/five-view-fine-teached-300.yaml'))
    config.add_argument('--dataset',required=True);config.add_argument('--model-dir',required=True)
    config.add_argument('--device',choices=['cuda','cpu'],required=True);config.add_argument('--out',required=True)
    check=commands.add_parser('check-data',help='read-only import checks; no policy or simulator')
    check.add_argument('--dataset',required=True);check.add_argument('--max-episodes',type=int,default=3)
    check.add_argument('--all-episodes',action='store_true');check.add_argument('--check-images',action='store_true')
    check.add_argument('--formal-config',help='also enforce the existing ACT balanced-success metadata gate')
    args=parser.parse_args(argv)
    from sim.config import load_config
    if args.command=='config':
        cfg=portable_config(load_config(args.base),dataset=args.dataset,model_dir=args.model_dir,device=args.device)
        content=cfg.to_dict();content.pop('_source_config',None)
        output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
        import yaml
        with output.open('x',encoding='utf-8') as stream:yaml.safe_dump(content,stream,sort_keys=False)
        print(json.dumps({'configuration':str(output.resolve()),'device':args.device,'training_started':False}))
    else:
        report=inspect_dataset(args.dataset,max_episodes=None if args.all_episodes else args.max_episodes,check_images=args.check_images)
        if args.formal_config:
            from sim.act.collect_dataset import validate_training_manifest
            cfg=load_config(args.formal_config)
            report['ACT_formal_metadata_gate']=validate_training_manifest(args.dataset,
                episodes_per_target=int(cfg.act.training_episodes_per_target),camera_views=list(cfg.act.camera_views),
                target_subset_mode=str(cfg.act.get('target_subset_mode','any')))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
