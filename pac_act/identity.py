"""Portable frozen-control/model identity and disjoint layout registries."""
import hashlib
import json
from pathlib import Path
import shutil

from .storage import sha256,write_json

PROJECT=Path(__file__).resolve().parents[1]
RANGES={'train':(6_000_000,7_000_000),'validation':(7_000_000,8_000_000),
        'preflight':(8_000_000,9_000_000)}
CONTROL_PROFILE_IDS={'dinov3_sync5'}


def controller_profile_id(manifest):
    """Resolve an explicitly bundled controller profile; absence is legacy."""
    value=manifest.get('control_profile_id')
    if value is None:
        return None
    if not isinstance(value,str) or value not in CONTROL_PROFILE_IDS:
        raise ValueError('unknown controller profile')
    return value


class FileIdentityWatch:
    """Cheap running guard; rehash only changed files, never modify inputs."""
    def __init__(self,files):
        self.files={Path(p):digest for p,digest in files.items()}
        self.signatures={p:self.signature(p) for p in self.files}
    @staticmethod
    def signature(path):
        s=path.stat(); return s.st_size,s.st_mtime_ns,s.st_ino
    def __call__(self):
        for path,expected in self.files.items():
            try:
                current=self.signature(path)
                if current!=self.signatures[path]:
                    if sha256(path)!=expected: raise RuntimeError('runtime identity drift: '+str(path))
                    self.signatures[path]=current
            except OSError as error:
                raise RuntimeError('runtime identity drift: '+str(path)) from error


def watch_bundle(root,frozen):
    files={Path(root)/name:digest for name,digest in frozen['manifest']['files'].items()}
    files.update({PROJECT/name:digest for name,digest in frozen['manifest']['sources'].items()})
    files[Path(root)/'manifest.json']=sha256(Path(root)/'manifest.json')
    return FileIdentityWatch(files)


def seed_spec(role,index,arm='actor'):
    start,end=RANGES[role]; seed=start+index
    if index<0 or seed>=end: raise ValueError('seed outside registered interval')
    return dict(role=role,seed=seed,target_count=index%6+1,arm=arm)


class LayoutRegistry:
    def __init__(self,excluded):
        self.excluded_seeds=set(excluded.get('seeds',[]))
        self.excluded_hashes=set(excluded.get('fingerprints',[]))
        self.used={}

    def register(self,spec,fingerprint):
        seed=spec['seed']; role=spec['role']
        if seed in self.excluded_seeds or fingerprint in self.excluded_hashes or 2_120_000<=seed<2_120_120 or 5_000_000<=seed<6_000_000:
            raise ValueError('held-out, residual, demonstration or prior layout collision')
        if role in RANGES and not RANGES[role][0]<=seed<RANGES[role][1]: raise ValueError('wrong seed role')
        prior=self.used.get(fingerprint)
        if prior and not (prior['seed']==seed and role==prior['role'] and role in ('validation','preflight')):
            raise ValueError('layout fingerprint collision')
        if any(v['seed']==seed and k!=fingerprint for k,v in self.used.items()):
            raise ValueError('seed initial pose changed')
        self.used[fingerprint]=spec.copy()


def source_files(project=PROJECT):
    paths=list((project/'sim').rglob('*.py'))+list((project/'pac_act').rglob('*.py'))
    paths+=list((project/'hybrid_trial4_snapshot').rglob('*'))
    paths+=[project/'scripts/train_pac_act.py',project/'scripts/queue_pac_preflight.py',project/'scripts/evaluate_hybrid_paired_batch.py']
    paths+=[project/'residual_rl'/name for name in ('bridge.py','rewards.py','storage.py')]
    paths+=list((project/'pac_act').glob('*.html'))+list(project.glob('requirements-pac-act*.txt'))
    paths+=[project/'pyproject.toml']
    paths+=list((project/'configs').glob('*.yaml'))
    paths+=list((project/'sim/model/assets').rglob('*'))
    return {str(p.relative_to(project)):sha256(p) for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts}


def bundle_digest(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def copy_source_tree(project,code):
    code=Path(code); code.mkdir()
    for name in ('sim','configs','pac_act','hybrid_trial4_snapshot'):
        shutil.copytree(project/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (code/'scripts').mkdir(); (code/'residual_rl').mkdir()
    for name in ('train_pac_act.py','queue_pac_preflight.py','monitor_pac_act.py','package_pac_act.py','prepare_pac_act_exclusions.py','evaluate_hybrid_paired_batch.py'):
        shutil.copyfile(project/'scripts'/name,code/'scripts'/name)
    for name in ('__init__.py','bridge.py','rewards.py','storage.py'):
        shutil.copyfile(project/'residual_rl'/name,code/'residual_rl'/name)
    for path in [project/'pyproject.toml',project/'README.md',*project.glob('requirements-pac-act*.txt')]:
        shutil.copyfile(path,code/path.name)


def _dino_identity(config_path):
    from sim.act.configuration_dinov3 import (
        DINO_MODEL_ID,DINO_REVISION,DINO_WEIGHT_SHA256,
    )

    try:
        config=json.loads(Path(config_path).read_text())
    except (OSError,json.JSONDecodeError) as error:
        raise ValueError('DINOv3 200K checkpoint config is missing or invalid') from error
    expected={
        'type':'act_dinov3',
        'vision_backbone':'dinov3_vits16',
        'dino_model_id':DINO_MODEL_ID,
        'dino_revision':DINO_REVISION,
        'dino_weight_sha256':DINO_WEIGHT_SHA256,
    }
    if any(config.get(key)!=value for key,value in expected.items()):
        raise ValueError('DINOv3 checkpoint differs from the pinned 200K identity')
    return dict(model_type=expected['type'],vision_backbone=expected['vision_backbone'],
        dino_model_id=expected['dino_model_id'],dino_revision=expected['dino_revision'],
        dino_weight_sha256=expected['dino_weight_sha256'])


def _checkpoint_files(root):
    root=Path(root).resolve()
    if not root.is_dir(): raise ValueError('checkpoint directory is missing')
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('checkpoint files may not be symlinks')
        if path.is_file(): result[path.relative_to(root).as_posix()]=path
    return result


def _validate_dino_checkpoint(root):
    root=Path(root).resolve()
    identity=_dino_identity(root/'config.json')
    files=_checkpoint_files(root)
    required={
        'config.json','model.safetensors','policy_preprocessor.json',
        'policy_postprocessor.json','training_state.pt',
    }
    missing=required-set(files)
    if missing: raise ValueError('DINOv3 checkpoint is incomplete: '+', '.join(sorted(missing)))
    for filename,registry in (
        ('policy_preprocessor.json','normalizer_processor'),
        ('policy_postprocessor.json','unnormalizer_processor'),
    ):
        try:
            processor=json.loads(files[filename].read_text())
        except (OSError,json.JSONDecodeError) as error:
            raise ValueError(f'invalid DINOv3 processor graph: {filename}') from error
        matches=[step for step in processor.get('steps',[])
                 if step.get('registry_name')==registry]
        if len(matches)!=1: raise ValueError(f'invalid DINOv3 processor state reference: {filename}')
        state_file=matches[0].get('state_file')
        relative=Path(state_file) if isinstance(state_file,str) else None
        if (relative is None or relative.is_absolute() or '..' in relative.parts
                or relative.as_posix() not in files):
            raise ValueError(f'missing or unsafe DINOv3 processor state: {filename}')
    return identity,files


def build_bundle(destination,model,excluded,project=PROJECT):
    destination=Path(destination); model=Path(model)
    if destination.exists(): raise FileExistsError('bundle must be new')
    if model.name=='step_100000':
        try:
            saved=json.loads((model/'config.json').read_text())
        except (OSError,json.JSONDecodeError) as error:
            raise ValueError('100K base checkpoint config is missing or invalid') from error
        if saved.get('type')=='act_dinov3' or saved.get('vision_backbone')=='dinov3_vits16':
            raise ValueError('DINOv3 base checkpoint must be step_200000')
        dino_identity=None
    elif model.name=='step_200000':
        dino_identity,checkpoint_files=_validate_dino_checkpoint(model)
    else:
        raise ValueError('PAC-ACT base checkpoint must be step_100000 ResNet or step_200000 DINOv3')
    # Exclusions are portable evidence, not external paths consulted on the GPU.
    if not excluded.get('complete_known_inputs') or not excluded.get('fingerprints'):
        raise ValueError('explicit complete known-layout exclusion manifest required')
    destination.mkdir(parents=True); target=destination/'base_model'; target.mkdir()
    files={}
    if dino_identity is None:
        for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json',
                     'policy_preprocessor_step_3_normalizer_processor.safetensors',
                     'policy_postprocessor_step_0_unnormalizer_processor.safetensors'):
            shutil.copyfile(model/name,target/name); files['base_model/'+name]=sha256(target/name)
        snapshot=project/'hybrid_trial4_snapshot'
        for name in ('evaluation_config.yaml','control_profile.json'):
            shutil.copyfile(snapshot/name,destination/name); files[name]=sha256(destination/name)
    else:
        checkpoint_digests={}
        for name,source in checkpoint_files.items():
            copied=target/name
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,copied)
            digest=sha256(copied)
            checkpoint_digests[name]=digest
            files['base_model/'+name]=digest
        profile_source=project/'hybrid_trial4_snapshot'/'dinov3_sync5'
        for name in ('control_profile.json','evaluation_config.yaml'):
            shutil.copyfile(profile_source/name,destination/name)
            files[name]=sha256(destination/name)
        from .dinov3_control import load_sync5_profile
        load_sync5_profile(destination)
    write_json(destination/'exclusions.json',excluded); files['exclusions.json']=sha256(destination/'exclusions.json')
    # A runnable source tree travels with the model. No reference to a Mac
    # checkout, Git object cache or historical batch directory is needed.
    copy_source_tree(project,destination/'source')
    manifest=dict(schema=1,base_step=100000,model='base_model',files=files,sources=source_files(project),
        reference_commit='1a8c9db3c551ce776091bff6203746b439219c78',
        schedule=dict(ranges=RANGES,random_seed=42,validation=[seed_spec('validation',i) for i in range(24)]))
    if dino_identity is not None:
        manifest.update(dino_identity)
        manifest.update(base_step=200000,checkpoint_files=checkpoint_digests,
            control_profile_id='dinov3_sync5',
            control_profile_files={name:files[name] for name in
                ('control_profile.json','evaluation_config.yaml')})
    write_json(destination/'manifest.json',manifest)
    return verify_bundle(destination,project)


def verify_bundle(root,project=PROJECT):
    root=Path(root).resolve(); data=json.loads((root/'manifest.json').read_text())
    has_dino_identity=any(key in data for key in (
        'model_type','vision_backbone','dino_model_id','dino_revision',
        'dino_weight_sha256','checkpoint_files','control_profile_id',
    ))
    if data.get('model_type')=='act_dinov3':
        if data.get('base_step')!=200000 or data.get('model')!='base_model':
            raise ValueError('DINOv3 bundle must identify the 200K base checkpoint')
        identity,actual_files=_validate_dino_checkpoint(root/data['model'])
        if any(data.get(key)!=value for key,value in identity.items()):
            raise ValueError('DINOv3 bundle pinned model identity mismatch')
        checkpoint_files=data.get('checkpoint_files')
        actual_digests={name:sha256(path) for name,path in actual_files.items()}
        if checkpoint_files!=actual_digests:
            raise ValueError('DINOv3 checkpoint file identity mismatch')
        for name,digest in checkpoint_files.items():
            if data.get('files',{}).get(f"{data['model']}/{name}")!=digest:
                raise ValueError('DINOv3 bundle file manifest mismatch: '+name)
        from .dinov3_control import SYNC5_PROFILE_ID,load_sync5_profile
        if controller_profile_id(data)!=SYNC5_PROFILE_ID:
            raise ValueError('DINOv3 bundle controller profile mismatch')
        profile_files=data.get('control_profile_files')
        expected_profile_files={name:data.get('files',{}).get(name) for name in
            ('control_profile.json','evaluation_config.yaml')}
        if profile_files!=expected_profile_files or any(value is None for value in profile_files.values()):
            raise ValueError('DINOv3 controller profile digest manifest mismatch')
        load_sync5_profile(root)
    elif has_dino_identity or data.get('base_step')!=100000 or data.get('model')!='base_model':
        raise ValueError('wrong or unsupported PAC-ACT base model')
    for name,expected in data['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256(path)!=expected: raise ValueError('bundle identity mismatch: '+name)
    if source_files(project)!=data['sources']: raise ValueError('executable source identity drift')
    source=json.loads((project/'hybrid_trial4_snapshot/source_manifest.json').read_text())
    for name,expected in source['source_fingerprints'].items():
        if sha256(project/name)!=expected: raise ValueError('frozen sim drift: '+name)
    for name,expected in source['files'].items():
        if sha256(project/name)!=expected['sha256']: raise ValueError('frozen Hybrid drift: '+name)
    return dict(identity=bundle_digest(data),manifest=data,model=root/data['model'])
