"""Immutable observation and scene contracts; absent metadata means legacy."""
from pathlib import Path
import hashlib
import json
import yaml

LEGACY='five_view_320'
NATIVE='rudra_front_dual_640x480'
_PROFILES={
    LEGACY:dict(views=('corner_xminus_yminus','corner_xminus_yplus','corner_xplus_yminus','corner_xplus_yplus','wrist'),
        size=(320,320),identity='bed662c46eda74ad30edd7c68f7e20275f65def25180296c4040a1df06250dc4',frames=203121),
    NATIVE:dict(views=('top','wrist'),size=(640,480),identity='899e044242f9933891c36afeb11dd43dc465557e1632da12886204f3d648793c',frames=207179,
        scene_sha256='5c715a4a76485db80bc8e16338b031251f000ffce36520e3a6bbf519fbbe1a77',
        task_source_sha256='016d16e22048c3dea0df1017603ea220bda3b4d9d4c9091b6b63b5304508a38e',
        task_contract_sha256='ebc52a3290b954cc7d739cbd7310c2417fb94192798f6f17ad8e97625ad64deb')}

def profile_spec(name=None):
    try:return dict(_PROFILES[name or LEGACY])
    except KeyError:raise ValueError(f'unknown observation profile: {name}') from None

def validate_profile_config(config):
    name=config.get('observation_profile',LEGACY);spec=profile_spec(name)
    if name!=LEGACY and (config.get('camera_views')!=list(spec['views']) or config.get('image_size')!=list(spec['size'])
            or config.get('scene_profile_sha256')!=spec['scene_sha256'] or config.get('backbone')!='transformer'):
        raise ValueError('native observation/scene contract mismatch')
    return spec

def validate_task(cfg,name=None):
    spec=profile_spec(name)
    if (tuple(cfg.act.camera_views)!=spec['views'] or tuple(cfg.act.image_size)!=spec['size']
            or float(cfg.act.action_hz)!=25 or float(cfg.act.camera_hz)!=25 or float(cfg.act.query_hz)!=5
            or int(cfg.act.execute_steps)!=5 or int(cfg.act.action_dim)!=4):
        raise ValueError('DP observation/action contract mismatch')
    if name==NATIVE:
        scene=getattr(cfg,'scene',None)
        if (scene is None or scene.backend!='rudra_native' or scene.contact_profile!='trial4_1n'
                or hashlib.sha256(Path(scene.profile).read_bytes()).hexdigest()!=spec['scene_sha256']):
            raise ValueError('native scene/contact contract mismatch')
        profile=yaml.safe_load(Path(scene.profile).read_text())
        source=Path(profile['sweep_task']['source_config'])
        if not source.is_absolute():source=Path(__file__).resolve().parents[2]/source
        if (hashlib.sha256(source.read_bytes()).hexdigest()!=spec['task_source_sha256']
                or task_contract_sha256(cfg)!=spec['task_contract_sha256']):
            raise ValueError('native frozen physical/state contract mismatch')


def task_contract_sha256(cfg):
    """Operational paths, seed and requested target may vary; semantics may not."""
    data=cfg.to_dict()
    sections=('sim','table','target','end_effector','workspace','controller',
              'components','perception','planner','task','episode')
    value={k:data[k] for k in sections}
    value['task']=dict(value['task']);value['task'].pop('target_count',None)
    fields=('image_size','camera_hz','query_hz','action_hz','chunk_size','execute_steps',
            'horizon_seconds','state_history','state_history_seconds','camera_views',
            'ordinary_target_count_only','action_dim','state_dim')
    value['act']={k:data['act'][k] for k in fields}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
