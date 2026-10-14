"""Migration workbench behavior; stored fixtures only, no rollout or training."""
import json
from pathlib import Path
import numpy as np
from sim.act.dataset import ActDatasetWriter
from sim.act.expert import expert_generation_protocol
from sim.config import load_config
from sim.web.workbench import WorkbenchState


def test_goal_only_wrong_count_clone_rewrites_nested_image_paths(tmp_path):
    cfg=load_config();previews=tmp_path/'previews'
    images={name:np.zeros((12,12,3),np.uint8) for name in cfg.act.camera_views}
    ActDatasetWriter(previews).add_episode('source_goal2',[
        {'images':images,'state':np.zeros(36,np.float32),'environment_state':np.array([2.],np.float32),
         'phase':'sweep'}],np.zeros((1,4),np.float32),True,
        {'preview':True,'seed':4,'target_count':2,'collected':2,'total_count':6,'fps':25.,
         'failure_reason':'','episode_kind':'expert_preview','split':'pilot',
         'expert_generation_protocol':expert_generation_protocol(cfg)})
    before=(previews/'source_goal2/arrays.npz').read_bytes()
    workbench=WorkbenchState(cfg,dataset_root=tmp_path/'data',preview_root=previews)
    result=workbench._clone_success_as_wrong_count(seed=7,target_count=1,failure_mode='wrong_count_over')
    records=[json.loads(line) for line in (previews/'manifest.jsonl').read_text().splitlines()]
    cloned=records[-1]
    assert cloned['success'] is False and cloned['derived_from']=='source_goal2'
    assert cloned['target_count']==1
    with np.load(previews/cloned['arrays']) as payload:
        assert payload['environment_state'].shape==(1,1)
        assert payload['environment_state'][0,0]==1.
    for paths in cloned['images'].values():
        assert all(Path(path).parts[0]==cloned['episode_id'] for path in paths)
        assert all((previews/path).is_file() for path in paths)
    assert (previews/'source_goal2/arrays.npz').read_bytes()==before
