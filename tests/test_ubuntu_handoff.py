"""Portable handoff checks: no policy, training, physics or network calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def setup_api():
    path=ROOT/'scripts/ubuntu_setup.py'
    if not path.exists():return SimpleNamespace()
    spec=importlib.util.spec_from_file_location('ubuntu_setup_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def call(module,name,*args,**kwargs):
    function=getattr(module,name,None)
    assert callable(function),f'{name} portable behavior missing'
    return function(*args,**kwargs)


def test_ubuntu_config_changes_paths_device_not_physics(tmp_path):
    from sim.config import load_config
    cfg=load_config(str(ROOT/'configs/five-view-fine-teached-300.yaml'))
    before=cfg.to_dict()
    result=call(setup_api(),'portable_config',cfg,dataset=str(tmp_path/'usb_data'),
                model_dir='runs/new_model',device='cuda')
    assert result.act.device=='cuda'
    assert result.act.dataset_dir==str(tmp_path/'usb_data')
    assert result.web.inference_video_dir=='runs/inference_videos'
    for key in ('workspace','controller','task','end_effector','sim','cameras'):
        assert result.to_dict().get(key)==before.get(key)
    assert cfg.to_dict()==before


@pytest.mark.parametrize('device',['mps','auto','cpu:bad'])
def test_unreviewed_device_not_silently_selected(device):
    from sim.config import load_config
    with pytest.raises(ValueError):
        call(setup_api(),'portable_config',load_config(),dataset='datasets/demo',model_dir='runs/model',device=device)


def fixture_dataset(tmp_path):
    root=tmp_path/'data';episode=root/'episode_01';episode.mkdir(parents=True)
    images=[]
    for i in range(2):
        path=episode/f'{i}.png';Image.new('RGB',(320,320),(20,40,60)).save(path)
        images.append(str(path.relative_to(root)))
    np.savez_compressed(episode/'arrays.npz',state=np.zeros((2,36),np.float32),
        action=np.zeros((2,4),np.float32),environment_state=np.ones((2,1),np.float32),
        t=np.array([0.,.04],np.float64),action_valid=np.array([True,False]))
    record={'episode_id':'episode_01','schema_version':5,'state_dim':36,'action_dim':4,
            'episode_kind':'expert','split':'train','success':True,'target_count':1,
            'length':2,'arrays':'episode_01/arrays.npz','images':{'wrist':images}}
    (root/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    return root,record


def test_import_checker_checks_payload_without_touching_data(tmp_path):
    root,_=fixture_dataset(tmp_path);before=(root/'manifest.jsonl').read_bytes()
    report=call(setup_api(),'inspect_dataset',root,max_episodes=3,check_images=True)
    assert report['records']==1
    assert report['payload_episodes_checked']==1
    assert report['fully_checked'] is True
    assert report['train_success_by_target']['1']==1
    assert (root/'manifest.jsonl').read_bytes()==before


def test_sampled_payload_check_not_claimed_full(tmp_path):
    root,record=fixture_dataset(tmp_path)
    second={**record,'episode_id':'other','success':False,'episode_kind':'inference'}
    (root/'manifest.jsonl').write_text(json.dumps(record)+'\n'+json.dumps(second)+'\n')
    report=call(setup_api(),'inspect_dataset',root,max_episodes=1,check_images=False)
    assert report['records']==2 and report['payload_episodes_checked']==1
    assert report['fully_checked'] is False


@pytest.mark.parametrize('relative',['../outside.npz','/absolute/private.npz'])
def test_import_path_escape_rejected(tmp_path,relative):
    root,record=fixture_dataset(tmp_path);record['arrays']=relative
    (root/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError):call(setup_api(),'inspect_dataset',root,max_episodes=3)


def test_nonfinite_state_and_unknown_goal_rejected(tmp_path):
    root,record=fixture_dataset(tmp_path)
    state=np.zeros((2,36),np.float32);state[0,0]=np.nan
    np.savez_compressed(root/record['arrays'],state=state,action=np.zeros((2,4),np.float32),
        environment_state=np.ones((2,1),np.float32),t=[0,.04],action_valid=[True,True])
    with pytest.raises(ValueError):call(setup_api(),'inspect_dataset',root,max_episodes=3)


@pytest.mark.parametrize('mask',[np.array([1.,np.nan]),np.array([1,0],np.int64)])
def test_nonboolean_supervision_mask_rejected(tmp_path,mask):
    root,record=fixture_dataset(tmp_path)
    with np.load(root/record['arrays']) as payload:arrays={name:payload[name] for name in payload.files}
    arrays['action_valid']=mask
    np.savez_compressed(root/record['arrays'],**arrays)
    with pytest.raises(ValueError,match='action_valid'):
        call(setup_api(),'inspect_dataset',root,max_episodes=3)


def test_noninteger_manifest_length_rejected(tmp_path):
    root,record=fixture_dataset(tmp_path);record['length']=2.9
    (root/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='length'):
        call(setup_api(),'inspect_dataset',root,max_episodes=3)


def test_vendor_runtime_imports_from_this_repository_without_old_data():
    path=ROOT/'experiments/hybrid_trial4/runtime.py'
    assert path.exists(),'vendored runtime importer missing'
    spec=importlib.util.spec_from_file_location('portable_runtime_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    runtime=module.load_runtime()
    assert runtime.PROJECT==ROOT
    assert runtime.TRIAL_SPEC.force_N==1.
    assert runtime.TRIAL_SPEC.seek==(10.,400.,10.)
    assert runtime.TRIAL_SPEC.maintain==(10.,800.,10.)
    assert Path(runtime.__file__).is_relative_to(ROOT)


def test_portable_controller_keeps_wrist_force_dwell_and_two_mm_bound():
    path=ROOT/'experiments/hybrid_trial4/runtime.py'
    assert path.exists(),'vendored runtime importer missing'
    spec=importlib.util.spec_from_file_location('portable_runtime_dwell_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    runtime=module.load_runtime();specification=runtime.TRIAL_SPEC
    controller=runtime.TwoStageAdmittance(.01,specification)
    for i in range(11):controller.observe(i*.01,.5)
    assert controller.phase=='settle'
    for i in range(12,34):controller.observe(i*.01,1.)
    assert controller.phase=='maintain'
    for _ in range(100):assert abs(controller.step(1.,0.))<=.002+1e-12


def test_hybrid_cli_paths_configurable_and_frozen_vendor_dependencies_included(tmp_path,monkeypatch):
    path=ROOT/'scripts/evaluate_hybrid_paired_batch.py'
    spec=importlib.util.spec_from_file_location('portable_batch_paths_test',path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    assert callable(getattr(api,'configure_locations',None)),'portable batch location configuration missing'
    api.configure_locations(tmp_path/'outputs',tmp_path/'old_act')
    assert api.EXPORTS==tmp_path/'outputs' and api.OLD==tmp_path/'old_act'
    assert api.CANDIDATE.is_relative_to(ROOT)
    dependencies=api.dependency_hashes(api.runtime())
    assert str(api.CANDIDATE/'runner.py') in dependencies
    assert all(Path(name).is_relative_to(ROOT) for name in dependencies)


def test_workbench_default_checkpoint_uses_configured_ubuntu_model_root(tmp_path,monkeypatch):
    from sim.web import app as module
    model_root=tmp_path/'models';checkpoint=model_root/'checkpoints'/'step_100000';checkpoint.mkdir(parents=True)
    for name in ('config.json','model.safetensors','policy_preprocessor.json',
                 'policy_preprocessor_step_3_normalizer_processor.safetensors','policy_postprocessor.json',
                 'policy_postprocessor_step_0_unnormalizer_processor.safetensors','training_state.pt'):
        (checkpoint/name).write_bytes(b'fixture')
    (model_root/'latest_checkpoint.txt').write_text('checkpoints/step_100000')
    try:actual=module.latest_inference_model(tmp_path,model_root=str(model_root))
    except TypeError:pytest.fail('configured model root is not supported')
    assert actual==str(checkpoint)


def test_workbench_missing_configured_checkpoint_rejects_before_start(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from sim.config import load_config
    from sim.web.app import create_app
    from sim.web.jobs import JobRegistry
    registry=JobRegistry(tmp_path)
    monkeypatch.setattr('sim.web.jobs.JobRegistry',lambda root:registry)
    cfg=load_config()
    cfg.set_path('act.dataset_dir',str(tmp_path/'data'))
    cfg.set_path('web.inference_video_dir',str(tmp_path/'videos'))
    cfg.set_path('web.inference_model_root',str(tmp_path/'missing'))
    calls=[]
    monkeypatch.setattr(registry,'start',lambda kind,argv:calls.append((kind,argv)))
    with TestClient(create_app(cfg)) as client:
        response=client.post('/api/jobs/inference',json={})
        assert response.status_code==422
        assert 'No complete checkpoint' in response.json()['detail']
    assert calls==[]


def test_live_view_counts_final_not_peak_and_never_false_completion():
    path=ROOT/'scripts/hybrid_live_view.py'
    spec=importlib.util.spec_from_file_location('portable_live_view_test',path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    matrix,totals=api.count_matrix([{'target_count':1,'final_collected':0,'peak_collected':1}])
    assert matrix[0][0]==1 and matrix[1][0]==0 and totals[0]==1
    status={'state':'complete','completed':120};summary={'complete':True,'completed':120}
    assert api.batch_finished(status,summary,120,False)
    assert not api.batch_finished(status,summary,120,True)
    assert not api.batch_finished(status,summary,119,False)


def test_sealed_hybrid_baseline_cannot_be_switched_before_attempt(tmp_path,monkeypatch):
    path=ROOT/'scripts/evaluate_hybrid_paired_batch.py'
    spec=importlib.util.spec_from_file_location('portable_sealed_baseline_test',path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    batch=tmp_path/'new_batch';batch.mkdir();(batch/'attempts').mkdir()
    baseline_a=tmp_path/'baseline_a';baseline_b=tmp_path/'baseline_b'
    metadata={'kind':api.LABEL,'profile':api.PROFILE,'paired_act_batch':str(baseline_a),
              'frozen_sha256':{},'source_fingerprints':{},
              'checkpoint':{'path':str(tmp_path/'model'),'checkpoint_files':{}}}
    api.write_json(batch/'metadata.json',metadata)
    api.write_json(batch/'seal.json',{'metadata_sha256':api.sha(batch/'metadata.json')})
    monkeypatch.setattr(api,'PROJECT',tmp_path/'isolated_project')
    monkeypatch.setattr(api,'OLD',baseline_a)
    assert api.verify_identity(batch)['paired_act_batch']==str(baseline_a)
    monkeypatch.setattr(api,'OLD',baseline_b)
    with pytest.raises(RuntimeError,match='paired baseline'):
        api.verify_identity(batch)
    assert list((batch/'attempts').iterdir())==[]
