from pathlib import Path

from fastapi.testclient import TestClient

from sim.config import load_config
from sim.web.app import create_app


def settings(tmp_path):
    cfg = load_config()
    cfg.set_path('act.dataset_dir', str(tmp_path/'dataset'))
    cfg.set_path('act.model_dir', str(tmp_path/'model'))
    cfg.set_path('web.inference_model_root', str(tmp_path/'model'))
    return cfg


def checkpoint(root, step):
    folder = root/'checkpoints'/f'step_{step:06d}'
    folder.mkdir(parents=True)
    for name in ('config.json','model.safetensors','policy_preprocessor.json',
                 'policy_preprocessor_step_3_normalizer_processor.safetensors',
                 'policy_postprocessor.json',
                 'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
                 'training_state.pt'):
        (folder/name).write_bytes(b'fixture')
    (root/'latest_checkpoint.txt').write_text(str(folder.relative_to(root)))
    return folder


def test_empty_training_root_is_visible_but_rejected_before_job_start(tmp_path, monkeypatch):
    cfg=settings(tmp_path)
    app=create_app(cfg)
    calls=[]
    monkeypatch.setattr(app.state.jobs,'start',lambda *args:calls.append(args))
    with TestClient(app) as client:
        html=client.get('/').text
        assert 'id="inferModel" value="'+str(tmp_path/'model')+'"' in html
        defaults=client.get('/api/inference/defaults').json()['act']
        assert defaults['ready'] is False and defaults['model_root']==str(tmp_path/'model')
        response=client.post('/api/jobs/inference',json={'policy_type':'act','model':str(tmp_path/'model')})
        assert response.status_code==422 and 'checkpoint' in response.json()['detail'].lower()
    assert calls==[]


def test_ready_pointer_is_resolved_at_request_time_with_original_config(tmp_path,monkeypatch):
    cfg=settings(tmp_path)
    app=create_app(cfg)
    calls=[]
    def start(kind,argv):calls.append((kind,argv));return {'kind':kind,'state':'starting'}
    monkeypatch.setattr(app.state.jobs,'start',start)
    with TestClient(app) as client:
        assert client.get('/api/inference/defaults').json()['act']['ready'] is False
        first=checkpoint(tmp_path/'model',100000)
        defaults=client.get('/api/inference/defaults').json()['act']
        assert defaults['ready'] is True and defaults['model']==str(first)
        second=checkpoint(tmp_path/'model',200000)
        response=client.post('/api/jobs/inference',json={'policy_type':'act','model':str(tmp_path/'model')})
        assert response.status_code==200
    argv=calls[0][1]
    assert argv[argv.index('--model')+1]==str(second)
    assert argv[argv.index('--config')+1]==str(cfg.get_path('_source_config'))


def test_inference_keeps_original_cross_chunk_average():
    cfg=load_config('configs/rudra-dinov3-act-cuda.yaml')
    from sim.act.realtime import ActionChunkScheduler
    scheduler=ActionChunkScheduler(cfg,allow_late=True)
    assert scheduler.temporal_ensemble is not None
    assert scheduler.temporal_ensemble.coeff==0.01
    assert list(cfg.act.camera_views)==['top','wrist']
    assert cfg.act.chunk_size==25 and cfg.act.execute_steps==5
