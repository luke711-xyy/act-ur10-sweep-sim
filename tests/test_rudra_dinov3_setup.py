"""Portable setup writes host-local paths without touching data or starting jobs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/configure_rudra_dinov3_act.py'


def configure(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str,args)],
                          cwd=ROOT, text=True, capture_output=True)


def test_setup_generates_local_config_and_scene_without_mutating_dataset(tmp_path):
    dataset = tmp_path / 'data with spaces'
    dataset.mkdir()
    manifest = dataset / 'manifest.jsonl'
    manifest.write_text('{"fixture":"configuration only"}\n')
    original = manifest.read_bytes()
    output = tmp_path / 'models'
    config = tmp_path / 'local-act.yaml'
    result = configure('--dataset',dataset,'--out',output,'--config-out',config,
                       '--device','cpu','--batch-size',4)
    assert result.returncode == 0, result.stderr
    cfg = yaml.safe_load(config.read_text())
    scene = yaml.safe_load(Path(cfg['scene']['profile']).read_text())
    assert cfg['act']['dataset_dir'] == str(dataset.resolve())
    assert cfg['act']['model_dir'] == str(output.resolve())
    assert cfg['web']['inference_model_root'] == str(output.resolve())
    assert cfg['act']['camera_views'] == ['top','wrist']
    assert cfg['act']['image_size'] == [640,480]
    assert cfg['act']['device'] == 'cpu' and cfg['act']['cuda_memory_fraction'] is None
    assert cfg['act']['batch_size'] == 4
    assert scene['sweep_task']['source_config'] == str(config.resolve())
    assert cfg['act']['checkpoint_every_steps'] == 5000
    assert cfg['act']['keep_checkpoints'] == 2
    assert cfg['act']['checkpoint_permanent_steps'] == [50000,100000,200000]
    assert not output.exists()  # A configuration command must not create model artifacts.
    assert manifest.read_bytes() == original
    assert 'sim.act.train' in json.loads(result.stdout)['train_argv']


def test_setup_preserves_existing_config(tmp_path):
    dataset=tmp_path/'data'; dataset.mkdir(); (dataset/'manifest.jsonl').write_text('{}\n')
    config=tmp_path/'local-act.yaml'; config.write_text('existing: keep\n')
    result=configure('--dataset',dataset,'--out',tmp_path/'model','--config-out',config)
    assert result.returncode != 0
    assert config.read_text() == 'existing: keep\n'
    assert not config.with_name('local-act-scene.yaml').exists()


@pytest.mark.parametrize('extra', [('--batch-size','0'),('--cuda-memory-fraction','nan'),
                                   ('--device','cpu','--cuda-memory-fraction','0.5')])
def test_invalid_setup_does_not_write_files(tmp_path,extra):
    dataset=tmp_path/'data'; dataset.mkdir(); (dataset/'manifest.jsonl').write_text('{}\n')
    config=tmp_path/'local-act.yaml'
    result=configure('--dataset',dataset,'--out',tmp_path/'model','--config-out',config,*extra)
    assert result.returncode != 0
    assert not config.exists()


def test_missing_manifest_is_reported_before_writing_config(tmp_path):
    config=tmp_path/'local-act.yaml'
    result=configure('--dataset',tmp_path/'missing','--out',tmp_path/'model','--config-out',config)
    assert result.returncode != 0 and 'manifest' in result.stderr.lower()
    assert not config.exists()


def test_setup_rejects_writing_configuration_inside_demonstrations(tmp_path):
    dataset=tmp_path/'data';dataset.mkdir();(dataset/'manifest.jsonl').write_text('{}\n')
    config=dataset/'local-act.yaml'
    result=configure('--dataset',dataset,'--out',tmp_path/'model','--config-out',config)
    assert result.returncode != 0
    assert sorted(path.name for path in dataset.iterdir()) == ['manifest.jsonl']
