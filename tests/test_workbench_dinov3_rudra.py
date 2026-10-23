from fastapi.testclient import TestClient

from sim.config import load_config
from sim.web.app import create_app


def test_copied_workbench_identifies_dino_and_uses_its_own_training_paths(tmp_path):
    cfg = load_config()
    cfg.set_path("act.dataset_dir", str(tmp_path / "new_dataset"))
    cfg.set_path("act.model_dir", str(tmp_path / "new_act_model"))
    cfg.set_path("act.vision_encoder", "dinov3")
    cfg.set_path("act.camera_views", ["top", "wrist"])
    cfg.set_path("scene.backend", "rudra_native")
    cfg.set_path("web.inference_policy", "act")
    cfg.set_path("web.readonly_dataset", True)
    with TestClient(create_app(cfg)) as client:
        html = client.get("/").text
    assert "DINOv3 ACT · Rudra Sweep Workbench" in html
    assert 'id="topImage"' in html and 'id="wristImage"' in html
    assert 'id="trainDataset" value="' + str(tmp_path / "new_dataset") + '"' in html
    assert 'id="trainOut" value="' + str(tmp_path / "new_act_model") + '"' in html
    assert "Diffusion input /" not in html
