"""Original sweep objects/layout, front table view and native wrist mount."""
import base64
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/rudra-sweep-scene.yaml"


def test_sweep_profile_preserves_layout_geometry_and_camera_mounts():
    from sim.config import load_config
    from sim.environments.layout import sample_layout
    from sim.web.rudra_preview import RudraPreview, upstream

    assert PROFILE.exists(), "the native table needs its own sweep task profile"
    viewer = RudraPreview(PROFILE)
    try:
        meta = viewer.metadata()
        scene = viewer.scene
        original = load_config(str(ROOT / "configs/five-view-fine-teached-300.yaml"))
        expected = sample_layout(original, np.random.default_rng(meta["sweep_task"]["seed"]))
        assert scene.local_layout == expected
        offset = np.array([0.005, -0.685, -0.002])
        for local, world in zip(expected, scene.layout, strict=True):
            assert {k:v for k,v in local.items() if k not in ("x","y")} == {
                k:v for k,v in world.items() if k not in ("x","y")}
            np.testing.assert_allclose([world["x"],world["y"]], [local["x"],local["y"]]+offset[:2])
        assert len(scene.layout) == 6
        assert meta["sweep_task"]["target_bounds_world"] == {
            "x_min": -0.415, "x_max": -0.275, "y_min": -0.865, "y_max": -0.505}
        assert meta["sweep_task"]["generation_rules"] == original.components.to_dict()

        native_cfg = upstream().load_config(ROOT / "configs/rudra-native-scene.yaml")
        for name in ("wrist","overview"):
            assert scene.cfg["cameras"][name] == native_cfg["cameras"][name]
        top=scene.cfg["cameras"]["top"]
        assert top["pos"] == [-.85,-.95,1.00]
        assert top["look_at"] == [-.035,-.70,-.002]
        assert top["up_hint"] == [0,0,1]
        for name in ("appearance","fov_h_deg","rpy_deg"):
            assert top[name] == native_cfg["cameras"]["top"][name]
        assert scene.cfg["table"] == native_cfg["table"]
        xml_native = ET.fromstring(upstream().build_xml(native_cfg))
        xml_new = ET.fromstring(scene.xml)
        for name in ("wrist","overview"):
            assert xml_new.find(f".//camera[@name='{name}']").attrib == xml_native.find(f".//camera[@name='{name}']").attrib
        for name in ("pincopen_base","wrist_yaw","wrist_mount","d435i"):
            assert xml_new.find(f".//body[@name='{name}']").attrib == xml_native.find(f".//body[@name='{name}']").attrib
        for name in ("peg","hole_circle","hole_square"):
            assert scene.bid(name) == -1
        assert scene.bid("target_tray") >= 0
        for i in range(6):
            assert scene.bid(f"comp_{i}") >= 0
            np.testing.assert_allclose(scene.model.body_mass[scene.bid(f"comp_{i}")], expected[i]["mass"], rtol=2e-5)
        for name in ("brush_head","brush_sole"):
            i = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert scene.model.geom_contype[i] == {"brush_head":2,"brush_sole":4}[name]
        table = mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_GEOM,"table")
        assert scene.model.geom_conaffinity[table] == 5
        assert np.isfinite(scene.data.qpos).all()
        assert np.max(np.abs(scene.data.qvel)) < 0.1
        for i in range(6):
            assert -0.003 < scene.data.xpos[scene.bid(f"comp_{i}")][2] < 0.025
    finally:
        viewer.close()


def test_brush_is_clamped_and_moves_rigidly_with_the_robot():
    from sim.web.rudra_preview import RudraPreview
    assert PROFILE.exists(), "missing sweep task profile"
    viewer = RudraPreview(PROFILE)
    try:
        state = viewer.metadata()
        scene = viewer.scene
        m,d = scene.model,scene.data
        brush,gripper = scene.bid("sweep_brush"),scene.bid("pincopen_base")
        assert m.body_parentid[brush] == gripper
        assert m.body_jntnum[brush] == 0
        task = state["sweep_task"]
        assert task["brush"]["width"] == 0.14
        assert task["brush"]["height"] == 0.06
        assert task["brush"]["handle_thickness"] == 0.012
        assert abs(task["brush"]["measured_jaw_gap_m"]-0.012) < 0.001
        assert task["brush"]["grasp_alignment_error_m"] < 0.001
        tcp = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"tcp_site")
        np.testing.assert_allclose(d.site_xpos[tcp],[0.425,-0.685,0.098],atol=5e-4)
        static = {n:(d.cam_xpos[scene.cid(n)].copy(),d.cam_xmat[scene.cid(n)].copy()) for n in ("top","overview")}
        local = d.xmat[gripper].reshape(3,3).T @ (d.xpos[brush]-d.xpos[gripper])
        wrist_before = d.cam_xpos[scene.cid("wrist")].copy()
        before = d.site_xpos[tcp].copy()
        d.qpos[scene.arm_adr[0]] += 0.1
        mujoco.mj_forward(m,d)
        assert np.linalg.norm(before-d.site_xpos[tcp]) > 0.01
        assert np.linalg.norm(wrist_before-d.cam_xpos[scene.cid("wrist")]) > 0.01
        np.testing.assert_allclose(d.xmat[gripper].reshape(3,3).T @ (d.xpos[brush]-d.xpos[gripper]),local,atol=1e-12)
        for n,(p,r) in static.items():
            np.testing.assert_array_equal(d.cam_xpos[scene.cid(n)],p)
            np.testing.assert_array_equal(d.cam_xmat[scene.cid(n)],r)
    finally:
        viewer.close()


def test_sweep_workbench_regenerates_seeds_without_changing_cameras():
    from fastapi.testclient import TestClient
    from sim.web.rudra_preview import create_rudra_preview_app
    assert PROFILE.exists(), "missing sweep task profile"
    with TestClient(create_rudra_preview_app(PROFILE)) as client:
        before = client.get("/api/scene").json()
        assert before["mode"] == "rudra_sweep_preview"
        response = client.get("/api/frames")
        assert response.status_code == 200
        for encoded in response.json()["images"].values():
            pixels = np.array(Image.open(BytesIO(base64.b64decode(encoded))))
            assert pixels.shape == (480,640,3) and pixels.std() > 8
        result = client.post("/api/reset",json={"seed":2120000})
        assert result.status_code == 200
        different = result.json()
        assert different["sweep_task"]["seed"] == 2120000
        assert different["sweep_task"]["layout_local"] != before["sweep_task"]["layout_local"]
        assert client.post("/api/reset",json={"seed":2120000}).json()["sweep_task"]["layout_local"] == different["sweep_task"]["layout_local"]
        assert different["cameras"] == before["cameras"]
        assert client.post("/api/reset",json={"seed":-1}).status_code == 422
        assert client.get("/api/scene").json()["sweep_task"]["seed"] == 2120000
