"""The external native scene is inspectable without starting task jobs."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image


def test_rudra_preview_loads_original_table_and_three_native_cameras():
    from sim.web.rudra_preview import RudraPreview
    viewer = RudraPreview()
    try:
        data = viewer.metadata()
        assert data["table"]["size_xy"] == [1.25, 1.15]
        assert data["table"]["z"] == -0.002
        assert set(data["cameras"]) == {"top", "wrist", "overview"}
        assert data["cameras"]["top"]["position"] == [0.0322, -0.8351, 1.068]
        assert data["cameras"]["wrist"]["mount"]["version"] == "V6_REINFORCED"
        assert data["cameras"]["wrist"]["pose_source"] == "cad_mount"
        assert data["cameras"]["wrist"]["position"] is None
        assert data["cameras"]["top"]["configuration"]["yaw_deg"] == -1.5
        assert np.asarray(data["cameras"]["wrist"]["world_rotation"]).shape == (3,3)
        assert data["robot"]["source"] == "home"
        assert data["source_commit"] == "436058dde9098c077f14d14d11e633fee9a02dee"
        assert np.isfinite(viewer.scene.data.qpos).all()
    finally:
        viewer.close()


def test_native_preview_routes_render_rgb_without_advancing_or_launching_jobs():
    from fastapi.testclient import TestClient
    from sim.web.rudra_preview import create_rudra_preview_app
    with TestClient(create_rudra_preview_app()) as client:
        html = client.get("/").text
        assert "Rudra" in html and "topImage" in html and "wristImage" in html
        before = client.get("/api/scene").json()
        response = client.get("/api/frames")
        assert response.status_code == 200
        frames = response.json()["images"]
        assert set(frames) == {"top", "wrist", "overview"}
        for data in frames.values():
            pixels = np.array(Image.open(BytesIO(base64.b64decode(data.split(",", 1)[-1]))))
            assert pixels.shape == (480, 640, 3)
            assert pixels.std() > 8
        after = client.get("/api/scene").json()
        assert after["time"] == before["time"]
        assert client.post("/api/jobs/train", json={}).status_code == 404
        assert client.post("/api/jobs/inference", json={}).status_code == 404
        assert client.post("/api/reset").status_code == 200
        reset = client.get("/api/scene").json()
        assert reset["robot"]["joints_deg"] == [0, -90, 90, -90, -90, 90]


def test_preview_page_assigns_browser_image_data_urls():
    import json
    import shutil
    import subprocess
    from sim.web.rudra_preview import preview_html
    script = preview_html().split("<script>",1)[1].split("</script>",1)[0]
    harness = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements={};const state={time:2.5,table:{size_xy:[1.25,1.15],z:-.002},source_commit:'436058d',config:'scene.yaml',cameras:{top:{configuration:{view_label:'前上方相机',view_description:'从收集区一侧斜看桌面'}}}};
const context={document:{getElementById:id=>elements[id]??={}},
fetch:async url=>({ok:true,json:async()=>url==='/api/scene'?state:{images:{top:'cGl4ZWxz',wrist:'cGl4ZWxz',overview:'cGl4ZWxz'}}})};
vm.createContext(context);vm.runInContext(SCRIPT,context).then(()=>{
 for(const name of ['top','wrist','overview'])assert.equal(elements[name+'Image'].src,'data:image/png;base64,cGl4ZWxz');
 assert.equal(elements.topCameraTitle.textContent,'前上方相机');
 assert.equal(elements.topCameraCaption.textContent,'从收集区一侧斜看桌面');
 assert.ok(!elements.status.textContent.includes('失败'));
}).catch(e=>{console.error(e);process.exitCode=1});
'''.replace("SCRIPT",json.dumps(script))
    node=shutil.which("node")
    assert node,"Node is required for the browser image assignment regression"
    result=subprocess.run([node,"-e",harness],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
