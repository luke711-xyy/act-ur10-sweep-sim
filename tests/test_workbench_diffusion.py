import numpy as np
import pytest
from sim.web.jobs import build_train_argv, build_inference_argv


def test_diffusion_api_uses_independent_defaults_and_resume(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from sim.config import load_config
    from sim.web.app import create_app
    app = create_app(load_config())
    monkeypatch.setattr(app.state.jobs,"start",lambda kind,argv:{"kind":kind,"argv":argv})
    with TestClient(app) as client:
        response = client.post("/api/jobs/train",json={"policy_type":"diffusion","resume":"run.pt"})
        assert response.status_code == 200
        argv = response.json()["argv"]
        assert "sim.diffusion.train" in argv
        assert argv[argv.index("--out")+1] == "runs/diffusion_policy_mvp"
        assert argv[argv.index("--resume")+1] == "run.pt"
        assert client.post("/api/jobs/train",json={"policy_type":"unknown"}).status_code == 422
        response = client.post("/api/jobs/inference",json={"policy_type":"diffusion","model":"x.pt","target_count":2})
        assert response.status_code == 200
        assert "--policy-type" in response.json()["argv"]
        argv = response.json()["argv"]
        video = argv[argv.index("--video-root")+1]
        from pathlib import Path
        assert Path(video).is_relative_to(Path(__file__).resolve().parents[1]/"runs")


def test_diffusion_workspace_browses_original_data_read_only(tmp_path):
    from sim.web.run import diffusion_workspace_config
    from sim.config import load_config
    from sim.diffusion.config import DEFAULT_DATASET
    cfg = diffusion_workspace_config(load_config(),tmp_path)
    assert str(cfg.act.dataset_dir) == DEFAULT_DATASET
    assert cfg.web.readonly_dataset
    assert str(cfg.web.inference_video_dir) == str(tmp_path/"runs/diffusion_policy_mvp/videos")
    assert cfg.act.camera_views == load_config().act.camera_views


def test_readonly_dataset_delete_is_rejected_without_touching_files(tmp_path):
    import json
    from sim.config import load_config
    from sim.web.workbench import WorkbenchState
    episode=tmp_path/"keep"
    episode.mkdir()
    np.savez(episode/"arrays.npz",action=np.zeros((1,4)))
    content=json.dumps(dict(episode_id="keep",arrays="keep/arrays.npz",frame_count=1))+"\n"
    (tmp_path/"manifest.jsonl").write_text(content)
    cfg=load_config(overrides=["web.readonly_dataset=true"])
    state=WorkbenchState(cfg,dataset_root=tmp_path,preview_root=tmp_path/"previews")
    with pytest.raises(PermissionError,match="read.only"):
        state.delete_episode("dataset:keep")
    assert episode.is_dir()
    assert (tmp_path/"manifest.jsonl").read_text() == content


def test_dp_inference_artifacts_cannot_escape_independent_worktree(tmp_path):
    from sim.diffusion.runtime import inference_resource_guard
    with pytest.raises(ValueError,match="worktree"):
        inference_resource_guard(tmp_path,tmp_path.parent/"external",tmp_path/"video")


def test_act_legacy_and_diffusion_commands_are_separate(tmp_path):
    assert "sim.act.train" in build_train_argv(tmp_path,"c","d","o")
    argv = build_train_argv(tmp_path,"c","d","o",resume="x",policy_type="diffusion")
    assert "sim.diffusion.train" in argv and argv[-2:] == ["--resume","x"]
    argv = build_inference_argv(tmp_path,"c",7,"model",3,policy_type="diffusion")
    assert argv[argv.index("--policy-type")+1] == "diffusion"
    with pytest.raises(ValueError):
        build_train_argv(tmp_path,"c","d","o",policy_type="bogus")


def test_history_is_25hz_bounded_causal_and_first_frame_repeated():
    from sim.diffusion.runtime import TwoFrameHistory
    from sim.diffusion.dataset import VIEWS
    def obs(t):
        return {"t":t,"observation.state":np.full(36,t,dtype="f4"),
                "observation.environment_state":np.array([3],dtype="f4"),
                **{f"observation.images.{v}":np.full((3,320,320),t,dtype="f4") for v in VIEWS},
                "object_pose":np.full((6,7),999)}
    history = TwoFrameHistory()
    history.append(obs(0))
    assert history.batch()["state"][:, :, 0].tolist() == [[0,0]]
    for i in range(1,10):
        history.append(obs(i*.04))
    assert len(history.frames) == 2
    assert history.batch()["state"][0,:,0].tolist() == pytest.approx([.32,.36])
    assert "object_pose" not in history.batch()
    with pytest.raises(ValueError,match="40ms"):
        history.append(obs(.6))


def test_rolling_chunk_executes_only_five_without_delay_or_averaging():
    from sim.diffusion.runtime import RollingChunk
    clock = RollingChunk()
    values = np.repeat(np.arange(25)[:,None],4,axis=1)
    clock.accept(1.,values,1.)
    assert clock.action_for(1.)[0] == 0
    assert clock.action_for(1.16)[0] == 4
    assert clock.query_due(1.2)
    with pytest.raises(RuntimeError):
        clock.action_for(1.2)
    clock.accept(1.2,values+100,1.2)
    assert clock.action_for(1.2)[0] == 100
    with pytest.raises(ValueError):
        clock.accept(1.4,np.full((25,4),np.nan),1.4)
