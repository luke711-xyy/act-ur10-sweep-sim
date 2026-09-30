"""Behavior tests for exporting saved workbench episodes as synchronized MP4."""

import numpy as np
from fastapi.testclient import TestClient

from sim.act.dataset import ActDatasetWriter
from sim.config import load_config
from sim.web.app import create_app


def _write_episode(root, episode_id="episode_sync"):
    colors = (
        ((240, 20, 20), (20, 240, 20), (20, 20, 240)),
        ((240, 240, 20), (20, 240, 240), (240, 20, 240)),
    )
    observations = []
    for frame_colors in colors:
        observations.append({
            "overhead": np.full((64, 64, 3), frame_colors[0], dtype=np.uint8),
            "wrist": np.full((64, 64, 3), frame_colors[1], dtype=np.uint8),
            "inspection": np.full((64, 64, 3), frame_colors[2], dtype=np.uint8),
            "state": np.zeros(36, dtype=np.float32),
            "environment_state": np.zeros(3, dtype=np.float32),
            "phase": "sweep",
        })
    ActDatasetWriter(str(root)).add_episode(
        episode_id,
        observations,
        np.zeros((2, 4), dtype=np.float32),
        True,
        {"seed": 11, "target_count": 2, "total_count": 6,
         "fps": 25.0, "episode_kind": "expert", "split": "train"},
    )


def _color_at(frame, panel, panel_width):
    return frame[frame.shape[0] // 2,
                 panel * panel_width + panel_width // 2]


def test_episode_video_download_is_synchronized_three_view_mp4(
        tmp_path, monkeypatch):
    from sim.web import jobs

    class EmptyJobRegistry:
        def __init__(self, _project_root):
            pass

    monkeypatch.setattr(jobs, "JobRegistry", EmptyJobRegistry)
    dataset_root = tmp_path / "dataset"
    _write_episode(dataset_root)
    app = create_app(load_config(), dataset_root=dataset_root,
                     preview_root=tmp_path / "previews")

    with TestClient(app) as client:
        response = client.get("/api/episodes/episode_sync/video")
        page = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert "episode_sync.mp4" in response.headers["content-disposition"]
    assert 'id="exportVideoButton"' in page.text

    import imageio.v2 as imageio

    path = tmp_path / "download.mp4"
    path.write_bytes(response.content)
    reader = imageio.get_reader(str(path))
    try:
        assert reader.count_frames() == 2
        first, second = reader.get_data(0), reader.get_data(1)
    finally:
        reader.close()

    panel_width = 64
    assert first.shape[1] == panel_width * 3
    assert np.argmax(_color_at(first, 0, panel_width)) == 0
    assert np.argmax(_color_at(first, 1, panel_width)) == 1
    assert np.argmax(_color_at(first, 2, panel_width)) == 2
    assert _color_at(second, 0, panel_width)[2] < 80
    assert _color_at(second, 1, panel_width)[0] < 80
    assert _color_at(second, 2, panel_width)[1] < 80


def test_episode_video_download_reports_unknown_episode(tmp_path, monkeypatch):
    from sim.web import jobs

    class EmptyJobRegistry:
        def __init__(self, _project_root):
            pass

    monkeypatch.setattr(jobs, "JobRegistry", EmptyJobRegistry)
    app = create_app(load_config(), dataset_root=tmp_path,
                     preview_root=tmp_path / "previews")

    with TestClient(app) as client:
        response = client.get("/api/episodes/not_saved/video")

    assert response.status_code == 404
    assert "not_saved" in response.json()["detail"]
