import numpy as np

from sim.act.camera_preview import (candidate_camera_specs,
                                    preview_render_size)
from sim.config import load_config


def test_candidate_camera_specs_make_four_symmetric_overhead_corner_views():
    cfg = load_config()
    specs = candidate_camera_specs(cfg)

    assert set(specs) == {
        "preview_xminus_yminus",
        "preview_xminus_yplus",
        "preview_xplus_yminus",
        "preview_xplus_yplus",
    }
    positions = np.asarray([spec["pos"] for spec in specs.values()])
    lookats = np.asarray([spec["lookat"] for spec in specs.values()])

    assert positions.shape == (4, 3)
    assert np.allclose(positions[:, 2], positions[0, 2])
    assert positions[0, 2] > cfg.table.top_z
    assert np.allclose(lookats, np.tile([0.0, 0.0, cfg.table.top_z], (4, 1)))
    assert len({tuple(position) for position in positions}) == 4
    assert np.allclose(np.abs(positions[:, 0]), np.abs(positions[0, 0]))
    assert np.allclose(np.abs(positions[:, 1]), np.abs(positions[0, 1]))
    assert all(spec["fovy_deg"] > 0.0 for spec in specs.values())


def test_candidate_camera_specs_keep_each_view_above_and_outside_table_corners():
    cfg = load_config()
    margin = 0.25
    specs = candidate_camera_specs(cfg, margin=margin, height=0.85)
    half_x, half_y, _ = cfg.table.half_size

    for spec in specs.values():
        x, y, z = spec["pos"]
        assert abs(x) >= half_x + margin
        assert abs(y) >= half_y + margin
        assert z >= cfg.table.top_z + 0.85
        assert spec["up"] == [0.0, 0.0, 1.0]


def test_candidate_view_export_defaults_to_policy_image_size():
    cfg = load_config()
    expected_size = tuple(int(value) for value in cfg.act.image_size)
    assert preview_render_size(cfg) == expected_size
    assert preview_render_size(cfg, width=640) == (640, expected_size[1])
