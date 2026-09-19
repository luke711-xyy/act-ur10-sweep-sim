import numpy as np


def test_v5_bev_is_reconstructed_from_predicted_sidecar_fields():
    from sim.act.v5_bev import build_bev_from_sidecar

    tokens = np.zeros((6, 29), dtype=np.float32)
    tokens[:, 24] = 0.9
    tokens[:, 28] = 1.0
    tokens[0, 3:5] = [0.1, 0.0]
    masks = np.zeros((6, 128, 160), dtype=bool)
    masks[0, 64, 96] = True
    valid = np.array([True, False, False, False, False, False])
    robot_state = np.zeros(36, dtype=np.float32)
    robot_state[6:8] = [0.1, 0.0]
    selection = np.array([True, False, False, False, False, False])
    bev = build_bev_from_sidecar(
        object_tokens=tokens,
        object_valid=valid,
        instance_bev=masks,
        robot_state=robot_state,
        selection_target=selection,
        tray_bounds=(-0.45, -0.25, -0.18, 0.18),
    )
    assert bev.shape == (6, 128, 160)
    assert bev.dtype == np.float32
    assert bev[0, 64, 96] > 0.0
    assert bev[1, 64, 96] > 0.0
    assert bev[4, 64, 96] > 0.0
