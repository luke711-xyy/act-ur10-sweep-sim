"""Catch temporal shifts, episode leakage and padded-statistic contamination."""
import json
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def demos(tmp_path):
    views = ("corner_xminus_yminus", "corner_xminus_yplus",
             "corner_xplus_yminus", "corner_xplus_yplus", "wrist")
    records = []
    for e, base in enumerate((0, 100)):
        n = 4
        states = np.repeat((np.arange(n) + base)[:, None], 36, axis=1).astype("f4")
        actions = np.repeat((np.arange(n) + base)[:, None], 4, axis=1).astype("f4")
        actions[2] = 9999  # invalid supervision must never fit statistics
        np.savez(tmp_path / f"{e}.npz", state=states, action=actions,
                 environment_state=np.ones((n, 1), "f4"),
                 action_valid=[True, True, False, True],
                 t=np.arange(n) * .04,
                 phase=["approach", "descent", "sweep", "sweep"])
        images = {}
        for view in views:
            images[view] = []
            for i in range(n):
                name = f"{e}_{view}_{i}.png"
                Image.fromarray(np.full((320,320,3), base+i, dtype="u1")).save(tmp_path/name)
                images[view].append(name)
        records.append(dict(episode_id=str(e), arrays=f"{e}.npz", images=images,
                            schema_version=5, action_dim=4, state_dim=36,
                            episode_kind="expert", split="train", success=True))
    (tmp_path/"manifest.jsonl").write_text("\n".join(map(json.dumps, records)))
    return tmp_path


def test_window_alignment_padding_and_episode_isolation(demos):
    from sim.diffusion.dataset import DiffusionDataset
    ds = DiffusionDataset(demos)
    start = ds[0]
    assert start["state"].shape == (2,36)
    assert start["state"][:,0].tolist() == [0,0]
    assert start["action"].shape == (28,4)
    assert start["valid_mask"][:5].tolist() == [False, True, True, False, True]
    assert not start["valid_mask"][26:].any()
    mid = ds[1]
    assert mid["state"][:,0].tolist() == [0,1]
    assert mid["action"][:4,0].tolist() == [0,1,0,3]
    assert mid["timestamps"].tolist() == pytest.approx([0,.04])
    assert set(mid["obs"]) == set(ds.camera_views) | {"state", "environment_state"}
    assert mid["obs"]["wrist"][:,0,0,0].tolist() == pytest.approx([0,1/255])
    end = ds[2]
    assert end["state"][:,0].tolist() == [2,3]
    assert end["valid_mask"][:3].tolist() == [False,True,False]
    other = ds[3]
    assert other["state"][:,0].tolist() == [100,100]


def test_valid_normalization_roundtrip_and_phase_sampling(demos):
    from sim.diffusion.dataset import DiffusionDataset, normalize, denormalize
    ds = DiffusionDataset(demos)
    stats = ds.normalization
    assert stats["action"]["max"] == [103.0]*4
    assert stats["action"]["min"] == [0.0]*4
    a = np.array([[0,1,2,3]], dtype="f4")
    np.testing.assert_allclose(denormalize(normalize(a,stats["action"]),stats["action"]),a,atol=1e-5)
    weights = ds.phase_sampling_weights(.35)
    assert weights[:2].sum()+weights[3:5].sum() == pytest.approx(.35)
    assert weights.sum() == pytest.approx(1)


def test_rejects_non_40ms_history(demos):
    from sim.diffusion.dataset import DiffusionDataset
    p = demos/"0.npz"
    with np.load(p) as z:
        data = dict(z)
    data["t"] = np.arange(4)*.2
    np.savez(p,**data)
    with pytest.raises(ValueError,match="40ms"):
        DiffusionDataset(demos)


def test_short_terminal_invalid_stability_frame_is_not_a_training_condition(demos):
    from sim.diffusion.dataset import DiffusionDataset
    p = demos/"0.npz"
    with np.load(p) as z:
        data = dict(z)
    data["action_valid"][3] = False
    data["t"][-1] = .1
    np.savez(p,**data)
    ds = DiffusionDataset(demos)
    assert ds[1]["timestamps"].tolist() == pytest.approx([0,.04])
    assert not ds[1]["valid_mask"][2:].any()
