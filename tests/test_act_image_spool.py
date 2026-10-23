import numpy as np
import pytest


def test_disk_backed_capture_preserves_both_views_and_previous_frames(tmp_path):
    from sim.act.image_spool import NativeImageSpool
    spool=NativeImageSpool(tmp_path,max_frames=12,flush_every=2)
    rng=np.random.default_rng(42)
    originals=[];stored=[]
    for i in range(10):
        images={view:rng.integers(0,256,(24,32,3),dtype=np.uint8) for view in ['top','wrist']}
        originals.append({k:v.copy() for k,v in images.items()})
        stored.append(spool.store(images))
        for value in images.values():value[:]=0
    for expected,actual in zip(originals,stored):
        for view in ['top','wrist']:
            assert isinstance(actual[view],np.memmap)
            np.testing.assert_array_equal(actual[view],expected[view])
    assert len(list(tmp_path.glob('*.rgb')))==1


def test_image_store_rejects_capacity_or_shape_changes(tmp_path):
    from sim.act.image_spool import NativeImageSpool
    spool=NativeImageSpool(tmp_path,max_frames=2)
    images={v:np.zeros((24,32,3),dtype=np.uint8) for v in ['top','wrist']}
    spool.store(images)
    with pytest.raises(ValueError,match='shape'):
        spool.store(dict(top=np.zeros((20,32,3),dtype=np.uint8),wrist=images['wrist']))
    spool.store(images)
    with pytest.raises(ValueError,match='capacity'):spool.store(images)


def test_rollout_spool_uses_the_real_episode_frame_budget(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from sim.config import load_config
    from sim.act.rollout import run_action_path
    from sim.act import image_spool
    cfg=load_config(overrides=['episode.max_frames=17'])
    cfg.set_path('act.image_spool_dir',str(tmp_path))
    assert cfg.get_path('act.max_frames',None) is None
    class BudgetObserved(Exception):pass
    def observed(directory,max_frames):
        assert max_frames==17
        raise BudgetObserved
    monkeypatch.setattr(image_spool,'NativeImageSpool',observed)
    env=SimpleNamespace(table_top_z=0.,layout=[0]*6,tcp=lambda:np.zeros(3),
                        component_positions=lambda:np.zeros((6,3)),
                        ee=SimpleNamespace(tcp_yaw=lambda:0.))
    with pytest.raises(BudgetObserved):
        run_action_path(env,cfg,np.empty((0,4)),collect_observations=True)


def test_spooled_writer_preserves_pixels_and_arrays_and_releases_read_pages(monkeypatch,tmp_path):
    import json
    from PIL import Image
    from sim.act.dataset import ActDatasetWriter
    from sim.act import image_spool as module
    spool=module.NativeImageSpool(tmp_path/'scratch',max_frames=10)
    observations=[]
    for i in range(9):
        images={view:np.full((24,32,3),i*20,dtype=np.uint8) for view in ['top','wrist']}
        observations.append(dict(images=spool.store(images),state=np.zeros(36),
                                 environment_state=np.array([1.]),t=i*.04,phase='sweep'))
    calls=[];original=module.discard_image_pages
    def released(images):calls.append(True);original(images)
    monkeypatch.setattr(module,'discard_image_pages',released)
    for name,cls in [('reference',ActDatasetWriter),('spooled',module.DiskBackedDatasetWriter)]:
        cls(str(tmp_path/name),image_format='webp-lossless').add_episode(
            'episode',observations,np.zeros((9,4)),True,{})
    assert calls
    with np.load(tmp_path/'reference/episode/arrays.npz') as a,np.load(tmp_path/'spooled/episode/arrays.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    record=json.loads((tmp_path/'spooled/manifest.jsonl').read_text())
    for view in ['top','wrist']:
        for i,name in enumerate(record['images'][view]):
            with Image.open(tmp_path/'spooled'/name) as image:
                np.testing.assert_array_equal(np.asarray(image),np.full((24,32,3),i*20,dtype=np.uint8))
