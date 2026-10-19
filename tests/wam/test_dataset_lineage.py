"""Task8 offline slot association and role/version contracts."""
import importlib
import importlib.util

import numpy as np
import pytest

from wam_sweep.contracts import CAMERA_VIEWS, OnlineObservation, RobotContext
from wam_sweep.native import CameraCalibration, CaptureMetadata
from wam_sweep.snapshots import EpisodeSnapshot, PhysicsSnapshot, materialize_images


def canonical_budget():
    import json
    from pathlib import Path
    from wam_sweep.artifacts import BudgetLedger
    from wam_sweep.config import WAMConfig
    path = Path('.venv/task-2-budget.json')
    return BudgetLedger(path,owned_roots=json.loads(path.read_bytes())['policy']['roots'],
                        config=WAMConfig.load('configs/wam_visual_v1.yaml').budget)


def test_store_deduplicates_pixels_and_retains_all_historical_calibration(tmp_path):
    from wam_sweep.data.store import EpisodeStore
    store = EpisodeStore(tmp_path,canonical_budget())
    robot = RobotContext([.1,.2,.3],0.,[0.,0.,0.],0.,'sweep',[0.,0.,0.],[0.,0.,0.],0.,True,True,0.,False)
    images = {v:np.full((320,320,3),j,np.uint8) for j,v in enumerate(CAMERA_VIEWS)}
    observations, calibration = [], {}
    for j in range(11):
        now = .4 + j*.2
        observations.append(OnlineObservation(now,(),robot,images))
        matrix = np.eye(4); matrix[0,3] = j*.01
        calibration[now] = CaptureMetadata(now,{v:CameraCalibration(np.eye(3),matrix) for v in CAMERA_VIEWS},'fixture')
    # Physics is deliberately a small schema fixture; real portability is separate.
    root = EpisodeSnapshot('fixture',PhysicsSnapshot((),np.zeros(1),None,{}, {},()),
                           {'capture':observations[-1]}, {'history':observations},'sweep:frame-boundary')
    first = store.save_snapshot(root,captures=calibration)
    second = store.save_snapshot(root,captures=calibration)
    assert first == second
    assert len(list((tmp_path/'images').iterdir())) == 5
    assert len(store.snapshot_captures(first)) == 11
    restored = materialize_images(store.load_snapshot(first),store.read_image)
    for old,new in zip(observations,restored.participants['history']):
        for view in CAMERA_VIEWS:
            np.testing.assert_array_equal(old.images[view],new.images[view])
    records = sorted(store.snapshot_captures(first),key=lambda pair:pair[0].timestamp)
    assert [pair[1].cameras['wrist'].camera_to_world[0,3] for pair in records] == pytest.approx(np.arange(11)*.01)
    with pytest.raises(ValueError,match='historical RGB'):
        store.save_snapshot(root,captures={})
    store.publish_cursor('trajectory',dict(last=first,complete=False))
    assert store.cursor('trajectory') == dict(last=first,complete=False)
    reference = next((tmp_path/'images').iterdir())
    # Deliberate owned corruption, via atomic API; never delete evidence.
    from wam_sweep.artifacts import atomic_checkpoint
    atomic_checkpoint(reference,b'corrupt',store.budget)
    with pytest.raises((ValueError,IndexError)):
        materialize_images(store.load_snapshot(first),store.read_image)


def lineage_api():
    assert importlib.util.find_spec('wam_sweep.data.datasets') is not None, 'Task8 dataset lineage missing'
    return importlib.import_module('wam_sweep.data.datasets')


def association_fixture(*, reverse_truth=False, ambiguous=False):
    from wam_sweep.contracts import ObjectBelief, supervision
    from wam_sweep.history import HistoryBuffer
    from wam_sweep.truth import CollisionBody, CollisionGeom, TruthFrame
    robot = RobotContext([.3,0.,.01],0.,[0.,0.,0.],0.,'sweep',[0.,0.,0.],[0.,0.,0.],0.,True,True,0.,False)
    images = {v:np.zeros((320,320,3),np.uint8) for v in CAMERA_VIEWS}
    # Slot zero is track9 at +.10; slot one is track2 at -.10. GT IDs and
    # record order both disagree. Slot allocation follows causal PAST appearance.
    def belief(track,x):
        return ObjectBelief([x,0.],[0.,0.],[.01,.01],0.,0.,[1.,0.],1.,[1.,0.,0.],
                            np.eye(2)*1e-6,np.zeros(16),track,True)
    history = HistoryBuffer()
    history.append(OnlineObservation(.8,(belief(9,.1),),robot,images))
    history.append(OnlineObservation(1.,(belief(2,-.1),belief(9,.1)),robot,images))
    def truth(t):
        ids = (88,4)
        xs = (-.1,.1) if not ambiguous else (.099,.101)
        objects = tuple(CollisionBody((CollisionGeom('sphere',[.01,0.,0.],[x+(t-1)*.02,0.,0.],np.eye(3)),),0.,'circle') for x in xs)
        if reverse_truth:
            ids,objects = ids[::-1],objects[::-1]
        brush = CollisionBody((CollisionGeom('box',[.01,.05,.01],[.3,0.,0.],np.eye(3)),),0.,'rectangle')
        return TruthFrame(t,supervision.RawTruth(dict(objects=objects,brush=brush,object_ids=ids,
            contacts=(),contacts_valid=True,native_collected=np.zeros(2,bool),native_partial=np.zeros(2,bool),
            native_valid=np.ones(2,bool),visibility=np.ones(2,bool),visibility_valid=np.ones(2,bool),target_bounds=None)))
    return history,truth


@pytest.mark.parametrize('reverse_truth',[False,True])
def test_offline_root_mapping_uses_geometry_and_carries_fixed_correspondence_for_25_steps(reverse_truth):
    module = lineage_api()
    history,truth = association_fixture(reverse_truth=reverse_truth)
    association = module.associate_root(history,truth(1.),perception_hash='p1',label_hash='l1')
    assert association.slot_object_ids[:2] == (4,88)
    assert association.valid.tolist() == [True,True,False,False,False,False]
    future = [truth(1.+(j+1)*.04) for j in range(25)]
    aligned = module.align_future(association,truth(1.),future)
    # Independently derived x(t): first slot .10+.02*dt, second -.10+.02*dt.
    np.testing.assert_allclose(aligned.object_mean[:,0,0],.1+.0008*np.arange(1,26),atol=1e-8)
    np.testing.assert_allclose(aligned.object_mean[:,1,0],-.1+.0008*np.arange(1,26),atol=1e-8)
    np.testing.assert_allclose(aligned.object_mean[:,:2,2],.02,atol=1e-8)
    assert aligned.object_valid[:,:2].all() and not aligned.object_valid[:,2:].any()
    assert not aligned.yaw_valid.any()  # circles do not acquire yaw/omega targets
    short = module.align_future(association,truth(1.),future[:7])
    assert short.valid_steps.tolist() == [True]*7+[False]*18
    assert not short.object_valid[7:].any()


def test_ambiguous_root_matching_masks_only_object_supervision():
    module = lineage_api()
    history,truth = association_fixture(ambiguous=True)
    association = module.associate_root(history,truth(1.),perception_hash='p2',label_hash='l1')
    assert not association.valid.any()
    aligned = module.align_future(association,truth(1.),[truth(1.04)])
    assert aligned.valid_steps[0] and not aligned.object_valid.any()


def role_root(*, success=True, role='teacher', current_valid=True, continuation_hash='c1', parent='p', n=1):
    import hashlib
    from wam_sweep.actions import ActionCodec
    from wam_sweep.contracts import CandidateProgram, EpisodeResult, EpisodeSpec, ReferencePreview, ModelVersion, InputIdentity
    from wam_sweep.data.collection import SafeInputs, RootRecord, RootSupervision, BranchRecord, ExecutionTrace, Selection
    module = lineage_api()
    history,truth = association_fixture()
    from wam_sweep.data.collection import ContextHistory
    context_history = ContextHistory('p1')
    for observation in history.snapshot()['observations']:
        context_history.append(observation)
    history = context_history
    association = module.associate_root(history,truth(1.),perception_hash='p1',label_hash='l1')
    inputs = SafeInputs(history.batch(),ReferencePreview(np.tile([.1,0.,.01,0.],(25,1)),
                        1.+np.arange(25)*.04,np.ones(25),1.,'known'),ActionCodec().snapshot())
    identity = InputIdentity('fixture','manifest','act',{k:'hash' for k in ('perception','control','preprocessing','config')})
    version = ModelVersion(-1,'act','fixture','act','p1','unused',continuation_hash,identity)
    future = [truth(1.+(j+1)*.04) for j in range(25)]
    trace = ExecutionTrace(truths=future)
    result = EpisodeResult('success' if success else 'task_failure','exact target count reached' if success else '45-second shared episode timeout')
    branch = BranchRecord(0,CandidateProgram(np.zeros((5,3))),trace,result,
                          selection=Selection(0,'reference',1.),
                          continuation_version=version,terminal_result=result,terminal_count=n)
    cache = module.PerceptionCache('p1',inputs.history,association,current_valid,('capture',),1.,history.entry.snapshot())
    seed = int(hashlib.sha256(parent.encode()).hexdigest()[:8],16)
    return RootRecord(parent+str(n),EpisodeSpec(parent,seed,n,'train',0,'act',0),version,'private',inputs,
                      ('capture',),RootSupervision(truth(1.),{},association),(branch,),role,perception_caches={'p1':cache})


def test_uncontinued_is_unknown_bootstrap_is_perception_only_and_all_fail_has_no_positive_teacher():
    from dataclasses import replace
    module = lineage_api()
    root = role_root()
    unknown = replace(root.branches[0],index=1,selection=None,terminal_result=None,continuation_version=None,terminal_count=None)
    root = replace(root,branches=(*root.branches,unknown))
    assert len(module.TerminalDataset([root],perception_hash='p1',continuation_hash='c1')) == 1
    assert len(module.TeacherDataset([role_root(success=False)],perception_hash='p1')) == 0
    bootstrap = role_root(role='bootstrap')
    assert len(module.TeacherDataset([bootstrap],perception_hash='p1')) == 0
    assert len(module.PredictionDataset([bootstrap],perception_hash='p1')) == 0
    sample = module.PredictionDataset([root],perception_hash='p1')[0]
    assert set(vars(sample['inputs'])) == {'history','reference','codec_state'}
    assert 'goal_n' not in sample and 'snapshot' not in sample and 'association' not in sample
    assert sample['supervision'].mapping_valid[:2].all()


def test_terminal_versions_and_current_perception_coverage_are_not_mixed():
    module = lineage_api()
    assert len(module.TerminalDataset([role_root(continuation_hash='old')],perception_hash='p1',continuation_hash='c1')) == 0
    assert len(module.TeacherDataset([role_root(current_valid=False)],perception_hash='p1')) == 0
    roots = [role_root(parent=f'p{i}',n=n) for n in range(1,7) for i in range(20)]
    report = module.coverage_report(roots,perception_hash='p1',phase=0)
    assert report.ready and all(x.successful_roots==20 and x.parent_layouts==20 for x in report.rows)
    repeated = [role_root(parent=f'p{i%9}',n=n) for n in range(1,7) for i in range(30)]
    assert not module.coverage_report(repeated,perception_hash='p1',phase=0).ready
    teacher = module.TeacherDataset([role_root()],perception_hash='p1')
    assert len(teacher)==1
    with pytest.raises(ValueError,match='not_ready'):
        teacher[0]


def test_requested_teacher_phase_cannot_return_an_undercovered_other_phase():
    from dataclasses import replace
    module = lineage_api()
    qualifying = [role_root(parent=f'phase0-{i}',n=n) for n in range(1,7) for i in range(20)]
    other = role_root(parent='phase1-only')
    other = replace(other,spec=replace(other.spec,phase=1))
    records = [other,*qualifying]  # put the undercovered positive FIRST
    phase0 = module.TeacherDataset(records,perception_hash='p1',phase=0)
    assert phase0.coverage.ready and len(phase0)==120
    assert {phase0[i]['audit']['root_id'] for i in range(len(phase0))} == {r.root_id for r in qualifying}
    phase1 = module.TeacherDataset(records,perception_hash='p1',phase=1)
    assert len(phase1)==1 and not phase1.coverage.ready
    with pytest.raises(ValueError,match='not_ready'):
        phase1[0]
    mixed = module.TeacherDataset(records,perception_hash='p1',phase=None)
    assert len(mixed)==121 and not mixed.coverage.ready
    with pytest.raises(ValueError,match='not_ready'):
        mixed[1]


def test_new_perception_hash_cannot_reuse_old_offline_association():
    from dataclasses import replace
    module = lineage_api()
    root = role_root()
    stale = replace(root.perception_caches['p1'],perception_hash='p2')
    root = replace(root,perception_caches={'p2':stale})
    with pytest.raises(ValueError,match='lineage'):
        module.TeacherDataset([root],perception_hash='p2')


def test_current_cache_rebuilds_its_own_visual_clock_from_poisoned_raw_context(tmp_path):
    from dataclasses import replace
    from test_collection import visual_context_fixture, collection_api
    module = lineage_api()
    collection = collection_api()
    store,tap,raws,ids,estimator_type = visual_context_fixture(tmp_path)
    root = role_root()
    _,truth = association_fixture()
    root = replace(root,capture_ids=ids,supervision=replace(root.supervision,truth=truth(2.2)))
    versions = []
    for event,expected in ((1.2,1.),(1.6,.6),(None,0.)):
        version = replace(root.version,perception_hash=collection.perception_identity(estimator_type(event)))
        updated = module.with_current_perception(root,version,store=store,tap=tap,
            estimator_factory=lambda v,m:estimator_type(event))
        cache = updated.perception_caches[version.perception_hash]
        assert float(cache.history.context[0,-1,13])==pytest.approx(expected)
        assert bool(cache.history.context[0,-1,14])==(event is not None)
        versions.append(version.perception_hash)
    assert len(set(versions))==3 and tap.forwards==21


def test_updated_current_cache_survives_duplicate_root_progress_versions():
    from dataclasses import replace
    module = lineage_api()
    root = role_root()
    cache = root.perception_caches['p1']
    current = replace(cache,perception_hash='p2',audit=replace(cache.audit,perception_hash='p2'),
                      context_state={**cache.context_state,'perception_hash':'p2'})
    updated = replace(root,perception_caches={**root.perception_caches,'p2':current})
    assert len(module.TeacherDataset([root,updated],perception_hash='p2'))==1


def test_root_storage_externalizes_completed_branch_traces_across_atomic_progress(tmp_path):
    from dataclasses import replace
    from wam_sweep.data.store import EpisodeStore, StoredRoot
    module = lineage_api()
    store = EpisodeStore(tmp_path,canonical_budget())
    root = role_root()
    first = store.put('roots',root)
    unknown = replace(root.branches[0],index=1,selection=None,terminal_result=None,
                      continuation_version=None,terminal_count=None)
    second = store.put('roots',replace(root,branches=(*root.branches,unknown)))
    # The two progress versions share ONE immutable physical trace blob.
    archive = store.root_archive(second)
    assert isinstance(archive,StoredRoot)
    assert archive.branches[0].trace_id == archive.branches[1].trace_id
    assert archive.metadata.branches == ()
    loaded = store.get('roots',second)
    assert len(loaded.branches)==2 and len(loaded.branches[0].trace.truths)==25
    np.testing.assert_array_equal(module.PredictionDataset([loaded],perception_hash='p1')[0]['supervision'].object_mean,
                                  module.PredictionDataset([root],perception_hash='p1')[0]['supervision'].object_mean)
    assert store.get('roots',first).branches[0].terminal_result.success
