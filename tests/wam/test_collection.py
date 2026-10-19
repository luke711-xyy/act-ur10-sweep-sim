"""Task8 persistence/collection contracts; real fixtures are bounded, not quotas."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def data_api():
    assert importlib.util.find_spec('wam_sweep.data') is not None, 'Task8 versioned data package missing'
    return importlib.import_module('wam_sweep.data.store')


def collection_api():
    assert importlib.util.find_spec('wam_sweep.data.collection') is not None, 'Task8 branch collection missing'
    return importlib.import_module('wam_sweep.data.collection')


def safe_inputs():
    from wam_sweep.actions import ActionCodec
    from wam_sweep.contracts import HistoryBatch, ReferencePreview
    import torch
    module = collection_api()
    history = HistoryBatch(torch.zeros(1,11,6,34),torch.zeros(1,11,9),torch.zeros(1,11,15),
                           torch.zeros(1,11,6),torch.zeros(1,11))
    reference = ReferencePreview(np.tile([.1,0.,.01,0.],(25,1)),1.+np.arange(25)*.04,np.ones(25),1.,'known')
    return module.SafeInputs(history,reference,ActionCodec().snapshot())


def test_selection_is_three_unique_causal_branches_and_probabilities_are_recorded():
    module = collection_api()
    inputs = safe_inputs()
    programs = module.generate_programs(inputs,seed=13,target_count=2)
    assert len(programs) == 12
    np.testing.assert_array_equal(programs[0].knots,np.zeros((5,3)))
    def score(safe,programs,goal):
        assert set(vars(safe)) == {'history','reference','codec_state'}
        assert goal == 2
        return np.array([-10.,3.,2.,1.,0.,0.,0.,0.,0.,0.,0.,0.])
    selected = module.choose_continuations(inputs,programs,target_count=2,seed=91,scorer=score,scorer_hash='fixture-current-score-v1')
    assert len({item.index for item in selected}) == 3
    assert [item.role for item in selected] == ['reference','hope','random']
    assert [item.index for item in selected[:2]] == [0,1]
    assert [item.probability for item in selected] == [1.,1.,.1]
    assert all(np.abs(p.knots).max()<=1. for p in programs)


@pytest.mark.parametrize('reason,status',[
    ('exact target count reached','success'),
    ('2-second collection window elapsed','task_failure'),
    ('45-second shared episode timeout','task_failure'),
    ('1200-frame shared episode timeout','task_failure'),
    ('temporary hybrid safety: force ceiling or nonfinite force','safety_stop'),
    ('unknown producer fault','infrastructure_error')])
def test_native_outcome_is_distinct_from_preparation_not_ready(reason,status):
    from types import SimpleNamespace
    module = collection_api()
    backend = SimpleNamespace(done=True,termination_reason=reason,
                               guards={'safety_reason':reason if status=='safety_stop' else ''})
    preparation = SimpleNamespace(status='native_failure',ready=False,paused=False,visual_reason='')
    result = module.episode_result(backend,preparation=preparation)
    assert (result.status,result.native_reason) == (status,reason)


def test_visual_closed_failure_and_valid_paused_checkpoint_are_distinct():
    from types import SimpleNamespace
    module = collection_api()
    backend = SimpleNamespace(done=False,closed=True,termination_reason='',guards={'safety_reason':''})
    failure = SimpleNamespace(status='perception_invalid',visual_reason='uncertain contour',paused=True)
    assert module.episode_result(backend,preparation=failure).status == 'perception_invalid'
    pause = SimpleNamespace(status='paused',visual_reason='',paused=True)
    backend.closed = False
    assert module.episode_result(backend,preparation=pause).status == 'incomplete'
    from wam_sweep.snapshots import EpisodeSnapshot
    checkpoint = EpisodeSnapshot('fixture',None,dict(paused=True,done=False,context=None),{},'preparation:post-physics/pre-ack')
    with pytest.raises(ValueError,match='valid paused checkpoint'):
        module.collect_root(checkpoint,(),None,backend,None)
    assert not backend.closed


def test_native_bootstrap_result_has_no_visual_status_and_does_not_hide_original_faults():
    from types import SimpleNamespace
    from wam_sweep.native import PreparationResult
    module = collection_api()
    native = PreparationResult(True,False,np.zeros((0,4)),'')
    backend = SimpleNamespace(done=False,termination_reason='',guards={})
    assert module.episode_result(backend,preparation=native).status=='incomplete'


def test_control_trace_consumes_actual_native_execution_timestamps_and_masks_partial_frames():
    from types import SimpleNamespace
    from wam_sweep.native import StepResult
    from wam_sweep.actions import ActionCodec
    module = collection_api()
    window = SimpleNamespace(truth='offline',label='label',recent_ticks=[])
    backend = SimpleNamespace(env=SimpleNamespace(time=1.),codec=ActionCodec(),
        participants={'offline_labels':window},_robot_context=lambda:'safe robot',
        reference_preview=lambda:SimpleNamespace(targets=np.zeros((25,4)),valid=np.zeros(25)),
        step_target=lambda target,raw:StepResult(np.zeros((2,4)),'native stop',True,(1.01,1.02)))
    trace = module.ExecutionTrace()
    module._execute(backend,np.zeros(3),trace)
    assert trace.controls[0].times == (1.01,1.02)
    assert trace.truths == [] and trace.labels == []


def test_raised_native_safety_keeps_original_cause_and_masks_the_unreturned_frame():
    from types import SimpleNamespace
    module = collection_api()
    reason = 'unsafe robot collision; terminate temporary trial'
    backend = SimpleNamespace(env=SimpleNamespace(time=1.),done=False,termination_reason='',guards={},
        _commands=[],_robot_context=lambda:'safe robot',
        reference_preview=lambda:SimpleNamespace(targets=np.zeros((25,4)),valid=np.zeros(25)),
        participants={'offline_labels':SimpleNamespace(recent_ticks=[])})
    def native_failure(target,raw):
        backend.env.time = 1.01
        backend._commands.append(np.zeros(4))
        backend.done,backend.termination_reason = True,reason
        raise RuntimeError(reason)
    backend.step_target = native_failure
    trace = module.ExecutionTrace()
    module._execute(backend,np.zeros(3),trace)
    assert trace.interruption['result'].status == 'safety_stop'
    assert trace.interruption['reason']==reason and trace.interruption['end_time']==1.01
    assert trace.controls==trace.truths==[]


def test_invalid_live_capture_is_forensic_evidence_not_a_trainable_prefix(tmp_path):
    from types import SimpleNamespace
    from wam_sweep.artifacts import BudgetLedger
    from wam_sweep.config import WAMConfig
    from wam_sweep.contracts import CAMERA_VIEWS, OnlineObservation, RobotContext
    from wam_sweep.native import CaptureMetadata, CameraCalibration
    from wam_sweep.data.store import EpisodeStore
    module = collection_api()
    account = Path('.venv/task-2-budget.json')
    budget = BudgetLedger(account,owned_roots=json.loads(account.read_bytes())['policy']['roots'],config=WAMConfig().budget)
    store = EpisodeStore(tmp_path,budget)
    robot = RobotContext([0.,0.,.1],0.,[0.,0.,0.],0.,'prepare',[0.,0.,0.],[0.,0.,0.],0.,False,False,0.,False)
    raw = OnlineObservation(1.,(),robot,{v:np.zeros((320,320,3),np.uint8) for v in CAMERA_VIEWS})
    calibration = CaptureMetadata(1.,{v:CameraCalibration(np.eye(3),np.eye(4)) for v in CAMERA_VIEWS},'fixture')
    def fail(raw): raise ValueError('controlled live pixel authentication failure')
    backend = SimpleNamespace(capture_metadata=lambda t:calibration)
    tap = SimpleNamespace(features_only=fail)
    retained = module.RetainedTap(tap,store,backend=backend)
    with pytest.raises(ValueError,match='authentication'):
        retained.features_only(raw)
    assert retained.capture_ids==[]
    failures = list((tmp_path/'failures').iterdir())
    assert len(failures)==1
    forensic = store.get('failures',failures[0].name)
    assert forensic['stage']=='capture_contract' and forensic['timestamp']==1.
    assert set(forensic['images'])==set(CAMERA_VIEWS)


@pytest.mark.parametrize('bootstrap',[True,False])
def test_collection_refuses_provisional_split_before_opening_any_backend(bootstrap):
    from wam_sweep.config import WAMConfig
    from wam_sweep.splits import make_split_manifest
    module = collection_api()
    split = make_split_manifest(WAMConfig(),dict(complete_known_inputs=True,seeds=[],fingerprints=[],reserved_seed_ranges=[]))
    def forbidden(spec): raise AssertionError('opened backend before all852 physical families frozen')
    with pytest.raises(ValueError,match='not frozen/ready'):
        if bootstrap:
            module.collect_bootstrap(forbidden,None,split,None,None)
        else:
            module.collect_phase(0,None,None,forbidden,None,split,None,None)


def test_closed_failed_main_is_retained_and_never_reopened_on_restart(tmp_path):
    from dataclasses import asdict
    from types import SimpleNamespace
    from wam_sweep.contracts import EpisodeSpec, InputIdentity, ModelVersion, EpisodeResult
    from wam_sweep.data.store import EpisodeStore
    from wam_sweep.artifacts import BudgetLedger
    from wam_sweep.config import WAMConfig
    from wam_sweep.inputs import digest_json
    module = collection_api()
    account = Path('.venv/task-2-budget.json')
    budget = BudgetLedger(account,owned_roots=json.loads(account.read_bytes())['policy']['roots'],config=WAMConfig().budget)
    store = EpisodeStore(tmp_path,budget)
    identity = InputIdentity('fixture','manifest','act',{k:'hash' for k in ('perception','control','preprocessing','config')})
    version = ModelVersion(-1,'act','fixture','act','p1','unused','c1',identity)
    spec = EpisodeSpec('closed-parent',17,1,'train',0,'act',0)
    episode_id = digest_json(dict(spec=asdict(spec),version=asdict(version)))
    record = module.MainRecord(episode_id,spec,version,'main',(),(),EpisodeResult('perception_invalid','bad contour'),None)
    key = module._publish_main(record,store)
    layout = SimpleNamespace(parent_id=spec.parent_id,seed=17,target_count=1,split='train',phase=0)
    def forbidden(spec): raise AssertionError('closed failed parent was retried')
    config = module.CollectionConfig(version,lambda *a:None,'owned-token')
    returned,roots,deferred = module._main(layout,forbidden,store,config,module.ACTReferenceController(version),bootstrap=False)
    assert returned==key and roots==deferred==[]


@pytest.mark.parametrize('sufficient',[False,True],ids=['insufficient','sufficient'])
def test_reserved_phase_forecast_controls_backend_admission_before_any_reset(tmp_path,sufficient):
    from dataclasses import asdict, replace
    from types import SimpleNamespace
    from wam_sweep.contracts import InputIdentity, ModelVersion
    from wam_sweep.artifacts import BudgetLedger
    from wam_sweep.config import WAMConfig
    from wam_sweep.data.store import EpisodeStore
    module = collection_api()
    path = Path('.venv/task-2-budget.json')
    state = json.loads(path.read_bytes())
    budget = BudgetLedger(path,owned_roots=state['policy']['roots'],config=WAMConfig().budget)
    identity = InputIdentity('fixture','manifest','act',{k:'hash' for k in ('perception','control','preprocessing','config')})
    version = ModelVersion(-1,'act','fixture','act','p1','unused','c1',identity)
    config = module.CollectionConfig(version,lambda *a:None,state['active']['reservation'])
    store = EpisodeStore(tmp_path,budget)
    remaining = module._remaining(budget,config)
    assert remaining>10., 'fixture needs time for its own admission assertions'
    # The caller may reserve 240s (focused) OR 1400s (full WAM). Set each
    # side relative to its ACTUAL remaining reservation, without mutating it,
    # mocking admission, changing the ledger clock, or starting another clock.
    config = replace(config,root_seconds_forecast=remaining/2 if sufficient else remaining+60.)
    print(f'phase admission sufficient={sufficient}: remaining={remaining}, forecast={config.root_seconds_forecast}')
    controller = module.ACTReferenceController(version)
    factory = module.ContinuationFactory(version,module.ACTReferenceController,lambda *a:None)
    # Structural manifest only: no settled fingerprints or real resets claimed.
    # Stop at the backend-factory boundary even when admission is sufficient.
    layout = SimpleNamespace(parent_id='fixture',seed=1,target_count=1,split='train',phase=0)
    split = SimpleNamespace(require_ready=lambda consumer:None,
        binding_metadata={'input_identity':asdict(identity)},records=(layout,),manifest_hash='fixture')
    opened = []
    class BackendBoundaryReached(RuntimeError): pass
    def boundary(spec):
        opened.append(spec)
        raise BackendBoundaryReached('admitted before any real reset')
    if sufficient:
        with pytest.raises(BackendBoundaryReached,match='admitted before any real reset'):
            module.collect_phase(0,controller,factory,boundary,store,split,budget,config)
        assert len(opened)==1 and opened[0].parent_id==layout.parent_id
        cursor = next((store.directory/'cursors').iterdir())
        record = store.get('trajectories',store.cursor(cursor.stem)['record'])
        assert record.result.status=='infrastructure_error'
        assert record.result.native_reason=='admitted before any real reset'
    else:
        summary = module.collect_phase(0,controller,factory,boundary,store,split,budget,config)
        assert summary.status=='not_ready' and summary.main_records==() and not summary.coverage.ready
        assert opened==[] and not (store.directory/'cursors').exists()


@pytest.fixture(scope='module')
def portable_directory(tmp_path_factory):
    data_api()
    directory = tmp_path_factory.mktemp('task8-portable')
    result = subprocess.run([sys.executable,'-c',PORTABLE_PROBE,'write',str(directory)],
                            capture_output=True,text=True,timeout=180,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==0,result.stdout+result.stderr
    return directory


def test_externalized_snapshot_restores_in_fresh_source_checked_process(portable_directory):
    # Break: retaining a live handle, dropping calibration/provider/RNG state,
    # or requiring the original process makes the second independent run fail.
    data_api()
    result = subprocess.run([sys.executable,'-c',PORTABLE_PROBE,'read',str(portable_directory)],
                            capture_output=True,text=True,timeout=180,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==0,result.stdout+result.stderr
    evidence = json.loads((portable_directory/'portable-evidence.json').read_text())
    assert evidence['fresh_process'] and evidence['no_replan']
    assert evidence['commands_equal'] and evidence['state_equal']
    assert evidence['history_equal'] and evidence['calibration_equal']
    assert evidence['monitor_equal'] and evidence['termination_equal']
    assert evidence['replayed_backbone'] == 5 and evidence['replayed_act'] == 0


def test_only_selected_branches_have_terminal_labels(portable_directory):
    result = subprocess.run([sys.executable,'-c',PORTABLE_PROBE,'branches',str(portable_directory)],
                            capture_output=True,text=True,timeout=360,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==0,result.stdout+result.stderr
    evidence = json.loads((portable_directory/'branch-evidence.json').read_text())
    assert evidence['branches']==12 and evidence['terminal_labels']==3
    assert evidence['steps']==[25]*12
    assert len(set(evidence['selected']))==3
    assert evidence['no_future_act'] and evidence['actual_tcp_codec']
    assert evidence['restored_origin_unchanged']


PORTABLE_PROBE = r'''
import copy, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
from wam_sweep.artifacts import BudgetLedger, atomic_checkpoint
from wam_sweep.config import WAMConfig
from wam_sweep.contracts import EpisodeSpec
from wam_sweep.data.store import EpisodeStore
from wam_sweep.data.collection import (RetainedTap, OfflineWindow, VisualSession, RootRecord,
    SafeInputs, RootSupervision, CollectionIdentity, perception_identity, ContinuationFactory, generate_programs, collect_root, ACTReferenceController)
from wam_sweep.data.datasets import associate_root, PerceptionCache
from wam_sweep.history import HistoryBuffer
from wam_sweep.inputs import verify_inputs, assert_native_origin
from wam_sweep.native import FrozenSweepBackend
from wam_sweep.preparation import prepare_visually, VisualPreparationMonitor
from wam_sweep.vision import FrozenFeatureTap, BEVEncoder, SixQueryDetector, VisualStateEstimator
from wam_sweep.vision.detector import DetectionOutput

mode, directory = sys.argv[1:]
root = Path.cwd()
account = root/'.venv/task-2-budget.json'
budget = BudgetLedger(account, owned_roots=json.loads(account.read_bytes())['policy']['roots'],
                      config=WAMConfig.load(root/'configs/wam_visual_v1.yaml').budget)
store = EpisodeStore(directory, budget)
identity = verify_inputs('/Users/huazhi_luke/.local/share/wam-sweep-v1/inputs/r8', allow_unlisted=False)
spec = EpisodeSpec('task8-portable-fixture',2120018,1,'train',0,'act',0)
backend = FrozenSweepBackend.open(identity,spec,budget=budget,native_lock=root/'.venv/task-5-native.lock')
tap = None
try:
    import torchvision.models._api as vision_api
    def forbidden(*a, **kw): raise AssertionError('download/oracle/replan forbidden')
    vision_api.load_state_dict_from_url = forbidden
    backend.prepare_bootstrap_oracle = forbidden
    torch.manual_seed(808)
    tap = FrozenFeatureTap(backend)
    bev, detector = BEVEncoder().to('mps'), SixQueryDetector().to('mps')
    # Explicit measurement fixture, independent of native truth; no training claim.
    centers = [[-.05,-.14],[-.08,-.02],[.04,.08],[-.15,.18],[.12,.23],[-.12,-.27]]
    def measurement(out):
        def t(a): return torch.tensor(a,dtype=torch.float32,device='mps')
        return DetectionOutput(out.timestamp,t([centers]),t(np.zeros((1,6,2))),
            t(np.full((1,6,2),.008)),t(np.tile([0.,1.],(1,6,1))),t(np.zeros((1,6,1))),
            t(np.tile([100.,-100.],(1,6,1))),t(np.full((1,6,1),20.)),
            t(np.tile([20.,-20.,-20.],(1,6,1))),t(np.tile(np.eye(2)*1e-6,(1,6,1,1))),
            t(np.eye(6,16)[None]))
    detector.forward = measurement
    estimator = VisualStateEstimator(bev,detector,identity,tap.capture_metadata)
    history = HistoryBuffer()
    retained = RetainedTap(tap,store,backend=backend)
    from wam_sweep.contracts import ModelVersion
    from wam_sweep.inputs import digest_json, file_hash
    from dataclasses import asdict
    source_version = ModelVersion(-1,'act',str(Path(identity.root)/'base_model'),identity.model_hash,
        perception_identity(estimator),'not-used-by-act',digest_json(dict(input=asdict(identity),
        perception_manifest=estimator.checkpoint()['manifest'],collection=file_hash('wam_sweep/data/collection.py'),
        store=file_hash('wam_sweep/data/store.py'),datasets=file_hash('wam_sweep/data/datasets.py'),
        codec=backend.codec.snapshot()['protocol'],helper=backend.codec.snapshot()['helper_sha256'],
        native=file_hash('wam_sweep/native.py'),snapshot=file_hash('wam_sweep/snapshots.py'),kind='act',phase=-1)),identity)
    offline = OfflineWindow()
    offline.tick(backend.capture_truth())
    backend.bind_offline_truth_tick(offline.tick)
    for name, provider in [('features',retained),('estimator',estimator),('history',history),('offline_labels',offline),
                           ('collection_identity',CollectionIdentity(source_version))]:
        backend.bind_provider(name,provider,beliefs=False)
    if mode == 'write':
        # Retain ALL historical calibration before native's latest-only channel advances.
        metadata = {}
        original_features = tap.features_only
        def retain(raw):
            metadata[raw.timestamp] = backend.capture_metadata(raw.timestamp)
            return original_features(raw)
        tap.features_only = retain
        import sim.act.expert as expert
        planner = expert.expert_plan_candidates
        calls = []
        def counted(*a, **kw):
            calls.append(1)
            return planner(*a, **kw)
        expert.expert_plan_candidates = counted
        paused = prepare_visually(backend,estimator,tap=retained,history=history,
                                  on_observation=lambda safe: False)
        assert paused.status == 'paused' and not backend.closed
        snapshot_id = store.save_snapshot(backend.snapshot(), captures=metadata)
        checkpoint_id = store.put('models',estimator.checkpoint())
        paused_history = history.batch()
        paused_monitor = backend.participants['visual_preparation'].snapshot()
        wrist = tap.capture_metadata(history.snapshot()['observations'][-1].timestamp)
        ready = prepare_visually(backend,estimator,tap=retained,history=history,resume=True)
        assert ready.ready, (ready.status,ready.visual_reason,ready.termination_reason)
        expected = backend.snapshot()
        assert len(calls) == 1
        session = VisualSession(backend,retained,estimator,history)
        session.from_raw(backend.advance_visual_clock(),query=False)
        # Bounded terminal fixture: genuinely execute the native inactive hold
        # on the SAME45s clock, without advancing time by assignment or querying
        # future ACT. Reacquire three actual5Hz captures after the masked gap.
        while backend.episode_time<42.4-1e-8 and not backend.done:
            backend.step_target(None,raw=np.zeros(3))
        assert not backend.done,backend.termination_reason
        session.observe(query=False)
        session.from_raw(backend.advance_visual_clock(),query=False)
        session.from_raw(backend.advance_visual_clock(),query=True)
        assert history.batch().object_valid[0,-1].sum()==6
        backend.codec.begin_program()
        ready_snapshot_id = store.save_snapshot(backend.snapshot(),captures=retained.metadata)
        # Real known-reference commands/state after a serialized READY root,
        # in addition to the earlier saved valid preparation PAUSE.
        ready_commands = []
        for _ in range(5):
            ready_commands.extend(backend.step_target(None,raw=np.zeros(3)).actions)
        ready_future = backend.snapshot()
        key = store.put('evidence',dict(pid=os.getpid(),snapshot_id=snapshot_id,checkpoint_id=checkpoint_id,
            ready_snapshot_id=ready_snapshot_id,ready_commands=np.asarray(ready_commands),
            ready_integration=ready_future.physics.integration,ready_time=float(backend.env.time),
            history=paused_history,monitor=paused_monitor,wrist=wrist,expected=dict(
                integration=expected.physics.integration,commands=expected.state['commands'],
                low_steps=expected.state['low_steps'],origin=expected.state['origin'],
                reason=expected.state['reason'],done=expected.state['done'])))
        store.publish_cursor('portable',dict(evidence=key))
    else:
        saved = store.get('evidence',store.cursor('portable')['evidence'])
        estimator.load_checkpoint(store.get('models',saved['checkpoint_id']))
        monitor = VisualPreparationMonitor(backend,estimator,retained,history)
        backend.bind_provider('visual_preparation',monitor,beliefs=False)
        # Loading source checked runtime and matching providers precedes unpickling native records.
        backend.restore(store.load_snapshot(saved['snapshot_id']),image_reader=store.read_image)
        assert os.getpid() != saved['pid']
        for name in ('objects','brush','context','object_valid','history_valid'):
            torch.testing.assert_close(getattr(history.batch(),name),getattr(saved['history'],name),rtol=0,atol=0)
        assert monitor.snapshot() == saved['monitor']
        actual_wrist = tap.capture_metadata(saved['wrist'].timestamp)
        np.testing.assert_array_equal(actual_wrist.cameras['wrist'].camera_to_world,
                                      saved['wrist'].cameras['wrist'].camera_to_world)
        counts = dict(act=0,backbone=0)
        binding = backend.bind_frozen_policy()
        handles = [binding.policy.model.register_forward_hook(lambda *a:counts.__setitem__('act',counts['act']+1)),
                   binding.policy.model.backbone.register_forward_hook(lambda *a:counts.__setitem__('backbone',counts['backbone']+1))]
        tap.replay_features()
        assert counts == dict(act=0,backbone=5)
        replay_counts = counts.copy()
        import sim.act.expert as expert
        expert.expert_plan_candidates = forbidden
        result = prepare_visually(backend,estimator,tap=retained,history=history,resume=True)
        assert result.ready
        actual, expected = backend.snapshot(), saved['expected']
        np.testing.assert_array_equal(actual.physics.integration,expected['integration'])
        np.testing.assert_array_equal(actual.state['commands'],expected['commands'])
        assert actual.state['low_steps'] == expected['low_steps']
        assert actual.state['origin'] == expected['origin']
        assert actual.state['reason'] == expected['reason']
        assert actual.state['done'] == expected['done']
        backend.restore(store.load_snapshot(saved['ready_snapshot_id']),image_reader=store.read_image)
        ready_root = backend.snapshot()
        ready_commands = []
        for _ in range(5):
            ready_commands.extend(backend.step_target(None,raw=np.zeros(3)).actions)
        np.testing.assert_array_equal(ready_commands,saved['ready_commands'])
        np.testing.assert_array_equal(backend.snapshot().physics.integration,saved['ready_integration'])
        assert float(backend.env.time)==saved['ready_time']
        backend.restore(store.load_snapshot(saved['ready_snapshot_id']),image_reader=store.read_image)
        assert_native_origin(identity)
        atomic_checkpoint(Path(directory)/'portable-evidence.json',json.dumps(dict(
            fresh_process=True,no_replan=True,commands_equal=True,state_equal=True,
            history_equal=True,calibration_equal=True,monitor_equal=True,termination_equal=True,
            replayed_backbone=replay_counts['backbone'],replayed_act=replay_counts['act'],
            fixture='fixed detector measurements; no trained perception claim',
            commands=len(actual.state['commands']),absolute_time=float(backend.env.time),
            episode_age=backend.episode_time)).encode(),budget)
        if mode=='branches':
            from wam_sweep.contracts import ModelVersion, CandidateProgram
            from wam_sweep.inputs import digest_json, file_hash
            from dataclasses import asdict, replace
            session = VisualSession(backend,retained,estimator,history)
            raw = history.snapshot()['observations'][-1]
            backend.codec.begin_program()
            version = source_version
            association = associate_root(history,offline.truth,perception_hash=version.perception_hash,
                                         label_hash=offline.labeler.config.config_hash)
            snapshot_id = saved['ready_snapshot_id']
            inputs = SafeInputs(history.batch(),backend.reference_preview(),backend.codec.snapshot())
            cache = PerceptionCache(version.perception_hash,history.batch(),association,True,
                                    tuple(retained.capture_ids),raw.timestamp)
            root_record = RootRecord('real-branch-fixture',spec,version,snapshot_id,inputs,tuple(retained.capture_ids),
                                     RootSupervision(offline.truth,offline.snapshot(),association),
                                     perception_caches={version.perception_hash:cache})
            synchronizations = []
            class ACTReference(ACTReferenceController):
                def synchronize(self,*,codec_state,executed):
                    super().synchronize(codec_state=codec_state,executed=executed)
                    assert all(not hasattr(e,'commands') and not hasattr(e,'truth') for e in executed)
                    synchronizations.append(len(executed))
            def load_estimator(version,metadata_provider):
                b,d = BEVEncoder().to('mps'),SixQueryDetector().to('mps')
                d.forward = measurement
                e = VisualStateEstimator(b,d,identity,metadata_provider)
                e.load_checkpoint(store.get('models',saved['checkpoint_id']))
                return e
            factory = ContinuationFactory(version,lambda version:ACTReference(version),load_estimator)
            collected = collect_root(store.load_snapshot(snapshot_id),generate_programs(inputs,seed=83,target_count=1),factory,backend,store)
            assert len(collected.branches)==12
            assert sum(b.terminal_result is not None for b in collected.branches)==3
            assert all(b.terminal_result is None for b in collected.branches if b.selection is None)
            assert all(len(b.trace.controls)==25 for b in collected.branches)
            assert all(len(b.trace.truth_ticks)==100 for b in collected.branches)
            from wam_sweep.actions import ActionCodec
            different_tcp = False
            for branch in collected.branches:
                decoder = ActionCodec(bounds=root_record.inputs.codec_state['bounds'])
                decoder.restore(root_record.inputs.codec_state)
                decoder.begin_program()
                for event in branch.trace.controls:
                    measured = np.r_[event.robot_before.tcp_xyz,event.robot_before.yaw]
                    expected = decoder.decode(measured,event.nominal_reference.astype(np.float32),event.raw,.04,event.reference_valid)
                    np.testing.assert_array_equal(expected.residual,event.robot.residual)
                    assert event.codec_state['program_anchor']==root_record.inputs.codec_state['program_anchor']
                    different_tcp |= event.reference_valid and not np.array_equal(measured,event.nominal_reference)
            assert different_tcp  # this probe could detect substituting nominal target for TCP
            assert all(b.continuation_version==version for b in collected.branches if b.selection is not None)
            assert synchronizations.count(25)==3  # full actual residual prefix synchronization
            assert not backend.done and backend.origin==actual.state['origin']
            # A distinct CURRENT perception version replays retained causal RGB
            # with the same backbone and recomputes OFFLINE slot correspondence.
            from wam_sweep.data.datasets import with_current_perception, TeacherDataset, TerminalDataset
            checkpoint = store.get('models',saved['checkpoint_id'])
            def current_estimator(version,metadata_provider):
                b,d = BEVEncoder().to('mps'),SixQueryDetector().to('mps')
                b.load_state_dict(checkpoint['weights']['bev'])
                d.load_state_dict(checkpoint['weights']['detector'])
                with torch.no_grad(): next(b.parameters()).add_(.00001)
                d.forward = measurement
                # Changed weights are bound BEFORE creating the new estimator.
                return VisualStateEstimator(b,d,identity,metadata_provider)
            probe = current_estimator(version,tap.capture_metadata)
            current_version = replace(version,perception_hash=perception_identity(probe))
            assert current_version.perception_hash!=version.perception_hash
            updated = with_current_perception(collected,current_version,estimator_factory=current_estimator,store=store,tap=tap)
            cache2 = updated.perception_caches[current_version.perception_hash]
            assert cache2.association.perception_hash==current_version.perception_hash
            assert cache2.association is not collected.association
            assert cache2.capture_ids==root_record.capture_ids and cache2.root_time==root_record.root_truth.timestamp
            assert len(TeacherDataset([updated],perception_hash=current_version.perception_hash))==sum(b.terminal_result.success for b in collected.branches if b.terminal_result is not None)
            assert len(TerminalDataset([updated],perception_hash=current_version.perception_hash,
                                      continuation_hash=version.continuation_hash))==3
            evidence = dict(branches=len(collected.branches),terminal_labels=3,
                steps=[len(b.trace.controls) for b in collected.branches],
                selected=[b.index for b in collected.branches if b.selection is not None],
                statuses=[b.terminal_result.status for b in collected.branches if b.selection is not None],
                reasons=[b.terminal_result.native_reason for b in collected.branches if b.selection is not None],
                no_future_act=True,actual_tcp_codec=True,restored_origin_unchanged=True,
                current_perception_replayed=True,current_mapping_recomputed=True,
                old_perception_hash=version.perception_hash,current_perception_hash=current_version.perception_hash,
                continuation_hash=version.continuation_hash,
                probabilities=[b.selection.probability for b in collected.branches if b.selection is not None])
            atomic_checkpoint(Path(directory)/'branch-evidence.json',json.dumps(evidence).encode(),budget)
finally:
    if tap: tap.close()
    backend.close()
'''
