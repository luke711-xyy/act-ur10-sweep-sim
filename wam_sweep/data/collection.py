"""Versioned collection at completed causal transactions, never training.

BackendFactory(EpisodeSpec) owns opening a source-checked backend. Frozen
ContinuationFactory below calls ControllerFactory(ModelVersion); future Task11
controllers additionally implement synchronize(codec_state=...,executed=...).
Only SafeInputs and explicit goal N reach scorers/proposals/controllers. Native
snapshots, truth, association and actual future commands remain private evidence.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np

from ..actions import ActionCodec
from ..contracts import CandidateProgram, Controller, EpisodeResult, EpisodeSpec, HistoryBatch, ModelVersion, OnlineObservation, ReferencePreview, RunIdentity
from ..history import HistoryBuffer
from ..inputs import digest_json, file_hash
from ..labels import CarriedLabeler
from ..preparation import prepare_visually
from .datasets import associate_root


def _diagnostic(primary, operation):
    """Secondary persistence must never replace the active producer exception."""
    try:
        return operation()
    except BaseException as secondary:
        if hasattr(primary,'add_note'):
            primary.add_note(f'secondary diagnostic/cleanup failure: {type(secondary).__name__}: {secondary}')


def _close_resources(*resources):
    """Attempt EVERY release; propagate first failure only without a primary."""
    primary = sys.exc_info()[1]
    first = None
    for resource in resources:
        if resource is None:
            continue
        if primary is not None:
            _diagnostic(primary,resource.close)
        else:
            try:
                resource.close()
            except BaseException as error:
                primary = first = error
    if first is not None:
        raise first


@dataclass(frozen=True)
class SafeInputs:
    history: HistoryBatch
    reference: ReferencePreview
    codec_state: dict

    def __post_init__(self):
        if type(self.history) is not HistoryBatch or type(self.reference) is not ReferencePreview:
            raise TypeError('strict history/reference required')
        codec = ActionCodec(bounds=self.codec_state.get('bounds'))
        codec.restore(self.codec_state)  # exact whitelist + protocol/helper identity
        object.__setattr__(self,'codec_state',copy.deepcopy(codec.snapshot()))


@dataclass(frozen=True)
class Selection:
    index: int
    role: str
    probability: float
    protocol: str = 'reference-hope-uniform-remaining-v1'
    seed: int = 0
    scorer_hash: str = 'current-visual-geometry-v1'


def generate_programs(inputs, *, seed, target_count, proposal=None):
    """Zero + seven smooth bounded perturbations + four visual curves.

    Optional later learned proposal callback receives ONLY (SafeInputs,N,4).
    Supplying it requires an explicitly versioned ContinuationFactory below;
    random and geometric curves are never described as learned action heads.
    """
    rng = np.random.default_rng(seed)
    programs = [CandidateProgram(np.zeros((5,3)))]
    for _ in range(7):
        end = rng.uniform(-.8,.8,3)
        programs.append(CandidateProgram(np.arange(1,6)[:,None]/5*end))
    if proposal is not None:
        additions = tuple(proposal(inputs,target_count,4))
        if len(additions)!=4:
            raise ValueError('exactly four versioned proposals required')
        programs.extend(CandidateProgram(p.knots) for p in additions)
    else:
        valid = inputs.history.object_valid[0,-1].cpu().numpy().astype(bool)
        centers = inputs.history.objects[0,-1,:,0:2].cpu().numpy()[valid]
        tcp = inputs.history.brush[0,-1,:2].cpu().numpy()
        delta = np.zeros(2) if not len(centers) else centers.mean(0)-tcp
        direction = delta/max(np.linalg.norm(delta),1e-9)
        for sign,scale in ((1.,.4),(-1.,.4),(1.,.8),(-1.,.8)):
            end = np.r_[direction*scale,sign*.35*scale]
            programs.append(CandidateProgram(np.arange(1,6)[:,None]/5*end))
    return tuple(programs)


def _geometric_scores(inputs, programs, target_count):
    valid = inputs.history.object_valid[0,-1].cpu().numpy().astype(bool)
    centers = inputs.history.objects[0,-1,:,:2].cpu().numpy()[valid]
    brush = inputs.history.brush[0,-1,:2].cpu().numpy()
    if len(centers):
        centers = centers[np.argsort(np.linalg.norm(centers-brush,axis=1),kind='stable')[:target_count]]
        goal = centers.mean(0)
    else:
        goal = brush
    # Engineering current-estimate score, not GT-future or learned accuracy.
    return np.asarray([-np.linalg.norm(brush + .02*p.knots[-1,:2]-goal) for p in programs])


def choose_continuations(inputs, programs, *, target_count, seed, scorer=None, scorer_hash=None):
    inputs = SafeInputs(inputs.history,inputs.reference,inputs.codec_state)
    if len(programs)!=12 or np.any(programs[0].knots):
        raise ValueError('twelve programs with index0 zero reference required')
    if scorer is not None and not scorer_hash:
        raise ValueError('current-model scorer requires its frozen dependency hash')
    scores = np.asarray((scorer or _geometric_scores)(inputs,programs,target_count),dtype=float)
    if scores.shape!=(12,) or not np.isfinite(scores).all():
        raise ValueError('finite current-model/visual scores required')
    hope = int(np.argmax(scores[1:]))+1
    remaining = [i for i in range(12) if i not in (0,hope)]
    random = int(np.random.default_rng(seed).choice(remaining))
    return tuple(Selection(index,role,probability,seed=seed,
                           scorer_hash=scorer_hash or 'current-visual-geometry-v1')
                 for index,role,probability in ((0,'reference',1.),(hope,'hope',1.),(random,'random',.1)))


def episode_result(backend, *, preparation=None):
    """Original native evidence has priority over preparation readiness.

    Known literal outcomes come from reviewed frozen native producers. Unknown
    native reasons fail closed as infrastructure_error, retaining exact text;
    they are never guessed from a 'native_failure' preparation classification.
    """
    reason = backend.termination_reason
    if backend.done:
        if (backend.guards.get('safety_reason') or reason in (
                'unsafe robot collision; terminate temporary trial',
                'nonfinite physical sensor/state; terminate temporary trial')):
            status = 'safety_stop'
        elif reason == 'exact target count reached':
            status = 'success'
        elif reason in ('2-second collection window elapsed','45-second shared episode timeout',
                        '1200-frame shared episode timeout','no_plan','no_handoff',
                        'target count was not collected','unexpected component entered the target region',
                        'component projection only partially overlaps the collection region',
                        'expert descent ended without reaching 3 mm; no force search/retry permitted'):
            status = 'task_failure'
        else:
            status = 'infrastructure_error'
        return EpisodeResult(status,reason or 'native terminal without a reason')
    if preparation is not None and getattr(preparation,'status',None)=='perception_invalid':
        return EpisodeResult('perception_invalid',reason or 'native_not_terminated')
    return EpisodeResult('incomplete',reason or 'native_not_terminated')


def perception_identity(estimator):
    """Manifest binds weights AND input/config/calibration/tracker dependencies."""
    return digest_json(estimator.checkpoint()['manifest'])


class VisualEntryContext:
    """Causal VISUAL feature, not the native first-entry/task clock.

    Shared by live/preparation histories and each version's full causal replay.
    Input is already computed beliefs; no truth, N, encoder or ACT access.
    """
    protocol = 'visual-first-estimated-collected-v1'
    threshold = .5

    def __init__(self, perception_hash):
        if not isinstance(perception_hash,str) or not perception_hash:
            raise ValueError('visual context requires frozen perception identity')
        self.perception_hash = perception_hash
        self.first_entry = self.timestamp = None

    def snapshot(self):
        return dict(schema=1,protocol=self.protocol,threshold=self.threshold,
                    perception_hash=self.perception_hash,first_entry=self.first_entry,timestamp=self.timestamp)

    def restore(self, state):
        expected = self.snapshot()
        if (not isinstance(state,dict) or set(state)!=set(expected)
                or any(state[k]!=expected[k] for k in ('schema','protocol','threshold','perception_hash'))):
            raise ValueError('incompatible visual context protocol/perception identity')
        first,last = state['first_entry'],state['timestamp']
        if (any(t is not None and (type(t) not in (float,int) or not math.isfinite(t) or t<0) for t in (first,last))
                or first is not None and (last is None or first>last)):
            raise ValueError('invalid causal visual context clock')
        self.first_entry,self.timestamp = first,last

    def pair(self, timestamp):
        return (0.,False) if self.first_entry is None or timestamp<self.first_entry else (timestamp-self.first_entry,True)

    def apply(self, observation):
        if type(observation) is not OnlineObservation:
            raise TypeError('visual context requires safe estimated observation')
        now = observation.timestamp
        if self.timestamp is not None and now<=self.timestamp:
            raise ValueError('visual context requires strictly causal captures')
        first = self.first_entry
        if first is None and any(b.valid and b.state_probabilities[2]>=self.threshold for b in observation.objects):
            first = now
        age,valid = (0.,False) if first is None else (now-first,True)
        # All stored kinematics/control/time/phase stay bit-for-bit causal.
        safe = replace(observation,robot=replace(observation.robot,visual_entry_age=age,visual_entry_valid=valid))
        self.first_entry,self.timestamp = first,now
        return safe


class ContextHistory(HistoryBuffer):
    """Task7-compatible history adapter with a versioned, restorable clock.

    append(safe) enriches ONLY the two visual fields using existing beliefs and
    returns that enriched observation. Task7 may ignore the return value; its
    model history is still enriched. VisualSession returns it to controllers.
    """
    def __init__(self, perception_hash, *, backend=None):
        super().__init__()
        self.entry = VisualEntryContext(perception_hash)
        self.backend = backend

    def bind_backend(self, backend):
        self.backend = backend
        if self.entry.timestamp is not None:
            backend.update_visual_entry(*self.entry.pair(self.entry.timestamp))

    def append(self, observation):
        previous = self.entry.snapshot()
        try:
            safe = self.entry.apply(observation)
            super().append(safe)
        except BaseException as error:
            _diagnostic(error,lambda:self.entry.restore(previous))
            raise
        if self.backend is not None:
            self.backend.update_visual_entry(safe.robot.visual_entry_age,safe.robot.visual_entry_valid)
        return safe

    def snapshot(self):
        return dict(super().snapshot(),visual_context=self.entry.snapshot())

    def restore(self, state):
        context = VisualEntryContext(self.entry.perception_hash)
        if not isinstance(state,dict) or 'visual_context' not in state:
            raise ValueError('missing versioned visual context; rebuild from retained raw captures')
        context.restore(state['visual_context'])
        history = {k:v for k,v in state.items() if k!='visual_context'}
        observations = history.get('observations',())
        if ((observations[-1].timestamp if observations else None)!=context.timestamp
                or any(not math.isclose(o.robot.visual_entry_age,context.pair(o.timestamp)[0],abs_tol=1e-7)
                       or o.robot.visual_entry_valid!=context.pair(o.timestamp)[1] for o in observations)):
            raise ValueError('history/visual context clock mismatch')
        super().restore(history)
        self.entry = context


class OfflineWindow:
    """Private named participant; never reset carry windows at branch roots."""
    def __init__(self):
        self.labeler = CarriedLabeler()
        self.truth = self.label = None
        self.recent_ticks = []

    def tick(self, truth):
        self.truth, self.label = truth,self.labeler.update(truth)
        self.recent_ticks = [*self.recent_ticks[-3:],truth]

    def snapshot(self):
        return copy.deepcopy(dict(schema=1,window=self.labeler.snapshot(),truth=self.truth,label=self.label))

    def restore(self, state):
        if state['schema']!=1:
            raise ValueError('offline window schema mismatch')
        self.labeler.restore(state['window'])
        self.truth,self.label = copy.deepcopy((state['truth'],state['label']))
        self.recent_ticks = []  # output evidence buffer, not label/filter memory


class CollectionIdentity:
    """Private restorable source/version/role binding, registered BEFORE prep."""
    def __init__(self, version, *, source_role='teacher', run_identity=None):
        if type(version) is not ModelVersion or source_role not in ('teacher','physical','bootstrap'):
            raise ValueError('full source version and explicit collection role required')
        self.version,self.source_role,self.run_identity = version,source_role,run_identity
        from pathlib import Path
        self.producer_hashes = {name:file_hash(Path(__file__).with_name(name+'.py')) for name in ('collection','store','datasets')}

    def snapshot(self):
        return copy.deepcopy(dict(schema=1,version=self.version,source_role=self.source_role,
                                  run_identity=self.run_identity,producer_hashes=self.producer_hashes,
                                  visual_context=VisualEntryContext(self.version.perception_hash).snapshot()))

    def validate(self, state):
        if (state.get('schema')!=1 or state.get('version')!=self.version
                or state.get('source_role')!=self.source_role or state.get('producer_hashes')!=self.producer_hashes):
            raise ValueError('collection source version/role/producer dependency mismatch')
        VisualEntryContext(self.version.perception_hash).restore(state.get('visual_context'))

    def restore(self, state):
        self.validate(state)
        self.run_identity = copy.deepcopy(state['run_identity'])


class RetainedTap:
    """Task6 facade adapter: durable RGB/calibration, zero pyramid disk cache."""
    def __init__(self, tap, store, *, backend, capture_supervision=None):
        self.tap,self.store = tap,store
        self.backend = backend
        self.capture_ids = []
        self.metadata = {}
        self.persist = True
        self.capture_supervision = capture_supervision

    def _packet(self, raw, function):
        calibration = None
        try:
            calibration = self.backend.capture_metadata(raw.timestamp)
            evidence = None if not self.persist or self.capture_supervision is None else self.capture_supervision(raw)
            capture_id = None if not self.persist else self.store.put_capture(raw,calibration,supervision=evidence)
            packet = function(raw)
        except BaseException as error:
            _diagnostic(error,lambda:self.store.put_failed_capture(raw,calibration,error))
            raise
        self.metadata[raw.timestamp] = calibration
        if capture_id is not None:
            self.capture_ids.append(capture_id)
        return packet

    def features_only(self, raw):
        return self._packet(raw,self.tap.features_only)

    def query_act_and_features(self, raw):
        return self._packet(raw,self.tap.query_act_and_features)

    def capture_metadata(self, timestamp):
        return self.tap.capture_metadata(timestamp)

    def snapshot(self):
        return copy.deepcopy(dict(schema=1,tap=self.tap.snapshot(),capture_ids=self.capture_ids,metadata=self.metadata))

    def restore(self, state):
        if state['schema']!=1:
            raise ValueError('retained feature schema mismatch')
        self.tap.restore(state['tap'])
        self.capture_ids,self.metadata = copy.deepcopy((state['capture_ids'],state['metadata']))


class VisualSession:
    """Concrete Task6/7 producer chain. Factories load matching weights first."""
    def __init__(self, backend, tap, estimator, history):
        if not isinstance(history,ContextHistory):
            raise TypeError('ContextHistory required for version-specific visual context')
        self.backend,self.tap,self.estimator,self.history = backend,tap,estimator,history
        history.bind_backend(backend)

    def from_raw(self, raw, *, query=False):
        packet = self.tap.query_act_and_features(raw) if query else self.tap.features_only(raw)
        safe = OnlineObservation(raw.timestamp,self.estimator.estimate(packet),raw.robot,raw.images)
        return self.history.append(safe)

    def observe(self, *, query=False):
        return self.from_raw(self.backend.observe(),query=query)


@dataclass(frozen=True)
class ControlEvent:
    command_time: float
    raw: np.ndarray
    commands: np.ndarray
    times: tuple
    robot: object  # safe RobotContext, actual TCP, not joint/wrench/native36
    codec_state: dict
    robot_before: object = None
    nominal_reference: np.ndarray | None = None
    reference_valid: bool = False


@dataclass(frozen=True)
class AppliedControl:
    timestamp: float
    residual: np.ndarray
    residual_rate: np.ndarray


def _synchronize(controller, backend, controls):
    # Synchronization receives ONLY actual past safe residual/control memory;
    # low-level commands, truth, native36 and private snapshots stay in audit.
    applied = tuple(AppliedControl(event.command_time,event.robot.residual.copy(),event.robot.residual_rate.copy())
                    for event in controls)
    controller.synchronize(codec_state=backend.codec.snapshot(),executed=applied)


@dataclass
class ExecutionTrace:
    controls: list = field(default_factory=list)
    truths: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    capture_ids: list = field(default_factory=list)
    truth_ticks: list = field(default_factory=list)
    interruption: dict | None = None


@dataclass(frozen=True)
class BranchRecord:
    index: int
    program: CandidateProgram
    trace: ExecutionTrace
    short_result: EpisodeResult
    selection: Selection | None = None
    continuation_version: ModelVersion | None = None
    terminal_result: EpisodeResult | None = None
    terminal_count: int | None = None
    continuation: ExecutionTrace | None = None

    def __post_init__(self):
        if type(self.index) is not int or not 0<=self.index<12:
            raise ValueError('branch index must be0..11')
        if self.terminal_result is not None and (self.selection is None or type(self.continuation_version) is not ModelVersion):
            raise ValueError('terminal result requires explicit selection and full frozen version')


@dataclass(frozen=True)
class RootSupervision:
    truth: object
    label_state: dict
    association: object


@dataclass(frozen=True)
class RootRecord:
    root_id: str
    spec: EpisodeSpec
    version: ModelVersion
    snapshot_id: str
    inputs: SafeInputs
    capture_ids: tuple[str, ...]
    supervision: RootSupervision
    branches: tuple[BranchRecord, ...] = ()
    source_role: str = 'teacher'
    proposal_version: str = 'bounded-geometric-v1'
    perception_caches: dict = field(default_factory=dict)
    run_identity: RunIdentity | None = None

    def __post_init__(self):
        if (type(self.spec) is not EpisodeSpec or type(self.version) is not ModelVersion
                or type(self.inputs) is not SafeInputs or type(self.supervision) is not RootSupervision):
            raise TypeError('root requires public spec/version and safe input types')
        if self.source_role not in ('teacher','physical','bootstrap'):
            raise ValueError('unknown root data role')
        indices = [b.index for b in self.branches]
        if len(set(indices))!=len(indices) or len(indices)>12:
            raise ValueError('root branch indices must be unique')

    @property
    def root_truth(self):
        return self.supervision.truth

    @property
    def label_state(self):
        return self.supervision.label_state

    @property
    def association(self):
        return self.supervision.association


@dataclass(frozen=True)
class ContinuationFactory:
    """Task9/11 supplies real ControllerFactory(version) and estimator loader.

    estimator_factory(version,tap.capture_metadata) constructs/loads that exact
    version BEFORE replay. Controller.synchronize receives actual applied safe
    controls and native codec state; no stale controller/filter context survives.
    scorer(SafeInputs,programs,N) is optional current-model hope scoring.
    """
    version: ModelVersion
    controller_factory: Callable
    estimator_factory: Callable
    scorer: Callable | None = None
    scorer_hash: str | None = None

    def __post_init__(self):
        if (type(self.version) is not ModelVersion or not callable(self.controller_factory)
                or not callable(self.estimator_factory) or (self.scorer is not None and
                    (not callable(self.scorer) or not self.scorer_hash))):
            raise ValueError('full continuation version and versioned scorer required')

    def __call__(self, version):
        if version!=self.version:
            raise ValueError('continuation version mismatch')
        controller = self.controller_factory(version)
        if (not isinstance(controller,Controller) or not callable(getattr(controller,'synchronize',None))
                or getattr(controller,'version',None)!=version):
            raise TypeError('Controller plus synchronize(codec_state,executed) required')
        return controller


def _session(backend):
    providers = backend.participants
    return VisualSession(backend,providers['features'],providers['estimator'],providers['history'])


def _execute(backend, raw, trace):
    now = float(backend.env.time)  # private audit clock, never neural input
    before = backend._robot_context()
    reference = backend.reference_preview()
    command_offset = len(getattr(backend,'_commands',()))
    try:
        result = backend.step_target(None,raw=raw)  # actual TCP + SAME live codec
    except Exception as error:
        # These are the two exact reviewed native measure() RuntimeErrors.
        # Preserve original native safety evidence without inventing a returned
        # StepResult, a complete40ms target, or a usable brush observation.
        reason = getattr(backend,'termination_reason','')
        if (type(error) is not RuntimeError or not backend.done or str(error)!=reason or reason not in (
                'unsafe robot collision; terminate temporary trial',
                'nonfinite physical sensor/state; terminate temporary trial')):
            raise
        trace.interruption = dict(result=episode_result(backend),reason=reason,
            exception_type=type(error).__name__,start_time=now,end_time=float(backend.env.time),
            commands=np.asarray(backend._commands[command_offset:]))
        trace.truth_ticks.extend(copy.deepcopy(t) for t in backend.participants['offline_labels'].recent_ticks if t.timestamp>now+1e-8)
        return None
    trace.controls.append(ControlEvent(now,np.array(raw,copy=True),result.actions.copy(),result.executed_at,
                                       backend._robot_context(),backend.codec.snapshot(),before,
                                       reference.targets[0].copy(),bool(reference.valid[0])))
    window = backend.participants['offline_labels']
    trace.truth_ticks.extend(copy.deepcopy(t) for t in window.recent_ticks if t.timestamp>now+1e-8)
    if window.truth is not None and len(result.executed_at)==4:
        trace.truths.append(copy.deepcopy(window.truth))
        trace.labels.append(copy.deepcopy(window.label))
    return result


def replay_perception(capture_ids, *, store, tap, estimator, controller=None):
    """Chronological current-version replay, using original RGB AND calibration.

    Public Task6 restore/replay_features regenerates each retained packet on the
    same frozen backbone, without ACT. Restore the live tap afterwards; the new
    estimator/history/controller remain at the causal endpoint. Caller charges
    all model/replay work on its existing active clock.
    """
    retained = tap.snapshot()
    history = ContextHistory(perception_identity(estimator))
    previous = None
    try:
        for key in capture_ids:
            raw,calibration = store.read_capture(key)
            if previous is not None and raw.timestamp<=previous:
                raise ValueError('replay must contain strictly increasing causal captures')
            packet_state = dict(retained,timestamp=raw.timestamp,observation=raw,metadata=calibration,
                                reference=None,receipt=None)
            tap.restore(packet_state)
            packet = tap.replay_features()
            safe = OnlineObservation(raw.timestamp,estimator.estimate(packet),raw.robot,raw.images)
            safe = history.append(safe)
            if controller is not None:
                if previous is None:
                    controller.reset(safe)
                controller.observe(safe)
            previous = raw.timestamp
    finally:
        primary = sys.exc_info()[1]
        if primary is None:
            tap.restore(retained)
        else:
            _diagnostic(primary,lambda:tap.restore(retained))
    if previous is None:
        raise ValueError('nonempty retained causal history required')
    return history


def collect_root(root, programs, continuation_factory, backend, store) -> RootRecord:
    # A valid paused preparation checkpoint is not consumed/closed by rejection.
    from ..snapshots import EpisodeSnapshot
    if type(root) is EpisodeSnapshot:
        _require_counterfactual_root(root)
    try:
        return _collect_root(root,programs,continuation_factory,backend,store)
    except BaseException:
        retained = backend.participants.get('features')
        _close_resources(getattr(retained,'tap',None),backend)
        raise


def _collect_root(root, programs, continuation_factory, backend, store) -> RootRecord:
    """Twelve fixed1s branches, only three unique versioned terminal continuations.

    No post-root ACT query during fixed branches. FiveHz features-only captures
    update the original providers. Different-version terminal controllers replay
    THEIR own perception and control context before issuing any new ACT query.
    """
    from ..snapshots import EpisodeSnapshot
    if type(root) is EpisodeSnapshot:
        _require_counterfactual_root(root)
        binding = backend.participants.get('collection_identity')
        if not isinstance(binding,CollectionIdentity) or 'collection_identity' not in root.participants:
            raise ValueError('EpisodeSnapshot needs collection_identity: bind_collection before preparation')
        binding.validate(root.participants['collection_identity'])
        backend.restore(root,image_reader=store.read_image)
        root = _root(backend,_session(backend),store,binding.version,backend.spec,None,binding.run_identity)
        root = replace(root,source_role=binding.source_role)
    elif type(root) is not RootRecord:
        raise TypeError('complete EpisodeSnapshot required (RootRecord is an internal persisted request)')
    if root.source_role=='bootstrap':
        raise ValueError('bootstrap is perception-only, never an action teacher')
    if len(programs)!=12 or np.any(programs[0].knots):
        raise ValueError('exactly twelve programs, first zero reference')
    version = continuation_factory.version
    if version.input_identity!=root.version.input_identity:
        raise ValueError('frozen native/ACT/input dependencies changed')
    selection_seed = int(digest_json(dict(seed=root.spec.seed,time=root.inputs.reference.issued_at,
                                         protocol='root-selection-v1'))[:15],16)
    selected = {s.index:s for s in choose_continuations(root.inputs,programs,target_count=root.spec.target_count,
                    seed=selection_seed,scorer=continuation_factory.scorer,scorer_hash=continuation_factory.scorer_hash)}
    snapshot = store.load_snapshot(root.snapshot_id)
    _require_counterfactual_root(snapshot)
    branches = []
    failed = False
    trace = continuation = None
    try:
        for i,program in enumerate(programs):
            continuation = None
            backend.restore(snapshot,image_reader=store.read_image)
            session = _session(backend)
            tap = session.tap
            if not isinstance(tap,RetainedTap):
                raise TypeError('collection root requires retained Task6 feature provider')
            query_count = len(backend._queries)  # private audit, not a policy feature
            start_ids = len(tap.capture_ids)
            trace = ExecutionTrace()
            anchor = backend.codec.begin_program()
            for j in range(25):
                if backend.done:
                    break
                raw = backend.codec.sample(program,j*.04,anchor)
                _execute(backend,raw,trace)
                if (j+1)%5==0 and not backend.done:
                    session.observe(query=False)
            trace.capture_ids = tap.capture_ids[start_ids:]
            if len(backend._queries)!=query_count:
                raise RuntimeError('fixed known-reference branch queried future ACT')
            short = episode_result(backend)
            branch = BranchRecord(i,CandidateProgram(program.knots),trace,short)
            if i in selected:
                continuation = ExecutionTrace()
                if not backend.done:
                    from ..snapshots import capture_random_state, restore_random_state
                    rng = capture_random_state()
                    try:
                        controller = continuation_factory(version)
                        if version.kind=='wam':
                            if not callable(getattr(controller,'set_goal',None)):
                                raise TypeError('WAM continuation requires explicit set_goal(N)')
                            controller.set_goal(root.spec.target_count)
                        estimator = continuation_factory.estimator_factory(version,tap.tap.capture_metadata)
                        if perception_identity(estimator)!=version.perception_hash:
                            raise ValueError('continuation frozen perception dependencies mismatch')
                        history = replay_perception((*root.capture_ids,*trace.capture_ids),store=store,
                                                    tap=tap.tap,estimator=estimator,controller=controller)
                    finally:
                        restore_random_state(rng)
                    _synchronize(controller,backend,trace.controls)
                    live = VisualSession(backend,tap,estimator,history)
                    tap.persist = False  # terminal paths retain state/action/contact evidence
                    # Boundary image was already encoded during forced execution.
                    # A new native ACT query waits for NEXT capture, preserving
                    # valid_from/hold, instead of encoding that image a second time.
                    while not backend.done:
                        program = CandidateProgram(controller.act(backend.reference_preview()).knots)
                        anchor = backend.codec.begin_program()
                        for k in range(5):
                            if backend.done:
                                break
                            _execute(backend,backend.codec.sample(program,k*.04,anchor),continuation)
                        if not backend.done:
                            safe = live.observe(query=True)
                            controller.observe(safe)
                            _synchronize(controller,backend,continuation.controls[-5:])
                    tap.persist = True
                result = episode_result(backend)
                try:
                    truth = backend.capture_truth()
                    count = int(truth.raw.fields['native_collected'].sum()) if truth.raw.fields['native_valid'].all() else None
                except ValueError:
                    if backend.termination_reason!='nonfinite physical sensor/state; terminate temporary trial':
                        raise
                    count = None  # explicit invalid count, not a fabricated zero
                branch = replace(branch,selection=selected[i],continuation_version=version,
                                 terminal_result=result,terminal_count=count,continuation=continuation)
            branches.append(branch)
            partial = replace(root,branches=tuple(branches))
            key = store.put('roots',partial)
            store.publish_cursor(root.root_id,dict(record=key,complete=len(branches)==12))
    except BaseException as error:
        failed = True
        _diagnostic(error,lambda:store.put('failures',dict(root_id=root.root_id,completed=tuple(branches),
                                  failed_branch=i,trace=trace,continuation=continuation,
                                  native_commands=np.asarray(backend._commands),
                                  native_reason=backend.termination_reason,
                                  exception_type=type(error).__name__,reason=str(error))))
        raise
    finally:
        if not failed and not backend.closed:
            backend.restore(snapshot,image_reader=store.read_image)
            _session(backend).tap.persist = True
    return replace(root,branches=tuple(branches))


def _require_counterfactual_root(snapshot):
    state = snapshot.state
    if state.get('paused'):
        raise ValueError('valid paused checkpoint must resume preparation before counterfactual collection')
    if state.get('done') or not state.get('context') or not state['context'].get('ready'):
        raise ValueError('counterfactual root requires completed ready open native state')


class ACTReferenceController:
    """Actual ACT arm: zero residual; native backend owns frozen ACT queries.

    This is not a learned WAM head. Later WAM controllers come ONLY from the
    supplied ControllerFactory(version), with matching immutable dependencies.
    """
    def __init__(self, version):
        if version.kind!='act':
            raise ValueError('reference controller requires frozen ACT version')
        self.version,self.state = version,{}

    def reset(self, initial):
        self.state = dict(timestamp=initial.timestamp)

    def observe(self, observation):
        self.state['timestamp'] = observation.timestamp

    def act(self, reference):
        return CandidateProgram(np.zeros((5,3)))

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, state):
        self.state = copy.deepcopy(state)

    def synchronize(self, *, codec_state, executed):
        self.state['codec_state'] = copy.deepcopy(codec_state)


@dataclass(frozen=True)
class CollectionConfig:
    """Explicit integration, not an implicit launch/training configuration.

    The caller already owns RunLease + linked phase activity/reservation.
    estimator_factory(version,metadata_provider) loads that exact frozen
    perception before observation; phase_reservation is the existing token.
    root_seconds_forecast reserves admission for remaining root work; extra
    roots are drawn only from pre-retained opportunities of these same parents.
    """
    version: ModelVersion
    estimator_factory: Callable
    phase_reservation: str
    root_seconds_forecast: float = 240.
    extra_roots_per_n: int = 50
    capture_visibility: bool = True
    proposal: Callable | None = None
    proposal_hash: str | None = None

    def __post_init__(self):
        if type(self.version) is not ModelVersion or not self.phase_reservation or not callable(self.estimator_factory):
            raise ValueError('full version and caller-owned phase reservation required')
        if (not math.isfinite(self.root_seconds_forecast) or self.root_seconds_forecast<=0
                or type(self.extra_roots_per_n) is not int or not 0<=self.extra_roots_per_n<=50):
            raise ValueError('positive root forecast and bounded extra-root allowance required')
        if self.proposal is not None and not self.proposal_hash:
            raise ValueError('learned proposal callback requires its frozen dependency hash')


@dataclass(frozen=True)
class MainRecord:
    episode_id: str
    spec: EpisodeSpec
    version: ModelVersion
    source_role: str
    capture_ids: tuple
    chunk_ids: tuple
    result: EpisodeResult
    preparation: object
    native_evidence_id: str | None = None
    root_records: tuple[str, ...] = ()
    deferred_records: tuple[str, ...] = ()
    run_identity: RunIdentity | None = None


@dataclass(frozen=True)
class CollectionSummary:
    status: str
    main_records: tuple[str, ...]
    roots: tuple[RootRecord, ...]
    coverage: object
    warnings: tuple[str, ...]


def _policy(split, budget, config):
    from dataclasses import asdict
    split.require_ready('collection')
    if type(config) is not CollectionConfig:
        raise TypeError('CollectionConfig with real factories/reservation required')
    state = json.loads(budget.path.read_bytes())
    if (state['active'] is None or state['active'].get('reservation')!=config.phase_reservation
            or config.phase_reservation not in state['reservations']):
        raise ValueError('collection must consume the existing owned phase activity')
    if split.binding_metadata['input_identity']!=asdict(config.version.input_identity):
        raise ValueError('split/native input dependency mismatch')
    budget.poll().require()


def _remaining(budget, config):
    budget.poll().require()
    state = json.loads(budget.path.read_bytes())
    return state['reservations'][config.phase_reservation]['seconds']


def _attach(backend, store, config, controller, *, monitor=False, source_role='teacher', run_identity=None):
    """Reconstruct closures/models/providers; never pickle a live env/function."""
    from ..vision import FrozenFeatureTap
    from ..preparation import VisualPreparationMonitor
    tap = FrozenFeatureTap(backend)
    estimator = config.estimator_factory(config.version,tap.capture_metadata)
    if perception_identity(estimator)!=config.version.perception_hash:
        tap.close()
        raise ValueError('main frozen perception dependency mismatch')
    history,offline = ContextHistory(config.version.perception_hash,backend=backend),OfflineWindow()
    offline.tick(backend.capture_truth())
    def supervision(raw):
        truth = backend.capture_camera_truth(raw.timestamp) if config.capture_visibility else backend.capture_truth()
        return dict(truth=truth,label=copy.deepcopy(offline.label),
                    label_hash=offline.labeler.config.config_hash,quality_reviewed=False)
    retained = RetainedTap(tap,store,backend=backend,capture_supervision=supervision)
    for name,provider in (('features',retained),('estimator',estimator),('history',history),
                          ('offline_labels',offline),('collection_controller',controller),
                          ('collection_identity',CollectionIdentity(config.version,source_role=source_role,run_identity=run_identity))):
        backend.bind_provider(name,provider,beliefs=False)
    backend.bind_offline_truth_tick(offline.tick)
    if monitor:
        backend.bind_provider('visual_preparation',VisualPreparationMonitor(backend,estimator,retained,history),beliefs=False)
    return VisualSession(backend,retained,estimator,history)


def bind_collection(backend, store, config, controller, *, monitor=False, source_role='teacher', run_identity=None):
    """Future Task11 producer API; call before prepare_visually/native roots.

    On fresh restoration, load config.version's exact estimator/controller,
    pass monitor=True to reconstruct VisualPreparationMonitor, then restore the
    complete stored snapshot. No raw Task7 snapshot is guessed to be versioned.
    """
    return _attach(backend,store,config,controller,monitor=monitor,source_role=source_role,run_identity=run_identity)


def _root(backend, session, store, version, spec, proposal_hash, run_identity=None):
    from .datasets import PerceptionCache
    backend.codec.begin_program()
    offline = backend.participants['offline_labels']
    inputs = SafeInputs(session.history.batch(),backend.reference_preview(),backend.codec.snapshot())
    association = associate_root(session.history,offline.truth,perception_hash=version.perception_hash,
                                 label_hash=offline.labeler.config.config_hash)
    snapshot_id = store.save_snapshot(backend.snapshot(),captures=session.tap.metadata)
    root_id = digest_json(dict(snapshot=snapshot_id,parent=spec.parent_id,phase=spec.phase,
                               time=offline.truth.timestamp))
    captures = tuple(session.tap.capture_ids)
    cache = PerceptionCache(version.perception_hash,inputs.history,association,
                            bool(inputs.history.object_valid[0,-1].sum()==6),captures,offline.truth.timestamp,
                            session.history.entry.snapshot())
    return RootRecord(root_id,spec,version,snapshot_id,inputs,captures,RootSupervision(offline.truth,offline.snapshot(),association),
                      proposal_version=proposal_hash or 'bounded-geometric-v1',
                      perception_caches={version.perception_hash:cache},run_identity=run_identity)


def _publish_main(record, store):
    key = store.put('trajectories',record)
    store.publish_cursor('main-'+record.episode_id,dict(record=key,kind='main',complete=record.result.status!='incomplete'))
    return key


def _main(layout, backend_factory, store, config, controller, *, bootstrap,
          root_quota=0, deferred_quota=0, continuation_factory=None, split_hash='bounded-fixture'):
    spec = EpisodeSpec(layout.parent_id,layout.seed,layout.target_count,layout.split,layout.phase,
                       'act' if config.version.kind=='act' else 'wam',0)
    backend,tap,session = None,None,None
    episode_id = digest_json(dict(spec=asdict(spec),version=asdict(config.version)))
    run_identity = RunIdentity(episode_id,digest_json(asdict(spec)),'pending_preparation',
        digest_json(dict(input_config=config.version.input_identity.native_hashes['config'],
                         perception=config.version.perception_hash,proposal=config.proposal_hash,
                         visibility=config.capture_visibility,extra_roots=config.extra_roots_per_n)),
        config.version.input_identity,split_hash)
    record = MainRecord(episode_id,spec,config.version,'bootstrap' if bootstrap else 'main',(),(),
                        EpisodeResult('incomplete','attempt claimed before native preparation'),None,run_identity=run_identity)
    try:
        previous_key = store.cursor('main-'+episode_id)['record']
    except FileNotFoundError:
        previous_key = None
    if previous_key is not None:
        previous = store.get('trajectories',previous_key)
        if previous.spec!=spec or previous.version!=config.version:
            raise ValueError('durable attempt dependency mismatch')
        if (previous.run_identity is not None and
                (previous.run_identity.config_hash!=run_identity.config_hash or previous.run_identity.split_hash!=split_hash)):
            raise ValueError('durable attempt config/split mismatch')
        if previous.result.status in ('incomplete','infrastructure_error'):
            raise RuntimeError('prior '+previous.result.status+' attempt requires explicit validated checkpoint recovery; no retry')
        extra = [] if previous.result.status in ('perception_invalid','model_invalid') else [store.get('roots',k) for k in previous.deferred_records]
        return previous_key,[store.get('roots',k) for k in previous.root_records],extra
    _publish_main(record,store)  # even opening/reset/identity errors retain the attempt
    roots,deferred,chunks = [],[],[]
    preparation = None
    trace = ExecutionTrace()
    try:
        backend = backend_factory(spec)
        if backend.spec!=spec or backend.input_identity!=config.version.input_identity:
            raise ValueError('backend factory episode/input mismatch')
        if backend.reset_fingerprint().sha256!=layout.physical_fingerprint:
            raise ValueError('settled physical family differs from frozen split')
        session = _attach(backend,store,config,controller,source_role='bootstrap' if bootstrap else 'teacher',run_identity=run_identity)
        tap = session.tap.tap
        if config.version.kind=='wam':
            if not callable(getattr(controller,'set_goal',None)):
                raise TypeError('WAM controller must accept explicit set_goal(N)')
            controller.set_goal(spec.target_count)
        if bootstrap:
            # Explicit privileged bootstrap ONLY. This path never makes roots.
            session.observe(query=False)
            preparation = backend.prepare_bootstrap_oracle(on_observation=lambda raw,metadata:session.from_raw(raw))
        else:
            preparation = prepare_visually(backend,session.estimator,tap=session.tap,history=session.history)
        from ..reference import _state
        plan = backend.selected_preparation_plan()
        run_identity = replace(run_identity,plan_hash='no_plan' if plan is None else digest_json(_state(vars(plan))))
        backend.participants['collection_identity'].run_identity = run_identity
        record = replace(record,preparation=preparation,capture_ids=tuple(session.tap.capture_ids),run_identity=run_identity)
        _publish_main(record,store)
        if preparation.ready:
            raw = backend.advance_visual_clock()
            if not backend.done:
                safe = session.from_raw(raw,query=True)
                controller.reset(safe)
                controller.observe(safe)
            opportunity = 0
            while not backend.done:
                if not bootstrap and not session.history.batch().object_valid[0,-1].all():
                    record = replace(record,result=EpisodeResult('perception_invalid','current six-track visual state unavailable'))
                    backend.close()
                    break
                if not bootstrap and opportunity<root_quota+deferred_quota:
                    candidate = _root(backend,session,store,config.version,spec,config.proposal_hash,run_identity)
                    if opportunity<root_quota:
                        if _remaining(store.budget,config)<config.root_seconds_forecast:
                            record = replace(record,result=EpisodeResult('incomplete','phase_root_forecast_exhausted'),
                                deferred_records=(*record.deferred_records,store.put('roots',candidate)))
                            break  # valid root retained; never relabel budget shortage as task failure
                        programs = generate_programs(candidate.inputs,seed=spec.seed+opportunity,
                            target_count=spec.target_count,proposal=config.proposal)
                        roots.append(collect_root(candidate,programs,continuation_factory,backend,store))
                        record = replace(record,root_records=(*record.root_records,store.cursor(candidate.root_id)['record']))
                    else:
                        deferred.append(candidate)
                        record = replace(record,deferred_records=(*record.deferred_records,store.put('roots',candidate)))
                    opportunity += 1
                    _publish_main(record,store)
                reference = backend.reference_preview()
                program = CandidateProgram(controller.act(reference).knots)
                anchor = backend.codec.begin_program()
                for j in range(5):
                    if backend.done:
                        break
                    _execute(backend,backend.codec.sample(program,j*.04,anchor),trace)
                block = trace
                chunks.append(store.put('trajectories',block))
                trace = ExecutionTrace()
                if not backend.done:
                    safe = session.observe(query=True)
                    controller.observe(safe)
                    _synchronize(controller,backend,block.controls)
                record = replace(record,capture_ids=tuple(session.tap.capture_ids),chunk_ids=tuple(chunks))
                _publish_main(record,store)
        if record.result.status=='incomplete' and record.result.native_reason!='phase_root_forecast_exhausted':
            record = replace(record,result=episode_result(backend,preparation=preparation))
        if record.result.status=='infrastructure_error':
            raise RuntimeError(record.result.native_reason)
        if record.result.status in ('perception_invalid','model_invalid'):
            record = replace(record,deferred_records=())
    except BaseException as error:
        if trace.controls:
            _diagnostic(error,lambda:chunks.append(store.put('trajectories',trace)))
        captures = () if session is None else tuple(session.tap.capture_ids)
        record = replace(record,capture_ids=captures,chunk_ids=tuple(chunks),
                         result=EpisodeResult('infrastructure_error',str(error) or type(error).__name__))
        _diagnostic(error,lambda:_publish_main(record,store))
        _diagnostic(error,lambda:store.put('failures',dict(episode_id=episode_id,type=type(error).__name__,reason=str(error))))
        raise  # stop the batch, retain the ORIGINAL producer exception
    finally:
        def finalize():
            final_record = record
            if backend is not None:
                evidence = dict(origin=getattr(backend,'origin',None),native_reason=getattr(backend,'termination_reason',None),
                                native_result=episode_result(backend,preparation=preparation),
                                commands=None if not hasattr(backend,'_commands') else np.asarray(backend._commands),
                                frames=getattr(backend,'frames',None),
                                preparation_status=None if preparation is None else getattr(preparation,'status','native_bootstrap'),
                                backend_closed=getattr(backend,'closed',None))
                evidence_id = store.put('evidence',evidence)
                final_record = replace(final_record,native_evidence_id=evidence_id)
            return _publish_main(final_record,store)
        try:
            primary = sys.exc_info()[1]
            final_key = finalize() if primary is None else _diagnostic(primary,finalize)
        finally:
            _close_resources(tap,backend)
    # Closed visual failures are never revived as deferred opportunities.
    if record.result.status in ('perception_invalid','model_invalid','infrastructure_error'):
        deferred = []
    return final_key,roots,deferred


def collect_bootstrap(backend_factory, store, split, budget, config) -> CollectionSummary:
    _policy(split,budget,config)
    if config.version.kind!='act' or config.version.phase!=-1:
        raise ValueError('bootstrap requires frozen ACT version')
    main = []
    for layout in split.records:
        if layout.split!='bootstrap':
            continue
        key,_,_ = _main(layout,backend_factory,store,config,ACTReferenceController(config.version),bootstrap=True,
                       split_hash=split.manifest_hash)
        main.append(key)
    return CollectionSummary('perception_only',tuple(main),(),None,budget.poll().warnings)


def collect_phase(phase, controller, continuation_factory, backend_factory, store, split, budget, config) -> CollectionSummary:
    from ..config import WAMConfig
    from .datasets import coverage_report
    _policy(split,budget,config)
    if phase not in (0,1,2) or config.version.phase!=phase-1 or continuation_factory.version!=config.version:
        raise ValueError('phase0/1/2 uses exact ACT/WAM0/WAM1 continuation')
    if (not isinstance(controller,Controller) or getattr(controller,'version',None)!=config.version
            or not callable(getattr(controller,'synchronize',None))):
        raise TypeError('fully versioned main Controller plus synchronize required')
    data = WAMConfig().data
    layouts = [r for r in split.records if r.split=='train' and r.phase==phase]
    per_n = data.teacher_starts[phase]//6
    quotas = {n:0 for n in range(1,7)}
    mains,roots,deferred = [],[],[]
    for layout in layouts:
        if _remaining(budget,config)<config.root_seconds_forecast:
            break
        n = layout.target_count
        parents_left = sum(r.target_count==n for r in layouts[layouts.index(layout):])
        root_quota = (per_n-quotas[n]+parents_left-1)//parents_left
        deferred_quota = (config.extra_roots_per_n+sum(r.target_count==n for r in layouts)-1)//sum(r.target_count==n for r in layouts)
        key,collected,extra = _main(layout,backend_factory,store,config,controller,bootstrap=False,
                                   root_quota=root_quota,deferred_quota=deferred_quota,
                                   continuation_factory=continuation_factory,split_hash=split.manifest_hash)
        mains.append(key); roots.extend(collected); deferred.extend(extra)
        if store.get('trajectories',key).result.status=='incomplete':
            break
        quotas[n] += root_quota  # failed parents keep their preassigned quota; no replacement
    report = coverage_report(roots,perception_hash=config.version.perception_hash,phase=phase)
    extra_counts = {n:0 for n in range(1,7)}
    for root in deferred:
        row = next(row for row in report.rows if row.target_count==root.spec.target_count)
        if row.ready or extra_counts[row.target_count]>=config.extra_roots_per_n:
            continue
        if _remaining(budget,config)<config.root_seconds_forecast:
            break
        backend = backend_factory(root.spec)
        session = None
        try:
            layout = next(r for r in layouts if r.parent_id==root.spec.parent_id)
            if (backend.spec!=root.spec or backend.input_identity!=config.version.input_identity
                    or backend.reset_fingerprint().sha256!=layout.physical_fingerprint):
                raise ValueError('deferred root backend/frozen family mismatch')
            session = _attach(backend,store,config,controller,monitor=True)
            backend.restore(store.load_snapshot(root.snapshot_id),image_reader=store.read_image)
            programs = generate_programs(root.inputs,seed=root.spec.seed+extra_counts[row.target_count]+10000,
                                         target_count=root.spec.target_count,proposal=config.proposal)
            roots.append(collect_root(root,programs,continuation_factory,backend,store))
        finally:
            _close_resources(None if session is None else session.tap.tap,backend)
        extra_counts[row.target_count] += 1
        report = coverage_report(roots,perception_hash=config.version.perception_hash,phase=phase)
    ready = report.ready and len(roots)>=data.teacher_starts[phase] and len(mains)==data.main_trajectories[phase]
    return CollectionSummary('ready' if ready else 'not_ready',tuple(mains),tuple(roots),report,budget.poll().warnings)
