"""OFFLINE association/targets and causal versioned training views.

Mapping is supervision/audit ONLY. Slots are persistent HistoryBuffer slots,
never native IDs. Physical targets keep that root correspondence for all25
steps. Missing/ambiguous matches mask object losses, not terminal evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..truth import project_envelopes

ASSOCIATION_VERSION = 'root-center-extent-v1'


@dataclass(frozen=True)
class RootAssociation:
    slot_object_ids: tuple[int | None, ...]
    valid: np.ndarray
    perception_hash: str
    label_hash: str
    root_time: float
    version: str = ASSOCIATION_VERSION
    max_distance_m: float = .04
    ambiguity_margin_m: float = .005


def associate_root(history, truth, *, perception_hash, label_hash):
    """Hungarian root-only cost: center L2 + .5 * half-extent L2 (meters).

    Center distance<=40mm; both row AND column nearest alternative cost must
    differ by>5mm. Ties mask both competing objects. No ID equality, future
    frame or terminal outcome enters the cost or validity decision.
    """
    if not perception_hash or not label_hash:
        raise ValueError('full perception/label identities required')
    state = history.snapshot()
    if not state['observations'] or abs(state['observations'][-1].timestamp-truth.timestamp)>1e-7:
        raise ValueError('association requires root-time causal history and truth')
    latest = {b.track_id:b for b in state['observations'][-1].objects if b.valid}
    slots = [i for i,track in enumerate(state['track_ids']) if track in latest]
    estimates = [latest[state['track_ids'][i]] for i in slots]
    target = project_envelopes(truth)
    mapping, valid = [None]*6, np.zeros(6,bool)
    if estimates:
        distance = np.linalg.norm(np.asarray([b.center_xy for b in estimates])[:,None]-target.center_xy[None],axis=-1)
        cost = distance + .5*np.linalg.norm(np.asarray([b.half_extents_xy for b in estimates])[:,None]-target.half_extents_xy[None],axis=-1)
        rows,cols = linear_sum_assignment(cost)
        for row,col in zip(rows,cols):
            other_row = np.delete(cost[row],col)
            other_col = np.delete(cost[:,col],row)
            unambiguous = (not len(other_row) or other_row.min()-cost[row,col]>.005) and (
                           not len(other_col) or other_col.min()-cost[row,col]>.005)
            if target.object_valid[col] and distance[row,col]<=.04 and unambiguous:
                slot = slots[row]
                mapping[slot], valid[slot] = truth.raw.fields['object_ids'][col], True
    return RootAssociation(tuple(mapping),valid,perception_hash,label_hash,truth.timestamp)


@dataclass(frozen=True)
class AlignedTargets:
    object_mean: np.ndarray
    object_valid: np.ndarray
    velocity_valid: np.ndarray
    yaw_valid: np.ndarray
    angular_valid: np.ndarray
    state: np.ndarray
    state_valid: np.ndarray
    visibility: np.ndarray
    visibility_valid: np.ndarray
    rectangle: np.ndarray
    valid_steps: np.ndarray
    mapping_valid: np.ndarray


def align_future(association, root_truth, frames, *, labels=()):
    """Actual consecutive future prefix; no fabricated terminal padding.

    Root predecessor gives the first40ms velocity secant. Frame record order
    may vary; physical identity stays fixed by root slot_object_ids, not by
    independently rematching later geometry. Carry state uses restored100Hz
    label windows, supplied separately from sparse25Hz geometric targets.
    """
    frames, labels = tuple(frames), tuple(labels)
    if not 1<=len(frames)<=25 or len(labels) not in (0,len(frames)):
        raise ValueError('nonempty <=25 future prefix and matching labels required')
    if abs(association.root_time-root_truth.timestamp)>1e-7:
        raise ValueError('association/root clock mismatch')
    if any(abs(f.timestamp-root_truth.timestamp-(i+1)*.04)>1e-7 for i,f in enumerate(frames)):
        raise ValueError('actual consecutive25Hz future required')
    mean = np.zeros((25,6,9),np.float32)
    masks = [np.zeros((25,6),bool) for _ in range(7)]
    object_valid, velocity_valid, yaw_valid, angular_valid, state_valid, visibility_valid, rectangle = masks
    state, visibility = np.zeros((25,6),np.int64),np.zeros((25,6),np.float32)
    for j,frame in enumerate(frames):
        # project_envelopes expects stable predecessor order. Canonicalize the
        # predecessor to CURRENT record order without changing root mapping.
        previous = root_truth if j==0 else frames[j-1]
        if previous.raw.fields['object_ids'] != frame.raw.fields['object_ids']:
            from dataclasses import replace
            from ..contracts import supervision
            ids = previous.raw.fields['object_ids']
            order = [ids.index(i) for i in frame.raw.fields['object_ids']]
            fields = dict(previous.raw.fields)
            fields['objects'] = tuple(fields['objects'][i] for i in order)
            fields['object_ids'] = frame.raw.fields['object_ids']
            for name in ('native_collected','native_partial','native_valid','visibility','visibility_valid'):
                fields[name] = fields[name][order]
            previous = replace(previous,raw=supervision.RawTruth(fields))
        target = project_envelopes(frame,previous=previous)
        ids = frame.raw.fields['object_ids']
        for slot,identity in enumerate(association.slot_object_ids):
            if not association.valid[slot] or identity not in ids:
                continue
            i = ids.index(identity)
            mean[j,slot] = np.r_[target.center_xy[i],target.velocity_xy[i],target.half_extents_xy[i],
                                 target.yaw_double[i],target.angular_velocity[i]]
            object_valid[j,slot] = target.object_valid[i]
            velocity_valid[j,slot] = target.velocity_valid[i]
            yaw_valid[j,slot] = target.yaw_valid[i]
            angular_valid[j,slot] = target.angular_velocity_valid[i]
            visibility[j,slot],visibility_valid[j,slot] = target.visibility[i],target.visibility_valid[i]
            rectangle[j,slot] = target.rectangle[i]
            if labels:
                if abs(labels[j].timestamp-frame.timestamp)>1e-7:
                    raise ValueError('state label timestamp mismatch')
                state[j,slot],state_valid[j,slot] = labels[j].state[i],labels[j].state_valid[i]
    steps = np.arange(25)<len(frames)
    return AlignedTargets(mean,object_valid,velocity_valid,yaw_valid,angular_valid,state,state_valid,
                          visibility,visibility_valid,rectangle,steps,association.valid.copy())


@dataclass(frozen=True)
class PerceptionCache:
    perception_hash: str
    history: object
    audit: RootAssociation
    valid: bool
    capture_ids: tuple[str, ...]
    root_time: float

    @property
    def association(self):
        return self.audit


def with_current_perception(root, version, *, estimator_factory, store, tap):
    """Recompute current estimates AND offline root association from causal RGB.

    Never reuse an old mapping under a new perception hash. Factory loads its
    full frozen estimator checkpoint before the first estimate. Reject future
    captures and identity/config/calibration drift before publishing any cache.
    Caller holds the same admitted budget clock, including every replay forward.
    """
    from dataclasses import replace
    from ..inputs import digest_json
    from .collection import perception_identity, replay_perception
    captures = [store.get('captures',key) for key in root.capture_ids]
    if (not captures or any(c.timestamp>root.root_truth.timestamp+1e-7 for c in captures)
            or abs(captures[-1].timestamp-root.root_truth.timestamp)>1e-7):
        raise ValueError('current perception requires only causal captures ending at root')
    from ..snapshots import capture_random_state, restore_random_state
    rng = capture_random_state()
    try:
        estimator = estimator_factory(version,tap.capture_metadata)
        if perception_identity(estimator)!=version.perception_hash:
            raise ValueError('current perception checkpoint/dependency mismatch')
        history = replay_perception(root.capture_ids,store=store,tap=tap,estimator=estimator)
    finally:
        restore_random_state(rng)
    association = associate_root(history,root.root_truth,perception_hash=version.perception_hash,
                                  label_hash=root.association.label_hash)
    valid = bool(history.batch().object_valid[0,-1].sum()==6)
    cache = PerceptionCache(version.perception_hash,history.batch(),association,valid,
                            root.capture_ids,root.root_truth.timestamp)
    # Cache lineage includes raw-frame sequence and complete label identity.
    lineage = digest_json(dict(root_id=root.root_id,perception_hash=version.perception_hash,
                               label_hash=association.label_hash,capture_ids=root.capture_ids))
    key = store.put('perception',dict(lineage=lineage,cache=cache))
    store.publish_cursor('perception-'+lineage,dict(record=key))
    return replace(root,perception_caches={**root.perception_caches,version.perception_hash:cache})


def _cache(root, perception_hash):
    cache = root.perception_caches.get(perception_hash)
    if cache is None:
        return None
    if (cache.perception_hash!=perception_hash or cache.association.perception_hash!=perception_hash
            or cache.capture_ids!=root.capture_ids or abs(cache.root_time-root.root_truth.timestamp)>1e-7
            or cache.association.label_hash!=root.association.label_hash):
        raise ValueError('stale/mismatched current-perception cache lineage')
    return cache


def _inputs(root, cache):
    from .collection import SafeInputs
    return SafeInputs(cache.history,root.inputs.reference,root.inputs.codec_state)


def _terminal(branch):
    return (branch.terminal_result is not None and branch.continuation_version is not None
            and branch.selection is not None
            and branch.terminal_result.status in ('success','task_failure','safety_stop'))


def _roots(records):
    from ..splits import assert_family_isolation
    records = tuple(records)
    assert_family_isolation([r.spec for r in records])
    latest = {}
    for record in records:
        previous = latest.get(record.root_id)
        if previous is None:
            latest[record.root_id] = record
            continue
        if (previous.spec!=record.spec or previous.version!=record.version
                or previous.snapshot_id!=record.snapshot_id or previous.capture_ids!=record.capture_ids):
            raise ValueError('duplicate root identity has inconsistent physical/source lineage')
        complete = record if len(record.branches)>=len(previous.branches) else previous
        latest[record.root_id] = replace(complete,perception_caches={**previous.perception_caches,**record.perception_caches})
    return tuple(latest.values())


class PredictionDataset:
    """Physical data may merge known continuation versions; no goal N input."""
    def __init__(self, records, *, perception_hash):
        self.rows = []
        for root in _roots(records):
            cache = _cache(root,perception_hash)
            if root.source_role=='bootstrap' or cache is None:
                continue
            for branch in root.branches:
                if branch.trace.truths:
                    self.rows.append((root,branch,cache))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        root,branch,cache = self.rows[index]
        targets = align_future(cache.association,root.root_truth,branch.trace.truths,labels=branch.trace.labels)
        brush,brush_valid = np.zeros((25,9),np.float32),np.zeros(25,bool)
        for j,event in enumerate(branch.trace.controls[:25]):
            if len(event.times)!=4:
                break
            robot = event.robot
            brush[j] = np.r_[robot.tcp_xyz,np.sin(robot.yaw),np.cos(robot.yaw),robot.linear_velocity,robot.yaw_velocity]
            brush_valid[j] = True
        return dict(inputs=_inputs(root,cache),program=branch.program,supervision=targets,
                    brush=brush,brush_valid=brush_valid,
                    audit=dict(root_id=root.root_id,physical_version=root.version,branch=branch.index))


class TerminalDataset:
    """Quality head sees ONLY one explicitly requested complete continuation."""
    def __init__(self, records, *, perception_hash, continuation_hash):
        if not continuation_hash:
            raise ValueError('explicit continuation hash required')
        self.rows = []
        for root in _roots(records):
            cache = _cache(root,perception_hash)
            if root.source_role=='bootstrap' or cache is None or not cache.valid:
                continue
            for branch in root.branches:
                if _terminal(branch) and branch.continuation_version.continuation_hash==continuation_hash:
                    self.rows.append((root,branch,cache))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        root,branch,cache = self.rows[index]
        return dict(inputs=_inputs(root,cache),goal_n=root.spec.target_count,program=branch.program,
                    supervision=dict(success=branch.terminal_result.success,final_count=branch.terminal_count,
                                     count_valid=branch.terminal_count is not None),
                    audit=dict(root_id=root.root_id,continuation_version=branch.continuation_version,
                               native_reason=branch.terminal_result.native_reason,selection=branch.selection))


@dataclass(frozen=True)
class CoverageRow:
    phase: int
    target_count: int
    successful_roots: int
    parent_layouts: int

    @property
    def ready(self):
        return self.successful_roots>=20 and self.parent_layouts>=10


@dataclass(frozen=True)
class CoverageReport:
    rows: tuple[CoverageRow, ...]

    @property
    def ready(self):
        return bool(self.rows) and all(row.ready for row in self.rows)

    @property
    def status(self):
        return 'ready' if self.ready else 'not_ready'

    def require_ready(self):
        if not self.ready:
            raise ValueError('not_ready: each phase/N needs20 successful current-perception roots from10 parents')
        return self


def coverage_report(records, *, perception_hash=None, phase=None):
    records = _roots(records)
    if not records:
        return CoverageReport(() if phase is None else tuple(CoverageRow(phase,n,0,0) for n in range(1,7)))
    if perception_hash is None:
        versions = {key for root in records for key in root.perception_caches}
        if len(versions)!=1:
            raise ValueError('explicit current perception hash required for mixed/empty caches')
        perception_hash = versions.pop()
    phases = (phase,) if phase is not None else sorted({r.spec.phase for r in records if r.source_role=='teacher' and r.spec.phase>=0})
    rows = []
    for p in phases:
        for n in range(1,7):
            roots,parents = set(),set()
            for root in records:
                cache = _cache(root,perception_hash)
                if (root.source_role!='teacher' or root.spec.phase!=p or root.spec.target_count!=n
                        or cache is None or not cache.valid):
                    continue
                if any(_terminal(b) and b.terminal_result.success for b in root.branches):
                    roots.add(root.root_id)
                    parents.add(root.spec.parent_id)
            rows.append(CoverageRow(p,n,len(roots),len(parents)))
    return CoverageReport(tuple(rows))


class TeacherDataset:
    """Successful action programs with version/evidence; positive training gate.

    len()/coverage are available for collection planning. __getitem__ refuses
    positive training until ALL phase/N gates hold; consumers cannot quietly
    lower the20-root/10-parent rule. Uncontinued/all-fail/bootstrap never enter.
    """
    def __init__(self, records, *, perception_hash, phase=None):
        records = _roots(records)
        self.coverage = coverage_report(records,perception_hash=perception_hash,phase=phase)
        self.rows = []
        for root in records:
            cache = _cache(root,perception_hash)
            if root.source_role!='teacher' or cache is None or not cache.valid:
                continue
            for branch in root.branches:
                if _terminal(branch) and branch.terminal_result.success:
                    self.rows.append((root,branch,cache))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.coverage.require_ready()
        root,branch,cache = self.rows[index]
        return dict(inputs=_inputs(root,cache),goal_n=root.spec.target_count,program=branch.program,
                    audit=dict(root_id=root.root_id,success=True,continuation_version=branch.continuation_version,
                               native_reason=branch.terminal_result.native_reason))
