"""One-shot all-visual preparation using the original native control prefix.

Caller supplies the SAME already-versioned tap/estimator/history used later.
No policy query, second planning call, oracle recovery or truth repair occurs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from .contracts import ObjectBelief, OnlineObservation
from .native import PreparationResult as NativePreparationResult
from .planning import PerceivedPlanningScene, PlanningGeometry, monitor_plan


class _InvalidVisualMeasurement(ValueError):
    """Only returned visual values failed a local, known validation leaf."""


class _PreparationContractError(BaseException):
    """Carry a producer fault past frozen prefix diagnostic Exception catches."""
    def __init__(self, error):
        self.error = error


def _visual_beliefs(beliefs):
    if not isinstance(beliefs, (tuple, list)) or any(type(b) is not ObjectBelief for b in beliefs):
        raise TypeError('estimator must return visual ObjectBeliefs')
    try:
        return tuple(replace(b) for b in beliefs)
    except (ValueError, TypeError) as error:
        raise _InvalidVisualMeasurement(str(error)) from error


@dataclass(frozen=True)
class PreparationResult:
    """Native fields retained verbatim, with explicit visual outcome evidence.

    ``paused`` reports the native prefix flag, including on visual failure.
    Only status='paused' is resumable; perception_invalid closes the backend.
    termination_reason is the original native reason, even when empty.
    Startup native termination maps to native_failure without processing its
    final capture. actions then has no preparation commands; native startup
    hold/state evidence remains owned by the terminated backend.
    """
    ready: bool
    paused: bool
    actions: np.ndarray
    termination_reason: str
    status: str
    visual_reason: str = ''


class VisualPreparationMonitor:
    """Restorable causal state; dependencies are bound in place, never copied.

    Snapshot at completed capture/native cutpoints, not during the external
    features->estimate->history transaction. Native prefix replay restores
    these providers at the saved cut; it does not emit past monitor callbacks.
    """
    def __init__(self, backend, estimator, tap, history):
        self.backend, self.estimator, self.tap, self.history = backend, estimator, tap, history
        self.geometry = PlanningGeometry.from_config(backend.cfg)
        self.track_ids = ()
        self.startup_ids = None
        self.startup_captures = 0
        self.timestamp = None
        self.scene = None
        self.visual_reason = ''

    def snapshot(self):
        return copy.deepcopy(dict(schema=1, geometry=self.geometry, track_ids=self.track_ids,
                                  startup_ids=self.startup_ids, startup_captures=self.startup_captures,
                                  timestamp=self.timestamp, scene=self.scene, visual_reason=self.visual_reason))

    def restore(self, state):
        if (set(state) != set(self.snapshot()) or state['schema'] != 1 or state['geometry'] != self.geometry
                or not 0 <= state['startup_captures'] <= 3
                or len(set(state['track_ids'])) != len(state['track_ids'])
                or len(state['track_ids']) not in (0,6)):
            raise ValueError('invalid visual preparation snapshot')
        for key, value in copy.deepcopy(state).items():
            if key not in ('schema', 'geometry'):
                setattr(self, key, value)

    def capture(self, raw):
        if self.visual_reason:
            raise RuntimeError('visual attempt is terminal')
        if self.timestamp is not None and abs(raw.timestamp-self.timestamp-.2) > 1e-7:
            raise ValueError('preparation needs distinct consecutive 5 Hz captures')
        packet = self.tap.features_only(raw)
        beliefs = _visual_beliefs(self.estimator.estimate(packet))
        safe = OnlineObservation(raw.timestamp, beliefs, raw.robot, raw.images)
        self.history.append(safe)
        self.timestamp = safe.timestamp
        return safe

    def perceived_scene(self, safe):
        # Only this estimated-geometry leaf is an expected visual rejection.
        # Tap/estimator/calibration/OnlineObservation/history faults propagate.
        try:
            return PerceivedPlanningScene(safe.objects, geometry=self.geometry, robot_yaw=safe.robot.yaw)
        except ValueError as error:
            raise _InvalidVisualMeasurement(str(error)) from error

    def initial(self, raw):
        safe = self.capture(raw)
        self.startup_captures += 1
        ids = tuple(sorted(b.track_id for b in safe.objects))
        if self.startup_ids is None:
            self.startup_ids = ids
        valid_measurements = (len(ids) == 6 and ids == self.startup_ids
                              and all(b.visibility >= .5 and np.linalg.eigvalsh(b.position_covariance).max() <= .012**2
                                      for b in safe.objects))
        if not valid_measurements:
            self.visual_reason = 'startup requires three visible six-track captures with <=12mm uncertainty'
            return
        if self.startup_captures == 3:
            self.scene = self.perceived_scene(safe)
            self.track_ids = self.scene.track_ids

    def observe(self, raw, metadata=None, *, on_observation=None):
        try:
            safe = self.capture(raw)
            scene = self.perceived_scene(safe)
            if scene.track_ids != self.track_ids:
                self.visual_reason = 'visual track association changed after initial planning'
                return False
            validity = monitor_plan(scene, self.backend.selected_preparation_plan())
            if not validity.valid:
                self.visual_reason = validity.reason
                return False
            self.scene = scene
            return on_observation(safe) if on_observation is not None else True
        except _InvalidVisualMeasurement as error:
            self.visual_reason = str(error)
            return False
        except Exception as error:
            raise _PreparationContractError(error) from error


def _bind(backend, name, provider):
    if name not in backend.participants:
        backend.bind_provider(name, provider, beliefs=False)
    elif backend.participants[name] is not provider:
        raise ValueError('preparation must share the bound ' + name + ' provider')


def prepare_visually(backend, estimator, *, tap, history, on_observation=None, resume=False) -> PreparationResult:
    """Capture at t,t+.2,t+.4, plan ONCE inside native.prepare, monitor at 5Hz.

    Optional on_observation(safe) returning False deliberately pauses at a
    completed monitored native capture; resume=True continues that saved prefix
    with the same providers. Visual invalidity is terminal and never resumes.
    Reconstruct named providers before new-process restore; callable hooks are
    caller-owned and must be supplied again. No two-second history wait.
    """
    if backend.closed:
        raise RuntimeError('backend is closed')
    for name, provider in (('features',tap), ('estimator',estimator), ('history',history)):
        _bind(backend, name, provider)
    monitor = backend.participants.get('visual_preparation')
    if resume:
        if monitor is None or monitor.visual_reason:
            raise RuntimeError('no resumable visual preparation')
    else:
        if monitor is not None:
            raise RuntimeError('one visual preparation per attempt')
        monitor = VisualPreparationMonitor(backend, estimator, tap, history)
        backend.bind_provider('visual_preparation', monitor, beliefs=False)
    native = None
    try:
        if not resume:
            for index in range(3):
                raw = backend.observe() if index == 0 else backend.advance_visual_clock()
                if backend.done:
                    native = NativePreparationResult(False, False, np.empty((0,4)), backend.termination_reason)
                    break
                try:
                    monitor.initial(raw)
                except _InvalidVisualMeasurement as error:
                    monitor.visual_reason = str(error)
                if monitor.visual_reason:
                    break
            if native is None and not monitor.visual_reason:
                native = backend.prepare(monitor.scene, on_observation=lambda raw, meta: monitor.observe(
                    raw, meta, on_observation=on_observation))
        else:
            native = backend.resume_preparation(on_observation=lambda raw, meta: monitor.observe(
                raw, meta, on_observation=on_observation))
    except _PreparationContractError as fault:
        backend.close()
        raise fault.error.with_traceback(fault.error.__traceback__) from None
    except BaseException:
        backend.close()
        raise
    if native is None:
        native = NativePreparationResult(False, False, np.empty((0,4)), '')
    if monitor.visual_reason:
        backend.close()
        return PreparationResult(False, native.paused, native.actions, native.termination_reason,
                                 'perception_invalid', monitor.visual_reason)
    status = 'ready' if native.ready else 'paused' if native.paused else 'native_failure'
    return PreparationResult(native.ready, native.paused, native.actions, native.termination_reason, status)
