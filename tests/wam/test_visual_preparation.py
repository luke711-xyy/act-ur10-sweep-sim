"""Task7 geometry boundaries; deterministic measurements are NOT trained vision."""
import copy
from dataclasses import replace
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wam_sweep.contracts import CAMERA_VIEWS, ObjectBelief, OnlineObservation, RobotContext
from wam_sweep.history import HistoryBuffer
from wam_sweep.native import FrozenSweepBackend, PreparationResult as NativeResult
from wam_sweep.vision.tracking import CausalTracker, VisualDetection


CENTERS = ((-.05, -.14), (-.08, -.02), (.04, .08), (-.15, .18), (.12, .23), (-.12, -.27))


def api():
    assert importlib.util.find_spec('wam_sweep.planning') is not None, 'Task7 perceived planning missing'
    assert importlib.util.find_spec('wam_sweep.preparation') is not None, 'Task7 visual preparation missing'
    return importlib.import_module('wam_sweep.planning'), importlib.import_module('wam_sweep.preparation')


def beliefs():
    return tuple(ObjectBelief(xy, [0., 0.], [.008, .008], 0., 0., [1., 0.], 1.,
                             [1., 0., 0.], np.eye(2)*1e-6, np.eye(6, 16)[i], i, True)
                 for i, xy in enumerate(CENTERS))


def config():
    values = {'planner.expert_curve_capture_margin': .04, 'planner.expert_capture_forward_cos': 0.,
              'task.target_count': 1, 'task.expert_landing_clearance': .03}
    return SimpleNamespace(workspace=SimpleNamespace(x_min=-.5, x_max=.5, y_min=-.4, y_max=.4),
                           table=SimpleNamespace(top_z=0.),
                           target=SimpleNamespace(x_min=-.42, x_max=-.28, y_min=-.18, y_max=.18),
                           end_effector=SimpleNamespace(brush_width=.14, brush_depth=.008),
                           get_path=lambda key, default=None: values.get(key, default))


def plan():
    return SimpleNamespace(feasible=True, target_indices=np.array([0]), yaw_offset=0.,
                           waypoints=np.array([[.42, 0.], [.25, -.14], [-.37, -.14]]),
                           landing_region={'part_x_max': .131, 'clearance': .03,
                                           'brush_x_half_extent': .004, 'yaw': 0., 'x_min': .165})


def scene(items=None):
    planning, _ = api()
    return planning.PerceivedPlanningScene(items or beliefs(), cfg=config(), robot_yaw=0.)


def test_estimated_geometry_is_copied_conservative_and_has_no_truth_fallback():
    items = list(beliefs())
    items[0] = replace(items[0], half_extents_xy=[.03, .04], shape_probabilities=[0., 1.],
                       position_covariance=[[9e-6, 0.], [0., 9e-6]])
    perceived = scene(tuple(items))
    assert perceived.layout[0]['nominal_radius'] == pytest.approx(.059)  # .05 + 3*.003
    assert perceived.component_collision_x_max() == pytest.approx(.131)
    items[0].center_xy[:] = 999
    perceived.component_positions()[:] = 999
    perceived.layout[0]['nominal_radius'] = 999
    assert perceived.component_positions()[0, 0] == -.05
    assert perceived.layout[0]['nominal_radius'] == pytest.approx(.059)
    for forbidden in ('env', 'data', 'model', 'collected_mask', 'component_quats', 'capture_truth'):
        assert not hasattr(perceived, forbidden)


@pytest.mark.parametrize('change', ['missing', 'invalid', 'correlated_uncertainty', 'mutated_contour'])
def test_scene_rejects_unknown_or_invalid_geometry(change):
    items = list(beliefs())
    if change == 'missing':
        items.pop()
    elif change == 'invalid':
        items[0] = replace(items[0], valid=False)
    elif change == 'correlated_uncertainty':
        items[0] = replace(items[0], position_covariance=[[.0004, .0003], [.0003, .0004]])
    else:
        items[0].half_extents_xy[:] = np.nan
    with pytest.raises(ValueError):
        scene(tuple(items))


def test_monitor_checks_exact_selected_path_without_planning():
    planning, _ = api()
    assert planning.monitor_plan(scene(), plan()).valid
    moved = list(beliefs())
    moved[1] = replace(moved[1], center_xy=[-.12, -.14])
    assert not planning.monitor_plan(scene(tuple(moved)), plan()).valid
    moved = list(beliefs())
    moved[0] = replace(moved[0], center_xy=[-.05, -.32])
    assert not planning.monitor_plan(scene(tuple(moved)), plan()).valid
    enlarged = list(beliefs())
    enlarged[1] = replace(enlarged[1], half_extents_xy=[.1, .1], shape_probabilities=[0., 1.])
    assert not planning.monitor_plan(scene(tuple(enlarged)), plan()).valid
    right = list(beliefs())
    right[4] = replace(right[4], center_xy=[.3, .23])
    assert not planning.monitor_plan(scene(tuple(right)), plan()).valid


def test_selected_plan_getter_cannot_mutate_native_plan():
    backend = FrozenSweepBackend()
    backend._plan = None
    assert hasattr(backend, 'selected_preparation_plan'), 'authorized read-only chosen-plan accessor missing'
    assert backend.selected_preparation_plan() is None
    backend._plan = plan()
    returned = backend.selected_preparation_plan()
    returned.waypoints[:] = 999
    returned.target_indices[:] = 5
    returned.landing_region['x_min'] = 999
    assert backend._plan.waypoints[1, 0] == .25
    assert backend._plan.target_indices.tolist() == [0]
    assert backend._plan.landing_region['x_min'] == .165


class FixtureTap:
    def __init__(self):
        self.timestamps = []

    def features_only(self, raw):
        assert not self.timestamps or raw.timestamp > self.timestamps[-1]
        self.timestamps.append(raw.timestamp)
        return SimpleNamespace(timestamp=raw.timestamp)

    def snapshot(self):
        return list(self.timestamps)

    def restore(self, state):
        self.timestamps = list(state)


class MeasurementFixture:
    """Explicit synthetic measurement source, real causal tracker; no native handle."""
    def __init__(self, failure=None):
        self.tracker = CausalTracker()
        self.failure = failure
        self.timestamps = []

    def estimate(self, packet):
        self.timestamps.append(packet.timestamp)
        detections = [VisualDetection(b.center_xy, b.half_extents_xy, b.yaw, 0., b.shape_probabilities,
                                      1., b.state_probabilities, b.position_covariance, b.appearance)
                      for b in beliefs()]
        result = self.tracker.update(packet.timestamp, detections)
        if len(self.timestamps) >= 4 and self.failure:
            changed = list(result)
            changed[1] = replace(changed[1], **self.failure)
            return tuple(changed)
        return result

    def snapshot(self):
        return dict(tracker=self.tracker.snapshot(), timestamps=list(self.timestamps))

    def restore(self, state):
        self.tracker.restore(state['tracker'])
        self.timestamps = list(state['timestamps'])


class PoisonBackend:
    """Small native PAUSE boundary double; real native coverage is below."""
    def __init__(self):
        self.cfg, self.participants = config(), {}
        self.now, self.origin = 10.4, 10.4
        self.closed, self.done = False, False
        self.termination_reason = ''
        self.planning_calls, self.actions = 0, []
        self.chosen = None

    @property
    def env(self):
        raise AssertionError('online access to native truth')

    def bind_provider(self, name, provider, *, beliefs=False):
        assert not beliefs and name not in self.participants
        self.participants[name] = provider

    def observe(self):
        assert not self.closed
        robot = RobotContext([.42, 0., .1], 0., np.zeros(3), 0., 'prepare', np.zeros(3),
                             np.zeros(3), self.now-self.origin, False, False, 0., False)
        return OnlineObservation(self.now, (), robot, {v: np.zeros((2, 2, 3), np.uint8) for v in CAMERA_VIEWS})

    def advance_visual_clock(self):
        assert not self.closed
        self.now += .2
        return self.observe()

    def prepare(self, perceived, *, on_observation):
        assert not self.closed
        self.planning_calls += 1
        assert self.planning_calls == 1
        assert self.participants['features'].timestamps == pytest.approx([10.4, 10.6, 10.8])
        assert all(b.valid for b in self.participants['history'].snapshot()['observations'][-1].objects)
        np.testing.assert_allclose(perceived.component_positions()[:, :2], CENTERS)
        self.chosen = plan()
        return self.resume_preparation(on_observation=on_observation)

    def selected_preparation_plan(self):
        return copy.deepcopy(self.chosen)

    def resume_preparation(self, *, on_observation=None):
        assert not self.closed
        for _ in range(2):
            self.actions.append([1., 2., 3., 4.])
            raw = self.advance_visual_clock()
            if on_observation and on_observation(raw, None) is False:
                return NativeResult(False, True, np.array(self.actions), '')
        return NativeResult(True, False, np.array(self.actions), '')

    def step_target(self, target):
        assert not self.closed
        self.actions.append(target)

    def close(self):
        self.closed = True


def test_three_distinct_captures_six_confirmed_tracks_and_same_providers():
    _, preparation = api()
    backend, estimator, tap, history = PoisonBackend(), MeasurementFixture(), FixtureTap(), HistoryBuffer()
    result = preparation.prepare_visually(backend, estimator, tap=tap, history=history)
    assert result.status == 'ready' and result.ready and not result.paused
    assert backend.planning_calls == 1 and not backend.closed
    assert estimator.timestamps == tap.timestamps == pytest.approx([10.4, 10.6, 10.8, 11., 11.2])
    assert history.batch().history_valid.sum() == 5  # no two-second warmup
    assert backend.participants['estimator'] is estimator and backend.participants['history'] is history
    assert backend.participants['features'] is tap
    assert all(not x.robot.visual_entry_valid for x in history.snapshot()['observations'])


@pytest.mark.parametrize('failure', [{'center_xy': [-.12, -.14]}, {'valid': False},
                                    {'half_extents_xy': [.1, .1], 'shape_probabilities': [0., 1.]}])
def test_invalid_path_never_replans_or_uses_truth(failure):
    _, preparation = api()
    backend, tap, history = PoisonBackend(), FixtureTap(), HistoryBuffer()
    result = preparation.prepare_visually(backend, MeasurementFixture(failure), tap=tap, history=history)
    assert backend.planning_calls == 1
    assert result.status == 'perception_invalid' and not result.ready
    assert result.paused and result.termination_reason == ''  # faithful native PAUSE evidence
    assert result.visual_reason and backend.closed
    before = copy.deepcopy(backend.actions)
    for action in (lambda: backend.resume_preparation(), lambda: backend.step_target([9.]*4),
                   lambda: preparation.prepare_visually(backend, MeasurementFixture(), tap=tap, history=history)):
        with pytest.raises((AssertionError, RuntimeError)):
            action()
    assert backend.actions == before and len(before) == 1


def test_explicit_checkpoint_pause_is_resumable_and_monitor_snapshot_is_owned():
    _, preparation = api()
    backend, estimator, tap, history = PoisonBackend(), MeasurementFixture(), FixtureTap(), HistoryBuffer()
    result = preparation.prepare_visually(backend, estimator, tap=tap, history=history,
                                          on_observation=lambda safe: False)
    assert result.status == 'paused' and result.paused and not backend.closed
    monitor = backend.participants['visual_preparation']
    saved = monitor.snapshot()
    saved['track_ids'] = (999,)
    assert monitor.snapshot()['track_ids'] == tuple(range(6))
    resumed = preparation.prepare_visually(backend, estimator, tap=tap, history=history, resume=True)
    assert resumed.status == 'ready' and backend.planning_calls == 1


def test_invalid_startup_never_plans_or_retries():
    _, preparation = api()
    estimator = MeasurementFixture()
    original = estimator.estimate
    estimator.estimate = lambda packet: tuple(replace(b, valid=False) for b in original(packet))
    backend = PoisonBackend()
    result = preparation.prepare_visually(backend, estimator, tap=FixtureTap(), history=HistoryBuffer())
    assert result.status == 'perception_invalid' and backend.closed
    assert backend.planning_calls == 0 and not backend.actions


def test_native_failure_fields_and_interface_errors_are_not_visual_failures():
    _, preparation = api()
    backend = PoisonBackend()
    original = backend.prepare
    def no_handoff(perceived, *, on_observation):
        original(perceived, on_observation=on_observation)
        return NativeResult(False, False, np.ones((3,4)), 'native acquisition rejected')
    backend.prepare = no_handoff
    result = preparation.prepare_visually(backend, MeasurementFixture(), tap=FixtureTap(), history=HistoryBuffer())
    assert result.status == 'native_failure' and not result.ready and not result.paused
    assert result.termination_reason == 'native acquisition rejected' and not result.visual_reason
    np.testing.assert_array_equal(result.actions, np.ones((3,4)))
    backend = PoisonBackend()
    def broken_interface(*args, **kwargs):
        raise ValueError('source interface/config ambiguity')
    backend.prepare = broken_interface
    with pytest.raises(ValueError, match='source interface/config ambiguity'):
        preparation.prepare_visually(backend, MeasurementFixture(), tap=FixtureTap(), history=HistoryBuffer())
    assert backend.closed


def test_duplicate_capture_and_provider_substitution_cannot_continue():
    _, preparation = api()
    backend, tap = PoisonBackend(), FixtureTap()
    backend.advance_visual_clock = backend.observe
    with pytest.raises(ValueError, match='distinct consecutive 5 Hz captures'):
        preparation.prepare_visually(backend, MeasurementFixture(), tap=tap, history=HistoryBuffer())
    assert backend.closed
    assert backend.planning_calls == 0 and tap.timestamps == [10.4]
    backend = PoisonBackend()
    backend.bind_provider('features', FixtureTap())
    with pytest.raises(ValueError, match='share'):
        preparation.prepare_visually(backend, MeasurementFixture(), tap=tap, history=HistoryBuffer())
    assert backend.planning_calls == 0


@pytest.mark.parametrize('terminal_capture,hold_delta', [(1,.2),(2,.07),(2,.2),(3,.2)],
                         ids=['initial','between-captures','exact-boundary','third-capture'])
@pytest.mark.parametrize('reason', ['temporary hybrid safety: force ceiling or nonfinite force',
                                   'target count was not collected',
                                   '45-second shared episode timeout',
                                   '1200-frame shared episode timeout'])
def test_startup_native_termination_stops_before_more_visual_work(terminal_capture, hold_delta, reason):
    _, preparation = api()
    backend, tap, estimator, history = PoisonBackend(), FixtureTap(), MeasurementFixture(), HistoryBuffer()
    observe = backend.observe
    captures, holds = [], []
    def terminating_capture():
        captures.append(backend.now)
        if len(captures) == terminal_capture:
            backend.done, backend.termination_reason = True, reason
        raw = observe()
        return replace(raw, robot=replace(raw.robot, phase='terminated')) if backend.done else raw
    def terminal_hold():
        assert not backend.done, 'hold after native termination'
        holds.append(backend.now)
        backend.actions.append([.42, 0., .1, 0.])  # native hold evidence, not prepare-result actions
        backend.now += hold_delta
        return terminating_capture()
    backend.observe, backend.advance_visual_clock = terminating_capture, terminal_hold
    result = preparation.prepare_visually(backend, estimator, tap=tap, history=history)
    assert result.status == 'native_failure' and result.termination_reason == reason
    assert not result.ready and not result.paused and not result.visual_reason
    assert backend.done and backend.termination_reason == reason
    assert len(captures) == terminal_capture and len(holds) == terminal_capture-1
    assert len(backend.actions) == len(holds)  # original native evidence remains available
    assert result.actions.shape == (0,4)  # no native.prepare call/result ever occurred
    assert len(tap.timestamps) == len(estimator.timestamps) == terminal_capture-1
    assert len(history.snapshot()['observations']) == terminal_capture-1
    assert backend.planning_calls == 0


@pytest.mark.parametrize('stage', ['startup','monitor'])
@pytest.mark.parametrize('fault', ['identity','pixels','calibration','camera_order','history','tap_type'])
def test_producer_contract_errors_close_and_propagate(stage, fault):
    _, preparation = api()
    backend, tap, estimator, history = PoisonBackend(), FixtureTap(), MeasurementFixture(), HistoryBuffer()
    trigger = 1 if stage == 'startup' else 4
    feature_call, estimate_call, append_call = tap.features_only, estimator.estimate, history.append
    calls = dict(tap=0, estimator=0, history=0)
    if fault == 'identity':
        import torch
        from wam_sweep.contracts import InputIdentity
        from wam_sweep.vision import BEVEncoder, SixQueryDetector, VisualStateEstimator
        from wam_sweep.vision.features import ActFeaturePacket, LAYERS
        identity = InputIdentity('fixture','manifest','model',
                                 {k:'hash' for k in ('perception','control','preprocessing','config')})
        # Real producer method and fully typed packet; fails identity before any
        # neural forward. This is NOT a substitute trained perception fixture.
        real_estimator = VisualStateEstimator(BEVEncoder(), SixQueryDetector(), identity, lambda t: None)
        layers = {name:torch.zeros((1,ch,side,side)) for name,(ch,side) in LAYERS.items()}
        bad_packet = ActFeaturePacket(0.,{view:layers for view in CAMERA_VIEWS},'wrong-manifest','model','hash')
    def features(raw):
        calls['tap'] += 1
        if calls['tap'] == trigger:
            if fault == 'pixels':
                from wam_sweep.vision import FrozenFeatureTap
                # Exercise the real live pixel validator against a mismatched
                # private RGB channel, without binding or running another model.
                private = {'observation.images.'+v:np.ones((3,2,2),np.float32) for v in CAMERA_VIEWS}
                owner = SimpleNamespace(_backend=SimpleNamespace(native_act_input=lambda t:private))
                FrozenFeatureTap._validate_capture_pixels(owner, raw)
            if fault == 'tap_type':
                raise TypeError('controlled tap provider contract fault')
        return feature_call(raw)
    def estimate(packet):
        calls['estimator'] += 1
        if calls['estimator'] == trigger:
            if fault == 'identity':
                return real_estimator.estimate(replace(bad_packet,timestamp=packet.timestamp))
            if fault in ('calibration','camera_order'):
                from wam_sweep.native import CaptureMetadata
                from wam_sweep.vision.features import validate_capture_metadata
                timestamp = packet.timestamp+.1 if fault == 'calibration' else packet.timestamp
                validate_capture_metadata(CaptureMetadata(timestamp,{},'fixture'),packet.timestamp,'fixture')
        return estimate_call(packet)
    def append(safe):
        calls['history'] += 1
        if fault == 'history' and calls['history'] == trigger:
            append_call(safe)
        append_call(safe)  # real HistoryBuffer rejects the controlled duplicate slot
    tap.features_only, estimator.estimate, history.append = features, estimate, append
    messages = dict(identity='estimator feature/input dependency identity mismatch',
                    pixels='feature pixels differ from capture',
                    calibration='calibration schema/timestamp does not match historical pixels',
                    camera_order='calibration camera order/types differ from feature order',
                    history='observation clock ticks must strictly increase',
                    tap_type='controlled tap provider contract fault')
    error_type = TypeError if fault == 'tap_type' else ValueError
    with pytest.raises(error_type, match=messages[fault]):
        preparation.prepare_visually(backend, estimator, tap=tap, history=history)
    assert backend.closed
    assert backend.planning_calls == (0 if stage == 'startup' else 1)
    assert len(backend.actions) == (0 if stage == 'startup' else 1)
    assert not backend.participants['visual_preparation'].visual_reason
    assert calls['tap'] == trigger and calls['estimator'] <= trigger and calls['history'] <= trigger


@pytest.mark.parametrize('stage', ['startup','monitor'])
@pytest.mark.parametrize('field,value', [('half_extents_xy',np.array([np.nan,.01])),
                                       ('position_covariance',np.array([[1.,2.],[2.,1.]]))])
def test_returned_invalid_visual_payload_remains_terminal(stage, field, value):
    _, preparation = api()
    backend, estimator = PoisonBackend(), MeasurementFixture()
    estimate = estimator.estimate
    trigger, calls = (1 if stage == 'startup' else 4), []
    def invalid_visual(packet):
        calls.append(packet.timestamp)
        items = estimate(packet)
        if len(calls) == trigger:
            getattr(items[0],field)[:] = value  # returned visual values fail leaf revalidation
        return items
    estimator.estimate = invalid_visual
    result = preparation.prepare_visually(backend, estimator, tap=FixtureTap(), history=HistoryBuffer())
    assert result.status == 'perception_invalid' and result.visual_reason and backend.closed
    assert backend.planning_calls == (0 if stage == 'startup' else 1)
    assert len(calls) == trigger


@pytest.mark.parametrize('invalidate', [False, True, 'producer-identity'],
                         ids=['ready-and-replay', 'terminal-visual-failure', 'producer-identity'])
def test_real_frozen_planner_and_native_preparation_use_conflicting_estimates(invalidate):
    api()  # RED is feature absence, not an incidental subprocess import error.
    body = 'invalidate = ' + repr(invalidate) + '\n' + r'''
import copy, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from wam_sweep.artifacts import BudgetLedger, atomic_checkpoint
from wam_sweep.config import WAMConfig
from wam_sweep.contracts import EpisodeSpec
from wam_sweep.history import HistoryBuffer
from wam_sweep.inputs import verify_inputs, assert_native_origin
from wam_sweep.native import FrozenSweepBackend
from wam_sweep.preparation import prepare_visually
from wam_sweep.vision import FrozenFeatureTap, BEVEncoder, SixQueryDetector, VisualStateEstimator
from wam_sweep.vision.detector import DetectionOutput

root = Path.cwd()
account = root/'.venv/task-2-budget.json'
budget = BudgetLedger(account, owned_roots=json.loads(account.read_bytes())['policy']['roots'],
                      config=WAMConfig.load(root/'configs/wam_visual_v1.yaml').budget)
identity = verify_inputs('/Users/huazhi_luke/.local/share/wam-sweep-v1/inputs/r8', allow_unlisted=False)
backend = FrozenSweepBackend.open(identity, EpisodeSpec('task7-bounded-fixture',2120018,1,'train',-1,'act',0),
                                 budget=budget, native_lock=root/'.venv/task-5-native.lock')
tap = None
try:
    # Never copied from native objects. Deliberately false fixed measurement field.
    centers = np.array([[-.05,-.14],[-.08,-.02],[.04,.08],[-.15,.18],[.12,.23],[-.12,-.27]])
    truth = backend.env.component_positions().copy()  # assertion/audit ONLY
    assert not np.allclose(truth[:,:2], centers)
    native_edge = backend.env.component_collision_x_max()
    native_radii = [p['nominal_radius'] for p in backend.env.layout]
    import torchvision.models._api as vision_api
    def forbidden(*args, **kwargs): raise AssertionError('forbidden external download/oracle/repair')
    vision_api.load_state_dict_from_url = forbidden
    backend.prepare_bootstrap_oracle = forbidden
    tap = FrozenFeatureTap(backend)
    counts = {'act':0, 'backbone':0}
    binding = backend.bind_frozen_policy()
    hooks = [binding.policy.model.register_forward_hook(lambda *args: counts.__setitem__('act',counts['act']+1)),
             binding.policy.model.backbone.register_forward_hook(lambda *args: counts.__setitem__('backbone',counts['backbone']+1))]
    bev, detector = BEVEncoder().to('mps'), SixQueryDetector().to('mps')
    # Test-only detector output injection. The real shared frozen RGB/tap, BEV,
    # estimator, tracker, history, planner, physics/force/Z remain in the chain.
    # No trained or calibrated accuracy is claimed for this measurement fixture.
    measurement_times = []
    def measurement(bev_output):
        measurement_times.append(bev_output.timestamp)
        def t(a): return torch.tensor(a,dtype=torch.float32,device='mps')
        extent = .16 if invalidate and len(measurement_times)>=4 else .008
        return DetectionOutput(bev_output.timestamp,t(centers[None]),t(np.zeros((1,6,2))),
            t(np.full((1,6,2),extent)),t(np.tile([0.,1.],(1,6,1))),t(np.zeros((1,6,1))),
            t(np.tile([100.,-100.],(1,6,1))),t(np.full((1,6,1),20.)),
            t(np.tile([20.,-20.,-20.],(1,6,1))),t(np.tile(np.eye(2)*1e-6,(1,6,1,1))),
            t(np.eye(6,16)[None]))
    detector.forward = measurement
    estimator = VisualStateEstimator(bev,detector,identity,tap.capture_metadata)
    history = HistoryBuffer()
    for name, provider in [('features',tap),('estimator',estimator),('history',history)]:
        backend.bind_provider(name,provider,beliefs=False)
    import sim.act.expert as expert
    real_planner, real_contact = expert.expert_plan_candidates, backend._runner.prepare_contact
    calls, consumed, monitor_times, root_cuts = [], [], [], []
    def observed_planner(scene,cfg,**kwargs):
        assert scene is not backend.env and not hasattr(scene,'env')
        for name in ('data','model','collected_mask','component_quats','capture_truth'):
            assert not hasattr(scene,name)
        np.testing.assert_allclose(scene.component_positions()[:,:2],centers,atol=1e-7)
        assert scene.component_collision_x_max() != native_edge
        assert not np.allclose([p['nominal_radius'] for p in scene.layout],native_radii)
        state = history.snapshot()
        assert len(state['observations']) == 3
        np.testing.assert_allclose([o.timestamp-backend.origin for o in state['observations']],[0.,.2,.4],atol=1e-8)
        assert all(b.valid for b in state['observations'][-1].objects)
        calls.append(scene)
        # Original expert gets only disposable copies of estimated geometry.
        # Any attempt to read another privileged field fails at this boundary.
        class PoisonFacade:
            layout = scene.layout
            ee = scene.ee
            def component_positions(self): return scene.component_positions()
            def component_collision_x_max(self): return scene.component_collision_x_max()
            def __getattr__(self,name): raise AssertionError('privileged planner field: '+name)
        return real_planner(PoisonFacade(),cfg,**kwargs)
    expert.expert_plan_candidates = observed_planner
    def observed_contact(env,cfg,chosen):
        consumed.append(copy.deepcopy(chosen))
        root_cuts.append(backend.snapshot())
        return real_contact(env,cfg,chosen)
    backend._runner.prepare_contact = observed_contact
    def completed(safe):
        monitor_times.append(safe.timestamp)
        assert safe.objects and len(safe.objects)==6
        assert not safe.robot.visual_entry_valid  # native timer cannot fill this
        assert backend.selected_preparation_plan().landing_region == consumed[0].landing_region
        if len(monitor_times)==1: return False  # deliberate resumable native PAUSE
    if invalidate == 'producer-identity':
        features = tap.features_only
        feature_times = []
        def wrong_dependency(raw):
            packet = features(raw)
            feature_times.append(raw.timestamp)
            return replace(packet,input_hash='wrong-manifest') if len(feature_times)==4 else packet
        tap.features_only = wrong_dependency
        try:
            prepare_visually(backend,estimator,tap=tap,history=history,on_observation=completed)
        except ValueError as error:
            assert str(error)=='estimator feature/input dependency identity mismatch'
        else:
            raise AssertionError('real monitored producer fault did not propagate')
        assert backend.closed and len(calls)==len(consumed)==1
        assert len(measurement_times)==3 and len(feature_times)==4 and not monitor_times
        action_count=len(backend._commands)
        assert action_count==60 and counts=={'act':0,'backbone':20}  #40 startup +20 native prefix
        try: backend.resume_preparation()
        except RuntimeError as error: assert 'closed' in str(error)
        else: raise AssertionError('producer-fault attempt continued')
        assert len(backend._commands)==action_count
        evidence=dict(error='estimator feature/input dependency identity mismatch',closed=True,
                      planning_calls=len(calls),commands=action_count,feature_captures=len(feature_times),
                      estimated_captures=len(measurement_times),counts=counts,no_later_actions=True)
        atomic_checkpoint(root/'.venv/task-7-fix1-producer-evidence.json',json.dumps(evidence,indent=2).encode(),budget)
        print(json.dumps(evidence))
        raise SystemExit(0)
    result = prepare_visually(backend,estimator,tap=tap,history=history,on_observation=completed)
    if invalidate:
        assert result.status=='perception_invalid' and not result.ready and result.paused
        assert result.actions.shape==(20,4) and result.termination_reason=='' and result.visual_reason
        assert backend.closed and len(calls)==len(consumed)==1 and len(measurement_times)==4
        action_count=len(backend._commands)
        for action in (lambda: backend.resume_preparation(),lambda: backend.step_target(None),
                       lambda: backend.prepare(calls[0])):
            try: action()
            except RuntimeError as error: assert 'closed' in str(error)
            else: raise AssertionError('invalid visual attempt continued')
        assert len(backend._commands)==action_count
        evidence=dict(status=result.status,paused=result.paused,visual_reason=result.visual_reason,
                      native_reason=result.termination_reason,actions=len(result.actions),planning_calls=len(calls),
                      captures=len(measurement_times),closed=True,no_later_actions=True)
        atomic_checkpoint(root/'.venv/task-7-invalid-evidence.json',json.dumps(evidence,indent=2).encode(),budget)
        print(json.dumps(evidence))
        raise SystemExit(0)
    assert result.status=='paused' and not backend.closed
    paused=backend.snapshot()
    assert paused.cutpoint=='preparation:post-physics/pre-ack'
    paused_providers=copy.deepcopy(paused.participants)
    result=prepare_visually(backend,estimator,tap=tap,history=history,resume=True,on_observation=completed)
    assert result.ready and result.status=='ready', (result.status,result.visual_reason,result.termination_reason)
    assert len(calls)==len(consumed)==1 and monitor_times
    chosen = backend.selected_preparation_plan()
    np.testing.assert_array_equal(chosen.waypoints,consumed[0].waypoints)
    assert chosen.landing_region == consumed[0].landing_region
    clone = backend.selected_preparation_plan()
    clone.waypoints[:]=999; clone.landing_region['x_min']=999
    np.testing.assert_array_equal(backend.selected_preparation_plan().waypoints,chosen.waypoints)
    assert not backend._queries and counts['act']==0
    assert counts['backbone']==5*(3+len(monitor_times))
    initial = root_cuts[0]
    assert set(initial.participants)=={'features','estimator','history','visual_preparation'}
    assert initial.state['low_steps']==40 and initial.state['origin']==backend.origin
    ready = backend.snapshot()
    saved_history = history.snapshot()
    # Alter private native count/timer evidence at a completed capture, without
    # running a physical step. Safe context/history cannot acquire that truth.
    native_mask=backend.env.collected_mask
    backend.env.collected_mask=lambda: np.ones(6,bool)
    backend.guards['collection_timer'].first_entry_time=-999.
    poisoned=backend.observe()
    assert not poisoned.robot.visual_entry_valid and poisoned.robot.visual_entry_age==0.
    assert all(not o.robot.visual_entry_valid for o in history.snapshot()['observations'])
    backend.env.collected_mask=native_mask
    backend.restore(ready)
    saved_wrist = tap.capture_metadata(saved_history['observations'][-1].timestamp).cameras['wrist'].camera_to_world.copy()
    backend.advance_visual_clock()  # later capture must not replace retained calibration
    np.testing.assert_array_equal(tap.capture_metadata(saved_history['observations'][-1].timestamp).cameras['wrist'].camera_to_world,saved_wrist)
    backend.restore(ready)
    assert history.snapshot()['observations'][-1].timestamp == saved_history['observations'][-1].timestamp
    assert backend.participants['estimator'] is estimator and backend.participants['features'] is tap
    assert_native_origin(identity)
    evidence = dict(initial_absolute_times=[o.timestamp for o in initial.participants['history']['observations']],
        origin=backend.origin,ready_absolute_time=float(backend.env.time),ready_episode_age=backend.episode_time,
        startup_low_steps=initial.state['low_steps'],planning_calls=len(calls),monitor_captures=len(monitor_times),
        chosen_waypoints=chosen.waypoints.tolist(),estimated_centers=centers.tolist(),
        native_centers_audit_only=truth[:,:2].tolist(),estimated_edge=calls[0].component_collision_x_max(),native_edge_audit_only=native_edge,
        chosen_landing_region=chosen.landing_region,counts=counts,root_cutpoint=initial.cutpoint,ready_cutpoint=ready.cutpoint,
        actions=len(result.actions),same_providers_restored=True,historical_calibration_retained=True,
        paused_cutpoint=paused.cutpoint,paused_provider_names=list(paused_providers),
        private_count_timer_poison_excluded=True,
        fixture='deterministic detector outputs, no trained perception claim')
    atomic_checkpoint(root/'.venv/task-7-positive-evidence.json',json.dumps(evidence,indent=2).encode(),budget)
    print(json.dumps(evidence))
finally:
    if tap: tap.close()
    backend.close()
'''
    result = subprocess.run([sys.executable, '-c', body], cwd=Path(__file__).resolve().parents[2],
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True,
                            text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
