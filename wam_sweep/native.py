"""Source-checked frozen r8 control, private ACT input and visual capture.

No PAC learner is imported. Native functions are reused from sealed bytes in
an isolated namespace; the source tree is never modified. Online and offline
channels are deliberately separate.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache, wraps
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import time
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Protocol

import numpy as np

from .actions import ActionCodec, DecodedAction
from .artifacts import RunLease
from .contracts import CAMERA_VIEWS, EpisodeSpec, InputIdentity, OnlineObservation, RobotContext
from .inputs import assert_native_origin, digest_json, file_hash, verify_inputs
from .reference import NativeTargetContext, preview_reference
from .snapshots import (EpisodeSnapshot, capture_physics, restore_physics,
                        capture_random_state, restore_random_state,
                        externalize_images, materialize_images)
from .truth import capture_truth

_RUNTIMES = {}


@dataclass
class FrozenRuntime:
    identity: InputIdentity
    formal: ModuleType
    runner: ModuleType
    solver_lease: type
    modules: dict

    def check_origin(self):
        assert_native_origin(self.identity, native_aliases=tuple(self.modules))
        assert_native_origin(self.identity, self.modules, native_aliases=tuple(self.modules))


def load_frozen_runtime(identity: InputIdentity) -> FrozenRuntime:
    """Extract only pinned load_trial4/SolverLease, including real imports.

    Origin checks precede sys.path changes. Existing contaminated modules are
    rejected, never removed/replaced. Unregistered legacy helper objects are
    traversed too, so their original filenames/hashes are checked individually.
    """
    assert_native_origin(identity)
    actual = verify_inputs(identity.root, allow_unlisted=False)
    if asdict(actual) != asdict(identity):
        raise ValueError('input identity mismatch')
    cache_key = digest_json(asdict(identity))
    if cache_key in _RUNTIMES:
        result = _RUNTIMES[cache_key]
        result.check_origin()
        return result
    source = Path(identity.root) / 'source'
    path = source / 'pac_act/hybrid.py'
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in {'load_trial4', 'SolverLease'}]
    if ([n.name for n in selected] != ['SolverLease', 'load_trial4']
            or selected[1].args.args
            or [a.arg for a in selected[0].body[1].args.args] != ['self', 'module', 'cls']):
        raise RuntimeError('frozen loader signature changed')
    namespace = dict(__package__='pac_act', PROJECT=source, np=np,
                     lru_cache=lru_cache, importlib=importlib, json=json,
                     linecache=linecache, Path=Path, sys=sys,
                     _solver_owner=None, _solver_users=0)
    previous = sys.path.copy()
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    try:
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        formal, runner = namespace['load_trial4']()
        pending = [formal, runner] + [v for v in list(sys.modules.values()) if isinstance(v, ModuleType)]
        seen, modules = set(), {}
        while pending:
            module = pending.pop()
            if id(module) in seen:
                continue
            seen.add(id(module))
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).is_relative_to(source):
                modules[module.__name__] = module
                pending.extend(v for v in vars(module).values() if isinstance(v, ModuleType))
        result = FrozenRuntime(identity, formal, runner, namespace['SolverLease'], modules)
        result.check_origin()
        # Legacy load_helper creates real source-checked module objects without
        # registering them. Register those exact objects for pickle restoration;
        # never synthesize __file__ or substitute namespace-only modules.
        for name, module in modules.items():
            if name in sys.modules and sys.modules[name] is not module:
                raise RuntimeError('conflicting frozen alias identity: ' + name)
            sys.modules[name] = module
        _RUNTIMES[cache_key] = result
        return result
    finally:
        sys.path[:] = previous


class PerceivedPlanningScene(Protocol):
    """Task7's estimated geometry facade; no native environment fallback."""
    layout: list
    def component_positions(self) -> np.ndarray: ...
    def component_collision_x_max(self) -> float: ...


class StatefulProvider(Protocol):
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class BeliefProvider(StatefulProvider, Protocol):
    def estimate(self, observation: OnlineObservation, metadata: CaptureMetadata) -> tuple: ...


@dataclass(frozen=True)
class CameraCalibration:
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    size: tuple[int, int] = (320, 320)
    convention: str = 'mujoco_x_right_y_up_minus_z_forward'


@dataclass(frozen=True)
class CaptureMetadata:
    timestamp: float
    cameras: dict[str, CameraCalibration]
    input_hash: str
    schema: int = 1


@dataclass(frozen=True)
class QueryReceipt:
    issued_at: float
    received_at: float
    valid_from: float
    accepted: bool
    inference_wall_seconds: float


@dataclass(frozen=True)
class FrozenPolicyBinding:
    """Borrowed, read-only model/processors owned by ONE live backend.

    Task6 attaches hooks to policy.model.backbone['layer1'/'layer2'/'layer4']
    BEFORE its first query. For pre-contact/fixed-branch features_only, pass
    ONLY safe RGB CHW float tensors /255 through preprocessor, then call this
    SAME backbone once per CAMERA_VIEWS entry under no_grad. Do not call
    select_action/reset/train, change weights/processors, or create a model.
    Query frames use query_act instead, so hooks see the same five passes.
    Native36/fused tokens never belong in the safe feature packet.
    """
    policy: object
    preprocessor: object
    postprocessor: object


@dataclass(frozen=True)
class ResetFingerprint:
    poses: np.ndarray
    sha256: str


@dataclass(frozen=True)
class StepResult:
    actions: np.ndarray
    termination_reason: str
    done: bool
    executed_at: tuple[float, ...]


@dataclass(frozen=True)
class PreparationResult:
    ready: bool
    paused: bool
    actions: np.ndarray
    termination_reason: str


class _PreparationPause(BaseException):
    pass


class _SharedLimit(BaseException):
    pass


class _ReplayMismatch(BaseException):
    """Escape legacy diagnostic rollback; fail explicitly at the outer API."""


def _same_native_values(first, second):
    if isinstance(first, np.ndarray):
        return (isinstance(second, np.ndarray) and first.dtype == second.dtype
                and np.array_equal(first, second))
    if isinstance(first, dict):
        return (isinstance(second, dict) and first.keys() == second.keys()
                and all(_same_native_values(v, second[k]) for k, v in first.items()))
    if isinstance(first, (tuple, list)):
        return (type(first) is type(second) and len(first) == len(second)
                and all(_same_native_values(a, b) for a, b in zip(first, second)))
    return first == second


def _frame_operation(function):
    """Synchronous operations have no serializable unfinished Python locals."""
    @wraps(function)
    def run(self, *args, **kwargs):
        previous = getattr(self, '_busy', None)
        self._busy = function.__name__
        try:
            return function(self, *args, **kwargs)
        finally:
            self._busy = previous
    return run


def _clone_functions(module, names, overrides=None):
    """Copy globals, leaving original code objects and sealed modules intact."""
    result = SimpleNamespace()
    namespace = vars(result)
    namespace.update(vars(module))
    for name in names:
        original = getattr(module, name)
        namespace[name] = FunctionType(original.__code__, namespace, name,
                                       original.__defaults__, original.__closure__)
        namespace[name].__kwdefaults__ = original.__kwdefaults__
    namespace.update(overrides or {})
    return result


class FrozenSweepBackend:
    """One reset, one initial plan, full branch restoration, no attempt claims.

    Caller owns the run lease and a linked BudgetLedger activity around ALL
    loading/physics/render/query/replay. This backend polls that activity and
    never opens a nested activity or inference lock. Only query_act acquires
    the explicit owned native lock, once. No construction/close writes an
    EpisodeStore result. ``env``/``context`` are private native/offline handles.
    """
    @classmethod
    def open(cls, input_identity, episode_spec, *, budget, native_lock):
        self = cls()
        self.budget = budget
        self.native_lock = budget.ensure_owned(native_lock)
        self._admit()
        if type(episode_spec) is not EpisodeSpec:
            raise TypeError('EpisodeSpec required')
        self.input_identity, self.spec = input_identity, episode_spec
        self.runtime = load_frozen_runtime(input_identity)
        from sim.config import load_config
        from sim.environments.sweep_env import SweepEnv
        from sim.environments.ur10_interface import UR10CB3EndEffector
        from sim.act.interface import ACTObservationBuilder
        from sim.act.realtime import ActionChunkScheduler
        from sim.act.rollout import CollectionTerminationTracker
        self.cfg = self.runtime.formal.candidate_config(
            load_config(str(Path(input_identity.root) / 'evaluation_config.yaml')),
            episode_spec.target_count)
        if (float(self.cfg.sim.control_hz) != 100. or float(self.cfg.act.action_hz) != 25.
                or float(self.cfg.act.query_hz) != 5.):
            raise RuntimeError('frozen native clocks changed')
        self.runtime.runner.TRIAL_SPEC.check_clock(.01)
        self.native_context = NativeTargetContext(
            ((self.cfg.workspace.x_min, self.cfg.workspace.x_max),
             (self.cfg.workspace.y_min, self.cfg.workspace.y_max)),
            (self.cfg.workspace.z_search_min, self.cfg.end_effector.z_home))
        self.codec = ActionCodec(bounds=self.native_context.xy_bounds)
        self._pending_decode = None
        self.participants = {}
        self.belief_provider = None
        self._policy = None
        self._processors = None
        self._policy_binding = None
        self.env = None
        self.closed = False
        self._solver = self.runtime.solver_lease(self.runtime.runner, UR10CB3EndEffector)
        self._solver.__enter__()
        try:
            self.env = SweepEnv(self.cfg, seed=episode_spec.seed)
            self.env.reset(seed=episode_spec.seed)
            self.origin = float(self.env.time)
            self.phase, self.done, self.termination_reason = 'prepare', False, ''
            self.frames, self._low_steps = 0, 0
            self.reference = np.asarray([*self.env.tcp(), self.env.ee.tcp_yaw()], np.float32)
            self.builder = ACTObservationBuilder(self.cfg, camera_views=CAMERA_VIEWS, environment_state_dim=1)
            self.scheduler = ActionChunkScheduler(self.cfg)
            self.scheduler.reset(self.env.time)
            self.guards = dict(force_gate=self.runtime.runner.old.geometry.ForceGate(
                self.cfg.controller.safe_max_force, self.cfg.controller.safe_force_dwell_steps),
                collection_timer=CollectionTerminationTracker(target_count=episode_spec.target_count),
                last_guard_time=None, safety_reason='', collection_reason='')
            self.context, self.governor = None, None
            self._preparation_context, self._saved_preparation_latch = None, False
            self._residual, self._residual_rate = np.zeros(3), np.zeros(3)
            self._reference_valid = False
            self._visual_entry = (0., False)
            self._captures, self._queries, self._commands = {}, [], []
            self._next_capture = self.origin + .2
            self._preparation_root, self._plan, self._replay_cut = None, None, None
            self._preparing, self._paused, self._replaying = False, False, False
            self._on_observation = None
            self._offline_tick = None
            self._bridge_hashes = {str(Path(__file__)): file_hash(__file__),
                                   capture_physics.__code__.co_filename: file_hash(capture_physics.__code__.co_filename)}
            self._identity = digest_json(dict(input=asdict(input_identity), spec=asdict(episode_spec),
                                              config=self.cfg.to_dict(), xml=self.env.xml, bridge=self._bridge_hashes))
            self._native_step = self.env.step_control
            self.env.step_control = self._physical_step
            self._make_preparation_adapter()
            self.runtime.check_origin()
            return self
        except BaseException:
            self.close()
            raise

    def _admit(self):
        # The linked activity continuously accrues wall time. Full disk-tree
        # inventory is bounded to once per second, not once per 100Hz command.
        now = time.monotonic()
        if now < getattr(self, '_next_budget_poll', 0.):
            return
        state = json.loads(self.budget.path.read_bytes())
        active = state['active']
        if active is None or active.get('reservation') not in state['reservations']:
            raise RuntimeError('native work requires an active linked budget reservation and owned run lease')
        self.budget.poll().require()
        self._next_budget_poll = now + 1.

    @property
    def episode_time(self):
        return max(0., float(self.env.time) - self.origin)

    def _live(self):
        if self.closed:
            raise RuntimeError('backend is closed')
        self._admit()

    def reset_fingerprint(self) -> ResetFingerprint:
        self._live()
        if self._low_steps or self.context is not None or self._plan is not None:
            raise RuntimeError('fingerprint is reset-only, before policy/preparation')
        import hashlib
        poses = np.c_[self.env.component_positions(), self.env.component_quats()].astype('<f4')
        return ResetFingerprint(poses, hashlib.sha256(poses.tobytes()).hexdigest())

    def bind_provider(self, name: str, provider: StatefulProvider, *, beliefs=False):
        """Register causal state before preparing (not a feature callback cycle).

        Task6/7 neural consumers use beliefs=False: observe raw safe RGB, read
        capture_metadata, then OUTSIDE observe run features_only OR query_act
        through bind_frozen_policy, estimate beliefs, construct the final
        OnlineObservation with those same pixels/context, and append history.
        The optional beliefs=True callback is for synchronous RGB-only
        estimators: the new private capture is not yet published inside it.
        Never query there or encode an active frame a second time.
        """
        if not name or name in self.participants or not callable(getattr(provider, 'snapshot', None)) or not callable(getattr(provider, 'restore', None)):
            raise ValueError('named restorable provider required')
        if self._preparation_root is not None:
            raise RuntimeError('bind providers before preparation/replay root')
        self.participants[name] = provider
        if beliefs:
            if self.belief_provider is not None:
                raise ValueError('belief provider already bound')
            self.belief_provider = provider

    def bind_offline_truth_tick(self, callback):
        """Label-only 100Hz evidence; never attached to safe capture/provider."""
        self._offline_tick = callback

    def update_visual_entry(self, age: float, valid: bool):
        """Task7 monitor supplies ONLY its visual first-entry age/validity.

        Does not read or alter the native task timer, force gate or outcome.
        Same-time RGB/calibration remain cached when permitted context updates.
        """
        self._live()
        context = replace(self._robot_context(), visual_entry_age=age, visual_entry_valid=valid)
        self._visual_entry = (context.visual_entry_age, context.visual_entry_valid)

    def _robot_context(self):
        import mujoco
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.env.model, self.env.data, mujoco.mjtObj.mjOBJ_SITE,
                                self.env.ee.tcp_site_id, velocity, 0)
        return RobotContext(self.env.tcp(), float(self.env.ee.tcp_yaw()), self.env.tcp_velocity(),
                            float(velocity[2]), self.phase, self._residual, self._residual_rate,
                            self.episode_time, bool(self.context and self.context['ready']),
                            self._reference_valid, *self._visual_entry)

    def _native_contact_latched(self):
        if self.context and self.context['ready']:
            return True
        context = self._preparation_context
        if context is not None:
            return bool('z_limiter' in context and context['admittance'].phase != 'seek')
        return self._saved_preparation_latch

    @_frame_operation
    def observe(self) -> OnlineObservation:
        self._live()
        now = float(self.env.time)
        token = (self.phase, bool(self.context and self.context['ready']), self._reference_valid,
                 tuple(self._residual), tuple(self._residual_rate), self._visual_entry,
                 self._native_contact_latched())
        cached = self._captures.get(now)
        if cached is not None and cached[3] == token:
            return copy.deepcopy(cached[0])
        import mujoco
        images, cameras = {}, {}
        for view in (() if cached is not None else CAMERA_VIEWS):
            camera = 'wrist_cam' if view == 'wrist' else view
            images[view] = self.env.render_rgb(camera, size=(320, 320)).copy()
            cid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cid < 0:
                raise RuntimeError('missing frozen camera ' + camera)
            f = 160. / np.tan(np.deg2rad(self.env.model.cam_fovy[cid]) / 2.)
            intrinsic = np.array([[f, 0, 159.5], [0, f, 159.5], [0, 0, 1.]])
            matrix = np.eye(4)
            matrix[:3, :3] = self.env.data.cam_xmat[cid].reshape(3, 3)
            matrix[:3, 3] = self.env.data.cam_xpos[cid]
            cameras[view] = CameraCalibration(intrinsic, matrix)
        if cached is not None:
            images, metadata, beliefs = cached[0].images, cached[1], cached[0].objects
        else:
            metadata = CaptureMetadata(now, cameras, self.input_identity.manifest_hash)
            beliefs = ()
        observation = OnlineObservation(now, beliefs, self._robot_context(), images)
        if self.belief_provider is not None and cached is None:
            beliefs = self.belief_provider.estimate(copy.deepcopy(observation), copy.deepcopy(metadata))
            observation = OnlineObservation(now, beliefs, observation.robot, images)
        # _state is the original causal 36-D robot channel. Calling observe()
        # would also read supervisor masks, even with include_images=False.
        private = {'observation.state': self.builder._state(self.env, contact_latched=self._native_contact_latched()),
                   'observation.environment_state': np.array([self.spec.target_count], np.float32)}
        self._captures = {now: (copy.deepcopy(observation), metadata, private, token)}
        return observation

    def capture_metadata(self, timestamp: float) -> CaptureMetadata:
        if timestamp not in self._captures:
            raise ValueError('timestamp has no retained capture; persist metadata with its pixels')
        return copy.deepcopy(self._captures[timestamp][1])

    def native_act_input(self, timestamp: float) -> dict:
        """Private timestamp-bound provider for Task6; never pass to WAM."""
        if timestamp not in self._captures:
            raise ValueError('timestamp has no retained private native input')
        result = copy.deepcopy(self._captures[timestamp][2])
        result.update({'observation.images.' + view: self._captures[timestamp][0].images[view].transpose(2, 0, 1).astype(np.float32) / 255.
                       for view in CAMERA_VIEWS})
        return result

    def capture_truth(self):
        self._live()
        return capture_truth(self.env)

    @_frame_operation
    def capture_camera_truth(self, timestamp: float):
        """Offline-only visibility union from actual five-view segmentation.

        Requires contemporary RGB/calibration; never adds labels to the online
        observation. Invoke at capture cadence, not at 100Hz truth-tick cadence.
        Subsequent truth ticks correctly treat this evidence as stale/UNKNOWN.
        """
        self._live()
        if timestamp != float(self.env.time) or timestamp not in self._captures:
            raise ValueError('offline camera evidence requires current capture pixels/calibration')
        import mujoco
        visible = np.zeros(len(self.env.component_body_ids), dtype=bool)
        for view in CAMERA_VIEWS:
            camera = 'wrist_cam' if view == 'wrist' else view
            renderer = self.env._camera_renderers[(camera, 320, 320)]
            try:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(self.env.data, camera=camera)
                segmentation = renderer.render()
                geom_ids = np.unique(segmentation[:, :, 0][
                    segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)])
                for gid in geom_ids:
                    index = self.env.geom_to_component.get(int(gid))
                    if index is not None:
                        visible[index] = True
            finally:
                renderer.disable_segmentation_rendering()
        self.env.wam_truth_evidence = dict(timestamp=float(timestamp), visibility=visible,
                                          visibility_valid=np.ones_like(visible))
        return capture_truth(self.env)

    @_frame_operation
    def advance_visual_clock(self) -> OnlineObservation:
        """Native hold to NEXT distinct 5Hz capture without a policy query.

        Before planning this is airborne startup hold (observe, then twice for
        0,.2,.4). After contact preparation, call once before the first query
        to align ACT/WAM with the SAME continuing history grid. Frozen Z and
        governor acknowledgements continue at 100Hz, even for a partial hold
        frame; the two-second governor timer still starts only on executable
        ACT. No oracle, new plan, recovery, task clock or history reset.
        """
        self._live()
        startup = self._plan is None and self.context is None
        first_query = (self.context is not None and self.context['ready'] and not self._queries
                       and self.governor.started_at is None)
        if self.done or self._paused or not (startup or first_query):
            raise RuntimeError('visual hold requires startup or ready contact before the first query')
        target = self.origin + (int(np.floor((self.episode_time + 1e-8) * 5)) + 1) * .2
        while self.env.time < target - 1e-8 and not self.done:
            remaining = int(round((target - self.env.time) / .01))
            self._execute_target_frame(None, low_steps=min(4, remaining))
        return self.observe()

    def _physical_step(self, command):
        self._live()
        if self.done:
            raise _SharedLimit()
        before = float(self.env.time)
        try:
            self._native_step(command)
        except BaseException as error:
            self.done, self.phase = True, 'terminated'
            self.termination_reason = str(error)
            raise
        if abs(float(self.env.time) - before - .01) > 1e-8:
            raise RuntimeError('native low-level physical clock drift')
        self._low_steps += 1
        self.frames = (self._low_steps + 3) // 4
        values = np.array([command.x, command.y, command.z, command.yaw], np.float64)
        self._commands.append(values.copy())
        try:
            self.runtime.runner.measure(self.env)
        except Exception as error:
            self.done, self.phase, self.termination_reason = True, 'terminated', str(error)
            raise
        row = dict(t=float(self.env.time), normal_force=float(self.env.normal_force()),
                   object_collected=self.env.collected_mask(),
                   trial_partial_overlap_mask=self.env.partial_collection_overlap_mask())
        self.runtime.runner.old.observe_control_row(self.guards, row)
        reason = self.guards['safety_reason'] or self.guards['collection_reason']
        if not reason and self.episode_time >= 45. - 1e-8:
            reason = '45-second shared episode timeout'
        if not reason and self._low_steps >= 1200 * 4:
            reason = '1200-frame shared episode timeout'
        if reason:
            self.done, self.phase, self.termination_reason = True, 'terminated', reason
        if self._low_steps % 4 == 0:
            self.builder.advance_state_history(self.env, contact_latched=self._native_contact_latched())
        if self._offline_tick is not None and not self._replaying:
            self._offline_tick(self.capture_truth())
        if self._preparing and self.env.time >= self._next_capture - 1e-8:
            observation = self.observe()
            self._next_capture += .2
            if not self._replaying and self._on_observation is not None:
                if self._on_observation(observation, self.capture_metadata(observation.timestamp)) is False:
                    self._paused = True
                    raise _PreparationPause()
        if self._replaying and self._low_steps == self._replay_cut.state['low_steps']:
            expected = self._replay_cut
            actual = capture_physics(self.env)
            if (not np.array_equal(actual.integration, expected.physics.integration)
                    or not _same_native_values(actual.end_effector, expected.physics.end_effector)
                    or not np.array_equal(np.asarray(self._commands), np.asarray(expected.state['commands']))):
                raise _ReplayMismatch('preparation prefix replay diverged at saved pending acknowledgement')
            for name, state in expected.participants.items():
                self.participants[name].restore(copy.deepcopy(state))
            # User monitor callbacks are deliberately not re-emitted during
            # replay. Restore their causal control/RNG state at the matched cut.
            self.codec.restore(expected.state['codec'])
            self._visual_entry = expected.state['visual_entry']
            self._residual = expected.state['residual'].copy()
            self._residual_rate = expected.state['residual_rate'].copy()
            self._captures = copy.deepcopy(expected.state['captures'])
            restore_random_state(expected.physics.random_state)
            self._replaying = False

    def _make_preparation_adapter(self):
        # Rebind only function globals. No shared native module monkeypatch.
        original = self.runtime.runner.old
        def shared_guard(context, row):
            self._preparation_context = context
            original.observe_control_row(self.guards, row)
            context.update({key: self.guards[key] for key in self.guards})
        def prefix_budget(phases):
            count = sum(p in ('approach', 'descent') for p in phases)
            if self.frames + count >= 1200:
                raise RuntimeError('planned expert prefix exhausts shared 1200-frame budget')
            return count
        old = _clone_functions(original, ('prepare_prefix', 'record_row', 'adapt_act'),
                               dict(observe_control_row=shared_guard, validate_prefix_budget=prefix_budget))
        self._runner = _clone_functions(self.runtime.runner,
                                       ('prepare_contact', 'build_act_driver'), {'old': old})

    def prepare(self, scene: PerceivedPlanningScene, *, on_observation=None) -> PreparationResult:
        self._live()
        if scene is self.env or hasattr(scene, 'data') or hasattr(scene, 'model'):
            raise TypeError('prepare requires perceived geometry, not native environment')
        from sim.act.expert import expert_plan_candidates
        if self.done or self._plan is not None:
            raise RuntimeError('one initial planning call per attempt; use resume_preparation')
        plans = expert_plan_candidates(scene, self.cfg, max_candidates=1)
        return self._prepare_plan(plans, on_observation)

    def selected_preparation_plan(self):
        """Read-only clone of the already chosen plan; never plans or reads GT."""
        return copy.deepcopy(self._plan)

    def prepare_bootstrap_oracle(self, *, on_observation=None) -> PreparationResult:
        """Explicit privileged bootstrap/test path; never a visual fallback."""
        self._live()
        from sim.act.expert import expert_plan_candidates
        if self.done or self._plan is not None:
            raise RuntimeError('oracle preparation cannot retry a plan')
        return self._prepare_plan(expert_plan_candidates(self.env, self.cfg, max_candidates=1), on_observation)

    def _prepare_plan(self, plans, callback):
        if not plans or not plans[0].feasible:
            self.done, self.phase, self.termination_reason = True, 'terminated', 'no_plan'
            return PreparationResult(False, False, np.empty((0, 4)), self.termination_reason)
        self._plan = copy.deepcopy(plans[0])
        self._next_capture = self.origin + (int(np.floor((self.episode_time + 1e-8) * 5)) + 1) * .2
        self._preparation_root = self.snapshot()
        return self._run_preparation(callback)

    def resume_preparation(self, *, on_observation=None) -> PreparationResult:
        self._live()
        if not self._paused or self._preparation_root is None:
            raise RuntimeError('no interrupted preparation to resume')
        cut = self.snapshot()
        origin = self._preparation_root
        self.restore(origin)
        self._preparation_root = origin
        self._replay_cut, self._replaying = cut, True
        return self._run_preparation(on_observation)

    def _run_preparation(self, callback):
        self._preparing, self._paused, self._on_observation = True, False, callback
        cut_steps = self._replay_cut.state['low_steps'] if self._replaying else self._low_steps
        try:
            context = self._runner.prepare_contact(self.env, self.cfg, self._plan)
            if self._replaying:
                raise RuntimeError('preparation returned before saved replay cutpoint')
            self.runtime.formal.validate_acquisition(context)
            self.context = context
            context.update({key: self.guards[key] for key in self.guards})
            if context['ready'] and not self.done:
                self.phase = 'handoff'
                self.reference = np.asarray(context['last_reference'], np.float32).copy()
                self.builder._history = copy.deepcopy(self.builder._history)
                self.builder._history.clear()
                self.builder._history.extend(copy.deepcopy(context['history']))
                self._build_driver()
            elif not self.done:
                self.done, self.phase = True, 'terminated'
                self.termination_reason = (context.get('safety_reason') or context.get('collection_reason')
                                           or context.get('terminal_reason') or context.get('error') or 'no_handoff')
        except _PreparationPause:
            pass
        except _ReplayMismatch as error:
            self.done, self.phase, self.termination_reason = True, 'terminated', str(error)
            raise RuntimeError(str(error)) from error
        except _SharedLimit:
            if not self.done:
                raise
        finally:
            self._preparing, self._on_observation = False, None
        actions = np.asarray(self._commands[max(0, cut_steps):], np.float64).reshape(-1, 4)
        return PreparationResult(bool(self.context and self.context['ready'] and not self.done),
                                 self._paused, actions, self.termination_reason)

    def _build_driver(self, governor_state=None):
        self.context['env'] = self.env
        driver, self.governor = self.runtime.formal.build_formal_act_driver(self._runner, self.cfg, self.context)
        if governor_state is not None:
            vars(self.governor).update(copy.deepcopy(governor_state))
        namespace = driver.__globals__
        self._reference_substep = namespace['_policy_reference_substep']
        self._record, self._chunk_ack = namespace['_trial_record'], namespace['_trial_chunk_ack']
        self._z_command = namespace['_contact_z_command']

    def accept_act_chunk(self, absolute_targets, *, issued_at, received_at) -> QueryReceipt:
        self._live()
        if self.done or self.context is None or not self.context['ready']:
            raise RuntimeError('ACT chunks require completed contact preparation')
        if abs(received_at - self.env.time) > 1e-8 or issued_at > received_at:
            raise ValueError('ACT receipt must match current physical time')
        targets = np.asarray(absolute_targets, np.float32)
        if targets.shape != (25, 4) or not np.isfinite(targets).all():
            raise ValueError('frozen ACT must return exactly 25 finite targets')
        accepted = self.scheduler.accept(issued_at, targets, received_at)
        self._chunk_ack(accepted)
        receipt = QueryReceipt(float(issued_at), float(received_at), float(issued_at) + self.scheduler.budget, accepted, 0.)
        self._queries.append(receipt)
        return receipt

    @_frame_operation
    def bind_frozen_policy(self) -> FrozenPolicyBinding:
        """Initialize/borrow the single frozen policy even BEFORE contact.

        No physics, scheduler/query/ACK or handoff mutation. All construction
        RNG is restored; full local weights load strictly with external
        backbone initialization disabled. Caller owns the linked budget/run
        activity, including any external features_only passes through binding.
        This acquires the native lock only during construction; query_act calls
        it BEFORE acquiring that lock for inference (never double-flock).
        """
        self._live()
        if self._policy_binding is not None:
            return self._policy_binding
        with RunLease(self.native_lock):
            random_state = capture_random_state()
            try:
                from sim.act.policy import (build_act_config, build_act_processors,
                                            require_act_dependencies, validate_act_policy_contract)
                model = str(Path(self.input_identity.root) / 'base_model')
                # The saved full ACT checkpoint includes every ResNet weight.
                # LeRobot otherwise requests ImageNet weights while constructing
                # the architecture, even with local_files_only=True. Suppress
                # ONLY that redundant initializer, strictly load all saved
                # tensors, and restore the serialized configuration attribute.
                _, _, config_class, policy_class = require_act_dependencies()
                cfg = config_class.from_pretrained(model, local_files_only=True)
                cfg.device = build_act_config(self.cfg).device
                initializer = cfg.pretrained_backbone_weights
                cfg.pretrained_backbone_weights = None
                try:
                    policy = policy_class.from_pretrained(model, config=cfg, local_files_only=True, strict=True)
                finally:
                    cfg.pretrained_backbone_weights = initializer
                validate_act_policy_contract(cfg, self.cfg)
                if tuple(k.removeprefix('observation.images.') for k in cfg.image_features) != CAMERA_VIEWS:
                    raise RuntimeError('frozen ACT camera ordering changed')
                policy.eval().requires_grad_(False)
                policy.reset()
                pre, post = build_act_processors(cfg, pretrained_path=model)
                self._policy, self._processors = policy, (pre, post)
                self._policy_binding = FrozenPolicyBinding(policy, pre, post)
            finally:
                restore_random_state(random_state)
        self.runtime.check_origin()
        return self._policy_binding

    @_frame_operation
    def query_act(self, observation: OnlineObservation, *, feature_hook=None) -> QueryReceipt:
        """One frozen forward with five spatial hooks, exact saved processors.

        Attach layer1/2/4 hooks through bind_frozen_policy BEFORE this call;
        they see the FIRST query's same five passes. The optional legacy
        feature_hook(view, detached_layer4) needs no second backbone pass.
        Safe spatial hooks never receive ACT36, goal/fusion tokens or truth.
        """
        self._live()
        if self.context is None or not self.context['ready'] or self.done:
            raise RuntimeError('ACT query requires ready live native contact')
        now = float(self.env.time)
        if observation.timestamp != now or not self.scheduler.query_due(now):
            raise ValueError('ACT query must use a current capture and 5Hz due time')
        inputs = self.native_act_input(now)
        for view in CAMERA_VIEWS:
            if not np.array_equal(observation.images[view], self._captures[now][0].images[view]):
                raise ValueError('ACT/WAM capture pixels do not match')
        import torch
        from sim.act.interface import action_deltas_to_absolute
        binding = self.bind_frozen_policy()
        with RunLease(self.native_lock):
            pre, post = binding.preprocessor, binding.postprocessor
            seen = []
            def hook(module, args, output):
                view = CAMERA_VIEWS[len(seen)]
                seen.append(view)
                if feature_hook is not None:
                    feature_hook(view, output['feature_map'].detach().clone())
            handle = binding.policy.model.backbone.register_forward_hook(hook)
            started = time.monotonic()
            try:
                batch = pre(self.builder.torch_batch(inputs))
                with torch.no_grad():
                    deltas = post(binding.policy.predict_action_chunk(batch)).detach().cpu().numpy()
                if tuple(seen) != CAMERA_VIEWS or deltas.shape != (1, 25, 4) or not np.isfinite(deltas).all():
                    raise RuntimeError('frozen ACT output/feature contract changed')
            finally:
                handle.remove()
            elapsed = time.monotonic() - started
        self.scheduler.mark_query(now)
        receipt = self.accept_act_chunk(action_deltas_to_absolute(self.reference.copy(), deltas[0]),
                                        issued_at=now, received_at=float(self.env.time))
        receipt = QueryReceipt(receipt.issued_at, receipt.received_at, receipt.valid_from, receipt.accepted, elapsed)
        self._queries[-1] = receipt
        self.runtime.check_origin()
        return receipt

    def reference_preview(self):
        return preview_reference(self.scheduler, self.reference, float(self.env.time), native_context=self.native_context)

    def _decode_frame(self):
        """Read current decode provenance WITHOUT pruning the live scheduler."""
        ready = bool(self.context and self.context['ready'])
        q = copy.deepcopy(self.scheduler).action_for(self.env.time) if ready else None
        scheduler = dict(vars(self.scheduler))
        scheduler['chunk'] = None if self.scheduler.chunk is None else asdict(self.scheduler.chunk)
        ensemble = self.scheduler.temporal_ensemble
        scheduler['temporal_ensemble'] = (None if ensemble is None else
            dict(vars(ensemble), chunks=[asdict(chunk) for chunk in ensemble.chunks]))
        return copy.deepcopy(dict(timestamp=float(self.env.time), low_steps=self._low_steps,
            phase=self.phase, ready=ready, active=q is not None,
            actual_tcp=np.r_[self.env.tcp(), self.env.ee.tcp_yaw()],
            nominal=self.reference.copy() if q is None else self.native_context.normalize(q),
            held_reference=self.reference.copy(), codec=self.codec.snapshot(), scheduler=scheduler))

    def decode_target(self, raw) -> DecodedAction:
        """Issue ONE executable owned action for this exact complete frame.

        Decode on an isolated clone using current measured TCP and normalized
        accepted ACT reference; real codec state commits only on consumption.
        Pass the exact returned object to step_target(action), not a copy or a
        direct codec.decode result. Payload/audit mutation, time/TCP/reference/
        scheduler/codec drift and reuse are rejected before physical stepping.
        Snapshot is forbidden until consumed/cancelled; restore invalidates it.
        No native mapping/bounds change and no post-clamping of corrections.
        """
        self._live()
        if (getattr(self, '_busy', None) or self._preparing or self.done or self._paused
                or self._pending_decode is not None):
            raise RuntimeError('decode requires a live complete frame with no pending decoded target')
        frame = self._decode_frame()
        if not frame['active']:
            raise ValueError('owned decode requires an executable accepted ACT reference')
        codec = copy.deepcopy(self.codec)
        action = codec.decode(frame['actual_tcp'], frame['nominal'], raw, .04, True)
        self._pending_decode = dict(action=action, payload=copy.deepcopy(action),
                                    frame=frame, owner=self.codec, after=codec.snapshot())
        return action

    def cancel_decoded_target(self):
        """Discard an unconsumed ticket; physics and real codec stay unchanged."""
        self._live()
        if getattr(self, '_busy', None) or self._preparing:
            raise RuntimeError('cancel requires a complete operation boundary')
        self._pending_decode = None

    @_frame_operation
    def step_target(self, target=None, *, raw=None) -> StepResult:
        """Execute one 25Hz frame through unchanged 100Hz native control.

        ``None`` uses accepted ACT or native hold; ``raw`` runs the owned codec.
        A supplied DecodedAction must be the exact single-use decode_target()
        result for this frame. Its float64 target is inserted AFTER nominal
        FP32 normalization and never requantized. Direct codec results fail.
        """
        return self._execute_target_frame(target, raw=raw)

    def _execute_target_frame(self, target, *, raw=None, low_steps=4):
        self._live()
        if self.done or self._paused:
            raise RuntimeError('cannot step a terminal/interrupted preparation')
        from sim.controllers.hybrid import Command
        from sim.act.evaluate import _policy_reference_substep
        if target is not None:
            pending = self._pending_decode
            if (type(target) is not DecodedAction or raw is not None or pending is None
                    or target is not pending['action'] or self.codec is not pending['owner']
                    or not _same_native_values(asdict(pending['payload']), asdict(target))
                    or not _same_native_values(pending['frame'], self._decode_frame())):
                raise ValueError('decoded target is foreign, consumed, tampered or stale for the current frame')
        elif self._pending_decode is not None:
            raise RuntimeError('consume or cancel the pending decoded target before raw/hold execution')
        ready = bool(self.context and self.context['ready'])
        q = self.scheduler.action_for(self.env.time) if ready else None
        active = q is not None
        if low_steps != 4 and active:
            raise RuntimeError('only native inactive hold can end at a partial-frame boundary')
        nominal = self.reference.copy() if not active else self.native_context.normalize(q)
        if target is not None:
            # Execute the private validated copy, never mutable caller arrays.
            decoded = copy.deepcopy(pending['payload'])
            self.codec.restore(pending['after'])
            self._pending_decode = None  # consume BEFORE any physical command
        else:
            decoded = self.codec.decode(np.r_[self.env.tcp(), self.env.ee.tcp_yaw()], nominal,
                                        np.zeros(3) if raw is None else raw, .04, active)
        self._residual, self._residual_rate = decoded.residual.copy(), decoded.residual_rate.copy()
        self._reference_valid = active
        if active:
            self.governor.action_ready(float(self.env.time))
        start = self.reference.copy()
        actions, times = [], []
        for k in range(1, low_steps + 1):
            if self.done:
                break
            if ready:
                z = self._z_command(self.cfg, self.context['admittance'], self.context['z_nominal'],
                                     float(self.env.normal_force()), relief=self.context['relief'])
                command = self._reference_substep(start, decoded.target, k, 4, .01,
                    float(self.cfg.controller.sweep_speed), float(self.cfg.controller.z_speed),
                    float(self.cfg.controller.yaw_speed_limit), True, admittance_z=z)
            else:
                command = _policy_reference_substep(start, nominal, k, 4, .01,
                    float(self.cfg.controller.travel_speed), float(self.cfg.controller.z_speed),
                    float(self.cfg.controller.yaw_speed_limit), False)
            self.env.step_control(Command(*map(float, command)))
            if ready:
                row = self.runtime.runner.physical_row(self.env, command, self.builder, self.phase, True)
                self._record(row, self.env)
            self.reference = np.asarray(command, np.float32).copy()
            actions.append(np.asarray(command).copy())
            times.append(float(self.env.time))
        if ready and not self.done and self.governor.started_at is not None and self.env.time - self.governor.started_at >= 2. - 1e-9:
            self.phase = 'sweep'
        return StepResult(np.asarray(actions).reshape(-1, 4), self.termination_reason, self.done, tuple(times))

    def snapshot(self, *, image_writer=None) -> EpisodeSnapshot:
        self._live()
        if getattr(self, '_busy', None):
            raise RuntimeError('snapshot requires a complete frame/capture/query boundary')
        if self._pending_decode is not None:
            raise RuntimeError('snapshot requires consumption/cancellation of the pending decoded target')
        context = None if self.context is None else copy.deepcopy({k: v for k, v in self.context.items()
                                                                   if k not in {'env', 'handoff_governor'}})
        state = dict(origin=self.origin, phase=self.phase, done=self.done, reason=self.termination_reason,
                     frames=self.frames, low_steps=self._low_steps, reference=self.reference,
                     builder={k: v for k, v in vars(self.builder).items() if k != 'cfg'},
                     scheduler=vars(self.scheduler), guards=self.guards,
                     context=context, governor=None if self.governor is None else vars(self.governor),
                     codec=self.codec.snapshot(), residual=self._residual, residual_rate=self._residual_rate,
                     reference_valid=self._reference_valid, visual_entry=self._visual_entry,
                     contact_latched=self._native_contact_latched(),
                     captures=self._captures, queries=self._queries, commands=self._commands,
                     next_capture=self._next_capture, plan=self._plan, paused=self._paused or self._preparing,
                     preparation_root=self._preparation_root)
        cutpoint = 'preparation:post-physics/pre-ack' if self._paused or self._preparing else self.phase + ':frame-boundary'
        result = EpisodeSnapshot(self._identity, capture_physics(self.env), copy.deepcopy(state),
                                  {k: copy.deepcopy(v.snapshot()) for k, v in self.participants.items()}, cutpoint)
        return result if image_writer is None else externalize_images(result, image_writer)

    def restore(self, snapshot: EpisodeSnapshot, *, image_reader=None):
        self._live()
        if getattr(self, '_busy', None) or self._preparing:
            raise RuntimeError('restore requires a completed operation boundary; pause preparation first')
        self.runtime.check_origin()
        if any(file_hash(path) != digest for path, digest in self._bridge_hashes.items()):
            raise ValueError('backend/snapshot implementation drift; reopen with matching provenance')
        if (type(snapshot) is not EpisodeSnapshot or snapshot.schema != 1 or snapshot.identity != self._identity
                or set(snapshot.participants) != set(self.participants)):
            raise ValueError('snapshot provenance/providers do not match')
        if snapshot.images_externalized:
            snapshot = materialize_images(snapshot, image_reader)
        state = copy.deepcopy(snapshot.state)
        self._pending_decode = None  # completed-root restore invalidates ALL old tickets
        restore_physics(self.env, snapshot.physics)
        self.origin, self.phase, self.done = state['origin'], state['phase'], state['done']
        self.termination_reason, self.frames, self._low_steps = state['reason'], state['frames'], state['low_steps']
        self.reference, self.guards, self.context = state['reference'], state['guards'], state['context']
        vars(self.builder).clear()
        self.builder.cfg = self.cfg
        vars(self.builder).update(state['builder'])
        vars(self.scheduler).clear()
        vars(self.scheduler).update(state['scheduler'])
        self.codec.restore(state['codec'])
        self._residual, self._residual_rate = state['residual'], state['residual_rate']
        self._reference_valid, self._visual_entry = state['reference_valid'], state['visual_entry']
        self._preparation_context, self._saved_preparation_latch = None, state['contact_latched']
        self._captures, self._queries, self._commands = state['captures'], state['queries'], state['commands']
        self._next_capture, self._plan, self._paused = state['next_capture'], state['plan'], state['paused']
        self._preparation_root = state['preparation_root']
        self._replay_cut, self._replaying, self._preparing = None, False, False
        self.governor = None
        self._make_preparation_adapter()
        if self.context is not None and self.context['ready']:
            self.context.update({key: self.guards[key] for key in self.guards})
            self._build_driver(state['governor'])
        for name, value in snapshot.participants.items():
            self.participants[name].restore(copy.deepcopy(value))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self._solver.__exit__(None, None, None)
