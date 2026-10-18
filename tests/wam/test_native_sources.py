"""Real sealed imports in fresh processes; current checkout must never win."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


SEALED = Path('/Users/huazhi_luke/.local/share/wam-sweep-v1/inputs/r8')


def native_script(body):
    assert importlib.util.find_spec('wam_sweep.native') is not None, 'frozen native backend missing'
    result = subprocess.run([sys.executable, '-c', body], cwd=Path(__file__).resolve().parents[2],
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, result.stdout
    return result.stdout


PREAMBLE = f'''
from wam_sweep.inputs import verify_inputs, assert_native_origin
from wam_sweep.native import load_frozen_runtime
identity = verify_inputs({str(SEALED)!r}, allow_unlisted=False)
'''

BACKEND = PREAMBLE + '''
import json, numpy as np
from pathlib import Path
from wam_sweep.artifacts import BudgetLedger
from wam_sweep.config import WAMConfig
from wam_sweep.contracts import EpisodeSpec, CAMERA_VIEWS
from wam_sweep.native import FrozenSweepBackend
root = Path.cwd()
account = root / '.venv/task-2-budget.json'
policy = json.loads(account.read_bytes())['policy']
budget = BudgetLedger(account, owned_roots=policy['roots'], config=WAMConfig.load(root/'configs/wam_visual_v1.yaml').budget)
spec = EpisodeSpec('task5-bounded-fixture', 2120018, 1, 'train', -1, 'act', 0)
backend = FrozenSweepBackend.open(identity, spec, budget=budget, native_lock=root/'.venv/task-5-native.lock')
'''


def test_reset_capture_private_input_poisoning_and_shared_clock():
    native_script(BACKEND + '''
try:
    assert backend.phase == 'prepare'
    assert backend.termination_reason == ''
    poses = backend.reset_fingerprint()
    assert poses.poses.shape == (6, 7) and poses.poses.dtype.str == '<f4'
    obs = backend.observe()
    metadata = backend.capture_metadata(obs.timestamp)
    native = backend.native_act_input(obs.timestamp)
    assert obs.objects == ()
    assert set(obs.images) == set(CAMERA_VIEWS)
    assert all(im.shape == (320,320,3) for im in obs.images.values())
    assert native['observation.state'].shape == (36,)
    np.testing.assert_array_equal(native['observation.environment_state'], [1.])
    assert set(native) == {'observation.state','observation.environment_state'} | {'observation.images.'+v for v in CAMERA_VIEWS}
    for view in CAMERA_VIEWS:
        np.testing.assert_array_equal(native['observation.images.'+view], obs.images[view].transpose(2,0,1).astype(np.float32)/255.)
        assert metadata.cameras[view].camera_to_world.shape == (4,4)
    root_state = backend.snapshot()
    backend.env.collected_mask = lambda: np.ones(6, bool)
    backend.guards['collection_timer'].first_entry_time = -999.
    backend._captures.clear()  # demand a fresh capture under poisoned supervisor
    poisoned = backend.observe()
    np.testing.assert_array_equal(backend.native_act_input(poisoned.timestamp)['observation.state'], native['observation.state'])
    assert poisoned.robot.visual_entry_valid is False
    assert poisoned.objects == ()
    backend.restore(root_state)
    # Initial visual waiting consumes the SAME preparation origin/frame budget.
    first = backend.step_target(None)
    assert first.actions.shape == (4,4)
    assert abs(backend.episode_time-.04) < 1e-8
    assert backend.frames == 1
    backend.restore(root_state)
    second = backend.step_target(None)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.termination_reason == second.termination_reason
    assert backend.capture_truth().timestamp == backend.env.time
    backend.restore(root_state)
    startup = [backend.observe(), backend.advance_visual_clock(), backend.advance_visual_clock()]
    np.testing.assert_allclose([x.timestamp-backend.origin for x in startup], [0., .2, .4], atol=1e-8)
    assert backend.frames == 10 and backend._plan is None and not backend._queries
    original_pixels = backend.observe().images
    backend._reference_valid = True
    assert backend.observe().robot.reference_valid
    for view in CAMERA_VIEWS:
        np.testing.assert_array_equal(backend.observe().images[view], original_pixels[view])
    backend.update_visual_entry(.3, True)
    assert backend.observe().robot.visual_entry_valid
    assert backend.observe().robot.visual_entry_age == .3
    backend.restore(root_state)
    assert not backend.observe().robot.visual_entry_valid
finally:
    backend.close()
assert backend.termination_reason == ''  # closing identity work is not failure
''')


def test_real_preparation_replay_handoff_and_sweep_restore():
    native_script(BACKEND + '''
try:
    seen = []
    inline_snapshots = []
    def pause(observation, metadata):
        seen.append(observation.timestamp)
        inline_snapshots.append(backend.snapshot())
        return False
    paused = backend.prepare_bootstrap_oracle(on_observation=pause)
    assert not paused.ready and paused.paused
    root_state = backend.snapshot()
    assert root_state.cutpoint == 'preparation:post-physics/pre-ack'
    assert inline_snapshots[0].cutpoint == root_state.cutpoint
    assert inline_snapshots[0].state['paused'] is True
    first = backend.resume_preparation()
    assert first.ready, first.termination_reason
    backend.restore(root_state)
    second = backend.resume_preparation()
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.termination_reason == second.termination_reason
    assert backend.phase == 'handoff'
    assert backend.frames > 0 and backend.episode_time > 0
    assert backend.guards['collection_timer'] is backend.context['collection_timer']
    assert backend.observe().robot.contact_ready
    # One genuine frozen ACT inference, five spatial hook outputs, no second encoder.
    features = []
    import torch, draccus
    cold = root/'.venv/task-5-cold-no-fetch'
    assert not cold.exists()
    torch.hub.get_dir = lambda: str(cold)
    import torchvision.models._api as vision_api
    def forbidden_download(*args, **kwargs):
        raise RuntimeError('external ImageNet initialization is forbidden')
    vision_api.load_state_dict_from_url = forbidden_download
    receipt = backend.query_act(backend.observe(), feature_hook=lambda view, feature: features.append((view, feature.shape)))
    assert [v for v, shape in features] == list(CAMERA_VIEWS)
    assert receipt.valid_from == receipt.issued_at + .2
    assert backend.scheduler.query_due(receipt.issued_at) is False
    assert not cold.exists(), 'cold construction accessed an ImageNet cache'
    from lerobot.configs.policies import PreTrainedConfig
    saved_config = json.loads((Path(identity.root)/'base_model/config.json').read_bytes())
    assert draccus.encode(backend._policy.config, PreTrainedConfig) == saved_config
    # The full sealed state_dict, including all backbone tensors, was loaded.
    from safetensors.torch import load_file
    sealed_weights = load_file(str(Path(identity.root)/'base_model/model.safetensors'))
    assert set(backend._policy.state_dict()) == set(sealed_weights)
    for name, value in backend._policy.state_dict().items():
        assert name in sealed_weights
        assert __import__('torch').equal(value.cpu(), sealed_weights[name]), name
    import pickle
    persisted = pickle.loads(pickle.dumps(backend.snapshot()))
    backend.restore(persisted)
    # Exercise the real governor immediately on an accepted executable chunk.
    backend.accept_act_chunk(np.tile(backend.reference, (25,1)), issued_at=backend.env.time-.2, received_at=backend.env.time)
    physical_evidence = {'preparation_first': first.actions, 'preparation_second': second.actions,
                         'preparation_root_commands': np.asarray(root_state.state['commands']),
                         'preparation_root_integration': root_state.physics.integration}
    for phase in ('handoff', 'sweep'):
        if phase == 'sweep':
            # Actually execute the complete two-second handoff. There is no
            # learner wait gate and no artificially advanced governor clock.
            while backend.env.time-backend.governor.started_at < 2.-1e-8:
                backend.accept_act_chunk(np.tile(backend.reference, (25,1)), issued_at=backend.env.time-.2, received_at=backend.env.time)
                assert not backend.step_target(None).done
            assert backend.phase == 'sweep'
            backend.accept_act_chunk(np.tile(backend.reference, (25,1)), issued_at=backend.env.time-.2, received_at=backend.env.time)
        start = backend.snapshot()
        first = backend.step_target(None, raw=np.array([.25,-.2,.1]))
        assert len(first.actions) == 4
        state1 = backend.snapshot()
        truth1 = backend.capture_truth()
        backend.restore(start)
        second = backend.step_target(None, raw=np.array([.25,-.2,.1]))
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(state1.physics.integration, backend.snapshot().physics.integration)
        assert first.termination_reason == second.termination_reason
        assert truth1.raw.fields['contacts'] == backend.capture_truth().raw.fields['contacts']
        assert backend.governor.last_time == backend.env.time
        assert backend.context['z_limiter'].last_time == backend.env.time
        assert backend.codec.snapshot()['frames'] == start.state['codec']['frames'] + 1
        assert backend.codec.snapshot() == state1.state['codec']
        repeated = backend.snapshot()
        assert len(state1.state['builder']['_history']) == len(repeated.state['builder']['_history'])
        for (t1,h1), (t2,h2) in zip(state1.state['builder']['_history'], repeated.state['builder']['_history']):
            assert t1 == t2
            np.testing.assert_array_equal(h1,h2)
        physical_evidence[phase+'_first'] = first.actions
        physical_evidence[phase+'_second'] = second.actions
        physical_evidence[phase+'_state_first'] = state1.physics.integration
        physical_evidence[phase+'_state_second'] = backend.snapshot().physics.integration
    np.savez_compressed(root/'.venv/task-5-physical-evidence.npz', **physical_evidence)
    (root/'.venv/task-5-control-evidence.json').write_text(json.dumps(dict(
        preparation_command_count=len(physical_evidence['preparation_first']),
        actual_lowlevel_steps=backend._low_steps, shared_frames=backend.frames,
        actual_episode_time=backend.episode_time, governor_started_at=backend.governor.started_at,
        actual_elapsed_handoff=backend.env.time-backend.governor.started_at,
        governor_mode=backend.context['act_trace'][-1]['handoff_smoothing']['mode'],
        hook_shapes=[(view,list(shape)) for view,shape in features],
        checkpoint_tensors_verified=len(sealed_weights),
        saved_config_equal=True, cold_no_fetch=True,
        queries=[dict(issued_at=q.issued_at,received_at=q.received_at,valid_from=q.valid_from,accepted=q.accepted) for q in backend._queries]
    ), sort_keys=True))
finally:
    backend.close()
''')


def test_sealed_sim_runner_and_all_aliases_are_verified():
    native_script(PREAMBLE + '''
runtime = load_frozen_runtime(identity)
import sim, sys
assert sim.__file__.startswith(identity.root + '/source/')
assert runtime.runner.__file__.startswith(identity.root + '/source/')
runtime.check_origin()
assert 'pac_act.hybrid' not in sys.modules
assert 'residual_rl.rewards' not in sys.modules
assert runtime.runner.TRIAL_SPEC.force_N == 1.
assert runtime.formal.PROFILE['total_Z_limit_m'] == .002
print(runtime.runner.__file__)
''')


def test_preexisting_current_checkout_sim_is_rejected_before_loading():
    native_script(PREAMBLE + '''
import sim
try:
    load_frozen_runtime(identity)
except RuntimeError as error:
    assert 'another/unverified checkout' in str(error)
else:
    raise AssertionError('contaminated sim accepted')
''')


def test_extension_alias_cannot_replace_default_guard():
    native_script(PREAMBLE + '''
from types import ModuleType
import sys
bad = ModuleType('runner')
bad.__file__ = '/tmp/unsealed/runner.py'
sys.modules['runner'] = bad
try:
    load_frozen_runtime(identity)
except RuntimeError:
    pass
else:
    raise AssertionError('unsealed runner accepted')
''')


def test_multiple_backends_share_solver_lease_and_restore_original():
    native_script(BACKEND.rsplit('backend = FrozenSweepBackend.open', 1)[0] + '''
runtime = load_frozen_runtime(identity)
from sim.environments.ur10_interface import UR10CB3EndEffector
original = UR10CB3EndEffector._solve_ik
one = FrozenSweepBackend.open(identity, spec, budget=budget, native_lock=root/'.venv/task-5-native.lock')
two = None
try:
    two = FrozenSweepBackend.open(identity, spec, budget=budget, native_lock=root/'.venv/task-5-native.lock')
    one.close()
finally:
    one.close()
    if two is not None: two.close()
assert UR10CB3EndEffector._solve_ik is original, 'last lease did not restore original IK'
''')


def test_shared_limits_collection_window_and_safety_priority_after_real_steps():
    native_script(BACKEND + '''
import copy, mujoco
try:
    reset = backend.snapshot()
    # Inject only known clock counters near each cap; the final steps are real.
    cap = copy.deepcopy(reset)
    cap.state['origin'] = backend.env.time - 44.98
    backend.restore(cap)
    first = backend.step_target(None)
    assert first.done and first.termination_reason == '45-second shared episode timeout'
    assert len(first.actions) == 2
    backend.restore(cap)
    second = backend.step_target(None)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.termination_reason == second.termination_reason
    cap = copy.deepcopy(reset)
    cap.state['low_steps'], cap.state['frames'] = 4799, 1200
    backend.restore(cap)
    assert backend.step_target(None).termination_reason == '1200-frame shared episode timeout'
    backend.restore(reset)
    # Use actual body poses and native masks; this is terminal-label testing,
    # never a perceived-scene/online object source.
    bid = backend.env.component_body_ids[0]
    joint = backend.env.model.body_jntadr[bid]
    adr = backend.env.model.jnt_qposadr[joint]
    backend.env.data.qpos[adr:adr+2] = [-.35, 0.]
    mujoco.mj_forward(backend.env.model, backend.env.data)
    result = backend.step_target(None)
    assert result.done and result.termination_reason == 'exact target count reached'
    success = backend.snapshot()
    assert backend.guards['collection_timer'].first_entry_time is not None
    backend.restore(reset)
    # A native first-entry timer never resets after objects leave/re-enter.
    backend.guards['collection_timer'].first_entry_time = backend.env.time - 1.99
    backend.guards['force_gate'].steps = 2
    backend.env.ee._normal_force_filtered = 80.
    simultaneous = backend.step_target(None)
    assert simultaneous.done
    assert simultaneous.termination_reason == 'temporary hybrid safety: force ceiling or nonfinite force'
    assert backend.guards['collection_reason'] == '2-second collection window elapsed'
finally:
    backend.close()
''')


def test_empty_perceived_scene_cannot_fall_back_to_real_six_objects():
    native_script(BACKEND + '''
try:
    observations = [backend.observe(), backend.advance_visual_clock(), backend.advance_visual_clock()]
    class EmptyVisualScene:
        layout = []
        def component_positions(self): return np.empty((0,3))
        def component_collision_x_max(self): return 0.
    result = backend.prepare(EmptyVisualScene())
    assert not result.ready and result.termination_reason == 'no_plan'
    assert backend.frames == 10
    assert len(backend.env.component_body_ids) == 6
    try:
        backend.prepare(EmptyVisualScene())
    except RuntimeError:
        pass
    else:
        raise AssertionError('terminal no-plan attempt was replanned')
finally:
    backend.close()
''')


def test_visual_preparation_callbacks_keep_startup_5hz_clock():
    native_script(BACKEND + '''
try:
    backend.observe()
    backend.advance_visual_clock()
    backend.advance_visual_clock()
    seen = []
    def pause(observation, metadata):
        seen.append(observation.timestamp - backend.origin)
        return False
    result = backend.prepare_bootstrap_oracle(on_observation=pause)
    assert result.paused
    np.testing.assert_allclose(seen, [.6], atol=1e-8)
    assert backend.frames == 15
finally:
    backend.close()
''')


def test_startup_hold_propagates_real_native_robot_collision():
    native_script(BACKEND + '''
import mujoco
try:
    env = backend.env
    # The frozen profile has no collidable protected robot geoms. Designate a
    # real floor-contacting fixture geom as protected; keep the native contact
    # predicate and actual physics intact, without mocking a collision boolean.
    env.robot_geom_ids = {g for g in env.component_geom_ids[0]
                          if env.model.geom_contype[g] or env.model.geom_conaffinity[g]}
    mujoco.mj_forward(env.model, env.data)
    assert env.unsafe_robot_collision(), 'fixture did not create a physical robot contact'
    try:
        backend.step_target(None)
    except RuntimeError as error:
        assert 'unsafe robot collision' in str(error)
        assert backend.done and backend.phase == 'terminated'
        assert backend.termination_reason == str(error)
    else:
        raise AssertionError('startup hold omitted native collision safety')
finally:
    backend.close()
''')


def test_retained_cache_and_source_identity_audit():
    output = native_script(BACKEND.rsplit('backend = FrozenSweepBackend.open', 1)[0] + '''
from wam_sweep.artifacts import owned_size
from wam_sweep.inputs import file_hash
import inspect
runtime = load_frozen_runtime(identity)
cache = root/'.venv/task1-cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
info = cache.stat()
roots = policy['roots']
containing = [p for p in roots if Path(p) in cache.parents]
assert containing == [str(root/'.venv')]
single_inventory = owned_size(roots)
duplicate_inventory = owned_size([*roots, cache])
assert single_inventory == duplicate_inventory
account_state = json.loads(account.read_bytes())
charges = [e for e in account_state['events'] if e.get('kind')=='controller_task5_cache_audit_conservative_1s']
assert len(charges)==1 and charges[0]['seconds']==1
print(json.dumps(dict(
    cache=dict(path=str(cache),bytes=info.st_size,allocated_bytes=info.st_blocks*512,
               sha256=file_hash(cache),inode=info.st_ino,nlink=info.st_nlink,
               containing_policy_root=containing[0],single_inventory_bytes=single_inventory,
               inventory_with_repeated_cache_root_bytes=duplicate_inventory,
               cache_charge_bytes=owned_size([cache])),
    manifest_hash=identity.manifest_hash,model_hash=identity.model_hash,
    selected_loader=dict(path=str(Path(identity.root)/'source/pac_act/hybrid.py'),
                         sha256=identity.native_hashes['source/pac_act/hybrid.py'],
                         definitions=['SolverLease(module,cls)','load_trial4()']),
    modules={n:dict(path=m.__file__,sha256=file_hash(m.__file__)) for n,m in runtime.modules.items()},
    roots=roots, controller_audit_prior_charges=charges), sort_keys=True))
''')
    evidence = __import__('json').loads(output.splitlines()[-1])
    (Path.cwd()/'.venv/task-5-source-evidence.json').write_text(__import__('json').dumps(evidence, indent=2, sort_keys=True))


def test_preparation_private_contact_latch_and_provider_restore():
    native_script(BACKEND + '''
try:
    class Tracker:
        def __init__(self): self.ticks = []
        def snapshot(self): return dict(ticks=list(self.ticks))
        def restore(self, state): self.ticks = list(state['ticks'])
        def estimate(self, observation, metadata):
            self.ticks.append(observation.timestamp)
            return ()
    tracker = Tracker()
    backend.bind_provider('visual-tracker-fixture', tracker, beliefs=True)
    original = backend._runner.old.record_row
    latest = []
    def recorded(context, row, env, measurement, *, prefix=False):
        latest[:] = [context]
        return original(context, row, env, measurement, prefix=prefix)
    # Only this episode's isolated function-global namespace is changed.
    backend._runner.old.record_row = recorded
    def monitor(observation, metadata):
        if latest and 'z_limiter' in latest[0] and latest[0]['admittance'].phase in ('settle','maintain'):
            assert backend.native_act_input(observation.timestamp)['observation.state'][17] == 1.
            return False
    paused = backend.prepare_bootstrap_oracle(on_observation=monitor)
    assert paused.paused, 'did not reach an actual force-latched preparation capture'
    snapshot = backend.snapshot()
    expected_ticks = list(tracker.ticks)
    expected_input = backend.native_act_input(backend.env.time)['observation.state']
    tracker.ticks.append(-1.)
    backend.restore(snapshot)
    assert tracker.ticks == expected_ticks
    np.testing.assert_array_equal(backend.native_act_input(backend.observe().timestamp)['observation.state'], expected_input)
    import copy
    mismatched = copy.deepcopy(snapshot)
    mismatched.physics.end_effector['_normal_force_filtered'] += 3.
    backend.restore(mismatched)
    try:
        backend.resume_preparation()
    except RuntimeError as error:
        assert 'prefix replay diverged' in str(error)
    else:
        raise AssertionError('accepted equal qpos with an unreproducible force-filter cutpoint')
    backend.restore(snapshot)
    result = backend.resume_preparation()
    assert result.ready
    assert len(tracker.ticks) == len(set(tracker.ticks))
finally:
    backend.close()
''')


def test_handoff_and_sweep_snapshots_are_only_at_complete_frame_boundaries():
    native_script(BACKEND + '''
try:
    rejected = []
    def during_tick(truth):
        try:
            backend.snapshot()
        except RuntimeError as error:
            assert 'boundary' in str(error)
            rejected.append(truth.timestamp)
        else:
            raise AssertionError('snapshot claimed to preserve unfinished frame locals')
    backend.bind_offline_truth_tick(during_tick)
    backend.step_target(None)
    assert len(rejected) == 4
    assert backend.snapshot().cutpoint == 'prepare:frame-boundary'
finally:
    backend.close()
''')


def test_offline_visibility_uses_same_time_real_five_camera_segmentation():
    native_script(BACKEND + '''
import mujoco
try:
    env = backend.env
    # A real rendering fixture hidden below the tabletop/FOV; no guessed or
    # all-visible labels, and no policy preparation/outcome is performed.
    adr = env.model.jnt_qposadr[env.model.body_jntadr[env.component_body_ids[0]]]
    env.data.qpos[adr:adr+3] = [0., 0., -10.]
    mujoco.mj_forward(env.model, env.data)
    observation = backend.observe()
    truth = backend.capture_camera_truth(observation.timestamp)
    assert truth.timestamp == observation.timestamp
    assert truth.raw.fields['visibility_valid'].all()
    assert not truth.raw.fields['visibility'][0]
    assert truth.raw.fields['visibility'][1:].any()
    assert backend.observe().objects == ()
    for view in CAMERA_VIEWS:
        np.testing.assert_array_equal(backend.observe().images[view], observation.images[view])
    backend.step_target(None)
    assert not backend.capture_truth().raw.fields['visibility_valid'].any()
    try:
        backend.capture_camera_truth(observation.timestamp)
    except ValueError:
        pass
    else:
        raise AssertionError('stale pixels relabeled with current camera geometry')
finally:
    backend.close()
''')


def test_first_query_alignment_preserves_preparation_history_and_native_z():
    native_script(BACKEND + '''
from wam_sweep.history import HistoryBuffer
try:
    history = HistoryBuffer()
    for observation in (backend.observe(), backend.advance_visual_clock(), backend.advance_visual_clock()):
        history.append(observation)
    backend.bind_provider('causal-history', history)
    def monitor(observation, metadata):
        history.append(observation)
    prepared = backend.prepare_bootstrap_oracle(on_observation=monitor)
    assert prepared.ready
    before, before_steps = backend.episode_time, backend._low_steps
    z_controller = backend.context['admittance']
    observation = backend.advance_visual_clock()
    assert before < backend.episode_time <= before + .2 + 1e-8
    history.append(observation)  # exact Task1 grid check, no history reset
    assert int(history.batch().history_valid.sum()) == 11
    assert backend.context['admittance'] is z_controller
    assert backend.context['z_limiter'].last_time == backend.env.time
    assert backend.governor.last_time == backend.env.time
    assert backend.governor.started_at is None and not backend._queries
    assert 0 < backend._low_steps-before_steps <= 20
    start = backend.snapshot()
    first = backend.step_target(None)
    backend.restore(start)
    np.testing.assert_array_equal(first.actions, backend.step_target(None).actions)
finally:
    backend.close()
''')


def test_public_policy_binding_supports_precontact_features_and_first_query():
    # Catches a contact-gated binding, model recreation, different visual
    # preprocessing, or a second encoder pass hidden inside the ACT query.
    native_script(BACKEND + '''
import pickle, random, torch
from wam_sweep.snapshots import capture_random_state
handles = []
try:
    observation = backend.observe()
    before = backend.snapshot()
    rng = capture_random_state()
    cold = root/'.venv/task-5-cold-no-fetch'
    assert not cold.exists()
    torch.hub.get_dir = lambda: str(cold)
    import torchvision.models._api as vision_api
    def forbidden_download(*args, **kwargs):
        raise RuntimeError('external weight fetching prohibited')
    vision_api.load_state_dict_from_url = forbidden_download
    binding = backend.bind_frozen_policy()
    assert backend.bind_frozen_policy() is binding
    assert random.getstate() == rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    if rng[4] is not None:
        assert torch.equal(torch.mps.get_rng_state(), rng[4])
    assert not binding.policy.training
    assert not any(p.requires_grad for p in binding.policy.parameters())
    def forbidden_second_model(*args, **kwargs):
        raise RuntimeError('a second online policy is forbidden')
    type(binding.policy).from_pretrained = forbidden_second_model
    layers, inputs, forwards = [], [], []
    backbone = binding.policy.model.backbone
    for name in ('layer1', 'layer2', 'layer4'):
        handles.append(backbone[name].register_forward_hook(
            lambda module, args, output, name=name: layers.append((name, tuple(output.shape)))))
    handles.append(backbone.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().cpu().clone())))
    handles.append(binding.policy.model.register_forward_hook(
        lambda module, args, output: forwards.append(1)))
    # External Task6 features_only path: ONLY safe RGB through the SAME saved
    # preprocessor and backbone, without native36 or predict_action_chunk.
    images = {'observation.images.'+view: torch.from_numpy(observation.images[view].copy()).permute(2,0,1).float()/255.
              for view in CAMERA_VIEWS}
    batch = binding.preprocessor(images)
    # LeRobot also returns default transition metadata (action=None, etc.);
    # only its observation keys feed features_only, never private ACT state.
    assert {key for key in batch if key.startswith('observation.')} == set(images)
    with torch.no_grad():
        for view in CAMERA_VIEWS:
            backbone(batch['observation.images.'+view])
    expected = [('layer1',(1,64,80,80)), ('layer2',(1,128,40,40)), ('layer4',(1,512,10,10))] * 5
    assert layers == expected and not forwards and len(inputs) == 5
    after = backend.snapshot()
    np.testing.assert_array_equal(after.physics.integration, before.physics.integration)
    for key in ('scheduler','builder','guards','reference','commands','queries','frames','low_steps','phase','context','governor'):
        assert pickle.dumps(after.state[key]) == pickle.dumps(before.state[key]), key
    assert backend.context is None and backend.governor is None and not backend._queries
    # Real contact preparation, then FIRST ACT forward; no active-frame
    # features_only pass. Hook all three layers BEFORE this first query.
    assert backend.prepare_bootstrap_oracle().ready
    observation = backend.advance_visual_clock()
    layers.clear(); inputs.clear()
    receipt = backend.query_act(observation)
    assert receipt.accepted and layers == expected and len(inputs) == 5
    assert len(forwards) == 1 and backend.bind_frozen_policy() is binding
    # Query input pixels use the identical saved visual processor. This
    # comparison preprocesses only, never re-encodes an active frame.
    image_batch = binding.preprocessor({'observation.images.'+view:
        torch.from_numpy(observation.images[view].copy()).permute(2,0,1).float()/255. for view in CAMERA_VIEWS})
    for index, view in enumerate(CAMERA_VIEWS):
        assert torch.equal(inputs[index], image_batch['observation.images.'+view].cpu())
    assert len(layers) == 15 and len(forwards) == 1 and not cold.exists()
    (root/'.venv/task-5-shared-model-evidence.json').write_text(json.dumps(dict(
        precontact_physics_unchanged=True, precontact_scheduler_and_ack_unchanged=True,
        construction_rng_preserved=True, cold_no_fetch=True,
        feature_only_backbone_passes=5, feature_only_act_forwards=0,
        first_query_backbone_passes=5, first_query_act_forwards=1,
        first_query_layers=layers, same_policy_binding=True, saved_visual_preprocessing_equal=True)))
finally:
    for handle in handles: handle.remove()
    backend.close()
''')
