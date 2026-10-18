"""Real sealed ResNet/ACT integration in a fresh source-guarded process."""
import importlib
from pathlib import Path
import os
import subprocess
import sys


def test_vision_import_does_not_contaminate_native_sources():
    body = '''
import sys
import wam_sweep.vision
from wam_sweep.inputs import NATIVE_NAMESPACES
assert not any(n.split('.')[0] in NATIVE_NAMESPACES for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', body], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_feature_tap_preserves_act_output():
    # Removing a layer tap, a second encoder, changing normalization/BN, or a
    # fetched initializer must fail this real model/physics boundary fixture.
    assert importlib.util.find_spec('wam_sweep.vision') is not None, 'Task6 vision package missing'
    body = r'''
import copy, json, pickle
from pathlib import Path
import numpy as np
import torch
from wam_sweep.inputs import verify_inputs, assert_native_origin
from wam_sweep.artifacts import BudgetLedger
from wam_sweep.config import WAMConfig
from wam_sweep.contracts import CAMERA_VIEWS, EpisodeSpec, OnlineObservation
from wam_sweep.history import HistoryBuffer
from wam_sweep.native import FrozenSweepBackend
from wam_sweep.snapshots import capture_random_state, restore_random_state
from wam_sweep.vision import FrozenFeatureTap, BEVEncoder, SixQueryDetector, VisualStateEstimator

root = Path.cwd()
identity = verify_inputs('/Users/huazhi_luke/.local/share/wam-sweep-v1/inputs/r8', allow_unlisted=False)
account = root/'.venv/task-2-budget.json'
policy = json.loads(account.read_bytes())['policy']
budget = BudgetLedger(account, owned_roots=policy['roots'], config=WAMConfig.load(root/'configs/wam_visual_v1.yaml').budget)
backend = FrozenSweepBackend.open(identity, EpisodeSpec('task6-bounded-fixture',2120018,1,'train',-1,'act',0),
                                 budget=budget, native_lock=root/'.venv/task-5-native.lock')
try:
    # Forbid fetching even if the accidentally retained initializer exists.
    import torchvision.models._api as vision_api
    def no_fetch(*args, **kwargs):
        raise AssertionError('external weights forbidden, including cached initializer')
    vision_api.load_state_dict_from_url = no_fetch
    cold = root/'.venv/task-6-cold-never-created'
    assert not cold.exists()
    torch.hub.get_dir = lambda: str(cold)
    tap = FrozenFeatureTap(backend)
    assert hasattr(tap,'capture_metadata'), 'retained RGB needs its own historical calibration provider'
    binding = backend.bind_frozen_policy()
    assert tap.binding is binding is backend.bind_frozen_policy()
    assert next(binding.policy.parameters()).device.type == 'mps'
    from safetensors.torch import load_file
    sealed = load_file(str(Path(identity.root)/'base_model/model.safetensors'))
    assert set(binding.policy.state_dict()) == set(sealed)
    for key, value in binding.policy.state_dict().items():
        assert torch.equal(value.cpu(), sealed[key]), key
    import draccus
    from lerobot.configs.policies import PreTrainedConfig
    assert draccus.encode(binding.policy.config, PreTrainedConfig) == json.loads((Path(identity.root)/'base_model/config.json').read_bytes())
    # No second online constructor after the shared binding.
    from sim.act.policy import require_act_dependencies
    policy_class = require_act_dependencies()[3]
    policy_class.from_pretrained = no_fetch
    counts = {'act':0,'backbone':0}
    handles = [binding.policy.model.register_forward_hook(lambda *args: counts.__setitem__('act',counts['act']+1)),
               binding.policy.model.backbone.register_forward_hook(lambda *args: counts.__setitem__('backbone',counts['backbone']+1))]
    raw = backend.observe()
    bev = BEVEncoder().to('mps')
    detector = SixQueryDetector().to('mps')
    estimator = VisualStateEstimator(bev, detector, identity, tap.capture_metadata)
    history = HistoryBuffer()
    for name, provider in [('features',tap),('estimator',estimator),('history',history)]:
        backend.bind_provider(name, provider, beliefs=False)
    packet = tap.features_only(raw)
    assert counts == {'act':0,'backbone':5} and not backend._queries
    assert set(vars(packet)) <= {'timestamp','spatial','input_hash','model_hash','preprocessing_hash','reference','receipt'}
    assert set(packet.spatial) == set(CAMERA_VIEWS)
    for view in CAMERA_VIEWS:
        assert [tuple(packet.spatial[view][l].shape) for l in ('layer1','layer2','layer4')] == [(1,64,80,80),(1,128,40,40),(1,512,10,10)]
    # Visual slice of the exact saved native processor, no state/goal to backbone.
    full = binding.preprocessor(backend.builder.torch_batch(backend.native_act_input(raw.timestamp)))
    safe = tap.preprocess_images(raw)
    for view in CAMERA_VIEWS:
        key = 'observation.images.'+view
        torch.testing.assert_close(full[key], safe[key], rtol=0, atol=0)
    beliefs = estimator.estimate(packet)
    assert len(beliefs) <= 6 and not any(b.valid for b in beliefs)
    history.append(OnlineObservation(raw.timestamp, beliefs, raw.robot, raw.images))
    initial = backend.snapshot()
    feature_state = tap.snapshot()
    assert not any(isinstance(v,torch.Tensor) for v in feature_state.values())
    # Restoring drops all transient pyramids; explicit replay uses RGB/backbone,
    # never a new/future ACT query. This bounded replay is charged by parent.
    backend.restore(initial)
    replay = tap.replay_features()
    for view in CAMERA_VIEWS:
        for layer in ('layer1','layer2','layer4'):
            torch.testing.assert_close(packet.spatial[view][layer], replay.spatial[view][layer], rtol=0, atol=0)
    assert not backend._queries
    # Repeat encoding the same live timestamp is rejected except explicit replay.
    try:
        tap.features_only(raw)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate online encoder pass')
    prepared = backend.prepare_bootstrap_oracle()
    assert prepared.ready
    raw = backend.observe()
    # Keep the same bound provider throughout preparation/query/restore. Save
    # exact input/RNG now, then remove hooks for the baseline AFTER integration.
    rng = capture_random_state()
    baseline_batch = binding.preprocessor(backend.builder.torch_batch(backend.native_act_input(raw.timestamp)))
    real_predict = binding.policy.predict_action_chunk
    actual = []
    def predict(batch):
        value = real_predict(batch)
        actual.append(value.clone())
        return value
    binding.policy.predict_action_chunk = predict
    seen_layers = {layer:[] for layer in ('layer1','layer2','layer4')}
    layer_handles = [binding.policy.model.backbone[layer].register_forward_hook(
        lambda module,args,out,layer=layer: seen_layers[layer].append(out.detach())) for layer in seen_layers]
    counts.update(act=0,backbone=0)
    packet2 = tap.query_act_and_features(raw)
    tapped_chunk = actual[-1].clone()
    assert counts == {'act':1,'backbone':5}
    active_counts = counts.copy()
    for layer,values in seen_layers.items():
        for view,value in zip(CAMERA_VIEWS,values):
            assert packet2.spatial[view][layer].data_ptr() == value.data_ptr()
    for handle in layer_handles: handle.remove()
    assert packet2.receipt.issued_at == raw.timestamp
    assert packet2.receipt.valid_from == raw.timestamp+.2
    np.testing.assert_array_equal(packet2.reference.targets,backend.reference_preview().targets)
    assert packet2.reference.source_hash == backend.reference_preview().source_hash
    old_metadata = backend.capture_metadata(raw.timestamp)
    # Snapshot contains real registered facade/estimator/history causal state;
    # RGB is content-addressed by the existing persistence boundary.
    images = {}
    persistent = backend.snapshot(image_writer=lambda digest,image: images.setdefault(digest,image.copy()) is not None and digest)
    assert len(images) <= 10
    backend.restore(persistent,image_reader=images.__getitem__)
    counts.update(act=0,backbone=0)
    replay_query = tap.replay_features()
    assert counts == {'act':0,'backbone':5}
    for view in CAMERA_VIEWS:
        for layer in ('layer1','layer2','layer4'):
            torch.testing.assert_close(packet2.spatial[view][layer],replay_query.spatial[view][layer],rtol=0,atol=0)
    # A held fixed-reference branch uses image-only extraction without future
    # ACT queries. Actually step five25Hz native frames, no simulated clock jump.
    backend.accept_act_chunk(np.tile(backend.reference,(25,1)),issued_at=backend.env.time-.2,received_at=backend.env.time)
    query_count = len(backend._queries)
    for _ in range(5):
        assert not backend.step_target(None).done
    fixed_raw = backend.observe()
    # Native retains only current capture. The feature facade must still pair
    # its older retained RGB with the exact OLD wrist transform for replay.
    old_from_tap = tap.capture_metadata(packet2.timestamp)
    for view in CAMERA_VIEWS:
        np.testing.assert_array_equal(old_from_tap.cameras[view].camera_to_world,old_metadata.cameras[view].camera_to_world)
    try:
        backend.capture_metadata(packet2.timestamp)
    except ValueError:
        pass
    else:
        raise AssertionError('fixture did not advance native capture')
    counts.update(act=0,backbone=0)
    old_replay = tap.replay_features()
    assert old_replay.timestamp == packet2.timestamp and counts == {'act':0,'backbone':5}
    counts.update(act=0,backbone=0)
    fixed = tap.features_only(fixed_raw)
    assert counts == {'act':0,'backbone':5} and len(backend._queries) == query_count
    assert fixed.receipt is None and fixed.reference is None
    assert old_metadata.timestamp == packet2.timestamp
    # Baseline really has no feature hooks, SAME saved input, model and RNG.
    tap.close()
    restore_random_state(rng)
    with torch.no_grad():
        baseline = real_predict(baseline_batch).clone()
    torch.testing.assert_close(tapped_chunk,baseline,rtol=0,atol=0)
    for key, value in binding.policy.state_dict().items():
        assert torch.equal(value.cpu(), sealed[key]), key
    assert not binding.policy.training and not any(p.requires_grad for p in binding.policy.parameters())
    assert not cold.exists()
    assert_native_origin(identity)
    (root/'.venv/task-6-feature-evidence.json').write_text(json.dumps(dict(
        shared_model=True, checkpoint_tensor_count=len(sealed), output_equal_bitwise=True,
        active_forwards=active_counts, same_forward_storage_identity=True, fixed_branch_no_act=True,
        persisted_distinct_images=len(images), registered_provider_restore=True, historical_calibration_retained=True,
        shape_per_view={l:list(packet2.spatial[CAMERA_VIEWS[0]][l].shape) for l in ('layer1','layer2','layer4')},
        input_identity=identity.__dict__ if hasattr(identity,'__dict__') else str(identity),
        device=str(next(binding.policy.parameters()).device), dtype=str(next(binding.policy.parameters()).dtype),
        cold_no_fetch=True, timestamp=raw.timestamp, feature_snapshot_bytes=len(pickle.dumps(feature_state)))))
    for handle in handles: handle.remove()
finally:
    backend.close()
'''
    result = subprocess.run([sys.executable, '-c', body], cwd=Path(__file__).resolve().parents[2],
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=110)
    assert result.returncode == 0, result.stdout
