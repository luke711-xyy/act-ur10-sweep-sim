"""Real native scheduler parity; no ACT weights or physical execution."""
import importlib

import numpy as np
import pytest

from residual_rl.features import FeatureMailbox
from sim.act.realtime import ActionChunkScheduler
from sim.config import load_config


def observed(box, *, coefficient=.01, allow_late=False):
    module = importlib.import_module('residual_rl.features')
    factory = getattr(module, 'observed_scheduler', None)
    assert callable(factory), 'temporal-ensemble observation adapter missing'
    cfg = load_config()
    cfg.act.temporal_ensemble_coeff = coefficient
    cfg.act.query_hz = 5.
    cfg.act.action_hz = 25.
    cfg.act.execute_steps = 5
    cfg.act.realtime_budget_seconds = .16
    return ActionChunkScheduler(cfg, allow_late=allow_late), factory(ActionChunkScheduler, box)(cfg, allow_late=allow_late)


def tokens(offset):
    return np.broadcast_to((offset+np.arange(25, dtype=np.float32))[:, None], (25, 512)).copy()


def actions(offset):
    result = np.zeros((25, 4), np.float32)
    result[:, :3] = (offset+np.arange(25))[:, None]
    result[:, 3] = 3.1 if offset == 0 else -3.1
    return result


def accept_pair(base, adapter, box, issued, now, offset):
    box.publish(tokens(offset))
    expected = base.accept(issued, actions(offset), now)
    assert adapter.accept(issued, actions(offset), now) == expected


def test_overlapping_native_outputs_are_bit_exact_and_features_use_same_causal_weights():
    box = FeatureMailbox()
    base, adapter = observed(box)
    base.reset(0.); adapter.reset(0.)
    accept_pair(base, adapter, box, 0., .16, 0)
    for t in (.16, .20):
        assert np.array_equal(adapter.action_for(t), base.action_for(t))
    accept_pair(base, adapter, box, .2, .36, 100)
    assert np.array_equal(adapter.action_for(.4), base.action_for(.4))
    view = box.current(.4)
    # Native candidates are old index10 and new index5, with exp(-.1), exp(-.05).
    old, new = np.exp(-.1), np.exp(-.05)
    assert np.allclose(view['feature'], (old*10+new*105)/(old+new), atol=2e-6)
    assert view['age'] == pytest.approx((old*.4+new*.2)/(old+new))
    assert view['index'] == pytest.approx((old*10+new*5)/(old+new))
    assert view['available'] and len(view['components']) == 2
    assert adapter.temporal_ensemble.coeff == .01
    assert adapter.chunk is None  # Never change native scheduling state for observability.
    assert adapter.observation_chunk.issued_at == .2
    assert np.array_equal(adapter.observation_chunk.actions, actions(100))


def test_unaccepted_not_yet_valid_and_expired_features_never_reach_actor():
    box = FeatureMailbox()
    base, adapter = observed(box)
    accept_pair(base, adapter, box, 0., .16, 0)
    accept_pair(base, adapter, box, .2, .2, 100)
    assert np.array_equal(adapter.action_for(.24), base.action_for(.24))
    assert np.array_equal(box.current(.24)['feature'], tokens(0)[6])
    # Reject a late result, then prove that its tokens do not contribute.
    accept_pair(base, adapter, box, .4, .8, 1000)
    assert np.array_equal(adapter.action_for(.8), base.action_for(.8))
    assert {c['issued_at'] for c in box.current(.8)['components']} == {0., .2}
    assert adapter.action_for(1.3) is base.action_for(1.3) is None
    assert box.current(1.3)['available'] is False
    assert adapter.observation_chunk is None


@pytest.mark.parametrize('coefficient', [None, 0.])
def test_no_ensemble_path_stays_bit_exact_and_reset_clears_old_features(coefficient):
    box = FeatureMailbox()
    base, adapter = observed(box, coefficient=coefficient)
    accept_pair(base, adapter, box, 0., .16, 0)
    assert np.array_equal(adapter.action_for(.2), base.action_for(.2))
    assert np.array_equal(box.current(.2)['feature'], tokens(0)[5])
    adapter.reset(2.); base.reset(2.)
    assert adapter.action_for(2.) is base.action_for(2.) is None
    assert box.current(2.)['available'] is False
    assert not box.accepted and box.pending is None


def test_preview_late_acceptance_and_eight_chunk_eviction_keep_native_outputs():
    box = FeatureMailbox()
    base, adapter = observed(box, allow_late=True)
    for i in range(12):
        issued = i*.04
        accept_pair(base, adapter, box, issued, issued+.2, i*100)
        now = issued+.2
        expected = base.action_for(now)
        assert np.array_equal(adapter.action_for(now), expected)
        view = box.current(now)
        assert view['available']
        assert all(c['issued_at'] in box.accepted for c in view['components'])
        assert len(box.accepted) <= 8
    assert adapter.late_results == base.late_results == 12


def test_missing_tokens_for_any_contributing_native_chunk_fail_closed():
    box = FeatureMailbox()
    base, adapter = observed(box)
    accept_pair(base, adapter, box, 0., .16, 0)
    del box.accepted[0.]
    with pytest.raises(RuntimeError, match='chunk mismatch'):
        adapter.action_for(.2)


def test_new_preflight_seed_schedule_cannot_reuse_previous_consumed_or_planned_layouts():
    from residual_rl.pilot import seed_schedule
    schedule = seed_schedule()
    seeds = {s['seed'] for s in schedule['preflight']}
    assert not seeds & {3150000, 3150001, 3150100, 3150101}
    assert len(seeds) == 4


def test_prior_consumed_layout_is_excluded_even_under_a_new_seed():
    from residual_rl.runtime import LayoutRegistry
    registry = LayoutRegistry([], prior_consumed=[dict(seed=3150000, layout_fingerprint='consumed')])
    with pytest.raises(ValueError, match='consumed'):
        registry.register(3150000, 'other', 'preflight_native')
    with pytest.raises(ValueError, match='consumed'):
        registry.register(3151000, 'consumed', 'preflight_native')
    registry.register(3151000, 'fresh', 'preflight_native')


def test_previous_attempt_exclusion_is_measured_and_sealed_not_reexecuted(tmp_path):
    from residual_rl import pilot
    reader = getattr(pilot, 'prior_consumed_layouts', None)
    assert callable(reader), 'previous-attempt exclusion reader missing'
    previous = tmp_path/'stopped'
    (previous/'attempts').mkdir(parents=True)
    episode = previous/'episodes/native'
    episode.mkdir(parents=True)
    import hashlib, json
    poses = np.zeros((6, 7), dtype='<f4')
    fingerprint = hashlib.sha256(poses.tobytes()).hexdigest()
    marker = previous/'attempts/native.json'
    marker.write_text(json.dumps(dict(episode_id='native', seed=3150000, target_count=1)))
    layout = episode/'layout.json'
    layout.write_text(json.dumps(dict(poses=poses.tolist(), layout_fingerprint=fingerprint)))
    excluded = reader(previous)
    assert len(excluded) == 1 and excluded[0]['seed'] == 3150000
    assert excluded[0]['layout_fingerprint'] == fingerprint
    assert excluded[0]['source_sha256'][str(marker)] == hashlib.sha256(marker.read_bytes()).hexdigest()
    poses[0, 0] = 1.
    layout.write_text(json.dumps(dict(poses=poses.tolist(), layout_fingerprint=fingerprint)))
    with pytest.raises(RuntimeError, match='fingerprint'):
        reader(previous)


def test_actual_strict_handoff_adapter_installs_native_ensemble_observer_without_loading_model():
    import subprocess, sys
    from pathlib import Path
    result = subprocess.run([sys.executable, '-B', '-c', '''
from types import SimpleNamespace
import numpy as np
from residual_rl.runtime import PROJECT,identity,load_formal,install_driver,PolicyCache
from residual_rl.baseline import pre_dino_sources
from residual_rl.features import FeatureMailbox
from residual_rl.control import ResidualGovernor
from residual_rl.shadow import ZeroResidualShadow
f=identity(); pre_dino_sources(PROJECT,f['source_fingerprints']).activate()
from sim.config import load_config
formal=load_formal(); module=formal.runtime()
cfg=formal.candidate_config(load_config(str(formal.OLD/'evaluation_config.yaml')),1)
env=SimpleNamespace(control_dt=.01,time=5.)
c=dict(env=env,admittance=object(),force_gate=object(),collection_timer=object(),
       trace=[dict(t=4.99,command=np.zeros(4)),dict(t=5.,command=np.zeros(4))],
       last_reference=np.zeros(4),act_trace=[])
collector=SimpleNamespace(mailbox=FeatureMailbox(),env=env,mode='native',
    control=ResidualGovernor([[cfg.workspace.x_min,cfg.workspace.x_max],
                              [cfg.workspace.y_min,cfg.workspace.y_max]]))
collector.shadow=ZeroResidualShadow(collector.control.bounds)
collector.is_active=lambda: (collector.governor is not None and
    collector.governor.started_at is not None and env.time-collector.governor.started_at>=2.-1e-9)
frames=[]
collector.begin=lambda nominal:frames.append(np.asarray(nominal).copy())
cache=PolicyCache()
driver=install_driver(formal,module,cfg,c,collector,cache)
scheduler=driver.__globals__['ActionChunkScheduler'](cfg,allow_late=True)
scheduler.reset(4.8)
collector.mailbox.publish(np.ones((25,512),np.float32))
accepted=scheduler.accept(4.8,np.zeros((25,4),np.float32),5.)
assert accepted
driver.__globals__['_trial_chunk_ack'](accepted)
assert np.array_equal(scheduler.action_for(5.),np.zeros(4,np.float32))
assert scheduler is collector.scheduler
assert scheduler.temporal_ensemble.coeff==.01
assert collector.mailbox.current(5.)['available']
assert cache.bundle is None
# Exercise the REAL installed handoff and native reference, but no policy or
# physics. Acknowledgement advances only the synthetic test clock/state.
reference=driver.__globals__['_policy_reference_substep']
governor=collector.governor
proposals=[]
propose=governor.propose
def counted(*args,**kwargs):
    proposals.append(float(env.time))
    return propose(*args,**kwargs)
governor.propose=counted
controller=c['admittance']; gate=c['force_gate']
last=np.zeros(4,np.float32)
for i in range(208):
    if i%4==0:
        start=last.copy(); target=start+np.array([.002,.001,0.,.01],np.float32)
    z=.001+i*1e-8
    command=reference(start,target,i%4+1,4,.01,.08,.1,.35,True,z)
    # Exactly one native handoff proposal per command; shadow does not mutate it.
    assert len(proposals)==i+1
    assert np.array_equal(command,governor.pending['command'])
    if governor.pending['mode']=='native_restored':
        assert command[2]==governor.pending['native_reference'][2]
    else: assert command[2]==z  # Original handoff retains float64 Z before trace rounding.
    last=np.asarray(command,dtype=np.float32)
    row=dict(t=env.time+.01,command=last,control_step_advanced=True)
    governor.prepare_record(row); governor.acknowledge(row)
    env.time=row['t']
proof=collector.shadow.summary()
assert proof['checked_steps']==208 and proof['active_checked_steps']==8
assert proof['native_sha256']==proof['zero_sha256'] and proof['mismatches']==0
assert len(frames)==52 and not collector.control.velocity.any()
assert c['admittance'] is controller and c['force_gate'] is gate
assert cache.bundle is None  # No model loading/query or environment.step call.
assert len(collector.mailbox.accepted)==1
print('strict handoff and same-stream shadow verified, no model or physical execution')
'''], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
