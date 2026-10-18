"""Physical restoration, including hidden state that equal qpos cannot prove."""
import copy
import importlib
import importlib.util
import random
import pickle
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch


def implementation():
    assert importlib.util.find_spec('wam_sweep.snapshots') is not None, 'full native snapshot implementation missing'
    return importlib.import_module('wam_sweep.snapshots')


@pytest.fixture
def contact_world():
    # A real actuated sliding shoe, initially touching the floor, with solver
    # warmstart and a causal filtered force/controller signal outside MjData.
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" iterations="3"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
       <body pos="0 0 .019"><joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 0 1"/>
        <geom type="box" size=".02 .02 .02" mass=".1"/>
       </body></worldbody><actuator><motor joint="0"/></actuator>
    </mujoco>'''.replace('<joint type="slide" axis="1 0 0"/>', '<joint name="x" type="slide" axis="1 0 0"/>').replace('joint="0"', 'joint="x"'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data, rng=np.random.default_rng(72),
                          last_raw_force_peak=None,
                          ee=SimpleNamespace(_normal_force_filtered=.3,
                                             tare=np.array([.1, .2]),
                                             cfg=None, model=model, data=data))
    return env


def execute(env):
    commands, contacts = [], []
    for _ in range(12):
        command = .04 - .02 * env.ee._normal_force_filtered
        env.data.ctrl[0] = command
        mujoco.mj_step(env.model, env.data)
        env.ee._normal_force_filtered = .8 * env.ee._normal_force_filtered + .2 * abs(env.data.qacc[1])
        commands.append([command, *env.data.qpos, *env.data.qvel])
        contacts.append(tuple((int(c.geom1), int(c.geom2), float(c.dist)) for c in env.data.contact))
    return np.asarray(commands), contacts


def test_snapshot_repeats_physical_commands(contact_world):
    api = implementation()
    env = contact_world
    root = api.capture_physics(env)
    first, contacts1 = execute(env)
    assert any(contacts1)
    api.restore_physics(env, root)
    second, contacts2 = execute(env)
    np.testing.assert_array_equal(first, second)
    assert contacts1 == contacts2


def test_equal_qpos_does_not_restore_force_filter_or_warmstart(contact_world):
    api = implementation()
    env = contact_world
    root = api.capture_physics(env)
    first, _ = execute(env)
    api.restore_physics(env, root)
    env.ee._normal_force_filtered += 7
    env.data.qacc_warmstart[:] += 10
    np.testing.assert_array_equal(env.data.qpos, root.integration[1:1 + env.model.nq])
    changed, _ = execute(env)
    assert not np.array_equal(first, changed)
    api.restore_physics(env, root)
    repeated, _ = execute(env)
    np.testing.assert_array_equal(first, repeated)


def test_rng_roundtrip_and_snapshot_is_independent(contact_world):
    api = implementation()
    env = contact_world
    root = api.capture_physics(env)
    expected = (random.random(), np.random.rand(), torch.rand(2), env.rng.normal())
    env.ee.tare[:] = 99
    api.restore_physics(env, root)
    actual = (random.random(), np.random.rand(), torch.rand(2), env.rng.normal())
    assert actual[0:2] == expected[0:2]
    assert torch.equal(actual[2], expected[2])
    assert actual[3] == expected[3]
    np.testing.assert_array_equal(env.ee.tare, [.1, .2])
    env.ee.tare[:] = 12
    api.restore_physics(env, root)
    np.testing.assert_array_equal(env.ee.tare, [.1, .2])


def test_physical_snapshot_can_be_persisted_without_renderer_handles(contact_world):
    api = implementation()
    root = api.capture_physics(contact_world)
    saved = pickle.loads(pickle.dumps(root))
    first, _ = execute(contact_world)
    api.restore_physics(contact_world, saved)
    second, _ = execute(contact_world)
    np.testing.assert_array_equal(first, second)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_random_stream_is_part_of_complete_restore(contact_world):
    api = implementation()
    torch.rand(2, device='mps')
    root = api.capture_physics(contact_world)
    first = torch.rand(5, device='mps').cpu()
    api.restore_physics(contact_world, root)
    second = torch.rand(5, device='mps').cpu()
    assert torch.equal(first, second)


def test_persistent_images_are_shared_verified_references(contact_world):
    api = implementation()
    assert hasattr(api, 'externalize_images'), 'persistent image-reference boundary missing'
    from wam_sweep.contracts import OnlineObservation, RobotContext, CAMERA_VIEWS
    robot = RobotContext(np.zeros(3), 0., np.zeros(3), 0., 'sweep', np.zeros(3), np.zeros(3), 0., True, True, 0., False)
    observation = OnlineObservation(0., (), robot, {view: np.full((320,320,3), j, np.uint8) for j, view in enumerate(CAMERA_VIEWS)})
    root = api.EpisodeSnapshot('sealed-fixture', api.capture_physics(contact_world),
                               {'capture': observation}, {'history': [observation]*11}, 'sweep:frame-boundary')
    images, writes = {}, []
    def put(digest, image):
        writes.append(digest)
        images[digest] = image.copy()
        return digest
    external = api.externalize_images(root, put)
    assert len(writes) == 5
    # Measure image-bearing channels separately: full MjData's solver caches
    # have an independent, necessary footprint even in this tiny contact model.
    assert len(pickle.dumps((external.state, external.participants))) < 32_000
    restored = api.materialize_images(pickle.loads(pickle.dumps(external)), images.__getitem__)
    for item in [restored.state['capture'], *restored.participants['history']]:
        for view in CAMERA_VIEWS:
            np.testing.assert_array_equal(item.images[view], observation.images[view])
    images[writes[0]][0,0,0] = 42
    with pytest.raises(ValueError, match='image'):
        api.materialize_images(external, images.__getitem__)
