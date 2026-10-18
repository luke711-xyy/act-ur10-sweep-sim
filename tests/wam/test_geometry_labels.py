"""Hand-derived collision envelopes; tiny real MuJoCo scenes, no native reset."""
import importlib.util
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("wam_sweep.truth"), "Task4 truth capture is absent"
    from wam_sweep import truth
    return truth


def scene(geoms, geometry="bolt", yaw=0., target=None, evidence=None):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0"/><worldbody>
      <body name="part" euler="0 0 {yaw}">{geoms}</body>
      <body name="brush" pos="-.1 0 0"><geom type="box" size=".01 .05 .01"/></body>
      </worldbody><compiler angle="radian"/></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    part = model.body("part").id
    brush = model.body("brush").id
    gids = np.flatnonzero(model.geom_bodyid == part).tolist()
    bgids = set(np.flatnonzero(model.geom_bodyid == brush).tolist())
    env = SimpleNamespace(model=model, data=data, component_body_ids=[part],
        component_geom_ids=[gids], brush_geom_ids=bgids,
        layout=[dict(geometry=geometry, nominal_radius=.0196)],
        ee=SimpleNamespace(tcp_yaw=lambda: 0., tip_geom_ids=bgids),
        cfg=SimpleNamespace(target=SimpleNamespace(**(target or dict(x_min=-1., x_max=1., y_min=-1., y_max=1.)))),
        collected_mask=lambda: np.array([True]),
        partial_collection_overlap_mask=lambda: np.array([False]))
    if evidence is not None:
        env.wam_truth_evidence = evidence
    return env


@pytest.mark.parametrize("geometry,shaft,shaft_r,head_offset,head_half,head_r,expected", [
    ("screw", .010, .0018, -.0115, .0015, .0037, [-.0015, .0115]),
    ("bolt", .014, .0028, -.0168, .0075, .0075, [-.00515, .01915]),
])
def test_offset_fastener_uses_actual_collision_union(geometry, shaft, shaft_r, head_offset, head_half, head_r, expected):
    env = scene(f'''<geom type="cylinder" size="{shaft_r} {shaft}" euler="0 1.5707963267948966 0"/>
      <geom type="cylinder" size="{head_r} {head_half}" pos="{head_offset} 0 0" euler="0 1.5707963267948966 0"/>
      <geom type="box" size="1 1 1" pos="2 0 0" contype="0" conaffinity="0"/>''', geometry)
    target = api().project_envelopes(api().capture_truth(env))
    np.testing.assert_allclose(target.center_xy[0], [expected[0], 0.], atol=1e-10)
    np.testing.assert_allclose(target.half_extents_xy[0], [expected[1], head_r], atol=1e-10)
    assert target.center_xy[0, 0] != env.data.xpos[env.component_body_ids[0], 0]


def test_rotated_offset_box_hand_bounds_and_pi_yaw_equivalence():
    targets = []
    for yaw in (np.pi / 2, 3 * np.pi / 2):
        frame = api().capture_truth(scene('<geom type="box" size=".02 .01 .003" pos="-.005 0 0"/>', yaw=yaw))
        targets.append(api().project_envelopes(frame))
    np.testing.assert_allclose(targets[0].center_xy[0], [0., -.005], atol=1e-10)
    np.testing.assert_allclose(targets[0].half_extents_xy[0], [.02, .01], atol=1e-10)
    np.testing.assert_allclose(targets[0].yaw_double[0], [0., -1.], atol=1e-10)
    np.testing.assert_allclose(targets[0].yaw_double, targets[1].yaw_double, atol=1e-10)


@pytest.mark.parametrize("geom,geometry,half", [
    ('<geom type="sphere" size=".02"/>', "cylinder", [.02, .02]),
    ('<geom type="cylinder" size=".02 .005"/>', "washer", [.02, .02]),
    ('<geom type="ellipsoid" size=".02 .01 .005"/>', "screw", [.02, .01]),
    ('<geom type="capsule" size=".003 .01" euler="0 1.5707963267948966 0"/>', "screw", [.013, .003]),
    ('<geom type="cylinder" size=".003 .01" euler="0 .7853981633974483 0"/>', "screw", [.013 / np.sqrt(2), .003]),
])
def test_primitive_projected_supports(geom, geometry, half):
    target = api().project_envelopes(api().capture_truth(scene(geom, geometry)))
    np.testing.assert_allclose(target.half_extents_xy[0], half, atol=1e-10)
    assert bool(target.yaw_valid[0]) == (geometry == "screw")


def test_capture_copies_evidence_and_preserves_rawtruth_public_identity():
    from wam_sweep import supervision
    from wam_sweep.contracts import supervision as contracts_supervision
    module = api()
    env = scene('<geom type="sphere" size=".02"/>', "cylinder")
    frame = module.capture_truth(env)
    assert module.RawTruth is supervision.RawTruth is contracts_supervision.RawTruth
    assert type(frame.raw) is supervision.RawTruth
    env.data.geom_xpos[:] = 100.
    targets = module.project_envelopes(frame)
    np.testing.assert_allclose(targets.center_xy, [[0., 0.]])
    assert not targets.velocity_valid.any()
    assert not targets.visibility_valid.any()
    assert not targets.visibility.any()


def test_native_nominal_masks_are_not_recomputed_from_collision_envelopes():
    env = scene('<geom type="box" size=".02 .01 .003" pos="-.005 0 0"/>',
                target=dict(x_min=-.021, x_max=.021, y_min=-.02, y_max=.02))
    targets = api().project_envelopes(api().capture_truth(env))
    assert targets.native_collected.tolist() == [True]
    assert targets.native_partial.tolist() == [False]
    assert targets.collision_inside.tolist() == [False]
    assert targets.collision_overlap.tolist() == [True]


def test_camera_tick_evidence_is_explicit_and_timestamp_aligned():
    env = scene('<geom type="sphere" size=".02"/>', "cylinder",
                evidence=dict(timestamp=0., visibility=[False], visibility_valid=[True]))
    targets = api().project_envelopes(api().capture_truth(env))
    assert targets.visibility_valid.tolist() == [True]
    assert targets.visibility.tolist() == [False]
    env.data.time = .01
    targets = api().project_envelopes(api().capture_truth(env))
    assert not targets.visibility_valid.any()


def test_real_contacts_capture_direct_and_indirect_graph_edges():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body name="brush"><freejoint/><geom type="sphere" size=".01"/></body>
      <body name="a" pos=".019 0 0"><freejoint/><geom type="sphere" size=".01"/></body>
      <body name="b" pos=".038 0 0"><freejoint/><geom type="sphere" size=".01"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    env = SimpleNamespace(model=model, data=data, component_body_ids=[2, 3],
        component_geom_ids=[[1], [2]], brush_geom_ids={0},
        layout=[dict(geometry="cylinder"), dict(geometry="cylinder")],
        ee=SimpleNamespace(tcp_yaw=lambda: 0.),
        collected_mask=lambda: np.zeros(2, bool),
        partial_collection_overlap_mask=lambda: np.zeros(2, bool))
    frame = api().capture_truth(env)
    assert set(frame.raw.fields["contacts"]) == {(-1, 0), (0, 1)}


def test_center_velocity_tracks_rotating_offset_geometry_not_body_origin():
    module = api()
    a = module.capture_truth(scene('<geom type="box" size=".02 .01 .003" pos="-.005 0 0"/>'))
    env = scene('<geom type="box" size=".02 .01 .003" pos="-.005 0 0"/>', yaw=np.pi / 2)
    env.data.time = .1
    b = module.capture_truth(env)
    targets = module.project_envelopes(b, previous=a)
    np.testing.assert_allclose(targets.velocity_xy, [[.05, -.05]], atol=1e-10)
    assert targets.velocity_valid.tolist() == [True]
    # A 90-degree pi-period jump has center-motion evidence but no signed yaw rate.
    assert not targets.angular_velocity_valid.any()


def test_collision_mesh_uses_transformed_compiled_vertices():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="tetra"
      vertex="0 0 0  .02 0 0  0 .01 0  0 0 .03"/></asset><worldbody>
      <body name="part"><geom type="mesh" mesh="tetra"/></body>
      <body name="brush" pos="-.1 0 0"><geom type="box" size=".01 .05 .01"/></body>
      </worldbody></mujoco>''')
    env = scene('<geom type="box" size=".01 .01 .01"/>', "screw")
    env.model, env.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, env.data)
    target = api().project_envelopes(api().capture_truth(env))
    np.testing.assert_allclose(target.center_xy[0], [.01, .005], atol=2e-9)
    np.testing.assert_allclose(target.half_extents_xy[0], [.01, .005], atol=2e-9)


def test_tilted_round_collision_changes_circle_scale_but_never_yaw_loss():
    env = scene('<geom type="cylinder" size=".02 .005" euler="0 1.5707963267948966 0"/>', "washer")
    target = api().project_envelopes(api().capture_truth(env))
    np.testing.assert_allclose(target.half_extents_xy, [[np.sqrt(.000425)] * 2], atol=1e-10)
    assert not target.yaw_valid.any()


def test_missing_predecessor_identity_invalidates_velocity_instead_of_borrowing():
    from dataclasses import replace
    module = api()
    f = module.capture_truth(scene('<geom type="sphere" size=".02"/>', "cylinder"))
    previous = replace(f, raw=module.RawTruth(dict(f.raw.fields, object_ids=(99,))))
    target = module.project_envelopes(replace(f, timestamp=.04), previous=previous)
    assert not target.velocity_valid.any()


def test_pi_half_turn_alias_boundary_has_no_signed_angular_velocity():
    module = api()
    first = module.capture_truth(scene('<geom type="box" size=".02 .01 .003"/>'))
    env = scene('<geom type="box" size=".02 .01 .003"/>', yaw=np.pi / 2)
    env.data.time = .04
    target = module.project_envelopes(module.capture_truth(env), previous=first)
    assert not target.angular_velocity_valid.any()
    assert not target.angular_velocity.any()
