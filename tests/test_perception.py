"""Unit tests for the camera model, occupancy grid and perception plumbing."""

import numpy as np
import pytest

from sim.config import load_config
from sim.model.scene_builder import camera_xyaxes
from sim.perception.base import GridSpec, SceneObservation, build_perception
from sim.perception.camera import PinholeCamera


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def camera(cfg):
    return PinholeCamera.from_config(cfg.perception.camera)


def test_camera_frame_is_orthonormal_and_faces_the_lookat(cfg, camera):
    assert np.allclose(camera.rotation.T @ camera.rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(camera.rotation) == pytest.approx(1.0)
    forward = -camera.rotation[:, 2]
    expected = np.asarray(cfg.perception.camera.lookat) - np.asarray(cfg.perception.camera.pos)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(forward, expected, atol=1e-9)


def test_pixel_to_plane_round_trip_is_exact(camera):
    world = np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [-0.4, -0.25, 0.0], [0.42, 0.3, 0.0]])
    pixels = camera.project(world)
    back = camera.pixels_to_plane(pixels[:, 0], pixels[:, 1], plane_z=0.0)
    assert np.allclose(back, world, atol=1e-9)


def test_principal_ray_hits_the_lookat_point(cfg, camera):
    point = camera.pixels_to_plane([camera.cy], [camera.cx], plane_z=0.0)[0]
    assert np.allclose(point[:2], np.asarray(cfg.perception.camera.lookat)[:2], atol=1e-9)


def test_plane_offset_shifts_the_back_projection_towards_the_camera(camera):
    at_table = camera.pixels_to_plane([300.0], [320.0], plane_z=0.0)[0]
    above = camera.pixels_to_plane([300.0], [320.0], plane_z=0.01)[0]
    assert np.linalg.norm(above - camera.position) < np.linalg.norm(at_table - camera.position)


def test_rays_that_miss_the_plane_are_nan(camera):
    """A downward-looking ray can never reach a plane above the camera."""
    above_camera = float(camera.position[2]) + 2.0
    missed = camera.pixels_to_plane([0.0], [camera.cx], plane_z=above_camera)
    assert np.isnan(missed).all()


def test_camera_xyaxes_matches_mujoco_convention():
    xyaxes = camera_xyaxes([1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    x_axis, y_axis = xyaxes[:3], xyaxes[3:]
    assert np.dot(x_axis, y_axis) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(x_axis) == pytest.approx(1.0)
    assert y_axis[2] > 0.0                       # camera "up" has a +Z component


def test_grid_indexing_round_trip(cfg):
    grid = GridSpec(-0.42, 0.46, -0.32, 0.32, res=0.01)
    xy = np.array([[0.0, 0.0], [0.30, -0.20], [-0.41, 0.31]])
    iy, ix, valid = grid.to_index(xy)
    assert valid.all()
    centres = grid.to_xy(iy, ix)
    assert np.all(np.abs(centres - xy) <= grid.res)


def test_grid_rejects_points_outside(cfg):
    grid = GridSpec.from_config(cfg, res=0.01)
    _, _, valid = grid.to_index(np.array([[5.0, 5.0], [0.0, 0.0]]))
    assert list(valid) == [False, True]


def test_rasterize_disks_marks_a_connected_blob(cfg):
    grid = GridSpec(-0.1, 0.1, -0.1, 0.1, res=0.005)
    occupancy = grid.rasterize_disks(np.array([[0.0, 0.0]]), 0.02)
    assert occupancy.any()
    ys, xs = np.nonzero(occupancy)
    centres = grid.to_xy(ys, xs)
    assert np.all(np.linalg.norm(centres, axis=1) <= 0.02 + grid.res)


def test_scene_observation_summary_fields(cfg):
    grid = GridSpec.from_config(cfg)
    obs = SceneObservation(t=0.0, points=np.array([[0.1, 0.0], [0.2, 0.0]]),
                           counts=np.array([1.0, 3.0]), areas=np.array([8e-5, 2.4e-4]),
                           occupancy=grid.empty(), grid=grid)
    assert obs.n_detected == 2
    assert obs.estimated_total == pytest.approx(4.0)


def test_build_perception_dispatch(cfg):
    assert build_perception(cfg, "ground_truth").name == "ground_truth"
    assert build_perception(cfg, "vision").name == "vision"
    with pytest.raises(ValueError):
        build_perception(cfg, "nope")
