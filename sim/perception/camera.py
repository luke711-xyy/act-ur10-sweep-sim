"""Pinhole camera model and image <-> table-plane conversion.

This module is pure NumPy: it contains the geometry needed to turn an image
mask into table-plane coordinates and is therefore unit-testable without MuJoCo
or a GPU.

Conventions match MuJoCo's camera:

* the camera looks along its local **-Z**, with local **+Y up** and **+X right**
* ``fovy`` is the *vertical* field of view in degrees
* pixel ``(row, col)`` has ``row`` increasing downwards from the top of the image
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    position: np.ndarray        # (3,) world position
    rotation: np.ndarray        # (3, 3) columns are the camera x/y/z axes in world
    width: int
    height: int
    fovy_deg: float

    # -- intrinsics ---------------------------------------------------------
    @property
    def focal_px(self) -> float:
        return (self.height / 2.0) / np.tan(np.deg2rad(self.fovy_deg) / 2.0)

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def intrinsics(self) -> np.ndarray:
        f = self.focal_px
        return np.array([[f, 0.0, self.cx], [0.0, f, self.cy], [0.0, 0.0, 1.0]])

    # -- projection ---------------------------------------------------------
    def ray_directions(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """World-frame unit ray directions for pixel coordinates ``(rows, cols)``."""
        rows = np.atleast_1d(np.asarray(rows, dtype=float))
        cols = np.atleast_1d(np.asarray(cols, dtype=float))
        f = self.focal_px
        x_cam = (cols - self.cx) / f
        y_cam = -(rows - self.cy) / f
        z_cam = -np.ones_like(x_cam)
        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        dirs_world = dirs_cam @ np.asarray(self.rotation).T
        norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
        return dirs_world / np.maximum(norms, 1e-12)

    def pixels_to_plane(self, rows: np.ndarray, cols: np.ndarray,
                        plane_z: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane ``z = plane_z``.

        Returns an ``(N, 3)`` array; rays that do not hit the plane in front of
        the camera are returned as NaN.
        """
        dirs = self.ray_directions(rows, cols)
        origin = np.asarray(self.position, dtype=float)
        denom = dirs[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (float(plane_z) - origin[2]) / denom
        points = origin[None, :] + t[:, None] * dirs
        bad = (~np.isfinite(t)) | (t <= 0.0) | (np.abs(denom) < 1e-9)
        points[bad] = np.nan
        return points

    def optical_axis_depth(self, points_world: np.ndarray) -> np.ndarray:
        """Depth along the camera's optical axis (the convention MuJoCo renders).

        MuJoCo's depth buffer stores the z-distance in the camera frame, not the
        Euclidean ray length, so this is what a rendered depth image must be
        compared against.
        """
        pts = np.atleast_2d(np.asarray(points_world, dtype=float))
        rel = pts - np.asarray(self.position, dtype=float)[None, :]
        forward = -np.asarray(self.rotation)[:, 2]
        return rel @ forward

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """World points -> ``(N, 2)`` pixel ``(row, col)``; NaN when behind the camera."""
        pts = np.atleast_2d(np.asarray(points_world, dtype=float))
        rel = pts - np.asarray(self.position, dtype=float)[None, :]
        cam = rel @ np.asarray(self.rotation)          # world -> camera frame
        depth = -cam[:, 2]
        f = self.focal_px
        with np.errstate(divide="ignore", invalid="ignore"):
            cols = cam[:, 0] / depth * f + self.cx
            rows = -cam[:, 1] / depth * f + self.cy
        out = np.stack([rows, cols], axis=1)
        out[depth <= 1e-6] = np.nan
        return out

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_config(cls, cam_cfg) -> "PinholeCamera":
        from ..model.scene_builder import camera_xyaxes

        xyaxes = camera_xyaxes(cam_cfg.pos, cam_cfg.lookat, cam_cfg.up)
        x_axis, y_axis = xyaxes[:3], xyaxes[3:]
        z_axis = np.cross(x_axis, y_axis)
        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        return cls(
            position=np.asarray(cam_cfg.pos, dtype=float),
            rotation=rotation,
            width=int(cam_cfg.width),
            height=int(cam_cfg.height),
            fovy_deg=float(cam_cfg.fovy_deg),
        )

    @classmethod
    def from_env(cls, env) -> "PinholeCamera":
        pos, rot = env.camera_pose()
        cam_cfg = env.cfg.perception.camera
        return cls(position=pos, rotation=rot, width=int(cam_cfg.width),
                   height=int(cam_cfg.height), fovy_deg=float(cam_cfg.fovy_deg))
