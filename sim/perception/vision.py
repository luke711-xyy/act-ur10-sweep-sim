"""Conventional (non-learned) vision backend.

Pipeline
--------
1. render one image from the **fixed** high-oblique camera (the same camera
   configuration is used for every episode and for dataset export)
2. produce a binary component mask in image space using either
   * MuJoCo segmentation ids  (``perception.segmentation: mujoco``)
   * a brightness/colour threshold (``color``)
   * a height-above-the-table test on the depth image (``depth``)
3. back-project every masked pixel onto the table plane (ray/plane intersection)
4. rasterise the resulting points into a table-plane occupancy grid
5. label connected occupied regions into *clusters* with a centroid, an area and
   an estimated component count

No semantic classification of screws/bolts/nuts is attempted: the planner only
needs to know **where occupied regions are**.  Optional sensor noise (position
jitter, dropout, false positives) is applied at the end so that the planner can
be stress-tested against imperfect perception.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from ..environments.layout import in_target_region
from .base import Perception, SceneObservation
from .camera import PinholeCamera


class ConventionalVisionPerception(Perception):
    name = "vision"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mode = str(cfg.perception.segmentation)
        self.camera: Optional[PinholeCamera] = None

    # ------------------------------------------------------------------ main
    def observe(self, env, rng: Optional[np.random.Generator] = None) -> SceneObservation:
        rng = rng if rng is not None else np.random.default_rng(0)
        self.camera = PinholeCamera.from_env(env)
        pcfg = self.cfg.perception

        rgb = env.render_rgb()
        depth = env.render_depth() if self.mode == "depth" else None
        mask = self._image_mask(env, rgb, depth)
        mask = self._despeckle(mask, int(pcfg.min_blob_pixels))

        plane_z = float(env.table_top_z) + float(pcfg.get("plane_offset", 0.003))
        points = self._mask_to_plane(mask, plane_z)
        occupancy = self._rasterize(points)
        occupancy = self._suppress_regions(occupancy, env)

        centroids, areas, counts = self._cluster(occupancy)
        centroids, areas, counts = self._apply_noise(centroids, areas, counts, rng)

        positions = env.component_positions()
        remaining_true = int((~in_target_region(positions[:, :2], env.cfg.target)).sum()) \
            if positions.size else 0

        return SceneObservation(
            t=env.time,
            points=centroids,
            counts=counts,
            areas=areas,
            occupancy=occupancy,
            grid=self.grid,
            tcp=env.tcp(),
            backend=self.name,
            rgb=rgb,
            depth=depth,
            mask=mask,
            n_remaining_true=remaining_true,
        )

    # ------------------------------------------------------------ mask making
    def _image_mask(self, env, rgb: np.ndarray, depth: Optional[np.ndarray]) -> np.ndarray:
        if self.mode == "mujoco":
            seg = env.render_segmentation()
            comp_geoms = np.array(sorted(env.geom_to_component.keys()), dtype=np.int32)
            return np.isin(seg, comp_geoms)

        if self.mode == "color":
            img = np.asarray(rgb, dtype=float) / 255.0
            gray = img @ np.array([0.299, 0.587, 0.114])
            bluish = (img[:, :, 2] - img[:, :, 0]) > 0.08       # the blue tray
            threshold = float(self.cfg.perception.get("color_threshold", 0.55))
            return (gray > threshold) & (~bluish)

        if self.mode == "depth":
            if depth is None:
                raise RuntimeError("depth segmentation requested but no depth image rendered")
            expected = self._expected_table_depth(float(env.table_top_z), depth.shape)
            height_above_table = expected - np.asarray(depth, dtype=float)
            return height_above_table > float(self.cfg.perception.depth_threshold)

        raise ValueError(f"unknown perception.segmentation {self.mode!r}")

    def _expected_table_depth(self, plane_z: float, shape: Tuple[int, int]) -> np.ndarray:
        """Per-pixel depth of the *empty* table plane, in MuJoCo's convention.

        MuJoCo's renderer returns the camera-frame z-distance, so the reference
        plane depth is computed the same way (``optical_axis`` by default).
        Rays that never reach the plane get ``inf`` so they are never segmented.
        """
        assert self.camera is not None
        rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        pts = self.camera.pixels_to_plane(rows.ravel(), cols.ravel(), plane_z=plane_z)
        convention = str(self.cfg.perception.get("depth_convention", "optical_axis"))
        if convention == "euclidean":
            dist = np.linalg.norm(pts - np.asarray(self.camera.position)[None, :], axis=1)
        else:
            dist = self.camera.optical_axis_depth(pts)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        return dist.reshape(shape)

    @staticmethod
    def _despeckle(mask: np.ndarray, min_pixels: int) -> np.ndarray:
        if min_pixels <= 1:
            return mask
        labels, n = ndimage.label(mask)
        if n == 0:
            return mask
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        keep = np.concatenate([[False], sizes >= min_pixels])
        return keep[labels]

    # -------------------------------------------------------- plane / grid ops
    def _mask_to_plane(self, mask: np.ndarray, plane_z: float) -> np.ndarray:
        assert self.camera is not None
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return np.zeros((0, 2))
        pts = self.camera.pixels_to_plane(rows, cols, plane_z=plane_z)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts[:, :2]

    def _rasterize(self, points: np.ndarray) -> np.ndarray:
        grid = self.grid.empty()
        if points.size == 0:
            return grid
        iy, ix, valid = self.grid.to_index(points)
        grid[iy[valid], ix[valid]] = True
        return grid

    def _suppress_regions(self, occupancy: np.ndarray, env) -> np.ndarray:
        """Drop occupancy inside the target tray (already collected components)."""
        tgt = env.cfg.target
        iy_all, ix_all = np.meshgrid(np.arange(self.grid.ny), np.arange(self.grid.nx),
                                     indexing="ij")
        centres = self.grid.to_xy(iy_all, ix_all)
        inside_target = (
            (centres[..., 0] >= float(tgt.x_min) - 0.01)
            & (centres[..., 0] <= float(tgt.x_max) + 0.01)
            & (centres[..., 1] >= float(tgt.y_min) - 0.01)
            & (centres[..., 1] <= float(tgt.y_max) + 0.01)
        )
        return occupancy & (~inside_target)

    # ------------------------------------------------------------- clustering
    def _cluster(self, occupancy: np.ndarray):
        structure = np.ones((3, 3), dtype=int)   # 8-connectivity
        labels, n = ndimage.label(occupancy, structure=structure)
        if n == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        index = np.arange(1, n + 1)
        sizes = np.asarray(ndimage.sum(occupancy, labels, index=index), dtype=float)
        coms = np.asarray(ndimage.center_of_mass(occupancy, labels, index=index), dtype=float)
        centroids = self.grid.to_xy(coms[:, 0], coms[:, 1])
        areas = sizes * self.grid.cell_area
        typical = float(self.cfg.perception.get("typical_component_area", 8e-5))
        counts = np.maximum(1.0, np.round(areas / max(typical, 1e-9)))
        return centroids.reshape(-1, 2), areas, counts

    # ------------------------------------------------------------------ noise
    def _apply_noise(self, centroids, areas, counts, rng):
        ncfg = self.cfg.perception.noise
        if centroids.shape[0] and float(ncfg.dropout) > 0.0:
            keep = rng.random(centroids.shape[0]) >= float(ncfg.dropout)
            centroids, areas, counts = centroids[keep], areas[keep], counts[keep]
        if centroids.shape[0] and float(ncfg.position_std) > 0.0:
            centroids = centroids + rng.normal(0.0, float(ncfg.position_std), centroids.shape)
        fp_rate = float(ncfg.false_positive_rate)
        if fp_rate > 0.0:
            n_fp = int(rng.poisson(fp_rate))
            if n_fp:
                ws = self.cfg.workspace
                fake = np.stack(
                    [rng.uniform(float(ws.x_min), float(ws.x_max), n_fp),
                     rng.uniform(float(ws.y_min), float(ws.y_max), n_fp)], axis=1
                )
                centroids = np.concatenate([centroids, fake], axis=0)
                areas = np.concatenate([areas, np.full(n_fp, 8e-5)])
                counts = np.concatenate([counts, np.ones(n_fp)])
        return centroids.reshape(-1, 2), np.asarray(areas, float), np.asarray(counts, float)
