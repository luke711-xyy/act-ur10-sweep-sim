"""Reproducible sampling of component layouts.

Every episode is fully determined by ``(config, seed)``: the same seed always
produces the same component count, geometries, poses, masses, frictions and
table-height perturbation, which is what makes the paired planner comparison
meaningful.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import List

import numpy as np

from ..model.geometries import make_component_spec, resolve_geometry_list


def sample_layout(cfg, rng: np.random.Generator) -> List[dict]:
    """Sample ``cfg.components.count`` component placements.

    Two spawn modes (``components.spawn_mode``):

    ``uniform``
        Components are scattered independently across the spawn box.  This is
        the distribution the *planner* comparison uses -- it is what makes the
        fixed full-cover baseline expensive and rewards re-planning.

    ``cluster``
        All components are drawn around one randomly placed cluster centre with
        a randomised spread.  This is the locked main task for demonstration
        collection: it keeps tool-object and object-object contact while
        removing the combinatorial initial-state space that makes a ~50-demo
        imitation dataset hopeless.

    Guarantees in both modes
    ------------------------
    * no component starts inside the target region (plus ``target_clearance``)
    * components are separated by at least ``components.min_separation``
    * all components start inside the spawn box, which lies inside the workspace
    """
    comp_cfg = cfg.components
    count = int(comp_cfg.count)
    geometries = resolve_geometry_list(str(comp_cfg.geometry), count, rng)

    spawn = comp_cfg.spawn
    clearance = float(comp_cfg.target_clearance)
    tgt = cfg.target
    min_sep = float(comp_cfg.min_separation)
    mode = str(comp_cfg.get("spawn_mode", "uniform"))
    if mode not in ("uniform", "cluster"):
        raise ValueError(f"unknown components.spawn_mode {mode!r}")

    centre, spread = None, None
    if mode == "cluster":
        centre, spread = _sample_cluster(cfg, rng)

    placed: List[dict] = []
    max_attempts = 4000
    for index in range(count):
        geometry = str(geometries[index])
        spec = make_component_spec(geometry, float(comp_cfg.size_scale))
        for attempt in range(max_attempts):
            if mode == "cluster":
                # widen the spread slowly if the cluster is too tight to fit them all
                scale = spread * (1.0 + attempt / 400.0)
                cluster = comp_cfg.cluster
                radius = float(cluster.get("max_radius", 0.12))
                # Keep the production task inside a bounded cluster while
                # allowing full lateral scatter.  The configured ellipse is
                # not narrowed to the brush footprint: the expert planner
                # must be able to produce multi-turn paths when needed.
                sweep_oriented = count <= 6
                lateral_ratio = (float(cluster.get(
                    "lateral_spread_ratio", 1.0)) if sweep_oriented else 1.0)
                lateral_radius = (float(cluster.get(
                    "max_lateral_radius", radius * lateral_ratio))
                    if sweep_oriented else radius)
                if radius <= 0.0 or lateral_ratio <= 0.0 or lateral_radius <= 0.0:
                    raise ValueError(
                        "cluster max_radius, lateral_spread_ratio, and "
                        "max_lateral_radius must be positive"
                    )
                offset = rng.normal(
                    0.0, [scale, scale * lateral_ratio], size=2)
                norm = float(np.hypot(offset[0] / radius,
                                      offset[1] / lateral_radius))
                if norm > 1.0:
                    offset = offset / norm
                x, y = float(centre[0] + offset[0]), float(centre[1] + offset[1])
                if not (spawn.x_min <= x <= spawn.x_max and spawn.y_min <= y <= spawn.y_max):
                    continue
            else:
                x = float(rng.uniform(spawn.x_min, spawn.x_max))
                y = float(rng.uniform(spawn.y_min, spawn.y_max))
            if _inside_target(x, y, tgt, clearance):
                continue
            if any((x - p["x"]) ** 2 + (y - p["y"]) ** 2 < min_sep ** 2 for p in placed):
                continue
            break
        else:  # pragma: no cover - only with pathological configs
            raise RuntimeError(
                f"could not place component {index} after {max_attempts} attempts; "
                "loosen components.min_separation or enlarge components.spawn/cluster"
            )
        yaw = float(rng.uniform(-np.pi, np.pi)) if bool(comp_cfg.randomize_orientation) else 0.0
        placed.append(
            {
                "index": index,
                "geometry": geometry,
                "x": x,
                "y": y,
                "yaw": yaw,
                "mass": float(rng.uniform(comp_cfg.mass.min, comp_cfg.mass.max)),
                "friction": float(rng.uniform(comp_cfg.friction.min, comp_cfg.friction.max)),
                "half_height": spec.half_height,
                "nominal_radius": spec.nominal_radius,
            }
        )
    return placed


def _sample_cluster(cfg, rng: np.random.Generator):
    """Sample a reproducible, random-but-concentrated cluster centre.

    ``uniform_disk`` directly samples the latent centre uniformly in area,
    independently of legacy centre bounds and offset-radius reservations.
    Individual part constraints can change the resulting part centroid.
    Absent an explicit distribution, retain the legacy draws unchanged so
    frozen configurations and seeds still reproduce their original layouts.
    """
    comp_cfg = cfg.components
    cluster = comp_cfg.cluster
    spawn = comp_cfg.spawn
    distribution = str(cluster.get("center_distribution", "legacy"))
    if distribution == "uniform_disk":
        disk = cluster.get("center_disk")
        if not isinstance(disk, Mapping):
            raise ValueError("components.cluster.center_disk must specify center and radius")
        try:
            origin = np.asarray(disk.get("center", []), dtype=float)
            radius = float(disk.get("radius", float("nan")))
        except (TypeError, ValueError) as exc:
            raise ValueError("components.cluster.center_disk requires numeric center and radius") from exc
        if (origin.shape != (2,) or not np.all(np.isfinite(origin))
                or not np.isfinite(radius) or radius <= 0.0):
            raise ValueError("components.cluster.center_disk requires a finite XY center and positive finite radius")
        # Check the whole configured disk against actual domains; never clip
        # it or reserve the individual offsets' radius around its centre.
        spawn_bounds = np.array([
            [float(spawn.x_min), float(spawn.x_max)],
            [float(spawn.y_min), float(spawn.y_max)],
        ])
        half_size = np.asarray(cfg.table.half_size, dtype=float)
        if (half_size.ndim != 1 or half_size.size < 2
                or not np.all(np.isfinite(half_size[:2]))
                or np.any(half_size[:2] <= 0.0)):
            raise ValueError("components.cluster.center_disk requires finite positive table XY half_size")
        # Accept exact tangencies despite roundoff (e.g. 0.05 + 0.10).
        # This tolerance only validates; it never moves or resizes the disk.
        tolerance = 8 * np.finfo(float).eps * max(1.0, radius, float(np.max(np.abs(origin))))
        if (not np.all(np.isfinite(spawn_bounds))
                or np.any(origin - radius < spawn_bounds[:, 0] - tolerance)
                or np.any(origin + radius > spawn_bounds[:, 1] + tolerance)
                or np.any(np.abs(origin) + radius > half_size[:2] + tolerance)):
            raise ValueError("components.cluster.center_disk must fit inside the actual spawn box and table")
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        radial_distance = radius * np.sqrt(rng.uniform(0.0, 1.0))
        centre = origin + radial_distance * np.array([np.cos(theta), np.sin(theta)])
        spread = float(rng.uniform(cluster.spread_std.min, cluster.spread_std.max))
        return centre, spread
    if distribution != "legacy":
        raise ValueError(f"unknown components.cluster.center_distribution {distribution!r}")

    pad = float(cluster.get("max_radius", 0.12))
    safe_bounds = np.array([
        [float(spawn.x_min) + pad * 0.5, float(spawn.x_max) - pad * 0.5],
        [float(spawn.y_min) + pad * 0.5, float(spawn.y_max) - pad * 0.5],
    ], dtype=float)

    configured_bounds = cluster.get("center_bounds")
    if configured_bounds is None:
        configured_bounds = {
            "x_min": float(cluster.center_x.min),
            "x_max": float(cluster.center_x.max),
            "y_min": float(cluster.center_y.min),
            "y_max": float(cluster.center_y.max),
        }
    requested_bounds = np.array([
        [float(configured_bounds.get("x_min", safe_bounds[0, 0])),
         float(configured_bounds.get("x_max", safe_bounds[0, 1]))],
        [float(configured_bounds.get("y_min", safe_bounds[1, 0])),
         float(configured_bounds.get("y_max", safe_bounds[1, 1]))],
    ], dtype=float)
    bounds = np.column_stack((
        np.maximum(safe_bounds[:, 0], requested_bounds[:, 0]),
        np.minimum(safe_bounds[:, 1], requested_bounds[:, 1]),
    ))
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("components.cluster.center_bounds do not fit in spawn")

    if cluster.get("center_mean") is not None:
        mean = np.asarray(cluster.center_mean, dtype=float).reshape(-1)
        std = np.asarray(cluster.get("center_std", [0.0, 0.0]), dtype=float).reshape(-1)
        if len(mean) != 2 or len(std) != 2 or np.any(std < 0):
            raise ValueError("components.cluster.center_mean/std must be length-2 non-negative values")
        centre = np.array([
            _sample_bounded_normal(rng, mean[axis], std[axis],
                                   bounds[axis, 0], bounds[axis, 1])
            for axis in range(2)
        ], dtype=float)
    else:
        centre = np.array([
            float(rng.uniform(bounds[0, 0], bounds[0, 1])),
            float(rng.uniform(bounds[1, 0], bounds[1, 1])),
        ])
    spread = float(rng.uniform(cluster.spread_std.min, cluster.spread_std.max))
    return centre, spread


def _sample_bounded_normal(rng: np.random.Generator, mean: float, std: float,
                           low: float, high: float) -> float:
    """Draw a normal variate inside ``[low, high]`` without scipy.

    Rejection keeps the centre distribution smooth in its useful range.  The
    clipped fallback is only a guard for pathological configurations and
    ensures layout generation cannot hang forever.
    """
    if std == 0.0:
        return float(np.clip(mean, low, high))
    for _ in range(64):
        value = float(rng.normal(mean, std))
        if low <= value <= high:
            return value
    return float(np.clip(rng.normal(mean, std), low, high))


def _inside_target(x: float, y: float, tgt, clearance: float) -> bool:
    return (
        (float(tgt.x_min) - clearance) <= x <= (float(tgt.x_max) + clearance)
        and (float(tgt.y_min) - clearance) <= y <= (float(tgt.y_max) + clearance)
    )


def in_target_region(xy: np.ndarray, tgt) -> np.ndarray:
    """Boolean mask of which ``(N, 2)`` planar positions lie inside the tray."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return (
        (xy[:, 0] >= float(tgt.x_min))
        & (xy[:, 0] <= float(tgt.x_max))
        & (xy[:, 1] >= float(tgt.y_min))
        & (xy[:, 1] <= float(tgt.y_max))
    )


def in_safe_workspace(xy: np.ndarray, ws) -> np.ndarray:
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return (
        (xy[:, 0] >= float(ws.safe_x_min))
        & (xy[:, 0] <= float(ws.safe_x_max))
        & (xy[:, 1] >= float(ws.safe_y_min))
        & (xy[:, 1] <= float(ws.safe_y_max))
    )
