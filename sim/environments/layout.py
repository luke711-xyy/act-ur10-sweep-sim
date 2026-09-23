"""Reproducible sampling of component layouts.

Every episode is fully determined by ``(config, seed)``: the same seed always
produces the same component count, geometries, poses, masses, frictions and
table-height perturbation, which is what makes the paired planner comparison
meaningful.
"""

from __future__ import annotations

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
                offset = rng.normal(0.0, scale, size=2)
                radius = float(comp_cfg.cluster.get("max_radius", 0.12))
                y_radius = float(comp_cfg.cluster.get("max_y_radius", radius))
                norm = float(np.linalg.norm(offset))
                if norm > radius:
                    offset = offset / norm * radius
                offset[1] = float(np.clip(offset[1], -y_radius, y_radius))
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
    """Random cluster centre and spread for the single-cluster task."""
    comp_cfg = cfg.components
    cluster = comp_cfg.cluster
    spawn = comp_cfg.spawn
    pad = float(cluster.get("max_radius", 0.12))
    y_pad = float(cluster.get("max_y_radius", pad))
    centre = np.array([
        float(rng.uniform(max(cluster.center_x.min, float(spawn.x_min) + pad * 0.5),
                          min(cluster.center_x.max, float(spawn.x_max) - pad * 0.5))),
        float(rng.uniform(max(cluster.center_y.min, float(spawn.y_min) + y_pad * 0.5),
                          min(cluster.center_y.max, float(spawn.y_max) - y_pad * 0.5))),
    ])
    spread = float(rng.uniform(cluster.spread_std.min, cluster.spread_std.max))
    return centre, spread


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
