"""Component geometry library.

The first version deliberately uses *simple convex primitives* (and one inline
hexagonal-prism mesh) rather than CAD meshes of real fasteners.  See the README
section "Known simulation limitations": this validates the control, planning and
data pipeline, **not** the detailed contact dynamics of real screws.

Each geometry is described by :class:`ComponentSpec`, a small declarative record
that :mod:`sim.model.scene_builder` turns into MJCF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

GEOMETRY_NAMES: Tuple[str, ...] = ("hex_nut", "cylinder", "screw", "bolt", "washer")


@dataclass
class GeomSpec:
    """A single MuJoCo geom belonging to a component body."""

    type: str                       # box | cylinder | capsule | mesh | sphere
    size: Sequence[float] = ()      # MuJoCo size semantics for ``type``
    pos: Sequence[float] = (0.0, 0.0, 0.0)
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    mesh: str | None = None
    rgba: Sequence[float] = (0.75, 0.75, 0.78, 1.0)


@dataclass
class ComponentSpec:
    """Full description of one component geometry family."""

    name: str
    geoms: List[GeomSpec]
    half_height: float              # resting half-height, used to place the body
    nominal_radius: float           # planar footprint radius, used by the planner
    meshes: Dict[str, np.ndarray] = field(default_factory=dict)


def _hex_prism_vertices(circumradius: float, half_height: float) -> np.ndarray:
    """12 vertices of a regular hexagonal prism, centred at the origin."""
    angles = np.arange(6) * (np.pi / 3.0)
    ring = np.stack([circumradius * np.cos(angles), circumradius * np.sin(angles)], axis=1)
    top = np.concatenate([ring, np.full((6, 1), +half_height)], axis=1)
    bottom = np.concatenate([ring, np.full((6, 1), -half_height)], axis=1)
    return np.concatenate([top, bottom], axis=0)


# Quaternion that rotates the local +z axis onto +x (used for lying shafts).
_Z_TO_X = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)


def make_component_spec(name: str, scale: float = 1.0) -> ComponentSpec:
    """Build the :class:`ComponentSpec` for ``name`` at the given size scale."""
    s = float(scale)
    if name == "hex_nut":
        # M8-ish nut: 13 mm across flats, 6.5 mm thick.
        across_flats = 0.013 * s
        circumradius = across_flats / np.sqrt(3.0)
        half_h = 0.00325 * s
        verts = _hex_prism_vertices(circumradius, half_h)
        return ComponentSpec(
            name=name,
            geoms=[GeomSpec(type="mesh", mesh="mesh_hex_nut", rgba=(0.72, 0.73, 0.76, 1.0))],
            half_height=half_h,
            nominal_radius=circumradius,
            meshes={"mesh_hex_nut": verts},
        )

    if name == "cylinder":
        radius, half_h = 0.006 * s, 0.004 * s
        return ComponentSpec(
            name=name,
            geoms=[GeomSpec(type="cylinder", size=(radius, half_h), rgba=(0.70, 0.72, 0.78, 1.0))],
            half_height=half_h,
            nominal_radius=radius,
        )

    if name == "washer":
        # Flat washer approximation: a thin disc (the central hole is ignored).
        radius, half_h = 0.0085 * s, 0.0009 * s
        return ComponentSpec(
            name=name,
            geoms=[GeomSpec(type="cylinder", size=(radius, half_h), rgba=(0.66, 0.68, 0.72, 1.0))],
            half_height=half_h,
            nominal_radius=radius,
        )

    if name == "screw":
        # Short screw lying on its side: shaft along local +x with a flat head.
        shaft_r, shaft_half_len = 0.0018 * s, 0.010 * s
        head_r, head_half_h = 0.0037 * s, 0.0015 * s
        return ComponentSpec(
            name=name,
            geoms=[
                GeomSpec(
                    type="cylinder",
                    size=(shaft_r, shaft_half_len),
                    pos=(0.0, 0.0, 0.0),
                    quat=_Z_TO_X,
                    rgba=(0.68, 0.70, 0.76, 1.0),
                ),
                GeomSpec(
                    type="cylinder",
                    size=(head_r, head_half_h),
                    pos=(-(shaft_half_len + head_half_h), 0.0, 0.0),
                    quat=_Z_TO_X,
                    rgba=(0.60, 0.62, 0.70, 1.0),
                ),
            ],
            half_height=head_r,
            nominal_radius=shaft_half_len + 2.0 * head_half_h,
        )

    if name == "bolt":
        # Bolt lying on its side: hexagonal head + longer shaft.
        shaft_r, shaft_half_len = 0.0028 * s, 0.014 * s
        head_circumradius, head_half_h = 0.0075 * s, 0.0028 * s
        verts = _hex_prism_vertices(head_circumradius, head_half_h)
        return ComponentSpec(
            name=name,
            geoms=[
                GeomSpec(
                    type="cylinder",
                    size=(shaft_r, shaft_half_len),
                    quat=_Z_TO_X,
                    rgba=(0.68, 0.70, 0.76, 1.0),
                ),
                GeomSpec(
                    type="mesh",
                    mesh="mesh_bolt_head",
                    pos=(-(shaft_half_len + head_half_h), 0.0, 0.0),
                    quat=_Z_TO_X,
                    rgba=(0.60, 0.62, 0.70, 1.0),
                ),
            ],
            half_height=head_circumradius,
            nominal_radius=shaft_half_len + 2.0 * head_half_h,
            meshes={"mesh_bolt_head": verts},
        )

    raise ValueError(f"unknown component geometry {name!r}; expected one of {GEOMETRY_NAMES}")


def resolve_geometry_list(geometry: str, count: int, rng: np.random.Generator) -> List[str]:
    """Expand the ``components.geometry`` config entry into one name per component."""
    if geometry == "mixed":
        return list(rng.choice(GEOMETRY_NAMES, size=count))
    if geometry not in GEOMETRY_NAMES:
        raise ValueError(f"unknown component geometry {geometry!r}")
    return [geometry] * count
