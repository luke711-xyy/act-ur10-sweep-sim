"""OFFLINE ONLY: copied collision evidence, never an online observation.

Collector contract (no renderer or physics step is called here): ``env`` supplies
model/data, component_body_ids/component_geom_ids in stable native object order,
layout[i]['geometry'], brush_geom_ids and ee.tcp_yaw(). Native collected_mask()
and partial_collection_overlap_mask() are copied verbatim, not rederived.

At real RGB capture ticks the collector may set ``env.wam_truth_evidence`` to
{timestamp: data.time, visibility: bool[N], visibility_valid: bool[N]}. These
must come from camera-aligned segmentation/occlusion evidence with a documented
camera aggregation rule. No evidence, invalid entries or a different timestamp
means unknown visibility. This module never renders five cameras at control Hz.

TruthFrame.raw.fields schema: objects: tuple[CollisionBody], brush: CollisionBody,
object_ids: stable tuple[int], contacts: undirected (object_id, object_id) edges
(brush=-1), contacts_valid: bool, native_collected/native_partial/native_valid:
bool[N], visibility/visibility_valid: bool[N], target_bounds: (xmin,xmax,ymin,ymax)
or None. Contacts must be contemporaneous real collision contacts; a collector
missing contact evidence must set contacts_valid=False. IDs cannot be recycled
within an episode. Raw poses alone supply neither velocity nor visibility.

Envelope convention v1: screw/bolt -> body-X aligned enclosing rectangle (pi
period); near-round types -> conservative enclosing circle (yaw loss masked).
Primitive support bounds and compiled collision mesh vertices include actual
world rotations/tilt. Circle bounds are exact for spheres/upright cylinders;
tilted/compound round bodies use a conservative radial bound, not a visual mesh.
Velocity is a secant between consecutive envelope centers, never body qvel.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from .contracts import supervision

RawTruth = supervision.RawTruth  # Preserve Task1 public identity.
GEOMETRY_VERSION = "wam-collision-envelope-v1"


def _array(value, shape, *, boolean=False):
    a = np.asarray(value)
    if a.shape != shape or (boolean and a.dtype != np.bool_):
        raise ValueError(f"expected {'bool ' if boolean else ''}array {shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("nonfinite truth evidence")
    a = np.array(a, dtype=bool if boolean else float, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class CollisionGeom:
    kind: str
    size: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    vertices: np.ndarray | None = None

    def __post_init__(self):
        if self.kind not in {"sphere", "box", "cylinder", "capsule", "ellipsoid", "mesh"}:
            raise ValueError("unsupported collision geometry")
        for name, shape in (("size", (3,)), ("position", (3,)), ("rotation", (3, 3))):
            object.__setattr__(self, name, _array(getattr(self, name), shape))
        if np.any(self.size < 0) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-8):
            raise ValueError("invalid collision size/rotation")
        if self.kind == "mesh":
            v = np.asarray(self.vertices)
            if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4:
                raise ValueError("collision mesh vertices required")
            object.__setattr__(self, "vertices", _array(v, v.shape))


@dataclass(frozen=True)
class CollisionBody:
    geoms: tuple[CollisionGeom, ...]
    yaw: float
    shape: str
    yaw_valid: bool = True

    def __post_init__(self):
        if not self.geoms or any(type(g) is not CollisionGeom for g in self.geoms):
            raise ValueError("nonempty collision geometry required")
        if self.shape not in {"circle", "rectangle"} or not math.isfinite(self.yaw):
            raise ValueError("invalid planar body")
        object.__setattr__(self, "geoms", tuple(self.geoms))


@dataclass(frozen=True)
class TruthFrame:
    timestamp: float
    raw: RawTruth

    def __post_init__(self):
        if not math.isfinite(self.timestamp) or self.timestamp < 0 or type(self.raw) is not RawTruth:
            raise ValueError("finite timestamp and supervision.RawTruth required")
        f = dict(self.raw.fields)
        objects = tuple(f["objects"])
        n = len(objects)
        if not 1 <= n <= 6 or any(type(x) is not CollisionBody for x in (*objects, f["brush"])):
            raise ValueError("one to six collision bodies and brush required")
        ids = tuple(f["object_ids"])
        if len(ids) != n or len(set(ids)) != n or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("unique stable object IDs required")
        contacts = tuple(sorted({tuple(sorted(edge)) for edge in f["contacts"]}))
        if any(len(e) != 2 or e[0] == e[1] or any(i not in (*ids, -1) for i in e) for e in contacts):
            raise ValueError("invalid contact graph")
        if type(f["contacts_valid"]) is not bool:
            raise ValueError("explicit contacts_valid required")
        for name in ("native_collected", "native_partial", "native_valid", "visibility", "visibility_valid"):
            f[name] = _array(f[name], (n,), boolean=True)
        if np.any(f["native_collected"] & f["native_partial"] & f["native_valid"]):
            raise ValueError("native full and partial masks conflict")
        bounds = f["target_bounds"]
        if bounds is not None:
            bounds = tuple(_array(bounds, (4,)))
            if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
                raise ValueError("invalid target bounds")
        f.update(objects=objects, object_ids=ids, contacts=contacts, target_bounds=bounds)
        object.__setattr__(self, "raw", RawTruth(f))


def capture_truth(env) -> TruthFrame:
    """Copy actual MuJoCo state and optional tick evidence; no stepping/rendering."""
    import mujoco

    model, data = env.model, env.data
    kinds = {int(getattr(mujoco.mjtGeom, "mjGEOM_" + k.upper())): k
             for k in ("sphere", "box", "cylinder", "capsule", "ellipsoid", "mesh")}

    def geoms(ids):
        result = []
        for gid in sorted(ids):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            kind = kinds.get(int(model.geom_type[gid]))
            if kind is None:
                raise ValueError("unsupported collision geometry")
            vertices = None
            if kind == "mesh":
                mesh = int(model.geom_dataid[gid])
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                vertices = model.mesh_vert[start:start + count]
            result.append(CollisionGeom(kind, model.geom_size[gid], data.geom_xpos[gid],
                                        data.geom_xmat[gid].reshape(3, 3), vertices))
        return tuple(result)

    objects = []
    for index, bid in enumerate(env.component_body_ids):
        kind = env.layout[index]["geometry"]
        if kind not in {"screw", "bolt", "hex_nut", "cylinder", "washer"}:
            raise ValueError("unknown native object geometry")
        axis = data.xmat[bid].reshape(3, 3)[:2, 0]
        yaw_valid = bool(np.linalg.norm(axis) > 1e-9)
        yaw = float(np.arctan2(axis[1], axis[0])) if yaw_valid else 0.
        objects.append(CollisionBody(geoms(env.component_geom_ids[index]), yaw,
                                    "rectangle" if kind in {"screw", "bolt"} else "circle", yaw_valid))
    brush_ids = set(env.brush_geom_ids)
    brush = CollisionBody(geoms(brush_ids), float(env.ee.tcp_yaw()), "rectangle")
    owners = {gid: index for index, group in enumerate(env.component_geom_ids) for gid in group}
    owners.update({gid: -1 for gid in brush_ids})
    contacts = set()
    for c in data.contact[:data.ncon]:
        if c.dist > 0:  # margin-only proximity is not actual contact evidence
            continue
        a, b = owners.get(int(c.geom1)), owners.get(int(c.geom2))
        if a is not None and b is not None and a != b:
            contacts.add(tuple(sorted((a, b))))
    n = len(objects)
    visibility, visibility_valid = np.zeros(n, bool), np.zeros(n, bool)
    evidence = getattr(env, "wam_truth_evidence", None)
    if evidence is not None and evidence["timestamp"] == float(data.time):
        visibility = _array(evidence["visibility"], (n,), boolean=True)
        visibility_valid = _array(evidence["visibility_valid"], (n,), boolean=True)
    bounds = None
    if hasattr(env, "cfg") and hasattr(env.cfg, "target"):
        target = env.cfg.target
        bounds = (target.x_min, target.x_max, target.y_min, target.y_max)
    return TruthFrame(float(data.time), RawTruth(dict(
        objects=tuple(objects), brush=brush, object_ids=tuple(range(n)),
        contacts=tuple(contacts), contacts_valid=True,
        native_collected=env.collected_mask(), native_partial=env.partial_collection_overlap_mask(),
        native_valid=np.ones(n, bool), visibility=visibility, visibility_valid=visibility_valid,
        target_bounds=bounds)))


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _bounds(body, yaw):
    """Exact support extrema along the requested horizontal axes."""
    axes = _rotation(yaw)
    directions = np.vstack((axes, np.zeros(2))).T
    lower, upper = [], []
    for g in body.geoms:
        rows = directions @ g.rotation
        center = directions @ g.position
        if g.kind == "mesh":
            values = g.vertices @ rows.T + center
            lower.append(values.min(axis=0))
            upper.append(values.max(axis=0))
            continue
        if g.kind == "sphere":
            extent = np.full(2, g.size[0])
        elif g.kind == "box":
            extent = np.abs(rows) @ g.size
        elif g.kind == "ellipsoid":
            extent = np.linalg.norm(rows * g.size, axis=1)
        elif g.kind == "capsule":
            extent = g.size[0] + g.size[1] * np.abs(rows[:, 2])
        else:  # cylinder axis is local Z
            extent = g.size[1] * np.abs(rows[:, 2]) + g.size[0] * np.linalg.norm(rows[:, :2], axis=1)
        lower.append(center - extent)
        upper.append(center + extent)
    return np.min(lower, axis=0), np.max(upper, axis=0)


def _project_body(body):
    yaw = body.yaw if body.yaw_valid and body.shape == "rectangle" else 0.
    low, high = _bounds(body, yaw)
    center = _rotation(yaw) @ ((low + high) / 2)
    half = (high - low) / 2
    if body.shape == "circle":
        radius = 0.
        for g in body.geoms:
            if g.kind == "mesh":
                vertices = g.vertices @ g.rotation.T + g.position
                bound = np.linalg.norm(vertices[:, :2] - center, axis=1).max()
            else:
                offset = np.linalg.norm(g.position[:2] - center)
                if g.kind == "sphere":
                    radial = g.size[0]
                elif g.kind in {"cylinder", "capsule"}:
                    tilt = np.linalg.norm(g.rotation[:2, 2])
                    radial = g.size[0] + g.size[1] * tilt
                    if g.kind == "cylinder":
                        radial = min(radial, np.hypot(g.size[0], g.size[1]))
                elif g.kind == "ellipsoid":
                    radial = np.linalg.norm(g.rotation[:2] * g.size, ord=2)
                else:
                    # A bounding box corner radius is conservative for any tilt.
                    radial = np.linalg.norm(np.abs(g.rotation[:2]) @ g.size)
                bound = offset + radial
            radius = max(radius, float(bound))
        half = np.full(2, radius)
    return center, half, yaw


@dataclass(frozen=True)
class ObjectTargets:
    center_xy: np.ndarray
    half_extents_xy: np.ndarray
    yaw_double: np.ndarray
    yaw_valid: np.ndarray
    velocity_xy: np.ndarray
    velocity_valid: np.ndarray
    angular_velocity: np.ndarray
    angular_velocity_valid: np.ndarray
    visibility: np.ndarray
    visibility_valid: np.ndarray
    rectangle: np.ndarray
    object_valid: np.ndarray
    native_collected: np.ndarray
    native_partial: np.ndarray
    native_valid: np.ndarray
    collision_inside: np.ndarray
    collision_overlap: np.ndarray
    collision_region_valid: np.ndarray
    valid_steps: np.ndarray


def _region(center, half, yaw, circle, bounds):
    if bounds is None:
        return False, False
    low = np.array([bounds[0], bounds[2]])
    high = np.array([bounds[1], bounds[3]])
    if circle:
        inside = np.all(center - half >= low) and np.all(center + half <= high)
        overlap = np.linalg.norm(center - np.clip(center, low, high)) <= half[0]
    else:
        rotation = _rotation(yaw)
        extent = np.abs(rotation) @ half
        inside = np.all(center - extent >= low) and np.all(center + extent <= high)
        tc, th = (low + high) / 2, (high - low) / 2
        # Separating-axis theorem, target axes plus object's axes.
        overlap = True
        for axis in (*np.eye(2), *rotation.T):
            distance = abs((center - tc) @ axis)
            radius = np.abs(axis) @ th + np.abs(axis @ rotation) @ half
            if distance > radius:
                overlap = False
                break
    return bool(inside), bool(overlap)


def _angular_secant(current, current_valid, previous, previous_valid, dt):
    """Shortest pi-period secant; a half-period jump has no signed evidence."""
    new_yaw = np.arctan2(current[:, 0], current[:, 1]) / 2
    old_yaw = np.arctan2(previous[:, 0], previous[:, 1]) / 2
    delta = new_yaw - old_yaw
    wrapped = np.arctan2(np.sin(2 * delta), np.cos(2 * delta)) / 2
    valid = current_valid & previous_valid & (np.abs(wrapped) < np.pi / 2 - 1e-9)
    return np.where(valid, wrapped / dt, 0.), valid


def project_envelopes(frame: TruthFrame, *, previous: TruthFrame | None = None) -> ObjectTargets:
    f = frame.raw.fields
    n = len(f["objects"])
    projected = [_project_body(body) for body in f["objects"]]
    center = np.array([p[0] for p in projected])
    half = np.array([p[1] for p in projected])
    yaw = np.array([p[2] for p in projected])
    rectangle = np.array([b.shape == "rectangle" for b in f["objects"]])
    yaw_valid = np.array([b.yaw_valid for b in f["objects"]]) & rectangle
    double = np.column_stack((np.sin(2 * yaw), np.cos(2 * yaw)))
    double[~yaw_valid] = 0.
    velocity, velocity_valid = np.zeros((n, 2)), np.zeros(n, bool)
    angular, angular_valid = np.zeros(n), np.zeros(n, bool)
    if previous is not None:
        dt = frame.timestamp - previous.timestamp
        if dt <= 0:
            raise ValueError("truth timestamps must be strictly increasing")
        if f["object_ids"] == previous.raw.fields["object_ids"]:
            old = project_envelopes(previous)
            velocity = (center - old.center_xy) / dt
            velocity_valid[:] = True
            angular, angular_valid = _angular_secant(double, yaw_valid, old.yaw_double, old.yaw_valid, dt)
    regions = [_region(c, h, y, not r, f["target_bounds"])
               for c, h, y, r in zip(center, half, yaw, rectangle)]
    visible_valid = f["visibility_valid"].copy()
    return ObjectTargets(center, half, double, yaw_valid, velocity, velocity_valid,
        angular, angular_valid, f["visibility"] & visible_valid, visible_valid,
        rectangle, np.ones(n, bool), f["native_collected"].copy(), f["native_partial"].copy(),
        f["native_valid"].copy(), np.array([r[0] for r in regions]),
        np.array([r[1] for r in regions]), np.full(n, f["target_bounds"] is not None), np.array(True))


def stack_targets(targets, steps):
    """Zero invalid future slots; never repeat a terminal state into the future."""
    if not targets or not 1 <= len(targets) <= steps:
        raise ValueError("nonempty observed prefix within horizon required")
    result = {}
    for field in fields(ObjectTargets):
        arrays = [getattr(t, field.name) for t in targets]
        if any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("object order/count must be stable")
        value = np.zeros((steps, *arrays[0].shape), dtype=arrays[0].dtype)
        value[:len(arrays)] = arrays
        result[field.name] = value
    return ObjectTargets(**result)
