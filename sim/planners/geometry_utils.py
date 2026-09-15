"""Small planar-geometry helpers shared by the planners."""

from __future__ import annotations

import numpy as np


def pusher_width(cfg) -> float:
    """Width of the closed gripper-tip pair along the tool's local Y axis.

    The two tips sit at +/-(gap/2 + tip_half_y) and are ``2 * tip_half_y`` wide
    each, so the outer-to-outer extent is ``gap + 4 * tip_half_y``.
    """
    tip_half = np.asarray(cfg.end_effector.tip_half, dtype=float)
    return float(cfg.end_effector.tip_gap) + 4.0 * float(tip_half[1])


def stroke_yaw(direction: np.ndarray) -> float:
    """Yaw that points the pusher face along ``direction``.

    At yaw = 0 the tip face normal is -X, so a pure right-to-left stroke needs
    yaw = 0.  The returned angle is wrapped into [-pi/2, pi/2] because the tip
    pair is symmetric.
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return 0.0
    d = d / norm
    yaw = float(np.arctan2(-d[1], -d[0]))
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


def point_segment_frames(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    """Along-track / cross-track coordinates of ``points`` w.r.t. a segment.

    Returns ``(along, lateral, length)`` where ``along`` is normalised to
    ``[0, 1]`` over the segment (values outside mean the point is before/after
    the segment) and ``lateral`` is the signed perpendicular distance.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return np.zeros(points.shape[0]), np.linalg.norm(points - start, axis=1), 0.0
    unit = vec / length
    normal = np.array([-unit[1], unit[0]])
    rel = points - start[None, :]
    along = rel @ unit / length
    lateral = rel @ normal
    return along, lateral, length


def single_link_clusters(points: np.ndarray, eps: float):
    """Single-link (union-find) clustering; returns a label per point."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n = points.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if n:
        d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        close = np.argwhere(d2 <= eps * eps)
        for a, b in close:
            if a < b:
                union(int(a), int(b))
    roots = [find(i) for i in range(n)]
    remap, labels = {}, []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return np.asarray(labels, dtype=int)
