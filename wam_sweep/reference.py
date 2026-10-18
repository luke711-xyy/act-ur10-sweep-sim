"""Causal previews of already accepted ACT chunks; no query/inference/rollout.

The caller owns loading a verified frozen scheduler and Task2 origin checks
before/after native activation, including its explicit alias. This module never
imports native code or repairs sys.path/sys.modules. Source and queue bytes are
bound into preview provenance, along with optional native normalization context.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import sys

import numpy as np

from .contracts import ReferencePreview
from .inputs import digest_json, file_hash


@dataclass(frozen=True, slots=True)
class NativeTargetContext:
    """Static frozen config: ((x_min,x_max),(y_min,y_max)), (z_search_min,z_home).

    Actual backend/controller callers MUST supply this context; omission is a
    raw queue diagnostic only. Normalize AFTER native temporal fusion, exactly
    as frozen sim/act/evaluate.py:300-317. Corrected targets are never clipped.
    """
    xy_bounds: tuple[tuple[float, float], tuple[float, float]]
    z_bounds: tuple[float, float]

    def __post_init__(self):
        xy, z = np.asarray(self.xy_bounds, dtype=float), np.asarray(self.z_bounds, dtype=float)
        if (xy.shape != (2, 2) or z.shape != (2,) or not np.isfinite(xy).all()
                or not np.isfinite(z).all() or np.any(xy[:, 0] >= xy[:, 1]) or z[0] >= z[1]):
            raise ValueError("invalid native workspace/target range")
        object.__setattr__(self, "xy_bounds", tuple(tuple(float(x) for x in row) for row in xy))
        object.__setattr__(self, "z_bounds", tuple(float(x) for x in z))

    def normalize(self, target):
        """Return a native FP32 nominal q; this does not own live Z control."""
        q = np.array(target, dtype=np.float32, copy=True)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("invalid native target")
        q[0] = np.clip(q[0], *self.xy_bounds[0])
        q[1] = np.clip(q[1], *self.xy_bounds[1])
        q[2] = np.clip(q[2], *self.z_bounds)
        q[3] = np.arctan2(np.sin(q[3]), np.cos(q[3]))
        return q


def _state(value):
    """Canonicalize scheduler state without losing array dtype/signed-zero bytes."""
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), bytes=value.tobytes().hex())
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _state(item) for key, item in value.items()}
    return _state(vars(value))


def preview_reference(scheduler, held_reference, now, *, native_context: NativeTargetContext | None = None) -> ReferencePreview:
    """25 commands at now+j*.04; resulting states occur at now+(j+1)*.04.

    Use only chunks accepted at the root: no pending results, query_due,
    mark_query or accept call. Copy BOTH scheduler and ensemble before action_for
    can expire chunks. Missing actions retain the supplied root held reference
    and mark residual inactive. These invalid placeholders do not simulate an
    accepted-reference trajectory; the native branch/executor owns live holds.
    ReferencePreview stores lossless FP64 values of normalized native FP32 q.
    """
    held = np.asarray(held_reference)
    if held.shape != (4,) or held.dtype.kind != "f" or not np.isfinite(held).all():
        raise ValueError("invalid held reference")
    if (isinstance(now, (bool, np.bool_)) or not np.isscalar(now) or not np.isfinite(now)
            or now < 0. or not np.isclose(scheduler.action_period, .04, rtol=0., atol=1e-12)):
        raise ValueError("preview requires a finite root time and 25 Hz scheduler")
    if native_context is not None and not isinstance(native_context, NativeTargetContext):
        raise ValueError("NativeTargetContext required")
    copied = copy.deepcopy(scheduler)
    chunks = (copied.temporal_ensemble.chunks if copied.temporal_ensemble is not None
              else ([] if copied.chunk is None else [copied.chunk]))
    if any(not np.isfinite(chunk.issued_at) or chunk.issued_at > float(now) for chunk in chunks):
        raise ValueError("invalid/future chunk issuance at preview root")
    module = sys.modules[type(copied).__module__]
    source = dict(path=module.__file__, sha256=file_hash(module.__file__), scheduler=type(copied).__qualname__)
    provenance = dict(protocol="wam_causal_accepted_act_25hz_v1", source=source,
                      scheduler=_state(vars(copied)), held=_state(held), now=float(now),
                      native_context=None if native_context is None else asdict(native_context))
    source_hash = digest_json(provenance)
    times = float(now) + np.arange(25) * .04
    valid = np.zeros(25)
    targets = []
    for j, timestamp in enumerate(times):
        target = held.copy()
        q = copied.action_for(float(timestamp))
        if q is not None:
            q = np.asarray(q)
            if q.shape != (4,) or q.dtype.kind != "f" or not np.isfinite(q).all():
                raise ValueError("invalid accepted ACT target")
            target = q.copy() if native_context is None else native_context.normalize(q)
            valid[j] = 1.
        targets.append(target)
    return ReferencePreview(np.asarray(targets), times, valid, float(now), source_hash)
