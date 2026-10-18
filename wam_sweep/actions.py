"""ACT-relative feasible targets, with independent, restorable WAM state.

No native/legacy package is imported. Only the two pinned pure mapping functions
are compiled into a private namespace; native contact and Z control stay outside
this codec. Every decode maps afresh from measured TCP to the supplied nominal q.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .contracts import CandidateProgram

HELPER_PATH = Path(__file__).resolve().parents[1] / "residual_rl/relative_action.py"
HELPER_SHA256 = "f185c594c0a539c16e100ef7bf3f6c88cd2393ee2276fdc5e857d425108a41d8"
PROTOCOL = "wam_relative_xy80_yaw45_pchip_tau200ms_hold_reset_v1"


def _load_mapping():
    source = HELPER_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != HELPER_SHA256:
        raise ValueError("correction helper hash drift")
    tree = ast.parse(source, filename=str(HELPER_PATH))
    names = ("square_to_disk", "generate_relative_action")
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if tuple(node.name for node in nodes) != names or any(node.decorator_list for node in nodes):
        raise ValueError("unexpected correction helper definitions")
    if ([arg.arg for arg in nodes[0].args.args] != ["xy"]
            or [arg.arg for arg in nodes[1].args.args] != ["actual_xy", "native_target", "action"]
            or [arg.arg for arg in nodes[1].args.kwonlyargs] != ["bounds", "ratio", "yaw_limit"]):
        raise ValueError("unexpected correction helper signature")
    # Normal Python builtins are needed by NumPy dtype formatting internally.
    # Isolation is achieved by pinned AST selection, not a partial sandbox.
    namespace = {"math": math, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPER_PATH), "exec"), namespace)
    return namespace["generate_relative_action"]


def _vector(value, size, name, *, normalized=False):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "fiu":
        raise ValueError(f"invalid {name}")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all() or (normalized and np.any(np.abs(result) > 1.)):
        raise ValueError(f"invalid {name}")
    return result


@dataclass(frozen=True, slots=True)
class DecodedAction:
    target: np.ndarray
    raw: np.ndarray
    filtered: np.ndarray
    residual: np.ndarray
    residual_rate: np.ndarray
    active: bool
    audit: dict


class ActionCodec:
    """Normalized PCHIP programs -> 0.2s filter -> measured-state feasible map.

    ``sample`` is pure. At each real replan, call ``anchor = begin_program()``
    once, snapshot the root, and pass that SAME anchor for all samples of that
    program. ``current_raw`` moves on decode; ``program_anchor`` does not. Restore
    a separate codec from the same root for each candidate, then decode using
    that branch's actual/predicted TCP. Only decode on the execution codec
    advances real filter state. Inactive/hold clears raw/filter/applied residual;
    it emits the original nominal bytes, even following an active correction.
    """
    def __init__(self, bounds=None):
        box = None if bounds is None else np.asarray(bounds, dtype=float)
        if box is not None and (box.shape != (2, 2) or not np.isfinite(box).all()
                                or np.any(box[:, 0] >= box[:, 1])):
            raise ValueError("invalid XY workspace")
        self._bounds = None if box is None else box.copy()
        self._mapping = _load_mapping()
        self._filtered = np.zeros(3)
        self._current_raw = np.zeros(3)
        self._program_anchor = np.zeros(3)
        self._residual = np.zeros(3)
        self._frames = 0

    @property
    def current_raw(self):
        """Copy of the last executed raw command, for the next t=0 anchor."""
        return self._current_raw.copy()

    def begin_program(self):
        """Capture latest real raw at a new 200ms program's immutable t=0."""
        self._program_anchor = self._current_raw.copy()
        return self._program_anchor.copy()

    def sample(self, program: CandidateProgram, t, current_raw) -> np.ndarray:
        if not isinstance(program, CandidateProgram):
            raise TypeError("CandidateProgram required")
        # Public contract arrays are mutable: revalidate at the consumer.
        knots = CandidateProgram(program.knots).knots
        anchor = _vector(current_raw, 3, "raw anchor", normalized=True)
        if isinstance(t, (bool, np.bool_)) or not np.isscalar(t) or not np.isfinite(t) or not 0. <= t <= 1.:
            raise ValueError("sample time must lie in [0, 1]")
        return np.asarray(PchipInterpolator(
            [0., .2, .4, .6, .8, 1.], np.vstack((anchor, knots)), axis=0,
            extrapolate=False)(float(t)))

    def decode(self, actual_tcp, native_target, raw, dt, active) -> DecodedAction:
        p = _vector(actual_tcp, 4, "actual TCP xyz/yaw")
        q = np.asarray(native_target)
        if q.shape != (4,) or q.dtype.kind != "f" or not np.isfinite(q).all():
            raise ValueError("invalid native target")
        action = _vector(raw, 3, "raw action", normalized=True)
        if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0.:
            raise ValueError("invalid decode clock")
        if not isinstance(active, (bool, np.bool_)):
            raise ValueError("active must be boolean")
        active = bool(active)
        filtered = (self._filtered + (action - self._filtered) * -np.expm1(-float(dt) / .2)
                    if active else np.zeros(3))
        # The original held reference is authoritative during inactive phases;
        # no workspace policy, yaw wrapping or legacy gate restriction applies.
        target, audit = self._mapping(p[:2], q, filtered,
                                      bounds=self._bounds if active else None,
                                      ratio=.8, yaw_limit=math.pi / 4)
        residual = np.r_[audit["xy_correction"], audit["yaw_correction"]]
        rate = (residual - self._residual) / float(dt)
        audit.update(protocol=PROTOCOL, helper_sha256=HELPER_SHA256,
                     helper_source=str(HELPER_PATH), filter_before=self._filtered.copy(),
                     program_anchor=self._program_anchor.copy(),
                     raw_action=action.copy(), filtered_action=filtered.copy(),
                     frame_index=self._frames + 1, dt=float(dt), active=active)
        # Publish only after validation and the independent geometry audit pass.
        self._filtered = filtered.copy()
        self._current_raw = action.copy() if active else np.zeros(3)
        if not active:
            self._program_anchor = np.zeros(3)
        self._residual = residual.copy()
        self._frames += 1
        return DecodedAction(target, action, filtered, residual, rate, active, audit)

    def snapshot(self):
        """Whitelisted JSON state; no episode snapshot, environment or truth.

        Safe initial state: program_anchor/current_raw/filtered/residual are all
        [0.,0.,0.], frames=0, with this exact protocol/helper hash and constructor
        workspace. ``filtered`` is normalized hidden actuator state, NOT inferred
        from physical residual (which can vanish at p=q for many filters).
        Program-relative time t remains an explicit sample argument.
        """
        return dict(protocol=PROTOCOL, helper_sha256=HELPER_SHA256,
                    bounds=None if self._bounds is None else self._bounds.tolist(),
                    program_anchor=self._program_anchor.tolist(),
                    filtered=self._filtered.tolist(), current_raw=self._current_raw.tolist(),
                    residual=self._residual.tolist(), frames=self._frames)

    def restore(self, state):
        expected = self.snapshot()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[key] != expected[key] for key in ("protocol", "helper_sha256", "bounds"))):
            raise ValueError("incompatible codec provenance/workspace")
        filtered = _vector(state["filtered"], 3, "filtered state", normalized=True)
        raw = _vector(state["current_raw"], 3, "raw state", normalized=True)
        anchor = _vector(state["program_anchor"], 3, "program anchor", normalized=True)
        residual = _vector(state["residual"], 3, "residual state")
        if (type(state["frames"]) is not int or state["frames"] < 0
                or abs(residual[2]) > math.pi / 4 + 1e-12):
            raise ValueError("invalid codec snapshot")
        mapping = _load_mapping()  # Drift cannot silently restore old state.
        self._filtered, self._current_raw, self._residual = filtered, raw, residual
        self._program_anchor = anchor
        self._frames, self._mapping = state["frames"], mapping
