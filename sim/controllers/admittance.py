"""One-dimensional admittance controller for the contact (Z) direction.

Model
-----
The outer loop implements the second-order admittance

    m_d * d2(w)/dt2 + b_d * d(w)/dt + k_d * w = F_desired - F_measured

where ``w`` is the position correction measured **along the inward contact
normal** (i.e. positive ``w`` means "press deeper into the table") and both
forces use the same pressing-positive convention, so a small *positive*
``F_desired`` is a light downward push.

Because the task frame has the table normal along +Z, the correction that the
position loop consumes is

    delta_z   = direction * w,        direction = -1 for a downward normal
    z_command = z_nominal + delta_z

which is exactly the form required by the task specification.  Writing the ODE
in the inward-normal frame is what keeps the signs correct: if the measured
force is below the target the right-hand side is positive, ``w`` grows, and the
commanded Z drops, which increases contact force.

Saturations
-----------
* ``delta_limit``      -- hard bound on |delta_z| (protects against runaway
                          integration when contact is lost)
* ``rate_limit``       -- hard bound on |d(delta_z)/dt|
Both use anti-windup: when a bound is hit the internal velocity is clipped (or
zeroed) instead of continuing to integrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdmittanceParams:
    m: float = 1.0
    b: float = 60.0
    k: float = 300.0

    def is_stable(self) -> bool:
        """Necessary and sufficient stability condition for this 2nd-order system."""
        return self.m > 0.0 and self.b > 0.0 and self.k >= 0.0

    @property
    def damping_ratio(self) -> float:
        if self.k <= 0.0:
            return float("inf")
        return self.b / (2.0 * np.sqrt(self.k * self.m))

    @property
    def natural_frequency(self) -> float:
        return float(np.sqrt(self.k / self.m)) if self.m > 0 else float("nan")


@dataclass
class OverforceReliefParams:
    """Rate-limited emergency unloading parameters for the contact loop."""

    activation_force: float = 10.0
    release_force: float = 7.0
    upward_rate: float = 0.02
    return_rate: float = 0.10
    max_offset: float = 0.00025

    def is_valid(self) -> bool:
        return (
            self.activation_force > self.release_force >= 0.0
            and self.upward_rate >= 0.0
            and self.return_rate >= 0.0
            and self.max_offset >= 0.0
        )


class OverforceReliefController1D:
    """Hysteretic, slew-limited upward relief layered over admittance."""

    def __init__(self, params: OverforceReliefParams, dt: float):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not params.is_valid():
            raise ValueError(f"invalid overforce relief parameters: {params}")
        self.params = params
        self.dt = float(dt)
        self._offset = 0.0

    def reset(self) -> None:
        self._offset = 0.0

    @property
    def offset(self) -> float:
        return self._offset

    def step(self, force_measured: float) -> float:
        force = max(0.0, float(force_measured))
        if force >= self.params.activation_force:
            self._offset = min(
                self.params.max_offset,
                self._offset + self.params.upward_rate * self.dt,
            )
        elif force < self.params.release_force:
            self._offset = max(
                0.0,
                self._offset - self.params.return_rate * self.dt,
            )
        return self._offset


class AdmittanceController1D:
    def __init__(
        self,
        params: AdmittanceParams,
        dt: float,
        delta_limit: float = 0.02,
        rate_limit: float = 0.05,
        direction: float = -1.0,
        upward_rate_limit: float | None = None,
        downward_rate_limit: float | None = None,
    ):
        if not params.is_stable():
            raise ValueError(f"non-physical admittance parameters: {params}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.params = params
        self.dt = float(dt)
        self.delta_limit = float(abs(delta_limit))
        self.rate_limit = float(abs(rate_limit))
        self.upward_rate_limit = float(abs(
            rate_limit if upward_rate_limit is None else upward_rate_limit))
        self.downward_rate_limit = float(abs(
            rate_limit if downward_rate_limit is None else downward_rate_limit))
        self.direction = float(np.sign(direction)) or -1.0
        self._w = 0.0   # correction along the inward normal
        self._dw = 0.0
        self.saturated = False

    # -- state ---------------------------------------------------------------
    def reset(self, delta_z: float = 0.0, velocity: float = 0.0) -> None:
        """Reset the internal state.  Called at the beginning of every sweep."""
        self._w = float(delta_z) / self.direction
        self._dw = float(velocity) / self.direction
        self.saturated = False

    @property
    def delta_z(self) -> float:
        return self.direction * self._w

    @property
    def delta_z_rate(self) -> float:
        return self.direction * self._dw

    # -- integration ---------------------------------------------------------
    def step(self, force_desired: float, force_measured: float) -> float:
        """Advance one control period; returns the new ``delta_z``."""
        p = self.params
        error = float(force_desired) - float(force_measured)
        acc = (error - p.b * self._dw - p.k * self._w) / p.m

        # semi-implicit (symplectic) Euler: more stable than explicit Euler for
        # stiff spring/damper pairs at 100 Hz.
        self._dw += acc * self.dt
        # Rate limits are specified in world Z, where positive is upward.
        # Convert through ``direction`` so the same controller also supports
        # contact normals that point along +Z.
        z_rate = self.direction * self._dw
        z_rate = float(np.clip(
            z_rate, -self.downward_rate_limit, self.upward_rate_limit))
        self._dw = self.direction * z_rate

        w_next = self._w + self._dw * self.dt
        w_limit = self.delta_limit
        if w_next > w_limit:
            w_next, self._dw, self.saturated = w_limit, min(self._dw, 0.0), True
        elif w_next < -w_limit:
            w_next, self._dw, self.saturated = -w_limit, max(self._dw, 0.0), True
        else:
            self.saturated = False
        self._w = w_next
        return self.delta_z

    def steady_state_delta(self, force_desired: float, force_measured: float) -> float:
        """Analytic equilibrium of the ODE (used by the unit tests)."""
        if self.params.k <= 0.0:
            return float("nan")
        w = (float(force_desired) - float(force_measured)) / self.params.k
        w = float(np.clip(w, -self.delta_limit, self.delta_limit))
        return self.direction * w
