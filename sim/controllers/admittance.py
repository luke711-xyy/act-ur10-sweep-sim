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


class AdmittanceController1D:
    def __init__(
        self,
        params: AdmittanceParams,
        dt: float,
        delta_limit: float = 0.02,
        rate_limit: float = 0.05,
        direction: float = -1.0,
    ):
        if not params.is_stable():
            raise ValueError(f"non-physical admittance parameters: {params}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.params = params
        self.dt = float(dt)
        self.delta_limit = float(abs(delta_limit))
        self.rate_limit = float(abs(rate_limit))
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
        w_rate_limit = self.rate_limit  # |direction| == 1, so limits carry over
        self._dw = float(np.clip(self._dw, -w_rate_limit, w_rate_limit))

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
