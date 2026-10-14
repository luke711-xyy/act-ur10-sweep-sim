"""THROWAWAY: short-motion probe, not a production controller or policy.

Reuse the sealed wrist-only force/dwell state machine. Only the per-stage
M/B/K and ordinary upward limit are configurable. Native emergency unloading,
accepted-physics-clock commit, 1-N setpoint, 2-mm/s descent and 5-mm range stay.
"""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
SEALED = Path(__file__).resolve().parent.parent/'hybrid_astar_act_wrist1n_twostage_20261014/phases.py'
spec = importlib.util.spec_from_file_location('sealed_wrist_phase_machine', SEALED)
sealed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sealed)
from sim.controllers.admittance import AdmittanceController1D, AdmittanceParams


def checked_gains(values, dt):
    m,b,k = map(float, values)
    if not all(math.isfinite(v) for v in (m,b,k,dt)) or dt <= 0:
        raise ValueError('nonfinite controller gains/clock')
    if not (1. <= m <= 100. and 100. <= b <= 4000. and k == 1.):
        raise ValueError('outside temporary gain scope')
    matrix = [[1-dt*dt*k/m, dt*(1-dt*b/m)], [-dt*k/m, 1-dt*b/m]]
    if max(abs(np.linalg.eigvals(matrix))) >= 1. + 1e-10:
        raise ValueError('discrete admittance gains unstable')
    return m,b,k


class ProbeController(sealed.TwoStageAdmittance):
    def __init__(self, dt, *, seek, maintain, up,maintain_down=.002):
        self.seek_gains = checked_gains(seek, float(dt))
        self.maintain_gains = checked_gains(maintain, float(dt))
        self.up = float(up)
        self.maintain_down=float(maintain_down)
        if self.maintain_down not in (.0002,.0005,.001,.002):
            raise ValueError('maintenance descent must stay within the 2-mm/s ceiling')
        if self.up not in (.0001,.001,.002):
            raise ValueError('only the three approved normal upward limits')
        super().__init__(dt)

    def _loop(self, damping_marker):
        # The sealed state machine requests 400 at acquisition, 4000 at
        # force-confirmed settling; refuse any unexpected parent contract.
        if damping_marker == 400.: values = self.seek_gains
        elif damping_marker == 4000.: values = self.maintain_gains
        else: raise RuntimeError('sealed phase-machine factory contract changed')
        return AdmittanceController1D(
            AdmittanceParams(*values), self.dt, delta_limit=.005,
            rate_limit=.002, direction=-1., upward_rate_limit=self.up,
            downward_rate_limit=.002 if damping_marker==400. else self.maintain_down)


class ProbeLimiter(sealed.ZCommandLimiter):
    def __init__(self, initial_z, dt, *, up, initial_time=0.):
        self.up = float(up)
        if self.up not in (.0001,.001,.002):
            raise ValueError('outside approved normal upward limits')
        super().__init__(initial_z,dt,initial_time)

    def apply(self, desired_z, wrist_force,*,down=.002):
        desired_z = float(desired_z)
        force = self._force(wrist_force)
        if not math.isfinite(desired_z): raise ValueError('nonfinite Z command')
        down=float(down)
        if not math.isfinite(down) or not 0.<down<=.002:raise ValueError('invalid descent ceiling')
        # Keep the preceding diagnostic's fast emergency bound unchanged.
        up = .0201 if force >= 10. else self.up
        return max(self.last_z-down*self.dt,
                   min(self.last_z+up*self.dt,desired_z))

    _force = staticmethod(sealed.TwoStageAdmittance._force)


def small_motion_xy(seconds):
    """5-mm out-and-back raised-cosine moves, never a task sweep.

    X/Y individually over 4 s, individually over 2 s, then both over 2 s.
    All joins have zero velocity; peak component speed is <8 mm/s.
    """
    t = max(0., float(seconds))
    segments = [(0.,4.,1.,0.), (4.,4.,0.,1.),
                (8.,2.,1.,0.), (10.,2.,0.,1.), (12.,2.,1.,1.)]
    for begin,duration,dx,dy in segments:
        if begin <= t < begin+duration:
            offset = .0025*(1.-math.cos(2.*math.pi*(t-begin)/duration))
            norm=math.hypot(dx,dy)
            return np.array([.42+dx*offset/norm,dy*offset/norm])
    return np.array([.42,0.])
