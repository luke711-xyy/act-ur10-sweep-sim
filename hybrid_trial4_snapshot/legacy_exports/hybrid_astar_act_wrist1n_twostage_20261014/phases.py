"""Disposable wrist-only, one-newton acquisition/maintenance controller.

No table contact truth, part contact classifier, model Z, or alternate force
channel is accepted by this controller. A force-confirmed handoff is not a
claim that the wrist force must originate exclusively from the table.
"""
import math

from sim.controllers.admittance import AdmittanceController1D, AdmittanceParams


class ZCommandLimiter:
    """Bound the composed command, not merely the internal force correction.

    Existing emergency upward unloading remains fast at >=10 N. Its return
    never creates a downward jump faster than the requested 2 mm/s.
    """
    def __init__(self, initial_z, dt, initial_time=0.):
        self.last_z, self.dt = float(initial_z),float(dt)
        self.last_time=float(initial_time)
        if not math.isfinite(self.last_z) or self.dt<=0: raise ValueError('invalid Z limiter state')

    def apply(self, desired_z, wrist_force):
        desired_z=float(desired_z)
        force=TwoStageAdmittance._force(wrist_force)
        if not math.isfinite(desired_z): raise ValueError('nonfinite Z command')
        up=.0201 if force>=10. else .0001
        z=max(self.last_z-.002*self.dt,min(self.last_z+up*self.dt,desired_z))
        return z

    def commit(self,accepted_z,physics_time):
        z,now=float(accepted_z),float(physics_time)
        if not math.isfinite(z) or not math.isfinite(now): raise ValueError('invalid accepted command')
        if now<self.last_time-1e-10: raise ValueError('accepted physics clock reversed')
        if now<=self.last_time+1e-10:return
        self.last_z,self.last_time=z,now


class TwoStageAdmittance:
    def __init__(self, dt):
        self.dt = float(dt)
        self.loop = self._loop(400.)
        self.phase = 'seek'
        self.force_since = None
        self.stable_since = None
        self.last_time = None
        self.transitions = []

    def _loop(self, damping):
        return AdmittanceController1D(
            AdmittanceParams(25., damping, 1.), self.dt,
            delta_limit=.005, rate_limit=.002, direction=-1.,
            upward_rate_limit=.0001, downward_rate_limit=.002)

    @property
    def params(self): return self.loop.params

    @property
    def ready(self): return self.phase == 'maintain'

    @property
    def delta_z(self): return self.loop.delta_z

    @property
    def delta_z_rate(self): return self.loop.delta_z_rate

    def reset(self, delta_z=0., velocity=0.):
        self.loop.reset(delta_z, velocity)

    @staticmethod
    def _force(force):
        force = float(force)
        if not math.isfinite(force) or force < 0:
            raise ValueError('invalid pressing-positive wrist force')
        return force

    def step(self, desired_force, wrist_force):
        if float(desired_force) != 1.:
            raise ValueError('this diagnostic requires exactly 1 N target force')
        return self.loop.step(1., self._force(wrist_force))

    def observe(self, now, wrist_force):
        now, force = float(now), self._force(wrist_force)
        if not math.isfinite(now): raise ValueError('invalid physics clock')
        if self.last_time is not None:
            if now < self.last_time - 1e-10: raise ValueError('physics clock reversed')
            if now <= self.last_time + 1e-10: return
        self.last_time = now
        if self.phase == 'seek':
            if force >= .5:
                if self.force_since is None: self.force_since = now
                if now - self.force_since >= .1 - 1e-9:
                    delta, velocity = self.delta_z, self.delta_z_rate
                    self.loop = self._loop(4000.)
                    self.loop.reset(delta, velocity)
                    self.phase = 'settle'
                    self.transitions.append({'t': now, 'to': 'settle', 'wrist_force_N': force})
            else:
                self.force_since = None
        elif self.phase == 'settle':
            if .8 <= force <= 1.2:
                if self.stable_since is None: self.stable_since = now
                if now - self.stable_since >= .2 - 1e-9:
                    self.phase = 'maintain'
                    self.transitions.append({'t': now, 'to': 'maintain', 'wrist_force_N': force})
            else:
                self.stable_since = None
