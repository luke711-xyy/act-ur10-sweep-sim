"""Disposable wrist-only gain search; total Z correction is bounded to 2 mm."""
import copy
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path

import numpy as np

BASE=Path(__file__).resolve().parent.parent/'hybrid_astar_act_gate1mm_acc50_2s_goal1_20261014'
_spec=importlib.util.spec_from_file_location('preserved_rate_phase_dependency',BASE/'phases.py')
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
sealed=previous.control.sealed
AdmittanceController1D=previous.control.AdmittanceController1D
AdmittanceParams=previous.control.AdmittanceParams

@dataclass(frozen=True)
class AdmittanceSpec:
    force_N:float=1.
    seek:tuple=(25.,1200.,1.)
    maintain:tuple=(25.,2000.,1.)

    def __post_init__(self):
        if self.force_N not in (1.,.3,.2):raise ValueError('only approved 1/0.3/0.2 N setpoints')
        for name in ('seek','maintain'):
            values=tuple(map(float,getattr(self,name)))
            if len(values)!=3:raise ValueError('M/B/K needs exactly three values')
            m,b,k=values
            if not all(map(math.isfinite,values)) or not (1.<=m<=100. and 100.<=b<=4000. and 1.<=k<=20.):
                raise ValueError('gain candidate outside bounded search scope')
            object.__setattr__(self,name,values)

    def check_clock(self,dt):
        if not math.isfinite(dt) or dt<=0:raise ValueError('invalid control clock')
        for m,b,k in (self.seek,self.maintain):
            matrix=[[1-dt*dt*k/m,dt*(1-dt*b/m)],[-dt*k/m,1-dt*b/m]]
            if max(abs(np.linalg.eigvals(matrix)))>=1.:raise ValueError('discrete gain candidate unstable')

class TwoStageAdmittance(sealed.TwoStageAdmittance):
    def __init__(self,dt,spec):
        self.spec=spec;spec.check_clock(float(dt));super().__init__(dt)

    def _loop(self,marker):
        if marker==400.:gains=self.spec.seek
        elif marker==4000.:gains=self.spec.maintain
        else:raise RuntimeError('preserved stage factory contract changed')
        return AdmittanceController1D(AdmittanceParams(*gains),self.dt,
            delta_limit=.002,rate_limit=.1,direction=-1.,
            upward_rate_limit=.02,downward_rate_limit=.1)

    def step(self,desired_force,wrist_force):
        if float(desired_force)!=self.spec.force_N:raise ValueError('setpoint differs from recorded trial')
        return self.loop.step(self.spec.force_N,self._force(wrist_force))

    def observe(self,now,wrist_force):
        now,force=float(now),self._force(wrist_force)
        if not math.isfinite(now):raise ValueError('invalid physics clock')
        if self.last_time is not None:
            if now<self.last_time-1e-10:raise ValueError('physics clock reversed')
            if now<=self.last_time+1e-10:return
        self.last_time=now
        if self.phase=='seek':
            if force>=.5*self.spec.force_N:
                if self.force_since is None:self.force_since=now
                if now-self.force_since>=.1-1e-9:
                    delta,velocity=self.delta_z,self.delta_z_rate
                    self.loop=self._loop(4000.);self.loop.reset(delta,velocity)
                    self.phase='settle'
                    self.transitions.append({'t':now,'to':'settle','wrist_force_N':force})
            else:self.force_since=None
        elif self.phase=='settle':
            if .8*self.spec.force_N-1e-12<=force<=1.2*self.spec.force_N+1e-12:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.2-1e-9:
                    self.phase='maintain'
                    self.transitions.append({'t':now,'to':'maintain','wrist_force_N':force})
            else:self.stable_since=None

class ZCommandLimiter(previous.ZCommandLimiter):
    """Reuse exactly the preceding 20/100 mm/s accepted-command limiter."""

def compose_z(nominal,actual_z,wrist_force,z_min,loop,limiter,relief):
    if not all(map(math.isfinite,map(float,(nominal,actual_z,wrist_force,z_min)))):
        raise ValueError('nonfinite position or wrist feedback')
    if z_min>nominal+.002:raise ValueError('workspace floor incompatible with two-mm envelope')
    correction=loop.step(loop.spec.force_N,wrist_force)+(0. if relief is None else relief.step(wrist_force))
    native=max(float(z_min),float(nominal)+max(-.002,min(.002,correction)))
    details=limiter.tracking.propose(nominal=nominal,native_z=native,
        last_command_z=limiter.last_z,actual_z=actual_z,wrist_force=wrist_force,z_min=z_min)
    requested=details['requested_z']
    details['requested_z']=max(z_min,nominal-.002,min(nominal+.002,requested))
    details['total_2mm_bound_clipped']=abs(details['requested_z']-requested)>1e-12
    z=limiter.apply(details['requested_z'],wrist_force,down=.1)
    if abs(z-nominal)>.002+1e-9:raise RuntimeError('composed Z outside total2mm range')
    limiter.last_proposal={**details,'nominal_z':float(nominal),'applied_z':z,
        'proposal_time':limiter.last_time,'force_used_N':float(wrist_force),
        'target_force_N':loop.spec.force_N,'total_correction_limit_m':.002}
    return z

def attach_follow(row,limiter):
    previous.attach_follow(row,limiter)
