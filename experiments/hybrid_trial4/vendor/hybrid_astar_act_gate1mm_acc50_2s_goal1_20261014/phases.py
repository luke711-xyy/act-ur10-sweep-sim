"""THROWAWAY: restore only the formal 20/100 mm/s direction ceilings."""
import copy,importlib.util,math
from pathlib import Path

SMALL=Path(__file__).resolve().parent.parent/'admittance_wrist1n_small_motion_20261014'
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,SMALL/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

control=load('frozen_small_motion_controller','control.py')
follow=load('frozen_z_follow_controller','follow.py')

class TwoStageAdmittance(control.ProbeController):
    def __init__(self,dt):
        super().__init__(dt,seek=(25.,1200.,1.),maintain=(25.,2000.,1.),up=.002,maintain_down=.002)
        self.up=.02

    def _loop(self,damping_marker):
        loop=super()._loop(damping_marker)
        loop.upward_rate_limit=.02
        loop.downward_rate_limit=.1
        return loop

class ZCommandLimiter(control.ProbeLimiter):
    def __init__(self,initial_z,dt,initial_time=0.):
        super().__init__(initial_z,dt,up=.002,initial_time=initial_time)
        self.up=.02
        self.tracking=follow.TrackingCompensator(.5)
        self.last_proposal=None

    def apply(self,desired_z,wrist_force,*,down=.1):
        desired_z=float(desired_z)
        force=self._force(wrist_force)
        if not math.isfinite(desired_z):raise ValueError('nonfinite Z command')
        down=float(down)
        if not math.isfinite(down) or not 0.<down<=.1:raise ValueError('invalid descent ceiling')
        up=.0201 if force>=10. else self.up
        return max(self.last_z-down*self.dt,min(self.last_z+up*self.dt,desired_z))

def compose_z(nominal,actual_z,wrist_force,z_min,loop,limiter,relief):
    correction=loop.step(1.,wrist_force)+(0. if relief is None else relief.step(wrist_force))
    native=max(float(z_min),float(nominal)+max(-.005,min(.005,correction)))
    details=limiter.tracking.propose(nominal=nominal,native_z=native,
        last_command_z=limiter.last_z,actual_z=actual_z,wrist_force=wrist_force,z_min=z_min)
    z=limiter.apply(details['requested_z'],wrist_force,down=.1)
    if abs(z-nominal)>.005+1e-9:raise RuntimeError('composed Z outside total5mm range')
    limiter.last_proposal={**details,'nominal_z':float(nominal),'applied_z':z,
        'proposal_time':limiter.last_time,'force_used_N':float(wrist_force)}
    return z

def attach_follow(row,limiter):
    p=limiter.last_proposal
    if (p is not None and float(row['t'])>limiter.last_time+1e-10
            and abs(float(row['command'][2])-p['applied_z'])<1e-8):
        row['z_follow']=copy.deepcopy(p)

