"""Opt-in bridge to the existing trial-4 1N, two-stage wrist controller."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def phase_module():
    path = Path(__file__).resolve().parents[2]/'experiments/hybrid_trial4/vendor/admittance_2mm_target_gain_search_goal1_20261014/phases.py'
    spec = importlib.util.spec_from_file_location('native_collection_trial4_phases',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trial4_config(cfg):
    cfg = cfg.copy()
    for name,value in {'desired_force':1.,'admittance.m':10.,'admittance.b':400.,
                       'admittance.k':10.,'delta_z_limit':.002,
                       'delta_z_upward_rate_limit':.02,'delta_z_downward_rate_limit':.1}.items():
        cfg.set_path('controller.'+name,value)
    cfg.set_path('scene.contact_profile','trial4_1n')
    return cfg


def demonstration_config(cfg):
    cfg = trial4_config(cfg)
    # The normal-force sole represents compliant bristles, not a rigid rubber
    # fingertip. Preserve the plate/part contact and all part/table friction.
    cfg.set_path('end_effector.brush_sole_friction',[.1,.0001,.0001])
    return cfg


class Trial4Contact:
    def __init__(self,env):
        self.env = env
        self.phase = phase_module()
        self.spec = self.phase.AdmittanceSpec(1.,(10.,400.,10.),(10.,800.,10.))
        self.reset()

    def reset(self):
        self.loop = self.phase.TwoStageAdmittance(self.env.control_dt,self.spec)
        self.limiter = None
        self.pending = None

    @property
    def ready(self):
        return self.loop.ready

    def command_z(self,cfg,nominal,force,relief):
        now = self.env.time
        if self.limiter is None:
            self.limiter = self.phase.ZCommandLimiter(float(self.env.tcp()[2]),self.env.control_dt,initial_time=now)
        elif self.pending is not None:
            self.limiter.commit(self.pending,now)
        self.loop.observe(now,force)
        z = self.phase.compose_z(nominal,float(self.env.tcp()[2]),force,
                                float(cfg.workspace.z_search_min),self.loop,self.limiter,relief)
        self.pending = z
        return z
