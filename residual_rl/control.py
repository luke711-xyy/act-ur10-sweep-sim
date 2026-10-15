"""Velocity residual composition; never owns or changes the Z command."""
import numpy as np

from .config import DEFAULT


def limit_norm(value, bound):
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value * (min(1., bound / norm) if norm else 1.)


class ResidualGovernor:
    def __init__(self, bounds, config=DEFAULT):
        self.bounds = np.asarray(bounds, dtype=np.float64).copy()
        if (self.bounds.shape != (2,2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:,0] >= self.bounds[:,1])):
            raise ValueError('invalid XY workspace')
        self.config = config
        self.velocity = np.zeros(3, dtype=np.float64)
        self.last_audit = None

    def step(self, start, native, action, *, dt, active):
        start = np.asarray(start, dtype=np.float64)
        native = np.asarray(native)
        action = np.asarray(action, dtype=np.float64)
        if (start.shape != (4,) or native.shape != (4,) or action.shape != (3,)
                or not np.isfinite(np.r_[start,native,action,dt]).all() or dt <= 0
                or np.any(np.abs(action) > 1. + 1e-7)):
            raise ValueError('invalid normalized residual/command/clock')
        previous = self.velocity.copy()
        cfg = self.config
        desired = np.r_[limit_norm(action[:2],1.)*cfg.residual_xy_speed,
                        action[2]*cfg.residual_yaw_speed]
        if active:
            self.velocity[:2] += limit_norm(desired[:2]-self.velocity[:2],
                                           cfg.residual_xy_acceleration*dt)
            self.velocity[2] += np.clip(desired[2]-self.velocity[2],
                                       -cfg.residual_yaw_acceleration*dt,
                                       cfg.residual_yaw_acceleration*dt)
        else:
            if np.any(previous):
                raise RuntimeError('residual gate cannot close during an episode')
            self.velocity[:] = 0.
        # The exact native path includes its original float32 rounding.
        if not np.any(self.velocity):
            result = native.copy()
        else:
            nominal_xy = (native[:2].astype(float)-start[:2])/dt
            nominal_yaw = np.arctan2(np.sin(float(native[3])-start[3]),
                                     np.cos(float(native[3])-start[3]))/dt
            result = native.copy()
            result[:2] = np.clip(start[:2] + limit_norm(nominal_xy+self.velocity[:2],
                                    cfg.total_xy_speed)*dt,
                                    self.bounds[:,0],self.bounds[:,1])
            result[3] = start[3] + np.clip(nominal_yaw+self.velocity[2],
                                -cfg.total_yaw_speed,cfg.total_yaw_speed)*dt
        if (np.any(result[:2] < self.bounds[:,0]-1e-7)
                or np.any(result[:2] > self.bounds[:,1]+1e-7)):
            raise RuntimeError('native workspace violation')
        if result[2] != native[2]:
            raise RuntimeError('residual attempted to alter Z')
        self.last_audit = dict(raw_action=action.copy(), desired_residual=desired,
            filtered_residual=self.velocity.copy(), residual_acceleration=(self.velocity-previous)/dt,
            native=native.copy(), applied=result.copy(), Z_unchanged=True, active=bool(active))
        return result


class ResidualReference:
    """Production route accumulation shared with the isolated zero shadow."""
    def __init__(self,control):
        self.control=control
        self.previous_native=None; self.previous_applied=None

    def begin_frame(self,start):
        start=np.asarray(start)
        if start.shape!=(4,) or not np.isfinite(start).all():
            raise ValueError('invalid residual frame start')
        self.previous_native=start.copy(); self.previous_applied=start.copy()

    def step(self,raw,action,*,dt,active):
        if self.previous_applied is None: raise RuntimeError('residual frame was not initialized')
        raw=np.asarray(raw); action=np.asarray(action)
        synthetic=raw.copy()
        if np.any(self.control.velocity) or np.any(action):
            synthetic[:2]=self.previous_applied[:2]+raw[:2]-self.previous_native[:2]
            synthetic[3]=self.previous_applied[3]+np.arctan2(
                np.sin(raw[3]-self.previous_native[3]),np.cos(raw[3]-self.previous_native[3]))
        filter_before=self.control.velocity.copy()
        result=self.control.step(self.previous_applied,synthetic,action,dt=dt,active=active)
        self.control.last_audit.update(start=self.previous_applied.copy(),dt=float(dt),
            filter_before=filter_before,bounds=self.control.bounds.copy(),
            composition_native=synthetic.copy(),native=raw.copy(),
            native_before=self.previous_native.copy())
        self.previous_native=raw.copy(); self.previous_applied=result.copy()
        return result
