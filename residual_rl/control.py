"""Historical fixed-pose helper, retained for old characterization tests only.

The current execution path uses relative_action.RelativeActionGenerator.
"""
import numpy as np

from .config import LEGACY_POSE as DEFAULT
from .pose_constraints import disk,halfplane,intersection


def limit_norm(value,bound):
    value=np.asarray(value,dtype=float); norm=np.linalg.norm(value)
    return value*min(1.,bound/norm) if norm else value.copy()


def wrap(value):
    return np.arctan2(np.sin(value),np.cos(value))


class ResidualGovernor:
    def __init__(self,bounds,config=DEFAULT):
        self.bounds=np.asarray(bounds,dtype=float).copy()
        if (self.bounds.shape!=(2,2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:,0]>=self.bounds[:,1])):
            raise ValueError('invalid XY workspace')
        self.config=config; self.offset=np.zeros(3); self.velocity=np.zeros(3); self.last_audit=None

    def remove_offset(self,reference):
        reference=np.asarray(reference)
        if reference.shape!=(4,) or not np.isfinite(reference).all():
            raise ValueError('invalid native anchor reference')
        if not np.any(self.offset): return reference.copy()
        result=reference.astype(float)
        result[:2]-=self.offset[:2]; result[3]-=self.offset[2]
        return result.astype(reference.dtype)

    def step(self,start,native,action,*,dt,active,native_before=None):
        start=np.asarray(start); native=np.asarray(native); action=np.asarray(action,dtype=float)
        previous_native=self.remove_offset(start) if native_before is None else np.asarray(native_before)
        if (start.shape!=(4,) or native.shape!=(4,) or previous_native.shape!=(4,) or action.shape!=(3,)
                or not np.isfinite(np.r_[start,native,previous_native,action,dt]).all() or dt<=0
                or np.any(np.abs(action)>1.+1e-7)):
            raise ValueError('invalid normalized pose action/reference/clock')
        cfg=self.config; old=self.offset.copy()
        if not active and np.any(old): raise RuntimeError('residual gate cannot close during an episode')
        desired=np.r_[limit_norm(action[:2],1.)*cfg.residual_xy_offset,action[2]*cfg.residual_yaw_offset]
        delta=np.r_[native[:2].astype(float)-previous_native[:2],wrap(float(native[3])-float(previous_native[3]))]
        if np.any(native[:2]<self.bounds[:,0]-1e-7) or np.any(native[:2]>self.bounds[:,1]+1e-7):
            raise RuntimeError('native workspace violation')
        result=native.copy(); candidate=old.copy()
        if active and (np.any(old) or np.any(action)):
            request=old+(desired-old)*(-np.expm1(-dt/cfg.offset_response_seconds))
            distance=float(np.linalg.norm(delta[:2]))
            if distance>1e-10:
                direction=delta[:2]/distance; normal=np.array([-direction[1],direction[0]])
                slope=np.tan(cfg.direction_cone_radians)
                projections=[disk(np.zeros(2),cfg.residual_xy_offset),disk(old[:2],cfg.residual_xy_speed*dt),
                    disk(start[:2].astype(float)-native[:2],cfg.total_xy_speed*dt),
                    halfplane(-direction,-float(direction@old[:2])+(1-cfg.min_forward_fraction)*distance)]
                for axis in range(2):
                    normal_axis=np.eye(2)[axis]
                    projections.extend([halfplane(normal_axis,self.bounds[axis,1]-float(native[axis])),
                        halfplane(-normal_axis,float(native[axis])-self.bounds[axis,0])])
                for plane in (normal-slope*direction,-normal-slope*direction):
                    projections.append(halfplane(plane,float(plane@old[:2])+slope*distance))
                candidate[:2]=intersection(request[:2],projections)
            # Native XY idle freezes displacement, not a snap back to zero.
            if distance>1e-10 or abs(delta[2])>1e-10:
                span=cfg.residual_yaw_speed*dt
                low=max(-cfg.residual_yaw_offset,old[2]-span,old[2]-delta[2]-cfg.total_yaw_speed*dt)
                high=min(cfg.residual_yaw_offset,old[2]+span,old[2]-delta[2]+cfg.total_yaw_speed*dt)
                if delta[2]>1e-10: low=max(low,old[2]-(1-cfg.min_forward_fraction)*delta[2])
                if delta[2]<-1e-10: high=min(high,old[2]-(1-cfg.min_forward_fraction)*delta[2])
                if low>high+1e-9: raise RuntimeError('yaw pose offset constraints infeasible')
                # Select within the interval in the actual command dtype, not
                # a real-valued interval whose rounded endpoint can exceed 3 deg.
                lower=np.asarray(float(native[3])+low,dtype=native.dtype)
                upper=np.asarray(float(native[3])+max(low,high),dtype=native.dtype)
                if float(lower)<float(native[3])+low: lower=np.nextafter(lower,np.asarray(np.inf,dtype=native.dtype))
                if float(upper)>float(native[3])+max(low,high): upper=np.nextafter(upper,np.asarray(-np.inf,dtype=native.dtype))
                if lower>upper: raise RuntimeError('yaw pose constraints infeasible in command dtype')
                value=np.clip(np.asarray(float(native[3])+request[2],dtype=native.dtype),lower,upper)
                candidate[2]=float(value)-float(native[3])
            result[:2]=native[:2].astype(float)+candidate[:2]; result[3]=float(native[3])+candidate[2]
        applied=np.r_[result[:2].astype(float)-native[:2],wrap(float(result[3])-float(native[3]))]
        velocity=(applied-old)/dt
        actual=np.r_[result[:2].astype(float)-start[:2],wrap(float(result[3])-float(start[3]))]
        # Validate after native float32 rounding, before committing any state.
        if (np.linalg.norm(applied[:2])>cfg.residual_xy_offset+1e-7 or abs(applied[2])>cfg.residual_yaw_offset+1e-7
                or np.linalg.norm(velocity[:2])>cfg.residual_xy_speed+2e-5
                or abs(velocity[2])>cfg.residual_yaw_speed+2e-5
                or np.any(result[:2]<self.bounds[:,0]-1e-7) or np.any(result[:2]>self.bounds[:,1]+1e-7)):
            raise RuntimeError('pose offset constraints infeasible after output quantization')
        if np.any(old) or np.any(action):
            if np.linalg.norm(actual[:2])>cfg.total_xy_speed*dt+2e-7 or abs(actual[2])>cfg.total_yaw_speed*dt+2e-7:
                raise RuntimeError('pose total speed constraints infeasible')
            distance=np.linalg.norm(delta[:2])
            if distance>1e-10:
                direction=delta[:2]/distance; forward=actual[:2]@direction
                side=actual[:2]@np.array([-direction[1],direction[0]])
                if forward<cfg.min_forward_fraction*distance-2e-7 or abs(side)>np.tan(cfg.direction_cone_radians)*forward+2e-7:
                    raise RuntimeError('pose forward progress constraints infeasible')
        if result[2]!=native[2]: raise RuntimeError('residual attempted to alter Z')
        self.offset=applied; self.velocity=velocity
        self.last_audit=dict(raw_action=action.copy(),desired_offset=desired,offset_before=old,
            applied_offset=applied.copy(),filtered_residual=velocity.copy(),native=native.copy(),
            native_before=previous_native.copy(),start=start.copy(),applied=result.copy(),
            dt=float(dt),bounds=self.bounds.copy(),Z_unchanged=True,active=bool(active),action_contract=cfg.action_contract)
        return result


class ResidualReference:
    def __init__(self,control):
        self.control=control; self.previous_native=None; self.previous_applied=None

    def begin_frame(self,start):
        self.previous_applied=np.asarray(start).copy(); self.previous_native=self.control.remove_offset(start)
        return self.previous_native.copy()

    def step(self,raw,action,*,dt,active):
        if self.previous_applied is None: raise RuntimeError('residual frame was not initialized')
        result=self.control.step(self.previous_applied,raw,action,dt=dt,active=active,native_before=self.previous_native)
        self.previous_native=np.asarray(raw).copy(); self.previous_applied=result.copy()
        return result


def nominal_reference_executor(base,control):
    """Snapshot synchronously; original async scheduling stays intact."""
    class NominalReferenceExecutor(base):
        def submit(self,fn,observation,start_reference):
            return super().submit(fn,observation,control.remove_offset(start_reference))
    return NominalReferenceExecutor
