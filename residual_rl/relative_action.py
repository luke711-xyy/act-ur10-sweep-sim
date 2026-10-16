"""ACT-conditioned feasible 25 Hz targets; never executes or changes Z control.

SAC densities remain densities of its original normalized three-dimensional
action. This state-conditioned map is the environment's deterministic actuator
interface, not an extra policy sample or a post-hoc action clipping operation.
"""
import math
import numpy as np

from .config import DEFAULT


def square_to_disk(xy):
    xy=np.asarray(xy,dtype=float)
    if xy.shape!=(2,) or not np.isfinite(xy).all() or np.any(np.abs(xy)>1.):
        raise ValueError('normalized XY coordinates must be in [-1,1]')
    x,y=xy
    return np.array([x*math.sqrt(1.-y*y/2.),y*math.sqrt(1.-x*x/2.)])


def generate_relative_action(actual_xy,native_target,action,*,bounds=None,ratio=.8,yaw_limit=math.pi/4):
    """Return a directly generated target and an auditable measured-state basis."""
    p=np.asarray(actual_xy,dtype=float);q=np.asarray(native_target);a=np.asarray(action,dtype=float)
    if (p.shape!=(2,) or q.shape!=(4,) or q.dtype.kind!='f' or a.shape!=(3,)
            or not np.isfinite(np.r_[p,q,a,ratio,yaw_limit]).all()
            or np.any(np.abs(a)>1.) or not 0.<=ratio<1. or not 0.<=yaw_limit<=math.pi):
        raise ValueError('invalid measured state / ACT target / relative action')
    box=None if bounds is None else np.asarray(bounds,dtype=float)
    if box is not None and (box.shape!=(2,2) or not np.isfinite(box).all()
            or np.any(box[:,0]>=box[:,1]) or np.any(q[:2]<box[:,0]-1e-7)
            or np.any(q[:2]>box[:,1]+1e-7)):
        raise ValueError('invalid native workspace / bounds')
    d=q[:2].astype(float)-p;length=float(np.linalg.norm(d));radius=ratio*length
    disk=square_to_disk(a[:2]);fraction=float(np.linalg.norm(disk))
    # Generate inside the representable disk, reserving coordinate-rounding room.
    # Sub-ULP moves must become zero, never a rounded correction of 100% of d.
    resolution_margin=4*float(np.linalg.norm(np.spacing(q[:2].astype(float))))
    correction=np.zeros(2);available=max(0.,radius-resolution_margin)
    if length>0. and fraction>0.:
        forward=d/length;left=np.array([-forward[1],forward[0]])
        direction=(disk[0]*forward+disk[1]*left)/fraction
        if box is not None:
            for axis in range(2):
                if direction[axis]>0.:
                    available=min(available,max(0.,(box[axis,1]-float(q[axis]))/direction[axis]))
                elif direction[axis]<0.:
                    available=min(available,max(0.,(box[axis,0]-float(q[axis]))/direction[axis]))
        correction=direction*(fraction*available)
    yaw_correction=float(a[2])*yaw_limit
    if not np.any(correction) and yaw_correction==0.:
        result=q.copy()  # Exact native bytes: no casts, wraps or re-integration.
    else:
        result=q.astype(float)
        result[:2]+=correction
        if yaw_correction:
            result[3]=math.atan2(math.sin(float(q[3])+yaw_correction),math.cos(float(q[3])+yaw_correction))
    actual_correction=result[:2].astype(float)-q[:2]
    yaw_actual=math.atan2(math.sin(float(result[3])-float(q[3])),math.cos(float(result[3])-float(q[3])))
    tolerance=32*np.finfo(float).eps*max(1.,float(np.linalg.norm(q[:2])))
    workspace_tolerance=max(tolerance,1e-7 if q.dtype==np.float32 else tolerance)
    if (np.linalg.norm(actual_correction)>radius*(1.+64*np.finfo(float).eps) or abs(yaw_actual)>yaw_limit+1e-12
            or result[2]!=q[2] or (box is not None and
                (np.any(result[:2]<box[:,0]-workspace_tolerance) or np.any(result[:2]>box[:,1]+workspace_tolerance)))):
        raise RuntimeError('relative action generation failed its independent geometry check')
    audit=dict(actual_xy=p.copy(),native_target=q.copy(),generated_target=result.copy(),
        original_xy_displacement=d,original_xy_length=length,correction_radius=radius,
        feasible_radius=available,resolution_margin=resolution_margin,normalized_disk=disk,xy_correction=actual_correction,
        yaw_correction=yaw_actual,yaw_limit=yaw_limit,ratio=ratio,Z_input_unchanged=True,
        bounds=box.copy() if box is not None else None,native_target_dtype=str(q.dtype))
    return result,audit


class RelativeActionGenerator:
    def __init__(self,bounds,config=DEFAULT):
        self.bounds=np.asarray(bounds,dtype=float).copy();self.config=config
        if self.bounds.shape!=(2,2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:,0]>=self.bounds[:,1]):
            raise ValueError('invalid XY workspace')
        self.filtered_action=np.zeros(3);self.offset=np.zeros(3);self.velocity=np.zeros(3)
        self.last_audit=None;self.frames=0

    def generate(self,actual_xy,native_target,action,*,dt,active,allow_motion=True):
        a=np.asarray(action,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.any(np.abs(a)>1.) or not np.isfinite(dt) or dt<=0.:
            raise ValueError('invalid relative action / clock')
        if not active and np.any(self.filtered_action):
            raise RuntimeError('relative gate cannot close during an episode')
        candidate=(self.filtered_action+(a-self.filtered_action)*
            (-np.expm1(-dt/self.config.offset_response_seconds))) if active else np.zeros(3)
        requested=candidate if active and allow_motion else np.zeros(3)
        target,audit=generate_relative_action(actual_xy,native_target,requested,bounds=self.bounds,
            ratio=getattr(self.config,'residual_xy_fraction',.8),
            yaw_limit=getattr(self.config,'relative_yaw_limit',math.pi/4))
        offset=np.r_[audit['xy_correction'],audit['yaw_correction']]
        velocity=(offset-self.offset)/dt
        audit.update(raw_action=a.copy(),filtered_action=candidate.copy(),generated_action=requested.copy(),
            action_contract=getattr(self.config,'action_contract','relative_xy80_yaw45_v1'),
            frame_index=self.frames+1,dt=float(dt),active=bool(active),allow_motion=bool(allow_motion),
            offset_before=self.offset.copy(),filter_before=self.filtered_action.copy())
        self.filtered_action=candidate;self.offset=offset;self.velocity=velocity
        self.frames+=1;self.last_audit=audit
        return target


def continuous_probe_action(elapsed):
    """Deterministic 25Hz test input, not a learned policy or random exploration."""
    if not np.isfinite(elapsed) or elapsed<0.: raise ValueError('invalid probe clock')
    ramp=.5-.5*math.cos(math.pi*min(float(elapsed),1.))
    phase=2*math.pi*elapsed/4.
    return ramp*np.array([.30*math.sin(phase),.30*math.cos(phase),.10*math.sin(2*math.pi*elapsed/5.)])
