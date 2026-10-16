"""Independent arithmetic audit of recorded 25Hz relative targets / 100Hz controls."""
import gzip
import json
import math
import numpy as np

from .config import DEFAULT
from .relative_action import continuous_probe_action


def audit_relative_rollout(rollout):
    row=rollout.row
    if row.get('error') or row.get('outcome')=='safety' or row.get('accepted_ACT_chunks',0)<1:
        raise RuntimeError('relative preflight stopped on physical / ACT failure')
    def vec(value,n):
        a=np.asarray(value,dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('invalid vector')
        return a
    def close(a,b,tol=1e-7):
        if not np.allclose(a,b,rtol=0.,atol=tol):raise ValueError('inconsistent saved relative evidence')
    def wrap(x):return math.atan2(math.sin(x),math.cos(x))
    frames=active=count=changing=exercised=xy_exercised=yaw_exercised=0
    previous=None;last_frame=None;previous_filter=np.zeros(3);latched=False;last_sub=0
    applied_frames=set()
    previous_raw=None;max_ratio=max_yaw=max_xy=max_speed=max_yaw_speed=0.
    try:
        with gzip.open(rollout.trace_path,'rt') as stream:
            for line in stream:
                r=json.loads(line);count+=1
                if r['control_index']!=count:raise ValueError('missing physical control record')
                command=vec(r['command'],4);before=vec(r['command_before'],4)
                tcp=vec(r['tcp'],3);tcp_before=vec(r['tcp_before'],3)
                t=float(r['t']);began=float(r['command_started_at']);close(t-began,.01)
                if previous is not None:
                    close(began,previous['t']);close(before,previous['command']);close(tcp_before,previous['tcp'])
                if (latched or r.get('phase')=='ACT_residual') and not r['active']:
                    raise ValueError('active control escaped the latched residual audit')
                if r['active']:
                    latched=True
                    active+=1;a=r['control_audit']
                    if (not a or a['action_contract']!=DEFAULT.action_contract or not a['active']
                            or a['filter_accepted'] is not True or a['Z_input_unchanged'] is not True
                            or not a['contact_latched'] or r['probe_stage']!='continuous_micro_adjust'):
                        raise ValueError('missing accepted relative / Z ownership audit')
                    sub=int(a['substep']);n=int(a['substeps']);dt=float(a['control_dt'])
                    if n!=4 or not 1<=sub<=4:raise ValueError('not original 25/100Hz control structure')
                    close(dt,.01);close(a['dt'],.04)
                    close(began,float(a['frame_started_at'])+(sub-1)*dt)
                    close(a['applied'],command);close(a['control_reference'],command)
                    q=vec(a['native_target'],4);generated=vec(a['generated_target'],4);p=vec(a['actual_xy'],2)
                    d=q[:2]-p;length=float(np.linalg.norm(d));correction=generated[:2]-q[:2]
                    close(a['original_xy_displacement'],d);close(a['original_xy_length'],length,1e-12)
                    close(a['ratio'],.8);close(a['correction_radius'],.8*length,1e-12)
                    close(a['xy_correction'],correction)
                    if np.linalg.norm(correction)>.8*length+1e-10:raise ValueError('80% correction exceeded')
                    if (generated[:2]-p)@d<.2*(d@d)-1e-12:raise ValueError('generated XY reversed ACT')
                    yaw=wrap(generated[3]-q[3]);close(yaw,a['yaw_correction'])
                    if abs(yaw)>math.pi/4+1e-10 or generated[2]!=q[2]:raise ValueError('yaw/Z generation violation')
                    box=np.asarray(a['bounds'],dtype=float)
                    workspace_tolerance=1e-7 if a.get('native_target_dtype')=='float32' else 1e-9
                    if (box.shape!=(2,2) or not np.isfinite(box).all() or np.any(box[:,0]>=box[:,1])
                            or np.any(q[:2]<box[:,0]-workspace_tolerance) or np.any(q[:2]>box[:,1]+workspace_tolerance)
                            or np.any(generated[:2]<box[:,0]-workspace_tolerance) or np.any(generated[:2]>box[:,1]+workspace_tolerance)):
                        raise ValueError('generated workspace violation')
                    current=int(a['frame_index'])
                    if last_frame is None or current!=last_frame['frame_index']:
                        if sub!=1:raise ValueError('missing first control of action frame')
                        close(p,tcp_before[:2])
                        if last_frame is not None:
                            if current!=last_frame['frame_index']+1 or last_sub!=4:raise ValueError('missing 25Hz frame / substep')
                            close(a['frame_started_at']-last_frame['frame_started_at'],.04)
                        close(a['filter_before'],previous_filter,1e-12)
                        raw=vec(a['raw_action'],3)
                        close(raw,continuous_probe_action(float(a['residual_elapsed'])),1e-10)
                        filtered=previous_filter+(raw-previous_filter)*(-math.expm1(-.04/DEFAULT.offset_response_seconds))
                        close(filtered,a['filtered_action'],1e-12)
                        chosen=filtered if a['allow_motion'] else np.zeros(3)
                        close(chosen,a['generated_action'],1e-12)
                        x,y=chosen[:2]
                        disk=np.array([x*math.sqrt(1-y*y/2),y*math.sqrt(1-x*x/2)])
                        close(disk,a['normalized_disk'],1e-12)
                        # Independently reconstruct ACT-forward/left direction and the
                        # workspace-conditioned radius, not merely its recorded norm.
                        fraction=float(np.linalg.norm(disk));expected=np.zeros(2)
                        margin=4*float(np.linalg.norm(np.spacing(q[:2])))
                        available=max(0.,.8*length-margin)
                        if length and fraction:
                            forward=d/length;left=np.array([-forward[1],forward[0]])
                            direction=(disk[0]*forward+disk[1]*left)/fraction
                            for axis in range(2):
                                if direction[axis]>0.:
                                    available=min(available,max(0.,(box[axis,1]-q[axis])/direction[axis]))
                                elif direction[axis]<0.:
                                    available=min(available,max(0.,(box[axis,0]-q[axis])/direction[axis]))
                            expected=direction*(fraction*available)
                        close(available,a['feasible_radius'],1e-12)
                        close(correction,expected,1e-12)
                        close(yaw,float(chosen[2])*math.pi/4,1e-10)
                        exercised+=int(bool(a['allow_motion']) and (np.linalg.norm(correction)>1e-8 or abs(yaw)>1e-8))
                        xy_exercised+=int(np.linalg.norm(correction)>1e-8)
                        yaw_exercised+=int(abs(yaw)>1e-8)
                        changing+=int(previous_raw is not None and np.linalg.norm(raw-previous_raw)>1e-8)
                        previous_raw=raw;previous_filter=filtered;frames+=1;last_frame=a
                    else:
                        if sub!=last_sub+1:raise ValueError('missing active control substep')
                        for key in ('native_target','generated_target','actual_xy','raw_action','filtered_action'):
                            close(a[key],last_frame[key],1e-12)
                    last_sub=sub
                    start=vec(a['execution_start'],4)
                    delta=generated[:2]-start[:2];distance=float(np.linalg.norm(delta))
                    if distance>.08*.04:delta*=.08*.04/distance
                    close(command[:2],start[:2]+delta*sub/4.)
                    angular=wrap(generated[3]-start[3]);angular=max(-.35*.04,min(.35*.04,angular))
                    close(wrap(command[3]-(start[3]+angular*sub/4.)),0.)
                    close(command[2],np.float32(a['admittance_z']),1e-10)
                    native_delta=q[:2]-start[:2];native_distance=float(np.linalg.norm(native_delta))
                    if native_distance>.08*.04:native_delta*=.08*.04/native_distance
                    native_xy=start[:2]+native_delta*sub/4.
                    native_yaw=start[3]+max(-.35*.04,min(.35*.04,wrap(q[3]-start[3])))*sub/4.
                    if np.linalg.norm(command[:2]-native_xy)>1e-7 or abs(wrap(command[3]-native_yaw))>1e-7:
                        applied_frames.add(current)
                    max_speed=max(max_speed,float(np.linalg.norm((command[:2]-before[:2])/.01)))
                    max_yaw_speed=max(max_yaw_speed,abs(wrap(command[3]-before[3])/.01))
                    max_ratio=max(max_ratio,float(np.linalg.norm(correction))/length if length else 0.)
                    max_xy=max(max_xy,float(np.linalg.norm(correction)));max_yaw=max(max_yaw,abs(yaw))
                previous=r
        if (count!=row['control_steps'] or frames<50 or changing<frames//2
                or exercised<50 or xy_exercised<25 or yaw_exercised<25 or len(applied_frames)<25):
            raise ValueError('continuous 25Hz exercise incomplete')
        if max_speed>.08002 or max_yaw_speed>.35002:raise ValueError('original execution speed exceeded')
    except (ValueError,KeyError,TypeError,OverflowError,OSError,EOFError) as error:
        raise RuntimeError(f'relative physical audit failed: {error}') from error
    return dict(passed=True,action_frames=frames,changing_action_frames=changing,
        exercised_action_frames=exercised,xy_corrected_frames=xy_exercised,yaw_corrected_frames=yaw_exercised,
        physically_changed_reference_frames=len(applied_frames),
        audited_control_steps=active,max_relative_xy_correction=max_ratio,max_xy_correction_m=max_xy,
        max_yaw_correction_rad=max_yaw,max_command_xy_m_s=max_speed,max_command_yaw_rad_s=max_yaw_speed,
        Z_changes=0,continuous_contact_verified=False,action_contract=DEFAULT.action_contract)
