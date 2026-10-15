"""Disposable live ACT XY/yaw governor; enabled for exactly two seconds only.

The timer starts at the first executable valid ACT action, not at chunk
acceptance, model loading or the A* force gate. After expiry, return the original interpolator output
unchanged. Acceleration bounds intentionally do NOT cover that release or
later ACT motion. No Z controller, force sensor, or termination rule lives here.
"""
import copy
import math
import numpy as np


def angle_delta(value):
    return float(math.atan2(math.sin(value), math.cos(value)))


def stopping_speed(distance, acceleration, dt):
    # Conservative discrete stopping distance, with no endpoint snapping.
    return math.sqrt((acceleration*dt)**2 + 2.*acceleration*distance) - acceleration*dt


class TwoSecondHandoff:
    def __init__(self, reference, dt, now, bounds, velocity_xy=(0.,0.), yaw_velocity=0.):
        self.last_reference = np.asarray(reference,dtype=np.float64).reshape(4).copy()
        self.dt, self.last_time = float(dt), float(now)
        self.bounds = np.asarray(bounds,dtype=np.float64).reshape(2,2)
        self.velocity_xy = np.asarray(velocity_xy,dtype=np.float64).reshape(2).copy()
        self.yaw_velocity = float(yaw_velocity)
        if (not np.isfinite(np.r_[self.last_reference,self.dt,self.last_time,
                self.bounds.ravel(),self.velocity_xy,self.yaw_velocity]).all()
                or self.dt <= 0. or np.any(self.bounds[:,0]>=self.bounds[:,1])
                or np.linalg.norm(self.velocity_xy)>.08+1e-8
                or abs(self.yaw_velocity)>.35+1e-8):
            raise ValueError('invalid accepted handoff state')
        if np.any(self.last_reference[:2]<self.bounds[:,0]) or np.any(self.last_reference[:2]>self.bounds[:,1]):
            raise ValueError('handoff reference outside workspace')
        self.started_at = None
        self.armed = False
        self.first_native_time = None
        self.pending = None
        self.frame_targets = []

    def chunk_ack(self, accepted, now):
        if not accepted or self.started_at is not None: return
        if not math.isfinite(float(now)) or abs(float(now)-self.last_time)>1e-8:
            raise RuntimeError('ACT chunk clock differs from accepted physical clock')
        self.armed = True

    def action_ready(self, now):
        if self.started_at is not None: return
        if not self.armed: raise RuntimeError('executable ACT action without accepted chunk')
        if not math.isfinite(float(now)) or abs(float(now)-self.last_time)>1e-8:
            raise RuntimeError('ACT action clock differs from accepted physical clock')
        self.started_at = float(now)

    def propose(self, now, native, target, admittance_z):
        native = np.asarray(native).reshape(4)
        target = np.asarray(target,dtype=np.float64).reshape(4)
        if not np.isfinite(np.r_[now,native,target,admittance_z]).all():
            raise ValueError('nonfinite ACT reference')
        if abs(float(now)-self.last_time)>1e-8:
            raise RuntimeError('reference clock differs from last accepted step')
        elapsed = None if self.started_at is None else float(now)-self.started_at
        active = elapsed is not None and elapsed < 2.-1e-9
        mode = 'waiting_for_ACT' if elapsed is None else 'handoff_2s' if active else 'native_restored'
        command = native.copy()
        if active:
            delta = target[:2]-self.last_reference[:2]
            distance = float(np.linalg.norm(delta))
            desired = (delta/distance*min(.08,stopping_speed(distance,.05,self.dt))
                       if distance>1e-12 else np.zeros(2))
            change = desired-self.velocity_xy
            length = float(np.linalg.norm(change))
            velocity = self.velocity_xy + change*min(1.,.05*self.dt/length) if length>1e-12 else desired
            yaw_error = angle_delta(target[3]-self.last_reference[3])
            desired_yaw = math.copysign(min(.35,stopping_speed(abs(yaw_error),.5,self.dt)),yaw_error)
            yaw_velocity = self.yaw_velocity + float(np.clip(desired_yaw-self.yaw_velocity,-.5*self.dt,.5*self.dt))
            command = self.last_reference.copy()
            command[:2] += velocity*self.dt
            command[2] = float(admittance_z)
            command[3] += yaw_velocity*self.dt
            if np.any(command[:2]<self.bounds[:,0]) or np.any(command[:2]>self.bounds[:,1]):
                raise RuntimeError('smooth handoff cannot remain inside workspace; no clamping/retry')
        velocity = (command[:2]-self.last_reference[:2])/self.dt
        yaw_velocity = angle_delta(float(command[3])-self.last_reference[3])/self.dt
        self.pending = dict(mode=mode,proposal_time=float(now),elapsed_s=elapsed,
            started_at=self.started_at,window_s=2.,command=command.copy(),
            command_before=self.last_reference.copy(),raw_ACT_target=target.copy(),
            native_reference=native.copy(),velocity_xy_m_s=velocity.copy(),
            yaw_velocity_rad_s=yaw_velocity,
            acceleration_xy_m_s2=(velocity-self.velocity_xy)/self.dt,
            yaw_acceleration_rad_s2=(yaw_velocity-self.yaw_velocity)/self.dt,
            xy_speed_limit_m_s=.08,xy_acceleration_limit_m_s2=.05,
            yaw_speed_limit_rad_s=.35,yaw_acceleration_limit_rad_s2=.5,
            acceleration_bounds_apply=active,Z_unchanged=True)
        return command

    def prepare_record(self, row):
        if self.pending is None: raise RuntimeError('physical row without reference proposal')
        details = copy.deepcopy(self.pending)
        advanced = bool(row.get('control_step_advanced',True)) and float(row['t'])>self.last_time+1e-9
        actual = np.asarray(row['command'],dtype=np.float64).reshape(4)
        clock_ok = abs(float(row['t'])-(self.last_time+self.dt))<1e-8
        # Native trace stores float32; allow only its rounding, never a hold fallback.
        match = bool(np.isfinite(actual).all() and
            np.max(np.abs(actual[:3]-details['command'][:3]))<1e-7 and
            abs(angle_delta(actual[3]-details['command'][3]))<1e-7)
        details.update(physical_time=float(row['t']),advanced=advanced,
                       accepted=advanced and clock_ok and match,
                       applied_command=actual.copy(),command_match=match,clock_match=clock_ok)
        row['handoff_smoothing'] = details

    def acknowledge(self, row):
        details = row['handoff_smoothing']
        if not details['advanced']: return
        if not details['accepted']:
            raise RuntimeError('accepted command differs from handoff proposal or physical clock')
        self.last_reference = np.asarray(details['command'],dtype=np.float64).copy()
        self.last_time = float(row['t'])
        self.velocity_xy = np.asarray(details['velocity_xy_m_s'],dtype=np.float64).copy()
        self.yaw_velocity = float(details['yaw_velocity_rad_s'])
        if details['mode']=='native_restored' and self.first_native_time is None:
            self.first_native_time = float(details['proposal_time'])


def install(adapted, env, governor):
    """Wire into the copied run_act_episode globals, never native module globals."""
    namespace = adapted.__globals__
    native = namespace['_policy_reference_substep']
    record = namespace['_trial_record']
    ack = namespace['_trial_chunk_ack']
    native_scheduler = namespace['ActionChunkScheduler']

    class ObservedScheduler(native_scheduler):
        def action_for(self, now):
            action = super().action_for(now)
            if action is not None:
                values = np.asarray(action)
                if values.shape==(4,) and np.isfinite(values).all():
                    governor.action_ready(float(now))
            return action

    def chunk_ack(accepted):
        ack(accepted)
        governor.chunk_ack(accepted,float(env.time))

    def reference(start,target,substep,substeps,dt,max_xy_speed,max_z_speed,
                  max_yaw_rate,contact_latched,admittance_z=None):
        if not contact_latched or admittance_z is None:
            raise RuntimeError('handoff is allowed only with existing Z admittance ownership')
        if (abs(float(dt)-governor.dt)>1e-12 or float(max_xy_speed)>.08+1e-8
                or float(max_yaw_rate)>.35+1e-8):
            raise RuntimeError('native rate settings outside approved handoff scope')
        raw = native(start,target,substep,substeps,dt,max_xy_speed,max_z_speed,
                     max_yaw_rate,contact_latched,admittance_z)
        if int(substep)==1:
            governor.frame_targets.append(np.asarray(target,dtype=np.float32).copy())
        return governor.propose(float(env.time),raw,target,admittance_z)

    def capture(row,current_env):
        governor.prepare_record(row)
        record(row,current_env)  # Retain the genuine physical row before validation fails.
        governor.acknowledge(row)

    namespace.update(_policy_reference_substep=reference,_trial_record=capture,
                     _trial_chunk_ack=chunk_ack,ActionChunkScheduler=ObservedScheduler)


def partial_actions(prefix, governor):
    selected = [] if governor is None else governor.frame_targets
    return np.asarray(list(prefix)+list(selected),dtype=np.float32).reshape(-1,4)


def metrics(rows):
    records = [r for r in rows if r.get('handoff_smoothing',{}).get('advanced')]
    active = [r for r in records if r['handoff_smoothing']['mode']=='handoff_2s']
    restored = [r for r in records if r['handoff_smoothing']['mode']=='native_restored']
    violations = [r['t'] for r in active if
        np.linalg.norm(r['handoff_smoothing']['velocity_xy_m_s'])>.08+1e-8 or
        np.linalg.norm(r['handoff_smoothing']['acceleration_xy_m_s2'])>.05+1e-8 or
        abs(r['handoff_smoothing']['yaw_velocity_rad_s'])>.35+1e-8 or
        abs(r['handoff_smoothing']['yaw_acceleration_rad_s2'])>.5+1e-8]
    return dict(active_steps=len(active),restored_native_steps=len(restored),
        accepted_ACT_control_steps=sum(r['handoff_smoothing']['accepted'] for r in active+restored),
        first_active_time=(active[0]['handoff_smoothing']['proposal_time'] if active else None),
        first_native_time=(restored[0]['handoff_smoothing']['proposal_time'] if restored else None),
        active_limit_violations=violations,
        active_peak_wrist_N=max((float(r['normal_force']) for r in active),default=None),
        restored_peak_wrist_N=max((float(r['normal_force']) for r in restored),default=None),
        release_step=(restored[0]['handoff_smoothing'] if restored else None),
        posthandoff_actual_Z_error_max_mm=max((1000.*abs(float(r['tcp'][2])-float(r['command'][2])) for r in records),default=None),
        rejected_advanced_rows=sum(not r['handoff_smoothing']['accepted'] for r in records),
        acceleration_limit_applies_only_inside_two_seconds=True)
