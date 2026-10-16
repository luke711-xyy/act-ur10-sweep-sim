"""Optional continuous ACT target following while the brush maintains contact."""
import math
from collections.abc import Mapping
import numpy as np


def configure(module, cfg, profile):
    base = getattr(module.handoff, '_original_governor_type', module.handoff.TwoSecondHandoff)
    module.handoff._original_governor_type = base
    module.handoff.TwoSecondHandoff = base
    module._contact_sweep_limits = None
    values = cfg.get_path('act.hybrid_sweep_smoothing')
    if values is None:
        return profile
    keys = {'xy_acceleration_m_s2', 'yaw_acceleration_rad_s2'}
    if not isinstance(values, Mapping) or set(values) != keys:
        raise ValueError('hybrid_sweep_smoothing requires XY and yaw acceleration limits')
    xy_accel = float(values['xy_acceleration_m_s2'])
    yaw_accel = float(values['yaw_acceleration_rad_s2'])
    if (not np.isfinite([xy_accel, yaw_accel]).all()
            or not .05 <= xy_accel <= .3 or not .1 <= yaw_accel <= .5):
        raise ValueError('contact sweep acceleration outside bounded tuning range')
    module._contact_sweep_limits = {
        'xy_acceleration_limit_m_s2': xy_accel, 'yaw_acceleration_limit_rad_s2': yaw_accel}

    class ContinuousContactHandoff(base):
        def prepare_record(self, row):
            super().prepare_record(row)
            details = row['handoff_smoothing']
            actual = np.asarray(row['command'])
            if details['mode'] == 'contact_sweep' and actual.dtype == np.float32:
                # Native telemetry rounds each real command to float32. Above
                # two radians a single rounding can exceed the old absolute
                # 1e-7 tolerance. Accept exactly that representation, with no
                # extra tolerance or relaxation of the accepted physics clock.
                expected = np.asarray(details['command'], dtype=np.float32)
                match = bool(np.isfinite(actual).all() and np.array_equal(actual, expected))
                details.update(command_match=match,
                               accepted=details['advanced'] and details['clock_match'] and match)

        def propose(self, now, native, target, admittance_z):
            command = super().propose(now, native, target, admittance_z)
            if self.pending['mode'] != 'native_restored':
                return command
            target = np.asarray(target, dtype=np.float64).reshape(4)
            delta = target[:2] - self.last_reference[:2]
            distance = float(np.linalg.norm(delta))
            speed = math.sqrt((xy_accel*self.dt)**2 + 2*xy_accel*distance) - xy_accel*self.dt
            desired = delta/distance*min(.08, speed) if distance > 1e-12 else np.zeros(2)
            change = desired - self.velocity_xy
            length = float(np.linalg.norm(change))
            velocity = (self.velocity_xy + change*min(1., xy_accel*self.dt/length)
                        if length > 1e-12 else desired)
            error = math.atan2(math.sin(target[3]-self.last_reference[3]),
                               math.cos(target[3]-self.last_reference[3]))
            speed = math.sqrt((yaw_accel*self.dt)**2 + 2*yaw_accel*abs(error)) - yaw_accel*self.dt
            desired_yaw = math.copysign(min(.35, speed), error)
            yaw_velocity = self.yaw_velocity + float(np.clip(
                desired_yaw-self.yaw_velocity, -yaw_accel*self.dt, yaw_accel*self.dt))
            command = self.last_reference.copy()
            command[:2] += velocity*self.dt
            command[2] = float(admittance_z)
            command[3] += yaw_velocity*self.dt
            if np.any(command[:2] < self.bounds[:, 0]) or np.any(command[:2] > self.bounds[:, 1]):
                raise RuntimeError('continuous contact sweep cannot remain in workspace')
            self.pending.update(mode='contact_sweep', command=command.copy(),
                velocity_xy_m_s=velocity, yaw_velocity_rad_s=yaw_velocity,
                acceleration_xy_m_s2=(velocity-self.velocity_xy)/self.dt,
                yaw_acceleration_rad_s2=(yaw_velocity-self.yaw_velocity)/self.dt,
                xy_acceleration_limit_m_s2=xy_accel, yaw_acceleration_limit_rad_s2=yaw_accel,
                acceleration_bounds_apply=True, continuous_contact_sweep=True)
            return command

    module.handoff.TwoSecondHandoff = ContinuousContactHandoff
    profile.update(sweep_smoothing={**values, 'xy_speed_m_s': .08, 'yaw_speed_rad_s': .35,
                                   'starts_after_handoff_s': 2.},
                   after_handoff_window='continuous_ACT_target_following_with_acceleration_limits')
    return profile
