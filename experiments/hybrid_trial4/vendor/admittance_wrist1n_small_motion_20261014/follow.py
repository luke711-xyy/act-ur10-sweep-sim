"""THROWAWAY bounded actual-Z feedback, not pure force admittance.

Nonintegrating negative feedback on actual TCP Z minus the previous accepted
command. The force reference remains generated solely by the wrist loop.
This is a hypothesis probe, not a certified inner position servo.
"""
import math


class TrackingCompensator:
    def __init__(self,gain):
        self.gain=float(gain)
        if not math.isfinite(self.gain) or not 0.<=self.gain<=1.:
            raise ValueError('position feedback gain outside independent probe scope')

    def propose(self,*,nominal,native_z,last_command_z,actual_z,wrist_force,z_min):
        values=list(map(float,(nominal,native_z,last_command_z,actual_z,wrist_force,z_min)))
        if not all(map(math.isfinite,values)):
            raise ValueError('nonfinite position or wrist feedback')
        nominal,native_z,last_command_z,actual_z,force,z_min=values
        if force<0. or z_min>nominal+.005:
            raise ValueError('invalid wrist force or incompatible workspace floor')
        error=actual_z-last_command_z
        bias=max(-.0002,min(.0002,-self.gain*error))
        if force>=10.:bias=0.  # Never counteract the native emergency relief.
        unbounded=native_z+bias
        requested=max(z_min,nominal-.005,min(nominal+.005,unbounded))
        return dict(native_requested_z=native_z,tracking_actual_z=actual_z,
            tracking_last_accepted_command_z=last_command_z,
            tracking_error_z=error,tracking_bias_z=bias,
            tracking_gain=self.gain,requested_z=requested,
            tracking_emergency_disabled=force>=10.,
            tracking_total_bound_clipped=abs(requested-unbounded)>1e-12)
