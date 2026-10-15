"""Explicit sensor-only Actor boundary, separate privileged Critic builder."""
import numpy as np

from .config import DEFAULT

ACTOR_DIM = 563


def actor_observation(*, robot_state, feature, target_count, elapsed,
                      feature_age, chunk_index, feature_available, active,
                      nominal_velocity, previous_action, residual_velocity):
    state = np.asarray(robot_state,dtype=np.float32)
    features = np.asarray(feature,dtype=np.float32)
    if state.shape != (36,) or features.shape != (512,) or not 1 <= target_count <= 6:
        raise ValueError('invalid sensor/feature/target contract')
    scale = np.tile(np.r_[np.full(6,np.pi),np.ones(5),np.full(6,20.),1.],2)
    result = np.r_[state/scale, features if feature_available else np.zeros(512),
        np.asarray(nominal_velocity)/np.array([.08,.08,.35]), previous_action,
        np.asarray(residual_velocity)/np.array([.16,.16,.7]),
        target_count/6.,max(0.,1.-elapsed/DEFAULT.max_seconds),
        min(float(feature_age),DEFAULT.max_seconds),chunk_index/25.,
        float(feature_available),float(active)].astype(np.float32)
    if result.shape != (ACTOR_DIM,) or not np.isfinite(result).all():
        raise ValueError('nonfinite or invalid Actor sensor input')
    return result


def critic_observation(actor, *, qpos, qvel, collected, seen, dwell,
                       collection_clock, controller_state, scheduled_actions):
    result = np.concatenate([np.asarray(v,dtype=np.float32).ravel() for v in
        (actor,qpos,qvel,collected,seen,dwell,collection_clock,
         controller_state,scheduled_actions)])
    if not np.isfinite(result).all():
        raise ValueError('nonfinite privileged Critic state')
    return result
