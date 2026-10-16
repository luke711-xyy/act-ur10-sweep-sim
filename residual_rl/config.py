"""Approved pilot constants. No simulator, model or historical file mutations."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class StabilityConfig:
    warmup_steps: int = 5000
    episodes_per_goal: int = 2
    steps_per_goal: int = 256
    steps_per_critic: int = 25
    critics_per_episode: int = 10
    critic_only_updates: int = 100
    critics_per_actor: int = 5
    actors_per_episode: int = 2
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    prior_weight: float = 1.
    probe_states_per_goal: int = 64
    probe_noise_samples: int = 8
    probe_seed: int = 20261016
    max_mean_kl: float = .01
    max_xy_change: float = .005/32
    max_xy_fraction_change: float = .8/32
    max_yaw_change: float = (math.pi/60)/28
    saturation_level: float = .9
    max_saturation_fraction: float = .01


STABILITY = StabilityConfig()


@dataclass(frozen=True)
class PilotConfig:
    schema: int = 3
    action_contract: str = 'relative_xy80_yaw45_v1'
    reward_contract: str = 'contact_only_safety_v1'
    run_profile: str = 'pilot_v1'
    checkpoint_interval_steps: int = 10000
    gamma: float = 1.
    actor_hz: int = 25
    control_hz: int = 100
    feature_hz: int = 5
    feature_dim: int = 512
    handoff_seconds: float = 2.
    max_frames: int = 2000
    max_seconds: float = 80.
    residual_xy_fraction: float = .8
    relative_yaw_limit: float = math.pi/4
    # Historical pose-helper units below are NOT the relative action limits.
    residual_xy_offset: float = .005
    residual_yaw_offset: float = math.pi/60
    residual_xy_speed: float = .005
    residual_yaw_speed: float = math.pi/60
    offset_response_seconds: float = .2
    min_forward_fraction: float = .8
    direction_cone_radians: float = math.pi/18
    total_xy_speed: float = .08
    total_yaw_speed: float = .35
    contact_dwell_seconds: float = .020
    max_wall_seconds: float = 21600.
    stop_collection_seconds: float = 18000.
    max_residual_steps: int = 50000
    min_disk_bytes: int = 8 * 2**30
    max_output_bytes: int = 4 * 2**30
    max_rss_bytes: int = 10 * 2**30
    pressure_dwell_seconds: float = 30.
    training_seed_base: int = 3300000
    validation_seed_base: int = 3140000
    preflight_seed_base: int = 3152300
    random_seed: int = 20261015

    def to_dict(self):
        return dict(asdict(self), learning_stability=asdict(STABILITY))


DEFAULT = PilotConfig()
LEGACY_POSE = replace(DEFAULT,schema=2,action_contract='bounded_pose_xy5mm_yaw3deg_v1')
LONG_TRAIN = replace(DEFAULT,run_profile='long_train_1m_v1',checkpoint_interval_steps=50000,
    max_residual_steps=1000000,max_wall_seconds=72*3600.,stop_collection_seconds=72*3600.-60.,
    max_output_bytes=20*2**30,training_seed_base=5000000)


def activate_long_training_profile():
    """Explicit fresh-process entry only; never mutate an already-bound runtime."""
    import sys
    if any(name.startswith('residual_rl.') and name not in {'residual_rl.config'}
           for name in sys.modules):
        raise RuntimeError('long profile must be selected before residual modules are imported')
    global DEFAULT
    DEFAULT=LONG_TRAIN
    return DEFAULT
