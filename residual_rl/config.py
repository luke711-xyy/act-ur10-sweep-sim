"""Approved pilot constants. No simulator, model or historical file mutations."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PilotConfig:
    schema: int = 1
    gamma: float = 1.
    actor_hz: int = 25
    control_hz: int = 100
    feature_hz: int = 5
    feature_dim: int = 512
    handoff_seconds: float = 2.
    max_frames: int = 2000
    max_seconds: float = 80.
    residual_xy_speed: float = .16
    residual_yaw_speed: float = .7
    residual_xy_acceleration: float = .2
    residual_yaw_acceleration: float = .5
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
    preflight_seed_base: int = 3150000
    random_seed: int = 20261015

    def to_dict(self):
        return asdict(self)


DEFAULT = PilotConfig()
