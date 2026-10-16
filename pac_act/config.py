from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    chunk: int = 25
    execute: int = 5
    action_hz: int = 25
    environments: int = 6
    blocks_per_env: int = 64
    gamma: float = 1.
    gae_lambda: float = .95
    actor_lr: float = 1e-6
    critic_lr: float = 1e-5
    epochs: int = 8
    batch: int = 32
    microbatch: int = 4
    clip: float = .2
    value_coef: float = .5
    entropy_coef: float = .001
    max_grad_norm: float = .5
    kl_coef: float = 3.
    prior_coef: float = 2.
    target_kl: float = .01
    early_kl: float = .015
    hard_kl: float = .02
    max_actions: int = 1_000_000
    milestone: int = 50_000
    hours: float = 72.
    collect_hours: float = 66.
    output_gib: float = 50.
    free_gib: float = 20.
    rss_gib: float = 10.
    rss_dwell: float = 30.

    def __post_init__(self):
        if (self.chunk != 25 or self.execute != 5 or self.action_hz != 25
                or self.microbatch not in (1,2,4) or self.batch != 32
                or self.environments != 6 or self.blocks_per_env != 64):
            raise ValueError('unapproved chunk, environment or logical batch contract')

    def asdict(self):
        return asdict(self)


DEFAULT = Config()
