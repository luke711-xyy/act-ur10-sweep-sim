# Task 2: Recurrent SAC/replay implementation contract

Worktree /Users/huazhi_luke/.codex/worktrees/act-five-view. Own ONLY
residual_rl/learning.py and tests/test_residual_learning.py. Do not edit other
files, frozen sim/ or existing scripts. Do not commit, run ACT/physics, launch
training, spawn agents, install dependencies. Use apply_patch and TDD: tests
fail first, then implementation. Real small CPU torch tests are allowed.

Implement this API (main runtime consumes it):

```
Episode(episode_id: str, target_count: int,
        actor_obs: np.ndarray[T+1,A], critic_obs: np.ndarray[T+1,C],
        actions: np.ndarray[T,3], rewards: np.ndarray[T],
        terminated: np.ndarray[T] bool, learn_mask: np.ndarray[T] bool)
EpisodicReplay(capacity=100_000, seed=0)
  .add(episode): validate, retain whole episodes, reject duplicate IDs/invalid data;
                evict whole oldest episodes to respect total T capacity
  .ready(min_steps=256): >=min_steps valid learned transitions and all goals1..6
  .sample(batch_size=4,length=32): target-balanced rotation, random eligible
       episode/start, return list of SequenceSample(episode,start,length).
       start must allow >=1 learnable real transition in window; prefix/padding
       remain non-learning. Never cross episode boundaries.
  .state_dict() / .load_state_dict(state): include episodes and RNG, validate.
RecurrentSAC(actor_dim, critic_dim, seed=0)
  .act(observation: np.ndarray[A], hidden=None, deterministic=False)
       -> (np.ndarray[3] raw normalized tanh action, hidden tensor)
       Whole-episode state. This is only sensor Actor input. Critic data separate.
  .update(replay) -> dict with finite actor_loss, critic_loss, updates etc
  .checkpoint() -> serializable dict including dimensions/config/model, targets,
       optimizer states, RNG states, update count. No automatic resume.
  .load_checkpoint(payload): validate schema/dimensions/config.
```

Single-layer GRU128 each actor and independent twin critics; small MLP128 heads.
Actor mean head initialized zero, log_std bias -4 (all output weights initially
zero). Clamp log_std [-5,0]. Squashed Gaussian corrected log probability refers
to RAW normalized action, not post-control applied action. CPU only.
lr3e-4, tau.005, gamma1, alpha.001 fixed, clip grad norm5. Use target critics.
Full episode prefix re-forwarded without gradients before each sample start,
then train unroll32; padded/prefix/invalid learn_mask entries get no loss.
For each target next step use correctly aligned recurrent histories INCLUDING
previous replay action (can concatenate action into critic current Q head,
not actor observation). Actor no reward or privileged observations. Critic may
read critic_obs; terminal bootstrap masked, ordinary truncated budgets are NOT
fabricated terminal environment failure. Actor parameters/hidden reset not
implicitly modified by update; caller updates only between episodes.
One optimizer update per call; caller owns warmup/update schedule. If replay is
not ready do not train (return explicit skipped status). Target-balanced sample
selection cannot silently drop a target or alias storage. Deterministic act
must be reproducible; checkpoint roundtrip must preserve predictions and RNG.

Tests: actual finite optimizer update changes residual weights not input data;
all goals/warmup gate; prefix-only/invalid/padded losses excluded; whole-prefix
burn-in influences action; reset prevents cross-episode memory; terminal next Q
not bootstrapped; physical applied residual not used as policy log-prob action;
full checkpoint resume roundtrip; whole-episode replay eviction and disjoint
sampling, schema/NaN validation. Report RED and GREEN commands/results and files.
