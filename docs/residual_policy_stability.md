# Residual policy stability revision — approved 2026-10-16

Action-interface amendment: [bounded pose offsets](residual_pose_offset.md)
supersedes the velocity bounds, unbounded route accumulation and velocity-unit
policy thresholds in this historical revision. Learning cadence and transactional
rollback remain; current guard thresholds are 1/32 of the 5 mm pose bound and
1/28 of the 3-degree yaw bound. Learner schema3 replaces schema2.

Implementation and CPU synthetic-data tests only. No actual preflight, ACT
inference, robot rollout, training run, old-run recovery or publication is
authorized. This revision supersedes only the learner/scheduling portions of
the initial residual pilot contract.

## Frozen behavior

The original Hybrid A* approach, ACT temporal ensembling/query cadence, wrist
force-only Z admittance, contact handoff, reward events and native termination
are unchanged. The 160mm/s / .7rad/s residual bounds, 80mm/s / .35rad/s composed
bounds and residual slew limits are unchanged. No hard accumulated route-offset
budget is added; braking and reversal remain available.

## Fresh-data update permissions

- Require >=5,000 learnable steps overall, >=2 complete sample-bearing episodes
  and >=256 learnable steps for each goal1..6. Successful and failed episodes
  both qualify. Administrative partial episodes never earn permissions.
- The first readiness transition grants zero permissions. Keep warmup data in
  replay, not as update debt. If readiness is lost, reset permission remainder.
- Afterwards every25 actually valid new transitions grant a Critic update,
  max10 per completed episode. Discard integer overflow and unused permissions
  at the next episode; only a post-warmup remainder0..24 carries forward.
- Critic lr3e-4; Actor lr3e-5. First100 Critics do not update Actor. The first
  Actor step is Critic105, then every5 Critics, max2 Actors per round.
- GRU128, batch4x32, gamma1, alpha.001, tau.005 and gradient clip5 remain fixed.

## Actor prior and transactional policy checks

The additional Actor objective is coefficient1 times the squared normalized
requested residual: radial XY norm normalized by160mm/s plus yaw normalized
by.7rad/s, squared and summed. This is optimizer regularization, NOT a change
to environment rewards or SAC density/action definitions.

Build a fixed bank of64 learnable states per goal, 384 total, deterministically
spread over the original available goal histories. Keep full real prefixes
and exact episode identities. Each update round also checks all learnable
states of the current episode. No truth state, reward or Critic input is
introduced into Actor observations.

One snapshot of Actor and Adam defines the reference for the WHOLE round.
Recompute each policy's own GRU history from the episode boundary. Check before
the round, after each Actor step, and before commit:

- Mean KL(old||candidate) of pre-tanh diagonal Gaussians <=.01.
- Maximum deterministic requested XY-vector change <=.005m/s.
- Maximum deterministic requested yaw-rate change <=.025rad/s.
- Deterministic and stochastic saturation fractions each <=.01. Saturation
  means requested XY norm or absolute yaw >=90% of its residual bound.
- Stochastic checks use eight fixed Gaussian samples/state from a separate
  local seed20261016, without consuming policy or global RNG.
- Nonfinite distributions reject immediately.

Rejecting a candidate restores the round-start Actor weights, complete Adam
state, Actor step count and local learner RNG; gradient buffers are cleared.
Critic steps already executed remain accounted for. The learner latches a
fatal stop and refuses subsequent updates AND actions. The pilot records the
guard rejection, stops collection and never turns it into an ordinary IK/task
failure, silent fallback, retry or automatic resume. Per-update metrics are
buffered until whole-round commit so a rolled-back intermediate Actor is not
reported as deployed.

Probe setup/resolution faults also latch this stop. Completed Critic Adam and
target updates are counted before any fallible Actor operation. Separate
completed, rolled-back and failed Actor-slot counters must match the exact
scheduled slots even for failed checkpoints. A normal stop_collection budget
cutoff rolls back the uncommitted round but does not mark its last committed
Actor invalid; the original final-validation path can still use that Actor.
Administrative partial/nonterminal episodes do not qualify for warmup and
never grant new update permissions.

Schema2 learner checkpoints bind the complete stability contract and separate
Critic/Actor optimizer counters. Fixed probes are compact episode/index
descriptors to avoid duplicating large histories inside compact checkpoints.
Learning after an explicitly authorized restore must resolve the same histories
from replay; missing histories cause failure, never a silently replaced bank.
This revision does not authorize such a restore.

## Verification boundaries

Tests use real CPU GRUs/Adam and synthetic arrays, not ACT weights or physical
rollouts. Three fixed seeds each attempt200 Critic updates (20 Actor steps),
with every accepted candidate checked for rapid saturation and policy change.
Fault injection after a real optimizer step verifies entire-round rollback,
including rejection at the second Actor step with existing Adam moments.
Other tests cover warmup debt, burst caps, valid masks, actual recurrent prefixes,
variance-only saturation, nonfinite candidates, checkpoint guards and cadence.

Passing these tests demonstrates the specified software guard behavior, not
task success, convergence, continuous table contact, or real-world safety.
Soft regularization and finite probe coverage do not guarantee zero long-term
route drift. Actual preflight and training require a separate user decision.
