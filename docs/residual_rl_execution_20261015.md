# First residual SAC pilot: publication and execution authorization

On 2026-10-15 the user explicitly requested publication to a new GitHub RL
post-training branch, followed by preflight and training. This supersedes the
earlier implementation-only restriction for this single bounded pilot, not for
any old ACT training run or future formal 240-episode comparison.

- Repository: luke711-xyy/act-ur10-sweep-sim.
- New branch: feat/act-residual-sac-post-training. Preserve the DINO branch.
- Effective simulator/policy baseline: pre-DINO commit
  9102680d8e018b225ab34d36422572cd4b547166, with all 67 historical source hashes
  verified in memory; five-view ordinary ACT step_100000 remains frozen.
- Use the existing pilot entry in `--mode pilot`; its real ACT same-stream
  zero-residual checks and bounded motion probes MUST pass before baseline
  validation, learner initialization and training. Do not run a second standalone
  preflight that consumes these same seeds before the pilot.
- Keep the approved trial-4 Hybrid, wrist-only Z control, recurrent SAC,
  rewards, new train/validation layouts, single opportunities and all guards.
- One new isolated output directory, one invocation, no automatic restart,
  retry, resume, extra budget, seed replacement or best-checkpoint selection.
- Total: at most 6 hours or 50,000 residual decisions; training collection stops
  by hour 5 and reserves final validation within the same budget. ACT uses MPS;
  residual learning uses CPU. Keep all failures and measured counts.
- Stop/save on failed preflight, infrastructure exception or resource guard.
  Do not stop unrelated user processes. Training videos remain disabled;
  paired validation videos and numeric evidence are retained.

Publication contains code, tests and documentation only. The runner currently
requires the existing local sealed Hybrid metadata, reviewed exported helper
code and checkpoint at the documented paths. These local assets are not bundled
or uploaded; this branch is not a self-contained portable training package.

Pre-publication scoped regression: 275 passed, no failures/errors/skips. Prior
full-project attempts were bounded and incomplete, not full-suite passes. Real
preflight/training results belong in the new run directory; this authorization
document is not evidence that either has passed.

## Authorized replacement invocation after adapter failure

The first invocation, `residual_sac_pilot_pre_dino_20261015_2143`, stopped before
any learner initialization or residual decision. Its original ACT configuration
has `temporal_ensemble_coeff=0.01`; the new residual scheduler incorrectly rejected
all temporal ensembles. The consumed native-preflight opportunity (seed3150000)
and its actual prefix, partial experience and exception evidence remain intact.

The user explicitly authorized fixing that adapter, adding tests, publishing the
fix, and starting ONE new invocation on new preflight layouts. Preserve the
native ensemble coefficient, accepted chunks, action calculation, query timing,
Z state, model and termination. Do not retry or rewrite the previous invocation.

Feature metadata follows the native contributors and horizon indices with the
same normalized exponential weights; the512-dimensional feature, age and index
are weighted summaries. Full contributing block/index/weight provenance is saved
in numeric traces. This is observation bookkeeping, not a claim that a blended
hidden token is a new native network activation. Critic queue context uses the
newest eligible contributing block. Native action calculation, including circular
yaw averaging, runs exactly once in the inherited scheduler and is not replaced.

New preflight seeds:3151000/3151001 for the native/zero pairs,3151100/3151101 for
motion probes. Training3300000+ and validation3140000–3140005 were not attempted
in the failed invocation and remain separately fixed. Check the first new reset
fingerprint against the preserved failed layout as well as old120 exclusions.
All prior six-hour/50k-decision/resource/no-auto-resume limits remain unchanged.

## Authorized same-stream preflight after asynchronous comparison failure

The second invocation, `residual_sac_pilot_ensemble_fixed_20261015_2200`,
successfully executed native temporal ensembling in both diagnostic arms on
seed3151000. Both ended with two collected parts and an ordinary native-window
failure. Learning did not start. The original exact cross-rollout gate stopped:
at the first differing command, one run had accepted a query issued at7.14 s and
the other had not. Both actual command streams and the measured failure counts
remain intact. This is not evidence that zero residual changed the same inputs.

The user explicitly allowed a same-stream shadow comparison, tests, publication,
and ONE new invocation. Preserve native asynchronous timing/ensemble/settings;
do not force synchrony or loosen exact equality. The native path alone executes.
An independent zero-residual composer checks every reference on those same inputs
without another policy query, physics step or controller/handoff update. During
handoff it shares the bypass; once the original2 s gate opens it uses the same
pure native reference and production residual composition. A single bit mismatch
stops the experiment. Evidence must include nonempty real ACT, active comparison
coverage, exact per-step agreement and matching full reference digests.

New seeds:3152000/3152001 (target1/6 shadows),3152100/3152101 (motion probes).
Seal consumed seed and pose hashes from both prior invocations; never reexecute
or rewrite them. Training/validation sequences and the entire bounded pilot
contract remain unchanged. One physical attempt per new preflight layout, no
automatic repair, restart or resume on another failed gate.
