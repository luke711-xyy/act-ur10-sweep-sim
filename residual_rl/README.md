# Hybrid ACT + recurrent residual SAC pilot

**Latest training authorization:** [long training contract](../docs/residual_long_training_20261016.md).
Success/failure terminal reward0; contact shaping and safety-60 retained.
New training-only entry: `scripts/train_residual_long.py` (default inspect).
72h/1M effective RL steps,50K full-checkpoint milestones,20GiB output,8GiB disk
reserve. No new preflight or six-layout validation. Older execution boundaries
below describe historical runs and do not override this explicit new request.

**Latest contract (2026-10-16): relative XY B/80%, yaw correction +/-45 degrees.**
Use [ACT-conditioned relative targets](../docs/residual_relative_action.md).
At 25Hz, XY correction norm <=80% of the displacement from measured TCP XY to
the accepted ACT target. Direct square-to-disk generation, not post-clipping;
original native 100Hz speed/workspace/wrist/Z protections remain unchanged.
The user authorized implementation/tests and ONE new zero/continuous target1
pair with videos/curves/IK evidence. NO training, resume or GitHub push.
The fixed-5mm/3-degree helpers in `control.py` are legacy characterization only;
their descriptions below are historical, not the current execution interface.

The authorized new target1 pair is complete (seed3158000): both arms retained
exactly one part, with no IK exception. The zero shadow matched all1758 native
ACT controls byte-for-byte. The continuous arm exercised390 changed 25Hz targets;
the saved-trace audit checked all1563 active100Hz controls. No learner initialized,
training/update counts remain0. Evidence:
`exports/residual_relative80_preflight_20261016_115913/videos/edit/`
(under the user's zhe workspace, not this Git checkout).

Current residual Actor input is572 dimensions; current action contract is
`relative_xy80_yaw45_v1`. Old fixed-pose/velocity snapshots are rejected. The
following pilot/pose sections retain historical design context; they do not
override the current relative-action specification or authorize new execution.

This is a new isolated runner, not a replacement of historical evaluation code.
`docs/residual_rl_pilot.md` is the approved implementation contract. Importing
the package or running the default command does not load ACT or start physics.

## Safe inspection / preparation

From the existing worktree:

```sh
.venv/bin/python -B scripts/train_residual_sac.py
.venv/bin/python -B scripts/train_residual_sac.py --help
.venv/bin/python -B scripts/train_residual_sac.py --mode prepare --out /absolute/new/run_directory
```

Preparation only seals configuration, a disjoint deterministic seed schedule,
the baseline identity and new code hashes. It does not consume an opportunity.
Every layout's actual initial-pose hash is checked against old120 and previous
training/validation layouts after reset and BEFORE planning/execution. A collision
is an error, never silently replaced with another seed. The registry binds each
seed to one initial-pose hash and its consumed roles. The current preflight uses
one physical rollout per layout; only base/final validation pairs reuse layouts.

**Selected baseline (2026-10-15):** restored from the published independent branch
`feat/hybrid-astar-act-trial4`, commit `eb887843c9849d4a72be116bb87c44519dc7ef74`,
whose simulator base is `9102680d8e018b225ab34d36422572cd4b547166`. The existing
five-view ResNet ACT `step_100000` and trial-4 Hybrid controls stay fixed. The
67 on-disk sim sources now exactly match the historical sealed SHA256 map;
DINO code is preserved in its own Git branch, not enabled in this worktree.
The isolated RL process executes these same 67 sources.
Hash-identical disk sources are reused in memory; only `policy.py` / `train.py`
are obtained from the pinned Git blobs when their disk versions differ. A missing
blob, wrong hash or other modified frozen source fails closed. Added sim modules,
including DINOv3, cannot fall through to the ordinary importer.

`frozen_identity.json` records both the effective source identity and the current
disk fingerprints, so disk drift during a run still stops it. Old metadata,
fingerprints and formal scripts are not rewritten. The private process must start
before any `sim` imports; a mixed current/legacy namespace is rejected rather than
replaced. Virtual source filenames preserve existing asset/config directories and
allow module/function inspection and the reviewed AST adapters to see pinned text;
no virtual source copy or bytecode file is created. Earlier invocations stopped in
preflight before learning; their actual traces and consumed layouts are preserved.
The latest restoration authorization permits one original no-RL target1 check,
then adapter reconnection only after a normal rollout. It does NOT authorize a
new pilot, preflight campaign, validation campaign, or training restart.
That was the2026-10-15 authorization boundary. On2026-10-16 the user explicitly
authorized ONE new bounded pilot; see [the new execution note](../docs/residual_rl_execution_20261016.md).
All safety and preflight integrity gates remain unchanged.

## Execution requires a separate explicit human request

The latest request is2026-10-16 for ONE new bounded pilot on the restored
independent Hybrid; see `docs/residual_rl_execution_20261016.md`. The previous
invocations and authorizations remain documented separately. This does not
authorize automatic restarts, old ACT training resumption or the future240-rollout
comparison. No execution result is implied by this authorization note.

With the selected source baseline, the commands below are interfaces, **not evidence
of authorization or of an executed experiment**:

```sh
# Real ACT preflight only: two same-stream zero shadows plus two motion probes.
.venv/bin/python -B scripts/train_residual_sac.py --mode preflight \
  --out /absolute/new/preflight_directory --acknowledge-preflight

# Entire one-shot pilot, including fresh preflight and six paired validations.
.venv/bin/python -B scripts/train_residual_sac.py --mode pilot \
  --out /absolute/new/pilot_directory --acknowledge-pilot-training
```

Only a newly prepared directory can start. `started.json` is exclusive and
non-resumable; missing formal records after an attempt require explicit review.
No scheduler, background automation, old training resumption, or 240-rollout
comparison is installed. The supervisor owns only its newly created worker group;
an occupied manual inference lease fails without stopping the user's task.

## Control, observations, and learning

The native Hybrid prefix and two-second handoff are reused through source-checked
process-local adapters. After that window the residual requests bounded XY/yaw pose offsets only.
Zero residual preserves the exact original float32 command path; Z loop state is
not reset. Offset rate limits are separate from native ACT acceleration. Per-control
audit records contain raw action, desired/applied offset, offset rate and prior
state, unmodified native reference, actual prior/applied commands and physical
timestamps, nominal velocity, probe stage and measured TCP pose/yaw. The probe
auditor reconstructs pose offset, XY/yaw command speed, rate and native progress,
and requires actual small-offset motion, hold and return. It separately reports
TCP motion; stage labels and low wrist force cannot establish continuous contact.
The Actor receives sensor state, ACT action
context, remaining hard-budget time and detached frozen action-head input tokens.
Its episode time starts after reset settles physics; native controller and
collection clocks and the 2,000-frame contract are unchanged.
Tokens become visible only when their timestamped ACT chunk was accepted and its
corresponding index selected. Privileged Critic state has a separate builder;
reward values and simulator identities never enter the Actor input.

The native temporal ensemble stays enabled at its frozen coefficient0.01. The
adapter does not compute or replace executed ACT actions. It aligns512-dimensional
feature tokens to every currently contributing accepted block/index and combines
them with the native normalized exponential weights. Age/index are weighted
summaries, not an invented single-block identity; full contributor provenance is
retained in the trace. Critic queue context is the newest contributing block,
not an unaccepted or not-yet-valid prediction. Actor input dimensions and its
sensor-only boundary are unchanged. Scheduler reset clears feature bookkeeping.
The2026-10-16 replacement uses preflight seeds3152300/3152301 and motion probes
3152400/3152401. Preparation seals consumed markers and initial-pose fingerprints
from all three prior stopped invocations and the original-Hybrid restore check
3152200 as exclusions. It never resumes or reruns any earlier opportunity.

### Exact zero check on one actual asynchronous ACT stream

Two independent rollouts can accept different asynchronous prediction results at
the same physical time, even with the same seed and unchanged settings. Their
trajectory hashes are therefore not an isolating zero-adapter test. The current
gate executes the native path once on each new preflight layout. At every ACT
100 Hz reference step, an isolated zero-residual shadow uses the same accepted
action, frame start and already computed admittance Z. It uses the production
residual reference composer, without another policy query, handoff/controller
advance, executed command or physics step. During the original handoff it follows
the same bypass; after two seconds it tests native reference plus zero residual.

Equality requires identical dtype, shape and bytes, including Z and yaw; even one
ULP difference stops the invocation. `zero_residual_shadow` records compared and
active counts, maximum difference and both command digests. The gate requires real
accepted ACT chunks, post-gate frames and complete coverage of actual ACT control
steps, not an empty/prefix-only pass. The trace retains per-step shadow evidence.
Native command selection, temporal ensembling and termination are unchanged. This
proves reference equivalence on the observed stream, not that asynchronous runs
will have identical action-block acceptance or task outcomes. All shadow/probe
checks must pass before baseline validation and learner initialization.

Geometry contact dwell is sampled every physics substep. Only signed contact
distance <=0 is considered. Twenty continuous milliseconds confirm one unique
component; interruption resets unconfirmed dwell. Prefix contacts consume quota
without retrospective RL credit. Shorter contacts receive no shaping reward.

Recurrent SAC uses raw normalized actions for density/Q, while replay also has
control/filter state in its observations. Whole-prefix burn-in reconstructs
hidden states with current weights; learning is masked outside residual execution.
The actor and independent twin critics are CPU GRU128 networks. Frozen ACT stays
on MPS, is never placed in an optimizer, and is cached across episodes. Updates
occur only between episodes. The 2026-10-16 stability revision requires 5,000
learnable steps, two sample-bearing complete episodes and 256 learnable steps
per goal. Warmup earns no update debt. Thereafter 25 new valid steps permit one
Critic update, capped at10/episode, with excess permissions discarded. The first
100 Critic updates freeze the Actor; afterwards one Actor update occurs every
five Critic updates (max2/episode). Actor lr=3e-5, Critic lr=3e-4. No old
trajectory imports. See `docs/residual_policy_stability.md` for exact guards.

The Actor alone has a coefficient1 zero-request prior; environmental rewards
are unchanged. A balanced fixed384-state real-history probe bank plus the new
episode checks the entire between-episode round against a single reference:
mean KL<=.01, deterministic requested XY pose change<=0.15625mm,
yaw pose change<=0.107143degrees, and <=1%
near-saturated deterministic or fixed-noise requests. Failed rounds restore
Actor/Adam/counter/RNG and latch a stop, never an automatic zero-residual fallback.
Learning checkpoint schema3 records independent optimizer counts, the guard
contract and compact probe descriptors. Old schema1/schema2 learner checkpoints are
rejected rather than silently interpreted under different settings. Inference
does not need probe histories; subsequent learning must resolve their exact
episode identities from replay or fail closed.

## Budgets, checkpointing and outcomes

The six-hour monotonic clock begins before initialization. Collection stops by
hour five. The 50,000-decision cap covers motion probes, training and final
residual validation; training conservatively reserves 12,000 decisions for six
worst-case validation episodes, so it can stop below 50,000 training steps.
Resource checks cover the experiment process tree, free disk and run output only.
macOS system memory pressure level >=4 sustained30s is dangerous; RSS>10GiB
sustained30s, disk<8GiB or output>=4GiB stops the run. No user data is deleted.

Work stops conservatively 60 seconds before the six-hour deadline, at disk
headroom <=8GiB+64MiB or output >=4GiB-64MiB. Sustained memory pressure requests
cleanup at20s, before its30s hard cutoff. The supervisor requests cooperative
worker cleanup (stop-request file plus SIGTERM), observes save acknowledgement,
and escalates only its owned process group by the cleanup deadline or immediately
at a hard resource cap. A separate wall watchdog covers a stalled monitor. All
cleanup fits within the original six hours; reserves do not enlarge any hard cap.

An administrative cutoff has no fabricated -20/-60 penalty: its actual count is
retained, `success=null`, outcome `budget_truncated`; valid recorded transitions
are nonterminal for bootstrap. Such attempts are reported separately from true
task failures. A normal failure BEFORE residual activation still remains in the
task-success denominator. Infrastructure exceptions stop rather than retry.

Checkpoints include residual networks/targets/optimizers, replay and RNG states;
they never replace an ACT checkpoint. Milestone snapshots occur at episode
boundaries after crossing10k intervals and are labelled with the ACTUAL decision
count. The final checkpoint is reloaded for deterministic final validation;
validation is not used to choose a better checkpoint. No automatic resume.

Initial/milestone/final full checkpoints remain intact. Two alternating compact
recovery snapshots are saved at safe episode boundaries; shutdown attempts a
compact snapshot containing residual networks, targets, optimizers and RNG, with
no replay copy. Compact writes are capped at32MiB, have a deadline, and recheck
fresh disk/output/RSS/pressure headroom, keeping1MiB for final evidence. They do
not call or bypass the exhausted work guard. Hard risk instead reuses a completed
checkpoint whose byte count and SHA256 still verify. `save_result.json`, summary
and status explicitly distinguish `succeeded`, `previous_reused`, `failed` and
`not_applicable` (learner not initialized), including save errors and rejected
checkpoint evidence. The supervisor also records acknowledgement/forced-stop
evidence. A compact snapshot is recovery evidence, not a resumable full replay.
Only a new partial checkpoint created by the current write is removed on failure.
Derived readable/metric reports run after primary summary/status; report failures
are recorded separately and do not overwrite stop/save evidence.

Before each completed or administratively truncated Episode enters the ledger,
its `episodes/<episode_id>/experience.npz` is atomically saved and hashed. It
contains float32 raw normalized actions, Actor/Critic observations and rewards,
plus bool termination/learning masks; no RGB or object arrays. Truncated episodes
keep their real nonterminal masks. Compact checkpoints reference the durable
ledger prefix (byte count and SHA256), whose rows carry experience hashes. An
unknown live-episode interruption attempts `partial_experience.npz` within the
same bounded headroom. Its metadata explicitly says incomplete/not a valid
Episode; available arrays may include an unexecuted action, and no terminal state
or result is invented. Unsafe or failed diagnostic saves are recorded explicitly.

Training RGB/video is not retained. Every actual physical control state and
numeric trace is streamed to compressed JSONL. Only validation videos are made,
by restoring saved integration states, never by executing ACT or commands again.
The output contains full decode/hash evidence. No-plan attempts have no execution
video; their failure remains in the ledger. Incomplete verification is not success.

Reports keep native strict success, final count error, contact/terminal rewards,
wrist force, Z error, low-wrist-force contact proxy and residual intervention
separate. Low wrist force is NOT proof of continuous table contact. The uncapped
reward can prefer avoiding extra contacts over some successful recovery routes;
six validation layouts cannot establish statistical improvement or convergence.
# Historical fixed-pose interface — superseded by B/80% above

At that earlier revision, only implementation/tests were authorized.
Its action contract was `bounded_pose_xy5mm_yaw3deg_v1`, superseding velocity
actions, and has now itself been replaced by `relative_xy80_yaw45_v1`.

SAC requests a current absolute correction around the causal native ACT
reference: radial XY <=5 mm, yaw +/-3 degrees, never Z. Requests are not summed
over time. Offset updates use a 0.2 s response and <=5 mm/s / 3 deg/s rates;
native XY motion retains >=80% forward progress inside a 10-degree cone, still
bounded by 80 mm/s. Native idle freezes displacement rather than snapping back.
Incompatible pose/rate/workspace constraints stop with diagnostics, not fallback.

The isolated adapter removes its own applied offset from the reference at query
submission and frame start, preventing corrected anchors from accumulating the
same shift again. Original ACT observations, 5/25/100 Hz timing, temporal
ensembling, Hybrid handoff and wrist-only Z admittance stay unchanged. The 5 mm
bound is relative to the current causal native ACT reference, not a separate
counterfactual zero-residual episode observing different images.

Actor dimension is now566, adding applied pose offset separately from its rate.
Replay schema2 and learner schema3 bind the action contract; old velocity-space
snapshots are rejected. Policy-change checks now use metres/radians, preserving
the prior fractional guard thresholds. Rewards/update cadence are unchanged.
That revision's probes were small_offset / hold_offset / return_offset, not
braking or reversal. See [the historical design](../docs/residual_pose_offset.md).
