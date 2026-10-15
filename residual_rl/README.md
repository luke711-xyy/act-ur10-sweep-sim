# Hybrid ACT + recurrent residual SAC pilot

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
seed to one initial-pose hash and its consumed roles. Only the exact native/zero
preflight pair or base/final validation pair can reuse that seed AND hash.

**Selected baseline (2026-10-15):** the user chose the pre-DINO implementation,
commit `9102680d8e018b225ab34d36422572cd4b547166`, with the existing five-view
ordinary ACT `step_100000` and trial-4 Hybrid controls. The current DINOv3 branch
and its 69 on-disk Python sources stay intact. The isolated RL process executes
exactly the 67 sources identified by the historical sealed Hybrid SHA256 map.
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
no source copy or bytecode file is created. No real ACT preflight has run yet.

## Execution requires a separate explicit human request

The user supplied that request on 2026-10-15 for one new bounded pilot after
GitHub publication; see `docs/residual_rl_execution_20261015.md`. This does not
authorize automatic restarts, old ACT training resumption or the future240-rollout
comparison. No execution result is implied by this authorization note.

With the selected source baseline, the commands below are interfaces, **not evidence
of authorization or of an executed experiment**:

```sh
# Real ACT preflight only: two native/zero pairs plus two bounded motion probes.
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
process-local adapters. After that window the residual owns XY/yaw velocity only.
Zero residual preserves the exact original float32 command path; Z loop state is
not reset. Residual slew is separate from native ACT acceleration. Per-control
audit records contain raw action, desired and filtered residual, acceleration,
filter state before/after acceptance, unmodified native reference, separately
identified composition reference, actual prior/applied commands and physical
timestamps, nominal velocity, probe stage and measured TCP pose/yaw. The probe
auditor reconstructs XY/yaw command speed, residual slew/integration and clipping,
and requires observed nonzero start, braking and reversal. It separately reports
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
The explicitly authorized replacement run uses preflight seeds3151000/3151001
and probes3151100/3151101, not the previous failed opportunity.

Geometry contact dwell is sampled every physics substep. Only signed contact
distance <=0 is considered. Twenty continuous milliseconds confirm one unique
component; interruption resets unconfirmed dwell. Prefix contacts consume quota
without retrospective RL credit. Shorter contacts receive no shaping reward.

Recurrent SAC uses raw normalized actions for density/Q, while replay also has
control/filter state in its observations. Whole-prefix burn-in reconstructs
hidden states with current weights; learning is masked outside residual execution.
The actor and independent twin critics are CPU GRU128 networks. Frozen ACT stays
on MPS, is never placed in an optimizer, and is cached across episodes. Updates
occur only between episodes, at one update/25 new residual decisions after
256 valid steps AND representation of all six goals. No old trajectory imports.

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
