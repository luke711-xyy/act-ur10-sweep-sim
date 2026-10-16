# User-authorized long residual SAC run — 2026-10-16

This supersedes the old six-hour/50K pilot budget, not its frozen Hybrid control.
Human authorization in the current chat includes code changes, direct training,
new monitor binding, and removal of the previous stopped RL run's data.

## Frozen learning/control boundary

- Frozen pre-DINO ResNet five-view ACT step100000, A* prefix, native temporal
  ensemble and wrist-only Z admittance remain unchanged.
- Current B/80% direct XY correction generation, yaw +/-45 degrees, normalized
  action smoothing and two-second handoff remain as actually preflight-tested.
- Success/failure terminal return is now0. Contact +60/N and excess -120/N,
  once per component after20ms, and safety termination -60 remain. Strict
  success and actual collected count are still measured independently.
- Changed reward contract rejects old replay/learner checkpoints. No old
  demonstrations, checkpoints or preflight arrays enter the new replay.
- Update warmup/cadence, conservative policy drift/saturation guard, rollback,
  unknown-exception stop and no automatic restart remain in force.

## New invocation

- `scripts/train_residual_long.py`: default is read-only inspection. Training
  needs explicit `--mode train --acknowledge-training` and a new sealed root.
- Fresh-process-only profile selection happens before residual/sim imports;
  no mutation of existing processes or original ACT/checkpoint files.
- 1,000,000 accepted effective25Hz RL training contexts or72 hours, whichever
  comes first.72 hours includes loading, sampling, learning and checkpointing.
  Stop new work60 seconds early to fit cleanup within the hard wall limit.
- Output cap20GiB; free disk must remain>=8GiB.10GiB experiment-tree RSS or
  dangerous system memory sustained30s triggers stop. Existing reserve margin
  requests cleanup earlier. No automatic deletion to make room.
- No new physical preflight and no six-layout baseline/final validation.
  Previous same-layout actual-ACT zero/continuous test is linked as evidence,
  not rerun or charged to training steps. Reference dimensions572/791 only.
- Lazy deterministic seed sequence5000000..5999999; goals1..6 round-robin.
  Each layout has one geometry plan and at most one physical execution. No
  retries, seed substitution, candidate physical screening or success filtering.
  Old120 and all available prior layout fingerprints are excluded. The prior
  deleted training run's raw poses cannot be reconstructed from deletion hashes;
  its old3300000-series seeds are disjoint from the fresh range.

## Counts and saves

- The actual recorded Episode learning mask counts steps; prefix/inactive and
  begun-but-unexecuted actions do not count. The last accepted25Hz frame may
  finish its native four100Hz controls; Collector.begin refuses another action.
  Administrative step/time truncation is not a task failure or safety penalty.
- Full snapshots at the first safe episode boundary after crossing each50K
  milestone, labeled with both threshold and actual step count. No best-checkpoint
  selection. Initial/final snapshots and two rotating compact recovery snapshots
  preserve the existing failure-save semantics. Full milestones are not deleted.
-72h/resource stop below1M is explicitly incomplete; unexpected/policy-guard
  faults preserve the actual scene/evidence and stop, never auto-resume.
- Training saves numerical traces/arrays only, no videos. The read-only monitor
  has no control endpoint and never loads/checksums .pt weights on refresh.
- Atomic `training_progress.json` publishes committed counts, Actor/Critic
  updates, current episode and next save milestone; current-episode steps remain
  visibly pending until ledger commit. Curves exclude nontraining records.

## Output isolation

New run: `exports/residual_sac_relative80_1m_20261016` under the zhe workspace.
Monitor: existing read-only `exports/residual_sac_live_panel_20261016`, port8896,
rebound to the new task. No scheduler/heartbeat is installed. No Git push is
included in this request. Old ACT, teaching data, formal results and the last
successful preflight/video remain preserved.
