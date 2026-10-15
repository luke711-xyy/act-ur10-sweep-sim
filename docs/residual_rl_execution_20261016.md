# Restored trial4 Hybrid: one newly authorized residual-SAC invocation

On2026-10-16 (Asia/Singapore), the user explicitly requested reconnecting the
residual adapter, starting training, and pushing the RL branch. They accepted
the prior no-RL check's force peak as brush interference at collection-area
entry, not an A*→ACT handoff defect. This is the user's interpretation, not a new
sensor-source identification experiment. Preserve that actual safety termination,
measured count0, saved trajectory and video; do not relabel it successful.

This supersedes the previous instruction to hold RL reconnection after that
check. It does NOT disable or raise force/collision/workspace guards, waive
preflight integrity, or permit retries of stopped attempts.

## Unchanged baseline and training contract

- Restored source commit `cfd6954bd1c7bb4d35fbadfaefd76420ecdb40b7`, based on
  independent `feat/hybrid-astar-act-trial4` reference
  `eb887843c9849d4a72be116bb87c44519dc7ef74`; all67 frozen sim sources match.
- Frozen five-view ResNet ACT `step_100000`, native temporal ensemble0.01,
  query5Hz/action25Hz/control100Hz; no DINO and no ACT/encoder updates.
- Fourth-group A* prefix, contact gate, two-stage wrist-only Z admittance,
  actual-Z compensation,1N/±2mm and original2s handoff are unchanged.
- Residual controls only XY/yaw after the native2s handoff. Keep original
  residual bounds, slew, recurrent SAC, reward accounting and safety guards.
- Six-hour total wall budget /50,000 residual-decision cap; stop training
  collection by hour5, final validation inside the same cap; ACT on MPS and
  residual learning on CPU. Keep original disk8GiB/output4GiB/memory protections.
- Original single-opportunity and no-auto-restart rules remain in force.
  Infrastructure errors stop the invocation, not ordinary-failure substitution.

## Fresh layouts and ordered gates

Use one new `--mode pilot --acknowledge-pilot-training` invocation. Its own
preflight runs exactly once; do not first run a separate preflight on these seeds.

- Same-stream zero shadows: new seeds3152300 and3152301, targets1 and6.
- Start/brake/reverse probes: new seeds3152400 and3152401, targets1 and6.
- Seal consumed seed/pose evidence from all three stopped residual invocations
  AND the separate native Hybrid check seed3152200. Exclude those poses even
  if produced under a different seed; old120 remain excluded as before.
- Baseline/final validation3140000–3140005 and training3300000+ are unchanged
  because none was consumed in the earlier stopped runs.
- Require actual nonempty ACT same-stream byte-exact zero proof, and real
  bounded start/brake/reverse motion probes, before baseline validation and
  learner initialization. A failed new gate stops; do not replace its seed.

Training videos stay disabled. Keep numerical training records/checkpoints and
the six paired baseline/final validation videos. No recurring automation,
ordinary-ACT training resumption or formal240-episode evaluation is authorized.
The new run's files, not this authorization note, establish actual progress.
