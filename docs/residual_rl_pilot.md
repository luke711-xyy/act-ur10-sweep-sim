# Hybrid ACT + recurrent residual SAC pilot

Implementation contract, approved 2026-10-15. Implementation and tests only;
**no pilot training, resumed training, scheduled job or 240-rollout evaluation is
authorized by this change**. The CLI must require an explicit run acknowledgement.

## Frozen contract

Reuse the completed trial-4 Hybrid A* -> five-view ACT batch and step_100000.
Do not edit sim/, existing batch scripts, exports, datasets, checkpoints, or
historical ledgers. Reuse the existing worktree, no commits/pushes/worktree copies.
The original 1 N wrist-only two-stage Z loop (seek 10/400/10, maintain 10/800/10,
combined correction +/-2 mm, 1 mm gate), first two-second ACT handoff, shared
2000 action frames at 25 Hz, and native collection termination remain intact.

Residual: XY/yaw velocity only, 25 Hz, starts after the original two-second
handoff; integrate at 100 Hz. Desired XY residual norm <= .16 m/s, yaw <= .7
rad/s; residual acceleration <= .2 m/s2 and .5 rad/s2. Executed norm <= .08 m/s,
yaw <= .35 rad/s. No accumulated displacement budget. Zero residual must use
the unchanged native command path. Never reset or command Z.

Reward: continuous physical geometry contact for .020 s, reset dwell on any
break; unique components counted once from episode start, including prefix.
First N earn 60/N, remaining earn -120/N. Prefix events consume quota but are
not retrospective RL credit. Simultaneous events use counts, not identity order.
Terminal success +60, ordinary failure (including IK) -20, safety -60, mutually
exclusive. No clipping, normalization, proximity/force/time/action shaping, or
penalty cap. Gamma 1. Infrastructure errors stop rather than fabricate failure.

Actor: frozen ACT action-head input features associated with the accepted
timestamped chunk/index; 5 Hz visual queries, 25 Hz proprioception/context.
No object truth, collected counts, contact identities, reward inputs, or
collection-window truth. Wrist sensor only. Whole-episode GRU memory including
prefix (features absent until ACT becomes available); reset per episode.
Privileged critics may see component poses/velocities, collected/contact masks,
and controller state but never simulator contact-force truth.

## Task 1: pure control, rewards, observation boundary (main)

New residual_rl/config.py, rewards.py, control.py, observations.py. Test first:
all six rewards; debounce, deduplication, prefix credit, simultaneity; terminal
mutual exclusion; exact zero path; speeds, residual slew, reverse, unchanged Z;
explicit actor allowlist / no privileged or reward leakage.

## Task 2: recurrent SAC and replay (worker)

Read docs/residual_rl_learner_brief.md for the exact shared API. Worker owns ONLY
residual_rl/learning.py and tests/test_residual_learning.py. Real CPU torch tests,
no ACT/model/physics/training workload. Main owns all other files.

## Task 3: frozen runtime adapter, run protection and pilot CLI (main)

Reuse original controller through process-local instrumentation; no source edits.
Capture contacts each 1 kHz physics substep without reading mj_contactForce.
Correlate frozen fusion tokens with accepted ACT scheduler chunks; stream numeric
experience rather than retain training RGB. Preserve raw trajectories and states.
Fresh balanced training only, disjoint six validation seeds; exclude old120 seeds
and pose fingerprints. Single opportunities, append-only ledger, attempt markers,
no recovery/retry on infrastructure error. Save actual count or null, never invent.
6h total, 50k residual decisions, stop training collection at 5h. Reserve final
validation within same budget. Disk >=8 GiB, outputs <4 GiB, RSS <=10 GiB sustained
30s, persistent dangerous system pressure => stop/save, never kill other tasks.
Only validation videos; use saved-state rendering (not a second physical attempt).
Preflight real ACT equivalence and bounded motion probes gate any learning.

## Task 4: integration, documentation and independent review

Focused tests then project suite with bounded execution; report incomplete suite
honestly. Hash frozen files before/after. CLI help and inspect must not initialize
MuJoCo or load ACT. No claim of physical validation without actual rollout proof.

## Rulings / progress

- User implementation request authorizes code and tests, not the separately gated
  six-hour training pilot. Preserve that boundary in CLI and handoff.
- Existing untracked benchmark scripts/tests are user work; leave them unchanged.
- Interfaces checked: reward/control -> runtime, numeric episode -> replay/learner,
  guards -> pilot orchestration. Prefix observations remain replay burn-in context,
  not learned transitions. No conflicting file ownership across the two lanes.
- Planned initial defaults: CPU residual actor/twin critics, GRU128, MLP128;
  lr .0003, tau .005, grad clip5, gamma1, entropy .001, log_std initial -4;
  100k context steps, batch4 x32, full prefix burn-in; warmup256 plus all goals,
  one update per25 newly collected residual decisions, updates only between episodes.

## Verification ledger (implementation, not a training run)

- Core reward/control TDD:16 expected missing-feature failures, then16 passed.
- Runtime/ledger/observation contracts: missing-feature failures before code;
  later administrative-truncation and output-isolation regressions also RED/GREEN.
- Recurrent learner worker:41 CPU tests passed; independent spec/quality review
  found no important issues. Actual ACT/physics was not run by that worker.
- Media worker:43 RED cases then50 GREEN; synthetic MuJoCo state restoration and
  tiny real ffmpeg encode/full-decode tests, no actual sweep inference.
- Integrated regression before final review fixes:216 passed,0 failures/errors/
  skips,4.59s. Includes residual tests and unchanged Hybrid/fresh-A* batch tests.
- Project-wide pytest attempted: interrupted after336.61s;203 named cases
  completed with0 failures/errors. This is NOT a full-suite pass. No claim of
  coverage of the unfinished expensive MuJoCo tests.
- Read-only startup audit: expected67 vs current69 simulator sources; added
  configuration_dinov3.py,dinov3.py; changed policy.py,train.py. Current HEAD
  21f433c includes prior DINOv3 commits. New code refuses preparation/physical
  preflight rather than re-sealing the old baseline. git diff/status for sim/
  remains empty. All seven step_100000 checkpoint file size/mtime identities
  match historical metadata (not a new full weight-hash assertion).
- Final whole-subsystem reviewer found3 Important issues: bounded shutdown save,
  executed-motion probe audit, and same-seed initial-pose pairing. Consolidated
  fix wave completed. The original reviewer marked all3 ADDRESSED and both
  scoped spec/quality reviews PASS, with no new Critical/Important regressions.
- Final independent regression command:
  `.venv/bin/python -B -m pytest -o addopts='' -q tests/test_residual_*.py tests/test_hybrid_paired_batch.py tests/test_astar_fresh_batch.py --junitxml=docs/residual_rl_verification.xml`
  Result:263 passed,0 failed/errors/skips in5.11s. Scoped re-review independently
  ran68 pilot/runtime/reporting tests:68 passed in1.59s. Media validation uses
  synthetic states/tiny ffmpeg fixtures, NOT real sweep demonstration evidence.
- Shutdown now reserves60s/64MiB before hard limits, admits bounded compact
  saves<=32MiB, keeps verified prior snapshots if a new save is unsafe, records
  explicit save outcomes, and uses a cooperative supervisor request before
  escalating only against its own worker group. No hard resource cap increased.
- Per-episode numeric experiences are atomically persisted before ledger commit;
  interrupted diagnostics are explicitly incomplete and never become valid
  replay/results. Probe audits reconstruct actual command deltas and separate
  TCP reversal evidence from contact claims. Paired reset pose hashes must match.
- Initial code implementation and scoped verification were complete. Real ACT preflight,
  OpenGL whole-scene validation, training, and formal240-rollout comparison have
  NOT run. At that checkpoint the frozen-source mismatch was a deliberate startup
  blocker requiring an explicit baseline decision; no automatic repair/resume.

## User-selected pre-DINO compatibility, 2026-10-15

- User decision: "用回 dino 改造之前的". This selects source baseline only;
  no training, physical preflight, old experiment resumption or Git rollback.
- Ruling: pin the isolated RL import namespace to the sealed historical67 source
  hashes and commit9102680d8e018b225ab34d36422572cd4b547166, preserving the current
  DINO branch and69 disk files. Both modified Git blobs (`policy.py`, `train.py`)
  independently match the old SHA256 map. Other65 source bytes are reused only
  when hash-identical; any unrelated mismatch is rejected. No old seal changes.
- One exploratory recursive Git tree read stalled; only our read process was
  terminated. The implementation uses bounded, targeted reads of the two blobs,
  not recursive Git traversal or a source-directory copy. No user task stopped.
- New `baseline.py` provides immutable verified bytes and a process-local sim
  import allowlist. Extra DINO modules cannot import; any already-loaded foreign
  sim namespace fails closed. Virtual sibling filenames preserve asset/config
  roots and pinned `inspect` text for the existing AST adapters, without disk files.
- `identity()` retains historical seal/profile/frozen-input/checkpoint-stat guards,
  independently verifies effective sources, and records current disk fingerprints
  in new-run evidence. Historical formal verifier and all historical inputs stay
  unchanged. Default inspection does not import sim, torch or MuJoCo.
- TDD:8 new expected failures before implementation, then8 pass; module-level
  inspection regression separately failed before the virtual-origin fix, then
  passed. Initial focused regression271 passed in5.80s before that inspection fix.
- Four small-fixture guard cases verify the original metadata seal, trial profile,
  frozen dependency hash and checkpoint stats still reject changes. Strengthening
  default inspection to use those same guards had4 expected failures before its
  fix, then12 baseline tests passed in0.93s. Only temporary fixture files changed.
- The actual `HybridRuntime` constructor was initialized read-only: effective67,
  native evaluate bound to the pre-DINO policy builder, empty policy cache, no
  episode executed. Historical metadata SHA256 remains
  d997bebbbf5e849fb35c2fea3ceea9c941e6f469499b1f84d59288be4cbe2961;
  formal script remains e42e3940d8732393b441a0497e27c990bdc118915b8f21b6e8d2625997537c71.
- New project-wide test attempt bounded at147.95s with SIGINT sent only to our
  verified pytest process:185 passed,1 upstream Starlette/httpx deprecation
  warning; no failures/errors observed. NOT a full-suite pass. This invocation
  did not launch the residual preflight/training CLI or a formal batch.
- Final fresh focused regression:275 passed,0 failures/errors/skips in6.11s;
  JUnit evidence at docs/residual_rl_verification.xml. Independent read-only
  review found no Critical/Important issue and independently passed78 baseline/
  runtime/pilot tests in2.58s. Its additional non-executing checks verified pinned
  streaming/strict/physics AST transformations, asset roots and default CLI's lack
  of sim/torch/MuJoCo imports. Non-blocking suggestion: retain those extra manual
  checks as dedicated regressions in future; core import/guard coverage is present.
- Review rulings: physical behavior/model loading and RL learning quality remain
  deliberately unverified because no real ACT preflight or pilot was authorized.
  Full-suite completion and unrelated learner/control changes are outside this
  compatibility review; no full-suite success is claimed. No resumed experiments,
  commits, branch changes, scheduled jobs, model writes or historical edits.

Implementation choices documented for handoff: fixed seed schedules are sealed
up front; their actual pose fingerprints are checked after reset but before any
commanded execution, rather than spending the pilot budget pre-simulating50k
layouts. Checkpoint milestones are saved between episodes and named with the
actual step count, not falsely labelled exact10k. Probe/training/final residual
decisions share the50k cap, reserving up to12k for final validation. None of these
choices authorizes an actual pilot run or weakens the physical safety envelope.
