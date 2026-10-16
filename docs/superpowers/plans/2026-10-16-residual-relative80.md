# Relative residual 80% implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Implement approved ACT-relative B/80% XY and +/-45-degree yaw actions,
then execute the authorized zero/continuous actual-ACT preflight without learning.

**Architecture:** A pure action generator owns geometry; runtime supplies actual
TCP and accepted ACT targets at 25Hz, and leaves native 100Hz execution/Z intact.
Legacy pose helpers remain only for historical characterization tests.

**Tech Stack:** NumPy, PyTorch, pytest, existing MuJoCo/ACT, saved-state movies.

**Spec:** `docs/residual_relative_action.md` (user approval in this chat).

## Global constraints

- No edits to 67 sim sources, frozen Hybrid entry/dependencies/models/results.
- No training, actual optimizer updates, Git commits/push, or new worktree.
- One fresh target1 layout, one zero arm and one continuous arm; no retries.
- Existing unrelated workbench-video plan and all user changes are preserved.

## Review focus

- Actual TCP vs previous command/virtual reference: ratio uses actual frame XY.
- Zero-length ACT motion, wrap/corners, nonfinite data: valid exact zero or stop.
- Raw SAC density vs generated target: same square-to-disk map in policy guard.
- Missing/stale chunks: no additional motion, no prediction timing changes.
- Frozen sources and no learner initialization: verify before physical execution.

### Task 1: Feasible relative action generation

- [x] Write literal geometry, scaling, zero, yaw/workspace and invalid-input tests.
- [x] Run RED, implement `relative_action.py`, run GREEN and stochastic properties.

### Task 2: Runtime, observation and persistence contract

- [x] Write accepted-target / actual-anchor / zero / missing-chunk integration tests.
- [x] Update runtime, action config, observation context and matching policy units.
- [x] Reject old action contracts; preserve native zero path, queries and Z owner.
- [x] Run residual suite and original Hybrid/ensemble regressions; document failures.

### Task 3: Preflight audit and review

- [x] Test continuous 25Hz signal and saved-trace audits including corruption.
- [x] Fresh read-only code review; fix important findings with RED/GREEN evidence.
- [x] Seal new artifacts and ensure resources/frozen inputs/lease are safe.

### Task 4: Authorized physical pair and media

- [x] Run native zero and continuous micro-adjustment on one new layout, once each.
- [x] Stop on IK/safety/unknown exception and retain actual final quantities.
- [x] Saved-state rendering only; stack zero above continuous at original 25fps.
- [x] Force/Z/offset curves, IK facts, complete MP4 decode/hash/visual verification.
- [x] Final source/ledger/zero-training audit; report honestly, do not start training.

## Ledger

User has explicitly requested implementation and the actual preflight in this
turn; execute inline. No additional execution authorization is inferred for RL
training. Evidence is written under a new exports directory, not old results.

## Completion evidence

- `exports/residual_relative80_verification_20261016_115913/`:310 focused passes;
  1046 broader passes/7 independently reproduced legacy failures; bounded full
  suite is explicitly not claimed as passing.
- `exports/residual_relative80_preflight_20261016_115913/videos/edit/`:
  seed3158000, target1, one physical attempt per arm; both strict successes1/1.
  Zero shadow1758 controls identical; continuous391 frames/390 corrections,
  all1563 active controls audited. Maximum XY ratio.22906, correction2.826mm,
  yaw4.365deg. No IK exception; no training/checkpoint.
- Stacked MP4:611 frames,25fps,24.44s,960x1408; full decode/hash verified and
  first/middle/late/final frames visually reviewed. Both source videos retained.
