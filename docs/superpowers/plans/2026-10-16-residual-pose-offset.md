# Bounded Residual Pose Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Replace unrestricted residual velocities with nonaccumulating 5 mm / 3 degree pose corrections.

**Architecture:** Keep an isolated nominal-reference anchor at query submission and control-frame boundaries. Project smoothly changing pose offsets onto the intersection of pose, rate, workspace, total-speed and forward-motion constraints. Rebind learner/observation/checkpoint units without changing frozen Hybrid/ACT/Z.

**Tech Stack:** Python, NumPy, PyTorch, pytest; no new dependencies.

**Spec:** `docs/residual_pose_offset.md` and the user's explicit implementation authorization.

## Global Constraints

- XY norm <=0.005 m; yaw offset <=pi/60 rad; no accumulated offset increments.
- Frozen 67 sim sources, formal Hybrid entry, ACT scheduler and Z source remain byte-identical.
- No actual preflight/training or publication. Preserve existing dirty work and historical results.
- No automatic retry, fallback on constraint failures, or old velocity-checkpoint compatibility.

## Review Focus

- Corrected starts becoming future query/frame anchors: test repeated asynchronous captured starts.
- Idle/turning native trajectories: test frozen offsets and constrained direction changes.
- Total velocity saturation vs lateral correction: test feasible projection at 80 mm/s.
- Workspace/rate/pose constraints conflicting: test rejection without state mutation.
- Same-shaped old snapshots and offset state leakage: test explicit contracts and Actor boundary.

### Task 1: Pose control and nominal anchoring

**Files:** `residual_rl/control.py`, `residual_rl/config.py`, new `residual_rl/pose_constraints.py`, `tests/test_residual_pose.py`.
**Interfaces:** `ResidualGovernor.step(..., native_before=None)`, `remove_offset(reference)`, `ResidualReference.begin_frame/step`, isolated executor snapshot wrapper.

- [x] Write failing numeric tests for zero, pose bounds, repeated anchors, forward motion, full-speed steering, idle, wraps and infeasibility.
- [x] Run tests RED; implement minimum bounded projection and anchoring; rerun GREEN.

### Task 2: Runtime, learning contract and probes

**Files:** `residual_rl/runtime.py`, `observations.py`, `policy_guard.py`, `learning.py`, `pilot.py`; associated residual tests and README.
**Interfaces:** Actor observation adds `residual_offset`; pose actions remain normalized raw vectors; replay schema2 / learner schema3 bind `action_contract`; new pose-probe audit.

- [x] Write failing integration/schema/unit tests; verify RED.
- [x] Integrate the control adapter and captured nominal query reference, preserve native scheduler timing, add state/audits and replace brake/reverse probe.
- [x] Migrate old velocity-specific tests to the approved pose behavior and verify GREEN.

### Task 3: Verification and handoff

**Files:** new output `exports/residual_pose_offset_tests_20261016`; source audit and reports.

- [x] Run all residual tests, whole-project pytest (bounded if slow), and available non-slow regressions; report every failure or incomplete run.
- [x] Fresh read-only review; reproduce/fix important findings by RED/GREEN.
- [x] Verify frozen code/model identity, no inference/training processes, no new rollout or training checkpoint; report evidence and limitations.
