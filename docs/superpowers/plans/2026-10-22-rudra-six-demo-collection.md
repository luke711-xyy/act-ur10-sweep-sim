# Rudra Contact Validation and Six Demonstrations Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation.

**Goal:** Verify original contact control in the new scene, then capture exactly one two-view expert demonstration for each goal1–6.

**Architecture:** Opt-in native dynamic environment behind the existing expert factory, original task-frame planning and dataset writer. Preserve ordinary inference defaults and the active trainer. Record failed control probes before accepting a passing profile.

**Tech Stack:** Python, NumPy, MuJoCo, original coverage A*, Matplotlib, WebP/schema-v5.

**Spec:** `docs/rudra-six-demo-collection.md`; user explicitly chose six previews with top+wrist.

## Global Constraints

- Native camera mounts/FOV remain unchanged; original RGB preprocessing at native resolution before320x320resize.
- Original dataset, training and unrelated local modifications remain intact.
- Preserve task geometry, part physics, sampling, exact-count success and frame/time budgets.
- Contact adaptation is independent and documented, including changed sole friction and original trial4 controller provenance.
- Failed probes remain evidence; no data generation before a matching passing physical report.

## Review Focus

- Frame translation and pi tool yaw must not corrupt commands, counts or images.
- Sensors/compensation must be real physics; servo commands must not teleport state.
- Contact rate must distinguish per10ms coverage, endpoint contact and1ms physics-step occupancy.
- Force acquisition must finish before XY motion.
- Exactly six pilot episodes; no300-demo gate bypass, mixed historical data or accidental training.

### Task 1: Dynamic native environment

- [x] Failing physical adapter tests then minimal servo/FT/frame implementation.
- [x] Opt-in expert environment routing and default-backend regression.
- [x] Reproduce lost1um correction; preserve base tolerance and tighten native IK.

### Task 2: Physical force check

- [x] Original0.1N and trial4 1N baselines, failed probes retained.
- [x] Independent review of force, attitude, clock and acquisition gates.
- [x] Isolate sole/table friction; preserve plate/part interaction.
- [x] Passing native physical contact/attitude report and source-matched configuration.

### Task 3: Six preview demonstrations

- [x] Failing gate/plan tests then original-rule six-entry collector.
- [x] Scoped28 tests passed before collection.
- [x] Review fixes: report physical occupancy; retry only known no-plan failure, stop unknown/recording errors.
- [x] Final scoped36 tests passed, including original exact-delivery termination regressions.
- [x] Persistent single-worker collection with bounded memory; each goal exactly once saved, exit0.
- [x] Verify saved NPZ, masks, targets,8914 images and actual outcomes; export six videos and force curves.
- [x] Final review, task-owned commit(fed143d) and documented actual artifact locations.
