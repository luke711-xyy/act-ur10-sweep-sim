# Rudra Sweep Overlay Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** Show the original sweep tray, six-part layout and clamped brush on the native Rudra table without moving its cameras.

**Architecture:** Reuse original task sampling and task MJCF objects in a separate overlay. Translate the task origin to the native tabletop; retain native robot/camera assembly and render path. Keep the original native profile and all other worktrees intact.

**Tech Stack:** Python, NumPy, SciPy, MuJoCo EGL, FastAPI, existing workbench HTML.

**Spec:** `docs/rudra-sweep-preview.md` describes the user's requested scene and coordinate/grasp contracts.

## Global Constraints

- Preserve native camera settings and CAD mount transforms.
- Reuse original dimensions, layout RNG and part attributes; rigid translation only.
- Keep original dataset and active training unchanged. No policy rollout or training in this task.
- Preserve unrelated local changes and all vendored upstream file hashes.

## Review Focus

- Same seed must reproduce the original layout with only translation.
- Freejoint parts must remain top-level world bodies.
- Brush collision masks must allow parts and the compliant sole/table pair.
- Handle must fit measured jaws and follow the robot rigidly.
- Invalid seeds must not destroy the current scene; camera changes must not sneak into reset.

### Task 1: Scene/reset adapter

Files: `sim/model/rudra_sweep.py`, `configs/rudra-sweep-scene.yaml`, `tests/test_workbench_rudra_sweep.py`.

- [x] Write/run failing tests for layout, camera/mount preservation, physical geometry and fixed grasp.
- [x] Implement translated task-object overlay and native-kinematic initial brush pose.
- [x] Investigate native opening calibration mismatch and measure actual pads before fixed mounting.
- [x] Run scene tests and inspect real EGL images.

### Task 2: Workbench delivery

Files: `sim/web/rudra_preview.py`, `docs/rudra-sweep-preview.md`, `docs/rudra-native-preview.md`.

- [x] Test seed regeneration, deterministic reset, invalid seed and three real RGB frames.
- [x] Connect overlay and seed controls while keeping native-profile behavior.
- [x] Run scoped workbench/scene regressions and check unchanged upstream hashes (86 tests, 118 hashes).
- [x] Switch only 8780 to the sweep profile; verify through the browser and save evidence.
- [x] Independent code review found no Critical or Important issues; commit only task-owned files.
