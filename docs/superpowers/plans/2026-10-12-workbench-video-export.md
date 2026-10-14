# Workbench Video Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the currently displayed saved episode as a complete five-view MP4 beside Play, with progress, view and download links.

**Architecture:** Reuse the existing background callable job registry. Keep media-only encoding in `sim/web/video_export.py`; the web app resolves a source-qualified record before starting the job. Never run simulation, policy or training for an export.

**Tech Stack:** Existing FastAPI, NumPy, Pillow, imageio-ffmpeg and inline JavaScript; no new dependencies.

**Spec:** User approval in this chat: button beside Play, complete saved five-camera demonstration or inference, status and view/download, no simulation rerun or source-data mutation.

## Global Constraints

- Live 300-demonstration collection remains running; collection/core sources and default config remain byte-identical to launch.
- Preserve all existing dirty changes. Modify only app/playback web helpers and add a web encoder helper and tests.
- No new worktree, dataset collection entrypoint, training, commit, push or deletion of source frames.
- Preserve old API raw episode IDs while adding source-qualified playback IDs to distinguish duplicate stores.
- Keep the original collector launch fingerprints unchanged; record any approved web-only revision separately and audit its exact hashes.

## Review Focus

- Duplicate episode IDs in dataset and preview must show and export the same source.
- Changing selection during encoding must not change the captured export record.
- Missing cameras, invalid paths, non-finite FPS and corrupt MP4 must fail without publishing misleading output.
- Inference images already cleaned: reuse only the identity-verified original video inside the configured video root.
- Concurrent/failed/repeated exports must not overwrite results, freeze the UI or leave served partial files.

### Task 1: Saved media and source-qualified records

**Files:** `sim/web/workbench.py`, new `sim/web/video_export.py`, new `tests/test_workbench_video_export.py`.

**Interfaces:** `WorkbenchState.export_source(playback_id) -> (Path, dict)`; `export_saved_episode(root, record, output_root, *, video_root, progress, cancel_event) -> dict`.

- [x] Write/run failing behavior tests using real saved five-view frames and real encoding.
- [x] Add qualified IDs without changing old raw-ID behavior or generation methods.
- [x] Encode all native frames at recorded FPS; validate full decode before atomic publication; record provenance outside source dataset.
- [x] Validate and reuse cleaned inference media without touching its manifest/images.
- [x] Run focused tests and existing video/playback tests.

### Task 2: HTTP and UI

**Files:** `sim/web/app.py`, feature tests.

**Interfaces:** `POST /api/jobs/video-export` receives `{episode_id: playback_id}`; job progress contains original episode/source, stage and final `video_url`/`download_url`; safe `GET /api/episode-videos/{filename}`.

- [x] Write/run failing HTTP tests for missing/unsafe ID, async progress, busy conflict, download and failed export.
- [x] Add button beside Play, disabled/loading/error/success feedback; poll the existing registry and restore visible export after refresh.
- [x] Source-qualified selection is used throughout frame/signals/export requests.
- [x] Run focused tests and independent read-only review; add RED/GREEN regressions for material findings.

### Task 3: Apply and verify

**Files:** New evidence directory under existing exports; exact web-only revision recognition in external collection auditor.

- [x] Record unchanged core/config/protected artifacts, original launch hash and exact allowed web revision hashes.
- [x] Restart only the identified workbench listener, preserving the collector and its children.
- [x] Reload existing browser tab; click Export on a real saved demo; inspect resulting complete MP4, status, view/download and source identity.
- [x] Verify live collector continues advancing and report the feature without claiming all 300 demonstrations complete.
- [ ] Read the complete final bare pytest run and compare every failing test against the pre-feature baseline; never claim full green for baseline failures.

## Execution ledger

Ruling: The user's explicit “可以，按这个做” approves this bounded native implementation; no second design/execution confirmation. Evidence stays in exports and no commits/worktree cleanup are performed, preserving the dirty active checkout.

2026-10-12 13:10 SGT: 18 new tests and 70 related tests passed (counts read from the final JUnit reports). Real browser export/download verified for dataset:preview_goal_6_0010: 918 frames, 25 FPS, 36.72 s, five views, all 4592 source files unchanged. Changing selection during export retained the captured original episode. Actual workbench PID 86377; original collector PID 46876 remains running, 16 formally saved episodes audited. Full final regression is still running; the interrupted pre-review run is not a final result. Evidence: exports/demo_rebuild_y_edges_20261012/workbench_video_export under the user's canonical zhe workspace.
