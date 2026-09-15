# Robot Learning Workbench Design

## Goal

Provide a local MuJoCo workbench that exposes the simulator and custom force-control state while borrowing proven LeLab, VTPRL, teleop-data-analyzer, and LeRobot Dataset Visualizer patterns for episode review, job control, and configuration.

## Scope

- Add a fixed斜前上方 inspection camera rendered by MuJoCo.
- Keep the inspection camera out of `act.observation_keys`; ACT continues to train on the existing overhead and wrist RGB observations.
- Let the user browse saved local ACT episodes, play their camera frames, and run an ephemeral expert preview when no dataset episode exists.
- Expose local ACT training and MuJoCo ACT inference as guarded, observable jobs using the repository's existing Python entry points.
- Expose an editable, validated parameter form with explicit apply/reset semantics.
- Keep detailed LeRobot dataset exploration delegated to the existing LeRobot visualizer/Rerun/Foxglove tools; the local UI is a MuJoCo control surface, not a second general-purpose dataset product.

## Non-goals

- No WAM support.
- No real-robot command, calibration, upload, HF authentication, or cloud billing action.
- No change to the ACT observation schema caused by adding an observation-only camera.
- No physics shortcut that prevents or hides physically valid lateral sliding.
- No automatic expert-data generation from a UI refresh or preview action.

## Architecture

The existing FastAPI app remains the local service boundary. A small in-memory workbench state owns the current MuJoCo environment, preview records, and local subprocess jobs. Read-only endpoints expose config, scene health, camera frames, episode metadata, and job status. Mutating endpoints are limited to reset, validated config apply, expert-preview generation, and starting/stopping the repository's own training/inference entry points.

The browser page becomes a compact five-area workbench: Simulation, Demonstrations, Training, Inference, and Parameters. Simulation uses three images: `overhead_cam`, `wrist_cam`, and `inspection_cam`. Demonstration playback reads the existing local `ActDatasetWriter` format and uses a browser video-like frame slider; formal dataset writing remains unchanged except for an optional inspection sidecar, which is never read by `ActDataset`.

## Interfaces

- `GET /api/frames` returns `overhead`, `wrist`, and `inspection` PNG data plus a `camera_roles` map.
- `GET /api/episodes` returns local manifest rows with episode id, success, seed, count, length, and failure reason.
- `GET /api/episodes/{episode_id}` returns episode metadata and frame counts.
- `GET /api/episodes/{episode_id}/frame/{frame_index}` returns the selected overhead, wrist, and optional inspection PNGs.
- `POST /api/preview` accepts `{seed: int, count: int | null}` and returns a preview id after running the existing expert loop in a separate preview directory.
- `GET /api/preview/{preview_id}` and `GET /api/preview/{preview_id}/frame/{frame_index}` mirror the episode endpoints.
- `GET /api/jobs` returns local job state and bounded recent logs.
- `POST /api/jobs/train` starts `python -m sim.act.train` with validated config/dataset/output/steps values.
- `POST /api/jobs/inference` starts `python -m sim.act.evaluate` with validated config/seed/model values.
- `POST /api/jobs/{job_id}/stop` stops only a job owned by this workbench.
- `POST /api/config/apply` accepts a restricted flat map of known numeric/string paths, validates ranges, persists a workbench config copy, and resets the scene.

## Safety and truthfulness

- Subprocess arguments are built from typed fields; the browser never supplies a shell command string.
- Only one local training job and one local inference job may run at once; duplicate starts return HTTP 409.
- Preview results are clearly marked `preview=true` and are not added to `cfg.act.dataset_dir` or its manifest.
- A failed episode retains its failure reason and physical trace; no success relabeling occurs in the UI.
- Jobs expose `running`, `done`, `failed`, or `stopped` state and bounded stdout/stderr for diagnosis.

## Verification

- Unit tests cover camera-role separation, episode listing/frame bounds, preview isolation, config validation, duplicate job rejection, and typed command construction.
- Existing full pytest suite must remain green.
- Start the local server, open the workbench in Chrome, verify all five areas are visible, verify the third camera is rendered, and exercise reset/refresh plus at least one episode or preview playback path.
