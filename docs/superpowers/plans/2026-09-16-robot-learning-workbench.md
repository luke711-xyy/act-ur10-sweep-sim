# Robot Learning Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the local MuJoCo ACT dashboard with an observation-only inspection camera, episode/expert-preview playback, local training and inference controls, and validated parameter editing.

**Architecture:** Keep FastAPI and the existing MuJoCo/ACT modules as the backend boundary. Add a small workbench service layer for local dataset reads, preview sidecars, typed subprocess jobs, and config validation, then replace the current minimal HTML with a five-area browser workbench. Use LeLab's API/page separation, VTPRL's bounded telemetry/jobs, teleop-data-analyzer's synchronized review ideas, and LeRobot's dataset conventions without copying unrelated hardware or cloud controls.

**Tech Stack:** Python 3, FastAPI, MuJoCo, NumPy, Pillow, vanilla browser JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-robot-learning-workbench-design.md`

## Global Constraints

- The inspection camera is observation-only and must not enter `act.observation_keys`.
- No WAM support and no real-robot/cloud/upload/auth side effects.
- Physically valid lateral sliding remains visible and failed episodes retain their failure reason.
- Preview generation never appends to the formal training dataset.
- Browser-provided values are typed and range-validated; arbitrary shell commands are rejected.
- Existing tests and legacy Cartesian tests must remain green.

---

### Task 1: Add the inspection camera and typed workbench data helpers

**Files:**
- Modify: `configs/default.yaml`
- Modify: `sim/model/scene_builder.py`
- Modify: `sim/environments/sweep_env.py`
- Create: `sim/web/workbench.py`
- Test: `tests/test_workbench.py`

**Interfaces:**
- `SweepEnv.render_rgb("inspection_cam", size=(w, h))` returns an RGB array.
- `WorkbenchState.list_episodes()` returns JSON-serializable episode metadata.
- `WorkbenchState.load_episode_frame(episode_id, frame_index)` returns camera payloads or raises a typed not-found/bounds exception.
- `WorkbenchState.build_preview(seed, count)` returns a preview id and stores preview files outside `cfg.act.dataset_dir`.

- [ ] **Step 1: Write failing tests for camera roles and episode helpers**

```python
def test_inspection_camera_is_not_an_act_observation(cfg):
    assert "inspection_cam" not in cfg.act.observation_keys
    env = SweepEnv(cfg, seed=0)
    env.reset(seed=0)
    try:
        frame = env.render_rgb("inspection_cam", size=(96, 96))
        assert frame.shape == (96, 96, 3)
    finally:
        env.close()
```

```python
def test_episode_frame_bounds_and_preview_isolation(tmp_path, cfg):
    state = WorkbenchState(cfg, dataset_root=tmp_path / "dataset", preview_root=tmp_path / "preview")
    with pytest.raises(EpisodeNotFound):
        state.load_episode_frame("missing", 0)
    preview_id = state.build_preview(seed=0, count=1)
    assert (tmp_path / "dataset" / "manifest.jsonl").exists() is False
    assert state.preview_metadata(preview_id)["preview"] is True
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_workbench.py -q`

Expected: FAIL because `inspection_cam` and `WorkbenchState` do not exist yet.

- [ ] **Step 3: Add the fixed inspection camera**

Add `inspection_cam` under `video.extra_cameras` with a fixed position above and in front of the table. Extend the scene builder's existing extra-camera path and keep `ACTObservationBuilder` unchanged. Add a `render_inspection_rgb` convenience method only if it reduces repeated camera-name strings.

- [ ] **Step 4: Implement dataset and preview helpers**

Implement `WorkbenchState` using `ActDatasetWriter`'s manifest format. Read only `manifest.jsonl`, `arrays.npz`, and image files for saved episodes. For previews, run `run_expert_episode` with a copied config whose count is either the requested count or the configured count, write to `preview_root/<preview_id>`, and never touch `cfg.act.dataset_dir`.

- [ ] **Step 5: Run the focused tests and confirm they pass**

Run: `pytest tests/test_workbench.py -q`

Expected: PASS with camera shape, missing-episode, and preview-isolation assertions green.

- [ ] **Step 6: Commit the task**

```bash
git add configs/default.yaml sim/model/scene_builder.py sim/environments/sweep_env.py sim/web/workbench.py tests/test_workbench.py
git commit -m "feat: add inspection camera and workbench episode helpers"
```

### Task 2: Add FastAPI routes for frames, episodes, previews, config, and jobs

**Files:**
- Modify: `sim/web/app.py`
- Create: `sim/web/jobs.py`
- Test: `tests/test_workbench.py`

**Interfaces:**
- `GET /api/frames` returns `overhead`, `wrist`, `inspection`, and `camera_roles`.
- `GET /api/episodes` and `GET /api/episodes/{id}/frame/{index}` expose saved episodes.
- `POST /api/preview` accepts JSON `{seed, count}` and returns `{preview_id, metadata}`.
- `GET /api/jobs` returns bounded job snapshots.
- `POST /api/jobs/train`, `POST /api/jobs/inference`, and `POST /api/jobs/{id}/stop` use `JobRegistry`.
- `POST /api/config/apply` validates a restricted map and resets the MuJoCo environment.

- [ ] **Step 1: Write failing route tests**

```python
def test_frames_expose_three_roles(client):
    response = client.get("/api/frames")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["camera_roles"]) == {"overhead", "wrist", "inspection"}
    assert payload["camera_roles"]["inspection"] == "inspection-only"
```

```python
def test_config_rejects_unknown_path(client):
    response = client.post("/api/config/apply", json={"shell.command": "rm -rf"})
    assert response.status_code == 422
```

- [ ] **Step 2: Run route tests to confirm failure**

Run: `pytest tests/test_workbench.py::test_frames_expose_three_roles tests/test_workbench.py::test_config_rejects_unknown_path -q`

Expected: FAIL because the new routes and registry do not exist.

- [ ] **Step 3: Implement typed job registry**

Create `JobRegistry` with `start(kind, args)`, `snapshot()`, and `stop(job_id)`. Launch only `sys.executable -m sim.act.train` or `sys.executable -m sim.act.evaluate`; store argv, timestamps, return code, and the last 200 combined output lines. Reject a second job of the same exclusive kind with HTTP 409.

- [ ] **Step 4: Add the FastAPI routes**

Use the workbench helpers for frame/episode/preview responses. Use `Pydantic` request models or explicit numeric checks for seed, count, steps, and path fields. Config apply must whitelist numeric paths such as `controller.desired_force`, `controller.sweep_speed`, `act.batch_size`, `act.chunk_size`, `act.max_steps`, and `episode.max_time`, and then call `save_config` to a workbench-local YAML copy before resetting the scene.

- [ ] **Step 5: Run focused route tests**

Run: `pytest tests/test_workbench.py -q`

Expected: PASS; job tests must use a fake executable or monkeypatch `subprocess.Popen`, never start a real long training run.

- [ ] **Step 6: Commit the task**

```bash
git add sim/web/app.py sim/web/jobs.py tests/test_workbench.py
git commit -m "feat: expose workbench control and job APIs"
```

### Task 3: Replace the minimal page with the five-area workbench

**Files:**
- Modify: `sim/web/app.py`
- Test: `tests/test_workbench.py`

**Interfaces:**
- Browser controls call only the typed `/api/*` routes from Task 2.
- Three camera cards show the third card with an `inspection-only` label.
- Episode controls support select, play/pause, previous/next frame, and a range slider.
- Training, inference, and parameters forms show status and errors without exposing shell command text as an editable field.

- [ ] **Step 1: Add HTML smoke assertions**

```python
def test_workbench_html_has_five_areas(client):
    html = client.get("/").text
    for label in ("Simulation", "Demonstrations", "Training", "Inference", "Parameters"):
        assert label in html
    assert "inspection-only" in html
    assert "Play" in html
```

- [ ] **Step 2: Run the HTML test and confirm failure**

Run: `pytest tests/test_workbench.py::test_workbench_html_has_five_areas -q`

Expected: FAIL because the current page has only two images and two buttons.

- [ ] **Step 3: Implement the five-area page**

Use semantic sections and small cards in the existing single-file HTML. `refresh()` updates health and all three images. `loadEpisodes()` populates a select. `playEpisode()` uses a bounded timer and `/api/episodes/{id}/frame/{index}`; it never treats a failed episode as success. Training/inference submit typed JSON forms and poll `/api/jobs`; parameters use numeric inputs and an explicit Apply button.

- [ ] **Step 4: Add external-tool links without duplicating them**

Add a short “Detailed dataset viewer” note with the local `lerobot-dataset-viz` command and a link to the LeRobot Dataset Visualizer. Do not add a second statistics/annotation implementation to this page.

- [ ] **Step 5: Run the HTML test**

Run: `pytest tests/test_workbench.py::test_workbench_html_has_five_areas -q`

Expected: PASS.

- [ ] **Step 6: Commit the task**

```bash
git add sim/web/app.py tests/test_workbench.py
git commit -m "feat: build five-area MuJoCo ACT workbench UI"
```

### Task 4: Verify end-to-end behavior and document the reference integration

**Files:**
- Modify: `README.md`
- Modify: `tests/test_mujoco_smoke.py` if the camera inventory test needs the observation-only camera assertion
- Test: `tests/test_workbench.py`, full test suite

**Interfaces:**
- The README contains exact startup and browser URL commands plus the five-area capability boundary.
- The running page serves three visible camera images and non-loading health status.

- [ ] **Step 1: Run all tests**

Run: `pytest -q`

Expected: PASS with zero failures.

- [ ] **Step 2: Start the workbench from the feature branch**

Run: `PYTHONPATH=. .venv/bin/python -m sim.web.run --config configs/default.yaml`

Expected: Uvicorn serves `http://127.0.0.1:8765` and `/api/health` returns HTTP 200.

- [ ] **Step 3: Verify browser behavior in Chrome**

Open `http://127.0.0.1:8765`, verify the three camera cards, click Refresh, select or generate one preview, and move the frame slider. Verify the Parameters Apply button resets the scene and the Training/Inference forms return a visible validation error when required paths are empty.

- [ ] **Step 4: Update README**

Document the local workbench URL, the five areas, the inspection-only camera contract, preview isolation, and the recommended LeRobot visualizer command for detailed datasets.

- [ ] **Step 5: Commit verification/docs**

```bash
git add README.md tests/test_workbench.py tests/test_mujoco_smoke.py
git commit -m "docs: document the robot learning workbench workflow"
```
