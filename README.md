# act-ur10-sweep-sim — MuJoCo UR10 ACT tabletop sweep

A minimal but runnable **MuJoCo** prototype of a contact-rich sweeping task: a robot uses
its **fixed brush directly as a pusher** to collect small industrial components
(nuts, screws, bolts, washers) from a flat table into a fixed shallow tray at the table
edge.

There is no WAM and no grasping DOF. The scene uses the vendored MuJoCo Menagerie UR10e
six-joint model, a fixed brush, an overhead camera, a tool-mounted wrist camera, and a
separate observation-only oblique camera. The project
keeps the original Cartesian pusher/planner implementation as a legacy diagnostic path;
the default ACT path is the UR10 brush scene.

The first version focuses on five things:

1. contact with the table,
2. hybrid force/position control,
3. conventional (non-learned) visual and trajectory planning,
4. reproducible experiments over component count and distribution,
5. automatic demonstrations, LeRobot 0.6.1 ACT training, and asynchronous inference.

No Isaac Sim, no MoveIt. Python MuJoCo + NumPy + Matplotlib (SciPy for image labelling).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 0. health check -- do this first on a new machine
python -m sim.smoke_test

# 1. single episodes
python -m sim.run_episode --planner fixed         --num-components 1 --seed 0
python -m sim.run_episode --planner visual_greedy --num-components 5 --seed 0

# 2. conventional vision instead of ground truth, saving camera frames and masks
python -m sim.run_episode --planner visual_greedy --perception vision \
    --num-components 5 --seed 0 --save-observations

# 3. paired batch comparison (the headline experiment)
python -m sim.run_experiment --planners fixed global_sweep visual_greedy \
    --counts 1 2 3 5 10 --episodes 20

# 3b. the locked main task: 3-5 parts in ONE loose cluster
python -m sim.run_experiment --planners fixed global_sweep visual_greedy \
    --counts 3 5 --episodes 20 --spawn-mode cluster

# 3c. pick the force setpoint: where does a single fixed F_z* stop working?
python -m sim.run_force_sweep --forces 1 2 3 5 8 12 \
    --geometries washer hex_nut bolt --episodes 5

# 3d. record a two-camera video of one episode
python -m sim.record_video --planner visual_greedy --num-components 5 --seed 0
python -m sim.record_video --compare fixed global_sweep visual_greedy \
    --num-components 5 --seed 0        # same layout, one video per planner

# 4. re-plot a saved experiment
python -m sim.visualize_results --run runs/experiment-*

# 5. export demonstrations for later ACT training (no training happens here)
python -m sim.export_dataset --planner visual_greedy --episodes 200

# 6. default UR10 + ACT vertical slice
uv pip install -e '.[act,web]'
python -m sim.act.collect_dataset --split train --episodes 300
python -m sim.act.train --config configs/default.yaml
python -m sim.act.evaluate --config configs/default.yaml --model runs/act_model
python -m sim.web.run --config configs/default.yaml

# tests
pytest -q                     # or: python tests/run_tests.py
```

### Local robot-learning workbench

The local browser workbench is the integration surface for the current MuJoCo
prototype:

```bash
python -m sim.web.run --config configs/default.yaml
# open http://127.0.0.1:8765/
```

It contains five connected areas: live simulation views, saved/preview expert
episode playback, local ACT training jobs, MuJoCo inference jobs, and a bounded
parameter editor. The simulation shows the fixed perception overhead camera,
the ACT wrist camera, and a third **inspection-only** oblique-front-above
camera. The third camera is rendered and stored for human review but is never
included in `act.observation_keys` or passed to the ACT policy.

“Generate expert preview” runs the existing force-controlled expert once and
stores it under `runs/workbench_previews/`; this is separate from the formal
dataset, so preview failures (including excessive normal force or a component
sliding out of the safe workspace) remain visible without silently becoming
training labels. Training and inference are local subprocesses with captured
logs and stop controls; the panel does not upload data or control a real arm.

The UR10 preview uses a continuous task-space-to-joint adapter. Its IK solve is
performed in scratch state, the physical joint state is restored before stepping,
and `end_effector.ik_max_joint_step` bounds each joint target update. This keeps
near-singular least-squares solves from causing a visible branch jump. Demo
playback renders into separate image elements from the live simulation views, so
the periodic live refresh cannot overwrite a selected demo frame.

For richer dataset browsing, video playback, action plots, filtering, and
annotations, use the upstream [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
instead of duplicating that functionality in this project.

Every run writes a timestamped directory under `runs/` containing the exact
configuration used, metrics, phase events, the dense control trace and the plots.

### Useful flags

| Flag | Effect |
|---|---|
| `--set KEY.PATH=VALUE` | override any config entry, e.g. `--set controller.desired_force=5` |
| `--geometry {hex_nut,cylinder,screw,bolt,washer,mixed}` | component geometry family |
| `--spawn-mode {uniform,cluster}` | scattered parts (planner comparison) or one loose cluster (main task) |
| `--perception {ground_truth,vision}` | perception backend |
| `--jobs N` (experiments) | parallel worker processes |
| `--seed`, `--seed-base` | reproducibility |

Examples of domain randomisation, all through `--set`:

```bash
--set components.friction.min=0.2 --set components.friction.max=1.0   # friction
--set components.mass.min=0.002 --set components.mass.max=0.02        # mass
--set table.height_perturb_std=0.002                                  # table height
--set perception.noise.position_std=0.004                             # camera noise
--set perception.noise.dropout=0.15                                   # missed detections
--set controller.control_delay_steps=3                                # actuation delay
--set controller.force_noise_std=0.2                                  # F/T sensor noise
```

---

## Coordinate and task convention

* Table plane is **XY**, table normal is **+Z**, and the table surface is `z = 0`.
  Every height in the config is relative to the table surface.
* The target tray is **fixed at the left (−X) edge** and is *not* randomised in this
  version. Successful sweeps therefore move components generally towards **−X**.
* Component start poses are randomised but reproducible from `(config, seed)`, and no
  component ever starts inside the target region (a clearance band is enforced).

Default layout (metres):

```
   y
  +0.32 ┌──────────────────── workspace ───────────────────┐
        │ ┌──── tray ────┐                                 │
  +0.18 │ │              │                                 │
        │ │   TARGET     │        components spawn here    │
  −0.18 │ │              │                                 │
        │ └──────────────┘                                 │
  −0.32 └─────────────────────────────────────────────────┘
        −0.48         −0.34                             +0.46   x
        <---------------- sweep direction
```

---

## Episode phases

```
APPROACH → SEARCH_CONTACT → CONTACT_DETECTED → FORCE_RAMP → SWEEP
         → FORCE_RELEASE → RETRACT        ... then SUCCESS or FAILURE
```

Phases 1–7 run once per sweeping stroke; `SUCCESS`/`FAILURE` are episode outcomes.
The state machine (`sim/controllers/state_machine.py`) is free of MuJoCo and of
controller internals, and is unit-tested on its own.

**The release guard is a hard rule.** Once the TCP enters the guard band in front of the
tray entrance (`controller.x_release_line + controller.release_margin`), the desired
normal force is ramped to zero and the tool retracts — regardless of what the trajectory
was doing. The Z force loop never keeps searching for contact past the table edge or into
the tray mouth. This is enforced in three independent places (state machine, force
schedule, axis commands) and is covered by
`tests/test_hybrid_controller.py::test_force_is_released_before_the_guard_line`.

---

## Hybrid force/position controller

| Axis | Mode | Command source |
|---|---|---|
| X, Y | position | planner → smooth Cartesian trajectory |
| Z | **force** | outer admittance loop |
| yaw | position | stroke yaw (pusher face ⟂ stroke direction) |

The outer loop implements

```
m_d * d²w/dt² + b_d * dw/dt + k_d * w = F_desired − F_measured
z_command = z_nominal + Δz
```

where `w` is the correction **along the inward contact normal** (positive = press deeper)
and `Δz = −w` because the table normal is `+Z`. Both forces use the same
*pressing-positive* convention, so `controller.desired_force` is a small **positive**
value (default 1 N), never zero. Writing the ODE in the inward-normal frame is what keeps
the signs unambiguous; see the module docstring in `sim/controllers/admittance.py`.

`z_nominal` is latched only after a post-step brush contact load is measured, and it uses
the actual TCP height at that instant rather than a guessed or teleported command. The
admittance state is **reset at the start of every sweep**. The default search height is
1 mm below the table reference (`workspace.z_search_start=-0.001`), a small controlled
preload that gives MuJoCo a real contact constraint instead of relying on an exact
zero-gap equilibrium.

Implemented safeguards:

* simulated F/T sensing at the fingertip frame (two backends, see below)
* first-order low-pass filter on the normal force (`force_filter_cutoff_hz`)
* force-target slew-rate ramping, up **and** down (`force_ramp_rate`)
* force-target saturation (`max_force`) and a hard measured-force abort (`safe_max_force`)
* transient-force dwell (`safe_force_dwell_steps`) before the hard abort is declared
* `Δz` saturation (`delta_z_limit`) and `dΔz/dt` limit (`delta_z_rate_limit`), both with
  anti-windup
* contact-detection threshold with a dwell counter (rejects single-sample spikes)
* contact-loss accounting during the sweep
* optional actuation delay and force-sensor noise
* controller rate `sim.control_hz` (default 100 Hz), physics timestep
  `sim.physics_dt` (default 1 ms) — the physics step is always smaller than the control
  period, and the runtime asserts it.

### Controller tuning — the one non-obvious point

The admittance sees the environment *in series*: the closed-loop stiffness is
`k_d + k_actuator ≈ 60 + 3000 ≈ 3060 N/m`, not `k_d`. Sizing `b_d` against `k_d` alone
(`ζ = b_d / 2√(k_d·m_d)`) looks safely overdamped and is in fact badly **under**damped
once contact closes the loop, producing 40–50 % force overshoot. The defaults size the
damping against the closed-loop stiffness instead:

```
ω_n = √((k_d + k_act)/m_d) ≈ 55 rad/s      b_d = 2·ζ·ω_n·m_d ≈ 110–120 N·s/m
```

The remaining steady-state error is inherent to a non-zero `k_d`:
`F_ss = F_desired · k_act/(k_act + k_d)` ≈ 2.96 N for a 3.0 N target. Lower `k_d` for a
smaller offset, at the cost of a softer response to table-height error.

### Force sensing backends

| `controller.force_sensor` | Source | Notes |
|---|---|---|
| `contact` *(default)* | `data.cfrc_ext` on the tool body | total external contact force on the fingertip body; sign is unambiguous |
| `wrist_ft` | MuJoCo `force`/`torque` sensor at `ft_site` | a simulated wrist F/T sensor; `controller.wrist_ft_sign` maps its Z to pressing-positive |

`python -m sim.smoke_test` prints both and cross-checks them, so you can confirm the sign
convention on your MuJoCo version in one command.

### The wrench is mandatory, not optional

The full 6-axis wrench is recorded on **every** control step, and the pipeline is built so
that a missing one is an error rather than a row of zeros:

* `EndEffectorInterface.wrench()` is part of the contract, not an extra;
* `run_episode` calls `require_wrench(env)` before the first step;
* the controller runs with `require_full_state = True` and raises if `step()` is called
  without a wrench;
* `export_dataset` refuses to write an episode whose wrench is identically zero while
  sweeping in contact.

Four checks for one value, because the failure they prevent is silent: a dead sensing
path produces perfectly valid-looking episodes, and you find out after collecting the
demonstrations.

## Force signal quality

The measured tangential force is dominated by the tip sliding on the **table**, not by the
parts. With the defaults:

```
tip–table drag      = mu_tip * F_n      = 0.5 * 3.0 N   = 1.50 N
pushing an 8 g nut  = mu_part * m * g   = 0.6 * 0.008 * 9.81 = 0.047 N   (3 % of the total)
```

So a contact event contributes roughly **1.5–5 %** of `sqrt(fx² + fy²)` for a single part.
That is not fatal — the drag baseline is quasi-constant while part events are transient,
so a high-pass or a baseline-subtracting feature front-end recovers them — but it does
mean a classifier fed the raw total wrench has to find a small signal in a large one, and
you should know the ratio before trusting a result.

Three things follow, and all three are built in:

**1. The simulator reports the decomposition as ground truth.** `contact_breakdown()` walks
the active contacts, keeps those involving a gripper tip, and splits them into the
component share and the table share:

| Array | Meaning |
|---|---|
| `wrench` | what a real F/T sensor would measure (total) |
| `wrench_parts` | the component-contact share — **simulator only** |
| `wrench_table` | `wrench − wrench_parts` |
| `n_part_contacts` | how many component contacts were active |
| `tangential_force`, `tangential_force_parts` | in-plane magnitudes of the two |

None of this exists on hardware. It is there so a classifier trained on the *total* wrench
can be checked against what the parts actually contributed — together with the exact
`phase` label, that is a supervised validation set no real rig can produce. `dataset.json`
marks these arrays `simulator_ground_truth_arrays`: **never feed them to a policy**, or the
result will not transfer.

MuJoCo's sign convention for `mj_contactForce` has varied across versions, so it is not
assumed: it is calibrated once, on the first meaningful contact, by checking which sign
makes the per-contact sum agree with `cfrc_ext`. The decomposition is then exact
(`wrench_parts + wrench_table == wrench`), and the smoke test verifies it.

**2. Two metrics quantify it per episode.** `part_contact_ratio` (share of SWEEP samples
actually touching a part) and `part_force_snr` (`|F_parts,xy| / |F_total,xy|` while
touching). If `part_force_snr` is ~0.02, any force-based claim needs a front-end that
removes the drag baseline first.

**3. The drag baseline scales with `F_z*`.** Raising the normal force raises the denominator
proportionally, so a higher setpoint makes parts easier to push *and* harder to hear. That
trade-off is visible in `run_force_sweep` output and is worth reporting.

---

## Perception

Both backends return the same `SceneObservation`, so a planner cannot tell them apart.

**1. Ground truth (`--perception ground_truth`).** Uses the simulator's true component
poses. **Debug only** — it exists to validate the controller and planners without
perception error. Report numbers with the vision backend as well.

**2. Conventional vision (`--perception vision`).** One **fixed** high-oblique camera
(never moves; the same configuration is used for every episode and for dataset export):

1. render RGB (+ depth if needed),
2. build a binary component mask — MuJoCo segmentation ids (default), a brightness/colour
   threshold, or a height-above-the-table test on the depth image,
3. back-project every masked pixel onto the table plane (exact ray/plane intersection),
4. rasterise into a table-plane occupancy grid,
5. label connected regions into clusters with a centroid, an area and an **estimated
   component count** (`area / typical_component_area`).

No semantic classification of screws/bolts/nuts is attempted: the planner only needs to
know **where occupied regions are**. Optional position jitter, dropout and false
positives are available for stress testing.

The camera model (`sim/perception/camera.py`) is pure NumPy and round-trips
world → pixel → table plane to ~1e-16 m, so it is unit-tested without a GPU.

---

## Planners

Both produce the same high-level action, a planar sweep stroke:

```
[action_x_start, action_y_start, action_x_end, action_y_end, action_yaw]
```

**Planner A — predefined directional full-cover baseline (`--planner fixed`).**
Parallel lanes across the workspace, each swept right → left towards the tray; the tool is
lifted before travelling to the next lane (the `APPROACH` phase always lifts to
`workspace.z_travel`, so **the return motion is never in contact**); the sequence ends with
a fixed consolidation stroke in front of the tray. **It never looks at the component
positions** — the stroke list is a pure function of the configuration. Lanes whose `y` lies
outside the tray opening are aimed diagonally so the stroke ends inside the tray; this is
still layout-independent.

**Planner C — global single sweep (`--planner global_sweep`).**
The same perception and the same execution stack as Planner B, but it plans **one
global stroke per iteration**, aimed at the weighted centroid of *every* detection at
once, instead of short strokes aimed at one cluster. Then it lifts, re-observes and
repeats.

This is the planner that makes the comparison interpretable. A and B differ in two
things at once — perception *and* stroke segmentation — so a two-planner experiment
cannot say which one produced the gap. C changes only the segmentation:

```
C − A  =  what perception buys
B − C  =  what stroke segmentation buys
```

**Planner B — conventional visual greedy planner (`--planner visual_greedy`).**
`observe → generate candidates → score → execute one stroke → lift → re-observe → re-plan`,
until success or the budget runs out. Candidates are strokes centred on each detected
cluster plus a uniform sweep of lane offsets; each starts just to the +X side of the
right-most cluster it would capture and ends inside the tray mouth. Scoring:

```
score(a) = estimated_collected(a) − λ · path_length(a) − μ · risk(a)
```

with `risk` penalising (i) clusters captured near the outer edge of the pusher (they squirt
sideways), (ii) clusters that would merely be grazed and knocked off-lane, (iii) large
lateral funnelling demand, and (iv) captures that would end outside the tray opening.
Tunable through `planner.greedy.{lambda_path, mu_risk, …}`.

Trajectories are straight Cartesian segments with a smooth scalar time scaling —
trapezoidal (default) or quintic. No sampling-based motion planner is used for the planar
sweeping.

---

## Metrics

Written for every episode (`*_metrics.json`, and `episodes.csv` for batches):

| Metric | Meaning |
|---|---|
| `collection_rate` | components in the target / initial count |
| `success` | `collection_rate ≥ episode.success_collection_rate` |
| `completion_time` | simulated seconds |
| `n_strokes` | number of sweeping strokes |
| `path_length`, `path_length_xy` | total TCP path |
| `n_pushed_out` | components outside the safe workspace |
| `peak_normal_force` | max filtered normal force |
| `rms_force_error` | RMS of `F_desired − F_measured` during `SWEEP` |
| `contact_loss_ratio` | fraction of `SWEEP` samples below the release threshold |
| `components_per_stroke` | collected / strokes |
| `n_ride_over` | components swept over without moving (see Failure modes) |
| `n_jam_events` | sustained tangential-force events (see Failure modes) |
| `touchdown_overshoot_max/mean` | peak normal force above target during force build-up |
| `part_contact_ratio` | share of SWEEP samples actually touching a component |
| `part_force_snr` | component share of the tangential force while touching |
| `mean_tangential_force` | mean in-plane force magnitude while sweeping |

Plus diagnostics: `failure_reason`, `aborts`, `strokes_with_contact`, `mean_contact_force`,
`wall_time`, and the seed / planner / perception / geometry that produced them.

### Paired comparison

`sim.run_experiment` runs every `(n_components, seed)` pair through **every** planner, so
the planners see *identical* initial layouts. The low-level controller, target force,
velocity limits, workspace and episode budget come from one config and are therefore
identical by construction — `experiment_force_quality.png` is the check that they really
are (peak force, RMS tracking error and contact-loss ratio should be
planner-independent). `paired_*.csv` holds the per-seed differences and
`experiment_paired_*.png` the box plots.

---

## Failure modes

The official Menagerie UR10e is mounted at `[0.62, 0, 0.22]` in the default scene. This
is a task-layout choice: it keeps the tray-mouth lane reachable while the full official
link and collision geometry remains in the model. The brush stops at the tray mouth
(`target.x_max - max(0.02, brush_depth/2)`) rather than asking IK to reach the tray back
wall outside the command workspace. Once all components are inside the tray, the expert
rollout terminates immediately; it does not run extra recovery lanes that could create a
new collision after success.

Collection rate alone does not say *why* a run went badly. Three failure modes are
detected per stroke and aggregated per episode.

**Ride-over (`n_ride_over`) — too little force.** The tip passes over a component
without moving it. Detected geometrically: a component that lay inside the pusher's
capture band along the executed stroke and moved less than
`metrics.ride_over_move_threshold` (4 mm). This is the thin-washer mode, and it is the
lower bound on a usable `desired_force`.

**Jam (`n_jam_events`) — too much force.** A part wedges against the tip and the tool
loads up in plane. Detected as a **tangential** force above
`metrics.jam_force_threshold` sustained for `metrics.jam_min_duration`. Two notes on
why it is defined this way:

* it is a *tangential* test because the normal force is regulated by design and is
  therefore uninformative — a jam shows up in `sqrt(fx² + fy²)`, not in `fz`;
* it is a *force* test, not a stall test, because the prototype's position actuators
  are strong enough to keep tracking the command straight through a jam. On a real
  position-controlled arm the same threshold is what precedes a protective stop, so
  the metric transfers; in this simulator a velocity-based test would simply never
  fire. The threshold must sit above the baseline `μ · F_normal` drag of the tip on
  the table, which is why it is configured rather than derived.

**Touchdown overshoot (`touchdown_overshoot_*`) — controller transient.** Peak filtered
normal force above the target during FORCE_RAMP plus the first
`metrics.touchdown_window` seconds of SWEEP. This is the number the admittance gains
are actually judged on; it is where actuation delay and sensor noise show up first.

## Choosing F_z*

The normal-force setpoint is a fixed engineering parameter, not an action. `run_force_sweep`
is how it gets chosen: it sweeps `controller.desired_force` against component geometry
and reports both failure modes, so the usable range is bracketed from below (ride-over)
and above (jams and ejected parts).

```bash
python -m sim.run_force_sweep --forces 1 2 3 5 8 12 \
    --geometries washer hex_nut bolt --episodes 5 --spawn-mode cluster
```

It writes `episodes.csv`, `summary.csv` and `force_sweep_summary.png` (six panels:
collection rate, ride-over, jams, touchdown overshoot, ejected parts, peak force — one
line per geometry), and prints a suggested `F_z*`: the force with the best worst-case
collection rate across geometries. **Treat that suggestion as a starting point for
hardware, not as a result.** The sim's contact parameters are not calibrated against
real fasteners (see Limitations).

## Task distributions

`components.spawn_mode` selects what an episode looks like:

| Mode | What it is | What it is for |
|---|---|---|
| `uniform` *(default)* | parts scattered independently across the spawn box | the **planner** comparison — this is what makes full-cover expensive and rewards re-planning |
| `cluster` | all parts drawn around one randomly placed centre with randomised spread (`components.cluster.*`) | the **demonstration** task — keeps tool-object and object-object contact while removing the combinatorial initial-state space that makes a ~50-demo imitation dataset hopeless |

Both are reproducible from `(config, seed)` and both enforce the same guarantees (nothing
starts inside the target, minimum separation, inside the spawn box).

---

## Plots

Per episode: `*_force.png` (desired vs measured normal force with phase shading and
contact/release events), `*_z_correction.png` (Δz and commanded vs measured height),
`*_xy.png` (TCP trajectory, in-contact segments highlighted, component start/end poses,
table and tray), `*_events.png` (phase and contact timeline).
With `--save-observations`, `observations/obs_*.png` shows the camera frame, the image-space
mask and the derived table-plane occupancy grid side by side.

Per experiment: `experiment_summary.png` (collection rate, components/stroke, strokes,
time, path length, components pushed out), `experiment_force_quality.png`,
`experiment_paired_*.png`.

---

## Video

```bash
# default: the perception camera + a top-down overhead view, side by side
python -m sim.record_video --planner visual_greedy --num-components 5 --seed 0

# three panes, including one that tracks the tool
python -m sim.record_video --planner fixed --num-components 10 --seed 1 \
    --cameras scene_cam side_cam follow_cam

# the same layout under every planner -- one file each, directly comparable
python -m sim.record_video --compare fixed global_sweep visual_greedy \
    --num-components 5 --seed 0
```

| Camera | View |
|---|---|
| `scene_cam` | the fixed perception camera — **what the planner actually sees** |
| `overhead_cam` | top-down; best for reading stroke geometry and the tray |
| `side_cam` | low side view; best for seeing tip–table contact and parts tipping |
| `follow_cam` | fixed position, rotates to keep the tool centred |
| `inspection_cam` | oblique front-above human inspection view; never an ACT input |

Every frame carries a HUD: simulated time, stroke index, phase, planner, collected count,
and a normal-force gauge showing measured fill against a tick at the setpoint, plus a
contact dot. The gauge is scaled to 2.5 × `desired_force` rather than to
`safe_max_force` — on a 25 N axis a 3 N reading fills 12 % of the bar and shows nothing.

Output is MP4 (`libx264` via `imageio-ffmpeg`) with an automatic GIF fallback; frames are
streamed to the writer, so a 200 s episode costs megabytes of RAM, not gigabytes. One
frame every `video.every_n_control_steps` control steps — the default 4 gives 25 fps at
real-time speed.

`overhead_cam`, `side_cam` and `follow_cam` are defined under `video.extra_cameras` and are
**recording only**. Perception looks up `scene_cam` by name and nothing else, so adding
view angles cannot leak information into a planner or into an exported dataset; a unit test
asserts the names never appear in the perception path.

### 2-D preview (no OpenGL)

`sim.preview` draws the same episode as a Matplotlib animation — top view plus a rolling
force trace — straight from the episode record. No renderer, no GL context, so it works
over SSH, in CI, and on a machine where MuJoCo rendering is not set up.

```bash
# one episode, 3x playback
python -m sim.record_video --preview --planner visual_greedy --speed 3

# all three planners on the SAME layout, one shared clock
python -m sim.record_video --preview --speed 8 \
    --compare fixed global_sweep visual_greedy --num-components 5 --seed 2
```

The comparison figure is the experimental claim as a picture: identical initial scene,
identical low-level controller, different stroke planner; each pane freezes when its
episode ends and prints its stroke count, time, path length and collection rate, so the
difference is visible directly rather than read off a bar chart.

It is a plot, not a camera — flat shapes seen from above. Use `sim.record_video` without
`--preview` when you want the rendered scene. Every frame carries a provenance banner
naming the data source, so a file that travels on its own cannot be mistaken for rendered
physics.

## Motion planning: where RRT belongs

`planner.transfer.mode` chooses how the **contact-free** legs are planned — lifting off,
travelling to the next stroke's start, and moving to the observation pose:

| Mode | Behaviour |
|---|---|
| `direct` *(default)* | lift → straight line at travel height → descend |
| `rrt` | lift → RRT-Connect path (shortcut, then time-scaled) → descend |

**The sweeping stroke is never planned by RRT, in either mode.** That is deliberate:

1. **A sweep is supposed to collide.** Its whole purpose is contact with the components.
   A sampling-based planner is a collision-*avoidance* algorithm; asking it to plan a path
   whose goal is contact is a category error, and an obstacle model permissive enough to
   let it through would let it through everything else too.
2. **Randomised geometry fights the force loop.** RRT paths are jagged; sparse
   macro-waypoints fed to a hybrid force/position controller inject acceleration
   transients exactly where the normal force is being regulated.
3. **Reproducibility.** The protocol is paired seeds on identical layouts. A stochastic
   planner inside the measured motion adds variance unrelated to the question.
4. **The action space.** A stroke exports as 5 numbers (or 16 waypoints) only because it
   is a simple geometric primitive.

The transfer phase has none of those constraints — it is contact-free by construction, the
force target is zero throughout, and it does not enter the action — so a planner that can
route around the tray walls, around parts already detected on the table, and later around
fixtures and a real arm's self-collisions belongs exactly there. Stop-and-go at waypoints
costs time but no force transient, because the force loop is not running.

**Honest note on when it earns its place.** At the default `workspace.z_travel` of 60 mm
the transfer already clears 12 mm parts and the 40 mm tray walls, so `rrt` will return the
straight line every time and change nothing. It starts to matter when the travel height is
reduced, when the bin or fixtures are taller, or — the real case — when a UR10e replaces
the Cartesian stage and joint limits and self-collision enter the picture. A unit test
pins both behaviours.

Implementation notes: the prototype's task space *is* its configuration space, so planning
happens in `(x, y, z)`; the tool is reduced to its TCP point by inflating every obstacle
horizontally by the tool circumradius (yaw-independent, hence conservative) and downwards
by the tool height (the TCP is the *bottom* of the tips). Obstacles come from the
*perception* observation, not from ground truth, so the transfer planner knows exactly what
the stroke planner knows. Planning is seeded from the episode RNG, the straight line is
taken without sampling when it is already free, and a failure falls back to the direct path
rather than deadlocking the episode. For a UR10e, swap `CollisionModel` for a joint-space
one; `rrt_connect` and `shortcut_path` are dimension-agnostic.

## Dataset export for future ACT training

`python -m sim.export_dataset` records what an imitation-learning policy would need.
**No training happens here.** Per exported episode:

```
<out_dir>/
  dataset.json        global metadata, action spec, full config
  index.jsonl         one line per episode
  episode_000000/
    meta.json         seed, planner, geometry, per-component mass/friction/start pose,
                      camera config, control rate, success label
    episode.npz       dense per-control-step arrays + high-level actions
    obs_000_rgb.png   fixed-camera frame at each re-planning step
    obs_000_mask.png  component occupancy mask
```

### What is in `episode.npz`

Per control step (100 Hz):

| Array | Shape | Notes |
|---|---|---|
| `t`, `phase`, `stroke_index`, `in_contact` | (T,) | `stroke_index = −1` during transfers |
| `tcp_position` | (T, 3) | **absolute — debug only** |
| `tcp_pose_relative` | (T, 3) | `[dx, dy, dψ]` w.r.t. the stroke's capture pose; **no absolute z** |
| `tcp_velocity`, `tcp_yaw` | (T, 3), (T,) | |
| `wrench` | (T, 6) | full external wrench on the tool, world frame `[fx, fy, fz, tx, ty, tz]` |
| `tangential_force` | (T,) | `sqrt(fx² + fy²)` — the contact-event channel |
| `wrench_parts`, `wrench_table` | (T, 6) | **simulator ground truth** — the component / table split of `wrench` |
| `tangential_force_parts`, `n_part_contacts` | (T,), (T,) | **simulator ground truth** |
| `force_desired`, `force_raw`, `force_filtered` | (T,) | |
| `command`, `delta_z`, `z_nominal` | (T, 4), (T,), (T,) | controller internals, debug only |

Per stroke (N strokes):

| Array | Shape | Notes |
|---|---|---|
| `actions` | (N, 5) | `[x_start, y_start, x_end, y_end, yaw]`, absolute table frame |
| `actions_waypoints` | (N, 16, 3) | `[dx, dy, dψ]` w.r.t. the capture pose, resampled by arc length from the **executed** in-contact path |
| `capture_poses` | (N, 3) | the TCP pose each observation was taken from |
| `stroke_ride_over`, `stroke_jam_events`, `stroke_touchdown_overshoot` | (N,) | failure modes per stroke |
| `stroke_t_start/end`, `stroke_collected_before/after`, `stroke_observation_index` | (N,) | |

Per observation (one per stroke):

`obs_XXX_mask` (image-space), `obs_XXX_occupancy` (table-plane grid),
`obs_XXX_capture_pose`, `obs_XXX_wrench_history` (the last
`dataset.wrench_history_steps` control steps of the 6-axis wrench), plus
`obs_XXX_points`/`counts` (debug). `goal_region` is a fixed binary channel on the same
grid as the occupancy map.

`dataset.json` lists which arrays are policy-facing and which are debug-only, so the
distinction survives outside this README.

### Two design rules baked into the export

> **The Z force is not an action.** The policy outputs the planar stroke only; normal-force
> regulation stays inside the hybrid controller. This keeps the learned policy away from
> the safety-critical loop and makes the action space low-dimensional and stable.

> **No absolute z, no absolute pose in the observation.** Absolute z encodes the table
> height — precisely the nuisance variable the force loop exists to absorb — and feeding
> it back invites overfitting to one table height. Poses are stored relative to a fixed
> capture pose per stroke.

### Two action parameterisations, both exported

`actions` (5-D stroke) and `actions_waypoints` (16 sparse waypoints) are written for every
episode, so the choice can be made at training time instead of at data-collection time.
They carry identical information for a straight stroke; the waypoint form additionally
represents curved strokes and matches the sparse-waypoint + interpolation structure that
avoids feeding sparse macro-waypoints straight to a force controller.

---

## Project structure

```
sim/
  config.py                  YAML config with dotted-path overrides
  metrics.py                 episode metrics, aggregation, paired comparison
  plotting.py                episode and experiment figures
  logging_utils.py           run directories, JSON/CSV/NPZ serialisation
  smoke_test.py              one-command install health check
  model/
    geometries.py            component geometry library (incl. inline hex-prism mesh)
    scene_builder.py         programmatic MJCF generation (no MuJoCo import)
  controllers/
    admittance.py            1-D admittance loop with saturations and anti-windup
    filters.py               low-pass, slew-rate limiter, delay buffer
    state_machine.py         episode/stroke phase FSM
    hybrid.py                hybrid force/position controller
  perception/
    base.py                  SceneObservation, occupancy grid, backend registry
    camera.py                pinhole model, pixel <-> table-plane (pure NumPy)
    ground_truth.py          debug backend
    vision.py                conventional segmentation + clustering backend
  planners/
    base.py                  SweepStroke (the 5-D action), Planner interface
    trajectory.py            quintic / trapezoidal Cartesian trajectories
    fixed_cover.py           Planner A (fixed full cover)
    visual_greedy.py         Planner B (visual greedy, segmented strokes)
    global_sweep.py          Planner C (one global sweep per iteration)
    rrt.py                   RRT-Connect for the contact-free transfer legs
    geometry_utils.py        planar helpers
  environments/
    ee_interface.py          end-effector abstraction (+ UR10e adapter)
    layout.py                reproducible layout sampling
    sweep_env.py             MuJoCo environment
    episode.py               episode runner
  video.py                   multi-camera recorder with a controller HUD
  preview.py                 2-D Matplotlib animation (no OpenGL required)
  run_episode.py  run_experiment.py  run_force_sweep.py  record_video.py
  visualize_results.py  export_dataset.py
configs/default.yaml         every controller and experiment parameter
tests/                       unit tests (+ MuJoCo smoke tests, auto-skipped)
```

---

## UR10 model boundary

Everything above `EndEffectorInterface` — planners, trajectory generation, the admittance
loop, the state machine, metrics and the dataset exporter — is written against a
**task-space** interface: an absolute TCP pose command, the measured TCP pose/velocity,
and the contact wrench at the fingertip frame. Nothing in them knows about slide joints.

The current default `end_effector.type: ur10e` uses the vendored MuJoCo Menagerie
UR10e six-joint model with official visual meshes and collision geometry. It
is suitable for validating the ACT observation/action plumbing and the hybrid
force-control loop, but it is not a calibrated hardware model.
The model is mounted just beyond the table's +X edge and starts from an above-table IK
seed so the human inspection view does not show a link passing through the work surface.
The task-specific brush, TCP, wrist F/T sites and wrist camera are attached at the model's
`attachment_site`. This is still an ACT-feasibility model rather than a calibrated hardware
digital twin: the Menagerie README notes that actuator values are not carefully tuned, and
the brush is a rigid task fixture with no gripper DOF.

The current adapter solves damped-least-squares IK on the official tool frame, then sends
bounded joint-position targets to the Menagerie actuators. The force-control framework
above it continues to use the same absolute-TCP-pose and normal-force contract. The
vendored model's `README.md` and `LICENSE` are kept beside the assets for provenance.

The original `cartesian3dof` scene remains available as a diagnostic baseline.
`end_effector.type` selects the scene and adapter; the planner, force controller,
state machine, metrics, and dataset exporter keep the same task-space contract.

---

## Assumptions and known simulation limitations

Read this before quoting any number from this simulator.

**Robot model**

1. The active end-effector is the six-joint UR10e Menagerie model with visual meshes,
   inertial links, collision geometry, joint limits and position actuators. It is not a
   calibrated hardware digital twin: actuator gains, contact parameters and the rigid
   brush fixture remain feasibility-level settings, and `controller.control_delay_steps`
   is still a crude stand-in for hardware latency.
2. The gripper is **permanently closed** and rigid: no finger joint, no finger compliance,
   no pad deformation. There is no grasping degree of freedom anywhere in the model, by
   construction (a unit test enforces this).
3. The TCP is defined as the **bottom centre point between the two closed tips**, which is
   also the origin of the body carrying the F/T site.

**Contact and components**

4. Components are **simple convex primitives** — a hexagonal prism mesh, cylinders, and
   two-geom screw/bolt approximations — not CAD meshes of real fasteners. Threads,
   chamfers, internal holes and the real mass distribution are absent. **This validates the
   control, planning and data pipeline, not the detailed contact dynamics of real screws.**
5. Friction is Coulomb with MuJoCo's elliptic cone; sliding, torsional and rolling
   coefficients are constant per component, isotropic, and independent of speed, wear and
   contamination. Real fasteners on a real bench roll, jam and stick far more irregularly.
6. The table is a perfectly flat, rigid plane. `table.height_perturb_std` shifts the whole
   surface; there is no waviness, tilt or local compliance.
7. Component–component interaction is modelled (they collide and shove each other) but
   stacking and interlocking are not tuned, and small parts can be squeezed out sideways —
   `n_pushed_out` is the metric that exposes this.
8. The tray floor **is** the table surface, with three walls and an open +X side, so parts
   slide in without climbing a lip. There is no "bounce-out" model beyond plain contact.

**Perception**

9. The conventional-vision backend renders from a fixed, noise-free, perfectly calibrated
   camera. MuJoCo segmentation ids give a mask with no classification error; the `color`
   and `depth` modes are more realistic but still ideal. Camera noise must be injected
   deliberately (`perception.noise.*`).
10. Back-projection assumes components lie on a known plane at
    `table_top_z + perception.plane_offset`. For an oblique camera this biases the recovered
    position of tall parts by roughly `height · tan(view angle)`. The offset partially
    compensates it; it does not remove it.
11. MuJoCo's depth buffer is the camera-frame z-distance, not the Euclidean ray length.
    `perception.depth_convention` selects which one the depth-segmentation mode compares
    against (`optical_axis` by default, which matches MuJoCo).
12. Re-observation moves the tool to `workspace.observe_pose` so it does not occlude the
    scene; this cost is included in `completion_time`.

**Control and evaluation**

13. The normal-force measurement is the *total* external force on the tool body, so a
    horizontal shove from a component contributes its (small) vertical component. It is not
    a calibrated, drift-affected, temperature-dependent F/T sensor.
14. Force regulation has an inherent steady-state offset proportional to `k_d` (see
    "Controller tuning"). Do not read `desired_force` as the achieved force — read
    `mean_contact_force` and `rms_force_error`.
15. "Collected" means the component's **centre** lies inside the tray footprint. A part
    balanced on the tray lip counts as collected.
16. Episode budgets (`sim.max_episode_time`, `planner.max_strokes`) truncate long episodes;
    `failure_reason` says which limit was hit. Truncated episodes drag the mean collection
    rate down, which is intended — the baseline is supposed to look expensive.
17. **Jam detection is a force threshold, not a measured stall.** It fires on sustained
    tangential load, which is the right physical signal but needs a threshold above the
    tip's own friction drag. If `desired_force` is changed a lot, re-check
    `metrics.jam_force_threshold` — it does not scale itself.
18. **Ride-over detection is geometric.** A component inside the capture band that moved
    less than 4 mm counts as ridden over. A part that was nudged by a *neighbour* rather
    than by the tip can therefore escape the count, and a part the tip legitimately only
    grazed can be counted. It is a screening metric, not a contact-level diagnosis.
19. `actions_waypoints` is resampled from the executed path, so it inherits any tracking
    error the position loop had. That is deliberate (it is what the robot actually did),
    but it means the waypoints are not exactly the commanded stroke.
20. **The part-contact force is a small fraction of the measured wrench** (see Force signal
    quality). Any force-based classification result should be reported together with the
    `part_force_snr` it was obtained at.
21. The contact decomposition is exact *in the simulator*. On hardware only the total
    wrench exists, so a method that needs the split will not transfer — the split is for
    supervision and validation, not for deployment.
22. **Transfer RRT plans against a box model, not the real geometry.** The tray walls and
    detected parts are axis-aligned boxes inflated by the tool circumradius. That is
    conservative, so it never plans *through* something, but it will refuse clearances a
    finer model would allow.
23. Video is a rendering of the same simulation, not independent evidence. A run that looks
    right and a metric that says otherwise disagree because the metric is measuring
    something the camera cannot see — trust the metric and find out which.
24. Everything is deterministic given `(config, seed)` on one machine and MuJoCo version.
    Across MuJoCo versions or platforms, contact solver differences can change trajectories;
    the *config hash* is recorded with every experiment so runs can be compared honestly.

---

## Which machine to run this on

Nothing in this repository is platform-specific — pure Python, `os.path.join` throughout,
`multiprocessing` via an explicit `spawn` context (what macOS and Windows need anyway), and
`imageio-ffmpeg` ships its own ffmpeg binary. MuJoCo itself has prebuilt binaries for
Windows, Linux and macOS on both x86_64 and arm64. So it runs anywhere; the question is
only where it is least friction.

| Machine | Good for | Notes |
|---|---|---|
| **macOS (Apple Silicon)** | development, single episodes, video | `pip install mujoco` just works; offscreen rendering needs no `MUJOCO_GL` setting |
| **Linux** | batch runs (`run_experiment`, `export_dataset`) | more cores for `--jobs N`; headless rendering with `MUJOCO_GL=egl` |
| **Windows** | — | works, but never easier than the other two, and the downstream imitation-learning stack expects WSL |

The one macOS quirk worth knowing: MuJoCo's **interactive** viewer (`launch_passive`) must be
started with `mjpython` instead of `python`, because macOS requires rendering on the main
thread. It does not affect this repository — `sim.video` and the vision backend use the
*offscreen* `mujoco.Renderer`, which has no such restriction. Use `mjpython` only if you
want to open the scene and poke at it by hand.

## Troubleshooting

**Offscreen rendering fails / `Renderer` raises a GL error.** MuJoCo picks a GL backend
via `MUJOCO_GL`. On macOS the default (CGL) normally works; on a headless Linux box set
`MUJOCO_GL=egl`, or `MUJOCO_GL=osmesa` if there is no GPU:

```bash
MUJOCO_GL=egl python -m sim.smoke_test
```

Only the `vision` perception backend, `--save-observations` and `sim.record_video` need
rendering — the `ground_truth` backend and every controller/planner experiment run without
it.

**MP4 writing fails.** `imageio-ffmpeg` ships its own ffmpeg binary; if it is missing the
recorder falls back to GIF automatically and says so. `--gif` forces it.

**`multiprocessing` and rendering.** `--jobs > 1` uses spawned processes, each building
its own MuJoCo model. That is fine for `ground_truth`; with `--perception vision` each
worker also creates its own GL context, which some drivers dislike. If a parallel vision
run hangs, drop to `--jobs 1`.

**Episodes end with "stroke budget exhausted".** Planner A needs about `n_lanes + 1`
strokes; `planner.max_strokes` must be at least that or the baseline is truncated (which
silently flatters it — fewer strokes, lower collection rate). `n_planned_strokes` is
recorded in every episode's `extra` field.

**The force never reaches the target.** Expected: a non-zero `k_d` leaves a steady-state
offset (see "Controller tuning"). Read `mean_contact_force`, not `desired_force`.

## What this prototype is for

The first goal is **not** photorealism and **not** an accurate model of real screws. It is
to establish where a fixed full-cover sweep, a single global sweep and a segmented visual
planner start to differ — as a function of component count and spatial distribution — on
top of a hybrid force/position controller that is provably identical across all three. The
three-way split separates what perception buys from what stroke segmentation buys. That
difference, the force sweep that fixes `F_z*`, and the demonstrations exported from both,
are the foundation for the learned policy that comes next.

## License

MIT.
