# Hybrid PAC-ACT post-training

This is an isolated, independent implementation of the approved action-chunk
PPO design. Reference: PAC-ACT `fslee2/PAC-ACT@1a8c9db3c551ce776091bff6203746b439219c78`
and <https://arxiv.org/html/2607.09590v1>. Upstream source is not vendored because
the inspected reference did not declare a license. This is not the residual
SAC controller and does not reuse its replay pool or train its frozen ACT.

## Boundaries

- The 67 `sim/` sources, `hybrid_trial4_snapshot/`, existing evaluation
  entry points and historical experiments are unchanged.
- The legacy package path starts from five-view **ResNet ACT step_100000**. The
  separate DINO extension below uses only the local **DINOv3 ACT step_200000**
  checkpoint. The trainable
  Actor uses its zero-latent inference path without the CVAE encoder.
  Backbone, Transformer and action head receive gradients with dropout off.
  The Critic has a separate copied visual/Transformer encoder and
  `512 -> 256 -> 256 -> 1` value head. Neither model receives object truth.
- Inputs are exactly five RGB cameras, robot state36 and target count1.
  The frozen normalizers and frozen reference ACT are never optimized.
- A* approach, actual 1 mm contact gate, trial-4 wrist-force acquisition and
  native safety/termination ordering are reused. Z is always the existing
  1 N, ±2 mm wrist-admittance/follow controller; predicted ACT Z is ignored.
  Seeking/maintaining M/B/K remain 10/400/10 and 10/800/10.
- The frozen reference executes the first 50 ACT actions (2 s) with the
  native handoff ramp. Then the trainable Actor executes the first 5 actions
  of each 25-action prediction. Queries are synchronous every 0.2 simulated
  seconds, with no cross-chunk temporal ensemble. Physics pauses during
  inference and optimization; native 100 Hz interpolation remains continuous.
- A* prefix, acquisition and ACT share 2,000 actions / 80 simulated seconds.
  The native two-second first-entry window is continuous across stages.

## Probability and rewards

Each PPO action is a normalized `5 x 3` array `(dx, dy, dyaw)`. The initial
physical exploration std is 0.10 mm, 0.10 mm and 0.10 degree per step;
15 independent learned scales are bounded to 0.01–0.30 mm / degree.
Raw samples, integrated targets and executed references are kept separately.
Only the physically executed prefix contributes to log probability, entropy,
KL and the frozen-reference prior. Z and unexecuted suffixes contribute none.

Actual brush/part geometric contact must persist continuously for 20 ms.
Each part is scored once per episode: the first N give `+60/N`, additional
parts give `-120/N`. Simultaneous contacts share the remaining quota without
part-ID preference. Prefix contacts consume quota but are not credited to
actions that PPO has not executed. Strict-success/ordinary-failure terminal
reward is zero; safety termination is -60. Rewards are not normalized/clipped.

For each block: `optimization_reward = sum(environment_rewards)
- 2 * mean((actor_mean - frozen_ACT_mean)^2)` over the executed normalized
XY/yaw prefix. Environment reward, prior cost and optimization reward are
logged separately. Gamma=1, GAE lambda=.95. A zero-execution IK termination
seals the preceding real transition but never creates a fake action sample.

## PPO and commit protocol

- Six time-sliced environments, one goal each, 64 valid blocks per goal.
  A single Actor revision collects all 384 blocks; data is discarded after
  its on-policy update. Episodes may continue across rollout boundaries.
- Actor/noise LR=1e-6, Critic LR=1e-5; logical minibatch32, microbatch4/2/1;
  at most8 epochs, clip=.2, value coefficient=.5, entropy coefficient=.001,
  gradient norm=.5, differentiable adjacent-policy KL penalty=3.
- Finite whole-rollout KL is logged and penalized, without threshold-based
  early stopping or rejection. Nonfinite values or execution/update exceptions
  still restore Actor, Critic, optimizer, noise and all RNG state and stop.
  Resource/deadline checks also run between optimizer microbatches/KL passes.
- The user explicitly removed the earlier .015 early-stop and .02 hard-KL
  gates on 2026-10-16. The referenced upstream implementation adds
  an empirical KL penalty and supports optional `target_kl` early stopping;
  its inspected chunk-training entry does not pass `target_kl`. This project
  uses analytic `KL(old || new)`, summed over the executed XY/yaw prefix and
  averaged over blocks. Do not interpret KL as a physical safety measurement
  or transfer thresholds between different estimators/scales without checking.
- Used-action counters and accepted loss curves are published only after
  the complete update checkpoint is durable. Its audit is in the checkpoint
  too, so a crash between checkpoint and JSONL publication can be reconciled.
- Snapshots stream directly to bounded files, fsync, deserialize and verify
  size/hash before atomically publishing the index. Two rolling generations
  are maintained; initial, 50K milestones and final snapshots are retained.
  Frozen ACT weights are referenced by bundle hash, not duplicated per save.
- Resume uses the last valid complete update, never a half physical episode.
  Physical action/attempt journals prevent counter rollback or seed retries.
  Uncommitted rollout actions remain counted as sampled/discarded, not used.
  Completed validation arms are never re-executed; interrupted validation
  opportunities remain visible and do not silently become passed validation.

## Rejected-update evidence and offline diagnosis

New instrumented runs publish `update_audits/rejected_<attempt_id>.json` and
an append-only `rejected_updates.jsonl` pointer/hash index. These are not
checkpoints and cannot authorize resume. They retain attempted/completed Adam
calls, gradient norms, configured learning rates, the original exception,
available proposal distributions (including explicitly partial/nonfinite
passes), physical mean-shift summaries, and pre/proposed/restored component
hashes for model, Actor, Critic, noise, optimizer, RNG and update count.
`rollback_verified=true` requires equality of the pre/restored hashes; a
missing comparison is null, never an invented pass. Audit failures must not
bypass restoration or replace the originating exception. Later media/save
failures identify the last accepted update separately, not as a rejection.

```sh
.venv/bin/python -B scripts/diagnose_pac_update.py \
  --run /path/to/failed_mac_acceptance --model /path/to/bundle/base_model \
  --out /new/path/numeric_diagnosis
```

The diagnostic reads numeric learning-block logs and frozen normalization
only: no policy execution, environment steps, optimizer or historical writes.
It reports reward sparsity, physical noise units and independent float64
Gaussian KL sensitivity. Historical proposed weights/distributions cannot be
reconstructed if they were not saved. Logged-field GAE is labeled separately:
the collector may seal a previous transition after a zero-action terminal
without rewriting its earlier JSONL line, so those logs alone do not prove
the exact advantages used by the historical learner. New code does not
retroactively certify an older rejected update or its rollback.

## Environment and portable bundle

Use an independent **Python3.12** environment. Pinned main packages are
PyTorch2.11.0, torchvision0.26.0, LeRobot0.6.1, MuJoCo3.14.0, SB3 2.9.0.
`requirements-pac-act.lock.txt` records the minimal complete local runtime.
On a GPU machine install the official PyTorch2.11 CUDA build appropriate to
that machine first; install the remaining lock without replacing it. Local
version suffixes such as `+cu...` are accepted, but a different torch version
is not. `inspect` is read-only and reports mismatches; execution refuses them.

First build a new numeric-only exclusion snapshot. Required inputs are the
old120 plan, the ACT dataset manifest, the residual seed schedule and result
ledger. Add every other known demonstration/preflight manifest using repeated
`--additional`. Paths to images in those manifests are never followed.

```sh
.venv/bin/python -B scripts/prepare_pac_act_exclusions.py \
  --old-plan /path/to/old120/episode_plan.json \
  --dataset-manifest /path/to/ACT_dataset/manifest.jsonl \
  --residual-schedule /path/to/residual_run/seed_schedule.json \
  --residual-ledger /path/to/residual_run/episodes.jsonl \
  --out /new/path/exclusions.json
.venv/bin/python -B scripts/package_pac_act.py \
  --model /path/to/checkpoints/step_100000 \
  --exclusions /new/path/exclusions.json --out /new/path/pac_bundle
.venv/bin/python -B scripts/train_pac_act.py --mode inspect \
  --bundle /new/path/pac_bundle
```

The bundle carries base weights/stats, effective config/profile, numeric
exclusions, source identity and runnable `source/` including assets. Transfer
the entire bundle; on GPU run from its `source/`. No Mac absolute path or
historical batch directory is consulted. Preserve the manifest unchanged.
Registered streams are train6M, validation7M and preflight8M; initial-pose
fingerprints are checked before task execution. Old120 and the full residual5M
interval are excluded. A* uses one geometry candidate, no physical screening,
no seed replacement, no recovery route and no failure retry.

### DINOv3 ACT 200K base

Package the exact local `step_200000` directory with the same numeric exclusion
manifest. The DINO bundle recursively copies and hashes every checkpoint file,
including `training_state.pt` and the saved processor/statistics states, and
binds the pinned DINOv3 model id, revision and weight digest. It also carries
the explicit `dinov3_sync5` control profile and its paired evaluation config.
The ResNet `step_100000` manifest and package layout remain unchanged.

```sh
.venv/bin/python -B scripts/package_pac_act.py \
  --model /path/to/checkpoints/step_200000 \
  --exclusions /new/path/exclusions.json --out /new/path/dinov3_pac_bundle
.venv/bin/python -B scripts/train_pac_act.py --mode inspect \
  --bundle /new/path/dinov3_pac_bundle
```

The DINO loader dispatches only the serialized `act_dinov3` /
`dinov3_vits16` pair and loads weights and processors from the bundle with
local-only loading. It never downloads or substitutes a DINO encoder. Actor,
reference and Critic use byte-identical frozen DINO encoders; PPO optimizes
only the ACT trainable modules and exploration scales. Frozen encoders are
omitted from learner snapshots, which carry a DINO base-identity marker for
strict restore. The five-view RGB processor, saved normalization statistics,
25-to-5 chunk schedule, and Z-admittance ownership stay bound to the checkpoint
and sync5 bundle profile.

## Mac acceptance (not formal training)

```sh
.venv/bin/python -B scripts/train_pac_act.py --mode preflight \
  --bundle /new/path/pac_bundle --out /new/path/mac_acceptance --device cpu
```

Maximum2 h including model load, 24 new-layout opportunities across arms,
5 GiB output, 8 GiB disk reserve, process-tree RSS10 GiB / 30 s and sustained
dangerous-memory protection. Do not run while another local inference owns
the slot. Native sibling-worktree locks are held nonblockingly; optional
`PAC_ACT_INFERENCE_LOCKS` adds platform-path-separated lock paths for other
checkouts. No other user's process is stopped or waited out.

Acceptance compares target1/6 old native temporal ensemble, new deterministic
chunk schedule and new stochastic schedule on the same two new layouts.
It then performs **two real PPO updates of32 blocks** and deterministic
post-update checks. All opportunities, including native failures, are kept.
Videos initialize the same scene using native reset IK/settling, then replay
saved integration states, without rerunning ACT or trajectory-command IK.
Outputs include `acceptance.json`, per-episode `result.json`,
`physical_trace.jsonl.gz`, `blocks.jsonl`, MP4 and wrist-force/Z curves.
Engineering pass is not a claim that strict success has improved.

An explicitly authorized new acceptance can select a fresh six-aligned
preflight substream with `--preflight-start-index 120`. It cannot change the
training/validation streams or give an existing opportunity another attempt.
This fresh acceptance and its queue enforce CPU execution before resource
claims/output creation; no accelerator learner is authorized in this workflow.
Known prior layout seeds/fingerprints must be in the new bundle exclusions;
all earlier run directories and bundles stay unchanged. The new 2 h budget
begins at actual preflight entry (including model loading), not while queued.

```sh
python -B /new/bundle/source/scripts/queue_pac_preflight.py \
  --bundle /new/bundle --after-batch /running/paired120 \
  --out /new/acceptance --queue-state /new/queue \
  --preflight-start-index 120 --device cpu
```

This one-shot queue requires all120 paired formal results,20/goal, one-shot
markers, matching status/summary and retained verified-video hashes, followed
by exit of the dependency's execution/encoding processes and a free local
inference lease. It never stops or changes that batch. Set
`PAC_ACT_INFERENCE_LOCKS` to the existing native/worktree lease paths when
running from a portable bundle. Dependency errors, drift, child failure or
an expired 24 h queue deadline stop the queue without recovery/retry. No path
in this queue invokes formal training. A passed preflight ends and reports.
Dependency evidence is revalidated after waiting for a lease and immediately
before launch. Queue/preflight outputs must be disjoint from each other and
from both sealed input/dependency trees, including resolved symlink aliases.

Exploration remains independently learned per executed position/component:
initial XY0.10mm/yaw0.10degree, bounded0.01–0.30mm/degree. These are exploration
standard deviations, not limits on how far Actor means can learn to change.

If a presentation-layer error stops acceptance before any PPO update or
stochastic comparison, continuation requires explicit human authorization.
Keep the original bundle unchanged and package the authorized repair as a new
bundle. `--continue-authorized-from /path/to/original_bundle` reuses the same
output directory and original deadline/resource/opportunity budgets. It
archives the failed report, verifies completed results and recovered media,
and claims a single-use continuation owner. Only unattempted comparison arms
execute. Orphan attempts, changed control/model inputs, stochastic history,
previous updates, or an already-consumed authorization are refused. This is
not generic learner resume and never re-executes a finished physical trial.

## Explicit GPU execution (separate authorization required)

```sh
python -B scripts/train_pac_act.py --mode train \
  --bundle ../ --out /new/path/gpu_run \
  --acceptance-report /path/to/mac_acceptance/acceptance.json \
  --acknowledge-gpu-training
```

CUDA is required; there is no CPU/MPS formal-training fallback. An actual
six-environment rendering/backward memory/throughput test runs before the
baseline validation. Microbatch may fall from4 to2 to1; no environment,
logical batch or trainable module is dropped. Failure to fit stops the run.

Evaluate the fixed24 layouts (four per goal) with the new-schedule frozen100K
baseline, at every50K effective-action threshold and at the final accepted
checkpoint. Milestones record both threshold and actual commit step; no
best-validation checkpoint is selected. Maximum1M actual effective actions
or72 h from the original start, whichever occurs first; stop new collection
at66 h, 50 GiB output, 20 GiB disk reserve, 10 GiB process-tree RSS sustained
30 s / dangerous pressure. GPU OOM stops; no automatic tuning/retry/resume.
`--mode resume` requires an explicit new launch, keeps the original deadline,
and retires interrupted opportunities without physical continuation.

```sh
python -B scripts/monitor_pac_act.py --out /path/to/gpu_run
tensorboard --logdir /path/to/gpu_run/tensorboard --port 6007
```

The read-only dashboard defaults to127.0.0.1:8897, independent of SAC8896.
It shows complete episode pagination, collected/used/discarded actions and
blocks, accepted updates, raw/EMA loss/KL/prior/noise/reward curves, training
success/count error, validation heatmaps and checkpoint records.
`--mode evaluate`/`infer` loads the last complete Actor plus frozen warmup ACT,
without building the Critic. Supply `--run` for verified checkpoint identity;
single-layout inference also needs `--seed` and `--target`.

The future old120 baseline-versus-final comparison is a separate240-episode
experiment. It is neither run nor included in the Mac/GPU budgets here.
