# Diffusion Policy MVP — approved implementation plan

User-approved 2026-10-21. Baseline 8c19ef18dd1624691b42972183f059bf037c6864.
Branch codex/diffusion-policy-mvp; separate worktree. No changes to other
worktrees, ACT checkpoints, perception, original demonstrations or models.

## Contract

Official Stanford DP at 5ba07ac6661db573af695b419a7947ecb704690f, MIT.
Independent randomly initialized GroupNorm ResNet18 for each of five 320x320
RGB views; ImageNet normalization, no crop/resize. State36 and target count1
unchanged; two causal frames at i-1,i (40ms apart).
ConditionalUnet1D [256,512,1024], timestep128, kernel5, groups8,
cond_predict_scale=True. 100 cosine diffusion training steps, epsilon MSE,
DDIM8 eta0 inference using EMA (power .75).
Internal horizon28: historical action at0, output [1:26] is25x4;
last two slots always masked, nonexistent history/end and action_valid=false
masked. No episode crossing or future observations.

Read-only reuse all300 successful schema5 expert trajectories,203121frames,
50 per N1..6. No validation split or best-validation model. Only valid train
samples fit state/action normalization; no object/supervisor truth in inputs.
Approach/descent sampling mass .35 unchanged.

Workbench optional policy_type=act|diffusion, default act, independent DP
outputs/config/checkpoints, train/stop/resume/infer/replay/five-view video.
Reject wrong checkpoint types. DP captures25Hz, queries5Hz, executes5 steps
then requeries, pauses simulated time during inference, no ACT averaging or
200ms offset. Shared physical execution, integration, Z admittance, safety
and termination unchanged. DP errors terminate explicitly; two-frame cache.

Independent Python3.12 environment, torch2.11.0 torchvision0.26.0
einops0.8.2 diffusers0.35.1. MPS FP32 seed42 batch1 accumulation2 workers0;
AdamW lr1e-4 betas(.95,.999) wd1e-6 clip1, warmup500 then cosine.
Max100000 updates or cumulative24h training (pilot included).
Recoverable checkpoint every500 updates, retain2 plus final latest EMA:
model/optimizer/LR/EMA/RNG/sampler cursor/time/config/data identity.
New artifacts <=20GiB, free>=25GiB; stop and recover, never delete old data.

Pilot: fixed12 actual train windows, <=500 updates /30min, fixed-noise loss
decrease>=30%, encoder+UNet actually update, no NaN/OOM. Failure stops with
no automatic architecture/resolution/history shrink. Pass => fresh seed42
initialization for full300 training. End export latest completed update EMA,
not checkpoint selection by inference.

Acceptance after training: seeds81000001..81000006 for N1..6, each once,
check layout against training fingerprints first, max20min wall/episode.
Task timeout unchanged. Duplicate layout or infrastructure failure stops;
no seed replacement or rerunning task failures. Record results, latency,
replay/video. Six runs do not establish generalization or superiority.

### Task 1: Environment, vendor and causal data contract

Produces: independent locked environment, minimal attributed official modules,
DiffusionDataset two-frame windows, valid-only normalization and data audit.
RED: windows/alignment/padding/masks/causality/normalization tests.
GREEN: minimal implementation, run tests and relevant ACT baseline.

### Task 2: Policy, recoverable trainer and resource gates

Consumes Task1; produces full official architecture, masked epsilon loss,
DDIM+EMA inference, deterministic sampler/checkpoint restoration, resource
and cumulative time limits, pilot/full training orchestration.
RED: masking, output alignment, checkpoint resume and stop/budget tests.
GREEN: CPU contracts then real MPS forward/backward with exact architecture.

### Task 3: Shared execution and workbench

Consumes Task2; optional policy_type defaultACT. Produces bounded two-frame
25Hz cache, synchronous paused DP inference and five-step executor plugged
into original physical loop, explicit errors and wall limit, UI/API
train/infer/resume selection; replay/video path shared.
RED: job commands, defaultACT, cache/timestamps, rolling execution, errors.
GREEN: related ACT regressions and full project test command.

### Task 4: Review, real pilot and conditional full training

One fresh whole-branch review and tested fixes. Run exact MPS pilot and
persist evidence. Only on pass start fresh seed42 full300 training; count
pilot against24h. Stop on MPS/OOM/pilot failure, do not shrink model.

### Task 5: Six native acceptance runs and evidence

Only after actual full training, latest EMA. Preflight distinct layouts.
Each registered seed once, export five-view media and honest results.
Deliver branch, lock/config, checkpoints, curves, actual update count and
latencies. If an earlier mandatory gate fails report remaining not_run.
