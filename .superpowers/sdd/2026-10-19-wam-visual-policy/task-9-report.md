# Task9 — predictive, proposal and terminal-quality neural modules/losses

Status: IMPLEMENTED_WITH_CONCERNS. The27-test focused CPU/MPS run and the one
complete current-WAM regression (400passed) are GREEN, self-review defects are
fixed, source/cache/account checks pass, and the eight implementation/test files
are committed. Independent task review remains controller-owned and pending;
this is not a controller approval or a trained-policy result.

Implementation commit: `1090c5df9e14664dda72d2c5335e61e768a5fcf2`
(`feat(wam): implement predictive proposal and terminal quality heads`). This
report is committed separately after that scoped implementation commit.

## Scope and reviewed authority

Worktree: `/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim`.
Branch: `codex/wam-visual-policy-v1`. Reviewed dispatch BASE:
`630aa1c1489e1d65ed8c9198f3ba1ce84b4f5547`; initial worktree/index were clean.
Read `task-9-brief.md`, its required `frame-interface-integration.md`, and the
Task8 consumer contracts/shared-context section. Tasks1–8 were not restarted.

The implementation owns exactly these six production and two test files:

- `wam_sweep/models/__init__.py`
- `wam_sweep/models/graph.py`
- `wam_sweep/models/dynamics.py`
- `wam_sweep/models/proposal.py`
- `wam_sweep/models/quality.py`
- `wam_sweep/models/losses.py`
- `tests/wam/test_models.py`
- `tests/wam/test_model_losses.py`

This requested report is the only additional tracked deliverable. Runtime
logs/accounts remain in the already-owned, ignored `.venv`. No helper script,
subagent/reviewer, dependency setup/download, producer change, actual quota
collection, fitting/training, throughput profile, preflight, formal run, push,
PR or publication was performed. Independent task review belongs to controller.

## Actual model APIs and safe batching for Task10

Exports from `wam_sweep.models`:

```python
HistoryEncoder()
HistoryEncoder.forward(history) -> EncodedHistory
GraphGRU(encoder=None)
GraphGRU.rollout(history, reference, candidate, *, codec_state) -> WorldPrediction
ProposalHead(encoder)
ProposalHead.propose(history, goal, reference) -> Tensor[B,4,5,3]
TaskQualityHead(dynamics, *, calibration_hash=None, allow_uncalibrated=False)
TaskQualityHead.evaluate_task(history, goal, candidate, prediction,
                              *, reference, codec_state) -> TaskScore
HistoryOnlyQualityHead(encoder, *, calibration_hash=None, allow_uncalibrated=False)
HistoryOnlyQualityHead.evaluate_task(history, goal, candidate,
                                     *, reference, codec_state) -> TaskScore
```

`history` is the original `HistoryBatch`, with all five FP32 tensors and the
original shapes/masks. Construction contracts are revalidated at consumption,
including finiteness, masked zeros and a common device. Stored rows already
have B=1: concatenate each of the five tensors along B; never default-stack an
extra dimension. No RootRecord/EpisodeSnapshot/GT/audit is a model argument.
`EncodedHistory` contains objects `[B,6,128]`, brush `[B,128]`, scene `[B,384]`,
and bool object/row validity derived only from the known history.

For B=1, retain the existing public `ReferencePreview`, `CandidateProgram`
and exact dictionary from `ActionCodec.snapshot()`. Also accepted are lists or
tuples of these types of length B. Genuine B>1 REQUIRES separate per-row
references and safe codec dictionaries; a single B=1 value never broadcasts.
Candidate additionally accepts normalized finite FP32 Torch `[B,5,3]` on the
history device (select one proposal mode first). References remain per-row
public types, not dictionaries or truth-bearing records. Goals are int64 Torch
`[B]`, values1..6, on the history device; only B=1 also accepts a Python int.
Different references, programs, codec memories and goals in two rows are
checked against independent B=1 forwards for world, proposal and both scorers.

Codec states have exactly protocol/helper_sha256/bounds/program_anchor/filtered/
current_raw/residual/frames. Each is validated by restoring a real ActionCodec
with its original workspace and then privately cloned. Extra privileged keys,
wrong protocol/helper identity, malformed numeric state and wrong batch lengths
are rejected. Caller state/history/reference are not advanced by hypotheticals.

Full quality owns a frozen GraphGRU. `prediction` must be that world's own
prediction for identical safe history/reference/program/codec and unchanged
parameters; alternatively pass `None` to compute it internally with no grad.
Changed prediction fields, in-place mutation, external truth-shaped futures or
changed conditioning/weights are rejected. A bounded32-entry ephemeral weak
provenance cache stores no learned feature or privileged data. Regular tensors
use identity/version counters; inference tensors use content hashes because
Torch inference mode intentionally has no version counters. Serialized/evicted
predictions must be recomputed, not loaded from physical target records.

Both scorers return the original `TaskScore` with `[B,1]` success logits and
`[B,7]` final-count logits. Without an externally supplied Task10 calibration
identity, evaluation raises unless the caller explicitly opts into
`allow_uncalibrated=True`; that path labels the score `uncalibrated-fixture`.
No Task9 constructor/test creates a trained/qualified WAM `ModelVersion`. Calibration
fitting, checkpoint dependency identities and qualification remain Task10.

## Computations, widths and ownership

History: shared object34→128 and brush/context24→128 inputs; three relational
message/GRU layers128 with pairwise relative XYZ and masked neighbors, followed
by a masked temporal GRUCell128 over11 causal observations. No slot/track/GT ID
embedding exists. Mean128/max128 over relation-aware object nodes plus brush128
produce scene384; changing a neighbor's geometry changes an untouched object's
encoding. Empty history is finite zero, not an invented current observation.
Models consume Task8-enriched15-wide context without deriving another clock.
Later checkpoint/version identities must bind
`visual-first-estimated-collected-v1`, threshold0.5, shared context source and
perception identity. Runtime first_entry/timestamp stay per-episode snapshot
state; no new categorical clock feature is introduced by these modules.

World: three message/GRU layers128 at every physical step, separate object/brush
GRUCells128 and learned continuous/distribution readouts. Exactly25 recurrent
40ms outcomes are computed. Positions/velocities/angular states recurse through
the network; scales are strictly positive via exponentiation of bounded log
scale, object yaw is a normalized double-angle rotation (pi period), brush yaw
a single-angle rotation (2pi period). Position/velocity and brush log-std are
finite and clamped[-7,3]. Public continuous object field is `object_mean`.
Known history/numerical validity alone determine forecast validity; true native
termination/padded-target masks never enter rollout. Nonfinite inputs/outputs
are rejected rather than repaired with truth. No goal argument enters physics.

Proposal: scene384 + own goal embedding16 + ACT-reference GRU64 + own mode
embedding16 =480. One shared MLP480→256→256→15, tanh, produces four5x3 programs.
The actor's reference encoder receives xyz, yaw sin/cos and validity. Invalid
preview placeholders are masked in the reference encoding. It uses no predicted
or true future states to generate actions.

Both quality heads: scene384 + their OWN goal embedding16 + known-control64
=464. Known-control input171 is knots15 + program-start current_raw/filter6 +
flattened25x6 reference. At adoption current_raw, not a stale program_anchor,
is the new immutable raw anchor. Both scorers receive this SAME safe information.
Full head alone adds WorldFuture128: predicted object/distribution and brush
features are encoded with a relation graph, pooled mean/max/brush384 at each
future step and passed through a GRU384→128. Final full input592 and no-future
input464 each use MLP→256→128, then separate1 and7 linear outputs. The
HistoryOnlyQualityHead has no GraphGRU member and never calls world rollout.

World/history are trainable during the physical-model stage. Constructing a
proposal head freezes its shared history encoder; constructing full quality
freezes the entire supplied world. Head `.train()` preserves frozen world/
history `.eval()`. Each proposal/quality goal embedding is independent. Actor
candidate tensors are detached in world conditioning AND quality known-control
encoding, so terminal failure gradients cannot update proposal weights through
the candidate. Quality future inputs and history are read-only; its own future
encoder/GRU/MLP learn from terminal supervision. Task10 should use separate
optimizers over trainable owned parameters and fit world before freezing it.
The same fitted/frozen proposal and candidate count must serve both evaluation
arms; no matched-arm collection/selection experiment is claimed here.

Actual parameter counts from the admitted test (shared parameters count once
within each module; do not sum module totals across shared ownership):

| Module | Total | Trainable after head construction |
|---|---:|---:|
| Shared history encoder |553344|0|
| Proposal including frozen history |760111|206767|
| Frozen GraphGRU including history |1118373|0|
| Full quality including frozen world |1983437|865064|
| History-only quality including history |736680|183336|

## Codec, recurrence and hold semantics

Every candidate row restores the real Task3 codec, calls begin_program ONCE,
and samples with that fixed anchor at j*.04 for j0..24. Predicted outcomes are
at (j+1)*.04. Decode uses its OWN predicted TCP xyz/atan2(yaw), normalized known
reference, dt.04 and the preview activity bit. No second mapping or legacy
residual_rl import exists. The codec's original pinned AST helpers, workspace
generation, XY0.8 ratio, yaw45deg, normalized exponential filter and inactive
reset are used unchanged. Nominal target Z is unchanged by the codec; actual
brush Z response is learned, not equated to target or native Z-control changes.
Protocol is `wam_relative_xy80_yaw45_pchip_tau200ms_hold_reset_v1`; pinned helper
SHA256 is `f185c594c0a539c16e100ef7bf3f6c88cd2393ee2276fdc5e857d425108a41d8`.

Only the codec conditioning branch is stop-gradient: candidate conversion,
predicted TCP conversion and returned NumPy command/filter features. This is
the specified external-decoder approximation. The25-step object/brush/hidden
state recurrence is NOT detached, including relation geometry. A final-step
loss reaches root visual/brush tensors and the first actual recurrent object
state; world loss does not reach candidate/proposal gradients.

An invalid preview target after root is a placeholder. Rollout does not feed
that row as an executed future hold command. Root's initial held reference is
known; after active response the model uses its own predicted brush pose as a
hold estimate and retains it during inactive frames, with inactive codec reset.
This is an explicit modeled hold estimate: the actual future native slew/Z
command is not available in the safe inputs and is NOT asserted equal to TCP
or manufactured from the placeholder. Actual brush response remains learned.
Poisoning later invalid placeholders does not change physical rollout. No
privileged producer extension was introduced to make this estimate exact.

The decoder test records all25 real decode calls, reconstructs the same program
on an independent execution codec with the recorded OWN predicted TCP, and
checks exact target equality, immutable current_raw anchor, no caller-state
pollution, Z preservation and sampling times. The independent hand fixture at
step0 uses raw(.6,-.3,.2), filter0 and nominal(.1,0,.2,0): filtered X is
0.10876154815321091, XY radius.08 and target is
(0.10869448873037765,-0.00433757736191349,.2,0.028473706722598323).

## Three losses and Task10 collation obligations

```python
prediction_loss(prediction, targets, brush, brush_valid) -> scalar Tensor
proposal_loss(programs, successful_sets, successful_mask) -> scalar Tensor
terminal_loss(score, success, final_count, count_valid) -> scalar Tensor
```

Physical targets: B=1 original `AlignedTargets`, or sequence of B original
AlignedTargets. No B=1 target broadcasts across a real batch. Original per-field
FP32/bool/int64 dtypes/shapes are enforced. Brush is FP32[B,25,9], validity
bool[B,25], with unbatched25x9/25 accepted only for B=1. All target validity
combines valid_steps, mapping_valid and relevant object/per-field mask;
rectangle is shape supervision, not a generic validity mask. Brush has its own
validity independent of object mapping. Unknowns/NaN padding are sanitized
before target arithmetic; they do not become negatives. Loss terms are Gaussian
NLL for XY/velocity and actual brush, smooth-L1 for scales/angular velocity,
double-angle cosine yaw, state/shape cross entropy and visibility BCE. Circle
yaw/angular supervision is masked; brush sin/cos chord likelihood has2pi period.
Each term is normalized over its valid elements. All-unknown loss is zero with
zero gradients. The omitted Gaussian constant has no trainable effect.

Proposal: normalized finite FP32 `[B,4,5,3]`; explicitly grouped successful
targets FP32 `[B,S,5,3]` and bool `[B,S]` mask. Independent per-root Hungarian
assignment matches distinct modes to distinct real successful programs, using
mean squared program distance; differentiation is through selected distances,
not discrete assignment. More than four successes selects four distinct pairs;
unmatched modes/targets receive no fabricated action supervision. No successes
returns graph-connected zero. Task10 MUST group by the SAME physical root,
current perception and required successful complete continuation after the
TeacherDataset phase/N20-root/10-parent readiness gates; Task9 does not invent
that producer/collator. Task10 MUST skip actor optimizer.step on all-fail
batches, since AdamW decay can move weights even with zero gradients.

Terminal: success FP32[B,1] known binary; final_count int64[B]; count_valid
bool[B]. B=1 still keeps these tensor batch dimensions. Convert dataset None
counts to an arbitrary int64 sentinel ONLY with count_valid=False; sentinel
is removed before CE and does not label count0. Known counts must be0..6.
Success BCE and masked count CE train independent outputs. TerminalDataset's
exact-continuation completed native outcome selection remains authoritative.

## RED/GREEN and verification evidence

All tests use the existing owned venv/offline TMP/XDG/MPL namespaces and
canonical `.venv/task-2-run.lock`/`.venv/task-2-budget.json`. No nested clock or
new ledger/root. Wrapper prefix for focused rows:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python .venv/task-7-run.py LABEL SELECTION
```

It runs the actual argv with `./.venv/bin/python -m pytest SELECTION -q
-o addopts= -o tmp_path_retention_count=1000
-o cache_dir=.venv/wam-pytest-cache`. Every log/account is retained as
`.venv/task-7-LABEL.log` / `-account.json`; a label containing green does not
override an actual failing exit/result. These are the existing wrapper's
filenames, not a new Task7 operation or a new helper script.

| LABEL | SELECTION | Actual output / exit |
|---|---|---|
|task9-models-red|tests/wam/test_models.py tests/wam/test_model_losses.py|21 failed in1.04s /1; expected assertions: Task9 modules/losses absent|
|task9-models-green1|same two files|21 passed in1.68s /0|
|task9-self-review-red|tests/wam/test_models.py -k 'quality_loss_cannot or inference_mode or prediction_provenance or safe_dataset_row'|2 failed,2 passed,11 deselected in1.02s /1|
|task9-self-review-green|both Task9 files|25 passed in1.85s /0|
|task9-models-mps-green|both Task9 files -rP|27 passed in8.46s /0; MPS executed, not skipped|
|task9-current-wam|tests/wam,1400s/3GiB wrapper below|400 passed in694.71s (0:11:34) /0|

The two self-review REDs were real integration defects: quality gradients
reached tensor-valued actor proposals; inference tensors lack a `_version`
counter. Fixes detach quality candidate input and use inference-aware provenance.
Their tests, actor-gradient isolation and real CPU/MPS forward/backward are green.

Permutation test uses the required exact assert on `object_mean` with
rtol1e-5/atol1e-6, and also verifies brush/distribution equivariance; proposal and
both scores are invariant. Learned25-step output responds to candidates and is
not reference copying. Tests cover positive scales, normalized periodic yaws,
finite distributions, first-to-last recurrent gradients, absent goal/GT physics
arguments, relation sensitivity, independent batches and masked history.
Actual SafeInputs-based terminal row poisoning changes GT/audit future controls
and labels: model logits stay unchanged, terminal loss changes. External/
mutated/stale world futures are rejected. No-future scoring has a poisoned
GraphGRU rollout that would raise if called; reference-only and filter-only
changes both affect its score. Successful left/right targets are matched
separately; mean programs lose, duplicate reuse and unmatched supervision fail.

Complete current-WAM command, using the EXISTING runner in memory and preserving
its admission/account/prefix verification (no new script):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python - task9-current-wam tests/wam <<'PY'
from pathlib import Path
source = Path('.venv/task-7-run.py').read_text()
source = source.replace('budget.reserve(256*2**20,seconds=240)',
                        'budget.reserve(3*2**30,seconds=1400)')
source = source.replace('timeout=240', 'timeout=1400')
exec(compile(source, '.venv/task-7-run.py', 'exec'))
PY
```

Forecast rationale: measured Task8 amended native-file cost553.36s, current
Task9 CPU/MPS focused cost8.46s, plus other WAM tests/process overhead and
substantial margin.1400s and3GiB are reservations, not elapsed cost or a volume
profile. Actual complete output was:

```text
........................................................................ [ 18%]
........................................................................ [ 36%]
........................................................................ [ 54%]
........................................................................ [ 72%]
........................................................................ [ 90%]
........................................                                 [100%]
400 passed in 694.71s (0:11:34)
```

The wrapper's wall time was696.2343763329845s and its actually charged activity
695.3924090862274s. It exited0 with no warning, active clock or reservation left.
This is WAM-only; the unbounded original-root baseline remains Task12 debt.
The original Task8 failed full attempt and its later45-pass amended-file proof
retain their distinct historical meanings and are not replaced by this run.

## Final source/account audit, commits and concerns

Final audit evidence:
`/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim/.venv/task-9-final-account.json`.
It was produced by a one-off inline source/cache/account audit under the
EXISTING canonical run lease,60s/1MiB linked reservation and activity
`task9_final_source_account_audit`; no helper script/agent/model/physics was
created or rerun by this audit. It verifies exact staged eight-file scope,
`git diff --cached --check`, no unstaged tracked changes, each code SHA256,
all prior Task8 source hashes and their exact BASE bytes, the complete test
account chain, inherited event prefixes, prior proof/cache and r8 identities.
The committed code matches those audited SHA256s; no production/test change
was made after the full GREEN run. The documentation-only follow-up does not
change the tested implementation. Git's normal automatic-identity notice was
retained: `luke <huazhi_luke@MacBook-Air-10.local>`; no identity configuration
or commit amendment was performed.

Canonical account is `.venv/task-2-budget.json`; original policy roots and
distinct `.venv/task-2-run.lock` / `.venv/task-5-native.lock` remain unchanged.
Dispatch was4818.071447153092 effective seconds /295events /pilot0 /
active=null /reservations={}. Final state is **5531.923959035874 effective
seconds /309events /pilot0 /active=null /reservations={}**, warnings=[].
Task9 added **713.852511882782 effective seconds** including BOTH failing
REDs, all successful focused CPU/MPS runs, the complete native WAM regression
and the final source/cache/account audit. No pilot/quota/training/profile run
or manual replacement charge was invented.

| Account/log label | Effective seconds before→after | End events |
|---|---|---:|
|task9-models-red|4818.071447153092→4819.720043201447|297|
|task9-models-green1|4819.720043201447→4821.961625118256|299|
|task9-self-review-red|4821.961625118256→4823.548316020966|301|
|task9-self-review-green|4823.548316020966→4825.971340913773|303|
|task9-models-mps-green|4825.971340913773→4835.047406930924|305|
|task9-current-wam|4835.047406930924→5530.439816017151|307|
|task9_final_source_account_audit|5530.439816017151→5531.923959035874|309|

Audit-poll observations: owned2361401176bytes /free79271727104bytes; these
are live observations, not a learned throughput/volume forecast or permission
to start any later quota. The account JSON itself adds a small owned file after
that poll. No budget admission warning occurred. No cache, raw evidence,
checkpoint, prior ledger event, owned root or resource was deleted/relocated.

Inherited217 prefixes167/179/191/197/209/217 match the prior audit hashes.
The exact277 prefix remains
`b8990a9540e797b574362a87e1769c7a2f459e90b51055c7c40b3569cfdf9e7b`.
The preserved295 prefix is
`16312e93f898ed7cf50036cdb8ac47fe5bda9f2506684aaeb4a7d18e0b411c69`.
Every test wrapper checks its entire inherited event prefix/policy against the
post-run ledger; the final audit verifies the six consecutive Task9 activity
pairs and exact before/after chain from the dispatch account. BOTH distinct
controller1s prior charges are still present exactly once. Retained migrated
root pytest cache hashes/inodes are unchanged and180224bytes are counted once;
original source/root `.pytest_cache` remains absent. Existing ResNet initializer
SHA256 remains `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
The sealed r8 manifest/model/control/perception/preprocessing/config hashes
all match the reviewed Task8 final account. No download or second encoder.

The old Task8 failed-full log SHA256/exit1 and the subsequent amended-file
complete-fix log SHA256 remain unchanged. Both native evidence JSONs at
`pytest-63/task8-portable0` equal their archived Task8 values. Current regression
produced separate fresh native proof at
`.venv/task1-tmp/pytest-of-huazhi_luke/pytest-64/task8-portable0/`; the final audit
retains its portable/branch evidence. Native branch/restore proofs concern the
existing bounded fixtures, not adoption of these untrained Task9 modules.

Concerns/limits and consumer obligations:

- These are implemented learned modules with untrained fixture weights. No
  calibrated score, trained WAM version, controller adoption, physical-policy
  success or task-success improvement is inferred from their tests.
- External NumPy codec conditioning is detached as required. Actual recurrent
  physical state remains differentiable; gradients through the feasibility map
  are intentionally unavailable. No alternate boundary mapping was introduced.
- Expired-reference holds use the explicit model estimate described above.
  Exact future native slew/Z command is not available in the whitelisted inputs;
  it is neither copied from the placeholder nor asserted equal to predicted TCP.
  Actual brush prediction is learned and requires Task10 fitting/validation.
- Task10 owns grouping real same-root successful sets, positive coverage gates,
  current-version terminal selection, calibration/dependency identities,
  world-before-head freezing and skipping unsupervised actor optimizer steps.
  The bounded future provenance cache is transient: recompute after eviction or
  checkpoint restore, using state_dict rather than serializing live weakrefs.
- The original-root baseline is still Task12 debt. The400-pass result is ONLY
  the complete current `tests/wam` suite, not the whole original repository.

No unresolved implementation test failure or out-of-scope producer change is
present. No missing producer context was required to implement the authorized
neural computations. Controller independent review is the next owner action.

## FIX ROUND1 — known root-held information parity

Status: scoped correction implemented and committed; controller independent
scoped re-review remains pending. Task9 is NOT declared complete or published,
and Task10 has NOT been dispatched. The original400-pass full-WAM result above
is historical evidence for PRE-FIX code, not a new amended-code full-suite
claim. This append supersedes the earlier root-reference masking description
only at the causally known root row; all prior report/evidence history remains.

Authority: `task-9-fix-round-1.md`, round1/5, continuing the SAME original
implementer and worktree/branch. FIX_BASE is
`26be92c53d9dbb603940dccf367349d8b5bbcaf7`. Read that exact fix brief first,
then the original Task9 brief and committed report. Dirac's review was already
closed; no helper/subagent/reviewer was started or contacted for this fix.

### Exact open finding (verbatim)

> **Important — Preserve the known root-held reference in both scoring arms.**
>
> [graph.py:48](/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim/wam_sweep/models/graph.py:48) zeros every reference pose whose validity bit is false. Both scoring heads consume that masked encoding through [quality.py:33](/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim/wam_sweep/models/quality.py:33).
>
> However, [dynamics.py:100](/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim/wam_sweep/models/dynamics.py:100) initializes the held command from `targets[0]` and uses it for inactive steps at line 108. For two entirely inactive previews differing only in their known root-held pose, the no-prediction head receives identical features, while the world receives different commands and can convey that information through its future predictions. This violates the matched-information requirement in [frame-interface-integration.md:13](/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim/.superpowers/sdd/2026-10-19-wam-visual-policy/frame-interface-integration.md:13).
>
> Preserve the causally known root-held pose in the shared reference encoding, retaining its inactive mode bit and masking later unknown placeholders. Add a focused regression covering an inactive root, reference sensitivity without world invocation, and continued immunity to later invalid-placeholder changes. Existing GREEN results do not answer this boundary case; no probe was run.

### Root cause, source fix and exact scope

The original shared `references()` multiplied pose features by ACT validity for
EVERY row. With an entirely inactive preview this erased the known root-held
xyz/yaw as well as the later unknown placeholders. The physical rollout still
received root `targets[0]`, producing the information mismatch identified by
review. The prior mixed-active→hold test did not cover this inactive-root case.

Production fix is ONLY `wam_sweep/models/graph.py::references`: clone the
validity tensor into `pose_known`, set ONLY `pose_known[:,0]=1`, and multiply
the five pose features by that availability mask. Concatenate the ORIGINAL
validity bit unchanged. Thus root `[xyz, sin(yaw), cos(yaw), active]` keeps its
five causally known pose values with active0; later inactive rows stay all-zero
features. Known active rows retain their existing behavior. Reference arrays,
codec memory, dynamics/hold/reset/governor and native execution are unmodified.

`tests/wam/test_models.py` adds a two-row fixture with identical history, goal,
candidate and codec but different root-held xyz/yaw, plus two tests parameterized
over real CPU and available MPS. Literal expected root rows are
`[.125,-.25,.375,1,0,0]` and `[-.375,.5,.625,-1,0,0]` (yaw±pi/2, atol1e-6).
All activity bits remain0 and later inactive features remain0. Real control
encoder pre-hooks verify BOTH scorers receive the same171-wide safe input and
the literal root rows at columns21..26. No head forward is mocked. The
no-prediction scorer must distinguish the roots while GraphGRU.rollout is
poisoned to raise if invoked; changing ONLY later placeholders leaves its
success/count logits unchanged. Fresh OWN world predictions, physical outputs
and full-head logits are likewise unchanged by those later placeholders.
Caller codec states remain untouched.

No fields, IDs, input/hidden widths or public signatures were added. Shared
reference width6, control171→64, full592/no-future464 and proposal480 remain
unchanged. Both scorers and the proposal consume the same corrected shared
encoding. No privileged future command/GT input or producer/control change.

Fix source/test commit: `16270351536fa993d2840f62977d65fb835d69bb`
(`fix(wam): preserve inactive root reference information`). It changes ONLY
`wam_sweep/models/graph.py` and `tests/wam/test_models.py`.
This documentation-only follow-up APPENDS to the SAME tracked
`.superpowers/sdd/2026-10-19-wam-visual-policy/task-9-report.md`.
No prior commit was amended. All paths are relative to the same absolute
worktree printed above. No other model/loss/test/producer/config/weight file
was changed, and no dependency setup/download, actual collection/quota,
training, profile, preflight, formal run or publication occurred.

### Actual RED before production fix, then complete covering GREEN

Both commands use the EXISTING admitted runner with240s/256MiB reservation,
canonical run lease/account, existing offline TMP/XDG/MPL namespaces, and
explicit `-o cache_dir=.venv/wam-pytest-cache`
`-o tmp_path_retention_count=1000` (plus `-o addopts=`). These are focused model
checks, for which that existing bound is sufficient. No new helper script,
ledger/root or nested active clock was created. RED was observed before the
production source edit; `git diff` then showed only the new test file changes.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python .venv/task-7-run.py task9-fix1-inactive-root-red \
  tests/wam/test_models.py -k inactive_root -rP
```

Actual RED: **4 failed,17 deselected in1.44s**, exit1. CPU and MPS BOTH executed
and failed independently: the expected root features were zeroed (8/12 elements
mismatched, maximum absolute difference1), and the no-prediction root logits
had exactly0 difference despite different known held poses. This was a real
feature/output failure against the old production implementation, not a missing
API, skipped device, mocked result or inferred reproduction.

After the shared-mask source fix, ran BOTH complete covering files:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python .venv/task-7-run.py task9-fix1-covering-green \
  tests/wam/test_models.py tests/wam/test_model_losses.py -rP
```

Actual GREEN: **31 passed in3.49s**, exit0, no skipped case. Captured real
device markers include:

```text
INACTIVE_ROOT cpu: real full/no forward; shared171 root pose/inactive0; no world call; later-placeholder immunity; independent two-row root conditioning
INACTIVE_ROOT mps: real full/no forward; shared171 root pose/inactive0; no world call; later-placeholder immunity; independent two-row root conditioning
31 passed in 3.49s
```

The complete covering run also retains the existing CPU/MPS forward/backward,
actor/world/quality gradient isolation and freeze checks, own-future provenance,
object/action/score permutation behavior, genuine two-row independent
conditioning, mixed active→hold physical-rollout immunity, unchanged codec
forward mapping and periodic/masked physical/proposal/terminal loss checks.
Existing parameter counts and hidden widths printed unchanged. No optimizer
step or actual training occurred. No duplicate native full-WAM or root run
was made;694.71s/400passed remains the PRE-FIX historical run only.

Exact logs/accounts are retained under the already-owned `.venv`:

- `task-7-task9-fix1-inactive-root-red.log` / `-account.json`
- `task-7-task9-fix1-covering-green.log` / `-account.json`
- `task-9-fix1-final-account.json` (source/cache/evidence/account audit)

### Account, unchanged-source/evidence audit and remaining concerns

Self-review confirmed the fix changes pose availability independently of ACT
activity, leaves original valid bits/arrays untouched and exposes no later
placeholder. Final source audit held the existing `.venv/task-2-run.lock` and
canonical ledger, with linked60s/1MiB reservation and activity
`task9_fix1_final_source_account_audit`; it reran no model or native probe.
It verified exact staged two-file source/test scope, `git diff --cached --check`,
no unstaged tracked changes, old report unchanged before this append, the
other SIX Task9 source/test hashes unchanged, every prior producer/config
source hash/exact FIX_BASE bytes unchanged, r8 manifest/model/dependency hashes,
and the complete new test-account chain. Audited amended SHA256s:

- graph.py: `2acbb23f8ab21e517b43cbea4a7a14765993206328ae220f72ecac5158c93ffd`
- test_models.py: `139fcc139c614100d53359621cad643356357161c89a75b05fb02bf69b24ecf8`

Dispatch:5531.923959035874 effective seconds /309events /pilot0 /
active=null /reservations={}. Final: **5539.608646173478 effective seconds /
315events /pilot0 /active=null /reservations={}**, warnings=[]. Added fix cost
**7.68468713760376 effective seconds**, including real RED, complete covering
CPU/MPS GREEN and final source/cache/account audit.

| Activity | Effective seconds before→after | End events |
|---|---|---:|
|task9-fix1-inactive-root-red|5531.923959035874→5533.949943084717|311|
|task9-fix1-covering-green|5533.949943084717→5538.049547929764|313|
|task9_fix1_final_source_account_audit|5538.049547929764→5539.608646173478|315|

All inherited309/295/277/217 and167/179/191/197/209 prefixes are preserved;
all previously recorded prefix digests match. The inherited309 prefix digest
is `9e0696fdfe273ba00e5bb629410d246c3a3c5e136aca93446239bbf4b6ccf434`.
Each runner verified its entire inherited event prefix and immutable policy
after the run. Both distinct controller1s priors remain exactly once. Canonical
roots and distinct run/native locks are unchanged. Audit-poll observations:
owned2361438040bytes /free79270068224bytes; these are observations, not a
throughput/volume profile. The audit JSON adds a small owned file after polling.

Original Task9 RED/GREEN/full-run logs AND their accounts match their archived
hashes/content; the original final account JSON was not replaced. Historical
Task8 failed-full and complete-fix log hashes remain unchanged. Prior native
proof JSONs in pytest-63 and pytest-64 match archived values. Retained root
pytest cache hashes/inodes and180224bytes counted once remain intact; original
cache source/root `.pytest_cache` remain absent. ResNet initializer and all
sealed r8 dependency hashes are unchanged. No old cache, raw evidence or other
resource was deleted, moved or overwritten; no duplicate native run or new
policy root/ledger was introduced.

No covering test failure or unresolved dependency gap remains for this scoped
fix. Original untrained/uncalibrated-model, detached-codec and modeled future
hold limitations remain as documented above. The repaired root-held pose is
causally KNOWN, distinct from that unchanged future-hold modeling limitation.
Controller alone owns independent scoped re-review and the final gate; this
append does not claim Task9 completion/publication or authorize Task10.
