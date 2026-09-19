# ObjectACT-BEV RGB-first implementation plan

## Goal and hard boundary

Add structured visual object tokens and a table-frame multi-channel BEV to the
ordinary MuJoCo UR10 ACT simulator while preserving the working schema-v4 ACT
baseline. The first policy-facing sample must already be produced by the RGB
perception pipeline. There is no truth-input warm-up stage: a MuJoCo truth
ablation, if used, is an offline detector-supervision/audit tool only.

The existing v4 code, successful demonstrations, historical checkpoints and
ordinary workbench path remain loadable. The new path is opt-in through
`act.policy_variant: objectact`, with separate v5 manifests, normalizers,
detector artifacts, checkpoints and replay metadata.

## Causal observation contract

```text
overhead RGB + wrist RGB
        -> grouped RGB instance detector
        -> calibrated table-plane projection and short-term tracking
        -> six 29-D object tokens + [6,128,160] BEV
        -> permutation-invariant exact Top-N selector
        -> ObjectACT CVAE/Transformer/action head
```

The BEV channels are, in order: all occupancy, selected occupancy, unselected
occupancy, tray signed distance, current brush footprint, and table/free-space
signed distance. The token fields include grouped class probabilities, current
and previous XY/extent/yaw features, velocity, brush/tray-relative geometry,
visibility, confidence, staleness and validity.

The policy receives no MuJoCo object position, planned A* identity, expert
future path or cached truth mask. The selection target is an offline label
matched from actual final complete collected track IDs; it is not an input.

## Data and detector gate

1. Keep ordinary v4 ACT data untouched. Reuse the current successful v4 expert
   RGB frames as detector images, replaying scenes only offline to create
   segmentation labels.
2. Split detector frames by `layout_id`, never by adjacent frames, and write a
   separate detector manifest. `sim.act.collect_detector_dataset` restores the
   robot/part state, renders offline geom segmentation, and writes only RGB plus
   dense semantic/centre/offset labels.
3. Train the frozen-or-optionally-ImageNet-initialized ResNet-18 FPN with a
   learned RGB objectness head, semantic class head, centre heatmap and
   offset head using `sim.act.train_detector`. The objectness head is the
   Push-Wiper-inspired soft binary occupancy representation: it is learned
   from RGB labels, not a MuJoCo mask at policy time. The raw RGB branch is
   retained for appearance and occlusion cues, while objectness supplies the
   spatial topology used to build tokens and BEV. ImageNet normalization and
   the 0.60 foreground threshold are shared by detector training, audit and
   online inference.
4. Report both raw mask diagnostics and the temporal centre/track metrics used
   by ObjectACT. A detector checkpoint is usable by v5 generation, replay or
   inference only if the causal temporal gate passes:

   - visible-mask IoU >= 0.70 as a minimum geometry diagnostic;
   - centre error <= 3 px;
   - centre miss rate <= 15% over visible validation frames;
   - centre false-positive rate <= 10%;
   - the first five detector observations of each held-out episode cover at
     least 95% of the visible object tracks;
   - warm-up absolute count error <= 0.25.

   Raw mask miss/false-positive/count rates remain in the report, but a brief
   mask loss caused by the robot or brush occluding a part is not itself a
   policy-input failure. The tracker carries the last causal token with an
   explicit staleness field until the configured age limit.

   Pixel binarization of the raw image is explicitly not the runtime
   contract: the checkerboard, shadows, robot and brush can all change
   appearance. A fixed empty-table calibration may be added later as a
   causal auxiliary feature, but it cannot replace the learned RGB objectness
   gate or introduce simulator truth.

   Missing or unverified detector metadata is a hard error. Randomly
   initialized detector weights may not silently create policy data.

## v5 policy and control

- Use 36-D robot state, 6-D visual task state, six 29-D object tokens, six BEV
  channels and two 320x320 RGB views.
- Keep ACT action chunks as `[Delta x, Delta y, Delta z, Delta yaw]` in the table frame.
- ACT owns all four axes before measured contact. After contact latch, only
  applied Z is replaced by the 1 N admittance controller; ACT continues to
  produce XY/yaw and its policy Z remains logged for audit.
- Preserve 5 Hz queries, 25 Hz execution, substep interpolation, speed limits,
  late preview alignment, rolling prediction and temporal ensembling.
- Preserve the approved 20 N sustained-force protection, 500-frame cap,
  exact-count scoring, full projected inclusion and inference visual timers.
- Training/inference paths must still run if A* entry points are configured to
  raise; A* is used only by the expert generator.

## Expert data and training

- Generate v5 observations with the verified RGB detector from the first frame.
- A* may generate the high-quality expert action path, but target identities
  never become policy inputs. The v5 manifest accepts only successful expert
  records with action dimension 4.
- Preserve the formal 120-success target: the existing 90 successful records
  plus 30 additional paired-layout records (eight paired layouts x six goals
  and twelve independent layouts per goal as specified by the data plan).
- Before the full run, pass the six-episode gate: autonomous contact within
  500 frames, first-contact XY median error <= 3 cm, peak force <= 20 N, and at
  least 5/6 exact successes.
- Train ObjectACT from scratch with ImageNet ResNet-18 settings, batch size 4,
  at most 80,000 steps or 24 hours, checkpoints every 2,500 steps, latest
  three retained plus milestones 2,500/10,000/40,000/80,000, and Trackio
  logging kept separate from ordinary ACT.

## Workbench and verification

The workbench exposes detector source/checkpoint and quality-gate metadata,
object-token tables, validity/visibility/staleness, BEV channel inspection,
selection logits, visual-vs-truth counts for audit, policy Z, applied Z,
Z-owner, force and termination reason. v4 and v5 replay manifests remain
separate.

Verification order:

1. ordinary baseline branch and new GitHub branch are pushed and readable;
2. v5 contract and detector unit tests pass;
3. one real MuJoCo replay creates detector supervision and one-step detector
   training writes a frontend-compatible checkpoint;
4. held-out detector gate passes;
5. v5 manifest has exactly the declared successful records;
6. six-episode gate passes;
7. start the 80,000-step ObjectACT run and verify checkpoint round trips,
   A*-isolation, contact handoff, latency and visual termination behavior.

## Revised boundary from the Push-Wiper discussion

This plan keeps the useful insight from binary-wipe work -- spatial topology is
an important policy input -- but uses a causal detector rather than feeding
simulator truth. RGB remains available for appearance and abnormal cases;
tokens and BEV make the object layout explicit enough for ACT's relatively
weak visual head.
