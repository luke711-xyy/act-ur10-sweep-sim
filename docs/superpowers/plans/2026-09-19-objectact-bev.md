# ObjectACT-BEV Implementation Plan

This branch executes the approved ObjectACT-BEV schema, perception, BEV, selection, policy, training, and workbench plan. See the full plan in the task record.

## Revision addendum: RGB-first, no truth-input warm-up stage

The original plan's proposed truth-input warm-up stage is removed. From the
first v5 sample and first v5 rollout, policy observations are produced by the
RGB detector, calibrated table projection, short-term tracker and BEV
rasterizer. MuJoCo truth is permitted only as offline supervision for the
detector and as an evaluation/audit reference; it must not become an ACT
observation, object token, selection input, target identity or future path.

- Use a frozen/optionally ImageNet-initialized ResNet-18 FPN with semantic,
  center-heatmap and offset heads to decode RGB instances. Report held-out
  mask IoU, center error, miss/false-positive rate and count error before an
  end-to-end claim.
- Fuse predicted overhead/wrist instances in the table frame, track stable IDs
  with occlusion age, then form six 29D object tokens and a six-channel
  `[6,128,160]` BEV. Reconstruct the dense BEV at the policy boundary from
  predicted fields rather than cached simulator truth.
- Use a permutation-invariant selector for exact Top-N. Selection supervision
  uses actual final complete collected track IDs only; A* target IDs are never
  policy inputs or labels.
- Keep ordinary v4 ACT, data and checkpoints untouched. v5 is opt-in through
  `act.policy_variant: objectact`, with its own manifest, normalizer,
  checkpoint directory and replay metadata.
- Keep A* expert-only. v5 training reads RGB-derived observations and v5 ACT
  inference starts at reset and learns the approach/descent location itself.
  Both paths must remain functional when A* entry points raise.

Acceptance order is detector quality audit → six-episode contact/force/
exact-count gate → 80,000-step training, while retaining the 2,500-step
checkpoint cadence and 500-frame/visual-timer inference contract.
