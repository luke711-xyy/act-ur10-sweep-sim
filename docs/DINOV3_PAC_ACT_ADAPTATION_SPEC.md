# Ubuntu DINOv3 PAC-ACT adaptation specification

Status: review required before implementation.

## Objective

Add an isolated PAC-ACT on-policy PPO post-training path for the locally trained DINOv3 + ACT checkpoint. All changes belong to `codex/ubuntu-dino-pac-act`, created from PAC-ACT reference commit `8c19ef18dd1624691b42972183f059bf037c6864`. Do not merge or switch the active worktree, change default inference behavior, or modify historical checkpoints and evaluations.

## Model and training contract

- Base checkpoint: `/home/luke/luke－projects/UR１０SWEEP/runs/dinov3_act_bs4_200k_20261016/model/checkpoints/step_200000`. Load locally and fail closed on missing or mismatched files; do not download a replacement encoder.
- Preserve the serialized `act_dinov3` / `dinov3_vits16` identity, pinned DINOv3 ViT-S/16 revision and weight digest, DINO RGB preprocessing, five-camera ordering, ACT processors, normalization statistics, 36-D robot state, 1-D target-count environment input, 4-D action normalization, ACT chunk length 25, and inference cadence of five actions per block.
- Strictly load every checkpoint tensor. Before any PPO update, run a deterministic comparison against the original checkpoint and require matching five-step normalized and physical action outputs within a documented numerical tolerance. Reject any processor, camera-order, feature-shape, normalization, or state/action mismatch.
- Keep the DINOv3 transformer frozen and in evaluation mode in the Actor, frozen reference, and Critic. It must receive no optimizer parameters or gradients. PPO may update ACT-owned image/state/latent projections, Transformer encoder/decoder, action head, and the learned XY/yaw exploration scales. Remove the ACT CVAE training-only path as PAC-ACT does and retain the zero-latent deterministic ACT inference path.
- Retain the original checkpoint normalizers and a frozen copy of the original DINOv3 ACT policy as the reference prior. Preserve the PAC-ACT PPO, executed-prefix probability accounting, reward definition, prior penalty, optimizer/update transaction, rollback/audit format, and checkpoint durability protocol unless a concrete incompatibility is found; stop and report such incompatibilities rather than silently tuning or changing the algorithm.
- Preserve action ownership and safety: PPO samples only executed XY/yaw actions; predicted ACT Z is ignored. Existing A* approach, contact/acquisition, handoff, admittance/follow, filtering, IK, rate/workspace limits, safety checks, task budget, and native success/termination logic remain unchanged.

## Data, layout and execution gates

- Package an independent, hash-verified bundle for the 200K DINOv3 checkpoint and this branch’s runnable source. Preserve all original model files and statistics; output new bundles, runs, checkpoints, journals and media into new paths.
- Build exclusions from all known prior demonstrations, preflights, residual runs and the existing 120-layout evaluation (seeds 2120000–2120119). Keep each layout opportunity one-shot: no retry, replacement seed, success filtering, or repetition of an interrupted physical attempt.
- Do not acquire the simulator/inference slot or start physical acceptance while the current 120-layout evaluation is active. After it finishes, run only the already-approved CPU engineering acceptance: up to 24 one-shot new-layout opportunities and two PPO updates, with all failures and attempts recorded. No separate physical smoke test.
- Start CUDA formal post-training only after CPU acceptance passes and a free inference lease is verified. Keep PAC-ACT resource limits, fixed validation schedule, milestone checkpointing and stop budgets. Never auto-resume after a fault; recover only from the last verified complete update under the PAC-ACT protocol.
- No training or physical evaluation runs during implementation and review of this specification.

## Implementation and verification scope

1. Add DINO-aware strict model loading, Actor/reference/Critic construction, checkpoint packaging and identity checks only in this branch.
2. Add focused tests for frozen DINO weights in all three model roles, trainable ACT modules, strict checkpoint reload, normalizer and camera-order preservation, zero predicted-Z use, initial deterministic action equivalence, and checkpoint round-trip/rollback identity.
3. Run narrow CPU unit/contract tests and an offline forward/load check. These are software checks, not evidence of policy performance or acceptance.
4. Review the final plan and tests before starting CPU acceptance; report the exact gate results before formal CUDA training.

## Out of scope

No ResNet model or checkpoint, PPO/SAC residual controller, default inference/config changes, training-data mutation, reward/control retuning, batch-evaluation changes, or broad test suite. Do not modify the active evaluation worktree or root checkout.

## Open acceptance checks

- Confirm that the current 120-layout evaluation has a complete ledger and that its dependency/lease checks pass before CPU acceptance.
- Confirm strict initial action equivalence for the exact 200K checkpoint and validate GPU memory/throughput under the PAC-ACT six-environment configuration before baseline validation.
- If either check fails, preserve outputs and report the specific incompatibility. Do not compensate by changing DINO preprocessing, normalization, control or PAC-ACT hyperparameters.