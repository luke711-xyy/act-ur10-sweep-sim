# Ubuntu DINOv3 PAC-ACT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在隔离的 PAC-ACT 分支上，为本机 DINOv3 ACT 200K checkpoint 接入冻结视觉编码器的 PPO 后训练，并在既定门槛通过后执行 CPU 验收与 CUDA 正式训练。

**Architecture:** 保留 PAC-ACT 的 rollout、奖励、更新事务、控制和安全代码，仅为 DINOv3 ACT 增加严格的本地加载、冻结参数和 checkpoint 打包适配。Actor、reference 和 Critic 都从同一已校验的 DINOv3 base 构建；optimizer 与 learner checkpoint 只含可训练 ACT 参数和 PPO 状态。批测结束后由显式 DINOv3-sync5 队列验证既有结果，再启动原 PAC-ACT CPU 验收和正式训练流程。

**Tech Stack:** Python 3.12；PyTorch 2.11.0 + CUDA 12.8；torchvision 0.26.0；LeRobot 0.6.1；timm 1.0.30；Stable-Baselines3 2.9.0；MuJoCo 3.14.0；TensorBoard 2.21.0；psutil 7.2.2。

**Spec:** `docs/DINOV3_PAC_ACT_ADAPTATION_SPEC.md`

## Global Constraints

- 仅在 `codex/ubuntu-dino-pac-act` 分支修改；不改 root checkout、活动评测 worktree、历史评测、默认配置或默认推理入口。
- Base checkpoint 固定为 `/home/luke/luke－projects/UR１０SWEEP/runs/dinov3_act_bs4_200k_20261016/model/checkpoints/step_200000`；模型身份为 `act_dinov3` / `dinov3_vits16`，权重和文件必须本地严格加载。
- 保留 DINOv3 ViT-S/16 的 pinned revision/digest、RGB 预处理、五路相机顺序、36 维状态、1 维 target-count、4 维动作统计、25 步 chunk、每块执行 5 步以及现有 Z 导纳控制权。
- DINOv3 encoder 在 Actor、reference、Critic 内都冻结且不进入 optimizer；PPO 只更新 ACT 模块和 XY/yaw exploration scales。预测 Z 不执行。
- 沿用 PAC-ACT 既有 PPO/reward/安全/回滚/记账协议；不调 reward、控制参数、KL 系数或模型统计。已有 ResNet 代码路径保持回归通过，但这次运行不得加载 ResNet checkpoint。
- PAC 冻结快照与活动 DINO sync5 导纳不一致。DINO 运行必须显式选择分支内 hash-bound 的 `dinov3_sync5` 控制快照：沿用 1 N、±2 mm、seek/maintain MBK 10/400/10 和 10/800/10、1 mm gate、2 s handoff；maintain force compliance 为 5e-05 m/N（cap 0.0005 m、仅 maintain、compensated filtered wrist Fz）；overforce relief 为 1.5/1.1 N、0.0005/0.0005 m/s、cap 0.00025 m。不得使用旧 PAC 快照的 10/7 N、0.02/0.1 m/s relief，也不得借此调参。
- 当前 120 布局批测未完成前不得占用推理槽或启动物理验收；其 120 个布局和所有已知历史数据必须列入 exclusion，机会不可重试或换 seed。
- CPU 验收限时 2 小时、最多 24 个一次性机会及 2 次 PPO 更新、输出不超过 5 GiB、保留至少 8 GiB 空间；验收失败不启动 CUDA 训练。
- 正式训练仅在通过且匹配 bundle 的 CPU 验收报告和空闲 CUDA/推理资源确认后启动；沿用最多 1,000,000 effective actions 或 72 小时、66 小时后停止新采样、50 GiB 输出上限、20 GiB 空间保留及 10 GiB / 30 秒 RSS 保护。
- checkpoint 在初始状态、每 50K effective-action milestone 和结束状态原子保存；固定 24 个 validation layouts，不挑选 best checkpoint，不自动恢复中断的物理机会。

## Review Focus

- 错误模型类型、checkpoint 文件缺失或 pinned DINO 身份不符：加载必须在创建环境前失败；由 Task 2 和 Task 4 的拒绝测试覆盖。
- DINO 权重被 Actor、Critic 或 optimizer 解冻：参数哈希/梯度和 optimizer 成员测试由 Task 3 覆盖。
- 相机顺序、RGB 归一化、状态/动作维度或 checkpoint processor 漂移：确定性真实 checkpoint 对照测试由 Task 2 与 Task 6 覆盖。
- checkpoint 恢复遗漏可训练张量或覆盖 frozen DINO base：快照往返与篡改拒绝测试由 Task 3 覆盖。
- 120 批测仍在运行、重启器或编码器未退出、账本或视频不完整：DINOv3-sync5 队列测试由 Task 5 覆盖。

---

### Task 1: Branch-local DINO sync5 controller-profile parity

**Files:**
- Create: hybrid_trial4_snapshot/dinov3_sync5/control_profile.json
- Create: hybrid_trial4_snapshot/dinov3_sync5/evaluation_config.yaml (or an equivalently explicit, bundle-contained representation)
- Modify: pac_act/hybrid.py
- Modify: pac_act/identity.py
- Modify: pac_act/lifecycle.py (manifest profile propagation into each episode)
- Create: tests/test_pac_dinov3_control_profile.py
- Modify: tests/test_pac_execution.py (construct a legacy bundle fixture with an explicit manifest)

**Interfaces:**
- Resolve the controller profile explicitly from the verified bundle manifest as dinov3_sync5; reject unknown profile ids or digest/config mismatches before environment construction.
- Preserve the legacy PAC snapshot and loader behavior for provenance and existing tests. The DINO bundle must never select its older relief settings.

- [x] **Step 1: Write failing tests** for exact sync5 control-profile values and fail-closed profile/config drift or unknown ids. Cover target force 1 N, ±0.002 m Z limit, seek/maintain MBK 10/400/10 and 10/800/10, 1 mm contact gate, force/stability thresholds and dwell, 2 s handoff, handoff-only XY/yaw acceleration limits, tracking/follow, wrist-Fz feedback, and ACT-predicted-Z disabled.
- [x] **Step 2: Add** the compliance and relief behavior recorded by the active sync5 metadata/config: maintain compliance 5e-05 m/N capped at 0.0005 m only in maintain; overforce relief activates/releases at 1.5/1.1 N with 0.0005/0.0005 m/s rates and 0.00025 m cap. Bind these values and the effective evaluation config to the bundle source/profile digests.
- [x] **Step 3: Implement** explicit profile selection in the PAC runtime and a bundle-contained profile contract. Task 4 wires the DINO source profile/config into the portable bundle manifest. Do not consult the active run directory at runtime; do not change any parameter or modify the root checkout, current evaluation worktree, or default inference route.
- [x] **Step 4: Test** Z correction/cap, relief activation/release, phase ownership and profile drift using synthetic force/state inputs only; no simulator episode, optimizer update or physical attempt. Require generated runtime settings to equal the frozen DINO sync5 profile.
- [x] **Step 5: Run** python -m pytest -q tests/test_pac_dinov3_control_profile.py; expect parity and fail-closed tests to pass.
- [x] **Step 6: Commit** with message fix: preserve DINO sync5 admittance profile in PAC-ACT.

---
### Task 2: DINOv3 ACT strict loader and PAC policy adapter

**Files:**
- Create: `sim/act/configuration_dinov3.py`
- Create: `sim/act/dinov3.py`
- Modify: `pac_act/policy.py`
- Create: `tests/test_pac_dinov3.py`

**Interfaces:**
- Keep `ChunkPolicy(base_path: str | Path, device: str = "cpu", build_critic: bool = True)` unchanged.
- DINO config/backbone exports `DINOv3ACTConfig`, `DINOv3ACTPolicy`, and `FrozenDINOv3Backbone`; DINO checkpoint dispatch is selected only by the exact serialized `type` and `vision_backbone` pair.
- Keep the existing `ChunkPolicy.prepare`, `means`, `values`, and `physical_actions` contracts.

- [x] **Step 1: Write failing tests** `test_dinov3_config_keeps_pinned_identity_and_five_view_contract`, `test_frozen_backbone_keeps_dino_eval_and_rgb_feature_map`, `test_chunk_policy_loads_dinov3_locally_and_rejects_wrong_contract`, and `test_dinov3_processor_preserves_camera_order_and_identity_visual_stats` in `tests/test_pac_dinov3.py`.
- [x] **Step 2: Run** `python -m pytest -q tests/test_pac_dinov3.py`; confirm the new DINO module and dispatch tests fail before implementation.
- [x] **Step 3: Implement** the config/backbone contract from the approved local DINO ACT implementation: pinned ViT-S/16 identity, strict local weight restoration, frozen eval-only encoder, ImageNet RGB normalization once inside the backbone, and original 2x2 spatial pooling. Never download or substitute pretrained weights.
- [x] **Step 4: Implement** DINO dispatch in `ChunkPolicy`; load saved pre/postprocessors and preserve serialized camera order. Reject any config except five 320×320 RGB views, state36, target-count1, action4, chunk25 and execute5. Keep ResNet loading code unchanged.
- [x] **Step 5: Run** `python -m pytest -q tests/test_pac_dinov3.py tests/test_pac_policy.py`; expect PASS.
- [x] **Step 6: Commit** with message `feat: load frozen DINOv3 ACT in PAC policy`.

### Task 3: Frozen-weight optimizer and learner checkpoint semantics

**Files:**
- Modify: `pac_act/policy.py`
- Modify: `pac_act/ppo.py`
- Modify: `tests/test_pac_dinov3.py`
- Modify: `tests/test_pac_ppo.py`
- Modify: `tests/test_pac_recovery_contract.py`

**Interfaces:**
- `ChunkPolicy.learned_state() -> dict[str, Tensor]` contains learnable Actor/Critic tensors, PPO noise and normalization buffers plus a tensor-encoded DINO base identity, but omits the frozen reference and DINO encoder weights.
- `ChunkPolicy.load_learned(state: Mapping[str, Tensor]) -> None` restores only those tensors, rejects unknown/missing learnable keys, and leaves each frozen DINO copy byte-identical to the base.
- PPO optimizer groups include only parameters with `requires_grad=True`; their existing actor/critic learning rates and role names stay unchanged.

- [x] **Step 1: Write failing tests** `test_dino_encoder_is_absent_from_actor_and_critic_optimizer_groups`, `test_ppo_updates_act_modules_without_changing_any_dino_encoder`, and `test_dino_learner_snapshot_round_trips_without_storing_base_encoder`.
- [x] **Step 2: Run** the three tests by name; confirm optimizer inclusion, state size/keys, or restore behavior fails before the fix.
- [x] **Step 3: Implement** frozen DINO enforcement after Actor and Critic construction; filter each PPO parameter group by `requires_grad`; omit frozen DINO and reference tensors from `learned_state`; make restore fail closed on missing learnable tensors or changed DINO base identity.
- [x] **Step 4: Run** `python -m pytest -q tests/test_pac_dinov3.py tests/test_pac_policy.py tests/test_pac_ppo.py tests/test_pac_recovery_contract.py`; expect the Actor, Critic, optimizer, accepted update and rollback contracts to pass.
- [x] **Step 5: Commit** with message `fix: keep DINOv3 frozen across PAC updates and restores`.

### Task 4: Portable DINOv3 200K bundle and pinned runtime dependency

**Files:**
- Modify: `pac_act/identity.py`
- Modify: `pyproject.toml`
- Modify: `requirements-pac-act.lock.txt`
- Modify: `docs/PAC_ACT.md`
- Modify: `tests/test_pac_bundle.py`
- Modify: `tests/test_pac_runtime_identity.py`

**Interfaces:**
- Keep `build_bundle(destination, model, excluded, project=PROJECT)` and `verify_bundle(root, project=PROJECT)` signatures.
- Add a fail-closed DINO base manifest identity: `model_type=act_dinov3`, `vision_backbone=dinov3_vits16`, `base_step=200000`, pinned DINO model/revision/digest, and per-file SHA-256 entries.
- Keep the existing 100K ResNet bundle contract passing; DINO run artifacts identify only the 200K checkpoint.

- [x] **Step 1: Write failing tests** `test_dinov3_200k_bundle_copies_and_hashes_every_checkpoint_file`, `test_dinov3_bundle_rejects_wrong_step_type_or_tampered_file`, and `test_existing_resnet_100k_bundle_contract_is_unchanged`.
- [x] **Step 2: Run** `python -m pytest -q tests/test_pac_bundle.py tests/test_pac_runtime_identity.py`; confirm DINO 200K packaging is rejected by the existing 100K guard.
- [x] **Step 3: Implement** an explicit DINO 200K bundle path that recursively copies and hashes all checkpoint files, including `training_state.pt` and the exact processor/statistics files, and records the DINO identity in the manifest. Preserve existing bundle paths and include the new model code in the sealed portable source.
- [x] **Step 4: Add** `timm==1.0.30` to the PAC-ACT optional dependencies and lock; update `docs/PAC_ACT.md` with the DINO 200K bundle command, strict offline load, and frozen encoder contract.
- [x] **Step 5: Run** `python -m pytest -q tests/test_pac_bundle.py tests/test_pac_runtime_identity.py`; expect both existing and DINO bundle checks to pass.
- [x] **Step 6: Commit** with message `feat: package the frozen DINOv3 ACT 200K base`.

### Task 5: Explicit completion queue for the current DINOv3 sync5 batch

**Files:**
- Modify: `pac_act/preflight_queue.py`
- Modify: `scripts/queue_pac_preflight.py`
- Modify: `tests/test_pac_preflight_queue.py`

**Interfaces:**
- Add explicit dependency profile `dinov3_sync5_200k` to `dependency_ready(root, *, active_pids, profile=...)` and `run_queue(..., dependency_profile=...)`; retain the existing profile as the default for compatibility.
- Add CLI option `--dependency-profile`; for this training run pass `dinov3_sync5_200k` and `--preflight-start-index 120` explicitly.
- Queue writes its own dependency certificate and never edits the source batch.

- [x] **Step 1: Write failing fixtures/tests** for a valid 120-row DINO batch, a running recovery launcher, incomplete status, wrong checkpoint/schedule, duplicate seed, missing/invalid video, out-of-root media path, and changed ledger after waiting for the inference slot.
- [x] **Step 2: Run** `python -m pytest -q tests/test_pac_preflight_queue.py`; verify the current DINO metadata and row format are refused by the legacy hard-coded 100K validator.
- [x] **Step 3: Implement** the DINO profile validator for the exact `hybrid_dinov3_200k` / step200000 model and schedule fields (`25`, `5`, `25Hz`, query5Hz, deterministic, no temporal ensemble, paused simulation during inference). Verify the plan digest, seeds 2120000–2120119 exactly once, target counts 1–6 exactly 20 each, complete ledger/summary/status, and one-shot result rows.
- [x] **Step 4: Implement** retained-video validation: require each result video under the batch `videos/` directory, reject symlinks/path escapes, decode at least one frame, and store size/hash/frame evidence in the queue certificate. Treat deferred or failed media as incomplete.
- [x] **Step 5: Extend** dependency process detection to the batch recovery launcher/episode entrypoint and video encoder. Never stop or signal them. Recheck status, files, processes and inference lease immediately before the exclusive preflight marker is written.
- [x] **Step 6: Run** `python -m pytest -q tests/test_pac_preflight_queue.py tests/test_pac_lifecycle.py`; expect old-profile behavior, DINO success fixture and no-retry failure cases to pass. Verified in the dedicated PAC environment, together with `tests/test_pac_safety.py`.
- [x] **Step 7: Commit** with message `feat: gate PAC preflight on completed DINO sync5 batch`; published on `codex/ubuntu-dino-pac-act`.

### Task 6: Isolated runtime, offline checkpoint verification and readiness gate

**Files:**
- Modify: `tests/test_pac_real_model.py`
- Add any missing DINO contract tests to: `tests/test_pac_dinov3.py`

**Interfaces:**
- The real-model test runs only when `PAC_ACT_BASE` points to the local `step_200000` directory; it does not create a MuJoCo environment or write training output.
- The deterministic equivalence assertion compares native DINO ACT processor/model outputs with `ChunkPolicy` reference output on identical fixed five-view observations, using `atol=1e-6, rtol=1e-6`.

- [x] **Step 1: Create** a dedicated Python 3.12 PAC-ACT environment; do not install into or alter the existing ACT `.venv`. Install the locked PAC-ACT packages plus `timm==1.0.30`, then run `pip check` and assert CUDA reports the expected RTX 3070 Ti / compatible PyTorch wheel. Verified Python 3.12.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, and NVIDIA GeForce RTX 3070 Ti Laptop GPU.
- [x] **Step 2: Write** `test_actual_dinov3_200k_reference_equivalence` in `tests/test_pac_real_model.py`; assert strict offline load, same camera order and processor statistics, finite `(1, 5, 3)` policy means, Z omission, ACT gradients, and unchanged DINO hashes.
- [x] **Step 3: Run** `PAC_ACT_BASE=/home/luke/luke－projects/UR１０SWEEP/runs/dinov3_act_bs4_200k_20261016/model/checkpoints/step_200000 python -B -m pytest -q tests/test_pac_real_model.py::test_actual_dinov3_200k_reference_equivalence`; expect PASS without network or physical rollout. Passed in CPU-only offline mode with equivalence at `atol=1e-6, rtol=1e-6` and frozen DINO/base hash checks.
- [x] **Step 4: Run** the focused suite: `python -B -m pytest -q tests/test_pac_dinov3.py tests/test_pac_policy.py tests/test_pac_ppo.py tests/test_pac_bundle.py tests/test_pac_runtime_identity.py tests/test_pac_preflight_queue.py tests/test_pac_recovery_contract.py`; expect PASS. Also passed `tests/test_pac_exclusions.py tests/test_pac_safety.py` in the same environment.
- [x] **Step 5: Verify** the newly built bundle at Task 7 Step 3 using `python -B scripts/train_pac_act.py --mode inspect --bundle <new-bundle>` and record runtime compatibility, DINO manifest identity, source hashes, and bundle identity. This check happens after the only approved exclusions-bound bundle exists and before CPU acceptance; do not begin acceptance on any mismatch.
- [ ] **Step 6: Commit** any test/doc fixes with message `test: verify DINOv3 PAC-ACT base contract`.

### Task 7: One-shot CPU acceptance followed by bounded CUDA post-training

**Files:**
- No edits to the active 120-layout batch or historical runs.
- New output root: `/home/luke/luke－projects/UR１０SWEEP/runs/pac_act_dinov3_200k_<UTC-run-id>/`

**Interfaces:**
- Bundle input: the verified `base_bundle` produced by Task 4.
- Acceptance certificate: new `preflight/acceptance.json`; formal training may read it only when `engineering_passed=true` and its `bundle_identity` matches exactly.
- Training output: a new `training/` directory with PAC-ACT journal, TensorBoard, videos, validation results and atomic checkpoints.

- [x] **Step 1: Check** the existing sync5 evaluation at `/home/luke/luke－projects/UR１０SWEEP/runs/hybrid_dinov3_200k_eval120_sync5_seed2120000_20261016`; do not launch anything until status and summary both say 120/120 complete and the Task 5 queue validator confirms all 119 available videos decode, the user-authorized seed 2120014 exception matches its incident/recovery report, no batch/encoder process is active, and the inference lease is free.
- [x] **Step 2: Build** a new exclusions snapshot from the dataset manifest, old layout plan, source-defined residual schedules, current DINO sync5 `episodes.jsonl`, and every other known preflight/demo ledger. The user authorized creating the missing residual schedule/ledger: regenerate the pilot and long-training seed schedules from the PAC source, create a separate history-inventory ledger that explicitly says the old attempt ledger was not found (do not invent episode rows or claim zero historical attempts), and preserve that status in the exclusion evidence. Exclude every source-scheduled seed, all known observed seeds/fingerprints, and the full reserved 5M–6M interval. Stop before physical acceptance if source schedules cannot be reproduced or a newly discovered record conflicts.
- [x] **Step 3: Package** the local 200K checkpoint and verified exclusions into `<new-output-root>/bundle` using `scripts/package_pac_act.py`; run `--mode inspect` and require the manifest/checkpoint identity to match the completed evaluation source checkpoint.
- [x] **Step 4: Skip CPU acceptance** per the user's direct instruction. The interrupted preflight attempt is preserved as incomplete evidence; it is not an acceptance pass.
- [x] **Step 5: Check launch resources** (CUDA availability, free inference lease, and disk reserve) without running CPU acceptance or a policy baseline rollout.
- [x] **Step 6: Start** the authorized formal run once with the run-scoped CPU-acceptance waiver and baseline-validation skip: `python -B <bundle>/source/scripts/train_pac_act.py --mode train --bundle <bundle> --out <new-output-root>/training --device cuda --waive-cpu-acceptance --skip-baseline-validation --acknowledge-gpu-training`. The six-environment backward-memory startup benchmark remains; it does not execute policy actions. It may only reduce microbatch 4→2→1.
- [ ] **Step 7: Monitor** the existing PAC-ACT dashboard/TensorBoard and status files. Preserve every attempt and failure; checkpoints/validation occur at each 50K effective-action threshold and final accepted checkpoint. Stop at the first existing action/time/resource cap; on any exception preserve evidence and do not auto-resume.
- [ ] **Step 8: Report** effective/sampled/used actions, update and checkpoint identities, fixed-layout strict-success and count-match rates, failure classes, validation and media paths. Label all results `DINOv3 ACT 200K + PAC-ACT PPO`; do not mix with the separate sync5 inference evaluation.

## Execution Notes

- The sync5 run finished 120/120 with `status=complete`, `summary.complete=true`, no active recovery/evaluator/encoder processes, and the inference slot released. The recovery report records 119 videos for 120 rows; seed 2120014 is a known IK-unreachable attempt with `replay_input_path=null`, `video_path=null`, and an incident record stating no frames were persisted and the layout will not be rerun. On 2026-10-17 the user authorized a run-scoped exception for this one row and directed us to proceed without further investigation; never rerun it or fabricate media. The live validator verified all 119 videos and the exact incident/recovery-report hashes. For this declared row only, the missing planning-call count comes from its `planner_attempts`, the missing controller profile from hashed batch metadata, and the missing inference strategy from its `planner_strategy`. All default queue behavior remains strict.
- The user authorized proceeding with PAC-ACT RL post-training without letting the legacy row block the run; the later, more specific instruction supersedes the earlier CPU acceptance gate for this invocation.
- On 2026-10-17 the user directly changed the launch sequence: skip CPU acceptance preflight and skip the initial policy baseline evaluation, then enter formal training. Both choices are explicit one-run flags, recorded in `run.json`, and do not mark engineering acceptance as passed. The GPU startup memory benchmark remains because it performs no policy rollout or optimizer update. The interrupted CPU preflight logs stay preserved as incomplete evidence.
- The direct formal run uses `training_direct_no_video_no_baseline` with bundle identity `93de8e104562e9ff8e95ef4140b3088e8895c4853af9cba639d60fd42cdfd1c3`. CUDA availability, inference lease, and disk reserve were clear. The six-environment backward benchmark passed at microbatch 4; the run entered `collecting384` on its first training opportunity. Its validation replay media adapter disables videos for this invocation only.
- On 2026-10-17 the user requested that formal training omit the full-rollout KL scan. `ChunkPPO.train()` now reports the existing differentiable minibatch KL term only; the KL coefficient and remaining PPO/update safeguards are unchanged. The earlier `training_framebufferfix_novideo` attempt was interrupted before any PPO update and is preserved with its partial baseline rows and initial checkpoint. A fresh bundle identity and matching CPU acceptance are required; do not resume or overwrite that stopped run.
- The legacy residual export root encoded in `residual_rl.runtime` is a macOS absolute path and is absent on this Ubuntu host. The source defines the pilot/long-training schedules, but no old per-episode residual ledger was found in local `runs` or worktrees. The replacement evidence must distinguish regenerated plans from unavailable historical attempts; never label it a recovered ledger.
- All `<...>` paths in Task 7 are values resolved from the one newly created run root; create that root once with exclusive creation, record it in queue authorization, and never reuse an existing output directory.
