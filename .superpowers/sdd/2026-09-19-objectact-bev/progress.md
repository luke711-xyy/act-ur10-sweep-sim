# SDD ledger — plan: docs/superpowers/plans/2026-09-19-objectact-bev.md

## Baseline

- Base branch: `feat/objectact-bev`
- Base commit: `e12a9ec`
- Ordinary ACT baseline was uploaded to `origin/feat/robot-learning-workbench`.
- Full baseline pytest suite passed before starting this plan.

## Pre-flight

- Task 1 produces schema-v5 contracts and sidecars consumed by Tasks 2–5.
- Task 2 produces predicted tracks and packed instance BEV consumed by Tasks 3–5.
- Task 3 produces selection and BEV tensors consumed by Task 4.
- Task 4 produces the v5 policy API consumed by Tasks 5–6.

Task 1: complete (commit 284faa3, tests: `pytest -q tests/test_objectact_v5_contract.py` and full `pytest -q` → pass)

Task 2: complete (predicted object tokens, camera-mask projection, dual-view
fusion, geometry-only tracking, tray features, and six-channel BEV; tests:
`pytest -q tests/test_object_tokens.py tests/test_object_bev.py` and full
`pytest -q` → pass)

Task 2 detector extension: complete (frozen/optionally ImageNet-initialized
ResNet18-FPN semantic, center, and offset heads plus RGB-only instance
decoding; tests: `pytest -q tests/test_object_detector.py` → pass)

Task 3: complete (permutation-invariant selection head, exact Top-N
straight-through selection, actual-collected-ID labels, and the 0–5k /
5k–20k / post-20k teacher schedule; tests: `pytest -q
tests/test_object_selection.py` → pass)

Task 4: complete (ObjectACTConfig/ObjectACTPolicy with separate robot/task,
object-token, BEV, and RGB branches; official LeRobot CVAE/decoder,
L1/KL loss, zero-latent inference, and temporal ensembling; tests:
`pytest -q tests/test_object_policy.py tests/test_v5_bev.py
tests/test_objectact_v5_contract.py` and full `pytest -q` → pass)

Task 4 selection integration: complete (policy-boundary selector applies
teacher-scheduled or predicted Top-N to BEV selected/unselected channels and
adds the BCE supervision term; focused policy tests pass)

Task 5A: complete (v5 modality-aware statistics, CPU/MPS-safe preprocessing,
resumable schema-v5 checkpoints, 80k/2500-step defaults, and milestone
retention; focused training tests pass)

Task 5B: complete (RGB-derived observation builder, camera-pose lookup,
contact-latched Z handoff, and a separate ObjectACT runtime loop with the
existing 5 Hz/25 Hz scheduler, late-preview alignment, temporal chunk
ensembling, 20 N protection, 500-frame cap, and visual collection timers;
ordinary v4 evaluator remains the default)

Task 5C: complete (schema-v5 atomic writer, RGB-derived expert observation
capture, offline actual-final-collected selection labels, and a guarded
single-episode v5 generation CLI; it refuses to generate policy data without
an explicit RGB detector checkpoint)

Task 6A: complete (workbench discovers v5 train/replay manifests separately,
loads v5 token/selection/BEV inspection payloads, exposes v5 perception API
data, and can launch ordinary or ObjectACT training modules without changing
the ordinary defaults; focused workbench regressions pass)

Task 2 detector data/training gate: complete (RGB detector supervision writer,
layout-grouped v4 replay source, dense semantic/centre/offset targets,
held-out quality metrics and hard checkpoint verification; focused detector
tests, real one-frame MuJoCo replay, one-step detector training and full
pytest suite pass)

Ruling: the revised RGB-first boundary removes the proposed truth-input warm-up
stage. MuJoCo segmentation is retained only for offline detector supervision
and audit because feeding it to the policy would not represent the intended
real-scene deployment boundary.

Pending: run the held-out detector gate, generate the v5 expert manifest, pass
the six-episode gate, and start the formal 80,000-step ObjectACT run. These
are intentionally not marked complete because no verified detector checkpoint
or v5 policy dataset has been produced in this branch yet.
