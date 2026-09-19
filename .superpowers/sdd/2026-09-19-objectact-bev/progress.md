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
