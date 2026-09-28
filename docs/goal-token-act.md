# ACT goal-count task token

This experiment adds a categorical target-count condition to the ordinary
LeRobot ACT encoder without FiLM or a new loss. The six task IDs are
`target_count - 1` (`goal 1 → id 0`, …, `goal 6 → id 5`). A learned
`nn.Embedding(6, 512)` produces one task token in the ACT transformer encoder's
existing environment-token slot. The token has its normal encoder position
embedding; it does not modulate the image backbone.

The policy still receives the two RGB cameras and the existing 36-D robot
state. In goal-token mode, the three-value total/target/collected environment
vector is omitted from dataset batches and inference batches. The task ID stays
integer-valued and is not dataset-normalized. The simulator may continue using
counts and object truth for task termination, playback telemetry, and scoring;
those are not policy inputs. Expert generation remains A*-based and unchanged.

## Train

The approved experiment reuses the 90 successful schema-v5 V10 demonstrations
without copying them or adding failed/inference episodes:

```bash
python -m sim.act.train \
  --config runs/act_dataset_curve_v10/train_config.yaml \
  --dataset runs/act_dataset_curve_v10 \
  --out runs/act_model_goal_token_v1 \
  --policy-variant goal_token \
  --preview-training \
  --steps 100000
```

This initializes a fresh ACT policy with an ImageNet-pretrained ResNet-18. The
ACT action-chunk prediction objective, four-dimensional action, latent behavior,
MPS device, batch size 4, and configured checkpoint cadence are unchanged.
Checkpoints are eligible from step 50,000, saved every 10,000 steps, with the
latest six retained. The V10 ordinary checkpoint is not used to initialize this
model; it remains an inference comparison baseline.

## Inference

```bash
python -m sim.act.evaluate \
  --config runs/act_dataset_curve_v10/train_config.yaml \
  --model runs/act_model_goal_token_v1 \
  --policy-variant goal_token \
  --target-count 4 \
  --preview --record-replay
```

The model root resolves through `latest_checkpoint.txt` while training is in
progress, and to the final saved policy after training completes. The inference
path supplies task ID 3 for target four; it does not invoke A* or read component
positions for the policy.

## Paired evaluation

After the candidate is trained, evaluate the baseline and candidate on identical
held-out layouts:

```bash
python -m sim.act.compare \
  --dataset runs/act_dataset_curve_v10 \
  --baseline-model runs/act_model_curve_v10/checkpoints/step_100000 \
  --candidate-model runs/act_model_goal_token_v1 \
  --out runs/evaluations/act_goal_token_v1
```

The comparison first chooses six unique validation layouts and twenty distinct
test layouts, excluding training seeds/fingerprints. Every layout is evaluated
at target counts 1–6 for both policies. `layout_plan.json`, per-episode
`episodes.jsonl`, and per-target/paired `summary.json` are written incrementally
or at completion. The test split must remain reserved for the final frozen
comparison rather than iterative tuning.
