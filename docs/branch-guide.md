# Branch guide

Branch heads below were checked on 2026-10-01. The hashes identify the local
starting points; branches may advance as experiments continue.

| Branch | Starting point | Purpose / important distinction |
| --- | --- | --- |
| `main` | `e8895a2` | Stable ordinary-ACT workbench baseline. |
| `feat/robot-learning-workbench` | `e8895a2` | Ordinary ACT workbench development line; at this snapshot it matches `main`. The 200k experiment used the ordinary policy with scalar target-count conditioning, not a learned task embedding. |
| `feat/mt-act-goal-embedding` | `1a0d484` | Optional categorical goal-token/embedding experiment. The branch name alone does not mean every run on it uses the embedding: the active 200k run selected the ordinary policy and `ordinary_target_count_only: true`, so it did not train or use an embedding head. |
| `feat/objectact-bev` | `5ce7a97` | Structured object/BEV perception experiment; separate from ordinary ACT and the camera-view work. |
| `feat/v4-residual-rl` | `9965e8c` | V4-based residual reinforcement-learning prototype. |
| `feat/act-five-view` | based on `feat/robot-learning-workbench` / `e8895a2` | Five-view policy-input development line: the four table-corner cameras and the wrist camera are all ACT observations. The separate whole-scene inspection camera is preview-only. Corner-camera poses are still candidates pending user review. Current Rudra UR10, fixed Robotiq, and brush model assets are carried over. |

## Which branch to start from

- Ordinary ACT with the scalar target count: start from `feat/robot-learning-workbench`.
- Learned categorical goal token: use `feat/mt-act-goal-embedding` and explicitly select that policy variant.
- Structured object tokens or BEV: use `feat/objectact-bev`.
- V4 residual RL: use `feat/v4-residual-rl`.
- Four additional overhead viewpoints as policy inputs: use `feat/act-five-view`; the four corner images and wrist image enter the ACT observation together. Wait for camera-pose review before freezing the candidate viewpoints for formal data collection.

The active 200k run's checkout happened to be named `feat/mt-act-goal-embedding`,
but its selected method was ordinary ACT. For reproducibility, identify a run by its
policy configuration and observation schema, not by the branch name alone.
