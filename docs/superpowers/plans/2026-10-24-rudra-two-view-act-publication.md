# Rudra two-view DINOv3 ACT publication

User-requested scope: publish the existing working dual-view ACT implementation on
`codex/rudra-dinov3-act-two-view` and provide a direct configuration handoff.

- [x] Preserve the active ACT and Diffusion processes and their launch configs.
- [x] Add a portable snapshot of the current ACT/scene profiles and a configuration-only CLI.
- [x] Include the frozen DINO adapter, ACT trainer/checkpoint/replay fixes and copied workbench dependencies.
- [x] Add dependency metadata, setup documentation, and bounded configuration tests.
- [x] Verify the staged tree from a clean source export without downloads, GPU training or extra policy rollouts.
- [ ] Commit, push the new branch normally, verify the remote ref, and hand off a branch-based prompt.

The live host-specific profiles are retained on disk; portable versions of those
same paths are staged for publication. Dataset/model/media/credentials are excluded.
