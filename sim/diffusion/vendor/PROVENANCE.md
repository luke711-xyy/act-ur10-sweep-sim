Source: https://github.com/real-stanford/diffusion_policy
Commit: 5ba07ac6661db573af695b419a7947ecb704690f (MIT, LICENSE retained).
Copied conditional U-Net, conv/embedding, EMA, MultiImageObsEncoder,
model_getter, ModuleAttrMixin and pytorch_util. Imports namespaced locally.
Encoder change: unused crop dependency removed; crop/resize explicitly rejected.
No other algorithm changes. Wrapper implements masked loss, data normalization,
DDIM diffusers==0.35.1 and checkpoint/task integration (not LeRobot DP).

TransformerForDiffusion added from the same pinned commit, import-prefix changes only.
Upstream SHA256: c4cfd71a2f1b4f5181c7ccc391d71f19ebce74c698fc12725bb88317b6f8fd8f
Official image-task8/4/256 defaults; wrapper retains this task's five-view ResNet
encoder, limits normalization, DDIM8 and receding-horizon execution.
