Source: https://github.com/real-stanford/diffusion_policy
Commit: 5ba07ac6661db573af695b419a7947ecb704690f (MIT, LICENSE retained).
Copied conditional U-Net, conv/embedding, EMA, MultiImageObsEncoder,
model_getter, ModuleAttrMixin and pytorch_util. Imports namespaced locally.
Encoder change: unused crop dependency removed; crop/resize explicitly rejected.
No other algorithm changes. Wrapper implements masked loss, data normalization,
DDIM diffusers==0.35.1 and checkpoint/task integration (not LeRobot DP).
