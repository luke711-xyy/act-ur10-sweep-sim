# Frozen DINOv3 + ACT

分支 `feat/act-dinov3-frozen`，基于 `feat/act-five-view` 的
`9102680d8e018b225ab34d36422572cd4b547166`。仅替换可选视觉编码路径：
冻结预训练 DINOv3，重新初始化并训练图像适配层及原 ACT/VAE。
不从旧 ResNet ACT 保存点继续训练，不修改旧示教、旧模型或正式评测结果。

## 接口与权重

```text
五路 RGB 320×320，各自共享同一个冻结 DINOv3 ViT-S/16
  → 去除模型声明的 CLS/register 前缀，只保留 20×20 空间 patch
  → 2×2 平均池化为 384×10×10
  → 可训练 1×1 投影为 512×10×10
  → 原 ACT Transformer/VAE → 25×4 动作，执行前 5 步
```

不是 YOLO、目标检测或 occupancy；没有加入这些额外监督。保留 36 维状态、1 维
float32 目标数量的 Linear 输入、五路视角顺序，以及原 ACT 的 loss、KL、chunking。
DINO 全部参数 `requires_grad=False`；调用 `policy.train()` 后编码器仍为 eval，
优化器不包含编码器。1×1 投影、Transformer、VAE、动作头从头训练。

原始 RGB 先转为 `[0,1]`，视觉 processor 为 IDENTITY，再在 DINO 包装器内部做
一次 ImageNet mean/std 归一化。不重复使用旧 ResNet 的图像统计归一化。
状态和动作仍使用本示教集统计，保存并重载 processors。

采用 timm 发布的预训练模型，不是随机初始化，也不是 DINOv2：

- 模型：[timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)
- revision：`3bf4720a82ec2066db88137180ff1f83a675cef0`
- `model.safetensors` SHA256：`2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`
- 模型及其权重受 [DINOv3 许可](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
  约束，不因为本仓库代码为 MIT 就变成 MIT。权重不上传本仓库。

使用 timm 1.0.30 标准实现及 FP32 RoPE；不宣称与 Meta 原始实现逐位相同。
加载预训练文件前核验固定 SHA256，失败立即报错，没有随机权重降级。
完整 ACT checkpoint 包含冻结权重，严格恢复时不再次下载；缺失权重会报错。
序列化身份为 `act_dinov3`，不会伪装成 `resnet18`。

## 环境

基础模拟器支持 Python 3.10+；本 ML 路径的 LeRobot 0.6.1 要求 **Python 3.12+**。
新环境使用与硬件适配的 PyTorch；本次已在 torch 2.11.0、torchvision 0.26.0、
timm 1.0.30、Python 3.12、Apple MPS 下验证。不要为安装 DINO 擅自升级现有训练环境。

```bash
uv venv --python 3.12
uv pip install -e '.[act,dino]'
```

已有完整 ACT 环境但没有 timm 时，本次只增装了以下包，没有升级 torch：

```bash
uv pip install --python .venv/bin/python --no-deps timm==1.0.30
```

## 独立 200K 训练

[新配置](../configs/five-view-dinov3-frozen-300.yaml) 以原便携配置为基础，
只增加 `act.vision_encoder: dinov3` 并更换输出目录。其余配置保持原值。
复用原 300 条成功示教（目标 1–6 各 50 条），五路 320×320，不重采集。
原始图像共 1,015,605 张，全部核验存在；代表性解码 4,500 张，
不是声称完整解码百万图片。有效 BC 样本 195,621。

```bash
.venv/bin/python -u -B -m sim.act.train \
  --config configs/five-view-dinov3-frozen-300.yaml \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --out runs/act_model_five_view_dinov3_vits16_frozen_300_200k_20261015 \
  --steps 200000 --batch-size 4 --no-amp \
  --checkpoint-start-step 100000 --checkpoint-every 100000 \
  --checkpoint-extra-steps 50000 --keep-checkpoints 3 --no-trackio
```

选择空的新输出目录；**不加 `--resume`**。保存 50K、100K、200K，不保存 150K。
batch 4、MPS/FP32、AdamW lr `1e-5`、weight decay `1e-4`，
approach/descent 采样占比 0.35、sweep 0.65，总时间预算 192 小时；
没有新增 scheduler、梯度裁剪或 AMP。训练随机序列不声称与旧训练逐项相同。
在线提取冻结视觉特征，不创建数百 GB 的全量特征缓存。

训练入口在读数据、创建输出目录之前拒绝 ResNet/DINO 交叉续训。
推理仍按 checkpoint 中真实类型自动加载，旧 ResNet 保存点继续可读。
后续选择本模型推理时要显式选择新的 checkpoint，而不是旧工作台默认模型路径。
训练和工作台推理会竞争 MPS，不自动停止用户工作台任务。

## 验证范围

已用固定预训练权重完成真实五视角、batch 4、FP32/MPS 的 8 步训练；
训练后编码器 162 个状态张量与源权重逐项相等，适配层和动作头优化器状态已更新。
重载保存模型及 processors 后，用原示教的 4 个真实样本产生有限值 `4×25×4` 动作。
专项测试覆盖冻结/eval、空间接口、输入归一化、更新范围、离线严格恢复、缺权重拒绝、
预训练哈希和不同视觉编码器交叉续训拒绝。

这只证明接入和训练/保存/重载链路，不证明扫动成功率，也不表示 200K 已完成。
完整训练后的策略表现必须另做真实物理 rollout；不据 8 步 loss 宣称优于 ResNet。
旧正式 ACT/A*/Hybrid 结果不因本分支变化被改写或重跑。

实现参考了已有 ACT+DINOv3 的空间特征接法，但保留本项目原 ACT/VAE 核心：

- [SSatoya ACT+DINOv3](https://github.com/SSatoya/lerobot_custom_v0.4.3/blob/0e0bf6c0b4cdb6fbf49c4b19b8531a7b4766a194/src/lerobot/policies/act_with_dinov3/modeling_act.py)
- [DINO encoder wrapper](https://github.com/zaixiabalala/lerobot_with_DINOv3_backbone/blob/699d85ef95473f2c00eb5cd639a3d29d9f18b5fd/src/lerobot/policies/act/dino/dino_encoder.py)

未复制其本机绝对路径、CUDA 假设、硬编码前缀数量或其他训练方法差异。
