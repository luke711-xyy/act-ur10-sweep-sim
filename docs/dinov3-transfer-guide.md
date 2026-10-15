# 在另一台电脑从头训练 DINOv3 + ACT

本分支只提供源码、配置及操作说明。clone、安装依赖、下载权重或验证配置都不启动训练；
推理结构现为第四组 Hybrid A*→ACT，不是纯 ACT 落刷；见 [接管与参数](hybrid-dinov3-guide.md)。
只有最后明确执行 `python -m sim.act.train` 才会训练。本机训练已按用户要求停止在 122 步，
不自动恢复。目标机器应从头训练，不使用本机中断点或旧 ResNet 保存点。

## 1. 取得源码和环境

```bash
git clone --branch feat/act-dinov3-frozen --single-branch https://github.com/luke711-xyy/act-ur10-sweep-sim.git
cd act-ur10-sweep-sim
uv venv --python 3.12.13
```

NVIDIA 机器使用 Linux，或 Windows 的 WSL2 Linux 环境。当前数据集模块使用 `fcntl`，
不把原生 Windows Python 当作已支持的平台。Linux 下使用 `.venv/bin/python`；
不要把 Mac 的 `.venv` 拷过去，必须在目标机器重新安装依赖。

核心版本约束保存在 [constraints-dinov3.txt](../configs/constraints-dinov3.txt)：
torch 2.11.0、torchvision 0.26.0、LeRobot 0.6.1、timm 1.0.30。
先根据目标机器的显卡/驱动，通过 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)
安装对应的 CUDA 版 torch/torchvision；这里不猜测显卡、驱动或 CUDA wheel 版本。
随后安装本项目，并保留核心版本约束：

```bash
uv pip install --python .venv/bin/python -e '.[act,dino]' -c configs/constraints-dinov3.txt
.venv/bin/python -c "import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__); assert torch.cuda.is_available(), 'CUDA unavailable: check the installed torch wheel and GPU driver'; print(torch.cuda.get_device_name(0))"
```

上面的命令不创建模型或优化器、不训练。CUDA 检查失败时先处理环境，不启动训练。
这里的 CUDA 配置已做配置一致性检查；尚未在目标机器 GPU 上做运行验证。
Apple Silicon 另用原 MPS 配置 `configs/five-view-dinov3-frozen-300.yaml`。
Hybrid 全部依赖现随仓库携带；不需复制本机历史 `exports` 或试验脚本文件夹。
代码归档中的旧绝对路径仅为来源证据；运行配置使用仓库相对路径，视频默认写入 `exports/`。

## 2. 单独迁移完整示教数据

GitHub **不包含** 原始示教图片、数值数组、模型权重和运行日志。必须另行复制整个目录：

```text
runs/act_dataset_five_view_rudra_300_smooth320_v1/
  manifest.jsonl
  <全部 300 个 episode 目录及其原始内容>
```

不要只复制 manifest 或 `.npz`，也不要把评测视频当作训练图像。
保留 manifest 中引用的相对路径和五个视角；不重新生成、重排或筛选示教。
原数据为目标 1–6 各 50 条、五路 RGB 320×320，全部引用图像共 1,015,605 张。

在目标机器校验 manifest（不训练、不读大模型）：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
from sim.act.collect_dataset import validate_training_manifest
from sim.config import load_config
cfg = load_config('configs/five-view-dinov3-frozen-300-cuda.yaml')
root = Path(cfg.act.dataset_dir)
expected = '0bec71a7c2c3f06f7b6749f2b708a438c951c125c18ed0f48fe4036c8734431c'
assert hashlib.sha256((root / 'manifest.jsonl').read_bytes()).hexdigest() == expected
print(validate_training_manifest(root, episodes_per_target=50, camera_views=list(cfg.act.camera_views)))
PY
```

此检查只证明 manifest 身份与平衡/接口约束，不证明百万图像已经完整复制或解码。
数据复制工具还应完成文件完整性核验，确认没有遗漏或损坏，再由用户决定开始训练。

## 3. 固定预训练 DINOv3

首次构建新策略时，代码会下载并核验固定 revision 的模型文件；无需下载旧 ACT 权重。
可先单独下载（这一步不训练）：

```bash
.venv/bin/hf download timm/vit_small_patch16_dinov3.lvd1689m model.safetensors --revision 3bf4720a82ec2066db88137180ff1f83a675cef0
```

代码要求 SHA256 为 `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`；
下载失败或哈希不同会停止，不退回随机权重。权重遵守 DINOv3 许可，不上传本 GitHub 仓库。
离线目标机可另行迁移对应 Hugging Face 缓存，保持 repo/revision 结构；不硬编码本机缓存路径。

## 4. 用户在目标机器手动启动

[CUDA 完整配置](../configs/five-view-dinov3-frozen-300-cuda.yaml) 与 MPS 完整配置相比，
**只改 `act.device` 和模型输出目录**。五路输入、冻结权重身份、适配层和 ACT/VAE 不变。
batch 4、FP32/AMP 关闭、200K、192 小时时间预算及 50K/100K/200K 保存点均不变。

确认输出目录不存在或是新的空目录，且数据迁移与 CUDA 检查完成，再手动执行：

```bash
.venv/bin/python -u -B -m sim.act.train \
  --config configs/five-view-dinov3-frozen-300-cuda.yaml \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --out runs/act_model_five_view_dinov3_vits16_frozen_300_200k_cuda \
  --steps 200000 --batch-size 4 --no-amp \
  --checkpoint-start-step 100000 --checkpoint-every 100000 \
  --checkpoint-extra-steps 50000 --keep-checkpoints 3 --no-trackio
```

**不加 `--resume`**；训练图像适配层和全新 ACT/VAE，DINO 全程冻结。
AdamW lr `1e-5`、weight decay `1e-4`、approach/descent 采样比例 0.35、
contact_build/sweep 0.65、chunk 25、execute 5。没有新增 AMP、scheduler 或梯度裁剪。
换硬件不意味着随机序列或浮点结果逐位一致，也不预先保证显存容量或收敛效果。

如目标机显存不足，先停止并讨论设置；不要悄悄改 batch、分辨率或改成微调 DINO。
完成 200K 要以真实 checkpoint 和训练摘要为证，不能把本机的 122 步中断点当作完成模型。
