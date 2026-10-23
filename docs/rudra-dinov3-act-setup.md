# 另一台电脑：Rudra双视角DINOv3 ACT

## 获取代码

```bash
git clone --branch codex/rudra-dinov3-act-two-view --single-branch https://github.com/luke711-xyy/act-ur10-sweep-sim.git
cd act-ur10-sweep-sim
```

在新目录和独立Python环境配置，不覆盖正在运行的其他模型工作树。
参考DINO原始分支无需再合并或手动移植；本分支已包含接入代码及原生Rudra资产。

## 环境

ML路径使用Python3.12。先为目标机器安装匹配的PyTorch/torchvision，保留已有兼容环境，
然后安装本项目依赖；本分支不安装或修改其他训练任务的环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# 按目标机器驱动/平台安装配套PyTorch与torchvision，然后：
python -m pip install -e '.[act,dino,web,dev]'
python -m pip check
```

本机验证环境为torch2.11.0+cu128、torchvision0.26.0+cu128、LeRobot0.6.1、timm1.0.30。
这是来源记录，不是要求不同操作系统安装CUDA wheel。Linux无桌面渲染使用`MUJOCO_GL=egl`；
CUDA可用与MuJoCo渲染可用应分别检查。完整clone含机器人/夹爪资产，不只复制Python包。

## 生成本机配置

把300条示教完整复制到目标机器：schema-v5，目标1–6各50条成功专家记录，
`top,wrist`双路640×480。保留manifest、数组、图像及元数据，不改写源示教。

```bash
python scripts/configure_rudra_dinov3_act.py \
  --dataset /YOUR/DATASET \
  --out runs/rudra_dinov3_act_200k/model \
  --config-out configs/local-rudra-dinov3-act.yaml \
  --device cuda --batch-size 6 --cuda-memory-fraction 0.375 \
  --check-data
```

脚本生成主配置与对应`-scene.yaml`，把数据、输出、相机、原生场景引用和工作台推理根目录
接到当前电脑。`configs/local-*.yaml`已被Git忽略。已有配置拒绝覆盖；换一个文件名即可。
模型输出和本机配置放在仓库内部，以支持工作台受目录限制的只读日志监控；示教数据可位于外部磁盘。
不创建模型输出目录、不启动作业、不下载权重。`--check-data`使用既有只读manifest门禁；
未加该标志只检查manifest存在，不能声称数据已通过训练门禁。

0.375是来源RTX3070Ti8GiB与其他训练共享时的分配器预算；它不是全局显存预留或进程全部显存上限。
按目标机器可用显存调整。batch6是来源实测选择，不保证所有机器都能放下。
CPU/MPS配置可选`--device cpu`或`--device mps`，不得携带CUDA比例参数。
可用`--port`选择独立工作台端口，默认8774。

## 模型与数据契约

DINO：`timm/vit_small_patch16_dinov3.lvd1689m`，固定revision
`3bf4720a82ec2066db88137180ff1f83a675cef0`，原始预训练文件SHA256
`2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`。
首次初始化需要可用的同一权重缓存或按DINOv3许可取得权重；下载/访问失败明确报错，
不会退回随机权重。DINOv3权重遵循其原许可，未随本Git分支发布。

- 两路共享冻结ViT-S/16；384维、12层、6头，DINO始终eval/no_grad且不进优化器。
- RGB `[B,3,480,640]`、float32 `[0,1]`；视觉processor为IDENTITY，
  DINO内部只做一次ImageNet mean/std归一化。
- 去除模型声明的全部前缀tokens，保留30×40空间patch，2×2平均池化为384×15×20，
  经可训练1×1卷积投影到512维；每路300个tokens，再接原ACT/VAE。
- 36维状态为当前与约200ms前各18维；环境条件仅1维目标数量；动作是固定坐标系
  `[dx,dy,dz,dyaw]`，先反归一化再积分，不是关节值或绝对位姿。
- ACT：512维、8头、encoder4层、decoder1层、FFN3200、dropout0.1；
  VAE latent32、encoder4层；masked L1 + 10×VAE latent KL。
- 模型内部`temporal_ensemble_coeff=None`，外部调度器系数0.01；不据此关闭动作融合。
- 完整checkpoint含冻结DINO、ACT与处理器；严格离线重载，不再换编码器。

## 工作台和训练命令

以下是配置后的手动操作说明。配置脚本本身不会执行这些命令。

```bash
MUJOCO_GL=egl python -B -m sim.web.run --config configs/local-rudra-dinov3-act.yaml
# 本地浏览器：http://127.0.0.1:8774/

# 只有在用户授权训练、确认资源可用且模型输出目录为空后：
MUJOCO_GL=egl python -u -B -m sim.act.train \
  --config configs/local-rudra-dinov3-act.yaml --no-trackio
```

不要给已有模型目录直接启动一次无`--resume`的训练。断点续训显式传入包含
`training_state.pt`的checkpoint目录：

```bash
MUJOCO_GL=egl python -u -B -m sim.act.train \
  --config configs/local-rudra-dinov3-act.yaml \
  --resume /YOUR/ACT_MODEL/checkpoints/step_050000 --no-trackio
```

训练使用batch6、FP32、AMP关闭、AdamW lr1e-5、weight decay1e-4，接近/下降35%、扫动65%。
每5000步保存两份滚动断点，50000/100000/200000永久断点不占滚动名额。
SIGTERM/SIGINT会完成当前更新并保存中断点。恢复模型、优化器、学习率和累计步数；
当前保存格式未包含CUDA随机数或采样器剩余索引，不能宣称逐位连续复现。

工作台读取日志提供实时loss，推理地址跟随本模型目录的完整`latest_checkpoint.txt`，
不回退到旧五视角模型。训练与工作台应使用独立服务；本机曾因推理子进程与工作台共用过小
内存预算导致页面离线。若用systemd限制资源，预算需包含推理子进程；来源预算为3GiB软限额、
4GiB硬限额，应结合目标机器实际资源判断。因果回放不再累计保存完整浮点查询图像。

本地工作台不自动接管原电脑的私有远程站点或桥接身份。

## 有界验证

```bash
CUDA_VISIBLE_DEVICES='' python -B -m pytest -q \
  tests/test_rudra_dinov3_setup.py tests/test_act_dinov3.py \
  tests/test_act_cuda_budget.py tests/test_act_checkpoint_retention.py \
  tests/test_act_tracking.py tests/test_rudra_act_inference_defaults.py \
  tests/test_workbench_dinov3_rudra.py tests/test_workbench_training_progress.py \
  tests/test_act_query_observation_memory.py tests/test_act_replay.py
```

这些是配置、CPU小模型及回放契约测试，不是正式训练、完整分辨率显存基准或真实任务成功证明。
