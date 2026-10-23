# Rudra 双视角 DINOv3 ACT 训练

本副本基于用户指定 `codex/rudra-native-sweep-environment` 的提交
`632f33e69d4a44b83d910c6f8cc3f8741fd8c53f`。
工作台从独立的 Rudra diffusion 工作树复制；原工作树、8773端口、运行中的
Diffusion训练和配置均未修改。复制来源及文件哈希见 `rudra-act-copy-provenance.json`。

使用现有 ACT Python/CUDA 环境只读运行，没有安装或改动 diffusion 的环境。
使用 `configs/rudra-dinov3-act-cuda.yaml`，同一配置副本及所有日志位于
`runs/rudra_dinov3_act_200k_20261023/`。

## 训练契约

- 300条新成功专家记录，目标1–6各50条；207179帧，199679个有效监督帧。
- 原生 `top, wrist` 两路640×480；36维状态、1维目标条件、4维增量动作。
- 冻结原来的预训练 DINOv3 ViT-S/16，训练ACT投影层、Transformer、VAE及动作头。
  新ACT从头初始化，不续训旧ACT或PPO权重。预训练DINO权重校验保持不变。
- 保留FP32、AMP关闭、学习率1e-5、weight decay 1e-4、35%接近/65%扫动采样，
  25步动作块、5步执行及原来的归一化和有效帧掩码。统计由新数据重新计算。
- 正式batch=6，200000步；每5000步保存完整可续训checkpoint，保留最近两份滚动断点。
  50000、100000和200000步永久保留，不计入两份滚动断点的数量。
- ACT单进程PyTorch CUDA分配预算为整卡37.5%（3GiB），与CUDA上下文等非分配器开销区分。
  本进程预算不修改Diffusion分配器。系统服务另设3GiB内存高水位、4GiB硬上限及禁用本服务swap。
- 仅有边界明确的真实数据batch探测；无物理评测、无示教改写、无训练视频生成。

## Batch探测

batch4/6/8的分配器保留峰值分别1758/2206/2778MiB。
batch12在本进程3GiB预算内拒绝分配，没有挤占原训练。
选择batch6以保留866MiB分配器余量；短测吞吐不代表最终训练时长或模型质量。
完整记录见运行目录 `batch-selection.json`。

## 启动和查看

首次启动：`bash runs/rudra_dinov3_act_200k_20261023/launch_training.sh`。
脚本拒绝覆盖已有模型或重复启动；它不负责自动重试或自动从头重跑。
2026-10-23按用户要求在39684步正常保存并停止，恢复模型权重、AdamW状态、
学习率、累计训练时间和步数后继续，保存策略改为5000步滚动与三个永久里程碑。
显式断点续训：`bash runs/rudra_dinov3_act_200k_20261023/resume_training.sh`。
脚本拒绝重复启动，读取`model/latest_checkpoint.txt`，不会从头初始化ACT。
本次恢复未改变数据、模型结构、batch、精度、学习率及显存/内存限制。
旧训练器未保存CUDA随机数和采样器剩余索引，因此恢复后随机批次和dropout序列
不保证逐位一致；这不影响已保存模型、优化器累计状态和训练步数的恢复。
训练由 `ur10-rudra-dinov3-act-20261023.service` 保持，工作台是另一个独立服务，
因此重启本工作台不重启训练。

远程连接离线时，分别检查桥接和8774本机接口。工作台启动的推理子进程也计入
工作台服务的内存组；2026-10-23已将该服务预算持久化为MemoryHigh=3GiB、
MemoryMax=4GiB，避免原1.5GiB软限额同时限流页面和推理。启用因果回放的ACT
推理只保留查询观测的数值元数据，不再累计完整浮点相机图像；实际策略输入、
控制、动作融合和回放状态轨迹不变。训练服务独立，不随此工作台的恢复而重启。

- ACT工作台：http://127.0.0.1:8774/ ，含实时loss曲线和只读数据浏览。
- TensorBoard：http://127.0.0.1:6007/ ，含总loss、L1、KLD及进度。
- 终端：`bash runs/rudra_dinov3_act_200k_20261023/view_progress.sh`。
- 训练日志：运行目录 `train.log`；数值指标：`metrics.latest.json`。
- 模型：运行目录 `model/checkpoints/step_XXXXXX`，最新断点由
  `model/latest_checkpoint.txt` 指定；永久断点是`step_050000`、`step_100000`和`step_200000`。

原训练每100步输出一次，当前约一分钟；本工作台将日志新鲜度窗口设为180秒，
页面每3秒读取一次，不把正常日志间隔误报为卡住。
本次验证只证明环境、数据和训练启动工作正常，不等于200K已完成或模型任务成功。
