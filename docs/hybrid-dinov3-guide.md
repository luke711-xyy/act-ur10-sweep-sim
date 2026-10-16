# 第四组 Hybrid A*→ACT + 冻结 DINOv3

本分支以独立 Hybrid 快照 `eb887843c9849d4a72be116bb87c44519dc7ef74`
为控制流程来源，将之前的冻结 DINOv3 视觉实现接入它。**不取消 A*，不让 ACT 自己落刷。**
没有增加 YOLO、检测框、分割、occupancy、残差学习或新的动作监督。

## 控制归属和固定参数

| 阶段 | XY / 转角 | Z |
|---|---|---|
| 接近、落刷 | 原 A* 几何路径；最多一个候选，不做物理筛选 | 原接近/下降路径 |
| 实际 TCP 到桌面上方 1 mm 后寻找接触 | A* 继续接近/调整 XY | 腕力导纳 + 既有有界实际 Z 跟随补偿 |
| 腕力稳定后扫动 | ACT；首个有效动作起 2 秒缓启动，之后恢复原生 ACT | 同一导纳/补偿状态持续接管；不执行 ACT 预测 Z |

完整数值契约在 `sim/act/hybrid_trial4/control_profile.json`：

- 目标力 1 N，总 Z 修正（导纳、卸载与跟随补偿合成）±2 mm。
- 寻找 M/B/K = 10/400/10，维持 10/800/10；切换保留位移和速度状态。
- 腕力 ≥0.5 N 持续 0.1 s 进入稳定阶段，0.8–1.2 N 持续 0.2 s 才允许 ACT 扫动。
- 交接 2 s 内 XY 加速度 50 mm/s²、XY 最高 80 mm/s；转角最高 0.35 rad/s、
  加速度 0.5 rad/s²。交接期结束后不继续加速度限幅。
- 上抬 20 mm/s、下探 100 mm/s；实际 Z 跟随 gain 0.5、补偿分量 ±0.2 mm，
  合成仍受 ±2 mm 约束。进程内 IK 提前收敛阈值 1e-6。
- 100 Hz 控制、25 Hz 动作；A* 前段、寻找接触、ACT 共用 2000 动作帧预算。
  原首次入区后 2 秒终止窗口跨阶段连续，不在交接时重新计时。
- 只用现有补偿/滤波腕部 Fz 作力反馈，5 Hz 滤波及 20 N / 3 步安全保护不变。
  腕力不是桌面独占压力；低腕力不能证明持续贴桌。
- 每次调用只有一次几何机会、最多一次物理执行。无计划直接失败；未知异常向外抛出，
  不择优、不重试、不换 seed、不在异常后发送 hold 或重查询兜底。

### 接线修复与历史结果边界

14 个内部依赖的函数/类 AST，以及 4 个正式配置/驱动适配函数，与独立快照保持一致。
迁移只改其模块级路径；旧试验 main/历史批次保护函数不由新入口调用。

本次审查额外发现，正式入口复制推理函数 globals 后，旧工厂的临时 Z 钩子可能丢失。
新组装入口 `build_driver()` 显式把**同一个第四组 Z 限幅/跟随函数**接回最终 namespace，
并拒绝 NaN/Inf 动作进入原生的无效动作 hold 分支。没有重设计导纳参数或接管算法。
这两项属于接线/异常保护修复，不能把新版本与旧正式结果的差异全部归因于视觉编码器。
旧快照、结果、模型保持原样；未回写或重新评测旧批次。

## 显式接触导纳调参

本地接触/扫动诊断可在配置中设置 `act.hybrid_admittance`，例如：

```yaml
act:
  hybrid_admittance:
    seek_MBK: [10.0, 400.0, 10.0]
    maintain_MBK: [10.0, 800.0, 10.0]
    force_compliance_m_per_N: 0.00005
controller:
  overforce_relief:
    activation_force: 1.5
    release_force: 1.1
    upward_rate: 0.0005
    return_rate: 0.0005
    max_offset: 0.00025
```

此设置在创建导纳环之前生效，离散稳定性检查通过后才创建物理环境。
寻找、稳定和 ACT 扫动复用同一个导纳状态，阶段切换保留位移和速度。
省略 `act.hybrid_admittance` 时仍使用上述归档 M/B/K。额外卸载参数沿用
`controller.overforce_relief` 配置入口；示例采用较早、较平缓的卸载与恢复。
显式配置的执行记录 `hybrid_control_profile` 写入实际增益和卸载参数，
媒体恢复也校验同一配置。1 N 目标力、±2 mm 总 Z 限制、
5 Hz 腕力滤波、20 N / 3 步保护以及 A* / ACT 的 XY 执行方式保留。

可选 `force_compliance_m_per_N` 只在 `maintain` 阶段增加弹性腕力反馈，
不重置 M/B/K 环的积分状态。示例为 0.05 mm/N，额外位移限于 ±0.5 mm，
最终 Z 仍受原有 ±2 mm 总边界和速度限制约束。力上升时更早卸载，回落时
释放该项以恢复贴桌；省略或设置为 0 则关闭。支持范围为 0–0.3 mm/N。
真实执行 profile 和媒体恢复校验都包含这一项，不能将其标作未修改的归档控制器。

如接管 2 秒后 ACT 目标转向突变仍引起贴桌冲击，可显式设置：

```yaml
act:
  hybrid_sweep_smoothing:
    xy_acceleration_m_s2: 0.1
    yaw_acceleration_rad_s2: 0.5
```

此选项让接管后的 XY/yaw 继续跟随 ACT 目标，同时限制速度为 0.08 m/s、0.35 rad/s，
使用上述连续加速度限制。原始 2 秒接管段保留 0.05 m/s²、0.5 rad/s² 的限制。
扫动会减速，实际执行轨迹会改变；ACT 仍决定目标，Z 仍属于导纳。
这时 profile 明确记录 `sweep_smoothing` 和持续限加速度模式，逐个物理步审计，
不能称为 2 秒后完全恢复原生 ACT。省略该选项则保留归档行为。

调参应分别报告落刷/扫动的力峰、未滤波腕力、低力比例与贴桌诊断。
刷头与槽壁的碰撞峰值不能直接当成导纳跟踪误差；小幅扫动改善也不能替代完整任务评测。

## DINO 与训练准备

五路 320×320 RGB → 同一个冻结 DINOv3 ViT-S/16 → 384×10×10 空间图 →
可训练 1×1 投影到 512 通道 → 原 ACT Transformer/VAE。ACT 输入、chunk 25/execute 5 不变。
DINO 权重固定 revision/哈希、全程 frozen/eval；没有使用随机编码器降级。
详见 [视觉实现](dinov3-act-guide.md) 与 [另一台电脑训练准备](dinov3-transfer-guide.md)。

原始示教数据可以复用，但必须在目标机器另行复制完整数据集。
ACT/适配层需要重新训练，旧 ResNet checkpoint 不能直接变成 DINO 策略。
完整 DINO ACT checkpoint 必须包含模型及 processors；不使用历史 ResNet 100K 模型做 DINO 推理。

## 可跨机的入口（仅供之后手动执行）

从克隆的仓库根目录运行，安装 Python 3.12 和对应设备依赖。CUDA 使用
`configs/five-view-dinov3-frozen-300-cuda.yaml`，MPS 使用无 `-cuda` 后缀的配置。
数据和模型路径均在 `runs/` 下，可用配置或 CLI 显式改到目标机器路径；视频默认 `exports/`。
运行时所有 14 个 Hybrid 依赖均在仓库内，不读取原电脑的历史 exports/批测目录。
`hybrid_trial4_snapshot/` 中的旧绝对路径仅作归档证据，不是可运行配置。

训练完成且用户决定测试后，才手动执行，例如：

```bash
.venv/bin/python -B -m sim.act.evaluate \
  --config configs/five-view-dinov3-frozen-300-cuda.yaml \
  --model runs/act_model_five_view_dinov3_vits16_frozen_300_200k_cuda/checkpoints/step_100000 \
  --target-count 1 --seed 2120000 --preview --record-replay
```

CLI 及工作台根据 checkpoint 的真实 `act_dinov3` 类型进入 Hybrid，即使工作台传入旧默认配置。
新 DINO 配置还显式设置 `act.inference_mode: hybrid_trial4`，缺失模型或误选 ResNet 会提前报错，
不会创建随机 ACT 后开始落刷。回放保存的是实际生效的第四组配置，不是旧工作台配置。

如需工作台，在用户决定启动时使用：

```bash
.venv/bin/python -B -m sim.web.run --config configs/five-view-dinov3-frozen-300-cuda.yaml
```

该配置的自动模型选择只跟随其 `act.model_dir/latest_checkpoint.txt` 中已完整发布的 DINO 保存点；
不会静默回退到旧 ResNet。手动输入完整 checkpoint 路径仍优先。
旧 ResNet 的默认模型选择与原生推理入口保留兼容。
通用批测入口保留，但 DINO 模型标为 `hybrid_dinov3_*`，不标作纯 ACT。
无几何计划保留真实失败和初始姿态身份，物理执行为 0、没有视频；不能因缺视频而重跑。
已保存轨迹的媒体恢复使用实际生效的第四组配置，不再次执行策略。
这里只验证了模拟边界的记录兼容，没有准备或启动真实批次，也没有恢复任何历史批次。

## 本次验证边界

只做代码/配置检查、控制器数值单测、源代码 AST 对照、模拟边界调度与模型加载/视觉前向检查。
未启动训练、优化器更新、ACT 物理推理、120 组评测或工作台服务。
这些检查不证明新 DINO 策略的扫动成功率、接触连续性或力峰改善；目标设备 CUDA 尚未运行验证。
