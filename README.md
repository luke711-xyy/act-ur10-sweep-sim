# Rudra 原生桌面扫零件仿真

分支：`codex/rudra-native-sweep-environment`。在 Ubuntu 交接代码上保留 Rudra
原生 UR10/PincOpen、木桌、腕部相机及支架，接入收集槽、黑色零件、夹持刷板、
原布局生成规则及前上方相机；双路 top＋wrist 示教保存原生 640×480。
第三方场景源码、模型资产、来源提交和许可证随分支提供。

使用已安装本项目依赖的 Python 环境启动工作台：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 python -m sim.web.run \
  --rudra-scene --scene-config configs/rudra-sweep-scene.yaml \
  --host 127.0.0.1 --port 8780
```

参见[场景、坐标与机位](docs/rudra-sweep-preview.md)、
[导纳检查与专家示教入口](docs/rudra-six-demo-collection.md)、
[300条原生尺寸示教与失败补足](docs/rudra-seed-matched-300.md)。
当前工作台用于新场景预览；Diffusion/ACT 策略推理尚未接入新基座和夹爪。
示教载荷、权重、视频、运行日志和本机 Python/CUDA 环境不上传。
开发回归中的浏览器脚本检查需要 PATH 中可找到 Node.js。

以下保留分支起点的 Ubuntu 交接说明；本分支的 Rudra 入口与验证范围以上述文档为准。

# Ubuntu 模型训练与推理交接：UR10 / MuJoCo Sweep

本分支 `feat/ubuntu-model-handoff` 是 2026-10-15 的**代码迁移与 AI 交接分支**，基于
`9102680d8e018b225ab34d36422572cd4b547166`。目标是让另一台 Ubuntu 电脑复用现有
示教、仿真、控制、回放和工作台，再接入 ACT 之外的模型。**其他模型尚未实现；工作台现有训练/推理按钮仍接 ACT。**
不要把迁移代码、模型加载测试或接口测试称为新模型训练完成、策略成功或 Ubuntu 实机验证通过。

这里包含代码、机器人资产及其许可证、配置、测试、文档和第四组 Hybrid 控制运行依赖。
示教数据由用户通过 U 盘另行传输；权重、视频、日志、正式评测结果及本机虚拟环境不上传。
Mac 历史说明保留在 [README.mac-act-history.md](README.mac-act-history.md)，分支与回滚点见
[分支导航](docs/branch-guide.md)。历史路径不是 Ubuntu 部署指令。

## 0. 接手 AI 先做什么

1. 先读本 README，再读 `sim/act/interface.py`、`dataset.py`、`policy.py`、`train.py`、
   `evaluate.py` 和 `sim/web/jobs.py`；不要先重写仿真器或示教格式。
2. 向用户确认 Ubuntu/Python 版本、GPU 型号/驱动、数据挂载路径、空闲磁盘、要训练的模型。
   选择该模型后再设计适配器，未获授权不下载大模型、不启动长训练或批测。
3. 完整复制示教根目录到新位置，核验格式和目标分布。现有格式不是 LeRobot Hub Parquet；
   `lerobot` 在此用于 ACT 实现，不表示原始数据能直接交给任意 LeRobot 命令。
4. 先验证机器人资产、渲染、数据读取和工作台，再做 ACT 接口参考检查；新模型接入后依次验证
   离线预测、小规模实际推理及正式评测。每一阶段分别报告证据。
5. 不改传入示教和历史正式结果；不同模型写独立输出目录。不要修改旧指纹来绕过验证，
   不把成功示教替代失败评测，不把“数量相同”当成严格任务成功。

## 1. 代码地图与复用边界

```text
sim/
  config.py, environments/         完整 YAML、布局、MuJoCo 物理环境与 IK
  model/, model/assets/            机器人/刷子/物体资产、场景构建
  controllers/                    导纳、混合控制与安全状态机
  planners/                       几何规划、路径与 A* 相关规划逻辑
  act/
    interface.py                  观测、动作、归一化维度契约
    dataset.py, collect_dataset.py 示教读写、采集协议与训练门禁
    expert.py                     示教选件、coverage A*、落刷与专家路线
    policy.py, train.py            ACT 模型包装、训练与保存
    evaluate.py                   观测到动作、实时调度与物理执行
  web/
    run.py, app.py, jobs.py        工作台入口、HTTP 路由、子进程任务
    workbench.py                  数据浏览、回放、派生示教操作
scripts/
  ubuntu_setup.py                 生成本机配置、只读核验 U 盘示教
  evaluate_astar_baseline.py      单次机会的纯 A* 批评测
  evaluate_hybrid_paired_batch.py 第四组 A*→ACT 配对批评测入口
  hybrid_live_view.py             可选外部热力图/已验证视频展示，不调用推理
experiments/hybrid_trial4/
  runtime.py, vendor/             纳入仓库的第四组控制依赖，见其 README
configs/                         完整配置；ubuntu-five-view.yaml 为模板
tests/, docs/                    契约测试、工作台与实验文档
```

| 可复用部分 | 新模型必须适配的部分 |
| --- | --- |
| 机器人/场景、物理、腕部力传感、安全保护、IK | 模型工厂、依赖、权重格式、预处理/后处理 |
| 现有示教文件、相机和低维状态契约 | 训练窗口、损失、优化器、padding/valid 掩码 |
| A*、导纳、动作执行、记录/回放 | 单步或动作块调度、reset、预测延迟约束 |
| 工作台页面、数据查看、视频导出及任务框架 | 模型类型选择、训练/推理命令与加载路由 |
| 数量/严格成功指标和配对布局机制 | 新模型正式评测入口与独立冻结元数据 |

不需要 ROS、MoveIt 或 Isaac Sim 才能运行当前流程。当前五视角默认是 Rudra UR10 CB3；
另一 UR10e 资产是比较配置，迁移时不要顺手换机器人或刷子物理参数。

## 2. Ubuntu 环境与渲染

建议先采用 Ubuntu 24.04 + Python 3.12；项目声明 Python ≥3.10。这是部署建议，**本分支尚未在目标
Ubuntu/GPU 上实测**。从完整 Git clone 使用 editable 安装，资产依赖仓库目录结构，不要只复制 Python 包。

```bash
git clone --branch feat/ubuntu-model-handoff https://github.com/luke711-xyy/act-ur10-sweep-sim.git
cd act-ur10-sweep-sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[web,dev]'
# 需要现有 ACT 参考流程时再安装；新模型的重依赖可放独立虚拟环境
python -m pip install -e '.[act]'
```

GPU 训练先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选适合目标驱动的
CPU/CUDA/ROCm wheel，Torch 与 torchvision 必须配套，并满足本仓库声明的版本范围。
不要把 Mac MPS 配置直接带过去，也不要猜 CUDA 下载链接。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip check
```

迁移来源环境曾使用 MuJoCo 3.14.0、NumPy 2.2.6、Torch 2.11.0、torchvision 0.26.0、
LeRobot 0.6.1；仅是来源版本记录，**不是 Ubuntu 已验证锁文件**。依赖约束以 `pyproject.toml` 为准。
GPU 加速主要用于网络；现有 MuJoCo 物理不因 `act.device=cuda` 自动改成 GPU 物理。

Ubuntu 渲染需要匹配的 OpenGL 驱动/库。下列是常见库示例，先检查系统已有包和显卡驱动再安装：

```bash
sudo apt-get install libgl1 libegl1 libosmesa6 libglfw3
# 无桌面 GPU 渲染：启动 Python 之前设置
export MUJOCO_GL=egl
# 没有可用 EGL 时，另一个新终端/进程可尝试 CPU 渲染：
# export MUJOCO_GL=osmesa
# 有桌面窗口则可使用 glfw
# 可选历史 Cartesian 健康检查（不是 UR10 ACT 成功证明）
python -m sim.smoke_test
```

Linux 的 EGL/OSMesa/GLFW 选择见 [MuJoCo 官方 GL 后端代码](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rendering/classic/gl_context.py)；
变量必须在导入 MuJoCo 前设置。Torch 的 CUDA 可用不等于 RGB 渲染可用。
MuJoCo Python wheel 包含其运行库，不需要沿用旧 `mujoco-py` 安装流程；
Mac 的 `mjpython` passive-viewer 要求也不是 Ubuntu 指令，见
[MuJoCo Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)。
生成本机配置后，可用 `python -m sim.act.camera_preview --config configs/local-ubuntu.yaml`
检查UR10资产和离屏渲染；它生成四个预览相机，不代替五个正式策略视角的核验。

## 3. U 盘示教：保持格式和相对路径

复制**完整数据根目录**：`manifest.jsonl`、每条 episode 元数据、NPZ、所有视角帧图、
采集配置/协议、trace/审计文件及已有 split。仅复制 NPZ 或仅复制图片均不足。
不要改写 manifest 的相对路径，不要把 USB 挂载路径写入每条历史记录。
如需复用 ACT checkpoint，另行复制完整 checkpoint 目录，包括模型、配置、统计与 pre/postprocessor；
续训还需要对应训练状态，不能只复制一个权重文件。

```bash
# 用新目的目录，避免覆盖已有数据；替换下面两个示例路径
mkdir -p runs
rsync -a /media/USER/USB/act_dataset/ runs/act_dataset_five_view_rudra_300_smooth320_v1/
python -B scripts/ubuntu_setup.py check-data \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1
# 确认规模和耗时后才做全部载荷/图片解码检查（仍为只读）
python -B scripts/ubuntu_setup.py check-data \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --all-episodes --check-images
```

默认核验全部 manifest/元数据，但只检查前三条 NPZ/图片引用，不扫描整库图片；
报告会写出检查范围。全量图片检查也**不等于 split 泄漏、重复布局或所有 trace 哈希的完整审计**。
训练前还要核验成功专家的数量、采集协议、来源指纹、动作有效掩码以及 train/validation 划分。
正式门禁可额外使用 `check-data --formal-config configs/local-ubuntu.yaml`；
当前五视角正式训练配置要求目标 1–6 各 50 条、共 300 条成功专家示教。
个别旧 help 文本中的 120/每目标20 不是当前门禁；不要删除门禁让不足或不同协议的数据混入。

## 4. 数据与动作契约：换模型前必须掌握

当前 schema_version=5。以 `sim/act/interface.py` 和 `dataset.py` 为准：

| 字段 | 当前形状/含义 |
| --- | --- |
| `state` | float32 `(T,36)`；当前 18 维 + 200 ms 前历史 18 维 |
| 每组 18 维 | 六关节 + TCP XYZ 3 + yaw 的 sin/cos 2 + 腕部 wrench 6 + contact latch 1 |
| `environment_state` | 当前 float32 `(T,1)`，仅目标数；兼容旧 `(T,3)` 时目标在列1，旧列0/2是总数/收集数 |
| `action` | float32 `(T,4)`：世界坐标固定参考系的指令增量 `Δx,Δy,Δz,Δyaw`，单位 m/rad |
| 时间 | `t` 单调递增；动作 25 Hz，40 ms/帧 |
| `action_valid` | 每帧有效监督掩码；不能把无效接近/静置标签直接当扫动标签 |
| RGB | 五路 320×320；文件 HWC uint8，reader 转 CHW uint8，训练再转 float/255 |

五个 camera 名：`corner_xminus_yminus`、`corner_xminus_yplus`、
`corner_xplus_yminus`、`corner_xplus_yplus`、`wrist`。
**必须核对实际配置中的相机键与顺序**，不能仅凭“5张图”猜映射；
若更换分辨率，现有 reader 不会替你自动 resize。
不足200ms历史时重复首帧；关闭历史时重复当前帧，既有接口仍保留36维，不能默默改状态结构。

动作不是关节角、实际 TCP 轨迹或任意模型的绝对 pose。现有推理以查询时刻的指令参考锚定动作块，
不能从执行时的实际 pose 重新解释同一动作块。ACT chunk=25、execute_steps=5；
单步模型需要自己的调度适配，不要重复五次同一输出冒充动作块。

接触 latch 后，Z 的执行由导纳管理，监督中相应 `dz` 为0；Hybrid 中 ACT 预测 Z 不执行，
只保留审计，ACT 负责 XY/yaw。原始 state 的 wrench 是传感器反馈转世界系，
补偿/滤波的接触 Fz 是另外的控制量；二者不能随意当成同一个数。
收集数、物体真值 pose、专家选件信息和成功判定不能作为新策略输入偷渡。

新模型数据适配必须保持：动作单位/坐标系、相机映射、历史语义、有效和 padding 掩码、
只用训练集拟合 normalization。派生自同一条示教的样本放同一 split；
别仅用随机帧划分导致数据泄漏。元数据门禁通过并不证明载荷和 split 全部正确。

## 5. 本机配置与 ACT 参考全流程

配置加载是读取**完整 YAML**，不是自动把短文件 merge 到默认配置。
迁移工具从现有完整五视角配置生成副本，只改设备、数据/模型/视频路径和关闭外部追踪；
物理、控制、相机、任务、训练结构保持原值。输出必须是新文件，不覆盖冻结配置。

```bash
python -B scripts/ubuntu_setup.py config \
  --base configs/five-view-fine-teached-300.yaml \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --model-dir runs/ubuntu_act_reference --device cuda \
  --out configs/local-ubuntu.yaml
# 无兼容 GPU 时用 --device cpu；不支持 mps/auto 静默回退
```

仓库已有 `configs/ubuntu-five-view.yaml` CUDA 模板，可作为参考。
训练、evaluate、web 命令本身没有统一的 `--device` 或 `--set` 参数，设备要写进完整配置。
`web.inference_model_root` 在本分支支持选择本机模型目录的最新完整 checkpoint；
工作台空模型输入没有可用完整 checkpoint 时会报错，不会悄悄使用随机权重。

下列是**迁移后的参考命令，不是自动启动授权**。先确认数据门禁、设备和显存；原 batch=4、FP32。
不在 Mac 恢复已停止的训练，103243 中断点不是200K，历史 step_100000 不改写。

```bash
# 未来得到用户训练授权后，Ubuntu 上的 ACT 参考训练
python -m sim.act.train --config configs/local-ubuntu.yaml \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --out runs/ubuntu_act_reference --steps 200000 --batch-size 4 --no-amp --no-trackio

# 仅在对应 checkpoint 已完整生成或转移后做实际推理；--model 必须显式给
python -m sim.act.evaluate --config configs/local-ubuntu.yaml \
  --model runs/ubuntu_act_reference/checkpoints/step_100000 \
  --seed 3000000 --target-count 1 --record-replay \
  --replay-root runs/workbench_previews --video-root runs/inference_videos

python -m sim.web.run --config configs/local-ubuntu.yaml --host 127.0.0.1 --port 8765
```

浏览器访问 `http://127.0.0.1:8765`；远程访问优先 SSH tunnel，不把无认证控制台直接公开到网络。
工作台可以查看传入示教、保存状态回放和视频；训练/推理仍使用现有 ACT 子进程，新增模型需改路由。
数据、checkpoint、回放和视频目录应写入本机配置；不要沿用 `/Users/...` 或 Mac 外部 exports 路径。

续训用 `--resume` 指向完整 checkpoint，并另设输出保护历史；`--steps` 是总步数上限，
不是额外步数。独立 evaluate 不指定模型可能创建未训练策略，所以这里明确要求 `--model`。
`--preview` 允许预测延迟暂停仿真，不是严格实时调度评测。

## 6. ACT 之外的模型如何接入

建议小步复用，而不是先把 ACT 类名全部替换：

1. 单独实现模型模块和配置，明确图像/低维/目标数输入、时间窗和输出。
   需要语言或其他模态时先与用户确认：现有示教没有凭空提供这些标签。
2. 数据适配读取原格式，按新模型组成 batch/窗口，保留 valid/padding 掩码；测试预处理和归一化。
3. 增加训练入口、模型 factory、loss/optimizer 和完整 checkpoint save/load。
   不改物理控制代码迁就训练 API。每个模型独立依赖环境和输出目录。
4. 推理适配器输出物理单位的 `(K,4)` 动作，并提供 reset/query 语义；
   单步/扩散动作块/绝对 pose 各需明确转换、缓存和延迟预算，不套错 ACT scheduler。
5. 在工作台配置、jobs 和模型选择路由中注册模型类型；测试新命令与完整 checkpoint 检查。
   先离线前向，再目标1/6短测试，再冻结正式布局/配置/来源；真实失败全部记录。

可直接复用环境、腕部力反馈、导纳、安全限制、IK、记录/状态回放和指标计算。
但策略 success、数量误差、接触稳定性、力和实际 Z 跟随误差要分别报告。
静止1N或收集到目标数不能证明扫动持续贴桌。

## 7. 第四组 Hybrid 运行依赖与评测边界

Mac 上此前散落在 exports 的14份控制 Python 依赖已纳入
[experiments/hybrid_trial4](experiments/hybrid_trial4/README.md)，路径改成仓库相对位置，
不带历史物理轨迹、权重或数据。查看其来源哈希与迁移哈希；不要运行 vendor 内旧实验 main，
它们有意保留历史实验结构，不是可单独部署的批测入口。

```bash
python -c "from experiments.hybrid_trial4.runtime import load_runtime; print(load_runtime().TRIAL_SPEC)"
```

第四组固定：1N目标力、导纳+实际Z跟随补偿**合计±2mm**，寻找 M/B/K=10/400/10，
维持=10/800/10；实际距桌面1mm启用导纳，腕力0.5N持续0.1s转稳定，
0.8–1.2N持续0.2s才交 ACT XY/yaw。Z全程由腕部力导纳和有界实际Z跟随补偿管理，
不执行 ACT 预测Z、不看桌面力真值。
首个有效 ACT 动作后2秒，XY加速度50mm/s²；之后精确恢复原生 ACT，不永久限扫动。
XY最高80mm/s，转角0.35rad/s及交接0.5rad/s²；正常上抬20mm/s、下探100mm/s，
跟随gain0.5/±0.2mm、进程内IK精度1e-6、5Hz力滤波及现有安全保护保留。

新入口支持 `--exports-root`、`--act-batch`，供 Ubuntu 本机**重新建立并正常完成的 ACT 基线**
生成新的 Hybrid 批次使用；所有阶段共享2000个25Hz动作帧，入区后两秒终止连续跨阶段。
每布局一次几何机会、最多一次物理执行；无方案也记失败，不换seed、筛成功、重试或增加恢复路线。
预测、调度、执行异常停止保留现场，需人决定。

**不能把 Mac 历史批次复制到 Ubuntu 后重写 metadata/seal/mtime 来强行恢复。**
正式冻结记录仍要求实际配置、数据/checkpoint和依赖指纹一致；现有迁移仅取消旧清理审查文件的
机器路径依赖，改冻结完整完成基线的计划/配置/ledger/summary/status/metadata。
新模型正式入口也须有同等级别保护。Hybrid 不是纯 ACT；
曾参与调参的布局不能宣称全新未调参测试集，不作单因素因果或显著性宣称。

可选 `scripts/hybrid_live_view.py` 是外部展示工具，不导入/调用评测器、不改正式推理。
它生成最终数量密度PNG/SVG/CSV，并从已核验视频保留查看副本；原视频已删除时不重新执行策略补视频。
**正常整批完成且执行进程退出后，该工具会按自身receipt核验并永久删除自己的视频副本。**
仅在用户同意这一“整批保留、结束清理”策略后启动；默认不自动开启，也不恢复定时任务。
新批次先创建 `BATCH/live_view`，再运行 `python -B scripts/hybrid_live_view.py --root /absolute/BATCH`；
展示目录已有journal时，只可用 `--resume-display-only` 恢复展示，不能恢复物理评测。

## 8. 已验证范围与目标机验收

本分支在来源 Mac 的既有环境做了有界专项测试，未启动训练、完整120批测或新模型实验：

当前专项结果：136项，132通过、4项专有历史证据测试跳过、0失败/错误；不是全套测试。
另做工作台回归35项全部通过；详情见 [验证范围](docs/ubuntu-handoff-validation.md)。

```bash
python -B -m pytest tests/test_ubuntu_handoff.py tests/test_workbench_ubuntu.py \
  tests/test_hybrid_paired_batch.py tests/test_workbench_ordinary_50k.py \
  tests/test_admittance.py tests/test_scene_and_config.py -q
```

记录层原始历史恢复的4个样本测试需要本机专有轨迹，默认明确跳过；
只有用户提供真实只读历史证据时才设置 `SWEEPSIM_HISTORICAL_HYBRID_BATCH`。
这不是全套 pytest 通过，也不是 Ubuntu 驱动/CUDA/EGL/策略验证。
勿直接无审查跑所有测试：部分历史测试会创建训练/物理工作负载。

目标机验收顺序：依赖检查 → 资产/场景 → RGB 渲染 → 数据检查 → 工作台数据/回放
→ checkpoint 完整加载/离线预测 → 短实际推理与控制审计 → 用户授权训练/正式评测。

常见问题：黑图先检查 GL 后端和驱动；维度相同不证明示教/归一化相同；
找不到模型先检查 `web.inference_model_root` 和完整 checkpoint。
旧记录中 MP4 绝对路径在 Ubuntu 缺文件不是重新推理的理由：
使用独立映射/新回放目录从保存状态生成媒体，保留旧正式 ledger 不变。
