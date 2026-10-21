# Linux / CUDA Diffusion Policy 训练提示词

把下方分隔线之后的内容整体发给 Linux 电脑上的 Codex。本文件是交接指令，
不是“Linux 已验证可运行”的报告。GitHub 只包含代码，不包含示教、模型或环境。

---

请直接在这台 Linux 电脑完成下面的配置、必要的最小 CUDA 适配、测试和真实训练启动，不要只给方案。
有缺数据、无可用GPU或依赖不兼容等硬阻碍时，停在该处并说明需要我提供什么；不能伪造通过。

## 1. 代码、隔离与目标

- 仓库：https://github.com/luke711-xyy/act-ur10-sweep-sim
- 源分支：`codex/diffusion-policy-mvp`，必须包含代码提交
  `2cd6b9c51d0d355d531b7d02384ce9a2975a30eb`（实时 loss 面板）。
- 从 GitHub 获取该分支，先阅读 AGENTS.md（如有）、
  `docs/diffusion-policy-mvp.md`、`docs/diffusion-policy-mvp-status.md`、
  `docs/superpowers/plans/2026-10-21-diffusion-policy-mvp.md`、
  `sim/diffusion/vendor/PROVENANCE.md` 和相关代码。状态文档中的 Mac 步数是历史快照。
- 在独立本地分支 `codex/diffusion-policy-linux-cuda` 实施 Linux 适配。
  已有未提交工作必须保留，必要时使用独立工作树；不 reset、不覆盖其他实验。
- 不连接、停止、迁移或修改 Mac 上正在运行的训练，不擅自推送 Linux 改动。
- 本次目标是单GPU CUDA 训练与原工作台可见的真实进度；不引入 WAM、任务感知头、RL、
  新任务或新示教，不换成另一套 LeRobot DP/Hydra 项目，不继承 ACT 权重。

## 2. 先检查硬件、环境和磁盘

读取 NVIDIA GPU 型号/UUID、驱动、可用显存、其他GPU作业、CPU、RAM、可用磁盘和 Python。
不能用 CPU 冒充 CUDA 成功，不终止其他进程来抢显存；无适用GPU则先报告。

创建独立 Python 3.12 环境，不改系统 Python 或其他 ACT 环境。
以 `requirements-diffusion.in` 为依赖意图，以现有锁定文件为 Mac 基线参考：
`torch==2.11.0`、`torchvision==0.26.0`、`einops==0.8.2`、
`diffusers==0.35.1`、`mujoco==3.14.0`，保留项目所需 web/dev 依赖。
依据 PyTorch 官方安装说明选择匹配 GPU/驱动的 CUDA 构建，实际检查 CUDA 可用与真实运算。
不要盲目复制 Mac 的传递依赖锁，生成独立 Linux/CUDA 锁定文件，记录 wheel 来源、版本和环境。
若这些固定版本没有兼容构建，说明原因后问我，不偷偷升级或退到 CPU。

重要：现有 `ResourceGuard` 扫描整个项目目录。把 CUDA 虚拟环境和原始示教放项目目录之外，
不要把大型CUDA依赖/数据复制进工作树导致20GiB预算误触发。
保留新实验20GiB上限、磁盘余量25GiB底线及下一份恢复点预留；不删除旧数据或其他实验。

## 3. 找到并核验原始示教

GitHub 的 `runs/`、数据集、检查点不随代码上传。先在已知项目/数据盘检查是否已有数据；
缺少则停下告诉我需要复制的目录，不采新数据、不换公开数据或玩具数据启动全量。

Mac 源目录（只用于说明需要搬什么，不能当作 Linux 可用路径）：

`/Users/huazhi_luke/.codex/worktrees/act-five-view/runs/act_dataset_five_view_rudra_300_smooth320_v1`

需要完整保留 `manifest.jsonl`、所有引用的数组、五视角图片和元数据的相对结构，原数据只读：

- 300条成功专家示教、203121帧、195621有效训练窗口；N=1..6各50条、300个独立布局。
- manifest＋数组的数据身份应为
  `bed662c46eda74ad30edd7c68f7e20275f65def25180296c4040a1df06250dc4`。
- 此身份不包含全部RGB文件内容；还需检查引用文件完整性、RGB尺寸和已有传输校验清单，
  不能仅凭身份匹配宣称像素已全部校验。缺少图像checksum证据要明确说明。
- 配置 Linux 本机数据路径，不能改写原 manifest、重编码图像或删改样本来“通过检查”。
- 所有300条用于训练，没有独立验证集，不擅自拆分、报告验证成功率或挑“最佳验证模型”。

## 4. 当前代码确实需要 CUDA 适配，不能只改 JSON

当前 `sim/diffusion/train.py` 是 MPS MVP：设备、随机种子、同步、固定噪声评估都含 MPS 硬编码，
`window_batch()` 单样本、`range(2)`、`loss/2` 等也没有读取通用 batch/accumulation 配置。
`sim/diffusion/training.py` 的随机状态仅覆盖 CPU/MPS；默认数据路径也是 Mac 绝对路径。
所以仅修改 `training_config.json` 不会真正改变计算。

请用最小代码修改实现：

- 集中处理实际 `cuda:0` 设备、张量搬运、CUDA seed、同步与峰值显存统计。
  CUDA 分支不可无条件调用 `torch.mps.*`；保留现有 MPS 默认行为和测试，不全局字符串替换。
- `batch_size`、`accumulation` 必须真的控制批次、loss缩放、优化器更新频率和
  `WeightedCursor` 消耗；支持不跨回合的多个训练窗口，保持有效动作mask的归一化正确。
- 检查点保存并恢复 CUDA RNG、模型、EMA、AdamW、LR、归一化、采样游标和累计训练时间；
  恢复后优化器张量设备正确，配置/数据身份不符仍明确拒绝。
- 给 CLI、工作台启动及恢复路径提供同一份真实生效的 Linux 配置，设备与 batch 参数不能漏传。
  Linux 数据路径必须同时用于训练和工作台只读浏览，不被 `--diffusion-workspace` 又覆盖成 Mac 路径。
- 当前没有实际 DataLoader，`workers=0` 先保留；不要仅把配置改成8就说已并行取数。
  如确需引入预取，必须保证游标/断点恢复不被未消费的预取样本推进。
- 训练日志继续输出 `stage/step/loss/budget_seconds`，与现有实时面板兼容；新增字段只能增量扩展。
- 本次先用 `sim.diffusion.train` 训练；`sim.diffusion.pipeline` 会自动触发六次闭环验收，
  不要在这次“训练启动”任务中未经说明就启动额外验收。推理/CUDA扩展仅做训练必需部分。

## 5. 保持的模型、观测和动作语义

以下均沿用 Mac 这版，不为提高 batch 偷偷减负载：

- 五视角：四个角落相机＋wrist；320×320 RGB；两帧条件为当前帧与40ms前一帧。
- 36维机器人状态，另加1维目标数量；不输入物体真值、已收集数量等特权监督字段。
- 每视角独立随机初始化 ResNet18、GroupNorm，与动作网络联合训练；不冻结视觉、不共享骨干、
  不继承 ACT 权重。RGB先到[0,1]再做现有 ImageNet 标准化，无新增裁剪/缩放。
- 条件一维 U-Net 通道[256,512,1024]、时间嵌入128、kernel5、GroupNorm组数8；
  完整现有模型193328324参数，不换成更小网络。
- 100个扩散训练时间步、`squaredcos_cap_v2`、epsilon预测、有效动作掩码MSE；
  EMA power0.75；预测检查保持8步DDIM、eta0、EMA权重。
- 内部28步，对外严格返回 `[1:26]` 的25×4增量动作 `[dx,dy,dz,dyaw]`。
  第0步为历史动作，最后两步仅补齐、不计loss、不执行；其余按现有有效mask处理。
- 首帧复制补历史，末尾与无效动作掩码，不跨回合、不用未来观测。
  保留原接近阶段采样比例0.35，归一化只使用有效训练样本。
- 若将来做在线推理，继续25Hz采图、5Hz决策、每次执行5步，暂停仿真等待推理，
  无ACT跨块平均或200ms生效偏移；物理执行、Z导纳、安全与任务终止规则不变。

上游固定为 `real-stanford/diffusion_policy` 提交
`5ba07ac6661db573af695b419a7947ecb704690f`，保留现有MIT许可与来源。
官方 DDIM hybrid 配置可核对：
https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_ddim_hybrid_workspace.yaml

## 6. 训练参数和 batch 选择

基线沿用：seed42、FP32、AdamW lr1e-4、betas(0.95,0.999)、weight_decay1e-6、
梯度裁剪1、500次优化器更新预热、余弦LR；最多100000次优化器更新或累计24小时，先到即停。
每500次更新保存完整可恢复检查点，每阶段保留最近两份，结束导出最后完成更新对应的 EMA。
不启用新的AMP、TF32、torch.compile或新的学习率缩放规则；这些需要另做独立验证。

Mac 的真实microbatch=1、累积2、等效batch=2，不是官方默认值。
官方上述CUDA参考配置是batch64、累积1，但本任务五视角320输入不能只按batch数照搬。
先让基线在 CUDA 真实跑通，再做总计最多10分钟、独立临时模型的短基准：
按1、2、4、8、16、32、64逐级评估，遇到显存不足即停止增大，不能缩相机/分辨率/历史/网络。
记录真实窗口/秒、秒/优化器更新、有效batch、显存allocated/reserved峰值、RAM和数据读取耗时，
优先选留有至少20%显存余量且吞吐好的配置，不把“最大batch”自动当“最快”。

优先测试batch2、累积1来保持等效batch2；如果更大batch明显有益，可依据实测选择并明确记录
它与Mac等效batch2的差异，不宣称两次训练严格等价。不要求一定上64，不盲目把累积设成64求数字一致。
LR等其余参数仍用上述值，不能只比较更新次数而忽略每次更新样本数的差别。
最终配置应在正式小样本门之前固化；基准权重不可进入正式训练。基准与小样本训练时间计入24小时账本，
环境安装/数据复制不计作训练。切换配置不能覆写原实验或重置预算。

## 7. 测试与启动顺序

1. 先补测试再改 CUDA 相关实现：批次真实shape/实际样本数、梯度累积与mask、设备分支、
   CUDA RNG/采样游标恢复、预算、首尾时间对齐和工作台训练参数透传。
   跑 Diffusion 与工作台专项回归，运行命令见第9节。不要为绿测试修改任务物理/专家策略。
2. 独立 Linux 新实验，不能把 Mac 的 MPS 权重/优化器或“小样本门已过”结论直接搬过来。
   固定12个原训练窗口，最多500步/30分钟；固定噪声评估loss比初始化下降至少30%，
   五个视觉骨干和U-Net均确实更新，loss/梯度有限，无OOM。
3. 在真实 CUDA 上验证保存恢复：固定种子EMA预测、后续优化器更新与EMA一致，
   沿用现有 rtol1e-5/atol1e-6；不因失败偷偷放宽阈值或用CPU测试替代。
4. 门通过后以seed42重新初始化，开始全部300条的全量训练，不使用小样本权重。
   单独输出到 Linux 工作树下 `runs/diffusion_policy_linux_cuda_<时间标记>`，不覆盖任何旧实验。
5. 如果数据、CUDA、小样本门或恢复检查失败，停在该问题并交付证据，不自动降模型/图像规格。

## 8. 持久训练与工作台

继续复用原工作台，训练进程必须独立于HTTP请求、SSH连接和本轮Codex会话存活，保留日志及PID。
通过真实 JobRegistry 启动作业，或实现受测试的登记/启动入口；不能手写虚假running记录。
网页服务重启后可以按持久作业记录接回。Start/Stop/Resume走同一最终配置，Stop在完整更新边界保存。
不要使用只在当前前台终端存活的方式冒充持久训练。

工作台绑定Linux本机 `127.0.0.1:8773` 或确认空闲的本机端口，不未经许可暴露到0.0.0.0。
保留 Training 面板每3秒刷新的原始/平滑loss、真实更新步数、秒/更新与更新/分钟，以及断更/退出提示。
浏览器刷新后自动接回当前训练，确认它不是旧 ACT 作业或旧Mac快照。
如从Mac查看，可提供SSH转发说明；Mac的8773正在使用时用本地8774转发到Linux8773，避免混淆两个工作台。
无显示器的MuJoCo渲染按本机驱动配置并实测；训练已运行但五视角浏览未验证时要分别报告，不能伪称都成功。

## 9. 验证、交付和边界

建议专项命令（按本机环境路径调整）：

```sh
python -m pytest -o addopts='' -q tests/test_diffusion* tests/test_workbench*
node --test tests/training_monitor.test.mjs tests/pac_dashboard.test.mjs
```

本分支Mac端已有111项上述Python专项、9项前端测试通过的历史证据，不等于Linux已通过。
全项目既有专家规划测试存在失败/长耗时，见状态文档；不得隐瞒失败，也不因它们改动任务物理。
不要把“改完代码/测试通过”当成真实CUDA训练已启动。

交付时请给出：

- 实际仓库/本地分支/提交、Linux依赖锁、GPU与环境、真实生效配置、batch实测选择依据。
- 原示教核验结果、数据路径、完整性校验范围和未验证项。
- 训练PID、持久启动/停止/恢复命令、日志路径、工作台URL/必要的SSH转发方式。
- 真实小样本门和CUDA恢复测试结果，全量最近步数、loss与吞吐连续推进的证据。
- 检查点保存后核对完整元数据与加载；不能把步0恢复点当作训练成果。
- 当前状态清楚区分 `blocked / running / stopped / failed / training_complete`，
  不提前宣称训练完成、六次任务成功、泛化或优于ACT。

不需要在本轮对话等满24小时，但交付“已启动”前至少确认20次完整全量更新、
稳定递增日志及CUDA实际占用，已验证可恢复保存，且作业不会随着会话结束退出。
本次不自动跑六种子闭环验收，不自动推送模型、数据或Linux代码改动。
