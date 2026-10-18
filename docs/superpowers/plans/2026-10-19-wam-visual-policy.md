# WAM Visual Policy v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户已确认书面规格及逐项独立审查方式；实施前还需审阅本执行清单。

**Goal:** 实现并在既定资源内运行全视觉准备、预测式 ACT 纠偏的数据—训练—校准—三臂评测完整链路，按证据分别报告实现、训练、测试与目标达成。

**Architecture:** 新增独立 `wam_sweep` 包。在线只使用视觉对象信念与允许的 TCP／控制上下文；冻结原生 ACT／Hybrid／Z 控制，通过纯感知规划接口和因果参考预览接入。新增 GraphGRU、四模式条件 MLP、未来序列 GRU 双输出评价头，分阶段训练并进行两轮补采。

**Tech Stack:** Python 3.12；现有 `requirements-pac-act.lock.txt` 锁定的 PyTorch 2.11／torchvision 0.26／LeRobot 0.6.1／MuJoCo 3.14；NumPy、SciPy、Pillow、PyYAML、pytest；单 MPS、FP32。不接入新云训练服务，不新增 RL 算法依赖。

**Spec:** `docs/superpowers/specs/2026-10-19-wam-visual-policy-design.md`（用户已书面确认；规格是约束权威）。

**Worktree:** `/Users/huazhi_luke/.codex/worktrees/wam-visual-policy-v1/act-ur10-sweep-sim`，分支 `codex/wam-visual-policy-v1`，规格提交 `a4147dc7586f71fdb43e6ea9930cd0af6fd5a2b8`。

## Global Constraints

- 冻结 r8 bundle 的五视角 ResNet18／100k ACT、权重和预处理。保留 Hybrid／A* 规划方法、接触准备、力控、Z-admittance、交接约束及原安全和任务终止条件。
- 准备阶段与扫动阶段的零件信息全部来自视觉估计。禁止在线用真实中心、半径、类型、碰撞边界、收集 mask 或真实首入区时间修补视觉。
- WAM 不接收关节位置／速度或直接腕部六维力／力矩。关节信息仅供网络外标定／可达性检查，ACT／原力控原有输入不变。
- 固定两秒观测历史：5 Hz 下包含当前时刻的 11 个观测；不足历史用有效掩码。预测／规划一秒，25 Hz 输出 25 步；每 200 ms 重规划，只执行前五步。
- XY 残差范数不超过 0.8 倍当前 ACT XY 位移范数，yaw 残差不超过 ±45°；不累加残差，不改 Z。零残差／零滤波状态保持 ACT 原输出；零 XY 位移对应零 XY 修正。
- 单模型置信度，不使用集成；只要求 ACT 局部前向，不新增全局 X 单调约束；不保留旧学习器的 50 帧零纠偏等待。
- 六个零件、目标任意 N=1～6；精确 N 且无其他部分重叠才成功，首入区触发两秒窗口且不重置；45 秒／1,200 帧共享预算及零恢复轮次。
- GraphGRU 隐藏宽度 128、三层消息传递；四模式共享提案 MLP 隐藏层 256、256，输出 15；未来 GRU 隐藏 128、评价 MLP 隐藏 256、128，输出成功 1＋数量 7。目标 N 不直接进入物理预测头。
- 感知启动 120 条，仅用于感知；随后主轨迹 240／120／120、教师起点 600／300／300；每起点 12 条短分支，仅三条完整续跑；续跑器 ACT／WAM0／WAM1，最终 WAM2。
- 每阶段每 N 至少 20 个成功教师起点、来自至少 10 个不同父布局；全失败起点没有动作正向监督。旧物理数据可合并，旧终局标签不得无区分混入新版评价头。
- 父布局族隔离；验证 60 父布局、最终校准另 60 父布局各一个起点；正式 120 布局，每 N 20 个，三臂一次共 360 尝试；不按 ACT 成败筛选。
- 全部采集／训练／对照／校准／评测最多 72 小时有效计算墙钟；首先不超过两小时测吞吐及体积并计入总额。
- 新工件上限 45 GiB，磁盘至少剩 25 GiB；40 GiB 或低于 30 GiB 余量预警。临时写入和其他进程用量计入保护，不自动删除旧实验／原始证据／唯一恢复依赖。
- 主门为净增加至少 12／120 次成功且单侧 exact McNemar p<0.05；主门通过后才检验预测贡献。不把样本十个百分点提升说成总体增益下界超过十个百分点。
- 不提交权重／数据，不创建 PR 或合并主分支；不修改原 root／PAC 未提交工作。不得把 scaffold、synthetic tests、预检、已训练或正式效果相互替代。

## Review Focus

1. 冻结源码与当前 checkout 同名模块混入：必须拒绝错误来源；Task 2／5 的子进程导入来源测试。
2. 图像、ACT 参考和实际执行有不同时钟：不得把未来实际指令当已知输入；Task 3／8 的延迟索引及未来投毒测试。
3. 中断／磁盘骤降发生于原子 checkpoint 中途：保留上一个完整恢复点且拒绝越过预算；Task 2／10 的中断写与峰值空间测试。
4. 部分分支提前终止或未做完整续跑：不得生成补齐未来／失败终局／正动作标签；Task 4／8／10 的有效掩码与未标注测试。
5. 推理期间短遮挡、无效状态及低成功分数混淆：前两者依协议处理，低分不能导致偷偷回退 ACT；Task 6／7／11 的遮挡、失效和低分候选测试。

## 文件与接口地图

所有相对路径均相对于上述工作树；训练工件放在 `/Users/huazhi_luke/.local/share/wam-sweep-v1/`，不放进 Git。`.venv` 和本次发布缓存也纳入新工件空间清单，嵌套路径不能重复计数。

| 所属 | 文件 | 职责 |
|---|---|---|
| Task 1 | `wam_sweep/contracts.py`, `config.py`, `history.py` | 严格输入类型、张量排列、默认配置、有效历史 |
| Task 2 | `wam_sweep/artifacts.py`, `inputs.py`, `splits.py` | 输入快照、哈希、资源账本、原子恢复、布局隔离 |
| Task 3 | `wam_sweep/actions.py`, `reference.py` | 唯一动作解码器、纯 ACT 队列预览 |
| Task 4 | `wam_sweep/truth.py`, `labels.py` | 监督专用投影／接触图／过渡标签 |
| Task 5 | `wam_sweep/native.py`, `snapshots.py` | 冻结原生运行时、物理／控制全恢复与无 HUD 观测 |
| Task 6 | `wam_sweep/vision/features.py`, `bev.py`, `detector.py`, `tracking.py` | 共享前向特征、多尺度 BEV、六对象检测及跟踪 |
| Task 7 | `wam_sweep/planning.py`, `preparation.py` | 视觉规划只读接口、一次规划与监测 |
| Task 8 | `wam_sweep/data/store.py`, `collection.py`, `datasets.py` | 主轨迹／12短分支／3终局续跑、版本化训练样本 |
| Task 9 | `wam_sweep/models/graph.py`, `dynamics.py`, `proposal.py`, `quality.py`, `losses.py` | 三头、集合监督、未来预测和消融评分器 |
| Task 10 | `wam_sweep/training.py`, `calibration.py` | 分阶段训练、冻结、校准和完整 checkpoint |
| Task 11 | `wam_sweep/controller.py`, `safety.py` | 五候选在线选择、25 Hz 执行、无效终止 |
| Task 12 | `wam_sweep/cli.py`, `pipeline.py`, `preflight.py`, `evaluation.py`, `reporting.py` | 统一入口、生命周期、预检、三臂测试和统计 |
| Task 13 | 工件中的 run／报告／检查点／回放；`docs/WAM_VISUAL_V1.md` | 真正运行完整实验及据实交付 |

### 跨任务公共类型（Task 1 定义，后续不得私自重命名）

- `ObjectBelief`：中心 XY、速度 XY、轮廓半尺度 XY、矩形 yaw／角速度、圆／矩形概率、可见性、位置协方差、三态概率、16 维外观、内部 `track_id` 和有效性。`track_id` 不进入网络。
- `RobotContext`：实测 TCP xyz／yaw 及速度、准备／交接／扫动／终止阶段、残差及变化率、已知回合时间、接触准备与参考有效标志，以及纯视觉推断的首入区年龄／有效标志。无关节／wrench／真值计数。
- `HistoryBatch(objects, brush, context, object_valid, history_valid)`：FP32，批维 B；对象 `[B,11,6,34]`、刷子 `[B,11,9]`、控制上下文 `[B,11,15]`。34 维顺序为中心2、速度2、半尺度2、矩形双角 sin/cos2、角速度1、轮廓概率2、可见1、三态概率3、对称协方差3、外观16。刷子为 xyz3、yaw sin/cos2、线速度3、yaw 速度1；上下文为阶段 one-hot4、已施加残差3、残差率3、回合时间1、准备好1、参考有效1、视觉入区年龄1、其有效1。
- `CandidateProgram(knots)`：归一化 `[5,3]`，五个未来控制点；`ReferencePreview(targets, times, valid, issued_at, source_hash)` 为已知绝对参考 `[25,4]`。
- `WorldPrediction`：未来 25 步连续对象均值 `[B,25,6,9]`（中心2、速度2、半尺度2、双角2、角速度1）、位置／速度 log-std `[B,25,6,4]`、三态 logits、轮廓 logits、可见 logits；实际刷子状态 `[B,25,9]`／不确定性和有效步掩码。矩形朝向按 π 周期，刷子 yaw 按 2π 周期。
- `TaskScore(success_logit, final_count_logits, calibration_hash)`：数量类别0～6；`EpisodeResult` 分离 success／task_failure／perception_invalid／model_invalid／safety_stop／infrastructure_error／incomplete 及原生终止原因。
- `Controller` 协议：`reset(initial)`, `observe(observation)`, `act(reference)`, `snapshot()`, `restore(state)`。`OnlineObservation` 与监督专用 `RawTruth` 必须为不同类型和命名空间。
- `EpisodeSpec(parent_id,seed,target_count,split,phase,arm,attempt_index)`、`InputIdentity(root,manifest_hash,model_hash,native_hashes)`、`RunIdentity(run_id,spec_hash,plan_hash,config_hash,input_identity,split_hash)`、`ModelVersion(phase,kind,checkpoint_path,model_hash,perception_hash,dynamics_hash,continuation_hash)` 统一身份语义。`BackendFactory(spec:EpisodeSpec)` 与 `ControllerFactory(version:ModelVersion)` 是明确可调用协议，不把真实环境对象放进可见观测。
- 原始ACT续跑版本用 `kind="act", phase=-1`，WAM版本phase为0／1／2；版本身份包含其冻结感知／原生控制／预处理与配置依赖，而不只是actor权重。训练工件缺少某依赖哈希必须拒绝恢复。

---

### Task 1: 契约、配置与有效历史

**Files:** Create `wam_sweep/__init__.py`, `contracts.py`, `config.py`, `history.py`, `configs/wam_visual_v1.yaml`, `tests/wam/test_contracts.py`; Modify `pyproject.toml`（只增加 `wam_sweep*` 包发现）。

**Interfaces:** Produces 上述所有公共类型、`WAMConfig.load(path) -> WAMConfig`、`HistoryBuffer.append(observation)`／`batch() -> HistoryBatch`／`snapshot()`／`restore(state)`。Goal 是独立 1～6 整数参数，不塞入物理历史。

核心测试 `test_startup_history_is_masked`：`assert batch.history_valid.sum().item() == 1`；`assert tuple(batch.objects.shape) == (1,11,6,34)`。测试中手工构造单帧合法观测，不用生产打包器生成期望值。

- [ ] 建立本工作树 `.venv`：`uv venv --python 3.12 .venv`，`uv pip sync --python .venv/bin/python requirements-pac-act.lock.txt`，`uv pip install --python .venv/bin/python --no-deps -e .`；不改已有 PAC venv。运行原仓库测试一次，记录既有失败／不适用条件，不把它们算成本分支修复或全通过。
- [ ] 先写输入投毒、历史掩码、非法时间和排列测试：第一帧只一个有效历史位、到两秒为11位；非法质量／摩擦／真实收集／关节／wrench字段必须拒绝；固定合法图像／TCP／参考时修改隔离真值不能改变打包结果。断言采用手写期望值，如 `history_valid.sum()==1`、`objects.shape==(1,11,6,34)`，不通过读取常量自证。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_contracts.py -q`，确认缺模块／行为失败；再实现严格校验、有限值检查和 deepcopy 历史恢复。正常启动不为补足两秒而阻塞。
- [ ] 重跑该测试并检查安装后可从工作树外导入 `wam_sweep`。配置完整写入规格中的配额、模型结构和预算，不导入旧 SAC／PAC 的 learner warmup。
- [ ] Commit `feat(wam): define visual policy contracts and configuration`。

### Task 2: 输入快照、布局分割和可恢复资源账本

**Files:** Create `wam_sweep/inputs.py`, `artifacts.py`, `splits.py`, `tests/wam/test_artifacts.py`, `test_splits.py`。

**Interfaces:** Consumes `WAMConfig`。Produces `verify_inputs(source) -> InputIdentity`、`snapshot_inputs(source,destination,budget) -> InputIdentity`、`BudgetLedger.admit/reserve/record/poll`、`atomic_checkpoint(path,payload,budget)`、`make_split_manifest(config,exclusions) -> SplitManifest`、`assert_family_isolation(records)`。

核心测试 `test_admission_rejects_24gib_free`：`assert decision.allowed is False`；`test_interrupted_checkpoint_preserves_previous`：`assert path.read_bytes() == b"previous-complete-checkpoint"`。覆盖真实文件写入／恢复，不只断言fake被调用。

- [ ] 先测小型真实文件树的哈希不符、源 symlink 越界、原子写中断、原件不变、空间峰值；伪磁盘探针只是外部系统边界，实际原子文件写入仍真实执行。断言剩余24 GiB或工件预期46 GiB时拒绝，原完整 checkpoint 字节不变；超过72小时不得新准入，暂停的空闲时间不计有效计算，数据采集／训练／测试均计时。
- [ ] 建立单运行非阻塞进程租约；与已有原生推理租约使用显式锁路径，不能默认写入只读输入快照的runs目录。已有活跃运行时拒绝第二实例，不根据锁文件单独认定进程存活，不杀用户其他工作；重型仿真／模型探针不得借测试名绕过计算账本。
- [ ] 先测同父布局的不同 N、分支和重叠窗口跨集合时拒绝；正式清单恰有120个唯一布局且每 N=20，最终校准60个不同父布局各一个根；种子和指纹同时与 r8 exclusions 核对，不能失败后换种子。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_artifacts.py tests/wam/test_splits.py -q` 验证 RED，再实现。默认训练种子区间使用独立新命名空间，持久化生成器／配置哈希；父布局族先分集合后生成所有派生记录，不以结果筛选。
- [ ] 将 r8 源复制到工件 `inputs/r8/`，非 hardlink，逐文件核验195个 manifest 项及权重哈希后只读；源 bundle 不变。加载冻结 `sim`／原生运行时前检查其模块路径，已从其他 checkout 导入同名模块就拒绝并在新进程启动，不动态混用。新输入类型不给在线模块 RawTruth 引用。
- [ ] 资源保护包含 owned artifact／新 venv／缓存，嵌套去重；每次大写入先预留旧＋新＋临时副本的峰值，不自动清理旧数据。建立固定集合：train 各阶段240／120／120父布局（分别40／20／20每N）、感知启动120父布局、验证60父布局（固定30选模／30开发校准）、最终校准60、预检12、正式120，全部互斥。默认seed起点分别90000000／90001000／90002000、90010000、90020000、90030000、90040000、90050000，组内连续编号、N按编号轮转；先核对exclusions及布局指纹，若冲突必须在看到结果前重新生成并冻结清单，不在失败后换布局。
- [ ] GREEN 后 Commit `feat(wam): guard inputs artifacts and layout lineage`。

### Task 3: ACT 相对解码器与因果参考预览

**Files:** Create `wam_sweep/actions.py`, `reference.py`, `tests/wam/test_actions.py`, `test_reference.py`。既有 `residual_rl/relative_action.py`、`sim/act/realtime.py` 为只读语义依据，不修改原控制。

**Interfaces:** Consumes `CandidateProgram`, `ReferencePreview`。Produces `ActionCodec.sample(program,t,current_raw) -> ndarray(3)`、`decode(actual_tcp,native_target,raw,dt,active) -> DecodedAction`、`snapshot/restore`；`preview_reference(scheduler,held_reference,now) -> ReferencePreview`。

核心测试 `test_zero_action_preserves_native_bytes`：`assert decoded.target.tobytes() == native_target.tobytes()`；`test_valid_from_uses_age_index_five`：`assert selected[0] == 5.0`，其中队列第j条的x由测试手工设为float(j)。

- [ ] 先写零残差逐字节相等、actual TCP 锚点、零位移、yaw跨±π、边界附近直接可行生成和重复调用不积累测试。手算 p=(0,0)、q=(0.1,0,0.2,0)、归一化局部前向1时修正上限0.08、Z始终0.2；另一 p 改变半径，不减去历史 anchor。
- [ ] 先写曲线和滤波测试：控制点在0.2～1.0秒，t=0锚定已有原始归一化指令，PCHIP各通道插值、原始指令经0.2秒指数平滑；每个40ms重新由实际 TCP 映射可行域。五个候选克隆同一滤波初态，预测某条不能污染下一条或实际执行。
- [ ] 先写队列测试：issued=0、valid_from=0.2、25Hz时在0.2读取索引5而非0；跨±π融合正确；预览不修改真实队列或过期列表，不插入未来查询，hold期无残差。命令时刻为 t+j*0.04，预测结果时刻为 t+(j+1)*0.04，j=0..24。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_actions.py tests/wam/test_reference.py -q` 确认RED，再封装现有纯可行域映射与独立滤波状态；预览仅使用 scheduler 深拷贝。接触 governor 仍在原生执行层，不能被新解码器替换。
- [ ] GREEN，含全部25步边界审计，Commit `feat(wam): add causal ACT preview and bounded action codec`。

### Task 4: 真值隔离、二维几何与稳定随刷标签

**Files:** Create `wam_sweep/truth.py`, `labels.py`, `tests/wam/test_labels.py`, `test_geometry_labels.py`。

**Interfaces:** Produces 监督专用 `RawTruth`／`TruthFrame`、`capture_truth(env) -> TruthFrame`、`project_envelopes(frame) -> ObjectTargets`、`CarriedLabeler.update(frame) -> LabelFrame`／`snapshot/restore`。在线代码不能调用这些接口。

核心测试 `test_early_termination_does_not_pad_future`：`assert target.valid_steps.sum() == 3`；`test_transient_contact_is_not_stable`：`assert not label.carried_valid[1]`。标签类型明确包含 `valid_steps`／`carried_valid`，未知与false不能合并。

- [ ] 先测偏置螺钉／螺栓的整体包围中心不等于 body 原点、圆与矩形投影随真实碰撞几何和旋转变化、圆的yaw损失被掩码、矩形按π周期等价。用已知单形／复合形体的手算边界，不读可视mesh当碰撞占据。
- [ ] 先测直接推动、接触链间接推动、短碰撞不稳定、掉队、短遮挡、全入区、额外部分重叠，以及提前终止有效步数。200ms证据窗、120ms加入／脱离滞回，不确定过渡没有稳定二分类标签；窗口可恢复且不累计“曾接触”身份。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_labels.py tests/wam/test_geometry_labels.py -q` 得到RED，再实现原始证据与派生标签分开存储。速度根据包围中心的时间变化，不误用偏置刚体原点速度。
- [ ] 默认标签门限作为版本化配置：共同平移速度差≤0.025m/s、200ms内刷子相对位移变化≤8mm；有效接触链须在证据窗内，刷前范围来自刷子几何和包围半尺度扩张。刷子运动不足以判定共动、接触／空间条件冲突或门限边缘均标unknown；不得把这些默认值当已审查标签质量。Task13必须先审查代表片段再批量训练。
- [ ] GREEN 后 Commit `feat(wam): isolate supervision and derive planar carry labels`。

### Task 5: 冻结原生仿真后端与完整恢复

**Files:** Create `wam_sweep/native.py`, `snapshots.py`, `tests/wam/test_native_snapshot.py`, `test_native_sources.py`。

**Interfaces:** Consumes `InputIdentity`, `RobotContext`, `ActionCodec`, `TruthFrame`。Produces `FrozenSweepBackend.open(input_identity,episode_spec)`、`observe() -> OnlineObservation`、`step_target(target) -> StepResult`、`snapshot() -> EpisodeSnapshot`／`restore(snapshot)`／`close()`；bootstrap-only oracle preparation 与后续 `prepare(PerceivedPlanningScene)` 接口分离。

核心测试 `test_snapshot_repeats_physical_commands`：`np.testing.assert_array_equal(first.actions, second.actions)`；`assert first.termination_reason == second.termination_reason`。两次均真实step，同一起点中间执行restore，不用录制结果冒充第二次仿真。

- [ ] 先在子进程测冻结 `sim`／原生 runner 的 `__file__` 必须处于输入快照，污染导入必须报错；保留冻结源哈希，禁止从其他 cwd／旧导出动态补文件。只重用被冻结的原生控制函数，不复用 `pac_act.HybridEpisode` 的真值规划、50帧gate或2000帧override。
- [ ] 用真实小型 MuJoCo 接触场景先测 snapshot→执行→restore→同指令重复，动作、控制状态、接触／终止事件一致。覆盖准备、交接早期及扫动起点；qpos相同但滤波／warmstart不同的负例必须暴露差异。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_native_snapshot.py tests/wam/test_native_sources.py -q` 确认RED，再实现 `mjSTATE_INTEGRATION` 加原生 controller／governor／contact latch、ACT builder历史／队列／reference、残差、跟踪与各随机数状态。渲染器句柄不序列化；恢复后相机外参从相同物理状态重新计算。
- [ ] 无HUD五视角320×320，记录标定与查询／接收／有效／执行时刻；冻结ACT query同时提供Task6的特征钩子。保留100Hz低层和原任务判定优先级；共享45秒／1200帧从同一reset／准备起点计时。
- [ ] GREEN 后 Commit `feat(wam): bridge frozen native control with full snapshots`。

### Task 6: 同次前向特征、BEV、六对象检测与跟踪

**Files:** Create `wam_sweep/vision/__init__.py`, `features.py`, `bev.py`, `detector.py`, `tracking.py`, `tests/wam/test_features.py`, `test_bev.py`, `test_tracking.py`。

**Interfaces:** Consumes `OnlineObservation`, `HistoryBuffer`, `InputIdentity`。Produces `FrozenFeatureTap.query_act_and_features(observation) -> ActFeaturePacket`、准备阶段及固定参考分支的 `features_only(observation)`、`BEVEncoder.forward(features,calibration)`、`VisualStateEstimator.estimate(packet) -> tuple[ObjectBelief,...]`／`snapshot/restore`、感知监督损失。

核心测试 `test_feature_tap_preserves_act_output`：`torch.testing.assert_close(chunk_with_tap, chunk_without_tap, rtol=0, atol=0)`，相同eval模型／输入／随机状态；`test_short_occlusion_keeps_identity`：`assert after.track_id == before.track_id`，随后超时必须invalid。

- [ ] 先测实际ResNet层钩子同一次ACT前向获得layer1／2／4；冻结权重、BN、预处理和ACT输出不变，摄像头次序／标定错配拒绝；只复用图像空间特征，不复用混有关节／wrench的ACT融合token。训练可单独对原始图像跑同一冻结主干，但在线不得重复编码。
- [ ] 先测已知投影点、被相机视锥遮掉的网格、腕部外参更新；BEV覆盖桌面加2cm边缘，5mm网格，默认208×168。采用有效视角掩码和多尺度融合，不用真实物体高度／分割作为在线输入。
- [ ] 六对象检测采用BEV六查询解码器：两层cross-attention、宽128／4头，Hungarian检测匹配；输出规格状态和16维局部外观。先测六对象无序监督、圆／矩形及可见性，匹配不依赖固定真实ID。
- [ ] 先测常速度Kalman＋Hungarian在交叉／短漏检时保持跟踪，未知状态不能伪造高置信；重现快照一致。持久身份仅用于关联，不入神经特征。默认关联位置Mahalanobis平方门限9.21；初始三次有效观测要求可见概率≥0.5且校准后最大位置标准差≤12mm；在线超过600ms未观测或位置标准差>25mm标invalid。短遮挡继续信念传播，不能读真值补齐。这些数值是待实验的工程预设，和不确定性校准一起写入checkpoint，仅验证集可调整，正式测试前冻结。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_features.py tests/wam/test_bev.py tests/wam/test_tracking.py -q` 记录RED／GREEN。持久化原始图像或版本化估计状态，默认不缓存全量多尺度特征；内存缓存有界，不把TB级特征隐形落盘。
- [ ] Commit `feat(wam): estimate tracked planar scenes from frozen visual features`。

### Task 7: 全视觉 Hybrid／A* 准备与一次规划监测

**Files:** Create `wam_sweep/planning.py`, `preparation.py`, `tests/wam/test_visual_preparation.py`。

**Interfaces:** Consumes `ObjectBelief`, `FrozenSweepBackend`。Produces `PerceivedPlanningScene`（只读中心／保守半径／估计最右边界及已知静态几何）、`prepare_visually(backend,estimator) -> PreparationResult`、`monitor_plan(scene,plan) -> PlanValidity`。

核心测试 `test_invalid_path_never_replans_or_uses_truth`：`assert planning_calls == 1`；`assert result.status == "perception_invalid"`；任何访问真实零件字段的调用直接使测试失败。

- [ ] 先用访问即抛错的特权环境代理测试现有 `expert_plan_candidates` 接收到的只有估计规划scene；中心、nominal_radius及component_collision_x_max全部由视觉估计导出，矩形用外接圆加校准不确定性。保留原算法，不传原env让其回退。
- [ ] 先测必须三次有效观测并建立六轨迹才规划；只规划一次；未满两秒历史不阻止有效ACT介入。漏检／错误轮廓／路径失效在5Hz监测时终止，不重复规划／重新接触。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_visual_preparation.py -q` RED后实现，调用Task5原生接触准备，Z／力控／governor不变。原生GT收集终止只在环境层，不能泄漏到神经历史的计数／入区时钟。
- [ ] GREEN 后 Commit `feat(wam): drive preparation exclusively from perceived scenes`。

### Task 8: 主轨迹、12短分支和3终局续跑的数据链路

**Files:** Create `wam_sweep/data/__init__.py`, `store.py`, `collection.py`, `datasets.py`, `tests/wam/test_collection.py`, `test_dataset_lineage.py`。

**Interfaces:** Consumes Task1～7公共类型／后端／预算／分割。Produces `EpisodeStore`、`collect_bootstrap(backend_factory,store,split,budget,config) -> CollectionSummary`、`collect_phase(phase,controller,continuation_factory,backend_factory,store,split,budget,config) -> CollectionSummary`、`collect_root(root,programs,continuation_factory,backend,store) -> RootRecord`、`PredictionDataset`／`TeacherDataset`／`TerminalDataset`、`coverage_report(records) -> CoverageReport`。根为完整 `EpisodeSnapshot`，工厂遵守Task1协议，split为Task2固定清单。

核心测试 `test_only_selected_branches_have_terminal_labels`：`assert len(record.branches) == 12`；`assert sum(b.terminal_result is not None for b in record.branches) == 3`，未续跑项保持None而非失败。`RootRecord.branches` 保存逐分支证据。

- [ ] 先测从同一起点恢复的12短分支均执行固定已知参考及候选一秒，只有三个唯一分支有完整终局；reference／hope／random的选择不读取未来truth，记录概率和续跑版本。希望候选用当前模型评分；尚无模型时用视觉估计几何评分，不读GT未来。
- [ ] 初期候选为零参考＋平滑有界扰动＋视觉几何引导曲线，有学习器后替换部分提案；共12条，不把随机扰动叫学习动作头。Teacher roots均衡到每N初始100／补采各50，父布局内可多个起点但不得跨split。
- [ ] 先测未续跑不等于失败、提前终止步掩码、bootstrap不能流入动作教师、all-fail无正例、同根共享历史图像、未来实际指令不可进入输入；完整一秒后续跑器的观察历史／调度／跟踪缓存已同步，不能从旧缓存启动。固定分支期间5Hz视觉观测用同一个冻结主干的features-only路径更新，不查询／偷用未来ACT动作；切换续跑器时的新查询仍尊重原生valid_from和hold，不能凭空注入根时刻未知参考。
- [ ] 续跑工厂绑定完整冻结版本；若根来自另一版本，按原始历史重建该续跑器自己的感知／跟踪／控制上下文，并用实际已施加残差同步滤波，不能偷换为当前新版视觉模块再沿用旧hash。必要重放成本计入账本；正式三臂共用最终视觉模块是另一条评测约束，不能混淆两者。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_collection.py tests/wam/test_dataset_lineage.py -q` RED，再实现持久化图像／动作／证据分区、原子游标、失败也保留的主轨迹。主轨迹无HUD lossless图像；短分支与续跑默认保存状态／动作／接触／必要回放证据，不重复保存完整历史及全部多尺度特征。
- [ ] 数据集物理样本可跨已知版本合并；评价头只取指定continuation_hash的完整结果；成功动作教师保留成功与版本证据。更新感知后，从保留原始主轨迹／根历史按时间因果重算当前感知版本的belief缓存，不能混用旧估计或用未来图像补根状态；当前估计不可用的教师不算可训练覆盖。每阶段每N成功根≥20且父布局≥10才允许动作正向训练；不足在已预留本阶段预算内补采，仍不足报告not_ready而非降门槛。
- [ ] GREEN 后 Commit `feat(wam): collect versioned counterfactual and terminal supervision`。

### Task 9: GraphGRU、四模式提案与未来评价模型

**Files:** Create `wam_sweep/models/__init__.py`, `graph.py`, `dynamics.py`, `proposal.py`, `quality.py`, `losses.py`, `tests/wam/test_models.py`, `test_model_losses.py`。

**Interfaces:** Consumes `HistoryBatch`, `ReferencePreview`, `CandidateProgram`。Produces `HistoryEncoder.forward(history)`、`GraphGRU.rollout(history,reference,candidate) -> WorldPrediction`、`ProposalHead.propose(history,goal,reference)`、`TaskQualityHead.evaluate_task(history,goal,candidate,prediction) -> TaskScore`、`HistoryOnlyQualityHead`及三类训练损失。

核心测试 `test_prediction_is_object_permutation_equivariant`：`torch.testing.assert_close(permuted.object_mean, original.object_mean[:,:,permutation], rtol=1e-5, atol=1e-6)`；`test_proposal_has_four_modes`：`assert tuple(programs.shape) == (2,4,5,3)`。`WorldPrediction` 连续对象字段固定名为 `object_mean`。

- [ ] 先测对象节点置换的物理等变、动作／评分不变；25步实际刷子响应不是复制指令；改变候选可影响预测，physics接口无goal。图隐藏128、三层关系消息传递／GRU；集合pool保留关系编码。连续尺度为正、yaw按正确周期，分布参数有限。
- [ ] 预测的有效掩码仅来自已知输入和数值有效性；真实提前终止掩码只在loss／指标侧使用，不能喂给预测器偷看终止时刻。物理头不因知道目标N而提前截断未来。
- [ ] 先测四模式共享MLP（256、256→15）输出 `[B,4,5,3]`；成功左右绕行集合做一对一匹配而非均值；无成功时动作loss为零且没有伪正例／未匹配监督。历史／reference编码可共享只读上下文，不用预测真值生成提案。
- [ ] 统一内部尺寸：历史对象mean／max各128，加刷子128组成384维场景编码；目标embedding16，模式embedding16，ACT参考GRU64（xyz、yaw sin/cos及有效位）；提案MLP输入480。评价候选knots经MLP编码64，未来关系集合384维序列经GRU128；评价输入592，去未来消融输入464。提案与评价各自持有目标embedding，只有冻结的历史编码共享，避免失败评价梯度暗中更新提案。
- [ ] 先测未来GRU128＋MLP256、128双输出1／7；task输入来自冻结GraphGRU自己预测的未来。固定模型输入并篡改数据记录中的GT未来／真实未来指令，评分不变；标签改变只影响监督loss。
- [ ] rollout逐步使用Task3同一codec和自己预测的TCP生成控制条件，不用真值TCP或未来实际指令替代。候选不通过world loss反向训练提案头；codec控制条件构造视为外生输入、仅该分支stop-gradient以共用NumPy可行域解码器，物理预测状态的25步递推路径不detach。记录此近似并测试采集／预测／执行解码前向一致性，不能另写一个不同边界的预测版decoder。
- [ ] 消融共享同一感知、历史与冻结提案头／候选，单独训练无未来输入的MLP256、128评分器，输出同样1／7；在线不得调用GraphGRU。匹配教师数据、训练／选模预算和候选数量，不把无预测等同少一个候选。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_models.py tests/wam/test_model_losses.py -q` RED／GREEN，再 Commit `feat(wam): implement predictive proposal and terminal quality heads`。

### Task 10: 分阶段训练、校准与恢复

**Files:** Create `wam_sweep/training.py`, `calibration.py`, `tests/wam/test_training.py`, `test_calibration.py`。

**Interfaces:** Consumes Task6感知模型、Task8数据集、Task9模型与Task2账本。Produces `train_perception(phase,store,inputs,config,budget) -> ModelVersion`、`train_dynamics(phase,perception,store,config,budget) -> ModelVersion`、`train_action_quality(phase,perception,dynamics,continuation_hash,store,config,budget) -> ModelVersion`、`calibrate(version,records,config) -> CalibrationBundle`、`load_checkpoint(path,expected_identity) -> CheckpointPayload`。记录类型为Task8版本化记录；checkpoint身份为 `RunIdentity`。

核心测试 `test_resume_rejects_wrong_native_identity`：`with pytest.raises(ValueError): load_checkpoint(path, wrong_identity)`；`test_all_failure_batch_does_not_update_actor`：`torch.testing.assert_close(actor_after, actor_before, rtol=0, atol=0)`。使用真实模型／optimizer，不能mock训练函数。

- [ ] 先测真实小批量优化确实降低可学习fixture损失（不是mock optimizer）；阶段冻结时参数、BN及梯度均不变。目标为位置／速度概率损失、轮廓／yaw／可见性／随刷监督与有效步掩码，task为BCE＋最终数量CE；提案为成功集合匹配loss。
- [ ] 动作与评价使用独立optimizer参数组；本批无成功教师时不step动作optimizer，连同模式／目标embedding保持不变，不能让AdamW零梯度weight decay冒充无监督更新。评价仍可训练该批真实终局，物理仍可使用失败短分支。
- [ ] 先测all-fail不训练动作、未续跑不进终局、错误continuation_hash拒绝、只合并物理数据；teacher覆盖不足返回not_ready而非保存伪合格actor。数据统计／normalizer仅由train拟合。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_training.py tests/wam/test_calibration.py -q` RED，再实现规格AdamW学习率3e-4／1e-4、batch4／64／64、maxepochs30／40／40、patience5、clip1、seed42、FP32单MPS。现有设备支持验证后使用MPS，不私自切CPU跑正式训练并冒称同协议。
- [ ] 单模型概率采用开发校准数据拟合temperature，位置／运动不确定性用留出残差尺度校准，最终60独立父布局只作最终校准、不选模；保存每个head／版本的校准哈希。每阶段对固定60验证父布局采60条视觉主轨迹、每父布局一个根×12短／3终局，续跑版本与本阶段一致；它们不计训练配额、不得混入训练，额外成本全部计入总资源。报告Brier／ECE／覆盖，不声称可靠发现所有OOD。
- [ ] checkpoint包含新增模型／optimizer／调度／CPU与MPS RNG／游标／split／normalizer／原生与感知依赖哈希；不复制冻结ResNet权重到每个checkpoint。原子写中断和恢复后下一更新一致测试必须通过；每阶段至少保存可恢复last与validation-best且不自动删除唯一恢复点。
- [ ] GREEN 后 Commit `feat(wam): train calibrate and resume staged model versions`。

### Task 11: 在线五候选选择、失效终止与消融

**Files:** Create `wam_sweep/controller.py`, `safety.py`, `tests/wam/test_controller.py`, `test_controller_safety.py`。

**Interfaces:** Consumes Task3／5／6／7／9／10。Produces `VisualACTController`, `WAMController`, `NoPredictionController`（均实现Task1 Controller）、`validate_prefix(candidate,backend) -> CandidateValidity`。

核心测试 `test_low_scores_choose_valid_candidate_without_fallback`：给定五个均有效候选的success logits为[-0.9,-0.8,-0.3,-0.7,-0.6]，`assert selection.candidate_index == 2`；`assert selection.execute_steps == 5`。选择结果必须包含这两个审计字段。

- [ ] 先测每200ms固定四条学习候选＋一条平滑归零候选，仅选一个并执行前五步；分数同分优先小残差，再固定顺序。所有候选从同一filter／history快照开始，未选分支无副作用；不能平均。
- [ ] 先测接触准备完成且第一有效ACT参考即可介入，早期两秒仍经原governor；不等待50帧，native hold期无残差。每40ms实际TCP重新映射，预测中未执行的后20步不能写入真实历史。
- [ ] 先测低分仍选有效候选，短遮挡按校准信念处理；NaN／协议不符／跟踪无效／无候选立即终止且之后无ACT回退动作。基础设施错误停批，不计普通策略失败。工作空间／IK检查仅作用即将执行前200ms，原生物理安全始终保留。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_controller.py tests/wam/test_controller_safety.py -q` RED后实现。三臂共用视觉准备、ACT队列与仿真时钟；消融没有未来预测调用但候选完全匹配。
- [ ] GREEN 后 Commit `feat(wam): select and execute visual residual candidates online`。

### Task 12: 统一入口、阶段准入、预检与三臂报告

**Files:** Create `wam_sweep/cli.py`, `pipeline.py`, `preflight.py`, `evaluation.py`, `reporting.py`, `tests/wam/test_cli.py`, `test_pipeline.py`, `test_evaluation.py`; Create `docs/WAM_VISUAL_V1.md`（实际使用与状态语义）。

**Interfaces:** Produces `python -m wam_sweep.cli {verify-inputs,bootstrap,collect,train,calibrate,profile,preflight,evaluate,report,run,resume}`；统一参数 `--config configs/wam_visual_v1.yaml --root /Users/huazhi_luke/.local/share/wam-sweep-v1/`。`run`固定整个有限阶段图，`resume`验证依赖后接续，未提供root不得写到cwd或旧实验。

核心测试 `test_exact_discordant_pair_tail`：b=12、c=0的手写120布局结果表应 `assert stats.p_value == 1/4096`、`assert stats.net_gain_count == 12`；删去一个正式attempt应 `assert report.test_complete is False`。

阶段入口固定：collect／train使用 `--phase 0|1|2`，train另有 `--head perception|dynamics|action-quality|all`；calibrate使用 `--scope development|final --version CHECKPOINT`；evaluate固定 `--split test --arms act,wam,no-prediction`；resume使用已有 `--run-id`，不能悄悄新建run规避原账本。profile临时模型仅用于吞吐／梯度成本测量，不能发布为已训练版本或给bootstrap产生动作正例。

- [ ] 先测预算拒绝、缺配额／模型／校准／预检时不能进入后续阶段；最后一轮只到WAM2，不加round3。profile≤7200秒且从259200总秒额中扣除；用实测吞吐和体积上界为余下所有配额、校准、预检与360正式尝试预留，不只估下一阶段。估计不足时停在not_admitted，不偷偷缩配额。
- [ ] 先测预检12布局完全独立，零残差与原生ACT对照／受限探针必须产生TCP、Z、腕部力、交接、IK、命令边界证据。探针归一化幅度默认≤0.1；零残差命令逐字节一致，原生Z链不变；unsafe／IK或协议不一致阻止正式评测。真实腕力仅作为审计，不回流WAM。
- [ ] 在最终冻结模型的12预检父布局中，各保存一个有效根并执行五个完整一秒候选短分支，作为独立物理预测诊断（另60短分支，无终局教师）。这些标签不用于训练、调参或选模；强制候选的提前安全终止按掩码和事件单列，不能拿反事实整秒控制当实际200ms滚动失败率。
- [ ] 先测正式结果严格120布局×3臂一次，包含视觉准备／链路失败；基础设施异常保留原行并停批，不自动替换／重跑。用手算配对表验证：b=12,c=0满足净+12且p=1/4096；b=12,c=12不满足提升。先主门后预测贡献检验；部分结果不能标test_complete／achieved。
- [ ] Run `./.venv/bin/python -m pytest tests/wam/test_cli.py tests/wam/test_pipeline.py tests/wam/test_evaluation.py -q` RED，再实现生命周期及真实组件调用；不以空壳命令、预填统计、mock全流程或fixture轨迹代替采集／训练。断点恢复不得造成重复训练阶段／正式尝试。
- [ ] 报告按N区间、视觉／安全／任务失败、物理预测对简单运动／保持基线的独立完整候选分支误差、随刷／校准、真实回退、残差、IK／接触／Z、教师覆盖、训练曲线和墙钟。预测对照必须使用实际执行一秒候选的分支标签，不能用滚动改策后的轨迹。
- [ ] GREEN 后跑 `./.venv/bin/python -m pytest tests/wam -q`，再跑原库回归与原基线差异对照；任何新回归修复或明确升级，不能用原有失败掩盖新失败。Commit `feat(wam): orchestrate guarded training preflight and paired evaluation`。全分支独立审查通过后才启动Task13。

### Task 13: 实际采集、三版训练与正式验证交付

**Files:** 工件根下 `runs/<run-id>/` 的配置、split、账本、轨迹、标签审查、模型／完整checkpoint、校准、曲线、预检、360结果、统计及代表回放；Modify `docs/WAM_VISUAL_V1.md` 仅写真实运行状态和可复现命令，不提交大工件。

**Interfaces:** Consumes Task12真实CLI及全部审查通过实现。Produces 可恢复WAM0／1／2、正式三臂结果和证据分层报告；未过任何门时只产生明确的incomplete／not_ready／not_admitted及原因，不冒称目标达成。

实际交付审查 `assert len(formal_attempts) == 360` 只是必要条件；还必须逐行核对120个父布局、三臂、模型哈希、真实终止证据及无选择性重跑，不能靠造一个长度360的数组通过。

- [ ] 先用 `./.venv/bin/python -m wam_sweep.cli verify-inputs --config configs/wam_visual_v1.yaml --root /Users/huazhi_luke/.local/share/wam-sweep-v1/` 核验复制输入与运行身份；运行profile（≤2小时且计入72小时），纳入五视角采集、共享ResNet、完整三头前向／训练、快照／续跑／压缩／磁盘峰值实测，不仅测无物理fixture。
- [ ] 准入后采集120条oracle感知启动数据，审查几何中心／接触链／遮挡／部分入区／掉队标签，训练感知；oracle仅感知监督。验证纯视觉准备与零残差／受限探针的原生控制协议，未过则停在具体原因。
- [ ] 依次采集并训练初始240条／600根（ACT续跑）→冻结WAM0；补采120／300（WAM0续跑）→WAM1；再120／300（WAM1续跑）→WAM2。每根12短分支／3终局，所有阶段审查覆盖和谱系；每阶段先感知、再动力学、最后动作／评价及校准，不跳过失败或配额。
- [ ] 冻结WAM2及同教师／同候选消融评分器后，用独立60父布局各一个根、各5候选做短分支和完整续跑最终校准；续跑器仍为该评价头定义的WAM1，不能悄悄换成WAM2。另有300短分支／300终局续跑，不计训练配额但计总资源；正式测的是滚动WAM2的实际成功率，不把条件评价校准当部署概率。缺有效校准根则记录缺口，不真值修补或替换失败父布局。
- [ ] 完成12布局原生对照／零残差／受限探针预检证据及另60条完整候选预测诊断短分支，准备失败也保留；计入至少36个预检回合和60短分支的预算。预检不能用于选择模型checkpoint。任何实际重放／前缀恢复成本计入总额。
- [ ] 冻结配置／checkpoint／测试清单，正式120布局三臂各一次共360。正式集不用于调参／选模；基础设施异常停批并保留当前状态，不自行挑选重跑。完成配对主门，过后才统计预测贡献；不足时准确报告已训练／测试完成但未达目标。
- [ ] 输出报告、每N指标与区间、预测／感知／安全分解、恢复验证、工件清单和代表回放。独立审查者按实际工件核对配额与指标，而非只读终端摘要。源代码最终测试及GitHub分支头核验后提交小型复现说明；只有实际达到完整规格目标才可把长期goal标complete。

## 自检与交接

原规格五大工作不缩减：契约／解码器→Task1～3；感知／视觉准备→Task4～7；分支与标签→Task4／5／8；多头训练与版本→Task9／10；预检／评测与完整运行→Task11～13。资源、恢复、统计与最终交付分别由Task2／10／12／13承接。

用户已选择 subagent-driven 执行。每项先RED后GREEN，提交后由独立任务审查者检查规格和质量；记录Task BASE，review diff不可用HEAD~1代替多提交范围。最后全分支审查，Task13再独立审查真实实验工件。没有批准任何删除旧实验、改全局Git代理／身份、PR／合并或真实机器人运行。

实施阶段允许配置工程默认值，但不能改变规格已锁定选择。算法尚待实验的部分明确为默认值，不把计划中的数字当成已得到的性能／预算证据；若门槛或预算无法满足，保留完整目标并报告缺口，不降低完成定义。
