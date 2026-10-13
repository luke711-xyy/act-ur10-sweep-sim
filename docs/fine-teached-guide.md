# fine-teached 五视角运行指南

分支 `feat/act-five-view`。命令从仓库根目录手工运行；保存文档或 checkout 不自动训练。
模型、示教图片、归一化统计与运行产物不上传 GitHub，复现需要另外取得对应数据和权重。

## 工作台

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[act,web]'
python -m sim.web.run --config configs/default.yaml --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765/`。五 RGB 视角、36 维状态（含 200 ms 历史）、
1 维 float32 目标数经 Linear 投影、4 维动作；ordinary ACT 不是类别 embedding。
Inference 可编辑完整 checkpoint 路径，刷新/空 API 默认选择本次训练最新完整保存点。
新视频名含实际 checkpoint；seed 仍在记录里。Play 旁导出已保存示教/回放，不重跑物理。
推理与训练会竞争 MPS。中断训练必须经用户明确要求，不自动续训。

## 布局与示教

潜在簇中心：圆心 `(0.05,0)` m、半径 0.10 m，`r=R*sqrt(U)` 面积均匀采样。
逐件最大散布 0.14 m、最小间距 0.07 m；去掉旧 X ≥ −0.08 m 截边。
逐件 spawn/目标排除约束仍在，不裁剪中心圆盘；零件最终质心不被宣称面积均匀。
没有显式圆盘模式的旧配置保留原分支和随机数消费顺序。

所有目标使用 coverage A*、Y 边缘连续子集选件、基于本布局最靠 +X 零件外缘的
7 cm 动态落刷边界和路径首次交点（含刷子半厚度，Y 随路径）。收齐全部目标后解除
专家高度阈值，其他安全/成功条件不变。需要新数据时选**空目录**，不覆盖现有数据：

```bash
python -m sim.act.collect_dataset --config configs/five-view-fine-teached-300.yaml \
  --out runs/new_fine_teached_dataset --approved-plan --episodes-per-target 50 \
  --composition mixed --image-format webp-lossless --workers 2
```

采集可筛选失败并换种子凑满成功示教；**评测不可以**。workers 2 是选择，不表示每台机器最快。

## 本次训练配置和保存点

[完整便携 YAML](../configs/five-view-fine-teached-300.yaml) 从实际冻结训练配置转存：
只把本机绝对数据/模型/视频路径换为仓库相对路径，去掉 `_source_config`。
默认 YAML batch 2 与本次 batch 4 分开记录。300 条、目标 1–6 各 50 条；FP32/MPS、AMP
关闭；AdamW `1e-5`；approach/descent 采样占比 0.35；chunk 25、execute 5。

```bash
python -u -m sim.act.train --config configs/five-view-fine-teached-300.yaml \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --out runs/act_model_five_view_ordinary_disk14_landing7_300_200k_20261013 \
  --steps 200000 --batch-size 4 --no-amp --checkpoint-start-step 100000 \
  --checkpoint-every 100000 --checkpoint-extra-steps 50000 --keep-checkpoints 3 --no-trackio
```

保存 50K、100K、200K，跳过 150K。中断可能另存实际中断步，不能冒充 200K 完成。
用户于 2026-10-14 要求停止，本次停在 103,243 步；原 100K 未改写。
以后**另获续训指令**，同一命令加上以下参数从 100K 恢复优化器/RNG：

```bash
--resume runs/act_model_five_view_ordinary_disk14_landing7_300_200k_20261013/checkpoints/step_100000
```

200K 是总步数，不是再训练 200K；开始前核对输出目录、checkpoint 保留策略、数据及配置身份。
不使用测速权重、不中途改 batch、不自动恢复旧定时任务。
若同一输出目录保留了额外的 `step_103243` 中断点，续训到 200K 时 `keep=3` 可能裁掉
计划中的 50K。此时需先明确保留策略，例如续训用 `--keep-checkpoints 4`，保留三份计划点
及中断点；不要未经授权移动或删除已有保存点。本轮没有启动任何续训。

## 120 条 ACT 与一次机会 A* baseline

```bash
python -u -m sim.act.batch_evaluate --config configs/default.yaml \
  --dataset runs/act_dataset_five_view_rudra_300_smooth320_v1 \
  --model-root runs/act_model_five_view_ordinary_disk14_landing7_300_200k_20261013 \
  --out exports/new_act_heldout_batch --seed-start 2120000 --pipeline-media
```

通用入口在准备时固定 ready 模型；若当时 200K 也完整发布，会包含两模型共 240 条。
**本次已经冻结的评测只有 100K/120 条**，不追逐以后的 latest。
目标各 20 条，种子和稳定布局独立、未用于训练；失败计入、媒体故障另报。
每条前核对 sim 源码、冻结 YAML/plan/checkpoint；运行中不改它们。

```bash
python -u scripts/evaluate_astar_baseline.py --act-batch exports/new_act_heldout_batch \
  --out exports/new_astar_single_attempt --wait-act
```

工具位于 sim 外，避免更改正在运行的 ACT 冻结源码；调用现有专家选件/落刷/执行路径和导纳，
不加载 ACT、不调用 `run_expert_episode`。仅以几何代价确定一条路线，最多一次物理执行；
不可行也计失败，不换候选或 seed，不保存 RGB/MP4。规划搜索节点不等于多次物理机会。
与 ACT 同一组 120 seed/目标/初始姿态配对；无论是否传 `--wait-act`，ACT 真正完成均为前提。
该参数只决定等待还是立即报错。每个布局消耗一次机会：无可行路线是失败，物理执行次数为 0，
不为了凑足物理次数增加兜底路线；汇总分别记录 120 个机会和实际物理执行数。
中断过但未正式登记的布局不能自动再跑，需要用户决定。
A* 读取仿真初始零件几何，与示教规划相同；这不是与 ACT 五路图像等价的感知输入对照。

A* 原生成功条件包含任意恰好 N 件全部入区、无部分重叠、交付/接触要求与稳定 1 秒；
选中零件索引只作规划来源记录。ACT 为首次入区后 2 秒窗口；原生严格成功另列。
两者的硬物理帧预算都是 2000 帧（25 Hz 下约 80 秒）；该评测上限覆盖示教采集较短的上限，
不改变 A* 路线、落刷或力控，也不代表两者的原生终止契约相同。
共同图表用最终留区数量误差 0/1/2/≥3、±1 比例、平均绝对误差，六目标等权。
四条：`ordinary act`、`goal-token act`、`fine-teached ordinary act · 100K`、`A* expert baseline`。
旧第三条五视角不加入新图；旧图不改写，旧两个模型计数协议未重新核验，不声称统计显著性。

本次用户授权临时 ACT 视频核验、整批完成后删除视频及残余帧；保留结果、CSV、图表、
配置/计划、日志、数组/数值轨迹。这不是全局删除规则，更不清理示教图片。
删除后不能直接恢复原媒体。A* 从一开始不渲染，避免空间/编码开销。

## 验证与回滚边界

专项测试覆盖布局、落刷、计数、互斥、保存点、工作台与回放/编码。
全套仿真测试曾主动中断，**不称全套通过**；发布源码不表示模型收敛或真实任务成功。
评测未满 120+120 条前不写最终成功率。分支名不替代模型/schema/config/manifest 身份。
安全回滚见 [分支导航](branch-guide.md)，运行中不要 checkout/reset/pull 合并。
