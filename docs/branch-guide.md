# 分支与版本导航

迁移入口补充日期：2026-10-15（新加坡时间）。下列历史锚点保留2026-10-14核对值，不随README提交移动。

| 分支 | 用途与入口 | 固定回滚锚点 |
| --- | --- | --- |
| [feat/ubuntu-model-handoff](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/feat/ubuntu-model-handoff) | Ubuntu迁移与新模型AI交接；完整代码/资产、便携配置、示教契约、工作台与第四组Hybrid依赖。非ACT模型尚需适配，不含权重/数据/历史结果。见 [交接README](../README.md)。 | 迁移基底 `9102680d8e018b225ab34d36422572cd4b547166`；不改其他分支。 |
| [feat/act-five-view](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/feat/act-five-view) | 当前最新：五视角 ordinary ACT、fine-teached 布局/示教/落刷，训练、推理、工作台、批测；见 [运行指南](fine-teached-guide.md)。 | 本次完整源码及 A* 输出保护快照 [`6aa8386998d068013962db0faebb5c0bddea666e`](https://github.com/luke711-xyy/act-ur10-sweep-sim/commit/6aa8386998d068013962db0faebb5c0bddea666e)；更早开发点 `239cc8d` 不含完整更新。 |
| [main](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/main) | 历史 ordinary-ACT/V10 基线，不是最新五视角任务。 | `e8895a2d4c30942e0bbc23436f1caf97a45958e0` |
| [feat/robot-learning-workbench](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/feat/robot-learning-workbench) | 历史工作台/ordinary-ACT 开发线；本轮只补导航，不合并五视角新代码。 | `60e72dc8bfebd732c69f2858fa96bcb0a58501b6` |
| [feat/mt-act-goal-embedding](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/feat/mt-act-goal-embedding) | 可选类别 goal-token 实验；该分支曾运行 scalar ordinary ACT，必须核对实际 policy/config。 | `09ebcf37e9fbd21da303d785d688c494df1fa506` |
| [feat/objectact-bev](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/feat/objectact-bev) | 结构化对象/BEV 感知实验，独立于五相机输入。 | `5ce7a97a3c8e5575148cf1ee5e7db3a80d40116f` |
| [feat/v4-residual-rl](https://github.com/luke711-xyy/act-ur10-sweep-sim/tree/feat/v4-residual-rl) | V4 residual-RL 原型，不是本次 ACT 训练或 A* baseline。 | `35ca5bdaa782cfc8d7dd1951f9bff7222848418d` |

## 快速定位

- 当前六零件任务：`feat/act-five-view`，正式 batch 4 便携快照 `configs/five-view-fine-teached-300.yaml`。
- 原 V10：main 固定 SHA；goal-token：对应分支且明确选 embedding；对象/BEV 与 RL 是独立实验。
- 图表四条：ordinary act、goal-token act、fine-teached ordinary act · 100K、A* expert baseline。
  新图不保留旧第三个五视角结果；A* 只一次物理执行，不按成功结果挑选候选。
- 潜在簇中心面积均匀，不意味着最终零件质心均匀。相同输入尺寸不意味同一数据/行为。
  新示教改变布局、选件、路线与落刷，本次 batch 4 也与旧设置不同；不能把差异全归因于一个参数。

## 安全查看与回滚

1. 先 `git status --short`、`git branch --show-current`、`git log -1 --oneline` 确认检出及未保存工作。
2. **训练或冻结批测运行中不 checkout、reset、pull 合并或改配置。** 每条批测都核验 sim 源码。
   历史 frozen YAML、plan、metadata、结果不随 Git 回滚改写。
3. 只读查看：`git show <固定SHA>:configs/default.yaml`；比较：
   `git diff <固定SHA>..feat/act-five-view -- sim configs`。
4. 作业结束且工作区干净后，才可 `git switch --detach <固定SHA>` 查看历史；
   回到最新用 `git switch feat/act-five-view`。不要用 reset --hard 覆盖数据或未提交修改。
5. 撤销已共享修改应先评估 `git revert <commit>` 保留历史；不强制推送、不自动合并到 main。

Git 只能还原源码/config，不会还原未上传的模型、归一化统计、示教、运行快照或已删媒体。
完整复现还需对应 checkpoint/processor/training_state、数据 manifest、冻结配置和种子计划。

## 发布范围

最新线提交代码/config/tests；五个已有远端分支仅补 README 导航，保留原实现。
不上传 runs、datasets、exports、虚拟环境、密钥、视频或运行日志。文档更新不是代码合并。
