# Rudra 双视角 DINOv3 + ACT：200K，120 布局评测

- 固定 checkpoint：200000 步；`model.safetensors` SHA-256 `72827d469c64400b475df1b6f3489d04e96b0ff5c7fc67a1bd9b831afbf0fb45`。
- 目标数量 1–6 各 20 次，种子 2120000–2120119，每个布局一次物理执行；使用现有 Rudra ACT 原生异步调度与外部动作融合系数 0.01，非 A* 前段或 PAC-ACT。
- 严格成功 29/120（24.17%）；最终数量匹配 30/120（25.00%）；平均绝对数量误差 1.167。二者定义不同。
- 91 次未严格成功：collection_window 89，timeout 2。

[统计摘要](summary.json) · [逐回合 CSV](episodes.csv) · [逐回合 JSONL](episodes.jsonl) · [布局计划](episode_plan.json) · [密度矩阵](density_matrix.csv) · [热力图 PNG](success_density.png) · [SVG](success_density.svg) · [元数据和校验和](metadata.json)

[工作台截图](workbench_screen.jpg) · [俯视输入 top](workbench_top.png) · [腕部输入 wrist](workbench_wrist.png)。这三张是截图时工作台当前画面，和下面的成功回放不是同一时刻。

[六零件严格成功回放，seed 2120119](success_target6_seed2120119.mp4)：根据原回合保存的仿真状态按需渲染，未重新运行策略。批测时自动生成的视频数量为零（见原 summary）；此视频是事后按用户要求导出的。

原始逐回合数据与计划中本机绝对路径已从公开副本移除，其他结果字段保留。checkpoint、数据集和各回合仿真状态未纳入仓库。[运行时评测脚本快照](../../experiments/rudra-act-200k-eval120/README.md)供审计，不能单独复现。
