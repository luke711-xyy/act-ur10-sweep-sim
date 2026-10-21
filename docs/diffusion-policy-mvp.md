# Diffusion Policy 本机 MVP

独立分支 `codex/diffusion-policy-mvp`，基线 `8c19ef1`。
算法来源、最小改动见 [vendor 来源](../sim/diffusion/vendor/PROVENANCE.md)，
完整约束见 [批准计划](superpowers/plans/2026-10-21-diffusion-policy-mvp.md)。

## 环境和工作台

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-diffusion.lock.txt
.venv/bin/python -m sim.web.run --port 8773 --diffusion-workspace
```

新工作树不复制示教与旧模型。工作台选择 Diffusion Policy 后自动使用原300条
示教的只读绝对路径，以及本工作树的 `runs/diffusion_policy_mvp`。
ACT 请求默认行为不变；新工作树运行旧 ACT 时需明确选择原模型的绝对路径。
不要在原 ACT 环境安装此锁定文件，不要把 DP 输出指向原实验目录。

### 实时训练面板

工作台 Training 区的 Loss 面板每3秒只读刷新当前作业：原始 loss、50条移动均值、
优化器更新步数/上限、最近100条 loss 均值、秒/更新和更新/分钟。
刷新页面或单独重启网页服务后会自动接回已登记的活动训练作业；左侧 Policy
是新作业的启动选项，曲线上方单独标明实际被监控的策略、阶段和作业ID。
DP小样本和全量训练不混画，图线不是验证集指标；100,000是更新次数上限，
原24小时累计预算不变。ACT沿用已有训练日志，未记录的指标显示为空。

面板不加载模型，不导入Torch，不改训练设置。日志增量读取，发送给浏览器的
曲线最多1200点并保留尖峰；速度按日志中的实际步数差和耗时差计算。
超过30秒无新指标会提示未更新（保存检查点也可能导致等待），不显示实时速度；
进程退出、训练结束、接口断连也会明确标注，保留最后可读曲线。

## 训练、停止、恢复

```sh
.venv/bin/python -m sim.diffusion.train
# 可选：仅跑小样本门（不会启动全量）
.venv/bin/python -m sim.diffusion.train --pilot-only
# 对同一已存在输出只能显式恢复，保持原 steps/config
.venv/bin/python -m sim.diffusion.train --resume runs/diffusion_policy_mvp/full/step_000500.pt
# 本计划的持久作业：全量完成后自动运行六个一次性验收，不挑结果重跑
.venv/bin/python -m sim.diffusion.pipeline --resume runs/diffusion_policy_mvp/pilot/step_000100.pt
```

第一次命令自动先跑固定12窗口小样本门，通过后重新seed42初始化全量模型。
失败停止，不改分辨率、相机、历史或网络。SIGTERM/工作台Stop在当前更新结束后
保存；OOM等异常保留最后已经完整发布的恢复点，不保存可能部分更新的权重。
没有验证集，`final_ema.pt` 是最后完成更新的EMA，不是最佳验证模型。
24小时累计预算、30分钟累计pilot预算与磁盘门由 `status.json` 保持，
显式恢复不能重置；检查点保存完整优化器、LR、EMA、随机数与采样游标。
恢复必须同时保留原run的 `status.json`，缺失或旧于检查点则明确拒绝。
DP工作台模式只读浏览原示教，拒绝删除原记录，新回放和视频保留在本工作树。

`curves.jsonl` 保存训练曲线，`training_config.json` / `data_audit.json`
记录配置与只读数据身份；`pilot/` 和 `full/` 各保留最近两份恢复点。
整个新工作树（包括环境和日志）纳入20GiB上限，余量25GiB；预留下一份恢复点
所需空间，不删旧数据、旧实验。

## 推理与一次性验收

```sh
.venv/bin/python -m sim.act.evaluate --policy-type diffusion \
  --model runs/diffusion_policy_mvp/final_ema.pt --seed 123 --target-count 3 \
  --record-replay --video-root runs/diffusion_policy_mvp/videos
# 全量训练正常结束后，运行全部六个预注册种子，每个只能一次
.venv/bin/python -m sim.diffusion.acceptance
```

DP 25Hz捕获两帧、5Hz同步暂停仿真推理、执行前5步；不做ACT跨块平均或200ms偏移。
物理执行、接触后Z导纳、安全限幅和任务终止与ACT共用。
DP预测/执行异常会终止作业并留下错误日志，不会悄悄持续保持姿态。
每次查询打印墙钟延迟；不声明200ms实时达标。

验收记录在 `acceptance/`：先核对六个实际布局与示教不同，再逐个执行
81000001..81000006对应N1..6。正常任务失败不重跑；基础设施故障停止，
不换种子、不按结果换检查点。六次结果不是正式泛化或优于ACT的证据。
