# PAC-ACT 只读工作台

沿用 residual 工作台的白底绿色布局、EMA 曲线与完整回合分页，使用 PAC-ACT 自己的数据接口；不读取或修改 SAC 旧数据。

```sh
.venv/bin/python -B scripts/monitor_pac_act.py --out /path/to/current/pac-act-run --port 8897
```

浏览器打开 <http://127.0.0.1:8897/>。`--out` 显式指定当前任务；面板不会自动寻找“最新目录”，也不会自动启动或恢复实验。

## 数据含义

- `status.json`、`preflight_design.json`、`acceptance.json`：预检阶段、预算、工程检查与是否启动正式训练。预检结束后使用报告中的已完成耗时，不继续累计。
- `progress.json`、动作/块 journal：正式训练计数。预检没有这些 journal 时，只读取 `preflight_*_actor/learning_blocks.jsonl`，按实际执行前缀计数；只有对应策略版本的更新已接受，才计入“已用于更新”。比较组和更新后验证不冒充 PPO 训练样本。
- `updates.jsonl`、`active_update.json`：当前有效谱系中接受的更新。曲线以更新次数为横轴，浅色原始值、绿色 EMA；缺失值断线，只有一个记录时显示数据点。
- `episodes.jsonl`：全量阶段/调度筛选、分页、跳页。预检各组与正式训练分别统计；行政截断不参与任务成功率或数量误差分母，未知数量不补零。
- 每回合 `media.json`、`replay.mp4`、`force_Z.png`：按需提供回放与曲线链接；视频支持 Range 分段读取，不重新推理或生成视频。
- `checkpoints/index.json`：区分验收、里程碑与滚动快照，只核对元数据、文件存在和大小；不加载权重，也不把这项核对称为重新确认可恢复。

预检中没有记录的先验代价、物理探索尺度或优化回报显示“暂无”；不会把默认参数、其他任务数据或权重推算值当作本次实测日志。

## 隔离与核验

服务只绑定本机地址，仅提供 GET API 和限定的媒体路径。没有训练控制接口，不导入模型/学习器、不写实验目录、不访问帧图或物理轨迹。单个普通 JSON/回合日志读取上限为 16 MiB，超出时告警，不悄悄截断为完整历史。

修改工作台时只改可编辑 checkout 的展示代码，不能改正在执行的冻结 bundle。切换任务只重启这个展示服务，不停止实验进程。

专项检查：

```sh
.venv/bin/python -B -m pytest tests/test_pac_monitoring.py
node --test tests/pac_dashboard.test.mjs
```
