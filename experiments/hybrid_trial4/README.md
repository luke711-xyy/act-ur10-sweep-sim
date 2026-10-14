# 第四组 Hybrid 控制代码的仓库内依赖

这里保留2026-10-14用户批准的第四组 A*→ACT 控制入口所需的14个 Python 依赖，
来源是当时评测 metadata 中冻结的 exports 代码。没有复制实验载荷、训练数据、模型或物理历史结果。

导入入口：

```python
from experiments.hybrid_trial4.runtime import load_runtime
runtime = load_runtime()  # 不创建策略、环境或执行任务
assert runtime.TRIAL_SPEC.force_N == 1.0
```

正式新批入口是仓库根的 `scripts/evaluate_hybrid_paired_batch.py`，
不是 vendor 子目录中的历史 main。后者可能引用未发布的历史批次文件，其独立 CLI 不受支持。
控制函数/类保持来源实现；仅把顶层 PROJECT/BATCH 机器路径机械改成仓库相对路径。
`provenance.json` 列出来源路径（相对 exports）、原SHA256、迁移SHA256及路径变化。

运行导入沿用历史模块结构，`phases` 临时模块绑定在导入后恢复；不对活跃正式批测热更新。
第四组规格由 `runtime.py`/批次入口明确选定，不能靠运行历史“多组调参”脚本隐式选组。
总Z修正±2mm、1N目标、寻找10/400/10、维持10/800/10及安全边界的详情见根README。

仅支持在新机器上生成新的、本机完成基线配对评测。旧 Mac 批次的绝对路径、纳秒时间戳、
checkpoint/数据/源码指纹和已消耗机会是正式历史，不允许通过更新seal绕过校验恢复。
这次迁移与独立专项测试不意味着控制成功、Ubuntu物理验证或120条评测完成。
