# 新前上方机位：复用旧300条实际种子的示教采集

2026-10-22用户授权：新机位采集300条，删除此前六条预览数据。
原始 `/home/luke/dataset-ur10` 和现有8773 Transformer训练保持原样。

入口 `scripts/collect_rudra_seed_matched.py` 读取原manifest的全部300条成功专家记录，
目标1–6各50条，逐条复用实际保存的 `(target_count, seed)` 和布局ID。
64条旧记录的实际seed与最初requested_seed不同，本次使用实际seed，
不使用旧申请seed重新生成清单。原manifest SHA256：
`0bec71a7c2c3f06f7b6749f2b708a438c951c125c18ed0f48fe4036c8734431c`。

当前原生尺寸结果目录：`runs/rudra_front_300_seedmatched_640x480_20261022/`。
最初的`runs/rudra_front_300_seedmatched_20261022/`曾保存60条320×320成功示教，
以及7个固定seed失败记录。60条已全部在原生尺寸目录重采，并核对全部NPZ数组、
所有原图引用及每路首末帧尺寸后，清除了过时的方形图像/数组副本，释放约7.1GiB；
原manifest、日志及配置保留为历史。审计`square-replacement-audit.json`与清理记录
`square-replacement-cleanup.json`位于原生尺寸目录。
`plan.json` 固定300个槽位，保留原manifest顺序和对应来源记录摘要。
`provenance.json` 固定原manifest副本、已通过的接触报告、任务源配置、
双机位场景配置和采集配置；原生场景重置也使用目录内冻结的任务源配置。

使用当前前上方top及原生wrist，保存原生完整640×480图像，不做方形缩放。
保留原A*规划、候选物理筛选、目标子集规则和成功判定；沿用此前验证的
trial4 1N两阶段导纳与刷底摩擦，详见 `rudra-six-demo-collection.md`。
这是A*专家示教生成，不是模型评测；同一seed内的原候选筛选照常进行。
不另加物理试跑，不变更控制参数，不启动新训练。
最初阶段不替换seed；2026-10-23用户明确授权失败重试或更换seed，直到六个目标共300条成功。
以下为最新补足模式，原始固定plan及来源manifest仍保留不变。

成功条目直接使用原 `ActDatasetWriter`，标记 `expert/train`，schema-v5，
36维状态、1维目标条件、4维动作、有效掩码、100Hz trace和双路无损WebP。
不额外导出逐条MP4；图像即示教观测。`frame_count`契约与训练数量门槛保持不变。

已知无计划或实际任务失败写入 `events.jsonl`，保留实际测得数量；未知数量保持null。
补足模式将失败槽位换为原seed之后的下一个全局未用seed，跳过原300seed及所有已分配seed。
分配在执行前写入并fsync到`attempt_allocated`事件；恢复时复用未记账完成的同次分配。
`seed`为实际执行seed，`requested_seed/source_seed`为原槽位seed，`generation_attempt`为当前尝试序号；
旧来源的requested seed和generation attempt另存于`source_requested_seed/source_generation_attempt`。
换seed成功条目有`seed_replaced=true`，不再将它表述为原布局的重采结果。
程序、写入或未知执行异常会停止，已有数据和失败记录保留；已知任务失败继续换seed补足。
不足12GiB时不派发新条目，等待在途完成释放暂存；仍不足则进入`waiting_for_disk_space`，
空间恢复后继续，不为腾空间擅自删除其他数据或权重。
`progress.json`区分saved、当前未修复的failed、累计failed_attempts和remaining；
成功条目不重采，每目标恰好50条，只有300条成功保存且原训练manifest校验通过才标记complete。
此前失败始终留在事件和最终attempt_failures中，不当作成功率评测。

最新持久进程使用用户systemd服务 `ur10-rudra-front-300-640x480-fill-20261023.service`，
独立布局并行，每路内部仅1个候选worker。使用4路，`worker-count.json`可在1–4路内
调整并发资源；减少时等待当前条目结束。CPU总上限8核、内存总上限3GiB，
不使用swap、不开自动重启。
日志 `fill-collection.log`。独占文件锁阻止同目录重复启动。
中断后的显式resume读取已提交manifest，避免已保存槽位因日志缺失被重复执行；
补足模式会为已知失败分配新seed；已提交的成功槽位不会重采。

```sh
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /home/luke/.venvs/ur10-diffusion-cuda/bin/python -u \
 scripts/collect_rudra_parallel.py \
 --out runs/rudra_front_300_seedmatched_640x480_20261022 --workers 4 --fill-missing
```

首次准备使用`collect_rudra_seed_matched.py`，需新空目录和 `--gate-report`；
使用 `--prepare-only` 只冻结清单/配置，
不运行物理；已有服务运行时不能并行执行上面的恢复命令。
并行入口通过独占锁恢复已提交条目，父进程唯一分配固定槽位，子进程使用原writer
的manifest锁提交。每条子进程完成后退出，避免长期积累MuJoCo/EGL缓存。
只在本次子进程启用uint8磁盘memmap图像缓存，采图和编码阶段每8帧释放缓存页；
像素不压缩变更，仍使用原lossless WebP method1。默认推理/采集未开启此缓存。
未知子进程错误会终止全部worker，不自动重试。已退出worker的原图暂存目录可安全清理，
已提交示教、异常写入现场与历史结果不作为暂存删除。

首条原生尺寸回合seed4200256，目标6，803帧/1606张图全部640×480解码通过；
所有NPZ数组与对应320×320旧采集逐项相同。报告`first-native-episode-audit.json`。
单路原生采集4分32秒，服务内存峰值约2.48GiB，作为后续并行的参考。
编码实测method4约省3.6%但耗时1.7倍；method6约省9.4%但耗时21倍，
因此保留method1，避免用显著更慢的编码冒充采集提速。
按原203121帧和首条原图大小估算约86GiB，空间不足时按12GiB余量门等待，
不会擅自删除原数据或训练权重。

2026-10-23补足切换时，先停止旧parent继续派发，并让四个在途worker全部成功提交后退出，
共保留96条成功：目标6为43、目标5为50、目标4为3。之后优先分配目标6的七个补足布局。
`completion-policy.json`记录本次用户授权；初始冻结plan与provenance文件不重写。

2026-10-23全量采集完成并正常退出：300条严格成功专家示教，目标1–6各50条，
共207179帧、414358张双路640×480无损WebP。原`validate_training_manifest`
通过，300个实际seed与布局指纹均唯一；全部图像引用、文件长度和无损尺寸头通过，
每条每路首尾帧共1200张完整解码通过，其余图像未逐张完整解码。
285条保留原实际seed，15条按授权更换seed（目标6/3/2/1分别7/3/1/4条）；
21次失败尝试全部保留在事件中。原来源manifest和冻结配置哈希保持不变。
本地结果目录中的`completion-audit.json`及`layout-seed-audit.json`记录最终审计；
示教载荷及运行目录不随代码分支上传，定时检查已停用。
