# Ubuntu 现有 300 条 A* 示教：兼容性审核与复用

本页是开发交接材料，不代表数据已经检查或新模型已经实现。用户确认 Ubuntu 上已有 **300 条双机位 A* 成功示教**；数据目录、每 N 数量、schema、图像分辨率、采样频率、标定及失败覆盖尚未核实。不要默认每 N 50 条，也不要默认它们完全符合新 WAM。

## 1. 数据集合与分区

- `D_source`：用户已有的 300 条源记录，先保留原样和原始 manifest。
- `D0`：审核后实际兼容、分配为训练用途的记录集合；不强制截成旧计划的 120 条，也不默认 300 条全部进入训练。
- 按 episode 和物理布局分组去重后冻结用途清单；同一布局的不同 N 记录归同一组。检查 layout ID、seed、fingerprint 及配置来源，不能仅按图片或帧随机切分。
- train／calib-fit 可用于对应拟合；dev 只验收／选模；最终 test 不训练、不拟合噪声／归一化、不选 checkpoint。已有成功数据也不能同时当未见测试集。
- 若尚无合格留出数据，先报告缺口；不能把训练误差包装成独立验收，也不擅自把全部源记录重采或删除。
- D0 大小确定后，BC 的一个等效 epoch 按实际合法训练窗口数计量。BC 100 等效 epoch、一次 60 场且至少 12 成功、MBRL 新增总上限 30,000 宏步等其余已定协议不因源数据有 300 条而自动扩张。

## 2. 分阶段复用矩阵

| 阶段／用途 | 可优先复用的真实已有内容 | 必须另核实的条件 |
|---|---|---|
| 数据与资源预检 | manifest、实际双路图像、NPZ 字段、采集配置和版本 | 真实尺寸、时刻、频率、缺帧、相机 ID、动作单位与记录来源；文件中有字段不等于值真实有效 |
| 感知／几何／ROI | 原分辨率 overhead／wrist 图像 | 实例 mask／类别与完整对象参考点标签、对应标定；RGB 本身不等于有监督标签 |
| 启动与 belief | reset 起的连续图像、实际 q／力／控制记录 | 0／0.2／0.4 秒三帧是否真实存在，因果同步、相机位姿和最小启动模型能否重编码；缺失不能用未来帧补 |
| 初始 BC | 同域专家增量命令、N、有效 mask、真实机器人上下文 | 输入由当前模型因果得到 belief/h；命令须对应 `U_cmd`，不是实际 TCP 位移或低层处理后的参考，且落在策略分布支持内 |
| 完整概率动力学／事件监督 | 有来源的对象轨迹、原生 q(6)＋dq(6)＋有符号 Z 力(1)，共 13 维，以及控制与事件记录 | 当前方案需要每 0.2 秒宏步的 20 个 100 Hz 子步及有效前缀、对象目标、`C_ctrl`、标定／事件。仅帧级 pose 或 q 不足以冒充完整监督 |
| 误差拟合和独立验收 | 标签／标定相容且用途明确的序列 | calib-fit 仍属训练侧；dev／test 不参与拟合。真实检测器的误差不能由干净 GT 点代替 |

记录可只适合部分用途；不要因缺少高频动力学字段就把可用 RGB 或合法 BC 标签全部丢掉，也不要把“部分可用”标成“全链兼容”。成功-only 数据对失败／跌落／碰撞等稀有事件覆盖不足须报告；不能伪造失败标签。后续真实执行中新产生的失败记录照常保留和计费。

## 3. 已知旧格式的具体风险

本仓库旧 ACT 格式通常为 `manifest.jsonl`、每回合 `arrays.npz` 与 `overhead_*.png`／`wrist_*.png`；这里只能把它作为待检查候选，不代表 Ubuntu 一定相同。

- `sim/act/dataset.py:96–103` 对部分缺字段补零，对缺时间使用序号；`112–131` 对对象 pose 和参考量也有默认值。审核应结合来源代码和实际值，不能仅检查 key 是否存在。
- `dataset.py:133–168` 写入帧级 q／wrench／object_pose／reference，未因此提供完整原生 dq 或 100 Hz 子步；旧 36 维 ACT state 也不是新 13 维接口。
- `sim/act/interface.py:148–153` 的旧环境向量含真实 collected count。保留作裁判／标签遥测，不能直接传进新 actor／filter。
- `sim/act/rollout.py:430–485` 在执行前记录动作 delta；`692–703` 又回填执行后的 applied reference／Z 信息。未来动作结果不能倒灌为当前根状态。
- `collect_dataset.py:508–550` 的目标 N、布局 fingerprint、fps、力来源等是有用元数据，但不能证明新 schema 已完整。旧 manifest 的固定数量验收不应直接套用到新 D0。
- `sim/export_dataset.py` 另有逐 stroke 的旧导出格式和不同动作定义；不要仅凭 NPZ 扩展名混读。LeRobot、HDF5 或其他结构先列库存，再写对应适配器。

## 4. 合法补齐与禁止做法

1. 优先寻找同一采集过程的原始高频日志、完整配置、模型／代码版本、相机标定和控制状态记录；所有恢复字段带来源及处理版本。
2. 相机几何只可依据准确的机位配置、图像变换、机器人模型／TCP、对应时刻 q 和手眼恢复；可见 mask 中心不直接等于完整对象中心。
3. 实例标签可来自已有真实标签、合格人工标注或经一致性验证的仿真重录。投影几何、可见实例 mask 与完整对象几何必须区分。
4. 若保留了完整初始物理／控制状态、模型配置和随机状态，可验证仿真回放并重新记录缺失标签。仅一个 seed 不能保证恢复原轨迹；不一致时不可把新 GT 拼到旧 RGB 上。重新记录的结果标明为重录数据并单列成本，不伪称原始记录本来有这些字段。
5. 不把低频位置差分冒充原生 dq／100 Hz 真值，不放大旧图冒充原始高分辨率，不补零、复制帧或查询真实未来为 actor 补状态。无法合法恢复时保留可用子任务，缺口定向处理，不默改已定世界模型结构。
6. 原数据只读；适配后的 manifest／缓存／审核报告另存新目录。不要把示教图像、NPZ、模型权重或包含本机绝对路径的整份清单提交到 GitHub。

## 5. 首次只读盘点命令

以下片段仅适用于上一节的 **legacy ACT manifest＋NPZ＋双图片列表**，需要 NumPy、Pillow。先把 `DATASET_ROOT` 指向 Ubuntu 上现有目录。它不修改数据、不训练、不输出绝对路径或大数组，也不自动认定兼容。

```bash
export DATASET_ROOT='/path/to/existing/astar_dataset'
python - "$DATASET_ROOT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(sys.argv[1]).resolve()
manifest = root / 'manifest.jsonl'
if not manifest.is_file():
    raise SystemExit('Legacy manifest layout not found; inventory actual format first.')
try:
    lines = manifest.read_text(encoding='utf-8').splitlines()
except Exception as exc:
    raise SystemExit('Manifest read failed: ' + type(exc).__name__)
records, invalid = [], 0
for line in lines:
    if not line.strip():
        continue
    try:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('not a record')
        records.append(value)
    except (ValueError, TypeError):
        invalid += 1
def numeric_tag(value):
    return str(value) if isinstance(value, (int, float)) else 'missing_or_invalid'
print(json.dumps({'manifest_lines': len(lines), 'valid_records': len(records), 'invalid_records': invalid,
    'target_counts': dict(Counter(numeric_tag(r.get('target_count')) for r in records)),
    'success_true': sum(r.get('success') is True for r in records)}, ensure_ascii=False))
if not records:
    raise SystemExit('No valid records; stop before adaptation.')
record = records[0]
print(json.dumps({'first_record_keys': sorted(record.keys())}, ensure_ascii=False))
def local_file(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate
array_path = local_file(record.get('arrays'))
if array_path is None:
    print('First arrays entry missing/outside root; stop NPZ inspection.')
else:
    with np.load(array_path, allow_pickle=False) as arrays:
        summary = {}
        for key in arrays.files:
            try:
                value = arrays[key]
                summary[key] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
            except Exception as exc:
                summary[key] = {'unreadable': type(exc).__name__}
        print(json.dumps({'first_npz_fields': summary}, ensure_ascii=False))
        if 't' in arrays.files:
            try:
                times = np.asarray(arrays['t'], dtype=np.float64)
                if times.ndim == 1 and times.size > 1 and np.isfinite(times).all():
                    delta = np.diff(times)
                    print(json.dumps({'first_episode_dt': {'count': int(delta.size),
                        'min': float(delta.min()), 'median': float(np.median(delta)),
                        'max': float(delta.max()), 'nonpositive': int((delta <= 0).sum())}}))
                else:
                    print('First t is not a finite 1-D sequence with multiple samples.')
            except Exception as exc:
                print('First t inspection failed: ' + type(exc).__name__)
for camera in ('overhead', 'wrist'):
    values = record.get(camera)
    path = local_file(values[0]) if isinstance(values, list) and values else None
    if path is None:
        print(json.dumps({'camera': camera, 'status': 'missing_or_unrecognized'}))
        continue
    try:
        with Image.open(path) as image:
            print(json.dumps({'camera': camera, 'first_image_size': list(image.size),
                'mode': image.mode, 'listed_frames': len(values)}))
    except Exception as exc:
        print(json.dumps({'camera': camera, 'unreadable': type(exc).__name__}))
PY
```

首回合盘点仅用于确认实际格式，不能替代全量字段、图片存在性、同步和兼容性审核。下一步逐回合汇总同样统计，不用大数组打印做诊断。

## 6. 审核报告模板

- 来源：用户报告 300 条；实际有效记录数／success 数／每 N 数量：`not_checked`。
- 采集代码、模型／配置与力／相机／动作版本；布局分组、重复和 split 冲突：`not_checked`。
- 原始分辨率、图像／时间完整性、频率、三帧启动与控制状态：`not_checked`。
- 各用途资格：RGB 可读／感知监督可用／启动可重编码／BC 可用／完整动力学监督可用，分别列通过数量和拒绝理由。
- 缺失字段及可追溯恢复来源；失败／稀有事件覆盖；需补数据：明确列出，未知写 `unknown`。
- D0／calib-fit／dev／test 冻结清单与布局交集检查；实际额外采集和处理成本。
- 结论使用 `ready`、`partially_usable`、`incompatible` 或 `needs_evidence`；这些是数据资格，不代表模型训练成功。
