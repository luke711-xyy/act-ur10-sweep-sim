# Rudra 双视角 DINOv3 + ACT

分支：`codex/rudra-dinov3-act-two-view`。
基于原生 Rudra 桌面环境，复用已发布的冻结 DINOv3 接入，提供双路640×480示教训练、
本地工作台、checkpoint续训、因果回放和视频导出。

**另一台电脑从这里开始：[双视角配置指南](docs/rudra-dinov3-act-setup.md)。**
配置脚本只生成本机YAML，可选择只读检查示教；不会启动训练、仿真或下载模型。
数据集、训练权重、视频、运行日志和虚拟环境不包含在Git仓库中。

| 项目 | 本分支配置 |
| --- | --- |
| 图像 | `top, wrist`，顺序固定，RGB 640×480 |
| 视觉编码器 | 冻结DINOv3 ViT-S/16，`timm/vit_small_patch16_dinov3.lvd1689m` |
| 接入 | 空间patch → 2×2平均池化 → 可训练384→512投影 → 原ACT/VAE |
| 观测/动作 | 36维状态、1维目标数量；25×4增量动作块 |
| 优化 | batch 6，FP32，AdamW lr 1e-5、weight decay 1e-4，200K步 |
| 保存 | 每5K步滚动保存两份；50K、100K、200K永久保留 |
| 推理 | 原25Hz动作/100Hz插值与接触Z导纳；外部动作融合系数0.01 |

本分支包含复制工作台所需的兼容模块；默认选择ACT，不自动运行Diffusion。
原先Ubuntu迁移说明保存在[历史README](README.ubuntu-handoff-history.md)。
本机实际训练历史见[训练记录](docs/rudra-dinov3-act-training.md)，其主机路径不是另一台电脑的默认路径。

核心源码：

- [DINO配置和固定权重身份](sim/act/configuration_dinov3.py)
- [冻结DINO与ACT适配器](sim/act/dinov3.py)
- [模型工厂与处理器](sim/act/policy.py)
- [可移植ACT配置](configs/rudra-dinov3-act-cuda.yaml)、[原生场景配置](configs/rudra-act-scene-profile.yaml)
- [本机配置生成器](scripts/configure_rudra_dinov3_act.py)

源码发布、离线单元测试、训练loss以及实际任务成功是不同的证据。
本次发布不代表新电脑已训练或完成策略评测。
