# Rudra 原生场景预览

工作树：`/home/luke/.codex/worktrees/f08d/UR１０SWEEP`。
工作台：http://127.0.0.1:8780/ 。当前页面已使用后续的
[扫零件场景适配](rudra-sweep-preview.md)；本文记录可独立加载的原生插孔配置。

来源是用户指定的`rudra-8000/ur10-mujoco-sim`，读取时`main`提交为
`436058dde9098c077f14d14d11e633fee9a02dee`。原仓库的118个文件
（ur10sim、README、NOTICE、requirements，合计25,226,242字节）原样保存于
`third_party/rudra_ur10_mujoco_sim/`，逐文件SHA256见`SOURCE.json`。
第三方版权说明及各模型许可证随原文件保留，并随
`codex/rudra-native-sweep-environment`分支一并提供。

## 场景与机位

直接调用上游`build_scene.Scene`，保留其MJCF装配、2500步静置、
UR10 CB3/PincOpen运动学机械臂与夹爪、桌板纹理、插销和孔块、光照、
相机标定及外观处理。

- 桌板：1.25×1.15m，中心(0.005,-0.685)m；桌面Z=-0.002m，
  坐标原点为机器人基座安装平面。
- `top`：D415俯视，相机位置(0.0322,-0.8351,1.068)m，水平FOV65°；
  look_at/up_hint以及额外yaw/roll按上游配置保留。
- `wrist`：D435i RGB，V6_REINFORCED安装架，安装角-14°，水平FOV69°。
  启用CAD支架时，原配置中的自由相机pos/look_at是备用值，不用于当前机位；
  原生模型中的支架和RGB镜头局部坐标决定实际位姿。
- `overview`：调试总览位置(1.6,-1.9,1.4)m，观察(0,-0.5,0.3)m。

完整本地配置是`configs/rudra-native-scene.yaml`，初始字节与上游
`ur10sim/scene_config.yaml`一致。页面配置详情显示完整相机设置、
编译后的世界位置/旋转矩阵以及水平/垂直FOV。三个视角均使用原生640×480。

## 工作台接入

`sim.web.run --rudra-scene`选择独立的原生预览模式，复用原工作台的CSS、
卡片、按钮和PNG编码。默认工作台入口保持原行为；原8773训练工作台独立运行。
新页面可刷新画面、重置并重新加载YAML、查看机位。
GET请求不推进仿真；静置后的图像会缓存，减少显存和GPU负载。
模型构建、渲染和释放保持在同一事件循环线程，避免EGL跨线程错误。

这次交付是原生环境检查。上游任务为夹爪抓取/插孔；现有扫零件任务的
刷子、托盘、六零件、五视角观测、控制器和模型动作接口尚未迁移到此环境。
预览应用不提供训练/推理作业接口，不会自动使用旧模型执行新任务。

```sh
cd '/home/luke/.codex/worktrees/f08d/UR１０SWEEP'
MUJOCO_GL=egl OMP_NUM_THREADS=1 /home/luke/.venvs/ur10-diffusion-cuda/bin/python \
  -m sim.web.run --rudra-scene --scene-config configs/rudra-native-scene.yaml \
  --host 127.0.0.1 --port 8780
```

最初使用用户systemd服务`ur10-rudra-scene-preview-f08d-20261022.service`。
扫零件适配后，8780改由`ur10-rudra-sweep-preview-f08d-20261022.service`运行。
修改YAML后点击“重置并重新加载配置”即可重建；改Python代码后重启服务。
服务为当前会话的transient服务，系统重启后需重新启动。

## 验证

原生模型成功编译，qpos有限，三个640×480 RGB视角均非空；浏览器图片
解码成功并实测重置。专门回归覆盖原生尺寸/机位、PNG、GET不推进时间、
重置和作业接口不开放、浏览器data URL赋值。相关原工作台/场景测试共83项通过；
机位详情补充后又单独复核原生测试。一个既有Starlette/httpx弃用警告。

证据：`runs/rudra_scene_preview_20261022/`中的`scene-proof.json`、
`regression.log`、`native-final.log`、各视角PNG及界面截图。
验证范围为环境加载、渲染和工作台显示；任务成功率尚未评测。
