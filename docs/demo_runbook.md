# 演示 Runbook（模型在线闭环 + 正弦编队）

> 当前状态（2026-08-01）：Jetson ↔ UE5 的图形 `-game` 闭环已经完成可复现的
> 核心验收。L6/S0 红色与蓝色各完成一轮真实执行器验证；动态 L6/S2 场景的
> 实体运动和执行器通过，但目标开始运动后安全门会按设计进入 hold，因此本文件
> 不把 S2 持续跟随写成已通过指标。录制视频请使用下面的 L6/S0 命令。

## 演示内容

红蓝两船并行走大正弦（波长 60 m、幅度 6 m、0.6 m/s，红蓝分居曲线两侧），
两白船前方直线干扰；被控 ASV 按指令**选择性地跟随红色或蓝色船**。

- **控制源**：新训练 ONNX 策略（sine_formation_v4）——seed 17/23/42 验证门 PASS、
  ONNX parity 已通过；旧的 96.2%（L6 97.9% / L6B 95.3%）是 v2 离线快照，不能
  自动当作 v4 的选择率
- **诚实声明**：本次演示画面由**学习策略**驱动；安全门拒绝的轨迹会变成
  hold，不绕过安全门，也不把专家对照结果写成模型结果。

## 前置条件（已就绪）

| 项 | 状态 |
|---|---|
| UE5 EDGEEditor | 已重建（S2 正弦 / L6+L6B 布局 / YawFixWholeRun） |
| Jetson models/policy.onnx | v4 当前部署模型；旧备份不再参与运行 |
| 语言 embedding | follow_red（row0 一致）+ follow_blue（row20 一致） |
| hardware_loop.launch.py | VLA + ESP32 链（use_fake_esp32 默认 true） |

## 步骤

### 1. Jetson 启动在线闭环（终端 A）

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy.onnx \
  embedding_path:=/home/jetson/jetson_asv_ws/models/demo_instruction_embedding.npy \
  execution_address:=192.168.137.1 execution_port:=8081 visual_device:=cuda
```

### 2. UE5 启动（终端 B = Windows）

```powershell
# 可直接录制的静态 L6 红色演示（目标约 3 m 处收敛并保持）：
D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map.Main_Map -game -log `
  -SceneAuto -Slot=DEMO-L6-S0-RED -Layout=L6 -Motion=S0 -Seed=23 `
  -SceneExecPort=8081 -MaxRuntimeSeconds=120 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed
```

（演示录制用**窗口模式**便于 OBS 捕获。`-game` 加上完整 `.uproject` 是关键，
不要把 `UnrealEditor-Cmd.exe` 当作没有项目参数的游戏程序启动。）

### 3. 指令切换（终端 C，演示中途）

```bash
# 跟红（默认已加载）→ 跟蓝：
ros2 param set /language_stub active_embedding \
  /home/jetson/jetson_asv_ws/models/follow_blue_embedding.npy
# 切回跟红：
ros2 param set /language_stub active_embedding \
  /home/jetson/jetson_asv_ws/models/demo_instruction_embedding.npy
```

### 4. 录制

- OBS：捕获 UE5 窗口 + 终端 A（实时日志：safety gate PASS / setpoint 输出）
- 可选叠字：`ros2 topic echo /decision/output` 或 ROS 2 bag

### 5. 台词脚本（诚实标注）

```
[开场] 系统概览：UE5 仿真 → 视觉/实体/语言编码 → 学习策略（ONNX, CPU）
       → 任务级二维轨迹 → 确定性安全门 → 单步运动学执行。
       底层推力控制器与这条任务链解耦；红蓝编队走正弦，指令决定跟随对象。

[选择段] 当前指令"跟随红色"：注意船向右（红）偏转，蓝船在左侧。
       切换指令到"跟随蓝色"：船改为向蓝色目标侧偏转。

[收尾] 指标：v4 验证门通过（ADE/FDE 改善约 67%+）、在线选择率以本次 ≥8 runs
       记录为准；安全门全程 fail-closed（任何不安全轨迹被拒，从不绕过）。
       本演示画面由训练后的模型驱动。
```

## 当前在线证据（真实 UE5 ↔ Jetson 运行）

| 运行 | 结果 | 证据 |
|---|---|---|
| L6/S0 红，seed=23 | 通过 | `SCENE_EXEC_BAD_PAYLOAD=0`；`SCENE_EXEC_APPLY` 出现；末端约 2.8 m |
| L6/S0 蓝，seed=42 | 通过（颜色方向） | `SCENE_EXEC_BAD_PAYLOAD=0`；ASV 向蓝色侧运动；末端约 3.7 m |
| L6/S2 红，seed=17 | 安全停止 | S2 参数和执行器通过；目标开始移动后安全门进入 hold，未宣称持续跟随 |

日志位于 UE 项目 `Saved/Logs/asv_demo_l6*_exec.log`；Jetson 端对应日志为
`/tmp/asv_final_vla_final.log`。上述记录证明的是当前核心在线闭环和执行器，不等于
所有布局/动态条件下的统计鲁棒性。

## 可选鲁棒性验收（≥8 runs）

| 指标 | 目标 | 记录 |
|---|---|---|
| 选择正确率（终点靠近指令色） | ≥9/10 runs | 尚未声称统计通过 |
| 稳态间距误差（5 m standoff） | ±1 m | |
| 安全门 PASS 占比 | 为主 | |
| 误 E-STOP / STALE 计数 | 0 | |

每 run 记录：seed、布局（L6/L6B）、指令（红/蓝）、日志路径。
**未达标不得在演示视频中声称"模型闭环通过"**——如实标注当前状态。

### 8-run 记录表（必须由 PIE 实测填写）

| # | Scene_Seed | 布局 | 指令 | 末端选中目标 | 稳态距离 m | Gate PASS % | STALE/E-STOP | 日志 |
|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | | L6 | red | | | | | |
| 2 | | L6B | red | | | | | |
| 3 | | L6 | blue | | | | | |
| 4 | | L6B | blue | | | | | |
| 5 | | L6 | red | | | | | |
| 6 | | L6B | red | | | | | |
| 7 | | L6 | blue | | | | | |
| 8 | | L6B | blue | | | | | |

## 后续边界

- 若要发布“8-run 统计鲁棒性”结论，继续填写表格并保留每轮日志；当前核心验收
  已由图形 `-game` + C++ 执行器完成，PIE 不是启动链路的硬前置条件。
- v4 离线选择率仍以 `training/evaluate_selection.py` 的独立报告为准，不从单次
  在线运行反推统计率。
- 30 分钟压力测试 + 资源日志（验收证据，P2）
- ESP32 实机接入（替换 use_fake_esp32=false，见 docs/esp32_interface.md）
