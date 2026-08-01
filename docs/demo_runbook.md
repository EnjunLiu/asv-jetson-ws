# 演示 Runbook（S2 近距离在线闭环）

> 当前状态（2026-08-02）：默认 launch 已切换到 near image/color 模型，演示链路
> 是 Jetson 先启动、UE5 图形 `-game` 后启动。L7/S2 seed=230906 与 seed=230902
> 各完成一轮真实在线验证；这两轮是当前日志可复核的证据，不等同于 8-run 统计
> 鲁棒性结论。

## 演示内容

红蓝两船在 L7/L7B 近距离布局中并行走 S2 正弦（默认波长 60 m、幅度 6 m、
0.6 m/s）；红蓝目标起点约 4.5 m，白色干扰船约 7 m。被控 ASV 按指令**选择性
地跟随红色或蓝色船**。

- **控制源**：`policy_sine_near_image_color_seed42.onnx` +
  `image_entity_color_calibrated_v1.npz`；输入只来自图像感知、时序实体、语言和
  ego，不把 UE `/ue/entities` 真值送入在线策略。
- **执行说明**：策略输出经过 `visual_standoff_guard` 的图像/跟踪几何约束，只修正
  首个 waypoint 的距离和步长；guard 不读取 UE 真值，也不接收专家轨迹。
- **诚实声明**：本次演示画面由**学习策略**驱动；安全门拒绝的轨迹会变成
  hold，不绕过安全门，也不把专家对照结果写成模型结果。

## 前置条件（已就绪）

| 项 | 状态 |
|---|---|
| UE5 EDGEEditor | 已重建（S2 正弦 / L7+L7B 近距离布局 / YawFixWholeRun） |
| Jetson policy | `policy_sine_near_image_color_seed42.onnx`（CPU ONNX） |
| Jetson perception | `image_entity_color_calibrated_v1.npz` |
| 语言 | Qwen3-Embedding-0.6B，CUDA 分阶段启动；可切换 stub 做 smoke |

## 步骤

### 1. Jetson 先启动在线闭环（终端 A）

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.onnx \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  language_backend:=qwen \
  language_model_path:=/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B \
  language_device:=cuda language_release_after_encode:=true \
  language_staging_delay_sec:=30.0 \
  task_text:="跟随红色目标船，保持3米距离" \
  embedding_path:=/home/jetson/jetson_asv_ws/models/demo_instruction_embedding.npy \
  execution_address:=192.168.137.1 execution_port:=8081 visual_device:=cuda
```

Qwen 使用分阶段 CUDA：先等待日志出现 `LANGUAGE_TASK_RECEIVED` 和
`LANGUAGE_READY_VALID`，确认首次文本已编码并释放模型；`language_staging_delay_sec`
到期后才初始化 MobileNet/视觉链。若 Qwen 未 READY，保持 Jetson 进程和日志不变，
不要并发启动视觉模型，也不要静默切 CPU。

### 2. UE5 启动（终端 B = Windows，Jetson 已就绪）

```powershell
# 已验证运行 1：L7/S2，red，seed=230906：
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map.Main_Map -game -log `
  -SceneAuto -Slot=DEMO-L7-S2-RED-230906 -Layout=L7 -Motion=S2 -Seed=230906 `
  -SceneExecPort=8081 -MaxRuntimeSeconds=35 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed

# 已验证运行 2：L7/S2，red，seed=230902：
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map.Main_Map -game -log `
  -SceneAuto -Slot=DEMO-L7-S2-RED-230902 -Layout=L7 -Motion=S2 -Seed=230902 `
  -SceneExecPort=8081 -MaxRuntimeSeconds=35 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed
```

（演示录制用**窗口模式**便于 OBS 捕获。`-game` 加上完整 `.uproject` 是关键，
不要把 `UnrealEditor-Cmd.exe` 当作没有项目参数的游戏程序启动。）

两次运行均应同时保留 UE5 日志和 Jetson 日志，并核对
`SCENE_EXEC_BAD_PAYLOAD=0`、`SCENE_EXEC_APPLY`、策略 `valid=true`、安全门
`PASS` 以及 `/ue/kinematic_setpoint` 的连续身份字段。

### 3. 指令切换（终端 C，演示中途）

```bash
# Qwen 在线文本：重新发布任务文本，等待 LANGUAGE_READY_VALID 后再观察策略：
ros2 topic pub --once /task/text std_msgs/msg/String \
  "{data: '跟随蓝色目标船，保持3米距离'}"
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

[收尾] 本次展示是 near image/color seed42 的真实在线 S2 运行；两次 seed 证据已登记，
       但不把两次运行扩大成统计选择率。安全门全程 fail-closed（任何不安全轨迹被拒，
       从不绕过）。
       本演示画面由训练后的模型驱动。
```

## 当前在线证据（真实 UE5 ↔ Jetson S2 近距离运行）

| 运行 | 结果 | 证据 |
|---|---|---|
| L7/S2，seed=230906 | 已验证 | Run_Id=`FEB0142041FF570A8149F9B6FD69B28C`；图像第 2 帧约 `(4.542, 2.436) m`；`SCENE_EXEC_APPLY` 约 350 次；初始/中位/末端距离 `5.094/3.291/3.461 m` |
| L7/S2，seed=230902 | 已验证 | Run_Id=`B6AFBD864B4CE41482A7ECA28BB9E39E`；图像第 2 帧约 `(4.406, 2.181) m`；`SCENE_EXEC_APPLY` 至少到 275；初始/中位/末端距离 `4.943/3.304/3.140 m` |
| L7/S2，seed=230906（clean install 回归） | 已验证 | Run_Id=`1C5612294974C8EA9402969B5991D79A`；图像第 2 帧约 `(4.462, 2.400) m`；`SCENE_EXEC_APPLY` 至少到 339；末端距离约 `3.369 m` |

日志位于 UE 项目 `Saved/Logs/asv_demo_l7*_exec.log`；Jetson 端对应日志应至少包含
`LANGUAGE_TASK_RECEIVED`、`LANGUAGE_READY_VALID`、`POLICY_TRACE` 和安全门结果。
上述两轮证明当前 S2 近距离在线闭环和执行器链路，不等于所有布局/动态条件下的
统计鲁棒性。

## 7 m OOD 边界（必须保持 fail-closed）

L7/L7B 的白色干扰船约 7 m，超出当前近距离图像校准的主要工作范围；它们是刻意的
OOD 边界，不得写成“白船检测通过”或用 UE 真值补齐。若颜色目标/几何证据缺失、
非有限或超出校准范围，图像节点和任务张量必须保持 `valid=false`，策略 guard 以
`VISUAL_TARGET_MISSING` 等原因拒绝，安全门输出 hold/`valid=false`。任何 7 m
干扰船导致的有效运动指令都算失败，应保留日志而不是放宽安全门。

## 可选鲁棒性验收（≥8 runs）

| 指标 | 目标 | 记录 |
|---|---|---|
| 选择正确率（终点靠近指令色） | ≥9/10 runs | 尚未声称统计通过 |
| 稳态间距误差（3 m standoff） | ±1 m | |
| 安全门 PASS 占比 | 为主 | |
| 误 E-STOP / STALE 计数 | 0 | |

每 run 记录：seed、布局（L7/L7B）、指令（红/蓝）、日志路径。
**未达标不得在演示视频中声称"模型闭环通过"**——如实标注当前状态。

### 8-run 记录表（必须由 PIE 实测填写）

| # | Scene_Seed | 布局 | 指令 | 末端选中目标 | 稳态距离 m | Gate PASS % | STALE/E-STOP | 日志 |
|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | 230906 | L7 | red | | | | | |
| 2 | 230902 | L7 | red | | | | | |
| 3 | | L7 | blue | | | | | |
| 4 | | L7B | blue | | | | | |
| 5 | | L7 | red | | | | | |
| 6 | | L7B | red | | | | | |
| 7 | | L7 | blue | | | | | |
| 8 | | L7B | blue | | | | | |

## 后续边界

- 若要发布“8-run 统计鲁棒性”结论，继续填写表格并保留每轮日志；当前核心验收
  已由图形 `-game` + C++ 执行器完成，PIE 不是启动链路的硬前置条件。
- near image/color 的离线指标仍以独立报告为准，不从两次在线运行反推统计率。
- 30 分钟压力测试 + 资源日志（验收证据，P2）
- ESP32 实机接入（替换 use_fake_esp32=false，见 docs/esp32_interface.md）
