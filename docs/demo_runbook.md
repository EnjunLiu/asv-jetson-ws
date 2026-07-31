# 演示 Runbook（模型在线闭环 + 正弦编队）

## 演示内容

红蓝两船并行走大正弦（波长 60 m、幅度 6 m、0.6 m/s，红蓝分居曲线两侧），
两白船前方直线干扰；被控 ASV 按指令**选择性地跟随红色或蓝色船**。

- **控制源**：新训练 ONNX 策略（sine_formation_v2）——验证门 PASS（3 seeds）、
  选择正确率 96.2%（L6 97.9% / L6B 95.3%）、ONNX parity 精确
- **诚实声明**：本次演示画面由**学习策略**驱动；若实测异常则回退
  expert 对照（规格允许路径）并如实标注

## 前置条件（已就绪）

| 项 | 状态 |
|---|---|
| UE5 EDGEEditor | 已重建（S2 正弦 / L6+L6B 布局 / YawFixWholeRun） |
| Jetson models/policy.onnx | 新策略已部署（旧模型备份 policy_day21v2_backup.onnx） |
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
  model_path:=/home/jetson/jetson_asv_ws/models/policy.onnx
```

### 2. UE5 Play（用户操作，终端 B = Windows）

```powershell
# 或窗口模式手动 Play：
D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game `
  -SceneAuto -Slot=DEMO-001 -Layout=L6 -Motion=S2 -Seed=200101 `
  -MaxRuntimeSeconds=180 -YawFixWholeRun
```

（演示录制用**窗口模式**便于 OBS 捕获；headless 已验证闭环不可行——
setpoint 不执行，见 docs/scene_verification.md §5）

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
       → 确定性安全门 → 运动控制。红蓝编队走正弦，指令决定跟随对象。

[选择段] 当前指令"跟随红色"：注意船向右（红）偏转，蓝船在左侧。
       切换指令到"跟随蓝色"：船改为向左（蓝）偏转。

[收尾] 指标：验证门通过（ADE 改善 67%+）、选择正确率 96.2%、
       安全门全程 fail-closed（任何不安全轨迹被拒，从不绕过）。
       本演示画面由训练后的模型驱动。
```

## 验收指标（Play 实测 ≥8 runs）

| 指标 | 目标 | 记录 |
|---|---|---|
| 选择正确率（终点靠近指令色） | ≥9/10 runs | |
| 稳态间距误差（5 m standoff） | ±1 m | |
| 安全门 PASS 占比 | 为主 | |
| 误 E-STOP / STALE 计数 | 0 | |

每 run 记录：seed、布局（L6/L6B）、指令（红/蓝）、日志路径。
**未达标不得在演示视频中声称"模型闭环通过"**——如实标注当前状态。

## 遗留事项

- PIE 模式实测（本 runbook 第 2 步）需用户 Play；headless 下 setpoint
  不执行（已证实，见 scene_verification.md §5）
- 30 分钟压力测试 + 资源日志（验收证据，P2）
- ESP32 实机接入（替换 use_fake_esp32=false，见 docs/esp32_interface.md）
