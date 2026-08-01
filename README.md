# ASV VLA — 无人船视觉-语言-动作控制

ROS 2 工作空间：UE5 仿真提供多模态观测，Jetson 上的 VLA 流水线输出二维期望位移/
轨迹，经确定性安全门后执行（仿真运动学执行或 ESP32 推力链）。**故障时 fail-closed
（宁可停止，不可执行不安全轨迹）**。

动作接口是任务级二维 body-frame 轨迹，而不是左右推力：策略输出 20 个累计位移
waypoint（`dt=0.2 s`），安全门通过后控制器取短前缀，仿真 adapter 每个观测帧只发送
一个 setpoint。底层推力控制器可以独立调参，不改变 VLA 的任务接口。

## 系统概览

```
UE5 (相机/实体/本船状态)
  → 视觉编码 (MobileNetV3 冻结特征) + 实体张量 + 指令 embedding
  → 策略 (ONNX, CPU) / 专家对照
  → 安全门 (唯一发布者) → 轨迹控制器 → /decision/output
  → 仿真: decision_setpoint_adapter → UE5
  → 硬件: control_input_mux → [ESP32] → safety_supervisor → thruster_allocator → UE5
```

详见 [ARCHITECTURE.md](ARCHITECTURE.md)（架构、任务级动作边界、接口契约、安全设计、ESP32 扩展路径）
与 [HISTORY.md](HISTORY.md)（根因分析、诚实验收记录）。

## 目录结构

```
src/                  ROS 2 包（9 个，见 ARCHITECTURE.md §3）
training/             训练管线（特征缓存构建、训练、验证门、ONNX 导出）
dataset/language/     指令集 (instructions.jsonl) 与对比对
models/               演示指令 embedding + manifest
tools/ue5/            UE5 采集/验证自动化（C++ 子系统 + PowerShell 脚本）
tools/pc_reference/   PC 侧参考运行脚本
scripts/              Jetson 采集脚本
docs/                 接口与契约文档
```

## 快速开始（Jetson）

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

- 单元测试：`PYTHONPATH=src/asv_vla python -m pytest -q src/asv_vla/test`
- VLA 闭环（仿真）：`ros2 launch asv_bringup vla_closed_loop.launch.py`
  （可用 `model_path:=...`、`embedding_path:=...`、`active_embedding:=...`、
  `execution_address:=192.168.137.1`、`execution_port:=8081`、`visual_device:=cuda`
  显式指定部署资源）
- 专家对照闭环：`ros2 launch asv_bringup expert_closed_loop.launch.py`
- 完整硬件链：`ros2 launch asv_bringup full_system.launch.py`
  （micro_ros_agent + control manager；ESP32 通过 `/control/control_input` 接入）

可录制的 UE5 图形闭环顺序是“先启动上述 Jetson launch，再启动 UE5 游戏窗口”；
项目文件必须是 UnrealEditor 的第一个参数：

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map.Main_Map -game -log `
  -SceneAuto -Slot=DEMO-L6-S0-RED -Layout=L6 -Motion=S0 -Seed=23 `
  -SceneExecPort=8081 -MaxRuntimeSeconds=120 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed
```

红/蓝指令可在运行中通过 `ros2 param set /language_stub active_embedding <npy>` 热切换。
详见 [`docs/demo_runbook.md`](docs/demo_runbook.md)。

## 数据采集与训练（PC）

1. 采集：`tools/ue5/collect.ps1`（自动化：Jetson 查 slot → UE5 headless 运行 →
   打包回传 + SHA-256 校验）
2. 特征：`training/build_feature_caches.py`（冻结 MobileNet + Qwen）
3. 训练：`training/train.py`（配置在 `training/config/`）
4. 导出：`training/export_onnx.py`（parity 校验）
5. 部署候选（provisional demo only）：`policy_image_seed17.onnx` +
   `demo_instruction_embedding.npy` → Jetson `models/`

## 平台

- Jetson Orin Nano 8 GB / Ubuntu 22.04 / ROS 2 Humble
- UE5 项目：`D:\Unreal Projects\VLA`（无限海洋地图，TCP :8080 对接）
- ESP32-P4 固件：独立仓库 `asv-esp32-firmware`（micro-ROS over UART2）
- PC 训练：RTX 5060 / PyTorch / Qwen3-Embedding-0.6B + MobileNetV3-small

## 许可

Apache-2.0（见 [LICENSE](LICENSE)）。
