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

详见 [ARCHITECTURE.md](ARCHITECTURE.md)（架构、任务级动作边界、接口契约、安全设计、ESP32 扩展路径）、
[TODO.md](TODO.md)（当前 S2 近距离证据）和 [docs/demo_runbook.md](docs/demo_runbook.md)（可复现实验步骤）；
[HISTORY.md](HISTORY.md) 保留历史根因与审计记录。

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
- VLA 闭环（仿真，默认 near image/color 模型）：`ros2 launch asv_bringup vla_closed_loop.launch.py`
  （可用 `model_path:=...`、`perception_model_path:=...`、`language_backend:=qwen`、
  `language_device:=cuda`、`execution_address:=192.168.137.1`、`execution_port:=8081`
  显式指定部署资源；Qwen 在线启动必须使用 CUDA 分阶段参数）
- 专家对照闭环：`ros2 launch asv_bringup expert_closed_loop.launch.py`
- 完整硬件链：`ros2 launch asv_bringup full_system.launch.py`
  （micro_ros_agent + control manager；ESP32 通过 `/control/control_input` 接入）

可录制的 UE5 图形闭环顺序是“先启动 Jetson launch，再启动 UE5 游戏窗口”；
项目文件必须是 UnrealEditor 的第一个参数：

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=DEMO-S2-230906 -Layout=L7 -Motion=S2 -Seed=230906 `
  -MaxRuntimeSeconds=35 -SceneExecPort=8081 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed
```

默认策略与感知模型分别为 `policy_sine_near_image_color_seed42.onnx` 和
`image_entity_color_calibrated_v1.npz`；在线语言使用 Qwen3-Embedding-0.6B 的 staged
CUDA 路径（先编码并释放 Qwen，再初始化 MobileNet）。两次已验证的 S2 seed 是
230906 与 230902；约 7 m 的 OOD 目标必须保持 `valid=false`/hold。详见
[`docs/demo_runbook.md`](docs/demo_runbook.md) 与 [`models/manifest.yaml`](models/manifest.yaml)。

## 数据采集与训练（PC）

1. 采集：`tools/ue5/collect.ps1`（自动化：Jetson 查 slot → UE5 headless 运行 →
   打包回传 + SHA-256 校验）
2. 特征：`training/build_feature_caches.py`（冻结 MobileNet + Qwen）
3. 训练：`training/train.py`（配置在 `training/config/`）
4. 导出：`training/export_onnx.py`（parity 校验）
5. 部署：校验并复制 `policy_sine_near_image_color_seed42.onnx`、
   `image_entity_color_calibrated_v1.npz` 与 Qwen 模型目录到 Jetson `models/`；
   SHA/验证门状态以 [`models/manifest.yaml`](models/manifest.yaml) 为准。

## 平台

- Jetson Orin Nano 8 GB / Ubuntu 22.04 / ROS 2 Humble
- UE5 项目：`D:\Unreal Projects\VLA`（无限海洋地图，TCP :8080 对接）
- ESP32-P4 固件：独立仓库 `asv-esp32-firmware`（micro-ROS over UART2）
- PC 训练：RTX 5060 / PyTorch / Qwen3-Embedding-0.6B + MobileNetV3-small

## 许可

Apache-2.0（见 [LICENSE](LICENSE)）。
