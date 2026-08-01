# 当前运行说明（2026-08-02）

本文件只记录当前可运行系统；历史失败实验请看 [HISTORY.md](HISTORY.md)，当前验收
和 TODO 以 [TODO.md](TODO.md) 为准。项目目标是 UE5 S2 正弦场景中的近距离真实在线
跟踪，不是专家轨迹回放。

## 当前结论

- L7/S2 `seed=230908` 已完成最终 35 秒真实在线运行（此前 `230906/230902` 为对照）：UE5 相机 JPEG
  进入 Jetson，图像感知识别红船，首次任务由 Qwen CUDA 编码后释放权重，Torch CUDA 策略和图像跟踪
  standoff guard 产生有效 setpoint，UE5 C++ 8081 executor 实际改变 ASV 世界位置。
- `/ue/entities` 只用于录制和离线监督，不能进入在线策略。单帧图像不提供速度；
  tracker 用相邻帧和 `Run_Id/Scene_Seed/Frame_Index/stamp_us` 计算速度。
- 当前图像校准器的主要工作范围是约 5 m 内。超过范围看不清时必须 hold/fail-closed，
  不得为了视频向 UE 真值或专家控制器回退。

## 双端路径

| 端 | 路径 |
|---|---|
| PC 工程 | `C:\Users\LIU\Documents\jetson_ws\asv_vla` |
| PC 数据/模型 | `C:\Users\LIU\Documents\jetson_ws\pc_datasets` |
| UE5 项目 | `D:\Unreal Projects\VLA\VLA.uproject` |
| Jetson | `jetson@192.168.137.100:~/jetson_asv_ws` |

Jetson 当前部署模型哈希：

```text
policy_sine_near_image_color_seed42.pt
6c4ed50d49a0ba9447a3d991cb09de130bbdbcc6eb98eca1b98c49dfd66d7685

image_entity_color_calibrated_v1.npz
985111c7cfeaea9a927bc59b6b3d6efb2bf40df7e68996d44aa82de4b2014a3c
```

## 在线数据流

```text
UE5 SceneCapture JPEG + /task/text + ego
  → image_entity_perception（image-only）
  → temporal_entity_tracker（跨帧速度）
  → MobileNet CUDA + TaskFeatures + Qwen CUDA
  → policy_sine_near_image_color_seed42.pt（Torch/CUDA）
  → visual_standoff_guard（图像/跟踪几何，约 3 m 首步）
  → smoothing → safety_gate（唯一发布者）
  → trajectory_controller → decision_setpoint_adapter
  → /ue/kinematic_setpoint → UE5 C++ executor:8081
```

guard 是透明的图像执行适配器：只修改策略第一 waypoint 的距离和步长，不读取
`/ue/entities`，不接收专家轨迹。Torch 策略仍决定 STOP/valid 和多模态路径，安全门可以
拒绝任何异常轨迹。

## 启动顺序

### Jetson

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  language_backend:=qwen \
  language_model_path:=/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B \
  language_device:=cuda language_release_after_encode:=true \
  language_staging_delay_sec:=20.0 \
  policy_backend:=torch_cuda policy_device:=cuda \
  execution_address:=192.168.137.1 execution_port:=8081
```

默认先等待 Qwen 的 `READY model=Qwen3-Embedding-0.6B;device=cuda` 以及首次任务的
`LANGUAGE_READY_VALID ... release_model=true`，并用
`/vla/language_embedding` 确认 embedding 仍 valid；`language_staging_delay_sec=20.0` 后再等待
`visual_encoder ... device=cuda`，然后启动 UE5。首次任务由真实 Qwen CUDA 编码，成功后
释放权重而保留 256-D embedding 在线；当前 S2 演示不承诺任务切换，切换任务需重启整套
闭环。只允许一个 `vla_closed_loop`；不要并行启动专家、采集或另一套 bridge。

### UE5（项目文件必须是第一个参数）

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=DEMO-S2-230906 -Layout=L7 -Motion=S2 -Seed=230906 `
  -MaxRuntimeSeconds=35 -SceneExecPort=8081 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed -stdout -FullStdOutLogOutput
```

视频录制保留 `-windowed`；无界面验收可使用 `-RenderOffscreen -unattended`。目标初始
距离应控制在约 5 m 内。

## 运行时证据

- `seed=230908`：Run_Id `E82B58E6415C9AE61F3797BB1C7B7D99`，图像第 2 帧约
  `(4.563, 2.398) m`，Jetson `PERCEPTION_TRACE=5`、有效 setpoint 约 32 条；UE5
  `SCENE_EXEC_APPLY` 约 350 次，ASV 世界 X 从约 `-10150 cm` 变化到 `-7899 cm`，并
  以 `SCENE_UE_COMPLETE runtime_seconds=35.00` 正常结束。
- `seed=230906`（对照）：Run_Id `FEB0142041FF570A8149F9B6FD69B28C`，图像第 2 帧约
  `(4.542, 2.436) m`，UE setpoint 约 350 次，距离中位数 3.291 m、末端 3.461 m。
- `seed=230902`：Run_Id `B6AFBD864B4CE41482A7ECA28BB9E39E`，图像第 2 帧约
  `(4.406, 2.181) m`，UE `SCENE_EXEC_APPLY` 至少到 count=275，距离中位数
  3.304 m、末端 3.140 m。
- `seed=230907`：初始约 7.117 m，前五帧目标不可见，没有有效 setpoint；这是正确
  的 OOD fail-closed 边界，不是允许放宽门限的理由。

## 验证命令

```bash
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
PYTHONPATH=src/asv_vla pytest -q                 # 238 passed, 7 skipped
python3 -m py_compile src/asv_bringup/launch/vla_closed_loop.launch.py
git diff --check
```

如需论文级结论，另行完成 L7/L7B、红/蓝任务的至少 8 个新 seed，并报告稳态距离、
颜色选择和 safety-gate hold 比例；两次成功运行不应扩大成统计泛化声明。
