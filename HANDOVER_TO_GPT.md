# 最终系统当前状态（2026-08-02）

这个文件只保留当前可执行系统的事实，避免把旧的 Day 计划、ONNX 后端或
专家在线路径误当成最终方案。项目的目标是 UE5 S2 近距离正弦跟踪：相机图像和
任务指令进入 Jetson，Jetson 在线推断目标并输出二维期望位移，UE5 执行该位移。

## 最终边界

```text
UE5 SceneCapture JPEG + /task/text + /ue/asv_state
    -> image_entity_perception (只读 JPEG，CUDA 图像模型；不读 /ue/entities)
    -> temporal_entity_tracker (跨帧计算相对速度)
    -> MobileNet CUDA + TaskFeatures + Qwen3-Embedding CUDA
    -> policy_sine_near_image_color_seed42.pt (Torch CUDA)
    -> safety_gate -> trajectory_controller -> decision_setpoint_adapter
    -> /ue/kinematic_setpoint -> UE5 C++ executor:8081
```

`/ue/entities` 仅用于录制和离线监督；它不能成为在线感知、TaskFeatures、视觉编码器
或策略的输入。单帧图像不输出速度，速度只由 tracker 根据连续图像推断。VLA 上层只
输出二维 `desired_x/desired_y`，不输出左右推力。

## 当前活动文件

- ROS 包：`asv_jetson_interfaces`、`asv_ue_bridge`、`asv_vla`、`asv_bringup`。
- 在线入口：`src/asv_bringup/launch/vla_closed_loop.launch.py`。
- 数据录制：`src/asv_bringup/launch/collect.launch.py`、`record_episode.launch.py`。
- 离线回放：`src/asv_bringup/launch/replay_episode.launch.py`。
- PC 训练入口：`training/`；最终近距离计划为
  `training/config/sine_near_collection_plan_v1.json`。
- PC 数据和二进制模型位于仓库外的 `pc_datasets/`，不提交 Git。

## Jetson 启动

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  language_model_path:=/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B \
  language_device:=cuda language_release_after_encode:=false \
  policy_device:=cuda visual_device:=cuda \
  execution_address:=192.168.137.1 execution_port:=8081
```

`language_release_after_encode=false` 是任务可热切换的默认路径；它要求 Jetson 的
Qwen 权重常驻 CUDA。如果设备内存不足，唯一允许的降级是显式使用
`language_release_after_encode:=true`，即保留已经得到的 256-D embedding、释放 Qwen
权重；该模式仍然使用真实 Qwen CUDA 编码，不是 `.npy` stub，也不允许切换 CPU。
当前已验收的图像几何是近距离红色目标；其他颜色/方位任务的文本解析存在，但视觉
跟踪仍应保持 fail-closed，直到对应校准数据和独立 Run 通过验收。

## UE5 S2 演示

UE 编辑器/游戏参数必须把 `.uproject` 放在第一个参数，并把初始目标保持在约 5 m 内：

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=DEMO-S2-230909 -Layout=L7 -Motion=S2 -Seed=230909 `
  -MaxRuntimeSeconds=35 -SceneExecPort=8081 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed -stdout -FullStdOutLogOutput
```

可录制的视频应同时展示 UE5 视图和 Jetson 日志中的 `PERCEPTION_TRACE`、
`POLICY_READY`、有效 setpoint 以及最终 `SCENE_UE_COMPLETE`。若目标超过校准范围，
正确结果是 perception/safety hold，而不是读取 UE 真值继续运动。

## 验收不变量

1. `image_entity_perception` 源码没有 `/ue/entities` 订阅。
2. `task_entity_tensor` 只接受 `image_perception` 或 `temporal_tracker` 来源。
3. policy 只使用 Torch CUDA checkpoint；加载失败输出 `valid=false` hold。
4. ego 只来自 `/ue/asv_state` 的实时 `surge_velocity/yaw_rate`，并参与输入身份匹配。
5. `run_id/scene_seed/frame_index/stamp_us` 在每一帧贯穿感知、tracker、策略、setpoint。
6. `/ue/kinematic_setpoint` 是唯一在线执行边界；`/ue/entities` 不得越过采集边界。

完整验收证据和待完成项目以 [TODO.md](TODO.md) 为准。
