# 最终系统状态与 TODO（2026-08-02）

## 目标

项目的可演示目标是：在 UE5 的 S2 正弦场景中，被控 ASV 从相机图像和任务文本出发，
在线识别红色目标船，连续保持约 3 m 间距；所有有效运动都由 Jetson 的在线链路产生，
而不是读取 UE 真值或并行的专家控制器。

当前结论：**近距离 S2 真实在线演示链已经通过，两次独立运行均实际驱动 UE5 中的
ASV 运动。** 这不是“所有距离、布局和随机种子都已泛化”的结论；正式演示应使用目标
初始距离不超过约 5 m 的 L7/L7B 近距离布局。

## 在线数据流（最终边界）

```text
UE5 SceneCapture JPEG + task text + ego
        ↓
image_entity_perception（只看 JPEG；不订阅 /ue/entities）
        ↓
temporal_entity_tracker（跨帧计算速度）
        ↓
visual_encoder（MobileNet，CUDA） + task_entity_tensor + Qwen task embedding（CUDA）
        ↓
policy_sine_near_image_color_seed42.onnx（ONNX/CPU，输出 20×2 轨迹和 STOP）
        ↓
visual_standoff_guard（只使用图像/跟踪张量，限制首个 waypoint 到约 3 m）
        ↓
5 帧平滑 → safety_gate（唯一有效发布者，fail-closed）
        ↓
trajectory_controller → decision_setpoint_adapter
        ↓
UE5 C++ kinematic executor（8081）→ ASV 世界位置变化 → 下一帧 JPEG
```

`/ue/entities` 只进入录制器、离线监督和几何评估，不能进入在线策略。单帧图像不填写
速度；速度来自 tracker 的相邻帧，并受 Run_Id、Scene_Seed、Frame_Index 和时间戳门控。
专家轨迹可以用于训练标签和采集阶段，但最终在线演示没有专家 publisher，也没有 UE
真值注入。

`visual_standoff_guard` 是透明的图像几何执行适配器：它不读取实体真值，不替代相机感知，
也不是把专家轨迹接回在线链路；它只将策略的第一个位移点约束到图像跟踪得到的目标距离，
避免小数据策略在动态水面上输出反向或过大的首步。策略仍负责有效位、STOP 和多模态
输入一致性，安全门仍可拒绝整个决策。

## 已完成且有证据

- 图像、任务、tracker、ego、策略、安全门、setpoint adapter、UE5 C++ executor 已接通。
- `/task/text` 使用可靠 + transient-local QoS；Jetson 先 launch、UE5 后 Play 时，Qwen
  和图像节点不会错过首条任务。
- Qwen3-Embedding-0.6B 在 Jetson 上真实 CUDA 编码一次后释放权重；MobileNet 随后在
  CUDA 上加载。策略 ONNX 保持 CPU 推理以避免 Orin 统一内存竞争，不是静默 CPU 降级。
- 近距离 image-only cache：12 runs / 1200 frames / 104720 samples；cache 不含
  `/ue/entities` 真值。当前图像校准模型为
  `pc_datasets/models/image_entity_color_calibrated_v1.npz`。
- 当前候选模型为
  `pc_datasets/models/policy_sine_near_image_color_seed42.onnx`，Jetson 已部署同名文件。
  该模型是三 seed 中唯一通过当前验证门的候选；seed17/23 结果保留但不加载。
- 本地完整测试：`231 passed, 5 skipped`；`py_compile` 和 `git diff --check` 通过。Jetson 的
  `asv_vla`/`asv_bringup` 已用当前源码重建。

## S2 在线验收记录

| 运行 | Jetson/UE 证据 | 结果 |
|---|---|---|
| `L7/S2/seed=230906`，Run_Id=`FEB0142041FF570A8149F9B6FD69B28C` | JPEG 感知第 2 帧得到红船 `(4.542, 2.436) m`；Qwen `LANGUAGE_READY_VALID`；策略 guard `STANDOFF_ADJUSTED`；UE `SCENE_EXEC_APPLY` 连续出现约 350 次 | 初始距离 5.094 m，最小 3.089 m，中位数 3.291 m，末端 3.461 m |
| `L7/S2/seed=230902`，Run_Id=`B6AFBD864B4CE41482A7ECA28BB9E39E` | JPEG 感知第 2 帧得到红船 `(4.406, 2.181) m`；重连后收到有效 setpoint；UE `SCENE_EXEC_APPLY` 至少到 count=275 | 初始距离 4.943 m，最小 3.029 m，中位数 3.304 m，末端 3.140 m |
| `L7/S2/seed=230907`，Run_Id=`75A78C924E8CA586E4B7808D071444D7` | 初始距离约 7.117 m；前 5 帧图像目标均不可见；没有有效 setpoint | 正确 fail-closed/hold。该 seed 超出当前近距离图像校准域，不算跟踪失败 |

前两次运行的运动来自在线 JPEG→感知→策略/guard→安全门→8081 setpoint 链；UE 日志中
可见 `SCENE_EXEC_CLIENT_CONNECTED`、`SCENE_EXEC_APPLY` 和 ASV 世界位置连续变化。
第三次是刻意保留的边界证据：看不清目标时系统不“猜”，而是停船。

清理后回归运行（同一 `seed=230906`，Run_Id=`1C5612294974C8EA9402969B5991D79A`）
再次通过：Jetson 日志显示图像第 2 帧 `(4.462, 2.400) m`、`POLICY_TRACE` 的
`STANDOFF_ADJUSTED` 和带 `Scene_Seed=230906` 的 `DECISION_VALID` setpoint；UE
日志从 t=0 的 ASV `(-10150,-10000)` 移动到 t=35 的 `(-7937.306,-10685.283)`，
`SCENE_EXEC_APPLY` 至少到 count=339，红船末端距离约 3.369 m。

## 当前唯一可录制的演示流程

### 1. Jetson 先启动

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
  execution_address:=192.168.137.1 execution_port:=8081
```

等待日志同时出现 `LANGUAGE_READY_VALID ... device=cuda` 和 `visual_encoder ... device=cuda`，
再启动 UE5。不要同时启动 `collect.launch.py`、`expert_closed_loop.launch.py` 或其它
会向同一控制 topic 发布的 launch。

### 2. UE5 启动（项目文件必须是第一个参数）

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=DEMO-S2-230906 -Layout=L7 -Motion=S2 -Seed=230906 `
  -MaxRuntimeSeconds=35 -SceneExecPort=8081 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed -stdout -FullStdOutLogOutput
```

要录制窗口可保留 `-windowed`；要做无界面验收可改成 `-RenderOffscreen -unattended`。
目标初始距离必须保持在约 5 m 以内，否则系统按设计保持位置。

## 后续 TODO（不阻塞当前演示）

- [x] 近距离 S2 图像数据采集、image-only cache、三 seed 训练和 ONNX 导出。
- [x] Jetson staged CUDA 启动、单一 publisher、QoS 启动时序和真实 8081 执行器验收。
- [x] 两次未见 Run_Id/Scene_Seed 的近距离 S2 在线运行。
- [ ] 如需论文级统计结论，再补充 L7/L7B × 红/蓝的至少 8 个新 seed，报告稳态距离、
  选中颜色、safe-gate PASS/hold 比例；在此之前不要宣称“8-run 泛化通过”。
- [ ] 如需超过约 5 m 的通用感知，替换当前 RGB 颜色校准器为真实检测器并重新采集、
  分组训练和在线验收；不能放宽当前可见性门来掩盖 OOD。
- [ ] 录制视频后保存 Jetson/UE 日志和本 TODO 版本，作为最终演示证据。

## 不允许回退

- 在线任何节点不得订阅 `/ue/entities`，不得把 UE BBox/速度或专家轨迹接入策略。
- 不得从单帧图像直接输出速度；不得用 `valid=true` 的安全停止伪装成有效动作。
- 不得并行启动多个 bridge、safety gate、trajectory controller 或专家 publisher。
- CUDA/Qwen 失败、目标不可见、身份错帧、策略非有限或安全门拒绝时必须 hold/fail-closed。

## 可恢复清理说明

历史 checkpoint、旧 cache、旧诊断日志和 Jetson 同步前源码保存在
`/tmp/asv_vla_cleanup_20260801` 或对应 archive 中，不能作为默认在线模型。当前默认
运行时只使用上面列出的 image calibration、near-image policy 和 Qwen 模型。
