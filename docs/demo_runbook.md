# S2 在线演示 Runbook

本 Runbook 只覆盖真实 JPEG 感知、真实 Qwen CUDA、结构化实体 tracker 和学习策略在线
驱动 UE5 的近距离 S2 演示。它不启动专家、不读取 UE Entities 真值，也不启动低层推力
控制器。

## 1. Jetson

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  language_model_path:=/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B \
  language_device:=cuda visual_device:=cuda policy_device:=cuda \
  language_release_after_encode:=true \
  task_text:="跟随红色目标船，保持3米距离" \
  execution_address:=192.168.137.1 execution_port:=8081
```

等待以下日志：

```text
image_entity_perception ... device=cuda
language_qwen: READY ... device=cuda
LANGUAGE_READY_VALID ... release_model=true
image_entity_perception ... device=cuda
POLICY_READY backend=torch_cuda device=cuda
```

**实测（2026-08-02）**：Orin Nano 8 GB 统一内存下常驻 Qwen
（`release_after_encode:=false`）与其余 CUDA 模型并发加载/推理会 OOM
（`NvMapMemAlloc error 12`）。已内置两层对策并验证：
1. launch 对视觉/感知/策略节点错峰启动（`TimerAction` 20 s，Qwen 先加载编码）；
2. `language_release_after_encode:=true`（默认）：Qwen 首次 CUDA 编码后释放权重，
   仍使用真实 Qwen 生成的 embedding，绝不切换 `.npy` 或 CPU。
此组合已完整跑通在线闭环（SCENE_EXEC_APPLY 连续）。

## 2. UE5

目标初始距离保持在约 5 m 内。项目文件必须是 UnrealEditor 的第一个参数：

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=FINAL-S2-230908 -Layout=L7 -Motion=S2 -Seed=230908 `
  -MaxRuntimeSeconds=35 -SceneExecPort=8081 -YawFixWholeRun `
  -SineAmplitude=200 -SineDelay=40 -ResX=1280 -ResY=720 -windowed -stdout -FullStdOutLogOutput
```

录制时保留 UE5 窗口、Jetson 日志和 `Saved/Logs/VLA.log`。成功标准：

- `/vla/perceived_entities` 的 `source=image_perception`，任务切换反映在
  `instruction_id` 和 `is_target`；
- `/vla/tracked_entities` 第一帧 `velocity_valid=false`，之后才出现时序速度；
- `POLICY_TRACE` 中 `lang_valid=true`、`entity_valid=true`，输出为一个单步
  `desired_x/desired_y`；
- UE5 连续输出 `SCENE_EXEC_APPLY`，最终输出 `SCENE_UE_COMPLETE`；
- 目标不可见或身份不匹配时必须 hold，不能继续运动。

## 3. 任务切换

常驻 Qwen 支持在线更新指令，但当前图像校准器只对近距离红色目标完成验收。因此最终
视频只使用红色跟踪或 `stop`；蓝色/左右任务可以验证解析和 fail-closed 行为，不能把
语言 embedding 的更新当成视觉跟踪已验收：

```bash
ros2 topic pub --once /task/text std_msgs/msg/String \
  "{data: '停止'}"
```

切换后应观察 `instruction_id=stop`、零位移点和 hold。若使用显式释放模式，任务切换
需要重启闭环；若要验收蓝色/左右跟踪，先完成对应 RGB 几何校准并更新 manifest。

## 4. 离线采集

采集和在线演示不能并行：

```bash
ros2 launch asv_bringup collect.launch.py \
  slot_id:=L7 layout_id:=L7 motion_state:=S2 scene_seed:=230908 \
  execution_address:=192.168.137.1 execution_port:=8081
```

采集器保存 JPEG、UEASVState、UE Entities、每帧单步专家点和身份元数据。PC 训练时
Entities 只作为监督标签；部署策略的输入是图像推断的结构化 Entities、语言 embedding
和上一控制帧实际通过 safety gate 的 action，不接收 ego。

## 5. 当前证据与边界

L7/S2 `seed=230908` 的历史运行已证明 JPEG→图像实体→tracker→CUDA policy→UE5
executor 的在线链路，曾记录约 350 次 `SCENE_EXEC_APPLY` 和正常 `SCENE_UE_COMPLETE`。
这证明当前 near S2 可录制，不等于所有布局、距离和颜色已经统计泛化。约 7 m 白色干扰
船是 OOD fail-closed 边界，不能用 UE 真值补齐。

## 6. 红蓝切换验证记录（2026-08-02）

- **红色**：指令解析 ✓，感知检测 target_red 4.9 m（confidence 1.0），策略 applied，
  ASV 向左（red 侧）跟随 30 m（Y -10017 → -10180），间距 3 m 保持
- **蓝色**：指令解析 ✓，感知无蓝色输出（模型仅校准红），守卫
  VISUAL_TARGET_MISSING 阻止，ASV fail-closed hold（t=50 后静止）
  —— 蓝色视觉跟踪是记录在案的未验收项，未伪装成已验收
