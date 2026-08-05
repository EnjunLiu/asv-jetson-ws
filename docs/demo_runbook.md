# S2 单点闭环演示 Runbook

本 Runbook 覆盖真实 JPEG 感知、真实 Qwen CUDA embedding、结构化实体 tracker、单点
policy 和 UE5 运动学执行。Jetson 只做 ROS 2 构建与 CUDA 在线推理，不训练、不启动
专家、不读取 `/ue/entities` 真值，也不启动低层推力控制器。

## 1. 准备 artifact

将 PC 产物复制到 Jetson `/home/jetson/jetson_asv_ws/models/`，并使用以下隔离文件名：

```text
policy_single_point_v3_full_seed17.pt
perception_image_conditioned_130_v1.npz
Qwen3-Embedding-0.6B/
```

来源和 SHA-256：

```text
policy source: /mnt/c/Temp/asv_vla_retrain_20260805/policy_v3_single_point_20260805/full_seed17/best.pt
policy sha256: f907d297dbcbedd10aa5bc009d4345655654db04d1e66282f68fad06abbead2c

perception source: /mnt/c/Temp/asv_vla_retrain_20260805/perception_image_conditioned_130_v1.npz
perception sha256: a1e7451642c51b879e8b9ce1d7037567c2057d534bcb547c483716188ceb5e6e
```

感知 artifact 的 metadata 必须保持：`model_version=image_entity_ridge_language_v3`、
`input=(camera_image_rgb,task_embedding_float32[256])->structured_entities`、feature
input dimension `4320`，以及输出字段 `relative_velocity_mps`、`velocity_valid`。
相对速度由 temporal tracker 计算，不能把 `velocity_output` 写成感知模型直接输出。

## 2. Jetson 构建和启动

```bash
cd /home/jetson/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select asv_jetson_interfaces asv_ue_bridge asv_vla asv_bringup
source install/setup.bash

sha256sum models/policy_single_point_v3_full_seed17.pt
sha256sum models/perception_image_conditioned_130_v1.npz

ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_single_point_v3_full_seed17.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/perception_image_conditioned_130_v1.npz \
  language_model_path:=/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B \
  language_model_id:=Qwen/Qwen3-Embedding-0.6B \
  language_device:=cuda visual_device:=cuda policy_device:=cuda \
  language_release_after_encode:=true \
  task_text:="跟随红色目标船，保持3米距离" \
  execution_address:=192.168.137.1 execution_port:=8081
```

当前 canonical language model ID 是 `Qwen/Qwen3-Embedding-0.6B`。启动时首次任务必须
由真实 Qwen CUDA 编码，并看到：

```text
LANGUAGE_READY_VALID ... release_model=true
```

`language_release_after_encode=true` 只释放 Qwen 权重，保留这次真实 CUDA 编码得到的
有效 embedding 供当前闭环使用，以满足 8 GB Orin 内存约束。它不是预计算 `.npy`、缓存
embedding 或 CPU fallback；CUDA 不可用时必须 invalid/hold。

继续等待：

```text
POLICY_READY backend=torch_cuda inputs=task_embedding+structured_entities+previous_action output=[desired_x,desired_y]
PERCEPTION_TRACE visible=True
POLICY_TRACE entity_valid=True
PERCEPTION_PERF_TRACE valid=True
```

policy 的实际决策输入是 `language`、`structured_entities/entity_geometry`、
`previous_action` 及 `language_valid`、`entity_geometry_mask`、`previous_action_valid`、
`policy_input_valid`。`global_visual`、`entity_visual`、`ego` 不在 policy 输入中。输出是
一个 `[desired_x, desired_y]` 单步 body-frame 期望位移，单位为米，不是轨迹序列或推力。

## 3. UE5 启动

项目文件必须是 UnrealEditor 的第一个参数：

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=FINAL-S2-230908-V5 -Layout=L7 -Motion=S2 -Seed=230908 `
  -MaxRuntimeSeconds=120 -SceneExecPort=8081 -YawFixWholeRun `
  -SineAmplitude=200 -SineDelay=40 -ResX=1280 -ResY=720 -windowed -stdout -FullStdOutLogOutput
```

运行时同时保存 UE5 窗口、Jetson 日志和 `Saved/Logs/VLA.log`。不要启动第二套 bridge、
policy、safety gate 或 recorder。

## 4. 成功判据

当前第三轮真实证据固定为：

```text
run: L7/S2/seed=230908
slot: FINAL-S2-230908-V5
MaxRuntimeSeconds: 120
Jetson log: /mnt/c/Temp/asv_vla_closed_loop_20260805/jetson_l7_s2_230908_20260805_third.log
UE log: /mnt/c/Temp/asv_vla_ue_l7_s2_230908_20260805_third.log
SCENE_UE_COMPLETE runtime_seconds=120.01
SCENE_EXEC_APPLY final count=450
```

成功证据必须能在对应日志中核对 `LANGUAGE_READY_VALID release_model=true`、
`POLICY_READY backend=torch_cuda ...`、`PERCEPTION_TRACE visible=True`、
`POLICY_TRACE entity_valid=True`、`PERCEPTION_PERF_TRACE valid=True`、连续
`SCENE_EXEC_APPLY` 和最终 `SCENE_UE_COMPLETE`。只看到 launch 进程、PC 单元测试、下载
模型或静态接口不等于 Jetson-UE5 闭环成功。

## 5. 数据与 previous action 规则

离线采集和在线演示不能并行：

```bash
ros2 launch asv_bringup collect.launch.py \
  slot_id:=L7 layout_id:=L7 motion_state:=S2 scene_seed:=230908 \
  execution_address:=192.168.137.1 execution_port:=8081
```

recorder 保存 JPEG、`UEASVState`、UE Entities、任务、身份元数据和每帧单步专家点。
PC 训练每个时刻只回归当前专家 `[desired_x, desired_y]`；`previous_action` 来自同一
run、同一 instruction 的相邻前帧，首帧为 `[0.0, 0.0]` 且
`previous_action_valid=false`。跨 run、instruction 切换、帧不连续、gate 拒绝或动作无效
时清零并置无效。UE Entities/ego 只作离线监督或验证，不能进入 Jetson policy。

## 6. 失败轮次和结果保全

失败轮次必须保留原始 Jetson/UE 日志并标为失败，不能混入成功证据。当前成功引用只使用
上面列出的第三轮日志；旧图、原始 learning curves 和历史失败记录不能覆盖或删除。新增
统计、解释和截图应追加到新的分析位置。

目标不可见、身份不匹配、CUDA/模型加载失败、输入陈旧或 safety gate 拒绝时必须
fail-closed，UE5 收到零位移 hold；不得用 UE 真值、旧专家 publisher、缓存 embedding 或
CPU/ONNX 后端把失败轮次改写成成功。

## 7. 第三轮轨迹分析

为避免覆盖旧 benchmark 图，本轮生成了独立分析文件：

```text
/mnt/c/Users/LIU/Desktop/track_world_closed_loop_single_point_l7_s2_230908_20260805.png
/mnt/c/Temp/asv_vla_closed_loop_20260805/l7_s2_230908_20260805_third_metrics.json
```

解析结果：UE5 实际 `SCENE_EXEC_APPLY` 为 `450` 次，ASV 世界坐标采样 `121` 点，世界
路径长度约 `48.59 m`；红色目标与 ASV 的采样 standoff 从 `4.53 m` 变为 `4.11 m`，
本轮不是精确 3 m standoff 验收。Jetson bridge 的 payload 日志限流为 `105` 个动作样本，
其中 `104` 个有效、`72` 个非零；记录到的单步位移上限为 `15.00 cm`，连续非零日志样本
的稳定段相邻动作差分 p95 为 `0.34 cm`。这些平滑指标只描述该次真实 L7/S2/seed 运行，
不能外推为跨场景统计泛化。
