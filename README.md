# ASV VLA

面向 Jetson/ROS 2 的 UE5 ASV 单点闭环。当前在线边界是：

```text
camera_image_rgb + task_text
  -> 真实 Qwen CUDA task embedding
  -> 感知：image + task embedding -> structured_entities
  -> temporal tracker -> entity_geometry（含相对速度和有效性）
  -> policy：language + entity_geometry + previous_action + validity masks
  -> 一个 body-frame [desired_x, desired_y] 米制期望位移点
  -> safety_gate -> UE5 运动学执行
```

决策 policy 明确排除 `global_visual`、`entity_visual` 和 `ego`。在线节点不得订阅
`/ue/entities`；UE Entities 只用于 recorder、PC 离线监督、回放和验证。上层不输出左右
推进器指令，真实船的低层控制器属于独立工程。

## ROS 2 包

| 包 | 责任 |
| --- | --- |
| `asv_jetson_interfaces` | CameraFrame、UEASVState、UEEntity、TaskFeatures、TaskEmbedding、DecisionPoint 等消息 |
| `asv_ue_bridge` | UE5 TCP JSON、JPEG、本船状态、真值标签发布，以及运动学 setpoint 输出 |
| `asv_vla` | 图像实体感知、任务筛选、跨帧 tracker、Qwen/策略、安全门、单点控制、采集和回放节点 |
| `asv_bringup` | `vla_closed_loop.launch.py`、`collect.launch.py`、`record_episode.launch.py`、`replay_episode.launch.py` |

在线 launch 不启动 ESP32、控制管理器或推进器分配器。

## 当前部署 artifact

二进制不提交 Git。将 PC 产物复制到 Jetson 的隔离部署名后，按
[`models/manifest.yaml`](models/manifest.yaml) 核对来源和 SHA-256：

| 组件 | PC 来源 | Jetson 文件名 | SHA-256 |
| --- | --- | --- | --- |
| policy | `/mnt/c/Temp/asv_vla_retrain_20260805/policy_v3_single_point_20260805/full_seed17/best.pt` | `policy_single_point_v3_full_seed17.pt` | `f907d297dbcbedd10aa5bc009d4345655654db04d1e66282f68fad06abbead2c` |
| perception | `/mnt/c/Temp/asv_vla_retrain_20260805/perception_image_conditioned_130_v1.npz` | `perception_image_conditioned_130_v1.npz` | `a1e7451642c51b879e8b9ce1d7037567c2057d534bcb547c483716188ceb5e6e` |

感知 NPZ 的 metadata 是 `model_version=image_entity_ridge_language_v3`，输入为
`(camera_image_rgb,task_embedding_float32[256])->structured_entities`，feature input
dimension 为 `4320`，输出包含 `relative_velocity_mps` 和 `velocity_valid`。相对速度由
跨帧 tracker 产生，不是感知模型直接输出。

语言模型 canonical ID 是 `Qwen/Qwen3-Embedding-0.6B`，本地目录仍可命名为
`Qwen3-Embedding-0.6B`。当前 8 GB Orin 闭环必须使用
`language_release_after_encode:=true`：首次任务仍由真实 Qwen CUDA 在线编码，发布有效
embedding 后释放 Qwen 权重以避免 OOM；这不是缓存 embedding，也不是 CPU fallback。释放
后只能复用该次真实在线编码结果；需要新任务时应按实现支持范围处理，不能伪造新的缓存输入。

## Jetson 构建与启动

Jetson 只负责 ROS 2 构建和 CUDA 在线推理，不训练 policy 或 perception。以下命令使用
当前隔离部署名：

```bash
cd /home/jetson/jetson_asv_ws
source /opt/ros/humble/setup.bash
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
  execution_address:=<UE5_HOST_IP> execution_port:=8081
```

必须看到真实 CUDA 链路的状态，而不是只看到进程启动：

```text
LANGUAGE_READY_VALID ... release_model=true
POLICY_READY backend=torch_cuda inputs=task_embedding+structured_entities+previous_action output=[desired_x,desired_y]
PERCEPTION_TRACE visible=True
POLICY_TRACE entity_valid=True
PERCEPTION_PERF_TRACE valid=True
```

UE5 场景另行启动并连接 bridge；headless C++ 运动学执行器使用 8081。UE5 工程启动参数
不由本仓库定义，不要把不存在的 UE5 命令当作项目命令。

## PC 训练边界

采集与在线闭环是两种独立模式，不能同时占用同一 UE5/TCP 通道：

```bash
ros2 launch asv_bringup collect.launch.py \
  slot_id:=<SLOT> layout_id:=<LAYOUT> motion_state:=S2 scene_seed:=<SEED> \
  execution_address:=<UE5_HOST_IP> execution_port:=8081
```

PC 训练在每个 `(run, instruction, frame)` 时刻使用一个当前帧专家
`[desired_x, desired_y]` 点，不把一段轨迹当作 policy 输出。`previous_action` 只取同一
run、同一 instruction 的相邻前帧；首帧是零向量且 `previous_action_valid=false`。训练时
可使用 UE Entities 作为离线监督；部署推理不读取 Entities，也不把 ego 送入 policy。

训练数据、日志、checkpoint、感知模型和 Qwen 目录放在仓库外；Jetson 不承担训练工作。

## 当前第三轮闭环证据

当前可引用的真实证据是 L7/S2/`seed=230908`、slot `FINAL-S2-230908-V5`、
`MaxRuntimeSeconds=120`：

- Jetson log：`/mnt/c/Temp/asv_vla_closed_loop_20260805/jetson_l7_s2_230908_20260805_third.log`
- UE log：`/mnt/c/Temp/asv_vla_ue_l7_s2_230908_20260805_third.log`
- UE `SCENE_UE_COMPLETE`：`runtime_seconds=120.01`
- UE `SCENE_EXEC_APPLY` 最终 count：`450`

以上日志同时包含 `LANGUAGE_READY_VALID release_model=true`、CUDA policy 的单点输入/输出
trace、`PERCEPTION_TRACE visible=True`、`POLICY_TRACE entity_valid=True` 和
`PERCEPTION_PERF_TRACE valid=True`。这些是当前闭环证据，不代表跨颜色、距离、seed 或
布局的统计泛化。

失败轮次必须保留并明确标为失败，不能作为成功证据；不能用 UE 真值、专家 publisher、
缓存 embedding、ONNX/CPU 后端补齐失败。新分析应追加到现有结果，不能覆盖旧图或原始
learning curves。

## 验证与故障安全

文档收尾至少检查：

```bash
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
git diff --check
```

CUDA 不可用、模型加载失败、任务/实体身份不匹配、数据陈旧、输入 mask 无效或 safety
gate 拒绝时，必须输出零位移 hold，并标记 `valid=false`；不能静默回退 CPU、读取缓存
embedding 或订阅 UE 真值。第一帧 tracker 速度无效，只有同一身份的相邻图像帧才能产生
有效相对速度。

更多消息字段和不变量见 [`docs/interfaces.md`](docs/interfaces.md) 与
[`ARCHITECTURE.md`](ARCHITECTURE.md)。
