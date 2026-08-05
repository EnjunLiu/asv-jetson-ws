# ASV VLA

面向 Jetson/ROS 2 的 UE5 ASV 图像条件 VLA 原型。在线闭环的输入边界是：

```text
JPEG 图像 (/ue/camera_frame) + 任务文本 (/task/text)
  -> 感知头：图像 + 任务嵌入 -> 结构化实体 -> 跨帧 tracker 计算相对速度
  -> 决策头：任务嵌入 + 结构化实体 -> 一个 body-frame 期望位移点
  -> safety_gate -> point_controller
  -> desired_x/desired_y -> UE5 运动学执行
```

`/ue/entities` 是 UE 真值通道，只能用于 recorder、replay 和 PC 离线监督；在线感知、
tracker、visual encoder、policy 和 safety gate 不得订阅它。在线 VLA 不输出左右推进器
指令，真实船的低层控制器属于独立工程。

## ROS 2 包

| 包 | 责任 |
| --- | --- |
| `asv_jetson_interfaces` | CameraFrame、UEASVState、UEEntity、TaskFeatures、TaskEmbedding、DecisionPoint 等消息 |
| `asv_ue_bridge` | UE5 TCP JSON、JPEG、本船状态、真值标签发布，以及运动学 setpoint 输出 |
| `asv_vla` | 图像实体感知、任务筛选、跨帧 tracker、Qwen/策略、安全门、单点控制、采集和回放节点 |
| `asv_bringup` | `vla_closed_loop.launch.py`、`collect.launch.py`、`record_episode.launch.py`、`replay_episode.launch.py` |

在线 launch 不启动 ESP32、控制管理器或推进器分配器。

## Jetson 构建与启动

以下命令中的 `<JETSON_WS>`、`<UE5_HOST_IP>` 和模型文件名是部署占位符。模型二进制
不在 Git 中，先把它们放到 `<JETSON_WS>/models/`；路径和模型契约见
[`models/manifest.yaml`](models/manifest.yaml)。

```bash
cd <JETSON_WS>
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select asv_jetson_interfaces asv_ue_bridge asv_vla asv_bringup
source install/setup.bash

ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=<JETSON_WS>/models/<POLICY_CHECKPOINT>.pt \
  perception_model_path:=<JETSON_WS>/models/<PERCEPTION_MODEL>.npz \
  language_model_path:=<JETSON_WS>/models/Qwen3-Embedding-0.6B \
  language_device:=cuda visual_device:=cuda policy_device:=cuda \
  language_release_after_encode:=<true-or-false> \
  execution_address:=<UE5_HOST_IP> execution_port:=8081
```

`vla_closed_loop.launch.py` 的默认路径分别是
`models/policy_sine_near_image_color_seed42.pt`、
`models/image_entity_color_calibrated_v1.npz` 和
`models/Qwen3-Embedding-0.6B`（均相对于 `/home/jetson/jetson_asv_ws`）。
8 GB Jetson 的最终选定日志使用了 `language_release_after_encode:=true`；这会释放
Qwen 权重但保留真实 Qwen CUDA embedding。manifest 的部署契约字段为
`release_model_after_encode: false`（即保留权重常驻），并标注
`qwen_weight_resident_after_encode: true`；应按设备内存明确选择，不能改成缓存 `.npy`
或 CPU 推理。

UE5 场景需另行启动，并连接 bridge 的 TCP 8080；如果使用 headless C++ 运动学执行器，
将 `execution_address` 指向 UE5 主机的 8081 端口。UE5 工程启动参数不由本仓库定义，
不要把不存在的 UE5 命令当作本项目命令。

## 采集与 PC 训练边界

采集 launch 是独立运行模式，不要和在线闭环同时启动：

```bash
ros2 launch asv_bringup collect.launch.py \
  slot_id:=<SLOT> layout_id:=<LAYOUT> motion_state:=S2 scene_seed:=<SEED> \
  execution_address:=<UE5_HOST_IP> execution_port:=8081
```

`collect.launch.py` 的 recorder 保存 JPEG、`UEASVState`、UE Entities、任务和身份元数据；
expert action 只用于离线标签。PC 侧可使用 UE Entities 训练图像实体模型，并由
相邻图像实体和时间戳计算速度，再结合冻结 Qwen 任务嵌入和结构化实体训练单点决策头。
部署到 Jetson 后，在线输入只有图像和任务文本，不读取 UE Entities 或 ego。

PC 数据、日志、checkpoint、感知模型和 Qwen 目录放在仓库外的 `pc_datasets/` 或部署
目录，不提交 Git。仓库只保留源码、合同和 `models/manifest.yaml`；当前 manifest 中的
`artifact_sha256` 用于核对外部模型，而不是模型文件本身。

## 2026-08-05 三场景结果

证据目录为 `pc_datasets/final_selected_best_per_scene_20260805/`。三次运行均为
L7 / S2 / scene seed 230908、每场景 186 个样本；下表的模型是按场景选择的组合：

| 场景 | 选定 policy | 选定 perception | 最终误差 | 平均误差 | 60 s 稳定窗口在目标带内 |
| --- | --- | --- | ---: | ---: | ---: |
| RED 3 m | `policy_near_rgb_v8_seed42.pt` | `perception_retrained_20260805_pc_native_aug_v1.npz` | 0.389 m | 0.508 m | 60.7% |
| BLUE 3 m | `policy_near_rgb_v7_seed23.pt` | `image_entity_color_calibrated_v7.npz` | 0.883 m | 0.801 m | 0.0% |
| RED 4 m | `policy_near_rgb_v8_seed42.pt` | `perception_retrained_20260805_pc_native_aug_v1.npz` | 0.394 m | 0.510 m | 49.2% |

三场景审计和 UE5 完成标记为 PASS；Jetson 日志观察到 10 次 `POLICY_DRIVEN`、9 次
`STANDOFF_BACKSTOP`，没有 `TARGET_LOST` 或 `FAIL_CLOSED`。这证明上述三个指定场景在
各自选定模型和该单一运行条件下完成了在线 CUDA 闭环，不证明一个统一模型跨颜色、距离、
seed 或布局的泛化：选择清单明确为 `best_per_scene_composite`，
`single_uniform_run_proof=false`。BLUE 3 m 的稳定窗口 0% 也应保留在结论中，不能写成
三场景均稳定跟踪。

## 验证

PC 代码检查：

```bash
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
PYTHONPATH=src/asv_vla pytest -q
python3 -m compileall -q src/asv_vla training
git diff --check
```

Jetson/UE5 闭环验收应从 launch 输出和 UE5 日志核对：

```text
LANGUAGE_READY_VALID ... device=cuda
image_entity_perception ... device=cuda
POLICY_READY backend=torch_cuda device=cuda
POLICY_TRACE ... entity_valid=true
SCENE_EXEC_APPLY ...
SCENE_UE_COMPLETE ...
```

## 故障安全约束

- CUDA 不可用、模型加载失败、输入消息无效、身份不匹配或数据陈旧时，必须
  `valid=false` 并 hold；不能静默回退 CPU、缓存 embedding 或 UE 真值。
- 第一帧没有有效速度；速度只能由相同 `run_id/scene_seed/frame_index/stamp_us` 的相邻
  图像实体计算。
- `safety_gate` 检查有限性、速度、曲率、碰撞余量、陈旧和身份；目标缺失或 OOD 必须
  fail-closed，standoff guard 只能输出安全轨迹或 hold。
- `point_controller` 和 adapter 只发布 `desired_x/desired_y`。不可执行或
  `safe_stop` 时，UE5 收到零位移 hold；不得把零位移冒充有效推进命令。
- 每种模式只启动一套 bridge、policy、safety gate 和 recorder/expert owner；采集与闭环
  不得并行占用同一 UE5/TCP 通道。

更多接口和不变量见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。
