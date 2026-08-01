# 最终接口契约

本文档只描述活动的 VLA/UE5 路径。Jetson 上层和底层实体执行解耦：上层只给二维
期望位移，UE5 仿真执行器或后续独立底层控制器负责把它变成运动。

## 输入边界

UE5 bridge 发布：

- `/ue/camera_frame`：SceneCapture 的 JPEG、`run_id`、`scene_seed`、`frame_index`、
  `stamp_us`。
- `/ue/asv_state`：实时自船状态。策略目前使用 `surge_velocity` 和 `yaw_rate`，不使用
  UE 目标真值。
- `/ue/entities`：原始实体标签和速度，仅供 `record_episode` 及离线监督；不能被在线
  VLA 节点订阅。

任务输入 `/task/text` 是事件驱动字符串，例如：

```text
跟随红色目标船，保持3米距离
follow the blue boat
follow the left target
stop
```

`image_entity_perception` 同时消费 JPEG 和任务指令。模型先从 JPEG 产生候选几何，
再用解析后的任务选择相关实体；这里的选择改变可见性/目标标记，不读取 `/ue/entities`。
当前部署 artifact 的几何验收范围是约 5 m 内的红色目标；蓝色/左右任务若没有对应
校准模型会安全地输出不可见/hold。

## 感知和速度

`/vla/perceived_entities` 的 `UEEntityArray` 是图像推断结果：

- `source=image_perception`；
- `relative_x/y/z` 来自图像模型（当前近距离红色目标使用可审计 RGB 校准器）；
- 首帧速度字段为零且 `velocity_valid=false`；
- `bbox_*` 是从图像推断几何投影出的诊断框，不是 UE 传入的真值框。

`/vla/tracked_entities` 由相邻图像帧的几何和时间戳计算相对速度，跨 Run 或帧身份不
连续时会清空历史并 fail-closed。任何单帧都不会声称直接观察到速度。

`task_entity_tensor` 只接受 `image_perception` 或 `temporal_tracker` 来源；传入其他来源
会发布无效消息。它输出固定形状的 `TaskFeatures`，供 visual encoder 和 policy 使用。

## 任务级多模态策略

```text
/ue/camera_frame + /task/text + /ue/asv_state
       |             |              |
       v             v              v
image perception  Qwen CUDA       ego normalization
       |             |              |
temporal tracker -> visual/task tensors
                       |
                       v
             Torch CUDA policy checkpoint
                       |
                       v
             SelectedTrajectory [H=20, 2]
```

语言向量为 256-D、有限、L2 归一化的真实 Qwen3-Embedding-0.6B 输出。在线默认权重
常驻 CUDA 并支持新的 `/task/text`；若显式设置 `language_release_after_encode=true`，
只释放 Qwen 权重而保留已编码向量，不能改用 `.npy` 或 CPU stub。

策略输入的实体、视觉、语言、ego 和身份必须全部匹配同一
`run_id/scene_seed/frame_index`。策略输出为一条 20 点二维位移序列
`[dx0,dy0,...,dx19,dy19]`（米，`base_link`，`dt=0.2 s`）。缺失 CUDA 模型、身份或
有效输入时，输出固定零轨迹并 `valid=false/safe_stop=true`。

## 执行边界

```text
/vla/policy_trajectory
        -> /vla/selected_trajectory (safety gate 唯一发布者)
        -> /decision/output (desired_x, desired_y)
        -> /ue/kinematic_setpoint
        -> UE5 C++ executor:8081
```

上层不发布左右推力。`+X` 为船体前方，`+Y` 为左舷/左方；`desired_x/y` 是米制二维
期望位移。UE5 仿真可以直接设置下一期望位姿用于演示；真实船接入时，应由独立底层
控制器消费同一 setpoint，不改变 VLA 契约。

## 数据和专家轨迹边界

UE `Entities` 与离线 `ExpertTrajectory` 只用于收集图像监督和训练标签：

```text
JPEG + Entities + Run metadata -> PC dataset -> frozen model checkpoint
JPEG + task + ego             -> Jetson online inference -> desired_x/y
```

专家轨迹不能发布到 `/vla/selected_trajectory`，不能直接连接执行器。速度标签可用于
训练/验证 tracker，但在线速度必须来自连续感知帧。

## 失败闭环

以下任一条件会让在线输出 hold：空任务、相机无效、感知模型加载失败、实体来源不可信、
Run identity 不匹配、Qwen/Torch CUDA 不可用、策略输出非有限或 safety gate 拒绝。
这类 hold 是最终系统的可接受行为，不得通过 UE 真值、旧专家 publisher、ONNX/CPU 后端
或预计算语言 embedding 绕过。
