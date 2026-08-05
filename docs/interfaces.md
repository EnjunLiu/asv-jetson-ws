# 最终接口契约

本文档只描述活动的 VLA/UE5 路径。Jetson 上层和底层实体执行解耦：上层只给二维
期望位移，UE5 仿真执行器或后续独立底层控制器负责把它变成运动。

## 输入边界

UE5 bridge 发布：

- `/ue/camera_frame`：SceneCapture 的 JPEG、`run_id`、`scene_seed`、`frame_index`、
  `stamp_us`。
- `/ue/asv_state`：实时自船状态，仅供录制和离线审计；决策头不接收 ego。
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
会发布无效消息。它输出固定形状的 `TaskFeatures`，作为决策头的结构化实体输入。

## 任务级多模态策略

```text
/ue/camera_frame + /task/text
       |             |
       v             v
感知头：图像 + Qwen CUDA 任务嵌入
       |
temporal tracker -> structured TaskFeatures（相对位置/颜色/相对速度）
       |
       v
决策头：Qwen 任务嵌入 + TaskFeatures
       |
       v
DecisionPoint [desired_x, desired_y]
```

语言向量为 256-D、有限、L2 归一化的真实 Qwen3-Embedding-0.6B 输出。在线默认权重
常驻 CUDA 并支持新的 `/task/text`；若显式设置 `language_release_after_encode=true`，
只释放 Qwen 权重而保留已编码向量，不能改用 `.npy` 或 CPU stub。

决策头输入是以下七项：

- `language`：任务指令的 256-D Qwen embedding；
- `entity_geometry`：感知头/跨帧 tracker 输出的固定 `16 x 16` 结构化几何，包含
  颜色槽位、相对位置、相对速度等信息；
- `previous_action`：上一控制帧实际通过 safety gate 的 `[desired_x, desired_y]`；
- `language_valid`、`entity_geometry_mask`、`previous_action_valid`、`policy_input_valid`。

图像由感知头消费，ego 不进入决策头。`TaskFeatures` 和逐帧输出必须匹配同一
`run_id/scene_seed/frame_index`；语言 embedding 是任务级消息，与 TaskFeatures 通过
instruction 文本/可用的 `instruction_id` 匹配，不把 Qwen 编码时刻当成相机帧。混帧或
任务不匹配时直接 fail-closed。策略输出一个二维点（米，`base_link`，`dt=0.2 s`），而
不是轨迹序列。首帧、Run/scene 切换、
帧不连续、上一动作无效或 gate 拒绝时，`previous_action` 为零且
`previous_action_valid=false`。缺失 CUDA 模型、身份或有效输入时，输出零位移并
`valid=false/safe_stop=true`。

## 执行边界

```text
/vla/policy_point
        -> /vla/selected_point (safety gate 唯一发布者)
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
JPEG + task                   -> Jetson online inference -> desired_x/y
```

每个 `(Run, instruction, frame)` 都是一个独立的单步专家样本，专家字段只有当前帧的
`desired_x/desired_y`、STOP/valid 和身份元数据；它不是一条要由策略回归的离线轨迹。
训练 cache 另外按同一 Run、同一 instruction、相邻前帧保存
`previous_expert_action`，用于动作平滑损失。专家 action 不能发布到
`/vla/selected_point`，不能直接连接执行器。速度标签可用于训练/验证 tracker，但在线
速度必须来自连续感知帧。

## 失败闭环

以下任一条件会让在线输出 hold：空任务、相机无效、感知模型加载失败、实体来源不可信、
Run identity 不匹配、Qwen/Torch CUDA 不可用、策略输出非有限或 safety gate 拒绝。
这类 hold 是最终系统的可接受行为，不得通过 UE 真值、旧专家 publisher、ONNX/CPU 后端
或预计算语言 embedding 绕过。
