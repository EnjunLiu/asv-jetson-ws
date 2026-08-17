# ASV Jetson 在线闭环源码教材

> 目标：从零理解并能够独立讲解、构建、调试 `UE5 -> ROS 2 -> Jetson VLA -> UE5` 闭环。

本文按系统数据流组织，而不是按文件名排序。源码链接采用 GitHub 相对路径；仓库上传后可直接跳到相应代码行。

## 目录

1. [学习方法](#第-0-章学习方法)
2. [系统目标与边界](#第-1-章系统目标与边界)
3. [仓库和 ROS 2 包](#第-2-章仓库和-ros-2-包)
4. [ROS 消息合同](#第-3-章ros-消息合同)
5. [UE5 Bridge](#第-4-章ue5-bridge)
6. [任务文本与语言编码](#第-5-章任务文本与语言编码)
7. [图像感知](#第-6-章图像感知)
8. [时序跟踪与实体张量](#第-7-章时序跟踪与实体张量)
9. [策略模型与在线推理](#第-8-章策略模型与在线推理)
10. [任务守卫与安全门](#第-9-章任务守卫与安全门)
11. [控制与 UE5 执行](#第-10-章控制与-ue5-执行)
12. [Launch、部署和模型](#第-11-章launch部署和模型)
13. [测试和调试](#第-12-章测试和调试)
14. [逐文件索引](#附录逐文件索引)

---

## 第 0 章：学习方法

### 0.1 阅读顺序

不要先通读最长的感知文件。正确顺序是：

```text
系统边界 -> ROS 消息 -> Launch 节点图 -> 节点输入输出
         -> 纯算法 -> 测试 -> Jetson/UE5 实机验证
```

每读一个模块，都写下：

```text
解决的问题：
输入 topic/参数：
输出 topic：
核心数据结构：
正常路径：
失败路径：
身份字段如何传播：
对应测试：
```

### 0.2 区分算法和 ROS 胶水

项目通常把功能分成两层。例如：

- [safety_gate.py](./src/vla/vla/safety_gate.py#L1)：安全判定、topic、参数、定时器和发布。

先读纯算法有利于理解数学和异常条件，再读节点代码理解它如何进入 ROS 图。

### 本章问题与答案

**问：为什么不能只看 `*_node.py`？**  
答：节点代码主要处理通信，真正的模型、几何和安全逻辑常在无 `_node` 的文件中。

**问：为什么先读消息？**  
答：消息是模块之间的正式合同。先理解输入输出，才能判断函数为什么检查某些字段。

---

## 第 1 章：系统目标与边界

### 1.1 完整数据流

总入口是 [vla_closed_loop.launch.py](./src/bringup/launch/vla_closed_loop.launch.py#L45)。

```text
UE5
 ├─ Camera/State/Truth
 v
bridge
 ├─ /ue/camera_frame
 v
image_entity_perception
 ├─ /vla/perceived_entities
 v
temporal_entity_tracker
 ├─ /vla/tracked_entities
 v
entity_features
 ├─ /vla/entity_features ───────────────┐
                                      |
task_instruction -> language_qwen     |
                    └─ embedding ─────┤
                                      v
                                  vla_policy
                                      |
                          /vla/policy_displacement
                                      v
                                  safety_gate
                                      |
                        /control/desired_displacement
                                      v
                           bridge -> UE5 executor
                                      |
                                      v
                                UE5 executor
```

### 1.2 动作接口

策略输出 `desired_x, desired_y`：ROS `base_link` 中的单步期望位移，单位米。合同位于 [trajectory_contract.py](./src/vla/vla/trajectory_contract.py#L10)：

- `DT_SEC = 0.2`
- `ACTION_DIM = 2`
- `MAX_DISPLACEMENT_M = 0.30`

它不是推进器转速，也不是多点轨迹。这样任务层策略可以与 UE5 执行器或未来真实低层控制器解耦。

### 1.3 特权真值边界

bridge 会发布 `/ue/entities`，但在线感知节点 [ImageEntityPerceptionNode](./src/vla/vla/image_entity_perception_node.py#L206) 不订阅它，只使用相机图像、任务文本和语言 embedding。

原因是现实世界不存在 UE5 真值。在线使用它会让仿真结果失去真实性。该约束由 [test_runtime_identity_contract.py](./src/vla/test/test_runtime_identity_contract.py#L1) 检查。

### 1.4 fail-closed

无法证明输入有效时，系统停止而不是猜测：

```text
模型失败 / 错帧 / 超时 / NaN / 碰撞风险
                    -> valid=false 或 safe_stop=true
                    -> hold_position
```

### 本章问题与答案

**问：为什么不直接输出推进器推力？**  
答：二维位移是任务层最小充分接口；推进器控制属于具体船体动力学和低层硬件。

**问：`/ue/entities` 为什么可以存在？**  
答：它可用于观测和离线监督，但不能进入在线视觉决策路径。

**问：fail-closed 比继续执行上一动作安全吗？**  
答：是。感知失效时旧动作可能持续把船带向危险区域，明确停止更加可预测。

---

## 第 2 章：仓库和 ROS 2 包

```text
asv-jetson-ws/
├── README.md
├── models/manifest.yaml
└── src/
    ├── interfaces  # 消息
    ├── bridge          # UE5 TCP/C++
    ├── vla                # VLA 算法和节点
    └── launch            # Launch
```

### 2.1 接口包

[接口 CMakeLists.txt](./src/interfaces/CMakeLists.txt#L1) 使用 `rosidl_generate_interfaces` 将 `.msg` 生成 Python/C++ 类型。[package.xml](./src/interfaces/package.xml) 声明 ROS 依赖。

### 2.2 Bridge 包

[bridge CMakeLists.txt](./src/bridge/CMakeLists.txt#L1) 编译 C++ 节点并链接 ROS 2、nlohmann JSON 和线程库。

### 2.3 Python VLA 包

[setup.py](./src/vla/setup.py#L24) 注册 `ros2 run` 入口。例如：

```text
vla_policy = vla.vla_policy_node:main
```

因此 `ros2 run vla vla_policy` 会调用该文件的 `main()`。

- [setup.cfg](./src/vla/setup.cfg)：脚本安装目录。
- [package.xml](./src/vla/package.xml)：ROS/Python 依赖。
- [resource/vla](./src/vla/resource/vla)：ament 索引标记，不能因内容为空而删除。
- [requirements-language.txt](./src/vla/requirements-language.txt)：Qwen 额外依赖。

### 本章问题与答案

**问：为什么 `resource/vla` 是空文件却有用？**  
答：它让 ament 索引识别 Python ROS 包。

**问：为什么 bridge 用 C++，模型用 Python？**  
答：socket/线程通信适合稳定低延迟 C++；模型生态和研究迭代主要位于 Python/PyTorch。

---

## 第 3 章：ROS 消息合同

### 3.1 传感器消息

[CameraFrame.msg](./src/interfaces/msg/CameraFrame.msg#L1) 保存 JPEG 字节和 `run_id/scene_seed/frame_index`。[ASVState.msg](./src/interfaces/msg/ASVState.msg#L1) 保存位姿、纵向速度和偏航角速度。

当前策略明确排除 ego 输入，所以 `ASVState` 主要是 bridge 状态接口，不进入决策头。

### 3.2 实体消息

[Entity.msg](./src/interfaces/msg/Entity.msg#L1) 描述一个实体：

- 类别、颜色、是否目标。
- ASV 局部坐标位置和速度。
- bbox、置信度和对应有效位。
- `source`：`ue_truth`、`image_perception` 或 `temporal_tracker`。

[EntityArray.msg](./src/interfaces/msg/EntityArray.msg#L1) 为一帧实体添加身份、来源、任务文本和整体有效性。

### 3.3 语言和张量

[TaskEmbedding.msg](./src/interfaces/msg/TaskEmbedding.msg#L1) 包含 256 维 embedding、模型 ID 和有效性。

[EntityFeatures.msg](./src/interfaces/msg/EntityFeatures.msg#L1) 包含固定形状实体张量：16 个槽位，每个槽位 16 维，并用 `mask` 区分真实实体与 padding。

### 3.4 位移和执行

- [DesiredDisplacement.msg](./src/interfaces/msg/DesiredDisplacement.msg#L1)：策略、安全门和 UE5 bridge 共用的最终二维位移合同，包含源身份、安全停止和拒绝原因。

### 3.5 身份传播

```text
run_id       一次运行
scene_seed   场景配置
frame_index  运行中的帧序号
stamp_us     时效性和时间计算
```

时间戳可能在不同运行中重复，所以不能单独作为身份。

### 本章问题与答案

**问：空实体数组和 `valid=false` 相同吗？**  
答：不同。空数组可能是有效观测中没有目标；`valid=false` 表示数据或处理过程不可信。

**问：为什么 padding 为零还需要 mask？**  
答：否则网络可能把全零槽位误认为位于原点的真实实体。

**问：为什么最终消息仍保留源帧？**  
答：用于证明动作来自哪一帧输入，并拒绝无身份或错帧执行。

---

## 第 4 章：UE5 Bridge

核心文件：[ue_object_deliverer_bridge_node.cpp](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L185)，配置：[ue_bridge.yaml](./src/bridge/config/ue_bridge.yaml#L1)。

### 4.1 接收路径

1. [构造函数](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L192) 读取参数，创建 publisher/subscriber。
2. [server_loop](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L420) 监听 UE5 连接。
3. [receive_loop](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L509) 累积字节并按 `__OD_END__` 拆包。
4. [process_json](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L553) 解析一帧。
5. [validate_frame_metadata](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L657) 检查身份。
6. [publish_optional_camera](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L853) 和 [publish_optional_entities](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L745) 发布 ROS 消息。

文件开头的 [JSON 读取辅助函数](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L40) 拒绝错误类型、空字符串和 NaN/Inf。

### 4.2 单位和坐标转换

UE5 使用厘米，ROS 使用米：

```text
入站 position_scale = 0.01
出站 kinematic_position_scale = 100.0
```

UE 局部 `+Y` 向右，ROS `base_link +Y` 向左，因此两侧横向符号均为 `-1.0`。该错误最直观的表现是船向目标相反侧移动。

### 4.3 出站路径

bridge 在 [第 278 行](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L278) 订阅最终 setpoint。[send_kinematic_setpoint](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L895) 检查命令、米转厘米、转换横向符号并序列化 JSON。

若配置 `execution_address`，由 [execution_loop](./src/bridge/src/ue_object_deliverer_bridge_node.cpp#L346) 维护独立 UE5 executor 连接。

### 本章问题与答案

**问：为什么入站和出站都要转换 Y 轴？**  
答：UE 和 ROS 横向轴定义相反，每个通信方向都必须转换。

**问：bridge 发布真值实体是否意味着策略使用真值？**  
答：否。在线策略使用 `/vla/tracked_entities`，感知节点没有订阅真值 topic。

**问：为什么需要 packet terminator？**  
答：TCP 是字节流，没有天然消息边界，需要应用层标记一条 JSON 在哪里结束。

---

## 第 5 章：任务文本与语言编码

### 5.1 任务发布

[task_instruction_node.py](./src/vla/vla/task_instruction_node.py#L39) 从 `task_text` 参数读取任务，[校验函数](./src/vla/vla/task_instruction_node.py#L30) 拒绝空文本，并以 transient-local QoS 发布 `/task/text`。这使晚启动的 Qwen 节点仍能收到当前任务。

### 5.2 Qwen 算法

[USVLanguageEncoder](./src/vla/vla/language_encoder.py#L53) 完成模型加载、冻结、文本标准化和 embedding 计算。

[_compute](./src/vla/vla/language_encoder.py#L194) 的流程：

```text
任务文本 -> ASV prompt -> tokenizer -> Qwen hidden states
        -> pooling -> 256维 -> finite检查/归一化
```

[encode_with_metadata](./src/vla/vla/language_encoder.py#L254) 增加进程内缓存。

### 5.3 Qwen ROS 节点

[LanguageQwenNode](./src/vla/vla/language_qwen_node.py#L113) 订阅任务，发布 embedding 和模块状态。主入口是 [on_task](./src/vla/vla/language_qwen_node.py#L248)。

失败时发布 256 维零向量且 `valid=false`，不能静默回退 CPU。[_release_encoder_model](./src/vla/vla/language_qwen_node.py#L216) 可在首次编码后释放 Qwen 权重，为感知和策略让出 Jetson 内存。

### 本章问题与答案

**问：为什么零 embedding 还需要 `valid=false`？**  
答：零只是固定 shape 占位；有效位明确告诉下游不能推理。

**问：为什么先启动 Qwen？**  
答：先获得任务向量并释放大型语言模型，可降低后续感知和策略同时加载时的内存峰值。

**问：进程内缓存是否等于读取静态假 embedding？**  
答：不等于。缓存内容来自本次真实 Qwen 推理，只是避免重复编码相同文本。

---

## 第 6 章：图像感知

### 6.1 ROS 节点

[ImageEntityPerceptionNode](./src/vla/vla/image_entity_perception_node.py#L206) 订阅相机、任务和语言 embedding，发布 `/vla/perceived_entities`。[on_frame](./src/vla/vla/image_entity_perception_node.py#L406) 是单帧处理主入口。

### 6.2 图像处理

- [decode_camera_image](./src/vla/vla/visual_encoder.py#L65)：解码 JPEG。
- [enhance_low_light_image](./src/vla/vla/visual_encoder.py#L83)：gamma、亮度、对比度增强。
- [project_target_to_pixel](./src/vla/vla/visual_encoder.py#L154)：几何投影辅助。
- [FrozenMobileNetEncoder](./src/vla/vla/visual_encoder.py#L282)：冻结视觉编码器；当前在线感知主要复用其中图像处理工具。

### 6.3 感知特征

[image_entity_perception.py](./src/vla/vla/image_entity_perception.py#L44) 将图像缩放为 `32 x 18` 网格，构造 RGB、颜色/亮度证据图和空间矩特征，再拼接 256 维任务 embedding。

- [extract_image_features](./src/vla/vla/image_entity_perception.py#L523)：图像特征。
- [validate_task_embedding](./src/vla/vla/image_entity_perception.py#L93)：语言维度和 finite 检查。
- [parse_task_instruction](./src/vla/vla/image_entity_perception.py#L137)：提取跟随/停止、颜色和距离。
- [select_task_entities](./src/vla/vla/image_entity_perception.py#L718)：选择任务相关实体。

### 6.4 加载和预测

[ImageEntityModel.load](./src/vla/vla/image_entity_perception.py#L816) 检查 `.npz` 内的权重形状、输入合同、模型标识和设备。[predict](./src/vla/vla/image_entity_perception.py#L887) 输出 [ImageEntityPrediction](./src/vla/vla/image_entity_perception.py#L628)：可见性、相对位置、bbox 和置信度。

单帧感知不宣称速度有效，速度交给后续 tracker。

### 6.5 颜色几何先验

[calibrated_color_geometry](./src/vla/vla/image_entity_perception.py#L251) 根据颜色连通域、面积和像素位置估计目标几何。[_predict_with_color_reference](./src/vla/vla/image_entity_perception_node.py#L81) 用它约束或补充模型结果。

优点是简单、可解释、适合当前红/蓝 UE5 任务；缺点是依赖颜色、光照和标定，不能夸大真实环境泛化。

### 本章问题与答案

**问：为什么感知需要语言 embedding？**  
答：同一图像可能包含多个实体，任务语义决定哪个实体与当前行为相关。

**问：为什么不由单帧直接输出可信速度？**  
答：速度需要位置随时间变化，跨帧估计更符合物理含义。

**问：代码中少量历史模型标识为什么未删除？**  
答：它们是现有权重内部兼容合同，随意改名会导致加载校验失败。

---

## 第 7 章：时序跟踪与实体张量

### 7.1 时序跟踪

[temporal_entity_tracker.py](./src/vla/vla/temporal_entity_tracker.py#L1) 的主要结构：

- [FrameMetadata](./src/vla/vla/temporal_entity_tracker.py#L28)：帧身份。
- [GeometryObservation](./src/vla/vla/temporal_entity_tracker.py#L58)：当前观测。
- [TrackedEntity](./src/vla/vla/temporal_entity_tracker.py#L134)：带速度输出。
- [TemporalEntityTracker](./src/vla/vla/temporal_entity_tracker.py#L212)：轨迹状态机。

[update](./src/vla/vla/temporal_entity_tracker.py#L258) 按实体 ID 更新轨迹，[_filter_velocity](./src/vla/vla/temporal_entity_tracker.py#L402) 用位置差和时间差估计速度，[_expire_tracks](./src/vla/vla/temporal_entity_tracker.py#L330) 删除过期轨迹。

[TemporalEntityTrackerNode](./src/vla/vla/temporal_entity_tracker.py#L571) 将其接入 `/vla/perceived_entities -> /vla/tracked_entities`。[_DropoutRecovery](./src/vla/vla/temporal_entity_tracker.py#L470) 短暂保留丢失目标，但运行/场景变化时必须清空。

### 7.2 实体张量

[build_entity_features](./src/vla/vla/entity_features.py#L169) 将可变长度实体转换为 `16 x 16`。

每个实体的 16 维由 [_entity_row](./src/vla/vla/entity_features.py#L136) 定义：

```text
0-2   x/y/z
3-5   vx/vy/vz
6     distance
7-8   bearing sin/cos
9     closing speed
10    time to CPA
11    CPA distance
12    is_target
13    is_risk
14    is_red
15    is_blue
```

[compute_entity_metrics](./src/vla/vla/entity_features.py#L63) 计算最近会遇点 CPA。实体按“目标 -> 风险 -> 普通”排序，避免无关近物体挤掉任务目标。

[EntityFeaturesNode](./src/vla/vla/entity_features.py#L283) 负责 ROS 转换；异常时通过 [_publish_invalid](./src/vla/vla/entity_features.py#L354) 显式发布无效输入。

### 本章问题与答案

**问：为什么场景切换必须清空 tracker？**  
答：否则旧位置会与新场景观测计算出虚假速度和幽灵轨迹。

**问：为什么方位用 sin/cos？**  
答：避免角度在 `-pi/pi` 处不连续。

**问：CPA 比当前距离多了什么？**  
答：它结合相对速度预测未来最近接近时间和距离。

---

## 第 8 章：策略模型与在线推理

### 8.1 输入合同

[模型清单](./models/manifest.yaml#L39) 明确策略输入：

```text
language [B,256]
entity_geometry [B,16,16]
previous_action [B,2]
以及 validity/mask
```

决策头明确排除原始图像、entity visual token 和 ego。图像信息已经在感知阶段转成结构化实体。

### 8.2 网络结构

[SmallActionPolicy](./src/vla/vla/policy_model.py#L116)：

```text
language -> MLP -------------------------+
16 entities -> shared MLP -> attention --+-> fusion -> action_head
previous action -> MLP ------------------+          -> stop_head
previous_action_valid -------------------+
```

[forward](./src/vla/vla/policy_model.py#L225) 先检查 shape/device/dtype，再用 mask 将无效实体清零。语言生成 query，与实体 token 计算注意力，因此不同任务能关注不同实体。

`previous_action_valid` 让网络区分“第一帧没有历史”和“上一帧真实执行零动作”。stop head 选择停止时，输出被精确置零。

### 8.3 模型运行器

[TorchPolicyRunner.load](./src/vla/vla/policy_model.py#L429) 加载 checkpoint、恢复配置、检查 state dict 和 CUDA 设备；请求 CUDA 失败时不允许静默 CPU fallback。[run](./src/vla/vla/policy_model.py#L495) 负责张量转换和推理。

### 8.4 ROS 策略节点

[VLAPolicyNode](./src/vla/vla/vla_policy_node.py#L289) 订阅语言、实体 tensor 和安全门最终动作，发布 `/vla/policy_displacement`。

[FrameSyncCache](./src/vla/vla/vla_policy_node.py#L214) 用 `(run_id, scene_seed, frame_index)` 同步异步消息。[_maybe_infer](./src/vla/vla/vla_policy_node.py#L701) 只有在输入完整、有效、未过期且身份一致时推理。

- [bound_policy_displacement](./src/vla/vla/vla_policy_node.py#L145)：动作范数限制。
- [smooth_policy_displacement](./src/vla/vla/vla_policy_node.py#L173)：相邻动作变化限制。
- [_publish_fail_closed](./src/vla/vla/vla_policy_node.py#L668)：异常停止。

策略使用安全门通过的实际动作作为下一帧历史，而不是可能已被拒绝的原始动作。

### 本章问题与答案

**问：为什么策略不再直接输入图像？**  
答：感知模块已将图像压缩成任务相关、可解释的结构化实体，降低决策头复杂度。

**问：为什么上一动作来自安全门输出？**  
答：那才是实际执行动作，策略历史应与环境真实反馈一致。

**问：FrameSyncCache 解决什么问题？**  
答：ROS 消息异步到达，它防止把一帧实体与另一帧身份或旧任务组合推理。

---

## 第 9 章：任务守卫与安全门

### 9.1 跟随距离守卫

[visual_standoff_guard.py](./src/vla/vla/visual_standoff_guard.py#L1) 为 FOLLOW 任务增加可解释几何约束：

1. [is_follow_instruction](./src/vla/vla/visual_standoff_guard.py#L79) 判断任务。
2. [extract_target_observation](./src/vla/vla/visual_standoff_guard.py#L146) 恢复目标位置/速度。
3. [compute_standoff_step](./src/vla/vla/visual_standoff_guard.py#L191) 计算维持距离动作。
4. [apply_standoff_guard](./src/vla/vla/visual_standoff_guard.py#L247) 决定保留、停止或替换策略动作。

模式包括 `policy_driven`、`deadband_hold`、`backstop`、`fail_closed` 和非 FOLLOW 的 `pass_through`。

### 9.2 通用安全门

[safety_gate.py](./src/vla/vla/safety_gate.py#L22) 定义拒绝码：过期、模态/shape 错误、NaN、超速、碰撞风险、不可达和急停。

[evaluate_safety_gate](./src/vla/vla/safety_gate.py#L197) 检查动作和实体，再返回 `SafetyGateResult`。

[SafetyGateNode](./src/vla/vla/safety_gate.py#L322) 订阅 `/vla/policy_displacement` 与 `/vla/tracked_entities`，发布最终 `/control/desired_displacement`。它同时调用 [limit_displacement_rate](./src/vla/vla/safety_gate.py#L85) 限制相邻动作变化；[_on_timeout](./src/vla/vla/safety_gate.py#L506) 在上游停止发布时主动停止。

任务守卫回答“动作是否符合跟随语义”，安全门回答“动作在数值、时效、运动学和碰撞上是否可执行”。

### 本章问题与答案

**问：为什么需要 deadband？**  
答：避免目标距离附近因微小误差反复前后振荡。

**问：backstop 与 fail-closed 的区别？**  
答：backstop 有可信目标，可用规则动作修正；fail-closed 连可信目标都没有，只能停止。

**问：为什么安全门需要独立超时定时器？**  
答：如果上游完全停止，就不会有新消息触发普通回调。

---

## 第 10 章：控制与 UE5 执行

安全门输出的 [DesiredDisplacement.msg](./src/interfaces/msg/DesiredDisplacement.msg#L1) 已经是经过时效、碰撞、位移和变化率检查的最终任务层控制命令。它是 UE5 和未来 ESP32 控制链共同的分叉点。

bridge 直接订阅 `/control/desired_displacement`，在执行边界检查运行身份、有限值、坐标系和 `valid/safe_stop`，然后将位移转换为 UE5 执行 JSON。无效命令统一转换为保持当前位置。

未来 ESP32 adapter 应并行订阅同一个 `/control/desired_displacement`，再补充真实速度和偏航角速度，转换成固件的 `ControlInput`。UE5 专用 setpoint 不应发送给 ESP32。

### 本章问题与答案

**问：为什么 bridge 还要检查身份？**  
答：执行边界不能盲目信任上游，必须拒绝其他节点构造的无身份命令。

**问：零位移和 `hold_position` 完全相同吗？**  
答：运动结果相近，但 `hold_position` 是明确执行语义，可区分安全停止与普通零动作。

**问：分层对真实船有什么价值？**  
答：可新增 adapter 将二维位移转成速度、航向或推进器命令，而不改 VLA。

---

## 第 11 章：Launch、部署和模型

### 11.1 节点编排

[launch 参数](./src/bringup/launch/vla_closed_loop.launch.py#L79) 包括模型路径、CUDA 设备、任务、UE5 executor 地址和启动延迟。[节点定义](./src/bringup/launch/vla_closed_loop.launch.py#L144) 启动 bridge、任务、语言、感知、跟踪、tensor、策略、安全、控制和 adapter。

感知和策略通过 `TimerAction` 延迟启动。Qwen 先完成首次 CUDA 编码并释放模型，减少 Jetson 同时加载三类模型的内存峰值。

### 11.2 模型布局

```text
asv-hil-runtime/
├── models/
│   ├── policy_single_point.pt
│   ├── perception_image_conditioned.npz
│   └── Qwen3-Embedding-0.6B/
└── jetson/  # 本仓库
```

[manifest.yaml](./models/manifest.yaml#L1) 保存模型名称、SHA-256、输入输出合同和设备要求。它只是期望清单，不能证明真实文件存在；部署时仍需重新计算哈希。

### 11.3 构建和启动

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select interfaces bridge vla launch
source install/setup.bash

ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=../models \
  execution_address:=<UE5_HOST_IP> execution_port:=8081 \
  language_device:=cuda visual_device:=cuda policy_device:=cuda \
  task_text:="跟随红色目标船，保持3米距离"
```

### 本章问题与答案

**问：为什么模型不进 Git？**  
答：权重大、许可证和更新周期不同，外置部署更清晰。

**问：清单里有哈希是否证明本地模型正确？**  
答：否，必须计算实际文件哈希并比较。

**问：为什么不同时启动所有 CUDA 节点？**  
答：同时加载会产生更高内存峰值，可能在 Jetson 上 OOM。

---

## 第 12 章：测试和调试

### 12.1 测试地图

- 感知：[test_image_entity_perception.py](./src/vla/test/test_image_entity_perception.py#L1)、[test_image_entity_perception_node.py](./src/vla/test/test_image_entity_perception_node.py#L1)
- 语言：[test_language_encoder.py](./src/vla/test/test_language_encoder.py#L1)、[test_language_qwen_node.py](./src/vla/test/test_language_qwen_node.py#L1)
- 跟踪/张量：[test_temporal_entity_tracker.py](./src/vla/test/test_temporal_entity_tracker.py#L1)、[test_entity_features.py](./src/vla/test/test_entity_features.py#L1)
- 策略：[test_policy_model.py](./src/vla/test/test_policy_model.py#L1)、[test_vla_policy_sync.py](./src/vla/test/test_vla_policy_sync.py#L1)、[test_vla_policy_smoothing.py](./src/vla/test/test_vla_policy_smoothing.py#L1)
- 安全/控制：[test_safety_gate.py](./src/vla/test/test_safety_gate.py#L1)、[test_visual_standoff_guard.py](./src/vla/test/test_visual_standoff_guard.py#L1)
- 接口/运行时：[test_entity_interface_contract.py](./src/vla/test/test_entity_interface_contract.py#L1)、[test_runtime_identity_contract.py](./src/vla/test/test_runtime_identity_contract.py#L1)

### 12.2 PC 检查

```powershell
$env:PYTHONPATH=(Resolve-Path 'src/vla').Path
python -m pytest -q src/vla/test
python -m compileall -q src/vla/vla src/vla/test src/bringup/launch
```

PC 测试不能证明 Jetson CUDA、C++ bridge、UE5 TCP 和同次闭环。

### 12.3 逐 topic 调试

```bash
ros2 topic echo /ue/camera_frame --once
ros2 topic echo /vla/language_embedding --once
ros2 topic echo /vla/perceived_entities --once
ros2 topic echo /vla/tracked_entities --once
ros2 topic echo /vla/entity_features --once
ros2 topic echo /vla/policy_displacement --once
ros2 topic echo /control/desired_displacement --once
ros2 topic info /control/desired_displacement
```

每一层检查 `valid/reason` 和 `run_id/scene_seed/frame_index`。选择一帧 `N`，确认 bridge 的执行 JSON 使用相同的 `frame_index=N`，这才是完整数据链证据。

### 12.4 主动实验

1. 改为停止任务，预测下游输出。
2. 构造旧时间戳，验证超时停止。
3. 改 frame index，验证错帧拒绝。
4. 移除目标，验证 FOLLOW fail-closed。
5. 注入 NaN，验证安全门拒绝。
6. 请求不可用 CUDA 设备，确认不回退 CPU。

### 本章问题与答案

**问：`148 passed` 能证明真实闭环吗？**  
答：不能。它只证明 PC 上的源码合同。

**问：topic 有消息为什么还不够？**  
答：消息可能 `valid=false`、过期或身份错配。

**问：如何证明动作来自某帧图像？**  
答：沿消息链检查相同身份，直到 bridge 的最终执行 JSON。

---

## 附录：逐文件索引

### 根目录和构建文件

| 文件 | 作用 |
|---|---|
| [README.md](./README.md) | 部署入口 |
| [models/manifest.yaml](./models/manifest.yaml) | 模型合同 |
| [.gitignore](./.gitignore) | 排除产物和权重 |
| [vla/setup.py](./src/vla/setup.py) | Python 节点入口 |
| [vla/setup.cfg](./src/vla/setup.cfg) | 安装路径 |
| [vla/package.xml](./src/vla/package.xml) | Python ROS 依赖 |
| [interfaces/CMakeLists.txt](./src/interfaces/CMakeLists.txt) | 消息生成 |
| [bridge/CMakeLists.txt](./src/bridge/CMakeLists.txt) | C++ bridge 构建 |
| [bringup/CMakeLists.txt](./src/bringup/CMakeLists.txt) | Launch 安装 |

### 在线代码

| 文件 | 作用 |
|---|---|
| [task_instruction_node.py](./src/vla/vla/task_instruction_node.py) | 发布任务 |
| [language_encoder.py](./src/vla/vla/language_encoder.py) | Qwen 编码算法 |
| [language_qwen_node.py](./src/vla/vla/language_qwen_node.py) | Qwen ROS 节点 |
| [visual_encoder.py](./src/vla/vla/visual_encoder.py) | 图像工具 |
| [image_entity_perception.py](./src/vla/vla/image_entity_perception.py) | 感知算法 |
| [image_entity_perception_node.py](./src/vla/vla/image_entity_perception_node.py) | 感知节点 |
| [temporal_entity_tracker.py](./src/vla/vla/temporal_entity_tracker.py) | 跟踪算法和 ROS 跟踪节点 |
| [entity_features.py](./src/vla/vla/entity_features.py) | 实体特征和 ROS 特征节点 |
| [policy_model.py](./src/vla/vla/policy_model.py) | 策略网络 |
| [vla_policy_node.py](./src/vla/vla/vla_policy_node.py) | 在线策略 |
| [visual_standoff_guard.py](./src/vla/vla/visual_standoff_guard.py) | 跟随距离守卫 |
| [safety_gate.py](./src/vla/vla/safety_gate.py) | 安全算法和 ROS 安全节点 |
| [trajectory_contract.py](./src/vla/vla/trajectory_contract.py) | 动作合同 |
| [ue_object_deliverer_bridge_node.cpp](./src/bridge/src/ue_object_deliverer_bridge_node.cpp) | UE5/ROS bridge |
| [vla_closed_loop.launch.py](./src/bringup/launch/vla_closed_loop.launch.py) | 系统编排 |

### 消息文件

| 文件 | 作用 |
|---|---|
| [CameraFrame.msg](./src/interfaces/msg/CameraFrame.msg) | JPEG 相机帧和帧身份 |
| [ASVState.msg](./src/interfaces/msg/ASVState.msg) | ASV 位姿和运动状态 |
| [Entity.msg](./src/interfaces/msg/Entity.msg) | 单个实体几何、语义和观测有效性 |
| [EntityArray.msg](./src/interfaces/msg/EntityArray.msg) | 一帧实体集合及其来源 |
| [TaskEmbedding.msg](./src/interfaces/msg/TaskEmbedding.msg) | 语言 embedding |
| [EntityFeatures.msg](./src/interfaces/msg/EntityFeatures.msg) | 固定形状实体 tensor |
| [DesiredDisplacement.msg](./src/interfaces/msg/DesiredDisplacement.msg) | 策略、安全门和执行适配器共用的位移命令 |

### 测试文件

| 文件 | 主要验证内容 |
|---|---|
| [test_entity_interface_contract.py](./src/vla/test/test_entity_interface_contract.py) | 实体消息字段和来源合同 |
| [test_image_entity_perception.py](./src/vla/test/test_image_entity_perception.py) | 图像特征、模型加载和预测 |
| [test_image_entity_perception_node.py](./src/vla/test/test_image_entity_perception_node.py) | 感知节点输入输出和失败路径 |
| [test_language_encoder.py](./src/vla/test/test_language_encoder.py) | Qwen 编码、缓存和异常 |
| [test_language_qwen_node.py](./src/vla/test/test_language_qwen_node.py) | 语言 ROS 节点合同 |
| [test_policy_model.py](./src/vla/test/test_policy_model.py) | 策略 shape、mask、停止头和加载 |
| [test_runtime_identity_contract.py](./src/vla/test/test_runtime_identity_contract.py) | Launch、模型、身份和特权边界 |
| [test_safety_gate.py](./src/vla/test/test_safety_gate.py) | 安全拒绝码和碰撞检查 |
| [test_entity_features.py](./src/vla/test/test_entity_features.py) | 16 维实体特征、排序和 mask |
| [test_task_instruction_node.py](./src/vla/test/test_task_instruction_node.py) | 任务文本校验和发布 |
| [test_temporal_entity_tracker.py](./src/vla/test/test_temporal_entity_tracker.py) | 速度估计、重置和轨迹过期 |
| [test_trajectory_contract.py](./src/vla/test/test_trajectory_contract.py) | 动作常量和 safe-stop 合同 |
| [test_visual_standoff_guard.py](./src/vla/test/test_visual_standoff_guard.py) | 跟随距离和后备动作 |
| [test_vla_policy_smoothing.py](./src/vla/test/test_vla_policy_smoothing.py) | 动作历史、变化率和平滑 |
| [test_vla_policy_sync.py](./src/vla/test/test_vla_policy_sync.py) | 异步消息帧同步和错帧拒绝 |

### 元数据和许可证

| 文件 | 作用 |
|---|---|
| [根 LICENSE](./LICENSE) | 仓库 Apache-2.0 许可证 |
| [interfaces LICENSE](./src/interfaces/LICENSE) | 接口包许可证副本 |
| [bridge LICENSE](./src/bridge/LICENSE) | bridge 包许可证副本 |
| [bringup LICENSE](./src/bringup/LICENSE) | bringup 包许可证副本 |
| [interfaces package.xml](./src/interfaces/package.xml) | 接口依赖元数据 |
| [bridge package.xml](./src/bridge/package.xml) | bridge 依赖元数据 |
| [bringup package.xml](./src/bringup/package.xml) | bringup 包元数据 |
| [ue_bridge.yaml](./src/bridge/config/ue_bridge.yaml) | 通信、单位、坐标和端口参数 |
| [requirements-language.txt](./src/vla/requirements-language.txt) | 语言模型额外依赖 |
| [resource/vla](./src/vla/resource/vla) | ament Python 包索引标记 |
| [__init__.py](./src/vla/vla/__init__.py) | Python 包标记 |

## 学习完成标准

你应当能够：

1. 不看 launch 画出节点和 topic。
2. 解释实体张量 16 个特征。
3. 从 `CameraFrame` 追踪到 `DesiredDisplacement`，再检查 bridge 的 UE5 执行 JSON。
4. 解释语言条件注意力和上一动作有效位。
5. 指出每层 fail-closed 位置。
6. 区分 PC 测试、Jetson CUDA、UE5 联调和同次闭环证据。

做到这些，才意味着你已经从“拥有 AI 生成的代码”转变为“能够对系统设计和实现负责的工程师”。
