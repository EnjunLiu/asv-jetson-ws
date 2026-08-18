# VLA 最少节点重构设计

日期：2026-08-18  
状态：已确认设计（修订版），待实施计划

## 1. 目标

在不改变 ASV 的二维任务级控制合同的前提下，重构 `src/vla`：

- 保留 `src/vla/vla` Python 包目录，不改为 `src/vla/src`。
- 将 VLA 包运行时节点收敛到工程上合理的最少数量。
- 不新增 ROS 节点、topic 或消息类型。
- 一个保留节点对应一个同名算法文件：`x_node.py` 只负责 ROS，`x.py` 负责算法。
- 节点文件的模块注释只说明订阅和发布合同。
- 删除只在内部阶段间转发数据的节点和 topic。
- 将 Jetson 图像增强删除，由 UE5 输出标准 sRGB JPEG，JPEG 质量目标为 95。

## 2. 非目标

- 不把视觉、语言和策略强行合并成一个巨型节点。
- 不改变 `DesiredDisplacement` 的二维动作合同。
- 不引入旧路径、旧节点、旧 topic 或旧入口的兼容转发层。
- 不顺手重构 bridge 的通信算法、interfaces 或 VLA 之外的无关代码；本次仅统一 bridge 的节点和 executable 名称。
- 不把 Windows 静态检查视为 Jetson 或 UE5 闭环验证。

## 3. 最终节点边界

### 3.1 VLA 包保留三个节点

#### `language`

订阅：

- `/task/text` (`std_msgs/String`)，用于外部运行时任务输入。

发布：

- `/vla/language_embedding` (`interfaces/TaskEmbedding`)。

职责：

- 从 launch 参数读取初始任务文本；没有独立的任务发布节点。
- 校验任务文本并调用 Qwen 编码器。
- 管理 Qwen CUDA 模型加载、编码、缓存和按配置释放。
- 发布唯一的任务语义消息 `TaskEmbedding`。

部署配置使用 `release_model_after_encode=true` 时，一次运行只接受首次有效任务；更换任务必须重启 launch。只有明确关闭模型释放时，才支持运行中通过 `/task/text` 更新任务。

#### `perception`

订阅：

- `/ue/camera_frame` (`interfaces/CameraFrame`)。
- `/vla/language_embedding` (`interfaces/TaskEmbedding`)。

发布：

- `/vla/entities` (`interfaces/EntityArray`)。

职责：

- 解码 UE5 JPEG。
- 执行任务条件化图像实体感知。
- 在节点内部维护跨帧实体状态。
- 计算速度、执行滤波，并进行有界短时丢失恢复。
- 发布已经完成检测、跟踪和速度估计的最终实体数组。

该节点不再订阅 `/task/text`；任务文本直接来自 `TaskEmbedding.instruction`，避免同一语义存在两个输入源。

#### `decision`

订阅：

- `/vla/language_embedding` (`interfaces/TaskEmbedding`)。
- `/vla/entities` (`interfaces/EntityArray`)。

发布：

- `/control/desired_displacement` (`interfaces/DesiredDisplacement`)。

职责：

- 同步任务语义和实体帧身份。
- 在内部构造策略需要的固定形状实体张量。
- 执行策略模型推理、目标距离保护和动作平滑。
- 在最终发布边界执行字段校验、动作限幅、变化率限制、碰撞检查和超时停车。
- 只发布经过安全判定的最终二维位移命令。

### 3.2 保留 bridge 节点

`bridge_node` 保持为独立 C++ 节点。它是 UE5 TCP/JSON 与 ROS 消息之间的传输边界，不属于 VLA 算法节点。

bridge 继续：

- 发布 UE5 相机帧和必要的仿真状态消息。
- 订阅 `/control/desired_displacement`。
- 在执行前进行消息有限值、坐标系和有效性复核。
- 将最终二维位移转换为现有 UE5 JSON 执行合同。

完整 UE5-Jetson 闭环共有四个 ROS 节点：三个 VLA 节点加一个 bridge 节点。

bridge 的重命名必须同步以下位置：

- `src/bridge/CMakeLists.txt` 的 target、依赖和 install target：`bridge_node`。
- launch 中的 executable 和 name：`bridge_node`。
- `src/bridge/config/ue_bridge.yaml` 的节点参数根键：`bridge_node`。
- C++ 构造函数的 ROS 节点名：`bridge_node`。
- 仓库内测试、文档和运行脚本中的旧名称引用。

源文件保留为 `src/bridge/src/bridge_node.cpp`，因为它已经是 bridge 的同名节点实现。

## 4. 删除的节点和中间接口

### 4.1 删除 `task_instruction`

删除：

- `task_instruction_node.py`。
- `setup.py` 中的 `task_instruction` console entry point。
- launch 中的 `task_instruction` 节点。

初始 `task_text` 由 `language` 参数接收。VLA 内部不再用一个节点周期转发固定字符串。部署时 Qwen 在首次编码后释放，因此任务按一次运行固定；需要动态任务的调试配置必须显式保留 Qwen 模型，才能从系统外部向 `/task/text` 发布新任务。

### 4.2 删除 `temporal_entity_tracker` 节点

删除：

- `TemporalEntityTrackerNode` 和 `main()`。
- `temporal_entity_tracker` console entry point。
- launch 中的 tracker 节点。
- `/vla/perceived_entities` 和 `/vla/tracked_entities`。

跟踪算法、速度有效性、滤波和丢失恢复并入 `perception.py`，由感知节点直接发布 `/vla/entities`。

### 4.3 删除 `safety_gate` 节点

删除：

- `SafetyGateNode` 和 `main()`。
- `safety_gate` console entry point。
- launch 中的 safety gate 节点。
- `/vla/policy_displacement`。

纯安全判定逻辑并入 `decision.py`，节点级超时状态和最终发布逻辑并入 `decision_node.py`。`/control/desired_displacement` 必须只有 `decision` 一个 VLA 发布者。

### 4.4 节点和算法文件重命名

节点、算法和 executable 使用同一组短名称：

| 节点 | 算法文件 | ROS executable |
|---|---|---|
| `language_node.py` | `language.py` | `language` |
| `perception_node.py` | `perception.py` | `perception` |
| `decision_node.py` | `decision.py` | `decision` |
| `bridge_node.cpp` | 节点内部算法 | `bridge_node` |

不保留 `language_qwen`、`image_entity_perception`、`vla_policy` 或 `ue_object_deliverer_bridge_node` 的兼容入口、旧 executable 或旧节点名。

## 5. 目标代码结构

```text
src/vla/vla/
  __init__.py

  language.py
  language_node.py

  perception.py
  perception_node.py

  decision.py
  decision_node.py

```

不再保留 `language_encoder.py`、`visual_encoder.py`、`policy_model.py`、`trajectory_contract.py` 和 `visual_standoff_guard.py` 这类独立算法支持模块。它们的代码全部归入对应的算法文件：

- `language_encoder.py` 的编码器、缓存和输入校验归入 `language.py`。
- `image_entity_perception.py`、`visual_encoder.py` 以及跟踪/丢失恢复算法归入 `perception.py`。
- `policy_model.py`、`trajectory_contract.py`、`visual_standoff_guard.py`、实体特征构造和安全判定归入 `decision.py`。

算法文件不提供 ROS console entry point；每个节点只通过同名的 `_node.py` 暴露一个入口。

## 6. 文件职责规则

每个 `x_node.py` 仅允许包含：

- ROS 参数声明和读取。
- publisher、subscription、timer 的创建。
- ROS 消息与纯 Python 输入/输出之间的转换。
- 对 `x.py` 公开算法接口的调用。
- 节点生命周期和节点级日志。
- `main()`。

每个 `x.py` 包含：

- 配置数据类和纯数据结构。
- 输入校验和错误类型。
- 模型加载与推理包装。
- 数值计算、跟踪、同步、安全判定和状态机。
- 不依赖 `rclpy` 的可测试算法接口。

节点模块的顶部注释只列出订阅和发布，不描述算法实现历史。

## 7. 消息和 topic 合同

最终 VLA 在线主链路为：

```text
/task/text
  -> language
  -> /vla/language_embedding

/ue/camera_frame + /vla/language_embedding
  -> perception
  -> /vla/entities

/vla/language_embedding + /vla/entities
  -> decision
  -> /control/desired_displacement
  -> bridge_node
  -> UE5
```

保留的主消息：

- `CameraFrame`：UE5 JPEG 帧和运行身份。
- `TaskEmbedding`：任务文本、模型身份和 256 维 embedding。
- `EntityArray`：任务相关实体的位置、速度、可见性和有效性。
- `DesiredDisplacement`：一个控制周期的 `desired_x, desired_y`。

`EntityFeatures.msg` 当前没有独立 topic 消费者。重构时删除该消息，把固定 `(16, 16)` 特征张量改为 `decision.py` 内部数据结构。删除前必须通过仓库残留搜索和 Jetson 运行图确认本项目闭环没有消费者；本设计不保留兼容消息。

## 8. 图像合同

### UE5

- 输出标准 sRGB JPEG。
- JPEG 质量目标固定为 95。
- 删除人为 gamma、brightness 和 contrast 增强。
- 保留图像尺寸、通道统计和可选 JPEG dump 诊断能力。
- 检查 Blueprint 是否保存了显式 JPEG quality pin；不能只依赖 C++ 默认参数。

### Jetson

- 删除 `enhance_low_light_image()`。
- 删除 `image_preprocess_enabled`、gamma、brightness 和 contrast 参数。
- 图像节点直接解码标准 JPEG，并将解码结果送入感知模型。
- 保留无效 JPEG、空数据和模型输入错误的 fail-closed 行为。

## 9. 安全与错误处理

- 任一任务 embedding、实体数组、帧身份或模型输出无效时，策略发布无效的零位移停车消息。
- 非有限值、错误坐标系、错误动作维度、超限位移和碰撞风险均被拒绝。
- 策略流超时后，`decision_node` 发布一次状态变化对应的停车消息，避免高频重复日志。
- bridge 保留最终传输边界检查，但不重复 VLA 的完整碰撞算法。
- 跟踪器在第一帧或时间戳不递增时将 `velocity_valid` 置为 `false`，不伪造速度。
- 场景身份变化时清空跟踪、同步、动作历史和安全状态。

## 10. 分阶段迁移

### 阶段一：结构合同测试

- 添加目标文件、console entry point、launch 节点和 topic 数量的合同测试。
- 测试先在旧结构上失败，证明测试确实约束目标结构。

### 阶段二：合并任务输入

- 将初始任务参数移入 `language_node.py`。
- 删除 `task_instruction` 节点和入口。
- 验证有效和空任务的 fail-closed 行为。

### 阶段三：合并跟踪

- 将跟踪和丢失恢复迁移到 `perception.py`。
- 图像节点直接发布 `/vla/entities`。
- 删除 tracker 节点、入口和两个旧实体 topic。
- 保留并迁移现有跟踪算法测试。

### 阶段四：合并安全边界

- 将安全算法迁移到 `decision.py`。
- 策略节点直接发布 `/control/desired_displacement`。
- 删除 safety gate 节点、入口和策略中间 topic。
- 删除策略对最终控制 topic 的反馈订阅，内部记录最后一个已接受动作。

### 阶段五：统一命名、算法文件和薄节点

- 将节点和 executable 重命名为 `language`、`perception`、`decision`。
- 将语言、感知、策略模型及其所有支持逻辑分别收拢到三个 `x.py` / `x_node.py` 对。
- 删除节点文件中的算法实现和冗余长注释。
- 删除不再使用的导入、变量和内部消息转换。

### 阶段六：UE5 图像修复

- 修改 UE5 编码和质量设置。
- 删除 Jetson 图像增强和 launch 参数。
- 用同一场景保存 UE5 原始 JPEG，检查亮度、颜色和压缩质量。

## 11. 验证层级

### 静态验证

- Python `compileall`。
- console entry point、launch、topic 和旧名称残留搜索。
- 节点文件不得包含核心算法实现；算法文件之间不得再通过辅助算法模块间接拆分。
- `language_encoder.py`、`visual_encoder.py`、`policy_model.py`、`trajectory_contract.py` 和 `visual_standoff_guard.py` 不得作为独立运行时 Python 文件残留。
- 仓库内不得存在旧节点入口或兼容转发文件。

### 算法测试

- 语言输入校验和 embedding 合同。
- 图像解码、实体检测、速度估计、滤波和丢失恢复。
- 策略输入同步、特征张量、动作平滑和安全拒绝。
- 超时、NaN/Inf、身份错配、速度无效和碰撞风险。

### Jetson ROS 2 验证

- `colcon build` 和完整测试通过。
- `ros2 node list` 中 VLA 节点恰好为 `language`、`perception`、`decision` 三个。
- bridge 节点名称和 executable 均为 `bridge_node`。
- `ros2 topic info -v` 证明每个最终 topic 只有预期发布者和订阅者。
- CUDA 日志证明 Qwen、视觉模型和策略模型按预期设备运行。

### UE5 闭环验证

- UE5 实例真实发布 JPEG。
- Jetson 产生同一次运行身份下的 `TaskEmbedding`、`EntityArray` 和 `DesiredDisplacement`。
- bridge 将最终命令发送给 UE5。
- UE5 ASV 产生与命令方向一致的实际位移。
- 保存同一次运行的 UE5、Jetson 和 bridge 身份字段及关键日志。

## 12. 成功标准

- `src/vla/vla` 路径保持不变。
- VLA 包只有 `language`、`perception`、`decision` 三个 ROS console entry point 和三个运行时节点。
- 仓库中不存在旧的 `language_qwen`、`image_entity_perception`、`vla_policy` 或 `ue_object_deliverer_bridge_node` 运行入口。
- 完整闭环只有三个 VLA 节点加一个 bridge 节点。
- 不存在 `task_instruction`、`temporal_entity_tracker` 或 `safety_gate` 节点。
- 不存在 `/vla/perceived_entities`、`/vla/tracked_entities` 或 `/vla/policy_displacement`。
- 三个节点分别发布 `TaskEmbedding`、`EntityArray` 和 `DesiredDisplacement`。
- 节点文件是薄 ROS 外壳，算法文件可独立测试。
- Jetson 不再执行图像亮度、gamma 或 contrast 增强。
- UE5 输出经真实文件检查的标准 sRGB JPEG，质量设置为 95。
- Jetson 构建和测试通过，并完成 UE5 参与的同次运行闭环验证。
