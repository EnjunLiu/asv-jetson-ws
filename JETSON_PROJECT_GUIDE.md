# Jetson 项目说明

## 1. 最终运行结构

Jetson 端只暴露四个 ROS 2 节点：

```text
UE5 TCP
  -> bridge_node
  -> /ue/camera_frame + /ue/asv_state
  -> perception.py / perception
  -> /vla/entities
  -> decision.py / decision
  -> /control/desired_displacement
  -> bridge_node
  -> UE5 执行器

/task/text -> language.py / language -> /vla/language_embedding
```

节点与算法一一对应：

| 节点文件 | 算法文件 | 作用 |
| --- | --- | --- |
| `src/vla/vla/language_node.py` | `src/vla/vla/language.py` | Qwen 文本 embedding |
| `src/vla/vla/perception_node.py` | `src/vla/vla/perception.py` | JPEG 感知、任务筛选和内部速度跟踪 |
| `src/vla/vla/decision_node.py` | `src/vla/vla/decision.py` | 实体特征、策略推理和安全检查 |
| `src/bridge/src/bridge_node.cpp` | bridge 内部实现 | UE5 TCP 与 ROS 消息转换 |

实体速度、CPA 风险、16x16 实体特征、动作限幅、动作平滑和 fail-closed
逻辑均在 `perception.py` 或 `decision.py` 内部实现，没有独立节点。

## 2. 消息边界

- `language`：订阅 `/task/text`，发布 `/vla/language_embedding`。
- `perception`：订阅 `/ue/camera_frame` 和 `/vla/language_embedding`，发布
  `/vla/entities`。
- `decision`：订阅 `/vla/entities`、`/vla/language_embedding` 和
  `/ue/asv_state`，发布 `/control/desired_displacement`。
- `bridge_node`：接收 UE5 TCP JSON 和最终位移命令，发布 UE5 状态、相机帧及
  实体集合，并把最终位移发送回 UE5。

在线控制不使用 `/ue/entities` 作为感知输入；它只用于采集和离线监督。

## 3. 图像链路

UE5 的 `SceneAutomationSubsystem` 将场景捕获设置为 FinalColorLDR，启用后处理
和 tone mapping，固定 `+1 EV` 曝光，并把 Render Target 设为 sRGB。JPEG 编码前
在 UE5 端完成曝光和标准 sRGB 输出。

Jetson 端直接解码 UE5 JPEG。运行时代码中没有亮度增强、gamma 修正或低光照图像
预处理。这样 Jetson 接收到的图像色彩空间与 UE5 显示端保持一致。

## 4. 构建与运行

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select interfaces bridge vla bringup
source install/setup.bash

ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=../models \
  execution_address:=<UE5_HOST_IP> execution_port:=8081 \
  task_text:="跟随红色目标船，保持3米距离"
```

模型权重不进入 Git。Jetson 上需要观察同一次运行中的：

```text
language READY ... device=cuda
perception ready ... device=cuda
POLICY_READY backend=torch_cuda
```

没有有效模型、CUDA 显存不足、输入身份不一致或消息过期时，系统保持当前位置。
`Hold_Position` 的通信结果不能当作有效策略推理证据。

## 5. 测试

```bash
PYTHONPATH=src/vla python -m pytest -q src/vla/test
```

主机测试验证纯算法、消息合同、节点入口和 fail-closed 行为。真实 CUDA 和 UE5
闭环必须在目标设备上用同一次运行日志单独确认。
