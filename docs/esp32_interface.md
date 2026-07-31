# ESP32 固件接口规范

本文档基于 ESP32 固件实际代码（`asv-esp32-firmware`，独立仓库）与 Jetson
`asv_control_manager` 链，定义仿真/硬件通道的接口契约。**本项目的整理与
重训不修改此链**（保护清单，见 ARCHITECTURE.md §5）。

## 1. 硬件拓扑

```
Jetson (ROS 2 Humble)                        ESP32-P4 (asv-esp32-firmware)
┌─────────────────────────────┐   UART2      ┌──────────────────────────────┐
│ /decision/output            │  (/dev/ttyUSB0│ /esp32_node (micro-ROS)     │
│   → control_input_mux ──────┼── micro-ROS ─┼→ 订阅 /control/control_input │
│   → /control/control_input  │              │  控制律: 预设时间收敛观测器   │
│                             │              │          + 预设时间导引律     │
│ safety_supervisor ←─────────┼── micro-ROS ─┼─ 发布 /control/asv_wrench    │
│   (订阅 /control/asv_wrench)│              │         + /control/debug     │
│   → /control/safe_wrench    │              └──────────────────────────────┘
│   → thruster_allocator      │
│   → /ue/thruster_command    │
│   → ue_object_deliverer_bridge_node (thruster 模式) → TCP → UE5
└─────────────────────────────┘
```

## 2. 消息契约

### 2.1 `/control/control_input`（Jetson → ESP32，订阅）

类型：`asv_interfaces/msg/ControlInput`

| 字段 | 类型 | 说明 |
|---|---|---|
| desired_x | float64 | 期望纵向位移 (m) |
| desired_y | float64 | 期望横向位移 (m) |
| surge_velocity | float64 | 期望前进速度 (m/s) |
| yaw_rate | float64 | 期望转艏角速度 (rad/s) |
| valid | bool | 有效标志（false = fail-closed 停车） |

来源：`control_input_mux_node`（订阅 `/decision/output` + `/ue/asv_state`）。

### 2.2 `/control/asv_wrench`（ESP32 → Jetson，发布）

类型：`asv_interfaces/msg/ASVWrench`

| 字段 | 类型 | 说明 |
|---|---|---|
| force | float64 | 期望合力 (N) |
| moment | float64 | 期望力矩 (N·m) |
| valid | bool | 有效标志 |

消费方：`safety_supervisor_node`（硬件闭环：ESP32 实测控制量进入安全监督）。
仿真替身：`fake_esp32_wrench_node`（无硬件时发布相同的消息）。

### 2.3 `/control/debug`（ESP32 → Jetson，发布）

调试数据通道（观测器/导引律内部状态），供监控与标定，不参与安全链。

### 2.4 参数下发（Jetson → ESP32）

`esp32_param_manager_node` 通过 `/esp32_node/set_parameters` 服务下发：
`v_max`（速度上限，默认 0.5）、`e_max`（位置误差上限，默认 0.2）、
`max_force`（推力上限，默认 0.5）等。参数变更即时生效，无需重刷固件。

## 3. 全链数据流（含仿真对照）

```
/decision/output (VLA/专家安全门后)
  → control_input_mux → /control/control_input
  → [ESP32 固件 | fake_esp32_wrench] → /control/asv_wrench
  → safety_supervisor → /control/safe_wrench
  → thruster_allocator → /ue/thruster_command
  → ue_object_deliverer_bridge_node (outbound_command_mode=thruster)
  → TCP → UE5
```

- `full_system.launch.py`：完整硬件链（micro_ros_agent serial + 全部 control
  manager 节点），保持原样。
- `vla_closed_loop.launch.py`：仿真路径（decision_setpoint_adapter →
  /ue/kinematic_setpoint），不启动 control manager——`UEKinematicSetpoint`
  消息头注明"永不发送给 ESP32"，仿真/硬件通道刻意分离。
- 无硬件时：`fake_esp32_wrench` 替代 ESP32 发布 wrench，链可完整烟测。

## 4. 接入清单（后续接真实 ESP32）

1. Jetson 侧启动 `full_system.launch.py`（或 hardware_loop launch，见下）
2. 确认 `/dev/ttyUSB0` 存在且用户组为 dialout；micro_ros_agent 以
   `serial --dev /dev/ttyUSB0` 启动（固件 UART2，见 `main.cpp` 的
   `CONFIG_MICROROS_UART_TXD/RXD`）
3. 固件烧录后 `/esp32_node` 上线，订阅 `/control/control_input`、
   发布 `/control/asv_wrench`
4. 用 `esp32_param_manager` 下发参数（v_max/e_max/max_force）
5. 闭环控制源切换：`hardware_loop.launch.py`（VLA + control manager 链，
   以 launch 参数切换 sim/hardware 通道）

## 5. 保护声明

本项目任何改动不得修改：话题名（`/control/control_input`、
`/control/asv_wrench`、`/control/safe_wrench`、`/ue/thruster_command`）、
`asv_control_manager` 5 个节点的功能语义、`full_system.launch.py`、
`fake_esp32_wrench_node`、bridge 的 thruster 模式。
