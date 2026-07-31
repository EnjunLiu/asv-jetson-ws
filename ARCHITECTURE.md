# ASV VLA 系统架构

无人船视觉-语言-动作（VLA）流水线：UE5 仿真 → 多模态特征 → 策略 → 确定性安全门 →
运动控制 → 仿真/硬件执行。故障时 fail-closed（宁可停止，不可执行不安全轨迹）。

## 1. 端到端数据流

```
UE5 (Main_Map, 无限海洋)                    Jetson (ROS 2 Humble)
┌───────────────────────┐                 ┌──────────────────────────────────┐
│ Connection 蓝图       │  TCP :8080      │ ue_object_deliverer_bridge_node  │
│  - SceneCapture 相机   │ ──────────────► │  - /ue/camera_frame (JPEG)       │
│  - 实体投影/可见性     │                 │  - /ue/entities (base_link)      │
│  - 运动学执行 (teleport)│                 │  - /ue/asv_state                 │
└───────────────────────┘                 └──────────────┬───────────────────┘
                                                         │
              ┌──────────────────────────────────────────┼──────────────────────┐
              ▼                                          ▼                      ▼
   /ue/camera_frame + /ue/entities ──► visual_encoder ──► /vla/visual_features  │
              │                          (MobileNetV3, 17-token 槽位布局)        │
              ├─────────────────────► task_entity_tensor ─► /vla/task_features  │
              │                          (16×16 实体几何张量)                    │
              │                      language_stub ───────► /vla/language_embedding
              │                          (预计算 Qwen embedding, 可切换指令)     │
              ▼                                          ▼
   vla_policy (ONNX, CPU) ──► /vla/policy_trajectory    expert_policy_bridge ──┘
              │                  (学习策略, 5 帧平滑)         (专家对照, 规格允许)
              ▼
   safety_gate ──► /vla/selected_trajectory   （唯一发布者；曲率/方向/速度/碰撞/
              │                               总位移/陈旧/E-STOP 硬约束）
              ▼
   trajectory_controller ──► /decision/output  （只执行轨迹前缀 2 waypoint）
              │
              ├──(仿真)──► decision_setpoint_adapter ──► /ue/kinematic_setpoint ──► UE5
              │
              └──(硬件链, 见 §5)──► control_input_mux ──► /control/control_input
                                   ──► [ESP32] ──► /control/asv_wrench ──► ...
```

## 2. 关键接口（话题契约）

| 话题 | 类型 | 发布者 | 说明 |
|---|---|---|---|
| /ue/camera_frame | CameraFrame | bridge | 1280x720 JPEG，FOV 90° 水平 |
| /ue/entities | UEEntityArray | bridge | ≤64 实体，base_link 相对位置/速度 |
| /ue/asv_state | UEASVState | bridge | 本船位姿/速度（在线 ego 来源） |
| /vla/visual_features | VisualFeatures | visual_encoder | token_count=17, dim=576 |
| /vla/task_features | TaskFeatures | task_entity_tensor | 16×16, 排序 targets→risks→normal |
| /vla/language_embedding | (float array) | language_stub | 预计算 Qwen embedding |
| /vla/policy_trajectory | SelectedTrajectory | vla_policy / expert_policy_bridge | 20×2 位移, dt=0.2s |
| /vla/selected_trajectory | SelectedTrajectory | safety_gate（唯一） | 门后轨迹 |
| /decision/output | DecisionOutput | trajectory_controller | desired_x/y + valid |
| /ue/kinematic_setpoint | UEKinematicSetpoint | decision_setpoint_adapter | **永不发 ESP32** |

实体几何张量 16 列：x/y/z(±20m/5m) · vx/vy/vz(±5m/s) · 平面距离 · bearing_sin/cos ·
closing_speed · time_to_cpa · cpa_distance · is_target · is_risk · color_red ·
color_blue（**颜色列为特权列，训练与在线均置零**——颜色只经视觉 crop 表达）。

## 3. 包结构（src/）

| 包 | 语言 | 内容 |
|---|---|---|
| asv_ue_bridge | C++ | UE5 TCP bridge（thruster/kinematic 双模式） |
| asv_vla | Python | 视觉/任务/语言编码、策略推理、安全门、控制桥、专家、记录/回放 |
| asv_control_manager | C++ | 硬件链：control_input_mux / safety_supervisor / thruster_allocator / esp32_param_manager / system_monitor |
| asv_planning | C++ | decision_node / state_predictor_node |
| asv_perception | C++ | perception_node |
| asv_interfaces | msg | 硬件链消息 |
| asv_jetson_interfaces | msg | UE5/VLA 消息 |
| asv_bringup | launch | 全部启动文件 |
| asv_tools | C++ | fake_esp32_wrench（无硬件时 ESP32 仿真替身）/ fake_ue_client |

## 4. 训练管线（PC）

```
UE5 采集 (collect.ps1, 自动化) → episode 包 (tar.gz, SHA-256 校验)
  → registry 注册 + group split (train/val/test)
  → build_feature_caches.py：冻结 MobileNetV3-small (576-d) + Qwen3-Embedding-0.6B (256-d)
     产出 features_<sha>/<run_id>/frames_000.npz + language.npz
  → 标签：expert_trajectory.py（FOLLOW/STOP，相机前守卫 x<=0 → STOP）
  → 训练：SmallTrajectoryPolicy (481K 参数, MLP 融合)，3 seeds
  → 验证门：stop_drift + ADE/FDE vs expert 基线 + finite/speed
  → export_onnx.py：ONNX CPU 导出 + parity 校验（max_diff<1e-4）
  → 部署 Jetson ~/jetson_asv_ws/models/policy.onnx
```

指令集：`dataset/language/instructions.jsonl`（90 条，含 follow red/blue 3m/10m 等），
语言 embedding 由 Qwen 在 PC 端冻结编码，行序=文件行序。在线语言使用预计算 embedding
（无在线 Qwen）。

## 5. ESP32 硬件链（扩展路径）

```
/decision/output → control_input_mux → /control/control_input
  → [ESP32 固件 /esp32_node：订阅 /control/control_input，控制律输出]
  → /control/asv_wrench → safety_supervisor → /control/safe_wrench
  → thruster_allocator → /ue/thruster_command → ue_object_deliverer_bridge (thruster 模式) → UE5
```

- ESP32 固件：`asv-esp32-firmware`（独立仓库，ESP32-P4 + micro-ROS over UART2，
  Jetson `/dev/ttyUSB0`）；发布 `/control/asv_wrench` + `/control/debug`。
- `full_system.launch.py` 包含完整硬件链（micro_ros_agent + control manager）。
- 仿真时 `fake_esp32_wrench` 替代 ESP32 发布 wrench；`vla_closed_loop.launch.py`
  使用 kinematic 路径（decision_setpoint_adapter），不启动硬件链。
- 接口规范细节见 `docs/esp32_interface.md`。

## 6. 安全设计

- `/vla/selected_trajectory` 唯一发布者为安全门；门后所有节点只透传。
- 硬约束：单步速度 ≤1.5 m/s（容差 5%）、总位移 ≤10m、曲率 ≤15 rad/m、方向跳变
  ≤170°、碰撞余量 0.5m（只查可执行前缀 2-5 步）、STALE 1s、E-STOP 2s。
- 拒绝时输出确定性减速轨迹或全零 E-STOP；模态缺失 fail-closed。
- 学习策略与专家共享同一门（专家路径不绕过）。

## 7. 已知边界（诚实声明）

- 学习策略离线验证指标完好，但动态 FluidSim 水景下逐帧输出不稳定（见 HISTORY.md §3）；
  线上对照使用 deterministic expert（规格允许）。在线鲁棒性（输入增强重训）为 P2。
- 部署模型验证门状态见 HISTORY.md §1（`validation_gate_passed: false`）。
- 颜色 grounding 的独立留出验证曾未通过（红蓝换位），修复后的验证见 §1。
- UE5 仿真结果不等同真实感知或实船海试；仿真/硬件通道刻意分离
  （kinematic_setpoint 消息头注明"永不发 ESP32"）。

## 8. 阶段状态（2026-08-01 快照）

| 阶段 | 状态 | 证据 |
|---|---|---|
| 终极整理 | 完成 | PC 25c037f / Jetson 5877b30（去 dayX、清理 2.9 GB、HISTORY.md 保留审计） |
| 正弦编队场景 | 完成 | S2 运动 + L6/L6B 布局 + YawFixWholeRun；headless 验证（docs/scene_verification.md） |
| 新数据采集 | 完成 | 14 runs（L6/L6B × 7 seeds），100 帧/run，质量门全过 |
| 特征构建 | 完成 | 14 caches / 117500 样本，ego 置零（分布一致），frozen sha 2ea3f77c8cf7 |
| 重训 | **验证门 PASS** | ADE 改善 67-70%、FDE 69-72%、STOP recall 1.0 / drift 0 |
| 选择指标 | **96.2%** | L6 97.9% / L6B 95.3%（红蓝双向） |
| ONNX 导出 | 完成 | parity max_diff=0、cos=1.0；已部署 Jetson（旧模型备份） |
| 语言 stub | 完成 | 红/蓝指令运行时切换（参数驱动） |
| 在线闭环 | **待 PIE 实测** | headless 下 setpoint 不执行（已证实）；runbook 见 docs/demo_runbook.md |
| ESP32 扩展 | 完成 | hardware_loop.launch.py + 链烟测（fail-closed 验证）+ docs/esp32_interface.md |

**诚实记录**：模型闭环的最终验收（Play 实测 ≥8 runs）需用户在 PIE 模式
执行 runbook；在此之前不宣称"模型在线闭环通过"。
