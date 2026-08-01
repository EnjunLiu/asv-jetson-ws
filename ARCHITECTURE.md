# ASV VLA 系统架构

无人船视觉-语言-动作（VLA）流水线：UE5 仿真 → 多模态特征 → 策略 → 确定性安全门 →
运动控制 → 仿真/硬件执行。故障时 fail-closed（宁可停止，不可执行不安全轨迹）。

本工程的动作抽象是**任务级二维期望轨迹**：策略只输出船体坐标系中的累计位移
序列，不输出左右推力，也不承担流体动力学建模。这样保留了“轨迹足以表达任务”
的研究主线，同时把底层控制器、推力分配和 ESP32 保持为可替换的执行层。当前在线
策略发布 20 个 waypoint（`dt=0.2 s`，4 s 视界），控制器只取安全轨迹的短前缀，
运动学 adapter 每个观测帧只向 UE5 发送一个 setpoint。

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
   /ue/camera_frame ──► image_perception ─► tracker ─► visual_encoder ──────────┤
              │                                      (MobileNetV3, 17-token)     │
              ├──────────────────────────────────► task_entity_tensor ──────────┤
              │                                      (16×16 实体几何张量)        │
              │                      language_backend ────► /vla/language_embedding
              │                   (Qwen CUDA 首指令分阶段；stub 仅测试)           │
              ▼                                          ▼
   vla_policy (Torch, CUDA) ──► /vla/policy_trajectory expert_policy_bridge ──┘
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

`image_entity_perception` 的输入只有 JPEG 和 `/task/text`。它先从图像产生候选实体，
再用当前指令选择任务相关实体并写入 `instruction_id`；未知/空指令 fail-closed。
`/ue/entities` 的真值实体只允许进入 recorder、离线标签和验证脚本，不能进入这条在线
链。单帧图像不填写速度，`temporal_entity_tracker` 依据相邻帧的身份字段和时间戳计算
速度；`/ue/asv_state` 是唯一 ego 来源。

| 话题 | 类型 | 发布者 | 说明 |
|---|---|---|---|
| /ue/camera_frame | CameraFrame | bridge | 1280x720 JPEG，FOV 90° 水平 |
| /ue/entities | UEEntityArray | bridge | ≤64 实体，base_link 相对位置/速度 |
| /ue/asv_state | UEASVState | bridge | 本船位姿/速度（在线 ego 来源） |
| /vla/visual_features | VisualFeatures | visual_encoder | token_count=17, dim=576 |
| /vla/task_features | TaskFeatures | task_entity_tensor | 16×16, 排序 targets→risks→normal |
| /vla/language_embedding | TaskEmbedding | language_qwen / language_stub | 首次指令真实 Qwen CUDA；成功后释放权重，embedding 继续在线；stub 仅测试 |
| /vla/policy_trajectory | SelectedTrajectory | vla_policy / expert_policy_bridge | 20×2 位移, dt=0.2s |
| /vla/selected_trajectory | SelectedTrajectory | safety_gate（唯一） | 门后轨迹 |
| /decision/output | DecisionOutput | trajectory_controller | desired_x/y + valid + source Run/Scene/Frame/Model |
| /ue/kinematic_setpoint | UEKinematicSetpoint | decision_setpoint_adapter | 单步位移 + source 元数据；**永不发 ESP32** |

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
  → 保留带 `model_config` 的 `pytorch_state_dict` checkpoint
  → 部署 Jetson `~/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt`
     并由 JetPack PyTorch CUDA strict-load
```

指令集：`dataset/language/instructions.jsonl`（90 条，含 follow red/blue 3m/10m 等），
训练缓存仍由 PC 端冻结 Qwen 编码；为适配 Jetson Orin Nano 8 GB，最终在线演示默认先由
Jetson 上 `Qwen3-Embedding-0.6B` 以 CUDA 编码首次任务文本，成功后释放 Qwen 权重，
256-D embedding 继续在线发布。任务切换需要重启当前闭环，不在当前 S2 演示承诺内；stub
仅保留为确定性 smoke/无模型环境的可选后端。`vla_closed_loop.launch.py` 默认
`language_release_after_encode=true`、`language_staging_delay_sec=20.0`。

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

## 6. 任务级动作与执行边界

| 层 | 输入 | 输出 | 责任边界 |
|---|---|---|---|
| VLA 策略 | 视觉 token、实体几何、语言 embedding | 20×2 累计位移（body frame）+ STOP | 选择任务相关的期望运动，不接触推力/动力学 |
| 安全门 | 策略或专家轨迹 | 唯一的 selected trajectory | 约束速度、曲率、碰撞、陈旧和 E-STOP；拒绝即停止 |
| 轨迹控制器 | 安全轨迹 | `DecisionOutput` 的短前缀 | 将轨迹变成单步期望位移，不实现物理控制 |
| UE5 运动学 adapter | 单步期望位移 + Run 元数据 | 一个 `UEKinematicSetpoint` | 仅仿真 teleport；永不进入 ESP32 推力链 |
| 物理控制层 | `DecisionOutput`/控制输入 | 推力或安全 wrench | 后续可替换的底层控制器，独立调参 |

当前实现不包含 world model、候选轨迹枚举或候选评分器；这些概念不应重新写回
运行时消息契约。未来若增加规划器，应作为安全门之前的可选任务层，并保持上述
“轨迹输出 / 执行层”边界不变。

## 7. 安全设计

- `/vla/selected_trajectory` 唯一发布者为安全门；门后所有节点只透传。
- 硬约束：单步速度 ≤1.5 m/s（容差 5%）、总位移 ≤10m、曲率 ≤15 rad/m、方向跳变
  ≤170°、碰撞余量 0.5m（只查可执行前缀 2-5 步）、STALE 1s、E-STOP 2s。
- 拒绝时输出确定性减速轨迹或全零 E-STOP；模态缺失 fail-closed。
- 学习策略与专家共享同一门（专家路径不绕过）。

## 8. 已知边界（诚实声明）

- 当前唯一可录制演示是 L7/S2 近距离 image/color 在线链；最终 seed=230908
  已在全 CUDA 栈实际驱动 UE5 ASV（230906/230902 为对照），详见 [TODO.md](TODO.md) 和
  [docs/demo_runbook.md](docs/demo_runbook.md)。这不是所有距离、布局和随机种子都
  已泛化的结论。
- L7/L7B 的白色干扰船约 7 m，超出当前图像校准域；目标不可见或证据非有限时必须
  `valid=false`/hold，不能用 UE `/ue/entities` 真值补齐，不能放宽安全门。
- 在线图像感知矩阵、MobileNet、Qwen 和策略均请求 CUDA；Qwen 默认只在首次指令编码
  期间短暂驻留，成功后释放权重而保留 embedding 在线发布。任一 CUDA/模型加载失败都
  保持 fail-closed，不静默降级到 CPU。时间跟踪、张量工程、安全门和控制器是确定性
  CPU 代码，不属于神经网络推理。
- UE5 仿真结果不等同真实感知或实船海试；仿真/硬件通道刻意分离
  （kinematic_setpoint 消息头注明“永不发 ESP32”）。专家路径仍可选，但不与最终在线
  策略并行发布。

## 9. 阶段状态（2026-08-02 快照）

| 阶段 | 状态 | 证据 |
|---|---|---|
| 终极整理 | 完成 | `cleanup/ultimate-restructure` 工作树（过期 near/ONNX/checkpoint 已移入 `/tmp/asv_vla_cleanup_20260802`；HISTORY.md 保留审计） |
| 正弦编队场景 | 完成 | S2 运动 + L7/L7B 近距离布局 + YawFixWholeRun；headless 验证（docs/scene_verification.md） |
| 近距离 image-only 数据 | 完成 | 12 runs / 1200 frames / 104720 samples；cache 不含 `/ue/entities` 真值 |
| 感知校准 | **在线可用** | `image_entity_color_calibrated_v1.npz`；7m OOD 按设计 fail-closed |
| near image/color 策略 | **训练门 PASS；CUDA 在线 PASS** | seed42 checkpoint；Jetson Torch CUDA strict-load；S2 连续有效 setpoint |
| 在线语言 | **CUDA 首指令分阶段 PASS** | 首次指令真实 Qwen CUDA 编码后释放权重，embedding 继续在线；任务切换需重启且不在当前 S2 承诺；无静默 CPU 降级 |
| 在线闭环（图形 `-game`） | **S2 近距离通过** | L7/S2 seed=230908；JPEG 感知、Qwen、guard、`SCENE_EXEC_APPLY` 连续链路，35 s `SCENE_UE_COMPLETE` |
| ESP32 扩展 | 完成 | hardware_loop.launch.py + 链烟测（fail-closed 验证）+ docs/esp32_interface.md |

**诚实记录**：上述两次 S2 近距离运行是当前可录制证据；不要将其扩大成统计鲁棒性
结论，后续 run 继续填写 [docs/demo_runbook.md](docs/demo_runbook.md) 的记录表。
