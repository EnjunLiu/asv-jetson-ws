# 交接手册（给后续 AI 助手）

> 写于 2026-08-01 中午。本手册包含让新助手接手所需的一切：项目背景、架构、
> 当前状态、根因分析、环境、命令、待办。**所有声明均为实测事实，未验证的不写。**

---

## 1. 项目是什么

无人船视觉-语言-动作（VLA）闭环项目：
- **UE5 仿真**（Windows）：海洋场景，红/蓝两艘目标船并行走正弦编队（波长 60m、
  幅度 6m、0.6m/s，分居曲线两侧），两艘白船直线干扰；被控 ASV 按语言指令
  （"跟随红色/蓝色目标船，保持3米距离"）选择性地跟随其中一艘。
- **Jetson Orin Nano**（ROS 2 Humble）：视觉编码（MobileNetV3 冻结特征）+
  实体张量 + 语言 embedding → 481K 参数 MLP 策略（ONNX/CPU）→ 确定性安全门 →
  运动控制 → UE5 执行。
- 目标：**真实模型在线推理闭环**（不是专家控制器），为 ESP32 硬件扩展留接口。

## 2. 双端环境

| | PC（训练侧） | Jetson（推理侧） |
|---|---|---|
| 位置 | `C:\Users\LIU\Documents\jetson_ws\asv_vla`（WSL 挂载 `/mnt/c/.../asv_vla`，旧路径 `day11_kinematic_work` 是软链接） | `jetson@192.168.137.100`，`~/jetson_asv_ws`（与 PC 同步的独立 git） |
| 关键 | Windows venv：`pc_datasets/.venv-day13/Scripts/python.exe`；UE5 项目 `D:\Unreal Projects\VLA` | SSH 密钥：`/tmp/asv_key`（或 `C:\Users\LIU\.ssh\asv_day12_ed25519`）；micro-ROS agent 已装 |
| 数据 | `pc_datasets/`（特征缓存/checkpoints/registry） | `~/jetson_asv_ws/models/policy.onnx` + embeddings |
| git | 分支 `cleanup/ultimate-restructure`（PC 提交 8a75649 最近） | 分支 `fix/day19-closed-loop` |

SSH 命令模板：
```bash
ssh -i /tmp/asv_key -o BatchMode=yes jetson@192.168.137.100 'cd ~/jetson_asv_ws && source /opt/ros/humble/setup.bash && source ~/microros_ws/install/setup.bash && source install/setup.bash && ...'
```

## 3. 架构与数据流（在线闭环）

```
UE5（Windows）                          Jetson（ROS2）
Connection 蓝图（场景/相机/实体上报）     ue_object_deliverer_bridge_node（TCP :8080 服务）
  - S2 正弦编队、L6/L6B 布局              ├─ /ue/camera_frame（1280x720 JPEG, FOV 90°）
  - 实体投影（base_link 相对坐标）        ├─ /ue/entities（4 船：位置/速度/颜色/可见性）
  - 蓝图"counterbalanced"巡航            └─ /ue/asv_state
C++ 执行器（SceneAutomationSubsystem）      │
  - 8081 端口收 setpoint → teleport 移动   ├─ visual_encoder → /vla/visual_features
  - 世界 Tick 结束后强制应用位移            │    (17-token: 全局帧+16 槽位实体 crop, MobileNet)
  - 第一条 setpoint 时锚定"当前巡航位置"    ├─ task_entity_tensor → /vla/task_features
                                          │    (16×16 张量；颜色列[14/15]置零=特权列)
                                          ├─ language_stub → /vla/language_embedding
                                          │    (预计算 Qwen embedding；红/蓝可运行时切换)
                                          ▼
                                   vla_policy（ONNX/CPU, 481K MLP）
                                          ├─ 输出 20×2 轨迹（dt=0.2s, 4s 视界）+ STOP logit
                                          └─ 5 帧平滑 → /vla/policy_trajectory
                                          ▼
                                   safety_gate（唯一发布者）
                                          ├─ 速度≤1.5m/s、曲率≤15、方向≤170°、碰撞 0.5m
                                          ├─ 只查可执行前缀（2-5 步）；STALE 1s/E-STOP 2s
                                          └─ /vla/selected_trajectory
                                          ▼
                                   trajectory_controller → /decision/output
                                          ▼
                                   decision_setpoint_adapter → /ue/kinematic_setpoint
                                          ▼ bridge → TCP → UE5 执行器 → ASV 移动 → 新一帧…
```

**训练侧**（PC）：UE5 采集（expert 驱动）→ episode 包 → 特征缓存（冻结 MobileNet+Qwen）
→ 训练（3 seeds，验证门）→ ONNX 导出（parity 校验）→ 部署 Jetson。

**ESP32 硬件链**（保护项，未改动）：`/decision/output → control_input_mux →
/control/control_input → [ESP32 固件] → /control/asv_wrench → safety_supervisor →
/control/safe_wrench → thruster_allocator → /ue/thruster_command → bridge(thruster) → UE5`。
`hardware_loop.launch.py` 已备（use_fake_esp32 默认 true）；接口规范见
`docs/esp32_interface.md`。**不要动**：asv_control_manager 5 节点、full_system.launch.py、
fake_esp32_wrench_node、上述话题名。

## 4. 已完成的工作（按时间）

### A. 终极整理（完成，PC 25c037f / Jetson 5877b30）
- 目录改名 `asv_vla`（旧路径软链接兼容）；31 个文件去 dayX 改名、70 文件内容清理
- pc_datasets 4.3GB→1.5GB；TODO.md 拆为 ARCHITECTURE.md + HISTORY.md（审计史保留）

### B. UE5 场景（完成，PC 003256e）
- S2 正弦运动（波长/幅度/速度命令行可调）+ L6（红左蓝右）/L6B（红右蓝左）布局
- `-YawFixWholeRun`：抑制蓝图中途 180° 翻转（headless 实测 70s yaw=0）
- `-SineDelay`：编队延迟前进（让闭环在训练分布内启动）
- 蓝图行为（headless 实测）：无 setpoint 时巡航（0.6m/s 追 target）；有 setpoint 流
  时停止巡航但不应用位移（**headless 下蓝图不执行 setpoint**）

### C. 数据采集与训练（完成，验证门 PASS）
- 采集 14 runs（L6/L6B × 7 seeds，100 帧/run，全过质量门）；`-YawFixWholeRun` 全程
- 特征缓存 14 个/117500 样本；**ego 置零**（训练/在线一致）
- 训练增强：几何噪声 + 槽位 dropout + 镜像（mirror_prob 0.3）+ 指令互换
  （instruction_swap_prob 0.4，重算专家标签）
- **checkpoints/sine_formation_v4**（当前最佳）：**VALIDATION_PASS seeds=17,23,42**
- 离线选择指标 96.2%（v2 时代；v4 待重测）
- ONNX 导出 parity 精确；已部署 Jetson（旧模型备份 policy_day21v2_backup.onnx）
- 语言 stub 支持红/蓝指令运行时切换（`ros2 param set /language_stub active_embedding <path>`）

### D. Headless 模型闭环（D2 完成，D3 进行中）
- **C++ setpoint 执行器**（SceneExecPort 8081）：headless 下蓝图不执行 setpoint，
  改为 C++ 执行（世界 Tick 结束后强制应用，赢过蓝图位置控制）
- **bridge 双连接**（execution_address/execution_port 参数）：setpoint 走执行器
- **stamp 单调化**（task_entity_tensor + vla_policy）：UE5 模拟时钟 headless 下回退
  → 消除每 run 2500+ 次 STALE 误判（修复后 0 次）
- 实测（干净环境）：ASV 持续前进 183-190m、门几乎全 PASS（仅少量 CURVATURE_LIMIT）
- **选择问题（未解决）**：在线 ASV 追"最近的"目标（blue）而非指令色（red）——
  根因分析见 §5

### E/F. ESP32 扩展 + 演示
- `hardware_loop.launch.py`；烟测验证 fail-closed 传播；`docs/esp32_interface.md`
- `docs/demo_runbook.md`（Play 模式演示流程）；`docs/scene_verification.md`（验证记录）

## 5. 当前核心问题：在线选择失败（进行中）

**现象**：follow-red 指令下，headless 在线闭环中 ASV 追"最近的"blue 而非 red。
**离线回放**（eval_online_replay.py，用 R16 真实在线输入喂模型）100% 追 blue。

**已排除**：crop 槽位错位（交换实验无影响）、语言条件（red vs blue embedding 有
差异但不够强）、时序错配（offset 实验）、stamp（已修复）。

**关键发现链**：
1. 训练缓存 follow-red 帧中 red 近占 78.5%（blue 近仅 21.5%）→ 模型倾向"追最近"
2. 镜像增强（v3）+ 指令互换增强（v4）→ 验证门过但 R16 回放仍 0%——几何层面
   增强无效
3. **在线视觉特征统计与训练缓存匹配**（均值/方差几乎一致）——不是全局分布问题
4. **最可能根因（正在验证）**：在线 ASV 被锚定在**起点**（静止视角），而采集时
   ASV 巡航（视角动态、target 近、crop 大）——**在线静止视角 OOD** →
   模型困惑 → fallback 追最近
5. **当前修复（R17 验证中）**：C++ 执行器第一条 setpoint 时锚定**巡航当前位置**
   （不是起点）——ASV 先被蓝图巡航带到 target 附近（视角与采集一致），策略从
   近距接管 → 选择应在分布内正确

**R17 验证方法**（重跑）：
```bash
# Windows:
cd tools/ue5 && powershell.exe -File verify_demo_seed.ps1 -SceneSeed 200101 -LayoutId L6 \
  -MotionState S2 -RunSeconds 250 -SlotId D3-R17 -YawFixWholeRun -SceneExecPort 8081 -SineDelay 45
# Jetson（UE5 启动 ~30s 后）:
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy.onnx execution_address:=192.168.137.1
# 抓实体: python3 /tmp/r12_capture.py（Jetson 上已有，写 /tmp/r12_ents.json）
```
分析：实体数据的 red/blue 距离——ASV 最终更接近谁（选择正确性）。
**注意**：每次运行前必须清理残留进程：
```bash
ssh ... 'ps aux | grep -E "install/asv_vla|install/asv_ue" | grep -v grep | awk "{print \$2}" | xargs -r kill; ss -tlnp | grep 8080'
```

**如果 R17 修复成功**：部署 v4 ONNX + 完成 8 runs 验收表（选择正确率/间距误差/门统计）。
**如果仍失败**：下一个方向 = 重新采集（observation_only 模式：ASV 静止 + target
从 25m 开始——与在线一致的静止视角分布）→ 重训。这是最彻底的修复。

## 6. 关键文件索引

| 文件 | 作用 |
|---|---|
| `tools/ue5/Source/EDGE/SceneAutomationSubsystem.{h,cpp}` | UE5 场景+S2+L6+执行器（核心） |
| `tools/ue5/install_ue_automation.ps1` | 复制 C++ 到 UE 项目+编译（含 EDGE.Build.cs） |
| `tools/ue5/verify_demo_seed.ps1` | headless UE5 运行（参数：SceneSeed/Layout/Motion/YawFixWholeRun/SceneExecPort/SineDelay） |
| `tools/ue5/collect.ps1` | 自动化采集（Jetson 查 slot→UE5 headless→打包） |
| `src/asv_ue_bridge/src/ue_object_deliverer_bridge_node.cpp` | bridge（execution 双连接） |
| `src/asv_vla/asv_vla/vla_policy_node.py` | 策略节点（stamp 单调化） |
| `src/asv_vla/asv_vla/task_entity_tensor_node.py` | 实体张量（stamp 单调化） |
| `src/asv_vla/asv_vla/language_stub_node.py` | 语言 stub（红/蓝切换） |
| `src/asv_bringup/launch/vla_closed_loop.launch.py` | 在线闭环 launch（execution_address 参数） |
| `training/dataset.py` | 数据集+增强（mirror/swap/dropout/噪声） |
| `training/config/train_sine_v1.yaml` | 训练配置（augment 参数） |
| `run_train_wrapper.py` / `build_feature_wrapper.py` | PC 训练/特征构建入口（Windows 路径注入） |
| `training/evaluate_selection.py` | 离线选择指标（缓存） |
| `eval_online_replay.py` | **在线输入回放评估**（R16 特征喂模型） |
| `pc_datasets/checkpoints/sine_formation_v4/` | 当前最佳模型（3 seeds 全过） |
| `pc_datasets/r16_feats.json` | R16 在线特征抓取（回放评估用） |
| `docs/scene_verification.md` / `docs/esp32_interface.md` / `docs/demo_runbook.md` | 验证/接口/演示文档 |

## 7. 常用命令速查

```bash
# UE5 重建（C++ 改动后）
cd tools/ue5 && powershell.exe -File install_ue_automation.ps1

# Jetson 同步 + 构建（注意 Python 包需 rm -rf build/asv_vla 强制重建）
rsync -avz -e "ssh -i /tmp/asv_key -o BatchMode=yes" --exclude '.git' --exclude 'build' \
  --exclude 'install' --exclude '.venv*' --exclude 'artifacts' --exclude '*.onnx' \
  --exclude '*.npy' --exclude 'models/Qwen*' ./ jetson@192.168.137.100:~/jetson_asv_ws/
ssh ... 'cd ~/jetson_asv_ws && rm -rf build/asv_vla && colcon build --symlink-install --packages-select asv_vla asv_ue_bridge asv_bringup'

# 训练（PC，Windows venv）
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
/mnt/c/Users/LIU/Documents/jetson_ws/pc_datasets/.venv-day13/Scripts/python.exe run_train_wrapper.py

# 导出 ONNX（改输出路径）
# 部署：scp policy_sine.onnx → Jetson models/policy.onnx（先备份）

# 指令切换（在线）
ros2 param set /language_stub active_embedding /home/jetson/jetson_asv_ws/models/follow_blue_embedding.npy
```

## 8. 诚实边界（审计要求）

- 验证门通过 + 指标达标才宣称"训练达标"；未达标如实记录
- 模型闭环的最终验收（8 runs 实测表）未完成前，不宣称"模型在线闭环通过"
- 在线选择问题（§5）未解决前，演示视频不得用"模型选对目标"的画面
- HISTORY.md 保留全部根因与失败记录（v1/v2/v3 迭代、旧模型门未过等）

## 9. 下一步优先级（按序）

1. **R17 验证**（巡航锚定修复）→ 若成功：部署 v4 → 8 runs 验收表
2. 若失败：重新采集（observation_only S2 静止视角）→ 重训 → 回放验证
3. 蓝指令在线验证（active_embedding 切换）
4. 提交双端 + 更新 ARCHITECTURE.md 状态表
5. Play 模式演示（docs/demo_runbook.md）+ OBS 录制
6. ESP32 实机接入（可选，需硬件）
