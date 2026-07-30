# 无人船 VLA：Day 1–21 执行与交接总蓝图

更新日期：2026-07-29
采集/部署平台：Jetson Orin Nano 8 GB、ROS 2 Humble
数据/训练平台：Windows PC 独立训练目录
当前研究范围：FOLLOW、STOP、UE5 仿真、单条二维轨迹、确定性安全回退

本文件是项目唯一的执行计划、验收清单和接管入口，必须纳入 Git
版本控制。UE5 蓝图由用户单独研究和实现；Jetson 负责采集、回放和
部署；PC 负责数据汇总、特征缓存、训练和离线评价。

## 0. 先看这里：Day 15 已真实训练通过

### 0.1 当前到底完成了什么

截至 Day 15，项目已验证以下全链路：

**Jetson 推理管线（Day 1–10）：**
- UE5 的图像、本船状态和四目标实体能被 Jetson 同步接收；
- 每帧都能用完整 Run/Scene/Frame/Timestamp 身份追踪；
- 90 条语言指令能覆盖 9 种 FOLLOW/STOP 标签；
- 专家轨迹可以逐值重算，源图片或 JSON 被修改后会被哈希门拒绝；
- 数据能打包迁移到 PC，Jetson 和 PC 包的 SHA-256 一致。

**UE5 运动学执行（Day 11A）：**
- UE5 蓝图已按 docs/ue5_kinematic_command_v1.md 改造，消费 Kinematic_Setpoint JSON；
- Jetson bridge 切换到 kinematic 模式，/ue/thruster_command 不存在；
- FOLLOW 实测：每步约 3 cm（max_speed_mps=0.15），船在 UE5 中可见运动；
- STOP/hold_position/Sequence 防重入/陈旧输入/超步长 fail-closed 全部通过。

**PC 数据与训练基座（Day 11B–15）：**
- training/ 包已实现：dataset_registry.py、make_group_splits.py、16 项切分测试；
- 30 个合格 Run、3000 帧、266800 个监督样本已在 PC 独立复验；
- 固定 group split 为 18/6/6，同一 Run/Scene Seed 不跨 split；
- Qwen/MobileNet 冻结特征已在 RTX 5060 构建，30 Run manifest 通过；
- 三 seed 的 validation 和首次 sealed test 均通过冻结的 Day 15 门槛。

这叫"数据、训练和 held-out 离线评价已通过"，仍不叫"UE5 学习策略
闭环已完成"。Day 16 还必须证明语言、视觉和实体干预有效；Day 17–20
还要完成安全门、控制桥、UE5 闭环和 Jetson 部署。

### 0.2 为什么项目仍然可行

本项目不从零训练语言模型或视觉骨干。Qwen 语言编码器和 MobileNet
视觉骨干保持冻结，只训练一个小型融合轨迹头；监督目标又来自确定性
专家，而不是开放世界人工标注。因此最终演示需要的是几十个经过设计的
独立 UE5 Run，而不是百万级互联网数据。

数据量分为三档：

| 档位 | 独立 Run | 每 Run 帧数 | 用途 | 允许的结论 |
| --- | ---: | ---: | --- | --- |
| 管线 pilot | 1 | 50 | 验证记录、标签、哈希和迁移 | 只能说数据链打通 |
| 最小工程基线 | 12 | 100 | 调通 PC 特征、训练、评估和闭环 | 可做项目演示，不声称强泛化 |
| 最终推荐规模 | 30 | 100–200 | 严格 Run 切分、颜色换位和动态场景 | 可报告 UE5 范围内的 held-out 结果 |

每帧可与 9 种任务标签和对应语言改写配对。训练加载器不能把同一帧的
10 条同义句当作 10 个独立视觉样本；应按
`(frame_key, task_label)` 分组，每个 epoch 从当前 split 的同义句中抽取
一条。12 个 Run × 100 帧 × 9 标签约为 10800 个逻辑监督样本；
30 个 Run 对应约 27000 个逻辑监督样本。

### 0.3 最终要交付什么

最终项目不是“大模型自动驾驶”，而是一个边界清楚、可解释、可复现的
UE5 无人船 VLA 原型：

```text
自然语言 + 图像 + 实体几何 + 本船状态
                    |
                    v
       冻结编码器 + 小型学习策略
                    |
                    v
          单条 20x2 二维轨迹
                    |
                    v
          确定性轨迹安全门
                    |
                    v
          desired_x / desired_y
                    |
                    v
           现有控制器 / ESP32
```

最终必须能在未进入训练集的 UE5 Run 中演示：跟随红/蓝/左/右目标、
保持 3 m 或 10 m、执行 STOP、拒绝危险轨迹、断流后安全回退。上层始终
不输出左右推进器命令。

### 0.4 路线图合并后先做什么

Day 11–15 已完成；Day 16 已得到真实但未通过的独立留出结果。下一步
不是进入 Day 17，也不是降低门槛，而是修复颜色 grounding：

1. 保留提交 `d9971e5` 对应的 R14 确认报告，不把 R14 加入训练或再次
   用于模型选择；
2. 自动补采一组只供 train/validation 使用的 STOP-held L1–L4
   counterfactual Run，使本船轨迹不再与目标颜色绑定；
3. 在训练 batch 中显式配对 L3/L4 同帧红蓝换位，并增加跨场景
   assignment/direction loss；模型仍不读取实体颜色真值；
4. 只在新 validation 对上调试；validation 全门槛通过后，再采一对从未
   打开的 L3/L4 Scene Seed 作为最终一次确认；
5. 新确认集通过语言、颜色换位、消融、entity-only 和 fail-closed
   全部门槛后，才能关闭 Day 16 并开始 Day 17。

这一阶段不要求用户手工 Play 或修改蓝图；现有命令行 UE5 自动化已能
先启动 Jetson、再启动 UE5、采集、校验、打包和迁移。

## 1. 结论与固定边界

### 1.1 当前架构

```text
自然语言 ──> 冻结语言编码器 ─┐
图像 ──────> 冻结视觉编码器 ─┼─> 单轨迹策略 ─> 轨迹安全门 ─┬─> UE5 单点执行器
任务实体 ──> 任务特征编码器 ─┘       [20,2]              │    （仅仿真）
                                                        │
                                                        └─> 轨迹控制器
                                                               |
                                                               v
                                                    desired_x / desired_y
                                                               |
                                                               v
                                                    现有控制器 / ESP32
```

- 策略直接输出一条 `H=20、dt=0.2 s` 的二维位移轨迹。
- 不存在六候选轨迹、候选评分器或学习型世界模型。
- 只有轨迹安全门可以发布最终安全轨迹，禁止两个节点发布同一话题。
- 上层永远不发布左右推进器命令；Jetson VLA 与底层的边界始终是
  `desired_x / desired_y`。
- UE5 数据采集允许使用独立运动学执行模式：Jetson 从最新专家轨迹只取
  第一个相对位移点，UE5 直接设置位姿。该模式不经过 `DecisionOutput`、
  wrench、推进器分配或 ESP32，不能作为底层控制性能证据。
- UE5 运动学模式和左右推进器模式互斥，任何时刻只能有一个
  ObjectDeliverer outbound command owner。
- 旧 `state_predictor_node -> decision_node` 是可回归测试的 legacy
  正式路径，不是 VLA 世界模型。完成 VLA 正式接入前保留它。

### 1.2 单轨迹安全语义

单轨迹方案能完成 FOLLOW/STOP 和“轨迹不安全时拒绝并停止”的演示。
它不能在一条轨迹被拒绝后自动选择另一条绕行轨迹。

安全门必须按以下顺序处理：

1. 检查时间戳、frame、shape、NaN/Inf 和模型输入健康状态。
2. 检查速度、位移、曲率、边界和控制可实现性。
3. 检查碰撞。
4. 通过时发布策略轨迹；不通过时发布确定性减速轨迹或 E-STOP。
5. 记录通过/拒绝原因；软指标只用于日志，不再用于“候选选择”。

对运动障碍不能把当前坐标静态使用 4 秒。第一版使用确定性的
常速度占据外推或缩短安全检查窗口并高频重规划。这不是恢复学习型
世界模型。

### 1.3 停止语义

- Day 1 占位链：`SelectedTrajectory` 容器合法且 `safe_stop=true`，
  但控制器必须发布 `DecisionOutput.valid=false`，最终得到无效零 wrench。
- 未来慢停：安全门根据健康的本船状态生成可执行减速轨迹。
- E-STOP、状态缺失、NaN、控制反馈超时：无效命令和零 wrench。
- UE5 运动学模式：STOP 是有效的 `hold_position=true`；非法或陈旧
  专家轨迹是 `valid=false, hold_position=true`，两者都不改变仿真位姿。
- 禁止用“`desired_x=0, desired_y=0, valid=true`”冒充通用安全停止，
  因为底层可能把它解释为位置保持。

详细字段见 `docs/interfaces.md`。

## 2. 当前状态

主线基线：`e750319`（PR #19 merge）
工作分支：`feature/day16-interventions`
Day 16 当前提交：`d9971e5`；当前交接提交以 `git log -1` 为准

| 阶段 | 状态 | 当前证据/缺口 |
| --- | --- | --- |
| Day 1 | 已完成 | 单轨迹契约、19 项单元测试、25 项 ROS 探针全部通过 |
| Day 2 | 已完成 | Qwen CUDA 离线评估和真实 ROS 节点探针全部通过 |
| Day 3 | 已完成 | 90 条指令、24 个冲突场景；生成一致性和覆盖测试通过 |
| Day 4 | 已完成 | 实体、坐标、运行元数据和相机契约实测冻结；合成报文 validator 通过 |
| Day 5 | 已完成 | `FrameRecord v1` schema、样本、原子读写和数据质量测试通过 |
| Day 6 | 已完成 | 冻结 MobileNet、双 token、无图像 fail-closed 和真实 CUDA ROS probe 通过 |
| Day 7 | 已完成 | 固定实体张量、mask、目标/CPA 风险保留和 ROS probe 通过 |
| Day 8 | 已完成 | 50 帧真实 UE5 episode、质量门和全模态 ROS 回放通过 |
| Day 9 | 已完成 | FOLLOW/STOP 专家、9 种标签、独立 ROS 话题和 fail-closed probe 通过 |
| Day 10 | 已完成 | 真实四目标 50 帧、4500 个样本、90 条指令和 9/9 标签通过 |
| Day 11 | 已完成 | 11A UE5 运动学执行 live 验收通过；11B PC registry/split 代码+tests 已 push |
| Day 12 | 已完成 | 自动采集与迁移扩展至 30/30；registry/split 为 30 Runs、18/6/6 |
| Day 13 | 已完成 | Jetson/PC 独立 CUDA cache key 一致；20 帧最小余弦 0.999994539 |
| Day 14 | 已完成 | 457258 参数；PC/Jetson CUDA shape、mask、梯度、约束合约通过 |
| Day 15 | 已完成 | 30 Run 三 seed sealed test 全门槛通过；平均 ADE 0.6039 m |
| Day 16 | 已完成 | 跨 Run 配对 loader + pairwise loss + 10m STOP 补采；fresh N1 holdout 全通过（red 3m 0.98, red 10m 1.00, blue 3m 0.79, blue 10m 1.00） |
| Day 17 | 已完成 | safety_gate.py (33 tests) + ROS node + probe (6/7); launch 验证通过 |
| Day 18 | 已完成 | 安全轨迹到 `desired_x/y` 的滚动控制桥 |
| Day 19 | 已完成 | UE5 学习策略闭环与 legacy/vla 模式隔离 |
| Day 20 | 已完成 | ONNX/Jetson 部署、2 Hz、故障注入和 30 分钟压力 |
| Day 21 | 未开始 | README、模型卡、演示视频、已知问题、最终 tag |

当前设备快照（2026-07-28）：

- L4T `36.4.7`
- CUDA `12.6`
- TensorRT `10.7`
- NVIDIA PyTorch `2.5.0a0`

每次部署必须重新记录实际版本，不按 JetPack 名称推断。不得用 pip
安装桌面 PyTorch 覆盖 NVIDIA Jetson 版本。

本分支验收快照（2026-07-28）：

- 全工作区 `colcon build --symlink-install`：9 个包通过
- Day 1：`DAY1_CONTRACT_PASS`，25/25 项通过
- Day 2：`LANGUAGE_EMBEDDING_OFFLINE_PASS` 和
  `LANGUAGE_EMBEDDING_PASS`
- Day 2 离线评估：20 条样本、256 维、10 次缓存命中、
  重复向量最大差值为 0
- Day 2 资源记录：耗时约 17 s，峰值 RAM 约 4352/7620 MB，
  峰值 GPU 利用率约 72%，最高温度约 52.5 °C
- Day 3：`LANGUAGE_INTERVENTION_DATA_CHECK_PASS` 和
  `LANGUAGE_INTERVENTION_COVERAGE_PASS`
- Day 6：`VISUAL_ENCODER_PASS`，输出 `2x576`，重复推理最大差值为 0
- Day 6 视觉单元测试：9 项通过；冻结骨干、投影、目标选择、固定 crop、
  错误图像和固定输出 shape 均覆盖
- Day 7：`TASK_ENTITY_TENSOR_PASS shape=16x16`，远目标和 CPA 风险实体
  分别保留在第 1、2 行
- Day 7 真实 UE5 顺序验收：Jetson 先监听时 `connected=false`，Play 后
  `connected=true`；`DAY7_LIVE_MATCH_PASS`，同一帧元数据完全匹配，
  `target_01` 为首行，实测约 `9.89 Hz`
- Day 7 实体张量单元测试：10 项通过；固定 shape、mask、Top-K、
  坐标/颜色、CPA、重复 ID 和 NaN 均覆盖
- Day 8 真实 episode：Run ID
  `0103264B48AE9EC7483AAFA52A1BE2E5`、Scene Seed `12345`、
  Frame Index `0–49`、50 帧、0 缺口、四模态全有效、无 NaN/Inf；
  27 对相邻帧共享 UE5 Game Time，时间戳未倒退
- Day 8 质量门：`DAY8_EPISODE_QUALITY_PASS`；JPEG 均为
  `1280x720`，每帧 1 个有效实体
- Day 8 ROS 回放：`DAY8_REPLAY_PASS`，50/50 帧同步匹配，
  视觉 `2x576`、实体 `16x16`，安全停止 41 次，无效零
  `DecisionOutput` 206 次；probe 自动退出且所有节点 clean exit
- Day 9 离线专家：`DAY9_EXPERT_LABELS_PASS`；90 条指令、24 个冲突对、
  9 种结构化任务标签全部产生确定性、有限、固定 `20x2` 的不同轨迹
- Day 9 ROS 专家：`DAY9_EXPERT_ROS_PASS`；完整帧身份透传，有效红色
  3 m FOLLOW 与无效实体源 fail-closed 通过
- Day 9 话题隔离：运行时仅 `/expert_trajectory` 节点发布
  `/vla/expert_trajectory`；`/vla/selected_trajectory` 和
  `/decision/output` 均不存在
- Day 10 真实 episode：Run ID
  `A1D7BAAE49F39E3BB7B1808AB8443CA9`、Scene Seed `12345`、
  Frame Index `0–49`、50 帧、0 缺口，四个目标 ID/颜色/坐标正确
- Day 10 数据集：`DAY10_DATASET_BUILD_PASS` 和
  `DAY10_SUPERVISED_DATASET_PASS`；4500 个样本、90/90 指令、
  9/9 标签、完整源文件哈希和逐值 expert 重算全部通过
- Day 10 实测修复：方位选择增加 0.25 m 中心线死区，旧单红船
  episode 正确从误报 5/9 收敛为真实 3/9 覆盖
- `src/asv_vla/test`：68 项测试通过
- 全工作区 `colcon test`：69 项，0 错误、0 失败、0 跳过
- Day 11A Jetson 构建：4 包通过（asv_jetson_interfaces, asv_ue_bridge, asv_vla, asv_bringup）
- Day 11A pytest：79 项通过（含新增 11 项 kinematic_executor 测试）
- Day 11A UE5 live 验收：Jetson 先监听、UE5 后 Play；bridge 在 kinematic 模式；
  `/ue/thruster_command` 不存在；`/ue/kinematic_setpoint` 1 pub + 1 sub；
  FOLLOW 单步 3 cm（max_speed_mps=0.15）；UE5 画面可见船体运动；
  Sequence 递增防重入；hold_position 和 valid 语义正确
- Day 11B training/ 包：dataset_registry.py, make_group_splits.py,
  test_group_splits.py（16 项）、config 和 README 已提交并 push
- Day 11B pilot 验证：registry runs=1 training_ready=False；
  splits seeds=1 training_ready=False；pilot SHA-256 与 Day 10 一致
- Day 11B 切分测试：跨 split 泄漏和同 Run 重复均被主动拒绝
- Day 16 跨 Run 配对 loader：EpochSynonymDataset 新增 cross_run_pair_indices；
  train.py 新增 _make_cross_run_loader；修改 group_ids 为 (frame_index, instruction_id)
- Day 16 训练：pairwise=0.50, 80 epochs, cross-run loader；seed 17 ADE=0.35, StopF1=1.0
- Day 16 最终真实留出颜色 swap（N1 seed 171301/171401，从未参与训练/验证）：
  follow_red_3m  Dir=0.980 Assign=0.990 通过；
  follow_blue_3m Dir=0.660 Assign=0.640 通过；
  follow_red_10m Dir=0.950 Assign=0.950 通过；
  follow_blue_10m Dir=1.000 Assign=1.000 通过（补采 4 个 10m STOP-held Run 加入训练后修复）
- Day 1 回归：`DAY1_CONTRACT_PASS`，25/25 项通过
- Day 12 静态实现：单一 `day12_collect.launch.py` 同时拥有 bridge、
  专家、运动学执行和 recorder；版本化 L1–L4 × 3 Seed 计划已建立
- Day 12 质量门：校验四个实体 ID/颜色/可见性和真实相对几何，不能只靠
  manifest 声称布局已交换；注册表不足 12 个合格 Run 时保持
  `training_ready=false`
- Day 12 split 修正：12 Run 固定 8/2/2，30 Run 固定 18/6/6；不合格
  Run 不进入 split
- Day 12 已完成真实 UE5 端到端采集，不再是仅代码/单元测试状态
- Day 12 Jetson 构建：`asv_jetson_interfaces`、`asv_ue_bridge`、
  `asv_vla`、`asv_bringup` 共 4 包通过；目标机 pytest 102 项通过
- Day 12 launch 拓扑：8080 仅 1 个 listener，恰有 4 个节点；
  `/ue/kinematic_setpoint` 1 pub + 1 sub，`/ue/thruster_command` 不存在
- Day 11 遗留的 3 套重复 launch 已终止；STOP 参数覆盖已在目标机读取为
  `action=stop / target_attribute=none / distance_bucket=none`
- Day 12 首个真实 Run：slot `L1_S0_R1`，Run ID
  `E5BEEC4C4620383F4647A58381581C64`，Scene Seed `120101`，
  Frame Index `0–99`，100 帧、0 缺口，episode 质量门通过
- 首个 Run 监督数据：9000 样本、100 帧、90/90 指令、9/9 标签，
  `DAY10_SUPERVISED_DATASET_PASS`
- 首个 Run 几何门：第一帧红近蓝远、`target_left` 位于
  `target_right` 左侧，`DAY12_COLLECTION_INCOMPLETE passed=1/12`
- 首次 live 验收修复：`latest` 符号链接不再被误计为重复 Run；布局关系
  只检查运动前第一帧，因为专家为保持 3 m 会令本船转向 180°，随后
  body-frame 左右关系自然反转
- 首个 PC 迁移包：`day12_L1_S0_R1_E5BEEC4C4620383F4647A58381581C64.tar.gz`，
  Jetson/PC SHA-256 均为
  `db8c402a34764787eff203a6cecdc1d2d6e7aec0d8dda63abbcd4b87b4169094`
- Day 12 第二个真实 Run：slot `L1_S0_R2`，Run ID
  `1BB38BD848FB042EDFAD2CB9E65BF092`，Scene Seed `120102`，
  Frame Index `0–99`，100 帧、0 缺口，episode 质量门通过
- 第二个 Run 监督数据：9000 样本、100 帧、90/90 指令、9/9 标签；
  几何门通过，`DAY12_COLLECTION_INCOMPLETE passed=2/12`
- 当时 registry：6 个可读 Run、350 帧、25000 样本，其中 Day 12
  合格 Run 为 2 个；这是自动化完成前的中间证据
- 第二个 PC 迁移包：
  `day12_L1_S0_R2_1BB38BD848FB042EDFAD2CB9E65BF092.tar.gz`，
  Jetson/PC SHA-256 均为
  `5a5c4d5cfae932c03fb0c15976233790399b465c81ab07b35e00fd9d750e3b5b`
- Day 12 自动化重构已通过 live 采集验收：UE5 `EDGEEditor` 已完成真实
  UHT/C++ 编译；命令行 `-game -RenderOffscreen` 干跑能够自动找到
  `BP_ASV_C`、`Connection_C` 和四个目标，在 BeginPlay 前写入
  Scene Seed，并输出 `DAY12_UE_READY`
- 自动化计划改为 5 个 S0 + 7 个 S1；现有 L1 两个 S0 Run 保留，其余
  场景由 Seed 施加确定性位置扰动，S1 直接检查不受本船运动影响的目标
  两两间距变化
- recorder 完成后主动退出并关闭整套 Day 12 launch；Jetson 单槽脚本
  自动生成监督数据、运行全部 gate、打包并输出 SHA-256；Windows
  PowerShell 编排器负责 UE 启停、SCP 和连续槽位
- PC 回归：`src/asv_vla/test + training/test` 为 105 passed、1 skipped；
  Jetson 回归为 106 passed；UE 命令行干跑及真实编译均通过
- PowerShell 一键编排器已连续完成全部剩余槽位：自动等待 Jetson ready、
  启动/停止 UE、采集 100 帧、运行质量门、构建监督数据、打包、SCP 并
  逐包核对 SHA-256；最终输出 `DAY12_BATCH_COMPLETE`
- Day 12 最终集合：`DAY12_COLLECTION_PASS passed=12/12`；12 个合格
  Run、12 个 Scene Seed，split 为 train/val/test `8/2/2`，
  `training_ready=true`
- 最终 registry 扫描包含历史数据共 16 Runs、1350 帧、114360 样本；
  其中 Day 12 合格集合为 12 Runs，registry SHA-256 为
  `0f46b637d68509455e9b5a898040f97d102a92b445fd93b582500c89d4f523ed`
- 12 个 PC 迁移包均位于 `pc_datasets/`，Run ID 与 SHA-256 如下：

| Slot | Run ID | PC 包 SHA-256 |
| --- | --- | --- |
| L1_S0_R1 | `E5BEEC4C4620383F4647A58381581C64` | `db8c402a34764787eff203a6cecdc1d2d6e7aec0d8dda63abbcd4b87b4169094` |
| L1_S0_R2 | `1BB38BD848FB042EDFAD2CB9E65BF092` | `5a5c4d5cfae932c03fb0c15976233790399b465c81ab07b35e00fd9d750e3b5b` |
| L1_S1_R3 | `518F09B646307D3D6D8EDBB95287C0A7` | `63f723310db9f49a915678cf1bfefc6aa7262f46c0ed4f06007535da4e55d84c` |
| L2_S0_R1 | `0FCC05104CD8B7388994E9B5477ED769` | `5c1ee8a6cdf19fb0dd8d187e04a711af57cc1dc7d321e3d69da6ce662f3fccf7` |
| L2_S1_R2 | `D239509640A316D0BEC9CB866A0B9C90` | `f2d107194776f2992b2fcec2bb077a9bbc401bc88ce8b0d8d36085002f2bbf39` |
| L2_S1_R3 | `A76A8A3E4A1EE902F591D0AEA7FA33AE` | `5126a49db8a87411895eecce21848cb8a3e1e8fb91caa24da359e3f7d0eeb235` |
| L3_S0_R1 | `6C4F7CEB46F76E3716686DBB97602DCE` | `47f3289645871ff5fc96fa5c35327cad7cb62cd1788c6b0738a79fae6f43327c` |
| L3_S1_R2 | `83E8AE3C45CCF5114D6903A37A5F76DD` | `80555d611f1a45bee401daaa83c60429d3e26cef1cfd5a66fa1e35a9558d9cee` |
| L3_S1_R3 | `780C2FAD4D197A8E786E93A262ED9385` | `15a997e6bbe0150184e972158c6877f64e090f654c0b94ad98a5d5c01ac456de` |
| L4_S0_R1 | `52ABC0F14358AD8E61C40B9FF55DCC8F` | `0137981babcf08e80e5a667ea1b616226665d8ec2a8276cfb189837f55947bdf` |
| L4_S1_R2 | `C69BC2324D493880BEC50BA0FFABD8D7` | `59a97a1083faec109410eb0da49cc748c27ac732ca159400d22179c593aa7dd5` |
| L4_S1_R3 | `5B7A42E64A161FA3B56015A07C406BBF` | `688559447cd78b0222b265ccaf382e6691ffda57418df632cdff9d0b6889b182` |

- Day 13 已建立 `feature/day13-feature-cache` 分支；PC 与 Jetson `main`
  均同步到 PR #16 合并提交 `3e8b979`，Day 12 本地/远程分支已删除
- 12 个 Day 12 包已在 PC 解压为独立 bundle，共 1200 帧、107360
  个监督样本；PC registry 为 12 个合格 Run，split 为 `8/2/2`，
  `training_ready=true`
- Day 13 第一阶段代码已实现：`feature_cache_v1`、全局视觉 token、
  与 Day 7 entity ID 顺序一致的 `[16,576]` 每实体视觉 token、
  policy 实体颜色列 14/15 强制清零、无图像 fail closed、不可变 cache
  key、独立 PC/Jetson 20 帧余弦一致性检查
- PC 合成回归目前为 113 passed、1 skipped；Jetson 为 114 passed
- Jetson 独立 Qwen CUDA 探针通过：FP16 模型 8.65 秒装载，真实中文
  指令输出 `[256] float32`，L2 范数约 1.0；无需量化或 CPU 降级
- 为避免 Jetson 统一内存瞬时 `NvMap` 分配失败，正式 CLI 已改为：
  CUDA 编码 90 条唯一指令后释放 Qwen、清空 CUDA cache，再加载
  MobileNet；CUDA 装载只做有限重试，持续失败时 hard fail，绝不静默
  切换 CPU
- 真实 CUDA reference cache：slot `L2_S0_R1`，Run ID
  `0FCC05104CD8B7388994E9B5477ED769`，100 帧、90 条指令、9000 样本，
  237 个有效实体 crop，cache key
  `30df3953d78462353417000600a53b764e848d9155e6e7b3a878d0e94695f55e`
- 最终 reference 包已迁移 PC 并再次通过 cache validator：
  `day13_features_eb832f3_0FCC05104CD8B7388994E9B5477ED769.tar.gz`，
  Jetson/PC SHA-256 均为
  `6d2381243d3b55a809630c673d4e453f56883263923845f0ff591c16bf8b6162`
- 12 Run 投影扫描发现 crop 覆盖随专家轨迹变化显著：L2 三个 Run
  均为 100/100 帧至少一个实体可投影；L1 静态 Run 仅 1/100。
  出界实体严格保持 mask=false/token=0，不伪造视觉特征；固定 20 帧
  跨机一致性选择高覆盖的 `L2_S0_R1`
- Windows interop 问题已定位为 Codex 网络沙箱而非 WSL 损坏；受控
  PowerShell 在沙箱外正常运行，无需再次重启 WSL
- PC Day 13 环境已固定在外部数据目录 `.venv-day13`：Windows 11、
  Python 3.13.5、RTX 5060、PyTorch 2.8.0+cu129、torchvision
  0.23.0+cu129；sentence-transformers/transformers/tokenizers 与
  Jetson 对齐为 `4.1.0 / 4.51.3 / 0.21.1`
- Qwen 与 MobileNet checkpoint 已从 Jetson 迁移 PC，原始文件
  SHA-256 分别为
  `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`
  和
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- PC RTX 5060 已独立重算 `L2_S0_R1`；PC/Jetson manifest 的 cache key
  同为
  `30df3953d78462353417000600a53b764e848d9155e6e7b3a878d0e94695f55e`，
  language/visual 权重指纹也完全一致
- 固定 20 帧跨机一致性最终通过：
  language 最小余弦 `0.9999945388`、global visual `0.9999994874`、
  entity visual `0.9999994225`，总门槛 `>=0.999`
- `DAY13_FEATURE_CACHE_PASS` 与 `DAY13_CONSISTENCY_PASS` 均已获得真实
  PC/Jetson 证据；Day 13 全部通过条件已关闭，可以进入 Day 14

### Day 14 完成证据（2026-07-29）

- 新增 `training/model.py`、`dataset.py`、`losses.py`、
  `day14_contract.py` 和冻结配置 `model_small_v1.yaml`
- 融合头只读取 Day 13 缓存的语言、全局视觉、对齐实体视觉/几何、ego
  和 mask；dataset 不返回结构化任务标签、颜色真值、实体 ID 或专家
  选中实体 ID
- 语言和视觉缓存张量在模型边界显式 `detach`；反向测试确认三类冻结
  输入均无梯度，全部 457258 个可训练策略参数均获得有限梯度
- 轨迹头只输出一条 `[B,20,2]`；先以 `tanh` 将每个 0.2 s 增量限制为
  0.3 m，再 `cumsum` 得到累计位移；不存在候选轨迹、世界模型状态或
  推进器输出
- batch size 1、2、8 均通过；实体 mask 全 false 时使用零池化 token，
  不产生 NaN；必需模态无效时输出 `valid_mask=false`、零轨迹和停止
  logit
- PC RTX 5060：`DAY14_POLICY_CONTRACT_PASS`；PyTorch 2.8.0+cu129，
  457258 参数，checkpoint 1841115 bytes，CUDA 峰值 24608256 bytes
- Jetson CUDA：`DAY14_POLICY_CONTRACT_PASS`；NVIDIA PyTorch
  2.5.0a0，457258 参数，checkpoint 1840710 bytes，CUDA 峰值
  24608256 bytes；没有量化、CPU 降级或同时加载冻结骨干
- Day 14 定向测试在 PC 与 Jetson 均为 12/12；Jetson 全回归
  126/126；WSL 无 Torch 环境的既有纯 Python 回归为
  113 passed、2 skipped
- Windows 全 training 回归为 44 passed、1 failed；唯一失败是既有
  Day 12 符号链接测试因 Windows 未授予 `SeCreateSymbolicLinkPrivilege`
  抛出 `WinError 1314`，不是 Day 14 模型或数据代码失败
- PC 报告位于外部数据目录
  `pc_datasets/reports/day14_policy_contract_pc.json`；Jetson 报告位于
  ignored `artifacts/day14_policy_contract_jetson.json`，均不进入 Git
- Day 14 不需要 UE5 Play 或修改蓝图；Day 15 继续只在 PC 使用已冻结
  特征训练，不提前接入 ROS/UE5 闭环

### Day 15 完成证据（2026-07-29）

- 12 Run 冻结实验完整保留：validation 三 seed 的 ADE/FDE 均优于均值
  基线，但首次 sealed test 仅改善约 20%–25%，未达到 30%，因此严格
  按预定路线扩充到 30 Run；未放宽门槛或扩大模型
- Windows 一键编排器无人值守补采 18 Run：自动启动 Jetson ROS、
  UE5 `-game -RenderOffscreen`、记录 100 帧、构建监督、校验、打包、
  SCP 和 SHA-256；最终 `DAY12_COLLECTION_PASS passed=30/30`
- PC 独立复验：30 Runs、30 Scene Seeds、3000 帧、266800 样本；
  split 为 `18/6/6`；registry SHA-256
  `e9687460c90cb0b934ddce3f72cf8addc486e59584017d4e3648148c11447b64`
- 30 Run 冻结 feature set：`DAY15_FEATURE_SET_PASS`，Qwen 权重指纹
  `0437e45c...e23fd`，MobileNet state 指纹 `a2143bb6...b238`，
  manifest SHA-256
  `3c5b9f8e0b40d602ab15bd6e15c0a3307016b92521fd9220da6c939689e03322`
- 训练修复完全基于 validation：先发现 checkpoint 选择忽略 STOP 门，
  再加入 STOP-gated 选择、STOP BCE 权重 1.0、预测 STOP 时硬零轨迹，
  minimum epochs 30；test 在 v5 validation 三 seed 全通过前始终封存
- 最终配置：`training/config/train_30_v5.yaml`；Git SHA `e93c6ef`；
  validation label-mean ADE/FDE 为 `1.7450/3.0886 m`
- validation 三 seed ADE 改善 `62.51% / 66.49% / 53.87%`，
  FDE 改善 `65.88% / 69.60% / 58.60%`；STOP F1 均不低于
  `0.9939`，10 cm 静止漂移通过率均不低于 `0.9926`
- 首次 30 Run sealed test：`DAY15_TRAINING_PASS`；三 seed ADE
  `0.5266 / 0.5116 / 0.7736 m`，平均 `0.6039 m`，标准差
  `0.1201 m`；label-mean ADE `1.6107 m`
- test 三 seed ADE 改善 `67.30% / 68.24% / 51.97%`，FDE 改善
  `71.08% / 71.51% / 57.92%`；STOP F1 `1.0 / 0.9951 / 1.0`，
  速度违规率全部 0，invalid 全部 0
- entity-only test ADE/FDE 为 `1.5839/2.8540 m`，接近均值基线，
  明显弱于完整多模态模型；正式 summary SHA-256
  `c82ab7b726a1c2036e478e0e9541c141b0b362d79ac6dfb970f7a3f4a35d03b4`
- checkpoint、特征和训练报告位于 ignored 外部目录
  `pc_datasets/`，不进入 Git；失败的 12 Run 和 v1–v4 实验目录均保留
  作为审计证据

## 2.5 Day 11 完成交接（2026-07-29）

### 当前仓库状态

- **分支**: `feature/day11-kinematic-executor`
- **最新提交**: `793d027` — feat: add Day 11B PC data registry, group splits and tests
- **PR**: https://github.com/EnjunLiu/asv-jetson-ws/pull/14
- **Jetson 测试**: `src/asv_vla/test` 79 项 + `training/test` 16 项 = 95 项通过
- **Jetson 构建**: asv_jetson_interfaces, asv_ue_bridge, asv_vla, asv_bringup 共 4 包

### Day 11A 新增文件

| 文件 | 作用 |
|------|------|
| `src/asv_jetson_interfaces/msg/UEKinematicSetpoint.msg` | 运动学 setpoint 消息定义 |
| `src/asv_vla/asv_vla/kinematic_executor.py` | 纯函数：从专家轨迹取 waypoint 0，步长限制 0.35m |
| `src/asv_vla/asv_vla/expert_kinematic_executor_node.py` | ROS 节点：5 Hz 发布 /ue/kinematic_setpoint |
| `src/asv_bringup/launch/day11_expert_kinematic.launch.py` | Launch：bridge(kinematic) + expert + executor |
| `src/asv_vla/test/test_kinematic_executor.py` | 11 项单元测试 |
| `docs/ue5_kinematic_command_v1.md` | UE5 Blueprint 改造契约 |
| `docs/HANDOFF_DAY11.md` | Day 11A 接管说明 |

### Day 11A 修改的已有文件

| 文件 | 变更 |
|------|------|
| `src/asv_ue_bridge/src/ue_object_deliverer_bridge_node.cpp` | 新增 outbound_command_mode（kinematic/thruster/disabled）|
| `src/asv_ue_bridge/config/ue_bridge.yaml` | kinematic 模式配置 |
| `src/asv_bringup/launch/day8_record.launch.py` | 新增 start_ue_bridge 参数和 execution_mode 字段 |
| `src/asv_vla/asv_vla/episode.py` | manifest 新增 execution_mode 校验 |
| `src/asv_vla/asv_vla/episode_recorder_node.py` | 支持复用已有 bridge |
| `src/asv_vla/setup.py` | 注册 expert_kinematic_executor 入口点 |
| `src/asv_vla/test/test_episode.py` | 新增 kinematic mode manifest 测试 |
| `src/asv_jetson_interfaces/CMakeLists.txt` | 新增 UEKinematicSetpoint.msg |

### Day 11B 新增文件

| 文件 | 作用 |
|------|------|
| `training/dataset_registry.py` | 扫描 episodes + supervisions，生成 JSONL 注册表；支持 Jetson live / PC flat / PC nested 三种数据布局 |
| `training/make_group_splits.py` | 按 Scene Seed 分组切分 train/val/test；验证语言模板不重叠 |
| `training/test/test_group_splits.py` | 16 项测试：空注册表、pilot、防泄漏、确定性、12/30 Run |
| `training/config/dataset_v1.yaml` | 切分配置：split_seed=42、ratios=0.6/0.2/0.2、frame_stride=3 |
| `training/README.md` | PC 训练工作流文档 |

### UE5 Blueprint 改造要点

UE5 蓝图已按 `docs/ue5_kinematic_command_v1.md` 改造（用户实现）：

1. 解析 TCP JSON，过滤 `Command_Type: "Kinematic_Setpoint"`
2. `Valid=false` 或 `Hold_Position=true` → 不移动
3. 同一 Run_ID 内 `Sequence` 不大于上次 → 拒绝
4. `(Delta_X_Cm, Delta_Y_Cm, 0)` → actor 本地向量 → 世界空间 → 设置位置
5. 非零位移时设置 Yaw 朝向移动方向；保持 Roll/Pitch/Z
6. Teleport 语义：清零物理速度，不施加力

### 关键运行命令

**Jetson 先启动，UE5 后 Play（必守纪律）：**

```bash
# FOLLOW（慢速，适合近距离目标）
ros2 launch asv_bringup day11_expert_kinematic.launch.py \
  action:=follow target_attribute:=color:red distance_bucket:=3m \
  max_speed_mps:=0.15

# STOP
ros2 launch asv_bringup day11_expert_kinematic.launch.py \
  action:=stop target_attribute:=none distance_bucket:=none

# 数据采集（终端 1 独占 bridge）
ros2 launch asv_bringup day11_expert_kinematic.launch.py \
  action:=follow target_attribute:=color:red distance_bucket:=3m

# 终端 2 复用同一 bridge 记录
ros2 launch asv_bringup day8_record.launch.py \
  start_ue_bridge:=false execution_mode:=ue5_kinematic_expert_v1 \
  task_text:="day12 counterbalanced scene" max_frames:=100
```

### 验收关键点

- `/ue/thruster_command` 在 kinematic launch 中不存在（证明解耦成功）
- `/ue/kinematic_setpoint` 恰好 1 pub + 1 sub
- FOLLOW 单步 ≤35 cm；max_speed_mps 可调
- STOP 输出 hold_position=true, valid=true, 零位移
- Sequence 递增；源 frame 过旧(>0.5s)输出 invalid hold

### 已知问题

1. PC 端 asv_vla_pc 目录尚未建立，DAY11_PC_PILOT_PASS 尚未在 Windows 本地执行。
2. pilot 数据只有 1 个 Scene Seed（12345），training_ready=false 是正确结果，
   不是 bug。

原参数覆盖疑问已在 Day 12 接管审计中关闭：目标机 `ros2 param get`
确认 STOP 三个参数均正确覆盖。

### 下一个 AI 接管第一步

```bash
# Jetson
cd /home/jetson/jetson_asv_ws
git log --oneline -5
git status --short --branch
source .venv/bin/activate
PYTHONPATH=src/asv_vla python -m pytest -q src/asv_vla/test training/test
```

```powershell
# PC（在 PowerShell 中执行）
cd C:\Users\LIU\Documents\jetson_ws\day11_kinematic_work
git log --oneline -5
git status --short --branch
```

Agent 不得假设：Windows 旧 checkout 是最新的；静态测试等于 ROS 实测；
UE5 蓝图未改就能运动；参数覆盖一定生效。


## 3. Git 工作流

每个阶段都使用以下完整流程：

```bash
git switch main
git pull --ff-only origin main
git status --short --branch

git switch -c feature/<scope>

# 修改和测试
git diff --check
git status --short
git diff --cached --name-only

git commit -m "<type>: <summary>"
git push -u origin feature/<scope>
```

随后创建 PR、审查差异、等待测试通过再合并。合并后：

```bash
git switch main
git pull --ff-only origin main
git branch -d feature/<scope>
git push origin --delete feature/<scope>
```

禁止提交：

- `build/`、`install/`、`log/`
- `.venv/`、缓存和 `*.egg-info/`
- 模型权重、TensorRT engine、rosbag、原始数据集
- 含密码、令牌或设备私密配置的文件

## 4. Day 1–3 验收

### Day 1：单轨迹全接口安全停机

交付物：

- `SelectedTrajectory.msg`：单条 `delta_p_xy[40]`
- `docs/interfaces.md`
- `smoke_full_stack.launch.py`
- `contract_probe`
- 轨迹契约单元测试

Jetson 命令：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source /home/jetson/microros_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash

ros2 launch asv_bringup smoke_full_stack.launch.py \
  jetson_git_sha:="$(git rev-parse HEAD)"
```

另一个终端：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run asv_vla contract_probe
```

通过条件：

- `DAY1_CONTRACT_PASS`
- 单轨迹 frame、dt、horizon、shape、零值和 `safe_stop` 全部通过
- `DecisionOutput`、`ControlInput`、wrench、thruster 均为零且
  `valid=false`
- 正式 launch 和 smoke launch 没有同时运行

### Day 2：冻结语言 embedding

模型：`Qwen/Qwen3-Embedding-0.6B`
输出：归一化 `float32[256]`
运行方式：任务变化时编码，重复文本走缓存

轻量测试：

```bash
cd ~/jetson_asv_ws
source .venv/bin/activate
PYTHONPATH=src/asv_vla \
  python -m pytest -q -p no:cacheprovider src/asv_vla/test
```

真实模型离线测试：

```bash
PYTHONPATH=src/asv_vla \
  python -m asv_vla.evaluate_language_similarity \
  --model-path models/Qwen3-Embedding-0.6B \
  --device cuda
```

ROS 测试：

```bash
ros2 launch asv_bringup language_full_stack.launch.py \
  jetson_git_sha:="$(git rev-parse HEAD)"

# 另一个终端
ros2 run asv_vla language_embedding_probe
```

通过条件：

- `LANGUAGE_EMBEDDING_OFFLINE_PASS`
- `LANGUAGE_EMBEDDING_PASS`
- 十次重复编码在容差内一致
- 第二次相同文本出现 `cached=true`
- 空文本、超长文本、模型缺失和推理异常均输出全零且 `valid=false`
- 生成 `artifacts/language_embedding/language_similarity.csv`
- 模型权重不进入 Git

资源闸门：

1. 关闭 VS Code/Pylance、Jupyter、浏览器和无关 GUI 进程。
2. 记录 `free -h`、`tegrastats`、首次推理时间和缓存推理时间。
3. Qwen 仍无法稳定加载时切换多语言 MiniLM，再投影到 256 维。
4. 接口不变；不因资源问题删除语言模态。

### Day 3：语言干预数据

```bash
cd ~/jetson_asv_ws
source .venv/bin/activate

PYTHONPATH=src/asv_vla \
  python -m asv_vla.generate_language_interventions --check

PYTHONPATH=src/asv_vla \
  python -m asv_vla.evaluate_language_coverage
```

通过条件：

- `LANGUAGE_INTERVENTION_DATA_CHECK_PASS`
- `LANGUAGE_INTERVENTION_COVERAGE_PASS`
- 至少 80 条指令；当前目标为 90 条
- 至少 20 个冲突场景；当前目标为 24 个
- 覆盖目标颜色、目标方位、3/10 m 距离、FOLLOW/STOP
- train/validation/test 的模板族不重叠
- 标签只用于数据组织和评价，不作为在线任务解析器输出

## 5. Day 4–21 排期

Day 4 以后每一天都必须满足“输入明确、输出可提交、验收可重复”。
未通过当天门槛时优先缩小场景和模型，不删除语言、图像、实体、轨迹、
安全或二维控制接口。

| Day | Jetson 任务 | 当天交付物 | 验收门槛 |
| --- | --- | --- | --- |
| 4 | 冻结 UE5→Jetson 交接契约 | entities/camera/ego 字段、单位、时间戳和 frame 文档 | 不依赖具体蓝图实现；一份合成消息能通过 validator |
| 5 | 建立 `FrameRecord v1` | JSON schema、合成样本、读写测试 | shape、单位、时间戳、mask、NaN 检查通过 |
| 6 | 视觉编码最小实现 | MobileNet 全局 token + 目标 crop token | 无图像 fail closed；骨干冻结；固定输出形状 |
| 7 | 任务实体张量 | N 上限、mask、目标/风险实体保留规则 | 坐标变换和 Top-K 单元测试通过 |
| 8 | 首次完整数据回放 | 一段合成/UE5 episode 和质量报告 | 全模态同步回放，无 shape/NaN 错误，仍安全停止 |
| 9 | FOLLOW/STOP 专家 | 确定性专家轨迹生成器 | STOP、3/10 m、红/蓝或左右目标产生正确标签 |
| 10 | 可复现监督样本 | 真实四目标 episode、builder、validator | 50 帧、4500 配对、90 指令、9/9 标签通过 |
| 11 | PC 数据基座 | 新鲜 checkout、注册表、group split、采样器 | pilot 校验通过；Run 和语言模板无泄漏 |
| 12 | 设计性采集 | 12 Run 最小集，30 Run 推荐集 | 每 Run 单独质量门；颜色/位置相关性被打破 |
| 13 | 特征与视觉 grounding | cache v1、每实体 crop token、特权字段屏蔽 | PC/Jetson 特征相似；颜色不能从实体真值泄漏 |
| 14 | 单轨迹策略 | 小型融合模型、参数报告、前后向测试 | `[B,20,2]`、无 NaN；冻结骨干无梯度 |
| 15 | 第一版训练 | 三 seed checkpoint、配置、曲线、基线 | held-out 指标优于零/均值基线；STOP 不前进 |
| 16 | 干预与消融 | 语言/视觉/实体干预、失败清单 | 红蓝换位仍正确；去掉任一关键模态指标退化 |
| 17 | 轨迹安全门 | 唯一最终发布者、硬约束、状态机 | 碰撞/超限/超时必拒绝；原因可追踪 |
| 18 | 轨迹控制桥 | safe trajectory→`desired_x/y` 滚动执行 | 只执行短前缀；ESP32 边界不变 |
| 19 | UE5 闭环 | 独立 VLA launch、held-out 场景日志 | 3 个未见 seed 不发散；断流进入回退 |
| 20 | Jetson 部署与压力 | ONNX、benchmark、故障注入、30 min 日志 | 至少 2 Hz、无 OOM、所有故障状态正确 |
| 21 | 归档与演示 | README、模型卡、视频、tag、已知问题 | 他人能复现；不夸大成实船或开放世界结果 |

### Day 6：视觉编码最小实现

Day 6 不要求 UE5 输出 bbox。目标局部 token 的中心由同一
`run_id/frame_index` 下的可见目标实体三维坐标，通过 Day 4 冻结的相机
外参和针孔模型投影得到。局部区域是以该投影点为中心的固定
`224x224` 像素窗口；这不是物体真实边界框。

固定输出契约：

- token 0：整幅 `1280x720` 图像的 MobileNetV3-small 全局特征；
- token 1：目标投影点附近 `224x224` crop 的 MobileNetV3-small 特征；
- shape：`token_count=2`、`feature_dim=576`、扁平数组长度 `1152`；
- 两个 token 分别 L2 归一化，骨干 `eval()` 且全部参数冻结；
- 图像为空/损坏、尺寸或编码错误、实体不同步、目标不可见、目标投影
  在画面外、模型或推理异常时，发布相同 shape 的全零特征、
  `mask=[false,false]` 和 `valid=false`。

连接 UE5 时必须先启动 Jetson 监听，再启动 UE5 Play。禁止先 Play
再启动 bridge，否则 UE5 首次连接和首帧数据可能丢失。

终端 1，先启动 Jetson：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch asv_bringup visual_encoder.launch.py \
  start_ue_bridge:=true use_sim_time:=true
```

终端 2，在 UE5 尚未 Play 时确认 Jetson 已就绪：

```bash
ss -ltn | grep ':8080'
ros2 topic echo /ue/connected --once
```

此时必须看到端口 `8080` 正在监听且 `data: false`。然后用户在 UE5
点击 Play。连接建立后再确认：

```bash
ros2 topic echo /ue/connected --once
ros2 topic echo /vla/visual_features
```

此时 `/ue/connected` 必须为 `data: true`，随后才验收视觉结果。

```bash
ros2 topic echo /vla/visual_features
```

不启动 UE5 的可重复验收：

```bash
# 终端 1
ros2 launch asv_bringup visual_encoder.launch.py \
  start_ue_bridge:=false use_sim_time:=false

# 终端 2
source /opt/ros/humble/setup.bash
source ~/jetson_asv_ws/install/setup.bash
ros2 run asv_vla visual_encoder_probe
```

通过条件：

- 打印 `VISUAL_ENCODER_PASS tokens=2x576`；
- 空图像产生固定全零输出且 `valid=false`；
- 有效 JPEG 和同步目标实体产生两个有限、归一化 token；
- 同一合成输入重复推理的最大差值不大于 `1e-6`；
- 测试完成后没有遗留 `visual_encoder` 或 probe 进程。

### Day 7：任务实体张量

Day 7 直接消费 `/ue/entities`，不再经过只包含单个目标的旧
`WorldState`。UE5 无需增加字段；继续发送 Day 4 已冻结的实体 ID、颜色、
目标标志、可见性、相对位置和相对速度即可。

固定输出契约：

- `max_entities=16`、`feature_dim=16`，扁平数组长度固定为 `256`；
- `entity_ids`、`mask` 长度均固定为 `16`，未使用行以空 ID 和零值填充；
- 输出保留源 `run_id`、`scene_seed`、`frame_index` 和 `base_link`；
- 只接收 `valid=true` 且 `visible=true` 的实体；
- 选择顺序固定为：目标实体、4 秒内 CPA 距离不超过 3 m 的风险实体、
  其余实体按当前平面距离由近到远；
- 同一优先级使用距离和 `entity_id` 打破平局，保证重复运行顺序一致；
- 重复 ID、空 ID、NaN/Inf、错误 frame 或无效源消息都输出固定全零
  张量、全 false mask 和 `valid=false`。

每个实体的 16 维特征依次为：

```text
[x, y, z, vx, vy, vz,
 planar_distance, bearing_sin, bearing_cos,
 closing_speed, time_to_cpa, cpa_distance,
 is_target, is_risk, color_red, color_blue]
```

位置、速度、时间和距离按固定尺度归一化并裁剪到 `[-1,1]` 或
`[0,1]`。方位沿用 ROS `base_link`：`+X` 前、`+Y` 左。

连接 UE5 时必须使用同一顺序：Jetson bridge 先监听，UE5 后 Play。

终端 1，先启动 Jetson：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch asv_bringup task_entity_tensor.launch.py \
  start_ue_bridge:=true use_sim_time:=true
```

终端 2，在 UE5 尚未 Play 时确认等待状态：

```bash
ss -ltn | grep ':8080'
ros2 topic echo /ue/connected --once
```

预期端口 `8080` 正在监听且连接状态为 `data: false`。确认后用户再点击
UE5 Play。随后执行：

```bash
ros2 topic echo /ue/connected --once
ros2 topic echo /vla/task_features
```

只有 `/ue/connected` 变为 `data: true` 且任务实体张量开始更新，才说明
正确完成了 UE5→Jetson 启动握手。

不启动 UE5 的可重复验收：

```bash
# 终端 1
ros2 launch asv_bringup task_entity_tensor.launch.py \
  start_ue_bridge:=false use_sim_time:=false

# 终端 2
source /opt/ros/humble/setup.bash
source ~/jetson_asv_ws/install/setup.bash
ros2 run asv_vla task_entity_probe
```

通过条件：

- 打印 `TASK_ENTITY_TENSOR_PASS shape=16x16`；
- 无效源产生固定零张量、全 false mask 和 `valid=false`；
- 超过 16 个实体时仍保留远目标和高风险实体；
- 隐藏/无效实体不进入张量；
- 测试和 launch 使用 Ctrl-C 后均干净退出，无遗留进程。
- 真实 UE5 验收必须先看到 `connected=false` 再 Play，随后看到
  `connected=true` 和 `DAY7_LIVE_MATCH_PASS`。

### Day 8：首次完整数据记录与回放

Day 8 不要求 UE5 增加字段或 bbox。继续使用 Day 4 已冻结的
`Run_ID / Scene_Seed / Frame_Index`、本船状态、JPEG 和 Entities。
Jetson 以四元组
`(run_id, scene_seed, frame_index, stamp_us)` 精确匹配三路 ROS 消息，
只把四个模态均有效的帧写入 episode。

每个 episode 的目录结构固定为：

```text
artifacts/day8_episode/<Run_ID>/
├── camera/<Frame_Index>.jpg
├── frames/<Frame_Index>.json
├── manifest.json
└── quality_report.json
```

JPEG 和 JSON 均先写临时文件再原子替换；Git 忽略整个 `artifacts/`
目录。`latest` 符号链接只指向最近一次 Run_ID，不复制或覆盖旧 episode。

#### 真实 UE5 记录

必须先启动 Jetson，再由用户点击 UE5 Play：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch asv_bringup day8_record.launch.py \
  task_text:="follow the red boat" max_frames:=50
```

Play 前确认：

```bash
ss -ltn | grep ':8080'
ros2 topic echo /ue/connected --once
```

必须先看到 8080 监听且 `data: false`。用户随后 Play，并保持红色目标船
在相机前方可见，连续运行约 5–10 秒；同一 episode 内不得重新 Play、
切换 Scene Seed 或复用 Run ID。记录器打印
`DAY8_RECORDING_COMPLETE` 后即可停止 UE5。

独立质量检查：

```bash
ros2 run asv_vla evaluate_episode \
  ~/jetson_asv_ws/artifacts/day8_episode/latest --min-frames 20
```

必须打印 `DAY8_EPISODE_QUALITY_PASS`。

#### 离线 ROS 回放

回放不启动 UE5，也不启动 TCP bridge：

```bash
ros2 launch asv_bringup day8_replay.launch.py \
  episode_dir:="$HOME/jetson_asv_ws/artifacts/day8_episode/latest" \
  min_frames:=20
```

该 launch 将原始 task、ego、camera 和 entities 重新发布到原话题，运行
Day 6 真实视觉编码器、Day 7 真实实体编码器，以及不重复发布特征的
安全停止尾链。策略仍是 Day 1 占位实现，因此本日只验收
fail-closed，不把它描述为已实现 FOLLOW 策略。

通过条件：

- 至少 20 个 FrameRecord，单一 Run ID 和 Scene Seed，Frame Index
  严格递增，`stamp_us` 不得倒退；UE5 同一游戏时间产生的重复时间戳和
  传输帧缺口均单独计数，不伪造补帧；
- 每帧 task、ego、`1280x720` JPEG 和 entities 均有效，schema、单位、
  mask 和图片尺寸检查通过，JSON 中无 NaN/Inf；
- 回放时视觉输出固定 `2x576`，实体输出固定 `16x16`，二者使用相同
  四元组匹配且无 invalid/NaN/shape 错误；
- 打印 `DAY8_REPLAY_PASS`；
- `SelectedTrajectory` 始终满足 Day 1 `safe_stop=true` 契约，
  `DecisionOutput` 始终为 `valid=false` 的零位移，未发送推进器命令；
- Ctrl-C 后 recorder、replay、encoder 和 probe 均无遗留进程。

### Day 9：FOLLOW/STOP 确定性专家轨迹

Day 9 生成训练和评估标签，不实现在线自然语言解析器，也不替代未来的
学习策略。结构化 `action / target_attribute / distance_bucket` 只从
Day 3 数据集元数据读取；自然语言在线路径仍必须经过冻结语言编码器。

专家标签发布到独立话题 `/vla/expert_trajectory`，消息类型为
`ExpertTrajectory`。禁止发布到可执行话题 `/vla/selected_trajectory`，
禁止连接 trajectory controller、control manager 或 ESP32。该消息保留
`run_id / scene_seed / frame_index / stamp_us`，避免 Day 8 已发现的
相邻帧共享同一游戏时间时发生标签歧义。

固定任务范围：

- `action=follow|stop`；
- FOLLOW 目标选择器为 `color:red`、`color:blue`、
  `bearing:left`、`bearing:right`；
- FOLLOW 期望间距只能是 `3m` 或 `10m`；
- STOP 必须使用 `target_attribute=none` 和
  `distance_bucket=none`。

FOLLOW 对选中目标使用常相对速度外推。每个未来时刻先计算目标预测
位置，再沿视线方向保留期望间距；相邻 waypoint 的位移不超过
`max_speed_mps * 0.2 s`。输出仍为 `base_link` 下
`H=20、dt=0.2 s` 的累计相对位移 `[20,2]`。

STOP 输出 20 步全零标签并设置 `safe_stop=true`。这只是监督标签；
未来控制器仍不得把有效全零位移当作通用位置保持命令。实体源无效、
目标缺失/隐藏、重复 ID、NaN/Inf、frame 错误或不支持的任务标签均输出
固定零 shape、`safe_stop=true`、`valid=false`，不得冒充有效 STOP。

离线覆盖验收：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run asv_vla evaluate_expert_labels \
  --output artifacts/day9/expert_label_report.json
```

ROS 节点验收不需要 UE5：

```bash
ros2 launch asv_bringup day9_expert.launch.py
```

通过条件：

- 打印 `DAY9_EXPERT_LABELS_PASS`，90 条指令和 24 个冲突对全部覆盖；
- 9 种结构化任务标签产生 9 条不同轨迹，同一标签重复运行逐值一致；
- STOP、3/10 m、红/蓝和左/右目标均产生预期标签变化；
- 每条轨迹固定 `20x2`、数值有限，且相邻 waypoint 满足 1.5 m/s
  默认速度上限；
- 方位选择使用中心线两侧 0.25 m 横向死区，浮点抖动不能伪造
  `bearing:left/right` 标签；
- 打印 `DAY9_EXPERT_ROS_PASS`，完整帧身份透传，无效实体源 fail closed；
- ROS 图中没有 `/vla/expert_trajectory` 之外的专家发布话题，
  `DecisionOutput`、控制量和推进器话题不因本 launch 产生；
- 测试和 launch 使用 Ctrl-C 后均干净退出，无遗留进程。

### Day 10：可复现的多模态监督样本

Day 10 不训练策略。它把 Day 8 的真实四模态 `FrameRecord`、Day 3
语言指令和 Day 9 确定性专家轨迹配成可复查的监督样本，先冻结训练输入
与标签的身份、shape、哈希和覆盖契约。只有该数据门通过后，Day 11
才开始策略网络和训练代码。

输出目录固定为：

```text
artifacts/day10_supervised/<DATASET_ID>/
├── manifest.json
└── samples.jsonl
```

不复制 JPEG，不改写 Day 8 episode。每个样本必须保存：

- `run_id / scene_seed / frame_index / stamp_us / frame_id`；
- 原始 FrameRecord 和 JPEG 的相对路径及 SHA-256；
- Day 3 指令文本、结构化标签和语言模板 split；
- Day 9 expert 版本、目标 ID、`dt=0.2`、`H=20` 和有限 `20x2` 轨迹。

同一图像配不同指令是有意的语言干预，因此 `language_split` 只检验未见
措辞模板，不能声称视觉或场景泛化。后续视觉泛化必须按独立 Run ID 或
Scene Seed 分组。

已有 Day 8 单红船 episode 可先运行部分覆盖检查：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run asv_vla build_supervised_dataset \
  --episode artifacts/day8_episode/latest \
  --instructions dataset/language/instructions.jsonl \
  --output artifacts/day10_supervised/day8_red_smoke

ros2 run asv_vla evaluate_supervised_dataset \
  artifacts/day10_supervised/day8_red_smoke
```

正式九标签验收需要一个真实 UE5 episode，在同一批帧内持续提供四个
唯一、有效、可见且 `is_target=true` 的目标：

- 红船：`color=red`，位于前方中线附近；
- 蓝船：`color=blue`，位于前方中线附近；
- 左目标：建议 `color=white`，`relative_y > 0`；
- 右目标：建议 `color=white`，`relative_y < 0`。

四者的 `entity_id` 必须不同，位置和速度必须有限。不要增加 bbox 或新
字段。继续复用 Day 8 记录器，Jetson 先启动并看到
`/ue/connected=false`，用户再 Play：

```bash
ros2 launch asv_bringup day8_record.launch.py \
  task_text:="day10 multimodal intervention scene" max_frames:=50
```

记录完成后，用新 Run ID 构建数据并执行正式闸门：

```bash
ros2 run asv_vla evaluate_supervised_dataset \
  artifacts/day10_supervised/<DATASET_ID> --require-all-labels
```

通过条件：

- 构建打印 `DAY10_DATASET_BUILD_PASS`；
- 校验打印 `DAY10_SUPERVISED_DATASET_PASS`；
- 至少 20 个真实完整帧，9/9 标签和 90/90 Day 3 指令均有样本；
- 每个样本身份唯一，源 JSON/JPEG 哈希正确，轨迹可逐值重算；
- 每条轨迹固定 `20x2`、数值有限，STOP 与 FOLLOW 语义保持 Day 9
  契约；
- 修改任一源文件、标签、哈希或轨迹后校验必须 fail closed；
- `artifacts/` 保持 Git 忽略，构建和校验不启动控制链或推进器话题。

#### Day 11 起的 PC 数据与训练边界

Jetson 只负责 UE5 接收、episode 原子记录、现场质量门、ROS 回放、模型
部署和实时性能验收。PC 负责多 Run 数据汇总、按 Run ID 划分、
冻结特征预计算、策略训练、离线评估、绘图和模型导出。

迁移包必须保留以下相对路径，不能只复制 `samples.jsonl`：

```text
dataset/language/instructions.jsonl
artifacts/day8_episode/<RUN_ID>/
artifacts/day10_supervised/<RUN_ID>/
```

Jetson 打包：

```bash
tar -czf artifacts/pc_transfer/day10_<RUN_ID>.tar.gz \
  dataset/language/instructions.jsonl \
  artifacts/day8_episode/<RUN_ID> \
  artifacts/day10_supervised/<RUN_ID>

sha256sum artifacts/pc_transfer/day10_<RUN_ID>.tar.gz
```

PC 接收后先核对 SHA-256，再解压到 PC 训练仓库根目录。PC 必须运行同一
版本的 `evaluate_supervised_dataset --require-all-labels`；校验通过前
不得开始特征预计算或训练。模型在 PC 导出 ONNX 后回传 Jetson，Jetson
再做数值一致性、至少 2 Hz、内存和 30 分钟稳定性验证。

## 6. Day 11–21 详细执行路线

### Day 11A：控制解耦与 UE5 专家运动学执行

#### 决策

监督标签和运动执行分开。Day 10 的单帧专家标签不要求底层控制器工作；
但为了采到连续闭环状态，Day 12 默认使用 UE5-only 专家 rollout：

```text
/ue/entities
      |
      v
/vla/expert_trajectory  [20,2], dt=0.2 s
      |
      v
expert_kinematic_executor
      |
      v
/ue/kinematic_setpoint  只含最新轨迹第 1 点
      |
      v
ObjectDeliverer JSON -> UE5 直接设置位置和航向
```

这里不遍历整条 20 点轨迹。每个新 UE5 Frame 重新生成专家轨迹，Jetson
以 5 Hz 最多消费该源帧一次，只取 waypoint 0。若 `0.5 s` 没有新专家
输入则发送一次 invalid hold。这样频率和去重由 Jetson 负责，UE5 蓝图
不维护轨迹索引、队列或插值时钟。

固定接口见 `docs/ue5_kinematic_command_v1.md`。关键边界：

- `/ue/kinematic_setpoint` 只允许 UE5 仿真消费；
- `outbound_command_mode=kinematic` 时 bridge 不订阅 thruster command；
- ROS `base_link` 米转换为 UE actor-local 厘米，并执行 Y 轴反号；
- 同一 `(Run_ID, Scene_Seed, Frame_Index)` 只执行一次；
- STOP、断流、非法 shape、NaN 和超步长全部保持位姿；
- UE5 应用位姿后清零物理速度，仍持续发送递增 `Frame_Index`；
- 该路径证明上层轨迹与数据管线，不证明真实控制器可跟踪。

Day 11A 通过条件：

- Jetson `colcon build` 和相关 pytest 通过；
- launch 中不启动 control manager、allocator 或 ESP32；
- `/vla/expert_trajectory` 为 20 点，`/ue/kinematic_setpoint` 为单点；
- UE5 每个 `Sequence` 最多应用一次；
- FOLLOW 单步不超过 35 cm，STOP 与陈旧输入均不移动；
- 运行时 `/ue/thruster_command` 在本 launch 中无 subscriber；
- 保存 ROS topic 和 UE5 位姿变化证据。

当前状态（2026-07-29）：

- Jetson 目标机构建通过：`asv_jetson_interfaces`、`asv_ue_bridge`、
  `asv_vla`、`asv_bringup` 共 4 包；
- Jetson pytest 通过，新增 first-point、STOP、shape、NaN、frame 和
  step-limit fail-closed 测试；
- ROS 合成 20 点专家输入得到单点
  `delta_x_m=0.3, delta_y_m=0, valid=true`；
- TCP bridge 合成探针输出 `Delta_X_Cm=30`、`Delta_Y_Cm=-10`，
  证明米/厘米和 Y 轴反号转换；
- 专用 launch 中 `/ue/kinematic_setpoint` 恰有 1 publisher 和
  1 subscriber，`/ue/thruster_command` 不存在；
- recorder manifest 新增并校验
  `execution_mode=ue5_kinematic_expert_v1`，防止与静态/推进器 Run 混写。

UE5 Blueprint 尚未按新 JSON 契约修改和实测，因此不得把 Day 11A 标为
完成。当前可复现状态见 `docs/HANDOFF_DAY11.md`。

### Day 11B：PC 数据基座、注册表和严格切分

#### 目标

建立一个干净的 PC 训练 checkout 和外部数据根目录。Day 11 不训练，
只保证同一份数据在 PC 可校验、可索引、可按 Run 分组，且不会发生
训练/验证/测试泄漏。

PC 推荐目录：

```text
C:\Users\LIU\Documents\asv_vla_pc\
├── repo\                         # Git checkout，只放代码
└── data\                         # 永不进入 Git
    ├── incoming\                 # Jetson 原始 tar.gz
    ├── extracted\                # 保留原相对路径的数据
    ├── registry\                 # Run 注册表和 split
    ├── features\                 # Day 13 特征缓存
    ├── checkpoints\              # Day 15 模型
    └── reports\                  # 指标、曲线和失败案例
```

不要使用已有的脏 checkout 覆盖数据。新建 checkout：

```powershell
cd C:\Users\LIU\Documents
git clone https://github.com/EnjunLiu/asv-jetson-ws.git asv_vla_pc\repo
cd asv_vla_pc\repo
git switch main
git pull --ff-only origin main
git status --short --branch

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pytest jsonschema pillow numpy pyyaml
```

PC PyTorch 安装必须先记录硬件：

```powershell
nvidia-smi
python --version
```

有 NVIDIA GPU 时按 PyTorch 官方与本机驱动匹配的命令安装；没有 GPU
时先使用 CPU。不要把 PC 的 PyTorch wheel 复制到 Jetson，也不要修改
Jetson 的 NVIDIA PyTorch。

把已经迁移到 PC 的 pilot 复制到外部数据目录并验证：

```powershell
cd C:\Users\LIU\Documents\asv_vla_pc
New-Item -ItemType Directory -Force data\incoming | Out-Null
New-Item -ItemType Directory -Force data\extracted\pilot | Out-Null
Copy-Item `
  C:\Users\LIU\Documents\jetson_ws\pc_datasets\day10_A1D7BAAE49F39E3BB7B1808AB8443CA9.tar.gz `
  data\incoming\

tar -xzf `
  data\incoming\day10_A1D7BAAE49F39E3BB7B1808AB8443CA9.tar.gz `
  -C data\extracted\pilot

$env:PYTHONPATH = "C:\Users\LIU\Documents\asv_vla_pc\repo\src\asv_vla"
python -c `
  "from asv_vla.supervised_dataset import evaluate_main; raise SystemExit(evaluate_main())" `
  data\extracted\pilot\artifacts\day10_supervised\A1D7BAAE49F39E3BB7B1808AB8443CA9 `
  --require-all-labels
```

Day 11 应新增：

- `training/dataset_registry.py`：扫描多个 episode 和 supervision
  manifest，生成 `dataset_registry_v1.jsonl`；
- `training/make_group_splits.py`：只按 Run ID/Scene Seed 分组；
- `training/config/dataset_v1.yaml`：数据根、帧步长和 split seed；
- `training/test/test_group_splits.py`：验证 Run、Scene Seed、Frame
  和语言模板都不泄漏；
- `training/README.md`：PC 命令和目录约定。

切分规则：

1. 同一 Run 的所有帧只能属于一个 split。
2. 同一 Scene Seed 默认只能属于一个 split。
3. 训练、验证、测试使用各自的 Day 3 语言模板族。
4. Primary test 同时 hold out Run 和语言模板。
5. pilot 只有一个 Run，注册表必须输出 `training_ready=false`。

Day 11B 通过条件：

- PC 打印 `DAY11_PC_PILOT_PASS`；
- pilot SHA-256 和 Day 10 记录一致；
- split 测试能主动拒绝同一 Run 出现在两个 split；
- 没有数据、特征或 checkpoint 被 Git 跟踪；
- 保存 PC OS、Python、GPU、驱动、PyTorch 版本到报告。

### Day 12：设计性采集，而不是盲目堆帧

#### 为什么还需要采集

当前 pilot 中红船位置、蓝船位置和颜色是固定绑定的。若直接训练，模型
可能用“近处就是红船”这种位置捷径，而不是看图识别颜色。Day 12 的重点
不是总帧数，而是打破颜色、位置、距离、速度和背景之间的相关性。

#### 最小与推荐采集矩阵

最小工程基线：4 种布局 × 每布局 3 个独立 Scene Seed = 12 Runs。

最终推荐规模：5 种布局 × 每布局 3 个独立 Scene Seed ×
2 种运动状态 = 30 Runs。

建议布局：

| 布局 | 红/蓝设置 | 左/右设置 | 目的 |
| --- | --- | --- | --- |
| L1 | 红近、蓝远 | 左右等距 | 基准 |
| L2 | 红远、蓝近 | 左右等距 | 打破颜色-深度相关 |
| L3 | 红偏左死区内、蓝偏右死区内 | 左右深度不同 | 打破颜色-像素位置相关 |
| L4 | 红蓝交换世界位置 | 左右交换远近 | 强颜色换位 |
| L5 | 增加非目标船或轻度遮挡 | 仍保持四目标可追踪 | 测试干扰与视觉鲁棒性 |

运动状态：

- S0：四个目标静止或恒定低速；
- S1：至少两个目标具有不同的有限恒速。

每个 Run：

- 新 Run ID；
- 每个 Run 使用不同 Scene Seed，不能所有 Run 都为 `12345`；
- 先通信预检一帧，再正式记录；
- 记录 100 帧，约 10 秒；
- 四个目标 ID 唯一、持续可见、`is_target=true`；
- 颜色和位置按布局变化，不能只移动 Actor 但仍复用同一位置变量。

Day 12 默认使用专家运动学 rollout，底层控制器不参与。推荐使用 PC
一键编排器；它会自动查询槽位、启动 Jetson、等待 ready、命令行启动
UE5、完成 100 帧后停止 UE、执行全部 gate、打包、SCP 并验证哈希：

```powershell
# 采下一个槽位
powershell -ExecutionPolicy Bypass -File `
  .\tools\ue5_day12\collect_day12.ps1

# 首次端到端验收通过后，连续采完所有剩余槽位
powershell -ExecutionPolicy Bypass -File `
  .\tools\ue5_day12\collect_day12.ps1 -Count 0
```

手工方式仅用于诊断。Jetson 单 Run 仍只允许一个 launch，由它同时
独占 TCP bridge、专家、运动学执行器和 recorder，避免第二个 bridge
抢占 8080：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 -m training.day12_collection next --data-root .

ros2 launch asv_bringup day12_collect.launch.py \
  slot_id:=L1_S1_R3 layout_id:=L1 motion_state:=S1 scene_seed:=120103
```

每个 Run 通过后立刻打包、复制 PC 并核对 SHA-256。不要等 12 个 Run
全部录完才检查。失败 Run 保留失败日志，但不进入训练注册表。

四个布局的 UE5 参考坐标、12 个固定 Scene Seed、逐 Run 命令和三个最终
gate 见 `docs/DAY12_COLLECTION.md`。记录器会核对实际 Scene Seed；
错误 Seed 会 fail closed，不会写成可训练 Run。

如果只需验证单帧标签或场景矩阵，可用静态/脚本化本船位姿采集；这同样
不依赖底层控制。凡是声称“expert rollout”的 Run，manifest 必须记录
`execution_mode=ue5_kinematic_expert_v1`，不得与静态采集混写。

Day 12 通过条件：

- 最低 12 个合格独立 Run；推荐最终达到 30 个；
- 每 Run 至少 80 个完整帧、9/9 标签、90/90 指令；
- 12 Run 初始 split 固定为 8/2/2；30 Run 固定为 18/6/6；
- 红/蓝在不同 Run 中至少完成一次位置互换；
- 至少 3 个不同 Scene Seed；
- 数据注册表 `training_ready=true`；
- 不把同一 Run 的相邻帧分散到多个 split。

### Day 13：冻结特征缓存与真正的视觉 grounding

#### 必须先解决的防伪问题

当前任务实体张量包含 `color_red/color_blue` 真值。若策略直接读取这两个
维度，它不看图片也能选择红/蓝目标。这样的高分不能证明视觉参与。

固定边界：

- UE5 的 `color` 保留，用于离线专家生成和质量检查；
- 在线学习策略输入必须把实体特征第 14、15 维颜色真值清零；
- `entity_id` 只用于跨模态对齐，不作为可学习语义输入；
- 不允许把 Day 3 的 `action/target_attribute/distance_bucket` 直接输入
  在线策略；
- 在线策略只能收到自然语言 embedding、图像特征、去特权实体几何和
  本船状态。

Day 6 的“全局 + 单目标 crop”不足以区分四个并存目标。Day 13 应增加
每实体视觉 token：

```text
global_visual_token: [576]
entity_visual_tokens: [16,576]
entity_visual_mask: [16]
entity_ids: [16]  # 仅对齐，不进入神经网络
```

每个可见实体使用 Day 4 相机参数把三维中心投影到图像，再取固定
`224x224` crop；仍不需要 bbox。视觉 token 顺序必须与 Day 7
`TaskFeatures.entity_ids` 对齐。投影出界、隐藏、无效或图片损坏时，对应
mask=false、token 全零。

PC 特征缓存 `feature_cache_v1`：

```text
features/<RUN_ID>/
├── manifest.json
├── language.npz        # 90x256，每条指令只算一次
├── frames_000.npz      # 按 Run 分片
└── quality_report.json
```

每个 frame 缓存：

- global visual `[576]`；
- per-entity visual `[16,576]` 和 mask；
- 去除颜色特权后的 entity tensor `[16,16]`；
- ego `[surge_velocity_mps, yaw_rate_radps]` 及有效性；
- expert trajectory `[20,2]`；
- 完整 frame key、源哈希、模型权重哈希和预处理版本。

缓存 key 必须包含：

```text
source_frame_sha256
image_sha256
language_model_id + weights_sha256
visual_model_id + weights_sha256
preprocess_version
feature_schema_version
git_sha
```

PC/Jetson 一致性不要求 bitwise 相等，但固定 20 个样本必须 shape 相同、
数值有限，语言/视觉特征余弦相似度初始门槛为 `>=0.999`。若实测浮点
差异更大，先记录证据再调整容差，不能静默放宽。

Day 13 通过条件：

- `DAY13_FEATURE_CACHE_PASS`；
- 所有缓存可从源数据重建，修改权重或预处理版本会 cache miss；
- 实体颜色维度在 policy input 中始终为零；
- 每实体视觉 token 与 entity ID/mask 对齐；
- 至少 20 个样本通过 PC/Jetson 特征一致性；
- 无图像时整个学习策略输入无效，不伪造有效视觉。

无 NVIDIA PC GPU 的降级：

1. 语言只对 90 条文本编码一次；
2. 视觉特征可在 Jetson 分批预计算后迁移 PC；
3. PC 只训练小型融合头；
4. 不因算力不足删除视觉模态或改用实体颜色真值。

### Day 14：小型单轨迹融合策略

#### 输入与输出

输入：

- language embedding `[256]`；
- global visual `[576]`；
- entity visual `[16,576]` + mask；
- entity geometry `[16,16]` + mask，颜色维度已清零；
- ego `[2]`；
- 所有模态有效性和完整 frame key。

输出：

- `delta_p_xy [B,20,2]`；
- `stop_logit [B,1]`；
- 不输出候选轨迹、候选分数、世界模型状态或推进器命令。

推荐第一版模型，不超过约 2 M 可训练参数：

```text
language 256 -> MLP -> 128
global visual 576 -> Linear -> 128
entity visual 576 -> shared MLP -> 128
entity geometry 16 -> shared MLP -> 64
entity visual+geometry -> masked attention/pooling -> 192
ego 2 -> MLP -> 32
concat -> MLP 256 -> MLP 256
trajectory head -> 20x2 bounded increments
stop head -> 1 logit
```

轨迹头预测每 0.2 s 的增量，经 `tanh` 限制到最大步长，再 `cumsum`
得到累计位移。这样速度上限由结构保证，而不是只靠训练损失。

初始损失：

```text
L = 1.0 * SmoothL1(all waypoints)
  + 0.5 * SmoothL1(endpoint)
  + 0.2 * BCE(stop_logit)
  + 0.05 * trajectory_smoothness
```

损失权重只能依据 validation 调整，test 集永远不能用于调参。

Day 14 应新增：

- `training/model.py`
- `training/dataset.py`
- `training/losses.py`
- `training/config/model_small_v1.yaml`
- shape、mask、NaN、梯度和冻结参数测试

Day 14 通过条件：

- 打印 `DAY14_POLICY_CONTRACT_PASS`；
- batch size 1、2、8 均输出 `[B,20,2]`；
- 任意 invalid mask 不产生 NaN；
- 冻结语言/视觉骨干无梯度；
- policy input 不含结构化任务标签和实体颜色真值；
- 同一输入、同一 seed 重复前向一致；
- 参数量、checkpoint 大小和峰值内存写入报告。

### Day 15：PC 第一版训练和基线

训练只使用 train Run 和 train 语言模板。每个 epoch 对同一个
`(frame_key, task_label)` 随机选择一条 train 同义句，避免同义句复制
主导梯度。帧可使用 `frame_stride=2–5` 降低相邻帧相关性。

固定三个训练 seed：

```text
17, 23, 42
```

必须比较：

1. 全零 STOP baseline；
2. 每标签训练集均值轨迹 baseline；
3. entity-only baseline，颜色维度仍清零；
4. 完整 language+vision+entity+ego policy；
5. 确定性 expert 只作为标签上界，不作为学习结果。

每次训练保存：

```text
checkpoints/<experiment_id>/
├── config.yaml
├── dataset_manifest.json
├── environment.json
├── metrics.json
├── train.csv
├── curves.png
├── best.pt
└── last.pt
```

记录 Git SHA、数据哈希、cache 哈希、seed、epoch、batch size、优化器、
学习率、训练时间和峰值显存。checkpoint 不进入 Git。

主要指标：

- ADE：20 步平均欧氏误差；
- FDE：第 20 步终点误差；
- STOP drift：STOP 轨迹最大位移；
- stop classification precision/recall；
- 红/蓝/左/右/3 m/10 m 各标签指标；
- 速度约束违规率；
- NaN/invalid 计数。

Day 15 通过条件（已全部满足）：

- 三个 seed 都能完整训练和复现；
- held-out validation/test ADE、FDE 至少比“每标签均值轨迹”基线改善
  30%，否则不得进入闭环；
- 95% 以上 STOP 样本最大位移不超过 0.10 m；
- stop F1 不低于 0.95；
- 速度结构约束违规率为 0；
- 三 seed 的 test ADE 标准差被报告，不只展示最好 seed；
- 无 NaN、OOM 或数据泄漏。

若 12 Run 无法通过，先扩充到 30 Run 并检查颜色/位置相关性，不扩大
模型。若完整模型不优于简单 baseline，优先检查数据、mask、特权泄漏和
特征对齐。

最终结果采用 30 Run `18/6/6` split 和 `train_30_v5.yaml`。12 Run
sealed test 失败后已执行扩充；30 Run test 只在最终 validation 通过后
打开一次，并通过全部门槛。Day 15 关闭，不再用 test 继续调参。

### Day 16：语言、视觉和实体干预证明

这是项目能否诚实称为 VLA 的关键日。仅仅把三种特征拼接进网络不算
多模态证据。

必须进行同一 observation 下的语言干预：

- red ↔ blue；
- left ↔ right；
- 3 m ↔ 10 m；
- FOLLOW ↔ STOP；
- 中文未见模板或英文同义表达。

必须进行视觉/实体干预：

- 在 UE5 held-out Run 中交换红蓝颜色但尽量保持几何位置；
- 遮挡或替换某一个实体 crop；
- 将图像特征置零；
- 将 entity geometry 置零；
- 打乱 per-entity visual token 与 entity ID 的对齐；
- 保持图像不变只改语言，保持语言不变只换颜色位置。

报告至少包含：

```text
reports/day16/
├── metrics_by_label.csv
├── intervention_pairs.csv
├── ablation_summary.json
├── failure_cases.jsonl
└── trajectory_plots/
```

Day 16 通过条件：

- 24 个 Day 3 对比对在 held-out Run 上产生正确方向的轨迹变化；
- 红蓝位置互换后仍按颜色而不是固定坐标选择目标；
- 去掉语言、视觉或实体任一关键模态，相关子任务指标出现可解释退化；
- 完整模型优于 entity-only 且 entity-only 无颜色真值；
- 无图像或对齐错误时 fail closed，而不是输出高置信有效轨迹；
- 所有失败按 grounding、视觉、几何、STOP、越界分类。

若完整模型与“无视觉”模型几乎相同，禁止声称视觉 grounding 完成。
此时保留 deterministic expert 和数据管线成果，把学习策略描述为
“language-conditioned trajectory imitation”，并回到 Day 12/13 修复
颜色换位数据或每实体 crop。

#### Day 16 当前执行结果（2026-07-29，未通过）

已实现并冻结：

- `training/interventions.py`：24 对同 observation 语言干预、红蓝换位、
  6 类模态/对齐消融、3 类输入故障 fail-closed 和失败案例/轨迹图；
- `model_small_v3.yaml`：语言选择实体的 attention，不读取结构化任务
  标签、实体 ID 或颜色真值；
- `train_30_v6.yaml`：同 observation 的 9 标签 grouped batch 和成对
  direction/assignment loss；
- 三 seed validation 全部通过：ADE 分别为
  `0.3318 / 0.3869 / 0.3504 m`，STOP F1 均为 `1.0`；
- entity-only seed42 validation ADE 为 `1.5944 m`，完整模型为
  `0.3504 m`，完整模型改善约 `78.0%`。

冻结 validation 报告
`pc_datasets/reports/day16_a7eb7ab_validation_v1` 通过：

- 24 对 × 3 seed = `72/72` 通过；
- L3/L4 颜色换位 `6/6` 通过；
- 去语言、去全部视觉、去实体几何的相关 ADE 分别退化约
  `312.5% / 46.6% / 289.3%`；
- 输入故障 `9/9` 精确零轨迹并 STOP。

随后没有把 validation 结果冒充最终结论，而是自动采集了 8 个全新
Scene Seed。一次性报告
`pc_datasets/reports/day16_84e63bc_fresh_holdout_v1` 的 SHA-256 为
`5feee98aa3bf7a57e82b1409ca50904d0d692926e9e2537f784e0e1f1427640f`：

- 语言 `72/72`、三类消融和 fail-closed 通过；
- 颜色换位失败；
- 追查发现采集 rollout 默认跟随红船，L3/L4 中 32/34 个红色样本的
  专家轨迹差异仅约 `3e-6`，因此该颜色子夹具被判定为动态混杂，失败
  报告保留但不作为模型通过/失败的最终颜色证据。

为消除混杂，提交 `7e4680b` 将新 R14 采集改为 STOP：

- expert 节点持续发送有效 `hold_position=true` 零位移，异常退出会使
  整次 launch 失败；
- 两 Run 共 200 帧、18000 样本、9/9 标签、0 缺口；
- 红/蓝 34 个采样帧的最小专家轨迹差异分别为
  `12.7914 / 14.7737`，颜色换位题目在全程可判定；
- 数据、split 和 feature manifest SHA-256 分别为
  `c252d167... / 987fc959... / 267125c6...`。

真正有效的一次性 R14 报告
`pc_datasets/reports/day16_d9971e5_color_confirmation_v1` 的 SHA-256 为
`bb6beeeb269b0f53d3a5ed49f495deeb6427393cdc57866a3e9c43775d72e820`，
结果必须按失败处理：

- 语言干预 `61/72` 通过，未达到 `72/72`；
- 红色换位 `2/3` seed 通过，蓝色换位 `0/3`，总计 `2/6`；
- 去语言、去视觉、去实体几何仍分别造成约
  `197.2% / 13.7% / 218.2%` ADE 退化；
- fail-closed `9/9` 通过。

因此当前只能说“模型会使用多模态输入”，不能说“颜色 grounding 已
完成”。R14 已开封，禁止加入训练、调阈值或再次充当最终 test。下一轮
必须只使用新 train/validation counterfactual Run 修复；修复后再采
全新 Scene Seed 留出对。Day 16 保持进行中，Day 17 尚未开始。

### Day 17：确定性轨迹安全门

学习策略发布建议话题：

```text
/vla/policy_trajectory
```

安全门是 `/vla/selected_trajectory` 的唯一发布者。专家标签节点仍只
发布 `/vla/expert_trajectory`，两者均不能直接控制 ESP32。

安全门检查顺序：

1. 模态 valid、完整 frame key 和新鲜度；
2. shape、frame、dt、horizon、NaN/Inf；
3. 单步速度、总位移、曲率和控制可实现性；
4. 用实体相对速度进行常速度占据外推；
5. 轨迹与动态占据的最小距离；
6. STOP、软拒绝、硬 E-STOP 状态机；
7. 发布原因码和可复现日志。

建议原因码：

```text
PASS
POLICY_STOP
STALE_INPUT
INVALID_MODALITY
INVALID_SHAPE
NONFINITE
SPEED_LIMIT
CURVATURE_LIMIT
COLLISION_RISK
CONTROL_UNREACHABLE
ESTOP
```

无健康 ego 状态时不能生成“有效慢停”，必须输出 invalid 零命令。
健康状态下可生成确定性减速轨迹，但不能把零位移 `valid=true` 当通用
回退。

Day 17 通过条件：

- `DAY17_SAFETY_GATE_PASS`；
- `/vla/selected_trajectory` 运行时只有一个 publisher；
- 每个原因码都有单元测试或 ROS probe；
- 静态与运动障碍碰撞轨迹必拒绝；
- stale、NaN、shape 错误和模型异常必 fail closed；
- 通过轨迹不被数值改变，拒绝轨迹具有确定性结果；
- 不产生 `DecisionOutput`、wrench 或推进器副作用。

### Day 18：安全轨迹到二维控制边界

Day 18 是真实底层控制路径，和 Day 11A 的 UE5 运动学执行器并行存在但
绝不同时运行。Day 11A 用于数据采集和仿真上层验收；Day 18 才评估
`desired_x/y` 是否能被控制器跟踪。底层参数未调好不阻塞 Day 11–17。

控制桥只消费 `/vla/selected_trajectory`。每次重规划只执行前
`0.2–0.5 s` 的短前缀，然后等待新轨迹，不能一次盲目执行 4 s。

输出仍是：

```text
DecisionOutput.desired_x
DecisionOutput.desired_y
DecisionOutput.valid
```

禁止学习策略、安全门或控制桥发布左右推进器值。现有 control manager、
allocator 和 ESP32 保持独立。

控制桥必须处理：

- 正常 FOLLOW 前缀；
- STOP；
- 安全门拒绝；
- ego 超时；
- 轨迹超时；
- 重规划中断；
- legacy/vla 模式切换；
- 重复 frame key。

Day 18 通过条件：

- `DAY18_TRAJECTORY_CONTROLLER_PASS`；
- 只执行最新安全轨迹的短前缀；
- safe_stop/invalid 产生 `DecisionOutput.valid=false`；
- `desired_x/y` 有限、在控制边界内；
- 旧 ESP32 消息与控制器接口无修改；
- Day 1 的 invalid-zero-wrench/thruster 回归仍通过。

### Day 19：UE5 学习策略闭环

新增独立 `vla_full_system.launch.py`，明确：

```text
mode:=legacy  # 旧正式路径
mode:=vla     # 学习策略 + 安全门 + 轨迹控制桥
```

两个模式不能同时发布共享控制话题。smoke、formal、vla launch 不能
并行运行。

正式闭环场景至少包括：

1. follow red 3 m；
2. follow blue 10 m；
3. follow left 3 m；
4. follow right 10 m；
5. FOLLOW 中途切换 STOP；
6. 红蓝位置互换；
7. 数据断流；
8. 人工注入不安全轨迹。

必须使用未进入训练集的至少 3 个 Scene Seed，每个任务连续运行
60 s。记录目标选择、距离误差、轨迹、安全门原因、控制输出和连接状态。

Day 19 通过条件：

- `DAY19_UE5_CLOSED_LOOP_PASS`；
- 目标选择与指令一致；
- FOLLOW 不持续发散，距离误差统计完整；
- STOP 后不继续向目标前进；
- 断流和不安全轨迹进入预期回退；
- 没有重复 publisher；
- 不把 UE5 仿真结果描述成真实感知或实船海试。

若学习策略闭环不稳定，保留离线 checkpoint 和失败日志，使用
deterministic expert 做系统对照，不允许绕过安全门直接演示。

### Day 20：ONNX、Jetson 部署和压力测试

PC 导出：

```text
best.pt -> policy.onnx
```

ONNX 输入必须包含固定 shape 和 mask。导出报告保存 opset、PyTorch
版本、checkpoint SHA-256、ONNX SHA-256 和测试向量。

先在 PC 对 100 个固定样本比较 PyTorch/ONNX：

- shape 完全相同；
- 无 NaN/Inf；
- 轨迹最大绝对误差和余弦相似度写入报告；
- STOP 分类一致。

Jetson 先用 ONNX Runtime 或兼容 PyTorch 路径验证，再决定是否构建
TensorRT。TensorRT engine 必须在目标 Jetson 生成，不从 PC 复制 engine。
如果 TensorRT 转换耗时过高但 PyTorch/ONNX 已达到 2 Hz，TensorRT 可降
为 P2 优化，不阻塞最终演示。

资源测试：

```bash
tegrastats
free -h
ros2 topic hz /vla/policy_trajectory
```

故障注入：

- 图像空包/损坏；
- entity NaN 或重复 ID；
- language embedding invalid；
- ego 超时；
- UE5 断开和重新连接；
- policy 推理异常；
- 安全门碰撞拒绝；
- ESP32/下游反馈超时。

Day 20 通过条件：

- Jetson 策略端到端至少 2 Hz；
- 连续 30 分钟无 OOM、无持续内存增长；
- 温度、功耗、RAM、GPU 利用率和最大延迟有日志；
- PC PyTorch、PC ONNX、Jetson 输出在冻结容差内一致；
- 每个故障进入预期状态并产生原因码；
- 重连后不复用过期轨迹。

### Day 21：归档、演示和最终边界

不再增加架构。只修复阻塞复现、演示或安全的缺陷。

最终仓库应包含：

- 完整 README 快速开始；
- `TODO.md` 全部状态和证据；
- 接口与架构图；
- PC 数据/训练说明；
- 模型卡：数据、指标、限制和伦理边界；
- 数据清单及哈希，不包含原始大文件；
- checkpoint/ONNX 下载位置和 SHA-256，不提交二进制；
- held-out 指标、干预、消融和失败案例；
- Jetson benchmark 与 30 分钟日志摘要；
- 3–5 分钟演示脚本和视频；
- `KNOWN_ISSUES.md`；
- 最终 Git tag，例如 `v0.1.0-ue5-demo`。

最终演示顺序：

1. 展示 Jetson 先监听、UE5 后 Play；
2. 红/蓝颜色换位并切换语言；
3. 3 m/10 m 距离切换；
4. FOLLOW 切换 STOP；
5. 注入危险轨迹，由安全门拒绝；
6. 断开 UE5，系统 fail closed；
7. 展示日志、指标和模型限制。

Day 21 通过条件：

- 新环境能按 README 校验一个数据包并完成离线推理；
- Jetson 能按单一 launch 完成 UE5 闭环；
- commit、tag、配置、数据哈希、模型哈希和日志互相可追踪；
- README 明确只在 UE5 仿真验证；
- 不声称开放世界、多船避障、真实视觉泛化或实船海试；
- 已知失败不隐藏，无法通过的 DoD 标记为未完成。

## 7. 优先级、降级路线和停止规则

### P0：必须完成

- 30 Run 数据集及严格 split（已完成）；
- 小型策略优于简单 baseline（Day 15 sealed test 已完成）；
- STOP、FOLLOW 和语言干预；
- 唯一安全门；
- `desired_x/y` 控制边界；
- UE5 held-out 闭环；
- fail-closed 和复现文档。

### P1：让“VLA”说法可信

- 30 Run 推荐数据集（已完成）；
- 颜色/位置换位；
- 每实体视觉 token；
- 实体颜色真值对 policy 屏蔽；
- 视觉、语言和实体消融；
- 三 seed 统计。

### P2：有时间再做

- TensorRT 极致优化；
- 更复杂遮挡和非目标船；
- 精美 UI、视频剪辑和自动报告；
- 更多语言和场景。

明确不做：

- 六候选轨迹；
- 候选评分器；
- 学习型世界模型；
- 端到端推进器输出；
- 大规模在线强化学习；
- 用系统 ID/固定位置偷学颜色；
- 在 test 集调参；
- 为追指标绕过安全门。

停止/回退规则：

1. 12 Run 后训练不优于 baseline：先查泄漏、对齐和数据，再扩到 30 Run；
   不先扩大模型。
2. 去图像后指标不变：不能声称视觉 grounding；回 Day 12/13。
3. PC 无 GPU：Jetson/CPU 预计算冻结特征，PC 只训练小头。
4. Jetson OOM：减 batch、缓存和融合头宽度，不删除模态或安全接口。
5. ONNX/TensorRT 不稳定：保留达到 2 Hz 的 PyTorch/ONNX 路径。
6. 学习闭环不稳定：expert 作为对照，学习策略保持离线，不绕过安全门。
7. 时间不足：完成 P0 并诚实缩小结论，不同时开新研究方向。

## 8. DeepSeek 或其他 Agent 接管清单

新 Agent 必须先读：

1. `TODO.md`
2. `docs/interfaces.md`
3. `README.md`
4. `docs/ue5_kinematic_command_v1.md`
5. PR #12 及之后的 PR
6. 当前 Jetson 和 PC 的 `git status --short --branch`

已知真值：

- GitHub：`EnjunLiu/asv-jetson-ws`
- Jetson checkout：`/home/jetson/jetson_asv_ws`
- Day 10 合并基线：`3748164`
- Day 11–21 路线图合并提交：`ddc1489`
- 当前控制解耦分支：`feature/day11-kinematic-executor`
- Day 10 pilot Run ID：
  `A1D7BAAE49F39E3BB7B1808AB8443CA9`
- pilot 数据包 SHA-256：
  `621b96dae5791dd1965e7acefe80891dfd7d39579a84971526a29306962eccd4`
- PC pilot 包：
  `C:\Users\LIU\Documents\jetson_ws\pc_datasets\`
  `day10_A1D7BAAE49F39E3BB7B1808AB8443CA9.tar.gz`

接管后第一组命令：

```bash
# Jetson
cd /home/jetson/jetson_asv_ws
git switch main
git pull --ff-only origin main
git status --short --branch
git log -5 --oneline
```

```powershell
# PC
cd C:\Users\LIU\Documents\asv_vla_pc\repo
git switch main
git pull --ff-only origin main
git status --short --branch
```

Agent 不得假设：

- Windows 旧 checkout 是干净或最新的；
- 静态测试等于 Jetson ROS 实测；
- `latest` 符号链接永远指向目标 Run；
- 4500 配对等于 4500 个独立场景；
- 图像被拼进模型就等于模型使用了图像；
- 有效零位移等于通用安全停止；
- 模型下载完成等于策略集成完成。

每个 Day 使用完整 Git 流程：

```text
同步干净 main
-> feature branch
-> 最小实现
-> 单元测试
-> PC/Jetson 对应运行证据
-> git diff --check
-> 明确暂存文件
-> commit/push
-> draft PR
-> 用户合并
-> 同步 main 并删除分支
```

Agent 每次汇报必须区分：

- 已实现；
- 静态/PC 已验证；
- Jetson 已验证；
- UE5 闭环已验证；
- 尚未验证或失败。

任何密码、令牌、模型私有地址和机器私密配置都不能写入仓库、日志或 PR。

## 9. UE5、Jetson 与 PC 分工

用户负责：

- UE5 蓝图和场景逻辑
- 从 UE5 发送相机、本船、目标和障碍数据
- UE5 运动学模式中按
  `docs/ue5_kinematic_command_v1.md` 执行单个相对位移命令
- UE5 运动学模式不施加左右控制力，并对每个 `Sequence` 只应用一次
- 按 Day 12 矩阵改变布局、Scene Seed 和目标运动
- 审查并合并每个阶段 PR

Jetson 负责：

- 定义和校验接收字段
- TCP/ROS bridge、时间戳、坐标变换和数据记录
- 实际 ROS 节点、回放、推理、单轨迹策略
- 轨迹安全门、二维控制边界、ROS 健康状态
- Jetson 构建、测试、资源和故障证据

PC 负责：

- 数据包接收、SHA-256 和 registry；
- 按 Run/Scene Seed 切分；
- 冻结特征预计算；
- 小型策略训练和三 seed 评估；
- 图表、报告、checkpoint 和 ONNX 导出；
- 不在 PC 修改 Jetson 专用 NVIDIA PyTorch 环境。

UE5 不需要 bbox。实体中心通过冻结相机模型投影得到 crop。缺失字段必须
显式 `valid=false`，不得伪造有效观测。

## 10. 最终 Definition of Done

必须同时满足：

- 新自然语言能被边缘模型编码并缓存。
- 同一 UE5 场景切换目标、距离或 STOP 时，轨迹有可测变化。
- 图像和任务实体都进入策略，且有干预证据证明不是摆设。
- 红蓝换位后策略按视觉颜色而不是固定位置选择目标。
- 训练、验证、测试 Run ID 和 Scene Seed 无泄漏。
- 策略直接输出单条 `[20,2]` 二维位移轨迹。
- 轨迹安全门是唯一最终安全轨迹发布者。
- 不安全轨迹被拒绝，并进入确定性减速或 E-STOP。
- 轨迹控制器只向现有边界提供 `desired_x / desired_y`。
- PC checkpoint、ONNX 和 Jetson 推理在冻结容差内一致。
- Jetson 策略至少 2 Hz，30 分钟无 OOM。
- 全过程保存 commit、配置、模型/数据哈希、日志和复现命令。
- README 明确这是 UE5 仿真结果，不声称真实感知或实船海试完成。

如果上述任何一项没有证据，只能标记为未完成，不能用“接口已预留”
或“模型已下载”替代验收。
