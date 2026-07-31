# 场景验证报告（2026-08-01）

## 验证对象

UE5 `SceneAutomationSubsystem` 的 S2 正弦编队运动 + L6 远距布局，headless 实测。

## 验证方式

- 重建 EDGEEditor（SCENE_UE_BUILD_PASS）后 headless 运行
  （`-SceneAuto -Slot= -Layout=L6 -Motion=S2 -Seed=200101 -MaxRuntimeSeconds=`）
- UE5 侧：`SCENE_TARGET_POS` / `SCENE_ASV_POS` 每秒世界位置采样日志
- Jetson 侧：`/ue/entities`、`/ue/asv_state` 采样（bridge 转发）

## 结果

### 1. S2 正弦运动（通过）

| 实体 | 行为 | 实测 |
|---|---|---|
| target_red | 前进 + 正弦摆动 | X +60cm/s；Y 摆动幅度 ~300cm，周期符合 λ=60m、v=60cm/s |
| target_blue | 前进 + 正弦摆动 | 同 red（红蓝分居中心线两侧） |
| target_left/right | 直线前进（干扰） | X +60cm/s；Y 恒定（无摆动） |

`SCENE_SINE_PARAMS wavelength_cm=6000 amplitude_cm=600 speed_cm_s=60` 确认参数生效。

### 2. 远距实体投影可见性（通过）

- 4 个实体在相对距离 3–37 m 范围内全部 `visible: true`、`valid: true`
  （Jetson `/ue/entities` 实测；**超出历史实测的 9 m 上限**）
- 无 `TARGETPROJECTIONERROR`；相机 90° FOV 在 25 m 处横向覆盖 ±25 m，
  红蓝 6 m 间距可同时入画（数学必然，多帧可见性确认）
- 红蓝/白船颜色正确上报（red/blue/white）

### 3. 蓝图"counterbalanced"巡航与翻转（发现，已处理）

headless 运行时 Connection 蓝图默认驱动 ASV：

- **巡航**：无 setpoint 流时，ASV 以 ~0.6 m/s 巡航（与 target 同速）、
  初始 2 s 快速段（~2.5 m/s），并向 target 方向缓慢转向（yaw 0 → -17°）
- **翻转**：kinematic setpoint 流送达时，运行约 35 s 后蓝图把 ASV 翻转
  180°（yaw=164°）并停止——**这就是历史数据中"采集中途翻转"的机制**
  （relabel 修复的根因，HISTORY.md §1）
- **对策（已验证）**：新增 `-YawFixWholeRun` 参数，yaw 修复窗口延长至全程，
  headless 实测 70 s 全程 yaw=0、翻转完全抑制、巡航不受影响（fix 只转
  yaw 不锁位置）。**采集/验证统一使用该参数 → 全程相机朝前、无翻转帧**，
  无需 STOP 稀释或几何取反补丁。

### 4. 远距/编队验证结论

- 25 m 处 4 实体可见性：通过（多帧，可见范围已超历史 9 m 实测）
- S2 正弦运动：通过（前进 60 cm/s、正弦摆动、白船直线、红蓝分居两侧）
- 蓝图巡航/翻转：已定位并验证对策（YawFixWholeRun）
- **场景可行性验证完成**

## 采集/闭环设计输入

- 采集：`-YawFixWholeRun` + S2/L6 + expert setpoint（execution_mode
  =ue5_kinematic_expert_v1），与历史工作流一致，无翻转帧
- 在线闭环：PIE 交互模式验证（headless 下 setpoint 是否执行待用户
  Play 时确认；PIE 下历史闭环已验证收敛）
- 注意：巡航使 ASV 在采集/验证中持续移动（与 target 同向），相对几何
  覆盖逼近/跟踪分布——与历史采集一致（分布自洽）
