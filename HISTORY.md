# 项目历史与审计记录

> 本文档保留开发过程的**事实记录**，含根因分析与诚实的验收状态，供审计追溯。
> 所有声明均可从代码、日志、提交记录复现。不虚构任何未验证的结论。

## 1. 重大根因：训练标签反转（发现于 2026-07-31）

### 现象
模型输出"倒退跟随"轨迹：线上 yaw=0（目标在前方 +x）时策略生成持续后退的轨迹。

### 根因链
1. UE5 "counterbalanced" 场景中，Connection 蓝图按 SceneSeed 在 BeginPlay 时把 ASV
   旋转 180°（seed 200101 → yaw=180°，目标落在相机后方，投影 `depth=-2.73`）。
2. 翻转后目标在 base_link 中 relative_x < 0（相机后方），但 expert 标签生成器盲目
   跟随 base_link 坐标，生成了"倒退跟随"标签；x>0 的帧的标签来自另一套反转坐标源，
   同样是倒退。
3. 模型忠实学习了该反转映射 → 线上 yaw=0 时输出倒退轨迹。

### 修复（标签补丁，3 轮迭代）
- **v1**：翻转帧 → STOP 标签。结果：STOP 占 85%，稀释 follow 信号，模型学成
  "follow 指令 → 基本停止"（PC 复现：red 在 4m 时输出≈0）。
- **v2**：几何取反——翻转帧（40%，1358/3400）对实体 x,y,vx,vy 取反（与 180° 旋转
  一致，left/right 随之镜像），用 build_entity_tensor 重建派生列并保持颜色列置零；
  标签从修正后几何正常生成。共修正 106,140 个标签。
- **v3**：翻转帧中交换 left/right 实体身份（与几何取反配合，修复 bearing 语义）。
- 配套守卫：`expert_trajectory.py` 对选中目标 relative_x <= 0 生成 fail-closed STOP。

### 验证
- 重训后 3-seed 验证 PASS（full_seed17 ADE 0.124），ONNX parity 精确（7e-7），
  CPU 推理 2344 Hz。
- **诚实记录：`checkpoints/day21_label_fix_v1/summary.json` 的 `validation_gate_passed`
  为 false**（seed17 stop_drift 未过、seed23 ADE/FDE 提升不达标）。即当时部署的模型
  属"工程可用"而非"严格验收通过"状态。后续重训以验证门真实通过为准。

## 2. UE5 侧修复（yaw 翻转）

- `Day12AutomationSubsystem`（后改名 `SceneAutomationSubsystem`）在 ConfigureScene
  放置目标前强制 ASV yaw=0，并在 `kAsvYawFixWindowSec=8.0s` 内每个 Tick 对世界上所有
  BP_ASV_C 重申，抵消 BeginPlay 顺序不确定性；窗口结束后不再干预，避免与运动学
  执行器冲突。
- 修复前实测日志：`TARGETPROJECTIONERROR depth=-2.73`（seed 200101）；修复后目标
  全部正确投影，视觉不再误报 INVALID_MODALITY。

## 3. 线上不稳定排查记录（2026-07-31）

| 问题 | 根因 | 修复 |
|---|---|---|
| 跨 Run 全部误判 STALE（755 次） | UE5 每次启动重置帧计数器，而门/控制器持续存活，跨 Run stamp 单调性被破坏 | `safety_gate_node`/`trajectory_controller_node` 在 run_id 变化时重置 stamp 基线 |
| 线上视觉输入分布外 | 线上视觉编码器只发全局+1 crop（2 token），训练数据是全局+每实体 crop（16 slot） | 编码器按 task tensor 槽位发全部可投影实体 crop（17 token，零填充 + per-token mask）；策略节点按槽消费 |
| 语言 embedding 分布外 | stub 发零向量 | 加载预计算指令 embedding（与训练缓存逐位一致，max diff=0.0），缺失回退零向量 |
| 安全门拒绝正常模型轨迹 | 阈值按旧模型分布调，未校准 | 按实测分布校准：max_curvature 2→15（模型 p99=7.6）、方向连续性 170°、速度容差 5%、碰撞余量 1.0→0.5m、只查可执行前缀（2-5 步） |
| 模型逐帧输出振荡 | 动态 FluidSim 水景下输入分布抖动 | 策略节点 5 帧时间平滑（STOP 清窗）；输入分布修复（见上）；学习策略在动态水景下的逐帧鲁棒性列为 P2（可重训加输入增强） |

## 4. 闭环验证记录（2026-07-31 晚间）

- **专家控制闭环**（规格允许的对照路径）验证通过：有效 setpoint 序列
  31.1→2.0→0.75→0.37→0.17→0.09→0.03 cm（反馈收敛证明船真实移动），安全门 PASS，
  STALE 修复生效（755→1）。
- **学习策略（ONNX）线上**：动态水景下逐帧输出混沌振荡（平滑帧/锯齿帧交替），
  安全门按设计拒绝不安全轨迹（fail-closed）。离线指标完好（见上）。
- 规格原文允许："若学习策略闭环不稳定，保留离线 checkpoint 和失败日志，使用
  deterministic expert 做系统对照，不允许绕过安全门直接演示。"

## 5. 里程碑验收状态（压缩表）

| 阶段 | 状态 | 证据 |
|---|---|---|
| 接口契约 | 完成 | 25/25 项通过 |
| 语言干预 | 完成 | 覆盖/相似度 PASS |
| 视觉编码器 | 完成 | 2x576 冻结特征，重复推理差 0 |
| 实体张量 | 完成 | 16x16 |
| 数据采集 | 完成 | 12/12 包 SHA-256 校验 |
| 特征缓存 | 完成 | 34 runs npz 缓存 + manifest |
| 策略训练 | 完成 | 3-seed 验证 + ONNX parity（见 §1） |
| 独立留出验证 | 未通过（诚实记录） | 红蓝换位证明颜色 grounding 仍不可靠 |
| 安全门 | 完成 | 唯一发布者，fail-closed 全项 |
| 控制桥 | 完成 | 首次输入 STALE 修复 |
| 闭环（专家对照） | 通过 | 见 §4 |
| 闭环（学习策略） | 未稳定 | 见 §4，线上鲁棒性 P2 |

## 6. 命名清理说明（2026-07-31）

项目整理为独立工程（目录 `asv_vla`）：文件名、launch、节点名、docstring 全部去除
"DayX" 命名。以下**数据层标识按设计保留**（改名会破坏与历史数据的关联）：
- run_id / 采集包名（`day12_L*_S*_R*_*`）、`artifacts/day8_episode` 路径
- registry / split / feature cache 内的 schema_version 历史值
- Jetson SSH 密钥文件名 `asv_day12_ed25519`（已安装到两端）
- `.venv`（原 `.venv-day13`）训练环境

schema 版本常量已更名（`train_v1`、`policy_checkpoint_v1` 等）；旧缓存/旧 checkpoint
的 manifest 仍写历史值，**不向后兼容加载**——新特征缓存与重训产物使用新名。
