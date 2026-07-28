# Jetson 无人船 VLA：21 天执行计划

更新日期：2026-07-28
目标平台：Jetson Orin Nano 8 GB、ROS 2 Humble
当前研究范围：FOLLOW、STOP、UE5 仿真、单条二维轨迹、确定性安全回退

本文件是 Jetson 端唯一的执行计划和验收清单，必须纳入 Git
版本控制。UE5 蓝图由用户单独研究和实现；本计划只定义 Jetson
需要接收的数据契约，不修改或假设具体蓝图实现。

## 1. 结论与固定边界

### 1.1 当前架构

```text
自然语言 ──> 冻结语言编码器 ─┐
图像 ──────> 冻结视觉编码器 ─┼─> 单轨迹策略 ─> 轨迹安全门 ─> 轨迹控制器
任务实体 ──> 任务特征编码器 ─┘       [20,2]                  |
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
- 禁止用“`desired_x=0, desired_y=0, valid=true`”冒充通用安全停止，
  因为底层可能把它解释为位置保持。

详细字段见 `docs/interfaces.md`。

## 2. 当前状态

基线提交：`8537e53`
工作分支：`refactor/day1-3-direct-trajectory`

| 阶段 | 状态 | 当前证据/缺口 |
| --- | --- | --- |
| Day 1 | 已完成 | 单轨迹契约、19 项单元测试、25 项 ROS 探针全部通过 |
| Day 2 | 已完成 | Qwen CUDA 离线评估和真实 ROS 节点探针全部通过 |
| Day 3 | 已完成 | 90 条指令、24 个冲突场景；生成一致性和覆盖测试通过 |
| Day 4 | 已完成 | 实体、坐标、运行元数据和相机契约实测冻结；合成报文 validator 通过 |
| Day 5–21 | 未开始 | 不以 stub、空文件或下载完成冒充实现完成 |

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
- `src/asv_vla/test`：19 项测试通过

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
| 10 | 批量采集入口 | recorder、episode manifest、失败日志 | scene_seed 可追踪；原始图像与结构数据都保留 |
| 11 | 数据切分与缓存 | 固定 split、语言/视觉缓存、哈希 | 无 scene_seed 泄漏；在线/缓存特征一致 |
| 12 | 单轨迹策略 | 轻量融合模型、参数报告、前后向测试 | `[B,20,2]`、无 NaN；冻结骨干无梯度 |
| 13 | 第一版训练 | checkpoint、配置、曲线、固定 seed | 验证 ADE/FDE 改善；STOP 不持续前进 |
| 14 | 离线语言/视觉干预 | 干预轨迹、held-out 指标、失败清单 | 换语言能改变目标/距离/停止；去视觉有可解释退化 |
| 15 | 轨迹安全门 | 单一最终发布者、硬约束、状态机、测试 | 碰撞/超限/超时必拒绝；日志给出明确原因 |
| 16 | 轨迹控制桥 | safe trajectory→`desired_x/y` 滚动执行 | 只执行前 0.2–0.5 s；旧 ESP32 边界不变 |
| 17 | UE5 闭环 | 独立 VLA launch、60 s 日志 | 至少 3 个 scene_seed 不发散；断流进入回退 |
| 18 | Jetson 部署 | ONNX/TensorRT 或 PyTorch 混合部署、benchmark | 策略至少 2 Hz；无 OOM；engine 在目标机生成 |
| 19 | 定向修复 | grounding/视觉/坐标失败的最小修复 | 至少解决一个高频失败；STOP 和安全测试不退化 |
| 20 | 压力与演示 | 30 min tegrastats、故障注入、演示脚本 | 无持续 OOM；所有故障进入预期状态 |
| 21 | 缓冲与归档 | 已知问题、最终 tag、复现说明 | 不增加新架构；他人可按 README 完成回放/仿真 |

## 6. UE5 与 Jetson 分工

用户负责：

- UE5 蓝图和场景逻辑
- 从 UE5 发送相机、本船、目标和障碍数据
- UE5 中执行 Jetson 返回的控制结果

Jetson 负责：

- 定义和校验接收字段
- TCP/ROS bridge、时间戳、坐标变换和数据记录
- 语言、视觉、任务实体、单轨迹策略
- 轨迹安全门、二维控制边界、ROS 健康状态
- Jetson 构建、测试、资源和故障证据

在 Day 4 接口冻结前，Jetson 不假定 UE5 已经能发送多实体、bbox 或
障碍速度；缺失字段必须显式 `valid=false`，不得伪造有效观测。

## 7. 最终三周 Definition of Done

必须同时满足：

- 新自然语言能被边缘模型编码并缓存。
- 同一 UE5 场景切换目标、距离或 STOP 时，轨迹有可测变化。
- 图像和任务实体都进入策略，且有干预证据证明不是摆设。
- 策略直接输出单条 `[20,2]` 二维位移轨迹。
- 轨迹安全门是唯一最终安全轨迹发布者。
- 不安全轨迹被拒绝，并进入确定性减速或 E-STOP。
- 轨迹控制器只向现有边界提供 `desired_x / desired_y`。
- Jetson 策略至少 2 Hz，30 分钟无 OOM。
- 全过程保存 commit、配置、模型/数据哈希、日志和复现命令。
- README 明确这是 UE5 仿真结果，不声称真实感知或实船海试完成。

如果上述任何一项没有证据，只能标记为未完成，不能用“接口已预留”
或“模型已下载”替代验收。
