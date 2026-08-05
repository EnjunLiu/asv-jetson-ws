# 交接文档（2026-08-05，单步闭环重构）

> 本文档保留上一轮三场景实验的历史证据；当前实现契约以本节和
> `ARCHITECTURE.md`、`docs/interfaces.md` 为准。用户已拍板将 VLA 拆成感知头和
> 决策头，并直接在线闭环输出单个期望位移点。

## 当前活动契约

```text
图像 + 任务 embedding
  -> 感知头 -> 结构化实体（颜色/相对位置/相对速度）
  -> 决策头(language + entity_geometry + previous_action + validity masks)
  -> 一个 [desired_x, desired_y]
  -> safety gate -> point controller -> UE5
```

- 感知头接收图像和任务 embedding；相对速度由跨帧 tracker 计算，不能从单帧图像伪造。
- 决策头不接收 global visual、entity crop、ego 或专家/真值实体；它不输出轨迹序列。
- 训练在 Windows PC PyTorch/CUDA 上完成。每个 `(Run, instruction, frame)` 是一个单步
  专家点；相邻前帧的 `previous_expert_action` 只在同一 Run、同一 instruction、前帧
  有效且非 STOP 时有效。
- 在线 `previous_action` 只来自上一帧实际通过 safety gate 的动作。首帧、任务/Run 切换、
  帧不连续、gate 拒绝或延迟结果无法对应时，清零并置 `previous_action_valid=false`。
- Jetson 只做部署、推理和闭环验收，禁止运行 `training.train` 或任何训练脚本。

## 1. 本会话完成的事（诚实清单）

### ✅ 已完成并验证

1. **在线 guard 改造为安全兜底**（`visual_standoff_guard.py`）：
   - 跟随指令 + 目标可见 → **策略第一步原样执行**（POLICY_DRIVEN）；
   - 策略第一步背离目标 >90° 或冻结而站距误差大 → 确定性径向步覆盖（STANDOFF_BACKSTOP）；
   - 目标缺失 → fail-closed（VISUAL_TARGET_MISSING，不变）。
   - 新增：**死区保持**（|站距误差|≤0.15m 时第一步归零——验证场的保持语义）。
2. **4m 数据扩展**（无需重新采集）：
   - `instructions.jsonl`：130 条（3m/4m/10m × 红/蓝/左/右 + stop），4m 族镜像 3m 族模板；
   - `expert_trajectory.py`：`distances` 加 `"4m": 4.0`；`REQUIRED_LABELS` 加 4m；
   - 24 runs 监督重生成（`tools/rebuild_supervision_v16.py` → `day10_supervised_v16/`，13000 samples/run）；
   - 特征缓存 v6：`features_near_rgb_v6`（24 runs / 307080 samples / 16-4-4，感知=calibrated_v5）；
   - 新注册表/拆分：`combined_v6_registry_v1.jsonl` + `combined_v6_split_v1.json`。
3. **统一策略 v6 训练**：`checkpoints/near_rgb_v6/`（seeds 29/23/42，git-sha near-rgb-v6-24）。
   - 红/左/右/10m/红4m 全部优秀（red 4m ADE 0.46-0.72；red 3m 0.61-0.71）；
   - **验证门未过**（提升 19-27% vs 要求 30%）：瓶颈是**蓝色聚合**（ADE 2.0-2.6 vs 均值 2.2）。
     根因：蓝感知几何噪声（v5 npz 蓝 RMSE 2.44m，红 0.19m）+ bluew2 验证 run 中 R4 的
     蓝注意力不稳定（R3 蓝 ADE 0.13m，R4 3.4m——同模型同场景族，run 间不稳定）。
   - 结论：**这是项目已知的蓝感知质量边界**（你的 v3/v4/v5 尝试均未过门），非本次回归。
4. **感知选择**：`tools/perception_blue_check.py` 离线评估 v3/v5/v6：
   - **v5 胜出**：红 98.8%/0.19m + 蓝 97.3%/2.44m（v3 蓝仅 41.5%，v6 红几何退化 0.67m）。
   - v5 npz 已部署 Jetson。缓存 v6 与在线同用 v5（一致性）。
5. **在线链路修复**（v5 栈卡死根因，全部有测试）：
   - `entity_wait_sec` 0.25→0.5（visual_encoder 等待窗口，v5 感知 110ms 延迟下 0.25 会饿死大部分帧）；
   - 执行步长上限 `MAX_DESIRED_M` 3.0→**0.12m**（验证场运动学包络 ≤0.15；策略的 0.3m 大步 + ~1s 延迟 → 过冲 2m → 目标冲出感知校准面积上限 → 检测死）；
   - 策略 sync 缓存**有界化**：新增 1s 过期定时器（`expire()` 原本从未被调用，缓存无限增长）+ `on_ego` 只遍历 matchable keys（原遍历全缓存 → executor 饱和 → DDS 背压 → ENT 流滞后 25s）；
   - **推理节流** 5Hz（`MIN_INFERENCE_INTERVAL_SEC=0.2`；三条输入流各触发推理 → 10Hz CUDA 推理占满单线程 executor）。
6. **工具**（仓库内）：`tools/plot_track.py`（复刻 Desktop/track_*.png 绘图）、
   `tools/run_scenario.sh`（一键三场景：Jetson launch + UE5 窗口 + 验收监控 + 证据收集）、
   `tools/policy_contrast_check.py`（指令对比离线证据）、`tools/rebuild_supervision_v16.py`、
   `tools/perception_blue_check.py`。
7. 本地测试：asv_vla 192 passed + training 70 passed（2 个预存环境失败：Win symlink 权限、
   策略契约配置漂移——与本会话无关）。陈旧测试 `test_image_entity_perception.py`（calibrated_red_geometry
   改名）已修复。

### ❌ 未完成（在线闭环仍断）

**红3m 在线闭环（v6 模型 + v5 感知 + 策略主导）每次 ~2-5 秒后断链**，症状与
**你自己 17:35-17:55 的 v5 蓝场运行完全相同**（`/tmp/run_blue_rgb.log` valid=1 仅 7 次，
`run_blue_probe3.log` 8 次；v1 验证场 `run_red_track.log` 1258 次）：

- 诊断链（逐节点 trace 已确认）：感知持续工作（4 实体/110ms）→ 但红船**可见性在 ~5s 后消失**
  （TRACK_TRACE `visible_source=0`）→ tracker/visual 空转 → 策略无有效输出 → E-STOP。
- 几何：ASV 从 4.5m 逼近红船到 ~2.0m 后停下——**红船 RGB 校准面积上限（COLOR_AREA_MAX=0.0172）
  在 ~2.3m 处杀死检测**（验证场最低 2.4m 存活）。
- 根因：管道端到端延迟 ~1s（感知 110ms + 等待窗口 + 编码 + 同步）与检测范围
  （2.4-5m）的物理矛盾：0.12m×5Hz=0.6m/s 逼近 × ~1-2s 延迟 ≈ 1-2m 过冲 → 冲过 2.4m 死区。
- 修复尝试（按序）：步长 0.12 / 死区 0.15 / 节流 5Hz / 缓存有界 / **死区 0.45m**。
  效果：吞吐 0→5Hz（节流正常）、ENT 滞后消除、缓存有界（13 keys）、
  **有效跟踪从 1-5s 延长到 ~15s**（run 13：63 次有效 setpoint 连续 15 秒）。
  仍断：接近阶段红船 RGB 检测在真距 ~2.3m 处死（近距几何估计偏低 → 死区 0.45 判据
  被偏置骗过 → ASV 过冲到检测死区）。治本在感知近距标定（补 2-3m 样本重训）。
- **相机流随机死亡**（~50% run）：UE5 相机发送端或 bridge 入站 ~5-10s 断流，全链路静默
  （runs 4/8/9 模式）；另一些 run 相机正常但无效（runs 1/2/3/5/6/7 模式）。非代码相关。

### 下一步建议（按优先级）

1. **继续闭环调参**（最快路径，PC 端无需动）：
   - 死区 0.15→0.4m（验证场包络 2.4-3.3m ≈ ±0.45m），让保持区避开 ~2.3m 检测死区；
   - 或步长改比例式（|误差|×k 而非常数上限），逼近时减速；
   - 或降低延迟：entity_wait_sec 0.5→0.35、检查 task_entity_tensor 滞后。
2. **感知近距标定**：补 2-3m 近距离红/蓝样本重训感知（治本——检测死区右移）。
3. **UE5 相机断流**：抓 UE5 侧相机发送器日志（ObjectDeliverer）定位随机断流。
4. 三场验证通过后：`tools/plot_track.py` 出图（脚本已就绪，日志格式已验证）。

## 2. 关键命令速查

```bash
# 三场景一键运行（WSL，每场 ~7 分钟）
MODEL=policy_near_rgb_v6_seed42.pt bash tools/run_scenario.sh TRACK-RED-3M "跟随红色目标船，保持3米距离" 3 /tmp/scene_red3m
MODEL=policy_near_rgb_v6_seed42.pt bash tools/run_scenario.sh TRACK-BLUE-3M "跟随蓝色目标船，保持3米距离" 3 /tmp/scene_blue3m
MODEL=policy_near_rgb_v6_seed42.pt bash tools/run_scenario.sh TRACK-RED-4M "跟随红色目标船，保持4米距离" 4 /tmp/scene_red4m
# 证据：/tmp/scene_*/vla_*.log + jetson_*.log + topic_hz.log

# 绘图（对每场日志）
"/mnt/d/Softwares/Python/Python313/python.exe" tools/plot_track.py \
  --log /tmp/scene_red3m/vla_TRACK-RED-3M.log --standoff 3 \
  --output-prefix "C:\Users\LIU\Desktop\track_world_red3m"

# 指令对比离线证据
"/mnt/d/Softwares/Python/Python313/python.exe" tools/policy_contrast_check.py \
  --checkpoint pc_datasets/checkpoints/near_rgb_v6/full_seed42/best.pt

# 感知蓝检查
"/mnt/d/Softwares/Python/Python313/python.exe" tools/perception_blue_check.py
```

## 3. 部署状态

- Jetson `~/jetson_asv_ws/`：已同步全部在线改动 + colcon 重建（asv_vla/asv_bringup/asv_ue_bridge/asv_jetson_interfaces 4 包）；
- 模型：`policy_near_rgb_v6_seed42.pt`（seed42 最佳：ADE 0.93，red4m 0.46）+ `image_entity_color_calibrated_v5.npz`（已部署）；
- launch 默认参数未改（运行时代入）；`models/manifest.yaml` 未更新（等门/在线验收后更新）。

## 4. 关键文件（本会话改动）

- `src/asv_vla/asv_vla/visual_standoff_guard.py`（策略优先 + 安全 backstop + 死区保持）
- `src/asv_vla/asv_vla/vla_policy_node.py`（单点输出、previous action、gate pending、身份 fail-closed）
- `src/asv_vla/asv_vla/policy_model.py`（language + structured entity + previous action 决策头）
- `src/asv_bringup/launch/vla_closed_loop.launch.py`（visual/policy/language CUDA 参数）
- `src/asv_vla/asv_vla/expert_trajectory.py`、`supervised_dataset.py`（每帧单步专家点）
- `src/asv_vla/asv_vla/generate_language_interventions.py`（4m 族 + 3m↔4m 对比对）、`language_intervention_dataset.py`（距离对比对校验）
- `evaluate_expert_labels.py`（13 标签 + canonical 实体 6.5m）、`training/build_feature_caches.py`（130）、`training/dataset.py`（130 + swap 距离正则）
- 诊断 trace（每 50-100 帧限流，保留）：PERCEPTION_PERF_TRACE / TRACK_TRACE / ENT_IN_TRACE /
  POLICY_TRACE；旧的 `VIS_IN_TRACE`/`EGO_TRACE` 不再是决策头输入。
- 工具：`tools/{plot_track,run_scenario,policy_contrast_check,perception_blue_check,rebuild_supervision_v16}.py/.sh`

## 5. 已知坑（新增）

1. **v5 栈在线断链**（见上）——你的旧 run 日志就是证据（`/tmp/run_blue_rgb.log` 7 次 valid）。
2. **WSL 传 PYTHONPATH 给 Windows python 无效**——训练/缓存一律用 `python.exe -c "sys.path.insert..."` 模式。
3. **Jetson 残留进程**：kill 模式要含 `jetson_asv_ws/install|ros2 launch`（`asv_vla` 匹配不到 python3 进程）。
4. 绘图日志坐标 cm（/100）；`SCENE_EXEC_APPLY` 日志限流（1st + 每 25 次），不是真实次数。
5. 三场起点一致性：同一 `-Layout=L7 -Motion=S2 -Seed=230908`（-Slot 只是标签）。
