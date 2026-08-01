# 最终系统 TODO（2026-08-02）

## 目标

在 UE5 S2 正弦场景中，Jetson 只接收真实 SceneCapture JPEG、任务文本和本船
`UEASVState`，在线完成：

```text
图像 + 指令 → 任务条件 Entities → 跨帧速度 → ego 对齐
          → CUDA 视觉/语言/策略 → 20×2 期望位移轨迹
          → 安全门 → 单步 desired_x/desired_y → UE5 运动学执行
```

最终视频必须能证明 ASV 是由这条在线链路运动，而不是 `/ue/entities` 真值、专家
publisher、左右推力控制器或录播轨迹驱动。

## 当前状态

### 已完成（源码与本地合同）

- [x] `/ue/entities` 已从在线感知/策略输入边界排除，只保留采集和离线监督用途。
- [x] `image_entity_perception` 订阅 `/task/text`，解析 follow red/blue/left/right
  与 stop；输出 `instruction_id`，只把任务相关实体标为 `is_target/visible`。
- [x] 图像实体模型的主要归一化/线性推理增加显式 CUDA 路径；CUDA 不可用时节点
  fail-closed，不静默改用 NumPy。
- [x] `SmallTrajectoryPolicy` 已下沉到可安装的 `asv_vla.policy_model`；PC 的
  `training/model.py` 只保留兼容导入。
- [x] 策略节点默认 `backend=torch_cuda`，strict-load `model_config` 与
  `model_state_dict`；ONNX 仅作为显式 `onnx_cpu` 兼容后端。
- [x] 本地无 ROS/CUDA 环境的合同测试通过：感知/任务选择、同步、安全和 Torch
  运行时静态合同已覆盖；当前本地结果为 `238 passed, 7 skipped`。

### 设备与在线验收（2026-08-02）

- [x] 当前源码与 canonical `.pt` 已同步到 `/home/jetson/jetson_asv_ws`，
  `colcon build --merge-install --symlink-install --packages-select asv_vla asv_bringup`
  通过。
- [x] Jetson 默认分阶段启动通过：真实 `LANGUAGE_READY_VALID ... release_model=true`，
  随后 `visual_encoder ... device=cuda` 与 `POLICY_READY backend=torch_cuda device=cuda`。
  感知节点启动 detail 也显式包含 `device=cuda`；无 CUDA/模型时仍 fail-closed。
- [x] 真实 UE5 `L7/S2/seed=230908` 在线闭环通过。单一
  `run_id=E82B58E6415C9AE61F3797BB1C7B7D99`、`scene_seed=230908` 链中，JPEG 感知
  连续输出红目标约 `4.6 m` 的相对位置；策略安全门后连续发送有效
  `Delta_X_Cm/Delta_Y_Cm`（Jetson 日志计数：`PERCEPTION_TRACE=5`、policy stamp
  trace 约 20、有效 setpoint 约 32）。UE5 记录约 350 次 `SCENE_EXEC_APPLY`，ASV
  世界 X 从约 `-10150 cm` 变化到 `-7899 cm`，并以 `SCENE_UE_COMPLETE runtime_seconds=35.00`
  正常结束。
- [x] 已修正 UE5 `SceneAutomationSubsystem`：bounded `MaxRuntimeSeconds` 现在记录
  `SCENE_UE_COMPLETE`，不再把正常演示结束误报为 `SCENE_UE_FAIL`。
- [x] PC 活动数据已收敛为 canonical near 数据、冻结特征、`full_seed42`、图像校准器
  与 Qwen；旧远距离/旧版本特征、ONNX、旧 checkpoint 和日志已移入可恢复归档：
  `/tmp/asv_vla_cleanup_20260802/pc_datasets`。

### 当前边界（仍需诚实保留）

- [ ] 当前 S2 演示的 Qwen 采用“首指令 CUDA 编码后释放权重”以适配 Jetson 8 GB；
  embedding 继续在线，但任务热切换需要重启闭环，不把热切换列为本次验收能力。
- [ ] 感知前处理和颜色几何标定仍是确定性 CPU/PIL/NumPy；模型归一化/线性矩阵、
  MobileNet、Qwen 首指令和 policy 请求 CUDA。若目标是全链路 GPU 前处理，需要单独
  的性能工程，不影响本次真实在线验收结论。

## Canonical 资源

| 用途 | 活动资源 |
|---|---|
| 原始近距离数据 | `pc_datasets/extracted_sine_near` + 12 个 Day12 tar 包 |
| 冻结特征 | `pc_datasets/features_sine_near_image_color_v1` |
| 策略 checkpoint | `pc_datasets/models/policy_sine_near_image_color_seed42.pt` |
| 图像实体模型 | `pc_datasets/models/image_entity_color_calibrated_v1.npz` |
| 语言模型 | `pc_datasets/models/Qwen3-Embedding-0.6B` |
| Jetson manifest | `models/manifest.yaml` |

策略 checkpoint SHA-256：
`6c4ed50d49a0ba9447a3d991cb09de130bbdbcc6eb98eca1b98c49dfd66d7685`。

## 启动与验收命令

Jetson 先启动：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  policy_backend:=torch_cuda policy_device:=cuda \
  language_backend:=qwen language_device:=cuda \
  language_release_after_encode:=true \
  language_staging_delay_sec:=20.0 \
  execution_address:=192.168.137.1 execution_port:=8081
```

然后用项目文件作为 UnrealEditor 第一个参数启动 UE5；录制时保留窗口模式。
启动后检查：

```bash
ros2 topic echo /vla/perceived_entities --once
ros2 topic echo /vla/tracked_entities --once
ros2 topic echo /vla/policy_trajectory --once
ros2 topic echo /decision/output --once
```

必须满足：

- Entities 的 `source` 是 `image_perception/temporal_tracker`，`instruction_id` 与
  当前任务一致；不出现 `/ue/entities` 作为输入来源。
- 速度字段第一帧 `velocity_valid=false`，后续才由 tracker 填写；不是单帧图像直接
  猜速度。
- 策略 `model_version=vla_torch_cuda_v1`、`valid=true` 时输出 20×2 body-frame
  累计位移；控制器只取短前缀并发布 `desired_x/desired_y`。
- `SCENE_EXEC_APPLY` 连续出现且 ASV 世界坐标变化；模型/身份/可见性失败时必须
  hold/fail-closed。

## 不允许回退

- 不得把 UE Entities、UE BBox、专家轨迹或左右推力接回最终在线策略。
- 不得把历史 ONNX/CPU 作为默认模型；不得在 CUDA 失败时偷偷降级 CPU。
- 不得并行启动第二套 bridge、policy、safety gate、controller 或 expert publisher。
- 不得用“valid=true 的零轨迹”掩盖模型加载失败；失败必须 `valid=false` 并 hold。

## 历史证据

L7/S2 `seed=230906`、`230902` 是历史 JPEG→ONNX/CPU→guard→8081 对照运行；本轮
`seed=230908` 已补齐真实 Qwen CUDA 首指令、CUDA policy、图像感知和 UE5 执行，
不再把历史 ONNX/CPU 运行当作最终证据。上述结果证明“当前 near S2 演示链”闭环可录制，
但不等于所有布局、距离、颜色和动态条件都已统计泛化；统计鲁棒性仍按下方可选 8-run
表格单独记录。
