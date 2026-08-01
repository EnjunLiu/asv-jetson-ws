# ASV VLA 最终架构

## 1. 在线闭环

```text
UE5 SceneCapture JPEG ───────────────┐
UE5 task text ───────────────────────┼─> Jetson ROS 2
UE5 ASV_Location/Rotation/Velocity ──┘
      │
      ├─ /ue/entities（真值：只录制/离线监督）
      ├─ /ue/camera_frame（在线输入）
      └─ /ue/asv_state（在线 ego 输入）

/task/text + /ue/camera_frame
  -> image_entity_perception (image + instruction)
  -> /vla/perceived_entities
  -> temporal_entity_tracker (跨帧位置差分)
  -> /vla/tracked_entities
  -> task_entity_tensor（任务相关实体的 16x16 几何/风险张量）
  -> visual_encoder（MobileNetV3 CUDA）
  -> language_qwen（Qwen3 CUDA，默认常驻）
  -> vla_policy（Torch CUDA + tracked entities + language + ego）
  -> safety_gate
  -> trajectory_controller
  -> decision_setpoint_adapter
  -> /ue/kinematic_setpoint
  -> UE5 C++ executor
```

这里的 Entities 是图像推断出的中间变量，不是 UE 真值转发。感知节点仍可输出固定
实体槽位，但只有符合当前指令的实体被标记为 `is_target=true/visible=true`，下游张量
只选择这些有效实体。颜色/左右目标的选择由图像和指令共同决定；当前部署的几何
校准器只对近距离红色目标完成验收，未校准的颜色/方位任务必须 fail-closed。

单帧图像不直接产生速度。`temporal_entity_tracker` 使用同一
`(run_id, scene_seed, frame_index, stamp_us)` 的相邻帧计算相对速度，并在第一帧设置
`velocity_valid=false`。

`/ue/asv_state` 来自 UE5 当前本船的世界位置、姿态、surge velocity 和 yaw rate；
策略只使用归一化的本船运动状态 `[surge_velocity, yaw_rate]` 作为 ego，不用目标真值
伪造本船状态。ego 必须与图像/Entities 具有相同的 Run/Scene/Frame 身份。

## 2. 动作边界

策略输出固定的 20 个 body-frame 累计位移点（`dt=0.2 s`），不是左右推力。安全门
检查有限性、速度、曲率、碰撞余量、陈旧和身份；轨迹控制器只取安全轨迹短前缀，
adapter 每个观测帧发送一个 `desired_x/desired_y` setpoint。UE5 仿真 executor 可以
直接设置位置和航向；真实船的底层控制器属于独立工程。

最终在线链中没有 world model、候选轨迹枚举、专家 publisher 或 ONNX/CPU policy。
专家轨迹只出现在 `collect.launch.py` 的离线标签采集路径。

## 3. ROS 包

| 包 | 责任 |
|---|---|
| `asv_jetson_interfaces` | CameraFrame、UEASVState、UEEntity、TaskFeatures、VisualFeatures、TaskEmbedding、SelectedTrajectory 等最终消息 |
| `asv_ue_bridge` | TCP JSON 校验、JPEG、本船状态、真值标签发布、运动学 setpoint 输出 |
| `asv_vla` | 图像实体模型、任务筛选、跨帧 tracker、视觉/Qwen/策略、安全门、轨迹控制、采集/回放 |
| `asv_bringup` | `vla_closed_loop.launch.py`、`collect.launch.py`、`record_episode.launch.py`、`replay_episode.launch.py` |

低层 ESP32/推力分配不再作为本仓库的在线 launch 或消息依赖。

## 4. 训练边界

```text
UE5 JPEG + UE Entities + UEASVState + task text
  -> PC frame records
  -> 图像实体模型训练（Entities 仅标签）
  -> 跨帧速度和 ego 特征
  -> 冻结视觉/Qwen 特征 + SmallTrajectoryPolicy 训练
  -> Jetson strict CUDA checkpoint
```

训练标签可以使用 UE Entities 和专家轨迹；部署时只复制模型和 Qwen 目录，在线不读取
UE Entities。数据与模型位于仓库外的 `pc_datasets`，Git 只保存代码、manifest 和合同。

## 5. CUDA 与内存策略

- 图像实体模型在 `device=cuda` 时用 torch CUDA 完成特征图、归一化和线性投影；JPEG/PIL
  解码不是神经网络推理。
- MobileNet、Qwen、policy 都显式使用 CUDA；任何 CUDA 不可用或模型加载失败都
  `valid=false`，不静默回退 CPU。
- Qwen 默认常驻，支持新的 `/task/text` 在线编码。若 Jetson 设备级实测峰值不足，
  只能显式启用 `language_release_after_encode=true`，释放权重但保留真实 Qwen 生成的
  embedding；禁止改回缓存 `.npy`。
- 训练/评估中的 NumPy 张量工程和安全门是确定性 CPU 代码，不应冒充神经网络推理。

## 6. 验收不变量

1. 在线策略订阅 `/vla/tracked_entities`、`/vla/visual_features`、
   `/vla/language_embedding` 和 `/ue/asv_state`，不订阅 `/ue/entities`。
2. `source=image_perception/temporal_tracker`，`instruction_id` 与任务一致。
3. 第一帧速度无效，后续速度来自 tracker。
4. ego 缺失、身份不匹配、CUDA/模型失败时必须 hold。
5. UE 日志出现连续 `SCENE_EXEC_APPLY` 和最终 `SCENE_UE_COMPLETE`。

## 7. 当前诚实边界

当前图像实体模型针对近距离 S2 红/蓝目标校准，约 5 m 是主要工作域；约 7 m 白色干扰
目标按 OOD fail-closed。一次或两次成功运行只能证明可录制在线闭环，不能替代多 seed、
多布局统计鲁棒性。
