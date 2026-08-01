# 最终系统状态与 TODO（2026-08-01）

## 已落地

- 真值/在线彻底分流：/ue/entities 只给录制、离线标签和专家基线；在线
  vla_closed_loop.launch.py 只允许
  /ue/camera_frame -> image_entity_perception -> temporal_entity_tracker ->
  /vla/tracked_entities。
- 单帧不输出速度：图像感知只输出位置、语义、可见性、bbox、置信度，并明确
  velocity_valid=false；速度由相邻 (Run_ID, Scene_Seed, Frame_Index, stamp_us)
  观测有限差分得到，首帧速度无效。
- 在线 ego 已接入：策略按完全相同的帧身份读取 /ue/asv_state 的 surge/yaw
  rate；缺失或错帧不再用零向量替代，而是 fail-closed。
- 图片+Entities 训练管线：录制器已保存 JPEG、ego、Entities 真值和 provenance；
  training/train_image_entity_perception.py 可从 extracted_sine 生成
  models/image_entity_perception_v1.npz。
- 近距离采集门：collect.launch.py 默认 max_target_distance_m=5，远距离帧不
  进入新训练集；训练器还默认剔除 |ego yaw|>0.1 rad 的疑似相机/真值不同步帧，
  并剔除 |surge_velocity|>1.0 m/s 的速度异常帧。
- ROS 契约和构建：UEEntity/UEEntityArray/TaskFeatures 增加来源、bbox、置信度、
  velocity_valid 和任务元数据；纯单测 149 passed，训练单测 39 passed；接口、VLA
  Python 包和 UE bridge 已在本机构建。
- Jetson 已同步并固化为 `4e2d9f3`：Humble 下接口、VLA、UE bridge、bringup 及四个
  底层包构建通过；无 UE bridge 的启动烟测通过，且在线视觉节点只使用
  `/vla/tracked_entities`。

## 当前诚实边界

- 当前 image_entity_ridge_v1 是第一版轻量监督模型，不是通用检测器。现有 14 个
  旧 run 经过 5 m 近距离和姿态过滤后仅约 32 帧，再经 surge 速度门只剩 17 帧，
  模型仍未验收；
  image_entity_perception_v1.json 明确标记 acceptance_ready=false。
- 5 m 稳定旧帧通过 surge 速度质量门后仅剩 17 帧（14 train/3 validation）；
  在样本量和验证误差同时达标前，模型仍未验收。
- 因此不能把当前模型称为最终在线感知验收通过；在线缺少模型、错帧或低置信度
  时必须停船。现有 policy.onnx 也尚未用真实 ego 和新感知输出重新训练。
- 旧数据里存在相机画面与实体姿态交替的迹象；新采集必须先通过近距离和时序质量门，
  再训练，不把 14 个旧 run 直接当成最终真值。

## 下一步（按阻塞顺序）

1. 用 collect.launch.py max_target_distance_m:=5 采集至少 8 个新的近距离 run；
   每个 run 先检查 JPEG 与真值投影/ego yaw 连续性，再放入 PC 数据集。
2. 在 PC 运行：

   PYTHONPATH=src/asv_vla python3 training/train_image_entity_perception.py
     --episodes ../pc_datasets/extracted_sine/artifacts/day8_episode
     --output models/image_entity_perception_v1.npz
     --max-primary-distance-m 5 --max-abs-yaw-rad 0.1
     --max-abs-surge-velocity-mps 1.0

   只有验证误差达标，才把 acceptance_ready 改为 true 并部署模型。
3. 用新感知特征（颜色列仍不接受 UE truth；速度来自 tracker，ego 来自
   UEASVState）重建 feature cache，重新训练/导出 policy.onnx；在 Jetson 用
   onnxruntime 做输入输出 parity 和延迟/内存测量。
4. 最后做在线验收：只启动 VLA launch，再 Play UE5；记录
   image_perception -> temporal_tracker -> policy -> safety_gate ->
   kinematic_setpoint 的同一 Run/Scene/Frame 链和真实画面，才录制演示视频。

## 不允许回退

- 在线策略、视觉编码器、任务张量和安全门不得订阅 /ue/entities。
- 不得从单帧图像填写任何速度；不得以专家输出替代最终策略验收。
- 不得把 valid=true 的安全停止当作有效动作；缺模态、错帧、低置信度一律
  valid=false/hold。
