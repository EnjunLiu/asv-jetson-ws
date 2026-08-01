# 最终系统状态与 TODO（2026-08-01，自动采集批次已完成）

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
  velocity_valid 和任务元数据；全量 Python 单测 197 passed、4 skipped；接口、VLA
  Python 包和 UE bridge 已在本机构建。
- Jetson 已同步并固化为 `4e2d9f3`：Humble 下接口、VLA、UE bridge、bringup 及四个
  底层包构建通过；无 UE bridge 的启动烟测通过，且在线视觉节点只使用
  `/vla/tracked_entities`。
- 自动采集链路已验证：先启动 Jetson，再由 `collect.ps1` 启动 UE5、等待连接、录制、
  打包并下载；当前已取得完整计划的 30 个真实 UE5 专家 Run（3000 帧），每个具体包
  均为 `quality_passed=true`、4/4 Entities 可见。部分包有 UE 传输丢帧 warning（不把
  warning 伪报成零间隙；时序跟踪按 Frame_Index/stamp_us 处理 gap）；新 Run 的近距离/
  速度门通过后有 2740 帧可用于图像感知训练。

## 当前诚实边界

- 当前 image_entity_ridge_v2 是轻量监督模型，不是通用检测器。数据已从旧的 14 个
  run 扩充到 44 个 run；新采集的 30 个 run 经过 5 m/姿态/速度门后有 2740 帧，跨
  Run 验证和正式 acceptance gate 已通过；报告中 `acceptance_ready=true`。这代表
  当前 UE5 场景分布内的图像感知验收，不等同于开放世界检测器。
- 当前重训报告为 2116 train/641 validation 帧，验证可见性 98.87%、投影可见目标几何
  RMSE 0.325 m；逐 Run 最差可见性 96.5%、几何 RMSE 0.462 m。正式 gate 已写入报告，
  同步 Jetson 后模型可部署；仍需完成真实在线闭环验收。
- 因此当前模型可以称为“场景内图像感知验收通过”，但不能替代最终在线闭环验收；
  在线缺少模型、错帧或低置信度时必须停船。现有 policy.onnx 也尚未用真实 ego 和
  新感知输出重新训练。
- 旧数据里存在相机画面与实体姿态交替的迹象；新采集必须先通过近距离和时序质量门，
  再训练，不把 14 个旧 run 直接当成最终真值。

## 下一步（按阻塞顺序）

1. 【已完成】用 collect.launch.py max_target_distance_m:=5 自动采集 30 个新的近距离
   run，并检查 JPEG、真值投影、ego yaw、速度和帧连续性；具体包已下载到
   `../pc_datasets/incoming`，episode 已解压到 canonical 数据集。
2. 【已完成】用 v2 特征、相机投影可见性 mask 和明确阈值完成图像感知模型验收，重新运行：

   PYTHONPATH=src/asv_vla python3 training/train_image_entity_perception.py
     --episodes ../pc_datasets/extracted_sine/artifacts/day8_episode
     --output models/image_entity_perception_v1.npz
     --max-primary-distance-m 5 --max-abs-yaw-rad 0.1
     --max-abs-surge-velocity-mps 1.0

   报告已满足可见性、几何误差和逐 Run 阈值，`acceptance_ready=true`；部署前仍需
   Jetson 同步、构建和真实 JPEG smoke。
3. 【cache 已完成，策略待训练】正式 cache 已用 `image_entity_ridge_v2` 重新生成：
   30 Runs/3000 帧/269380 samples，实体几何与 crop 来自 JPEG 图像模型，速度来自
   与 Jetson 一致的 EMA tracker（`ttl_frames=2, ttl_sec=0.5, alpha=0.6, beta=0.85`），
   ego 来自 UEASVState；每个 cache manifest 都固定模型权重和 tracker provenance。
   下一步用 `training/config/train_30_v8_image_only.yaml` 重新训练/导出 policy.onnx；
   在 Jetson 用 onnxruntime 做输入输出 parity 和延迟/内存测量。
4. 最后做在线验收：只启动 VLA launch，再 Play UE5；记录
   image_perception -> temporal_tracker -> policy -> safety_gate ->
   kinematic_setpoint 的同一 Run/Scene/Frame 链和真实画面，才录制演示视频。

## 不允许回退

- 在线策略、视觉编码器、任务张量和安全门不得订阅 /ue/entities。
- 不得从单帧图像填写任何速度；不得以专家输出替代最终策略验收。
- 不得把 valid=true 的安全停止当作有效动作；缺模态、错帧、低置信度一律
  valid=false/hold。
