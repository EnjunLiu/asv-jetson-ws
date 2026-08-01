# 最终系统状态与 TODO（2026-08-01，在线闭环已实测，策略验收仍阻塞）

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
  velocity_valid 和任务元数据；全量 Python 单测 **201 passed、5 skipped**；接口、
  VLA Python 包和 UE bridge 已在本机构建。
- Jetson 已以当前源码重建 9 个 ROS 包（Humble，`colcon build` 全部 finished）；
  无 UE bridge 的启动烟测通过，视觉编码器使用 CUDA，在线视觉节点只使用
  `/vla/tracked_entities`。当前运行时模型 SHA-256 已核对。
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
  Jetson 同步、CUDA 视觉启动和真实在线 JPEG smoke 已通过；这只是感知模型 gate。
- 因此当前模型可以称为“场景内图像感知验收通过”，但不能替代最终策略验收；在线
  缺少模型、错帧或低置信度时必须停船。当前策略候选为
  `current_policy_image_v8_seed17`（`policy_image_seed17.onnx`），导出 parity
  `max_diff=0.00e+00, cosine=1.000000`，PC ONNX benchmark 为 5528.95 Hz
  （p50 0.17 ms、p95 0.26 ms），但它只能标记为 **provisional_demo_only**。
- 策略三 seed 验证门仍为 FAIL，不能称为最终训练完成：

  | seed | ADE (m) | FDE (m) | 主要失败 |
  |---|---:|---:|---|
  | 17 | 0.437 | 0.662 | stop F1=0.936、within 0.10m=0.897 |
  | 23 | 0.675 | 1.100 | ADE/FDE improvement 与 stop gate |
  | 42 | 0.745 | 1.213 | ADE/FDE improvement 与 stop gate |

  `summary.json` 的 `validation_gate_passed=false` 是最终策略状态；Jetson 上的
  seed17 仅用于在线链路演示，不得用于宣称泛化通过。
- 旧数据里存在相机画面与实体姿态交替的迹象；新采集必须先通过近距离和时序质量门，
  再训练，不把 14 个旧 run 直接当成最终真值。

## 实测在线闭环（2026-08-01）

- 第一次自动 UE 运行暴露出确定性故障：图像模型对一个画面外预测调用投影时抛出
  `TargetProjectionError`，感知节点退出，随后视觉编码器产生大量
  `ENTITY_FRAME_TIMEOUT`。修复提交 `d51aab3` 将单目标越界降级为
  `bbox_valid=false/visible=false`，整帧仍发布；新增回归测试覆盖该场景。
- 修复后的第二次自动运行使用 `Scene_Seed=120102`、近距离 L1 场景：
  `process has died=0`、`PERCEPTION_ERROR=0`、`ENTITY_FRAME_TIMEOUT=0`；
  bridge 收到带 Run_ID/Scene_Seed/Source_Frame_Index 的运动点，策略曾产生有效
  setpoint。UE 日志中 ASV 从 t=0 的 `(-10150,-10000)` 移到 t=3 的
  `(-10384.678,-9970.687)`，随后安全门因 `CURVATURE_LIMIT`/`COLLISION_RISK`
  进入 hold。这证明真实图片→时序实体→策略→安全门→运动执行链路已连通，**不等于
  策略行为验收通过**。

## 下一步（按阻塞顺序）

1. 【已完成】用 collect.launch.py max_target_distance_m:=5 自动采集 30 个新的近距离
  run，并检查 JPEG、真值投影、ego yaw、速度和帧连续性；具体包已下载到
  `../pc_datasets/incoming`，episode 已解压并核验到 canonical 数据集；原始中间包
  已移入可恢复清理目录。
2. 【已完成】用 v2 特征、相机投影可见性 mask 和明确阈值完成图像感知模型验收，重新运行：

   PYTHONPATH=src/asv_vla python3 training/train_image_entity_perception.py
     --episodes ../pc_datasets/extracted_sine/artifacts/day8_episode
     --output models/image_entity_perception_v1.npz
     --max-primary-distance-m 5 --max-abs-yaw-rad 0.1
     --max-abs-surge-velocity-mps 1.0

   报告已满足可见性、几何误差和逐 Run 阈值，`acceptance_ready=true`；Jetson 同步、
   重建和真实 JPEG smoke 已通过。
3. 【已完成，策略门 FAIL】正式 cache 已用 `image_entity_ridge_v2` 重新生成：
  30 Runs/3000 帧/269380 samples，实体几何与 crop 来自 JPEG 图像模型，速度来自
  与 Jetson 一致的 EMA tracker（`ttl_frames=2, ttl_sec=0.5, alpha=0.6, beta=0.85`），
  ego 来自 UEASVState；每个 cache manifest 都固定模型权重和 tracker provenance。
  `train_30_v8_image_only.yaml` 已完成三 seed 训练，保留失败报告与 seed17 候选；
  不要覆盖候选模型或删除失败证据。
4. 重新平衡近距离数据（红/蓝目标在相同 5 m 内的可见距离分布、相机视角和运动状态），
   让 stop/10m 标签覆盖真实在线分布；然后只在独立 Run 分组上重建 image-only
   cache、三 seed 重训、导出并重新执行 validation gate。
5. 只有当策略 gate 通过后，重复至少 3 个不同 `Run_ID/Scene_Seed` 的 UE 在线运行，
   统计有效策略点、hold/E-STOP、碰撞门拒绝和目标距离，再录制最终演示视频。当前
   provisional demo 可用于展示模块数据流，不能替代这一步。

## 不允许回退

- 在线策略、视觉编码器、任务张量和安全门不得订阅 /ue/entities。
- 不得从单帧图像填写任何速度；不得以专家输出替代最终策略验收。
- 不得把 valid=true 的安全停止当作有效动作；缺模态、错帧、低置信度一律
  valid=false/hold。

## 清理状态

- 已确认无用的旧特征缓存、旧 checkpoint、重复监督数据、旧 tar 包和生成目录已移入
  可恢复隔离目录 `/tmp/asv_vla_cleanup_20260801`；canonical 数据集、当前 image-only
  cache、训练失败报告、候选 checkpoint 和 Qwen 训练模型保留。
- Jetson 仅保留源码、`install/`、当前模型和运行时资源；本次 `build/`、`log/`、重复
  测试副本及生成缓存已移入 Jetson 的 `/tmp/asv_vla_cleanup_20260801`。
