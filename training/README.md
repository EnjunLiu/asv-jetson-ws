# PC 数据与训练流水线

`training/` 是离线 PC 流水线。它只读取已录制的图像、`Entities` 监督和真实自船
状态，生成冻结特征、训练 checkpoint 和验证报告；它不会启动 ROS，也不会向 UE5
发送控制命令。

## 外部数据布局

仓库外的 `pc_datasets/` 是唯一数据根目录：

```text
pc_datasets/
├── incoming/       # Jetson/UE5 录制压缩包
├── extracted/      # day8_episode + day10_supervised
├── registry/       # Run/Scene-Seed 分组注册表
├── features/       # 冻结语言/视觉/实体/ego 特征
├── checkpoints/    # 训练模型和 summary.json
├── models/         # image_entity_color_calibrated_v1.npz/json、Qwen
└── reports/        # 训练与验证报告
```

Git 中只保存代码、配置和接口；原始 JPEG、实体真值、权重和生成报告不进入仓库。

## 当前采集计划

最终演示和训练使用
`training/config/sine_near_collection_plan_v1.json`：12 个 L7/L7B、S2、近距离
（约 5 m）slot。它要求每个 Run 至少 80 帧，并保留 `run_id/scene_seed/frame_index`
以及 JPEG/Entities 的 SHA-256。`Entities` 是监督标签，不是在线策略输入。

Jetson 端批量采集入口：

```bash
cd ~/jetson_asv_ws
bash scripts/remote_collect.sh \
  L7_S2_R1 L7 S2 220701 \
  training/config/sine_near_collection_plan_v1.json follow
```

Windows 自动化入口为 `tools/ue5/collect.ps1`，其默认计划也指向同一个 near S2
计划；UE5 必须以 `.uproject` 作为第一个参数启动。

## 从录制到 checkpoint

1. `training.dataset_registry` 扫描每个 Run，检查帧数、身份、质量和监督完整性。
2. `training.make_group_splits` 按 Run/Scene Seed 分组，避免相邻帧泄漏到验证集。
3. `training.feature_cache` 在 PC CUDA 上冻结 Qwen、MobileNet、图像实体和 ego 特征。
   相对速度只使用相邻帧 tracker 结果；不能从单张图像标签直接伪造速度。
4. `training.train` 训练 20 点二维位移策略；策略不接收颜色真值、实体 ID、专家候选
   轨迹或左右推力。
5. `training.evaluate_selection`、`training.contract_checks` 和近距离 S2 回放生成
   外部报告，只有通过验证的 checkpoint 才能复制到 Jetson `models/`。

典型命令（在仓库根目录，`PYTHONPATH=src/asv_vla`）：

```bash
python3 -m training.dataset_registry \
  --data-root /path/to/pc_datasets/extracted \
  --output /path/to/pc_datasets/registry/dataset_registry_v1.jsonl

python3 -m training.make_group_splits \
  --registry /path/to/pc_datasets/registry/dataset_registry_v1.jsonl \
  --output /path/to/pc_datasets/registry/group_split_v1.json

python3 -m training.feature_cache build \
  --episode /path/to/pc_datasets/extracted/artifacts/day8_episode/RUN_ID \
  --supervision /path/to/pc_datasets/extracted/artifacts/day10_supervised/RUN_ID \
  --instructions dataset/language/instructions.jsonl \
  --output-root /path/to/pc_datasets/features \
  --language-model-path /path/to/pc_datasets/models/Qwen3-Embedding-0.6B \
  --device cuda
```

Jetson 内存紧张时，feature-cache 允许分阶段加载 Qwen 后释放 CUDA 权重再加载视觉
骨干；持久化 CUDA 分配失败必须报错，不能静默改用 CPU。

## 在线模型契约

部署模型由 `models/manifest.yaml` 指定：

- `image_entity_color_calibrated_v1.npz`：JPEG -> 近距离图像几何；约 5 m 外 fail-closed；
- `policy_sine_near_image_color_seed42.pt`：Torch CUDA，输入固定多模态 481k 合同；
- `Qwen3-Embedding-0.6B`：真实 CUDA 语言编码，默认常驻，可显式释放权重但不允许 `.npy`。

在线链路永远是：

```text
JPEG + task text + real ego
  -> image entities -> temporal velocity -> visual/task/language/ego
  -> Torch CUDA policy -> safety gate -> desired_x/y
```

## 验证

```bash
PYTHONPATH=src/asv_vla pytest -q training/test
PYTHONPATH=src/asv_vla pytest -q
```

提交新的模型前，至少提供独立 Scene Seed 的 S2 近距离闭环日志、图像感知 trace、
策略 CUDA ready、有效 setpoint 和 UE5 executor apply 计数。一次成功演示不等于跨场景
泛化；统计结论应在新增 Runs 后再报告。
