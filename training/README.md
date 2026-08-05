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
├── features/       # 冻结语言/结构化实体特征与感知审计数据
├── checkpoints/    # 训练模型和 summary.json
├── models/         # perception_image_conditioned_130_v1.npz、policy、Qwen
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
3. `training.feature_cache` 在 PC CUDA 上冻结 Qwen 和图像感知/结构化实体特征。
   启用帧级图像感知时固定使用 instructions manifest 第 0 行的语言 embedding；这个
   选择记录在每个 cache 的 `manifest.json` 和汇总的 `feature_set_manifest.json` 中。
   相对速度只使用相邻帧 tracker 结果；不能从单张图像标签直接伪造速度。
4. `training.train` 训练单步二维期望位移策略；每个样本是某个
   `(Run, instruction, frame)` 时刻的专家点 `[desired_x, desired_y]`，不是整条专家
   轨迹。决策样本仍按每帧 `instruction_id` 从语言表配对；这与帧级感知使用第 0 行
   embedding 是两个独立策略。决策头接收 `language + entity_geometry + previous_action`
   及有效性 mask；
   `previous_action` 只从同一 Run、同一 instruction 的相邻前帧专家点生成，首帧或前帧
   STOP 使用零值并置无效。策略不接收全局视觉 token、实体 crop token、ego、颜色真值、
   实体 ID 或左右推力。缓存中的视觉/ego 数组只用于感知审计，不会进入决策头。
5. `training.evaluate_selection`、`training.contract_checks` 和近距离 S2 回放生成
   外部报告，只有通过验证的 checkpoint 才能复制到 Jetson `models/`。

单点策略的 checkpoint 选择指标是 `action_error_m`；它不能使用轨迹级 ADE/FDE。
验证报告同时输出 `lateral_action_error_m` 和每个语言任务的横向误差，用于单独检查
红 4m、蓝 3m 的左右泛化。当前 12-Run 数据的 PC 重训命令如下，输出目录必须是
新的空目录：

```bash
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
export PYTHONPATH=src/asv_vla
export TRAIN_OUTPUT=/mnt/c/Temp/asv_vla_retrain_20260805/policy_single_point_v4_20260805

python3 -m training.train train \
  --config training/config/train_sine_near_image_v3.yaml \
  --model-config training/config/model_small_v3.yaml \
  --features /mnt/c/Temp/asv_vla_retrain_20260805/features_v3 \
  --split /mnt/c/Temp/asv_vla_retrain_20260805/group_split_v2.json \
  --instructions dataset/language/instructions.jsonl \
  --output-root "$TRAIN_OUTPUT" \
  --git-sha "$(git rev-parse --short HEAD)" \
  --device cuda \
  --execution-target pc

python3 -m training.train evaluate-test \
  --config training/config/train_sine_near_image_v3.yaml \
  --model-config training/config/model_small_v3.yaml \
  --features /mnt/c/Temp/asv_vla_retrain_20260805/features_v3 \
  --split /mnt/c/Temp/asv_vla_retrain_20260805/group_split_v2.json \
  --instructions dataset/language/instructions.jsonl \
  --output-root "$TRAIN_OUTPUT" \
  --git-sha "$(git rev-parse --short HEAD)" \
  --device cuda \
  --execution-target pc
```

接受 checkpoint 至少需要：三 seed 的 sealed test 均通过 `action_error_m` 相对
label-mean 提升 `>=30%`、STOP F1 `>=0.90`、STOP 零动作在 `0.10m` 内的比例
`>=0.95`、动作越界率为 `0`、无 invalid；另外红 4m 和蓝 3m 各自的
`per_label[*].lateral_action_error_m` 建议 `<=0.08m`。既有
`policy_v3_single_point_20260805/full_seed17/best.pt` 的总体 test action error 为
`0.0511m`，但仍应使用上述命令在修复后的选择逻辑下重新生成并复核。

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

- `perception_image_conditioned_130_v1.npz`：JPEG + task embedding -> 结构化实体；相对
  速度由 temporal tracker 计算，模型不直接从单帧输出速度；
- `policy_single_point_v3_full_seed17.pt`：Torch CUDA，输入为任务嵌入、结构化实体、上一
  个放行动作及有效性 mask，输出一个 `[desired_x, desired_y]` body-frame 单步位移；
- `Qwen3-Embedding-0.6B`：真实 CUDA 语言编码；当前 Jetson 闭环使用
  `language_release_after_encode=true`，释放权重但保留真实 embedding，不允许 `.npy` 或 CPU fallback。

在线链路永远是：

```text
JPEG + task text
  -> image + task embedding perception -> temporal velocity -> structured entities
  -> task embedding + structured entities + previous gated action
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

## PC 合成几何训练工具

`training.synthetic_geometry_train` 是一个隔离的 PC-only smoke/training 工具。它复用
现有 `asv_vla.task_entity_tensor` 的 16 维结构化实体行和
`asv_vla.expert_trajectory` 的单步专家公式，覆盖红/蓝目标、3m/4m/10m，以及带余量的
L7/S2 运行时范围 `x=[3.80,4.85] m`、`y=[-1.35,0.55] m`。前 12 个样本固定包含
`(4.74,0.49)` 与 `(3.94,-1.27)` 两个回归点的全部颜色/距离组合；同时生成上一帧有效性
和 `[desired_x, desired_y]` 标签。

在 Windows CUDA Python 上运行真实语言条件训练（语言文件来自已经完成的 PC Qwen
CUDA 编码；WSL 的 `python3` 不承担训练）：

```powershell
Set-Location 'C:\Users\LIU\Documents\jetson_ws\asv_vla'
& 'D:\Softwares\Python\Python313\python.exe' -m training.synthetic_geometry_train `
  --model-config training/config/model_small_v3.yaml `
  --output 'C:\Temp\asv_vla_synthetic_qwen_l7_20260805\policy_synthetic_qwen_l7_seed23.pt' `
  --language-embeddings 'C:\Temp\asv_vla_retrain_20260805\language_embeddings_130.npz' `
  --sample-count 16384 --epochs 250 --batch-size 1024 --learning-rate 0.002 --device cuda
```

输出的 checkpoint 保持现有 `model_config` + `model_state_dict` 严格加载格式，因此可由
Jetson 的 `SmallPolicyConfig`/`SmallActionPolicy` 做结构检查；checkpoint metadata 会记录
完整的输入/输出张量契约、seed、数据 schema、训练超参数和语言文件 SHA-256。默认 CLI
拒绝没有真实 embedding 的部署训练。仅做无 Torch 的几何
smoke 时才显式添加 `--synthetic-language`；该产物不能部署，也不能作为真实语言闭环证据。
当前 WSL 若无 Torch，仍可运行 `training/test/test_synthetic_geometry_train.py` 的几何、
专家方向、真实 embedding 对齐和数据序列化测试；checkpoint round-trip 测试会明确跳过，
需在安装了 PC Torch 的环境运行。
