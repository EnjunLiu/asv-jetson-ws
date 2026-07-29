# Day 12 设计性采集指南

Day 12 的目标不是训练，而是得到 12 个通过质量门的独立 Run。每个 Run
由版本控制中的 `slot_id` 唯一标识；UE5 生成自己的新 `Run_ID`，两者不能
混用。

## UE5 自动化契约

保持 Day 10 的四个实体 ID 和颜色：

- `target_red`：`color=red`
- `target_blue`：`color=blue`
- `target_left`：颜色可为 `white`
- `target_right`：颜色可为 `white`

四者始终保持 `is_target=true`、`visible=true`、ID 唯一。命令行自动化
按当前工程中的固定 Blueprint 类完成映射：

- `BP_Target_C` → `target_red`
- `BP_Target1_C` → `target_blue`
- `BP_Target2_C` → `target_left`
- `BP_Target3_C` → `target_right`

`UDay12AutomationSubsystem` 只在传入 `-Day12Auto` 时创建。它会在所有
Actor 的 BeginPlay 之前读取 slot/layout/motion/seed，写入
`Connection_C.SceneSeed` 并布置目标，所以不需要人工改 Details 面板。
每次命令行启动新的 UE 进程，现有 Blueprint 继续负责生成新 `Run_ID`
和从 0 递增的 `Frame_Index`。

以下为可直接使用的初始参考位置，单位为 UE 厘米，假设无人船初始位置
为 `(0,0)`、Yaw 为 0，UE actor-local `+X` 向前、`+Y` 向右：

| 布局 | target_red (X,Y) | target_blue (X,Y) | target_left (X,Y) | target_right (X,Y) |
| --- | --- | --- | --- | --- |
| L1 | (150, 0) | (400, 0) | (250, -150) | (250, 150) |
| L2 | (400, 0) | (150, 0) | (250, -150) | (250, 150) |
| L3 | (250, -120) | (250, 120) | (150, -180) | (400, 180) |
| L4 | (250, 120) | (250, -120) | (400, -180) | (150, 180) |

位置可以按场景等比例放大，但正式记录的第一帧中，下列关系必须至少有
0.25 m 间隔：

- L1：红船比蓝船近，`target_left` 在 `target_right` 左侧；
- L2：蓝船比红船近，`target_left` 在 `target_right` 左侧；
- L3：红船在蓝船左侧，`target_left` 比 `target_right` 近；
- L4：蓝船在红船左侧，`target_right` 比 `target_left` 近。

Scene Seed 会对四个目标施加不超过 15 cm 的确定性位置扰动。12 Run
最小集改为 S0/S1 混合，避免只得到重复静态数据：

| 布局 | R1 | R2 | R3 |
| --- | --- | --- | --- |
| L1 | S0（已录） | S0（已录） | S1 |
| L2 | S0 | S1 | S1 |
| L3 | S0 | S1 | S1 |
| L4 | S0 | S1 | S1 |

S0 强制目标保持初始位置；S1 为四个目标提供可复现且互不相同的有限速度。
S1 质量门会直接检查记录中的目标两两相对速度差，至少 60% 的前 50 帧
必须达到 0.03 m/s，不能只在 manifest 中声称目标在运动。

## 推荐：PC 一键采集

UE 自动化源码的一次性安装和真实编译：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\ue5_day12\install_day12_automation.ps1
```

一次性安装专用 SSH key；只有这一步会要求输入 Jetson 密码：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\ue5_day12\setup_day12_ssh.ps1
```

自动采集下一个槽位：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\ue5_day12\collect_day12.ps1
```

验证一个自动 Run 后，可一次完成所有剩余槽位：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\ue5_day12\collect_day12.ps1 -Count 0
```

编排器会查询 Jetson 上的下一个未完成槽位，先启动唯一 TCP bridge、
专家、运动学执行器和 recorder，确认 8080/recorder ready 后再用
`UnrealEditor.exe -game -RenderOffscreen` 启动 UE。100 帧完成后 ROS
launch 自动退出，Jetson 自动完成质量检查、监督数据构建、registry、
split 和压缩；PC 自动 SCP 并核对 SHA-256，然后才进入下一个槽位。

## 手工诊断回退

自动编排失败时，仍可在 Jetson 查询并手工启动单槽：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 -m training.day12_collection next --data-root .

ros2 launch asv_bringup day12_collect.launch.py \
  slot_id:=L1_S1_R3 layout_id:=L1 motion_state:=S1 scene_seed:=120103
```

不要同时启动 `day11_expert_kinematic.launch.py` 或第二个 bridge。只有
episode、9/9 专家标签、四目标实体、初始布局几何以及 S1 真实运动门
全部通过，该槽才算完成。初始几何只检查第一帧，因为运动学专家可能为
保持间距而将船转向 180°，此后 body-frame 左右关系会自然反转。失败
Run 保留日志，不修改 manifest 冒充通过。

## 12 Run 总验收

```bash
python3 -m training.dataset_registry \
  --data-root . \
  --output artifacts/day12_registry/dataset_registry_v1.jsonl

python3 -m training.make_group_splits \
  --registry artifacts/day12_registry/dataset_registry_v1.jsonl \
  --output artifacts/day12_registry/group_split_v1.json \
  --instructions dataset/language/instructions.jsonl

python3 -m training.day12_collection validate \
  --data-root . \
  --report artifacts/day12_collection_report_v1.json
```

最终必须同时看到：

- `DAY11_REGISTRY_PASS ... eligible_runs=12 training_ready=True`
- `DAY11_SPLIT_PASS ... train=8 val=2 test=2 training_ready=True`
- `DAY12_COLLECTION_PASS passed=12/12`

上述三个 gate 通过后才能开始 Day 13 特征缓存。
