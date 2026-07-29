# Day 12 设计性采集指南

Day 12 的目标不是训练，而是得到 12 个通过质量门的独立 Run。每个 Run
由版本控制中的 `slot_id` 唯一标识；UE5 生成自己的新 `Run_ID`，两者不能
混用。

## UE5 一次性准备

保持 Day 10 的四个实体 ID 和颜色：

- `target_red`：`color=red`
- `target_blue`：`color=blue`
- `target_left`：颜色可为 `white`
- `target_right`：颜色可为 `white`

四者始终保持 `is_target=true`、`visible=true`、ID 唯一。每次 Play：

1. 把 `Scene_Seed` 设置为当前采集槽给出的整数；
2. 生成新的非空 `Run_ID`；
3. `Frame_Index` 从 0 开始递增；
4. 不改变 Day 4 已冻结的数据字段、单位和坐标转换；
5. 继续消费 `Kinematic_Setpoint`，不要启用左右推进器路径。

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

Day 12 最小集全部使用 `S0`：目标静止或保持相同低速。动态 `S1` 和 L5
留到 30 Run 推荐集，不在当前 12 Run gate 内。

## 每个 Run 的固定流程

Jetson 先执行：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 -m training.day12_collection next --data-root .
```

工具会打印下一个槽位及唯一正确的 launch 命令，例如：

```bash
ros2 launch asv_bringup day12_collect.launch.py \
  slot_id:=L1_S0_R1 layout_id:=L1 motion_state:=S0 scene_seed:=120101
```

launch 同时启动唯一 TCP bridge、FOLLOW 专家、单点运动学执行器和 100 帧
记录器。看到 `listening` 与 `DAY8_RECORDER_READY` 后，用户再在 UE5
点击 Play。不要再启动 `day11_expert_kinematic.launch.py` 或第二个
bridge。

记录器打印 `DAY8_RECORDING_COMPLETE` 后停止 Play，然后生成和检查该
Run 的监督数据：

```bash
RUN_ID=<刚刚生成的实际 Run_ID>

ros2 run asv_vla evaluate_episode \
  artifacts/day8_episode/$RUN_ID --min-frames 80

ros2 run asv_vla build_supervised_dataset \
  --episode artifacts/day8_episode/$RUN_ID \
  --instructions dataset/language/instructions.jsonl \
  --output artifacts/day10_supervised/$RUN_ID

ros2 run asv_vla evaluate_supervised_dataset \
  artifacts/day10_supervised/$RUN_ID --require-all-labels

python3 -m training.day12_collection status \
  --data-root . \
  --report artifacts/day12_collection_report_v1.json
```

只有 episode、9/9 专家标签、四目标实体和初始布局几何全部通过，该槽
才算完成。初始几何只检查第一帧，因为运动学专家可能为保持间距而将船
转向 180°，此后 body-frame 左右关系会自然反转。失败 Run 保留日志，
不修改 manifest 冒充通过；修复 UE5 后用新 `Run_ID` 重录。

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
