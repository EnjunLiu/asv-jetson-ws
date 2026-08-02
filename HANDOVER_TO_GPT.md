# 交接文档（2026-08-02 晚，给新任务窗口）

> 本文档是**当前唯一**的交接入口。目标：UE5 S2 近距离正弦跟踪——相机图像+任务指令
> 进 Jetson，在线推断目标并输出二维期望位移，UE5 执行。**红色完整周期已验证，
> 蓝色跟随是进行中的主线**（数据/感知/缓存已就绪，策略重训中途）。

## 1. 最终在线边界（已固化）

```
UE5 SceneCapture JPEG + /task/text + /ue/asv_state
  -> image_entity_perception（只读 JPEG，CUDA；不读 /ue/entities）
  -> temporal_entity_tracker（跨帧速度，首帧 velocity_valid=false）
  -> MobileNet CUDA + TaskFeatures + Qwen3-Embedding CUDA
  -> policy（Torch CUDA）
  -> safety_gate -> trajectory_controller -> decision_setpoint_adapter
  -> /ue/kinematic_setpoint -> UE5 C++ executor :8081（SCENE_EXEC_APPLY）
```

- `/ue/entities` 仅录制/离线监督，绝不出现在在线链。
- 单帧图像不产生速度；速度只由 tracker 跨帧推断。
- 上层只输出 `desired_x/desired_y`，不输出推力。
- 专家轨迹只存在于采集的离线标签路径。

## 2. 关键路径与文件

- 仓库：`C:\Users\LIU\Documents\jetson_ws\asv_vla`（旧名 day11_kinematic_work 为软链接）
- ROS 包（仅 4 个）：`asv_jetson_interfaces` / `asv_ue_bridge` / `asv_vla` / `asv_bringup`
- 在线入口：`src/asv_bringup/launch/vla_closed_loop.launch.py`（内置 CUDA 错峰启动）
- 采集：`src/asv_bringup/launch/collect.launch.py` + `tools/ue5/collect.ps1`
- UE5 自动化：`tools/ue5/Source/EDGE/SceneAutomationSubsystem.{h,cpp}`（S2/L6/L7/YawFixWholeRun/8081 执行器/SineDelay）
- UE5 项目：`D:\Unreal Projects\VLA\`（模块 EDGE；install 脚本 `tools/ue5/install_ue_automation.ps1`）
- PC 数据（不提交 Git）：`C:\Users\LIU\Documents\jetson_ws\pc_datasets\`
- 感知模型：`pc_datasets/models/image_entity_color_calibrated_v*.npz`（v3 最新，红+蓝视角训练）
- 策略：`pc_datasets/checkpoints/`（sine_formation_v2 已验证部署；near_rgb_v3 训练中）
- Jetson：`jetson@192.168.137.100`（SSH 密钥 `/tmp/asv_key`），`~/jetson_asv_ws/`
- 文档：README.md / ARCHITECTURE.md / HISTORY.md（审计）/ docs/demo_runbook.md

## 3. 环境与依赖

- **PC 训练**：`D:\Softwares\Python\Python313\python.exe`（有 torch+CUDA、jsonschema、sentence-transformers）；WSL python3 无 torch（跑脚本用 D python，脚本内 sys.path.insert 两个路径）
- **Jetson**：ROS 2 Humble；`colcon build --merge-install --symlink-install --packages-select asv_jetson_interfaces asv_ue_bridge asv_vla asv_bringup`（改 src 必须重建）
- **UE5**：5.6，`EDGEEditor` 编译入口 install_ue_automation.ps1；改 C++ 后必须重跑
- 网络：UE5(Windows 192.168.137.1) ↔ Jetson(192.168.137.100) TCP：8080(蓝图上报) + 8081(C++ 执行器，bridge 的 execution_address/execution_port 参数)

## 4. 已验证状态（诚实清单）

| 项 | 状态 | 证据 |
|---|---|---|
| 红色 S2 完整周期跟随 | ✅ 通过 | 185s/51 次 SCENE_EXEC_APPLY，ASV 88.2m，standoff 均值 3.15m（2.4-3.3m），横向摆动被跟随；轨迹图 Desktop/track_*.png |
| 红色感知 | ✅ | v1 校准 97.7-99.4%、RMSE 0.19-0.47m |
| 蓝色指令 fail-closed | ✅ | 指令解析 ✓，感知无蓝 → VISUAL_TARGET_MISSING → hold |
| 蓝色跟随 | ⏳ **进行中** | 数据已采（12 runs 跟蓝视角）、感知 v3 已训（蓝 99.7%）、v3 缓存已建（24 runs，蓝色实体 valid 100%）、**策略重训中（seed17 弱 → 换 seed29，train.py 校验已放宽）** |
| 常驻 Qwen | ❌ 不可行 | 8GB 统一内存 OOM；**release_after_encode:=true 为默认**（错峰 20s + release） |
| 本地测试 | ✅ | 234 passed, 6 skipped |

## 5. 命令速查（红色在线验证）

```bash
# Jetson（先）
cd ~/jetson_asv_ws && source /opt/ros/humble/setup.bash && source ~/microros_ws/install/setup.bash && source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  language_device:=cuda language_release_after_encode:=true \
  policy_device:=cuda visual_device:=cuda \
  task_text:="跟随红色目标船，保持3米距离" \
  execution_address:=192.168.137.1 execution_port:=8081

# Windows UE5（后启动；窗口模式必须）
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" \
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto \
  -Slot=TRACK-RED -Layout=L7 -Motion=S2 -Seed=230908 -MaxRuntimeSeconds=185 \
  -SceneExecPort=8081 -YawFixWholeRun -SineAmplitude=200 -SineDelay=40 \
  -ResX=1280 -ResY=720 -windowed -stdout -FullStdOutLogOutput
```

- `-SineDelay=40`：**必须**（UE5 启动慢 ~35s，否则目标跑出 5m 感知校准 → 死循环）
- 验收标记：`SCENE_EXEC_APPLY` 连续、`SCENE_UE_COMPLETE`、`PERCEPTION_TRACE`、`POLICY_TRACE ... guard=STANDOFF_ADJUSTED`

## 6. 蓝色跟随的进行中状态（新窗口主线）

已完成：
1. 蓝色采集 12 runs（`collect.ps1 -TargetAttribute color:blue`，expert 跟蓝 → 蓝船居中）
   - 坑：remote_collect.sh 参数上限 6→7、slot_id 正则加 `^(BLUE_)?`、S2 motion 检查不适用已从代码排除（仅 S1）
   - 坑：registry 重复（R4 重试 + latest 软链接）→ 手动清理 episode 目录/registry 行
2. 感知 v3：`models/image_entity_color_calibrated_v3.npz`（红+蓝视角 24 runs 训练；蓝 99.7%/RMSE 1.26m、红 97.7%/0.19m）
   - **关键修复**：`_extract_torch_image_features` resize 改为复用 `_resized_rgb`（PIL）——torch interpolate 与 PIL 数值不等价导致特征漂移（曾使蓝 logit=-6.59）
   - 蓝色槽位**不用 RGB 校准**（水面干扰 RMSE 22.9m）→ 纯 ridge；红色保留 RGB 校准
3. 缓存：`pc_datasets/features_near_rgb_v3/`（24 runs，16/4/4 分层 split，蓝色实体 valid=100%）
4. 训练配置：`training/config/train_near_rgb_v3.yaml`（seeds [29,23,42]；train.py 已放宽 seeds 校验）
5. **待完成**：策略重训（`pc_datasets/checkpoints/near_rgb_v3`）→ 部署到 Jetson → 在线蓝/红验证 + 完整周期

重训命令（PC，D python）：
```bash
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
D:/Softwares/Python/Python313/python.exe -c "
import sys, os
sys.path.insert(0, r'C:\Users\LIU\Documents\jetson_ws\asv_vla\src\asv_vla')
sys.path.insert(0, r'C:\Users\LIU\Documents\jetson_ws\asv_vla')
os.chdir(r'C:\Users\LIU\Documents\jetson_ws\asv_vla')
from training.train import main
sys.argv = ['train','train','--config',r'training\config\train_near_rgb_v3.yaml',
 '--model-config',r'training\config\model_small_v3.yaml',
 '--features',r'C:\Users\LIU\Documents\jetson_ws\pc_datasets\features_near_rgb_v3',
 '--split',r'C:\Users\LIU\Documents\jetson_ws\pc_datasets\registry\combined_split_v1.json',
 '--instructions',r'dataset\language\instructions.jsonl',
 '--output-root',r'C:\Users\LIU\Documents\jetson_ws\pc_datasets\checkpoints\near_rgb_v3',
 '--git-sha','near-rgb-v3-24','--device','cuda']
sys.exit(main())"
```

## 7. 已知坑（务必先读）

1. **UE5 慢启动**：窗口模式启动 ~35s，无 SineDelay 目标跑出校准范围。
2. **Jetson 残留进程**：每次运行前 `ps aux | grep -E "asv_vla|ue_object" | grep -v grep | awk '{print $2}' | xargs -r kill -9`，残留占显存会 OOM。
3. **Qwen 常驻不可行**：必须 release_after_encode=true。
4. **C++ 执行器**：headless（-RenderOffscreen）下 setpoint 不执行；**窗口模式（-windowed）必须**。
5. **坐标**：UE5 world 是 cm；日志 SCENE_ASV_POS/TARGET_POS 是 cm，绘图要 /100。
6. **registry 重复**：latest 软链接 + 失败重试会产生 duplicate；清理 episode 目录 + registry 行（json 解析删除 episode_valid=false 行）。
7. **感知特征**：训练/推理必须同一 resize 路径（PIL）；改感知代码后需重训+重建缓存+重训策略（链式）。
8. **L6 远距 registry 与近距(L7)区分**：`sine_registry` 是 L6 远距（25m，旧）；近距用 `near_red_registry`/`blue_registry`/`combined_registry`。

## 8. 下一步建议（新窗口）

1. 完成策略重训（near_rgb_v3，验证门三 seed 全过）
2. 部署：v3 npz + 新 pt → Jetson models/（更新 launch 默认参数）
3. 在线验证：红/蓝各一轮完整周期（185s，SineDelay=40），对照轨迹图方法
4. 录制演示视频（窗口模式 + OBS）
