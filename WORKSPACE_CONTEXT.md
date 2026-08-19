# Jetson 工作区事实与交接文档

> 这是 Jetson 端最终运行工作区的唯一交接入口。每次上下文压缩、开启新对话或部署/闭环验证前，先阅读本文件，再阅读 UE5 端 `D:\asv-unreal-simulation\WORKSPACE_CONTEXT.md`。本文只记录已检查过的事实；未验证内容必须标为未验证。

## 1. 唯一工作区与职责

- 唯一 Jetson 工作区：`/home/jetson/jetson_asv_ws`
- SSH：`ssh jetson@192.168.137.100`
- Git remote：`https://github.com/EnjunLiu/asv-jetson-ws.git`
- 本文建立时的代码基线 commit：`151fd37358016e1ce6e8e69962f2cc0cc989f4a5`（`chore: finalize closed-loop Jetson workspace`）。文档提交会产生新的 HEAD；每次任务开始必须执行 `git log -1 --oneline`，不能把此基线号当作永远不变的当前 HEAD。
- 本端职责：ROS 2 bridge、语言 embedding、图像感知、决策推理、安全拒止和向 UE5 发送二维机体坐标期望位移。
- 本端不负责训练、不启动 ESP32 控制器、不启动推进器分配器，也不把 UE 真值作为在线感知输入。
- `C:\Users\LIU\Documents\jetson_ws\asv_vla` 是旧的 PC 训练/历史工作区，不是当前 Jetson 运行源；不得从那里覆盖整个 Jetson 工作区。

## 2. 当前 ROS 节点与算法

系统只暴露四个节点：

| 节点 | 算法代码 | 订阅 | 发布 |
|---|---|---|---|
| `bridge_node` | `src/bridge/src/bridge_node.cpp` | TCP JSON、`/control/desired_displacement` | `/ue/camera_frame`、`/ue/asv_state`、`/ue/entities` |
| `language` | `src/vla/vla/language.py` | `/task/text` | `/vla/language_embedding` |
| `perception` | `src/vla/vla/perception.py`（节点入口 `perception_node.py`） | `/ue/camera_frame`、`/vla/language_embedding` | `/vla/entities` |
| `decision` | `src/vla/vla/decision.py`（节点入口 `decision_node.py`） | `/vla/entities`、`/vla/language_embedding`、`/ue/asv_state` | `/control/desired_displacement` |

时间跟踪、实体特征、策略推理、安全检查均为内部算法，不是独立 ROS 节点。`/ue/entities` 只用于采集/离线监督。

## 3. 消息和图像合同

- `CameraFrame.msg`：`encoding`、原始 `uint8[] data`、运行身份字段。`data` 是 UE5 原始 JPEG 字节。
- UE5 已在 SceneCapture 路径完成 FinalColorLDR、后处理、tonemapper 和一次标准 sRGB 转换。
- Jetson 必须直接解码 JPEG；禁止增加曝光、gamma、亮度、对比度或低光照预处理。
- `DesiredDisplacement.msg` 输出单步、机体坐标系二维期望位移，单位米；不是推进器 PWM。
- `Entity.bbox_*` 是感知输出中的像素框，用于感知结果/离线监督；当前决策输入明确排除 bbox，决策使用实体几何、速度/跟踪特征和任务 embedding。

## 4. 构建与启动

```bash
cd /home/jetson/jetson_asv_ws
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select interfaces bridge vla bringup
source install/setup.bash

ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=/home/jetson/jetson_asv_ws/models \
  execution_address:=192.168.137.1 execution_port:=8081 \
  task_text:="跟随红色目标船，保持3米距离"
```

默认 launch 依次启动 bridge、Qwen language、perception 和 decision；为缓解显存压力，perception/decision 有启动延迟，且 Qwen 配置为首轮编码后释放权重。一次验证只允许一套 launch 和一套 bridge。

## 5. 模型部署合同

部署目录固定为 `/home/jetson/jetson_asv_ws/models/`。当前已检查到：

| 文件 | 当前状态 |
|---|---|
| `policy.pt` | 已部署最终重训模型，SHA-256 `d9c159613c5ad37cfb61dc6aa39b80b334fe07b1663ea250083ce26c2cc1e674` |
| `perception.npz` | 已部署最终重训模型，SHA-256 `f78af3d972be0e31fae35fd9eaa45c8c14bbf08daa1c489d5daac5adbbccf11e` |
| `perception.json` | 存在但未跟踪，SHA-256 `9824f3880824df4d30367024352d7e0cf2c0cc52756ecfa2b5266465e5c0721a` |
| `Qwen3-Embedding-0.6B/model.safetensors` | 完整存在，SHA-256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| `manifest.yaml` | 已更新并通过模型文件 SHA-256 一致性检查 |

训练实验目录为 `D:\asv-vla-training\experiments\final_retrain`，数据目录为 `D:\asv-vla-training\data\episodes\asv_final_episodes`。按 R1-R4 训练、R5 验证、R6 测试，无 run/seed 泄漏。PC 端严格回读、Jetson 构建和隔离测试已通过；真实 UE5 同次运行闭环仍待验收，不能仅凭部署状态称为闭环完成。

离线 held-out rollout 最终 signed standoff error：RED 3m `0.0288 m`、BLUE 3m `0.0244 m`、RED 4m `-0.0041 m`，三类均未发散。测试集感知几何 RMSE `0.4663 m`，策略动作 RMSE `0.00208 m`，policy-driven ratio `99.34%`。

## 6. 正确的重训后部署登记

训练在 PC/D 盘进行，Jetson 只接收明确的部署包。每次部署必须同时记录：

```text
训练实验目录：D:\...
训练数据目录：D:\...
感知来源文件和 SHA-256
决策来源 checkpoint 和 SHA-256
Qwen 目录及来源/版本
perception/decision 输入输出契约
Jetson 目标路径
复制前后 SHA-256
训练时间、部署时间、Git commit
```

建议先复制到临时目录并校验，再原子替换明确文件；不能删除旧实验，不能复制整个 PC 工作区覆盖 Jetson，不能只复制一个 `.pt` 文件而不更新 manifest。

## 7. 验收证据等级

1. 源码/主机测试通过：只能证明静态逻辑。
2. `bridge_node` 与 UE5 TCP 建立连接：只能证明通信。
3. `/ue/camera_frame` 收到本次 JPEG：证明图像传输，必须可保存并检查实际文件。
4. `language READY`、`perception ready`、`POLICY_READY backend=torch_cuda`：证明对应模型实际加载。
5. 有效 `DesiredDisplacement` 且非 fail-closed：证明一次有效决策。
6. 同一次运行中 UE5 输入、CUDA 推理、位移输出和 UE5 executor apply 全部出现：才算完整闭环。

旧日志、缓存 embedding、真值 publisher、静态 checkpoint、单独 Hold_Position 或仅通信成功都不能冒充有效策略闭环。

## 8. 模型丢失/混乱的根因防护

- 代码、模型、训练数据、构建产物和运行日志必须分开；Git 只管理源码和小型合同/配置，不管理部署权重。
- `/home/jetson/jetson_asv_ws/models/` 是设备部署目录，不是训练历史库；每个文件必须有来源和 SHA-256 记录。
- 启动前先检查唯一工作区、commit、运行进程、模型路径、哈希、契约和 Qwen 完整性。
- 发现文件不存在、路径错误、哈希不匹配、契约不兼容或 CUDA OOM 时立即停止闭环并标记 fail-closed；不能用旧模型或 CPU fallback 静默替代。
- 任何清理操作必须先列出目标和备份/恢复方式；不删除 `microros_ws`，不删除训练数据，不删除旧实验作为“整理”。

## 9. 新对话固定检查顺序

```bash
ssh jetson@192.168.137.100
cd /home/jetson/jetson_asv_ws
git status --short
git log -1 --oneline
find models -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
find models -maxdepth 1 -type f -print0 | xargs -0 -r sha256sum
pgrep -af 'ros2 launch|bridge_node|language|perception|decision' || true
```

然后阅读 UE5 文档，确认 `D:\asv-unreal-simulation`、UE5 commit/源码状态、图像合同和本次运行参数。只有两份文档与现场检查一致，才能继续重训或闭环。

## 10. 2026-08-19 最终重训与闭环验收

- GitHub 推送已恢复并验证：remote 为 `https://github.com/EnjunLiu/asv-jetson-ws.git`，验证前 HEAD 为 `a5b2950`，`git push --dry-run` 返回 `Everything up-to-date`。用户已有未跟踪文件 `models/perception.json` 保留不动。
- 最终重训已完成，不得重复启动训练。产物位于 `D:\asv-vla-training\experiments\final_retrain`；感知测试几何 RMSE `0.4663 m`，策略 action RMSE `0.00208 m`，策略驱动比例 `99.34%`。
- 真实闭环已在同次运行通过：`run_id=2BEE9C1048010DE6B4320FB20F6E6034` 出现 CUDA language、perception、policy ready，bridge 发送 `Valid=true`、`Hold_Position=false` 的位移，UE5 日志出现 `SCENE_EXEC_APPLY`。示例命令为 `Delta_X_Cm=26.9896`、`Delta_Y_Cm=-12.6315`；UE5 executor 第 25 次应用记录为 `dx_cm=27.234`、`dy_cm=-12.096`。
- 启动初期的 `MISSING_OR_MISMATCHED_EGO_STATE` 是严格逐帧同步下状态尚未对齐时的 fail-closed，不是模型或 CUDA 加载失败；同次运行在缓存对齐后持续产生有效命令。不得通过取消身份校验掩盖该保护逻辑。
- 本次新鲜 JPEG 为 UE5 工作区 `Saved/CaptureDiagnostics/final_diag_camera.jpg`：`run_id=2C4BD279456EF557BE18828964E506D1`、`frame_index=8`、1280x720、48,705 bytes，亮度均值 71.64、P99 186.18、最大 219、过亮像素比例 0%，无额外 Jetson 提亮。
- 诊断场景真值目标约 80 m，和训练中的 3-4 m 评估分布不同；当前证据证明端到端闭环执行，不代表 3 m 在线跟随精度。

最后核验时间：2026-08-19（Asia/Shanghai）。
