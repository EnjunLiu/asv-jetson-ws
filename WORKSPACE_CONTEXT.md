# Jetson 工作区事实与交接文档

## 0. 当前任务状态（2026-08-22）

- 当前唯一目标：已完成最终三场景软件 HIL 闭环验收，并完成 GitHub 同步。
- 当前阶段：已验收。
- 已完成证据（现场验证）：
  - `models/manifest.yaml` policy SHA-256 已与 live `policy.pt` 对齐：`E0975763559D60E47A3CDC78A16C93512017B204826D979F9B2BCAD7A0CADA36`（`chase_standoff_candidate`）
  - RED3 / BLUE3 / RED4 launch 均出现 `LANGUAGE_READY_VALID` + CUDA perception ready + `POLICY_READY` + `0.0.0.0:8080`
  - UE 侧各场 `SCENE_EXEC_APPLY≈300` / 180s；结果图在 PC 桌面 `track_world_chase_standoff_2x3.png`
  - 指标摘要：RED4 mean_abs≈0.35 final≈-0.03；RED3 mean_abs≈1.24 final≈1.02；BLUE3 mean_abs≈1.50 final≈1.35
  - Git：`asv-jetson-ws` 已推送 `9d18815`；总览 README：`asv-hil-platform@b5c574d`
- 当前阻塞：无。注意 Orin 内存紧张时 Qwen CUDA 加载会 OOM，需先清掉残留 VLA 进程再 launch。
- 下一步唯一动作：空闲。VLA 进程已停止。

## 1. 唯一工作区与职责

- 唯一 Jetson 工作区：`/home/jetson/jetson_asv_ws`
- SSH：`ssh jetson@192.168.137.100`
- 本端职责：ROS 2 bridge、语言 embedding、图像感知、决策推理、安全拒止和向 UE5 发送二维机体坐标期望位移。
- 本端不负责训练、不启动 ESP32 控制器、不启动推进器分配器，也不把 UE 真值作为在线感知输入。
## 1. 唯一工作区与职责

- 唯一 Jetson 工作区：`/home/jetson/jetson_asv_ws`
- SSH：`ssh jetson@192.168.137.100`
- Git remote：`https://github.com/EnjunLiu/asv-jetson-ws.git`
- 本端职责：ROS 2 bridge、语言 embedding、图像感知、决策推理、向 UE5 发送二维机体坐标期望位移
- 对应 UE5：`D:\asv-unreal-simulation`
- 对应训练：`D:\asv-vla-training`
- 旧 PC 路径 `C:\Users\LIU\Documents\jetson_ws\asv_vla` 已清除

## 2. 活动模型（清理后）

```text
/home/jetson/jetson_asv_ws/models/policy.pt
/home/jetson/jetson_asv_ws/models/perception.npz
/home/jetson/jetson_asv_ws/models/perception.json
/home/jetson/jetson_asv_ws/models/qwen_final_embeddings.npz
/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B/
/home/jetson/jetson_asv_ws/models/manifest.yaml
```

## 3. 闭环要点

- launch：`ros2 launch bringup vla_closed_loop.launch.py`
- `execution_address:=192.168.137.1` `execution_port:=8081`
- 须等 language / perception / policy CUDA ready 且 `0.0.0.0:8080` 监听后再开 UE5
- 结束进程时按 **PID** 杀掉，勿对路径字符串盲目 `pkill -f`
