# Jetson 当前任务状态（2026-08-22）

- 当前目标：暂无。
- 当前阶段：已验收 / 空闲。
- 当前阻塞：无。
- 下一步动作：空闲。


# Jetson 工作区事实与交接文档

> 这是 Jetson 端最终运行工作区的唯一交接入口。每次上下文压缩、开启新对话或部署/闭环验证前，先阅读本文件，再阅读 UE5 端 `D:\asv-unreal-simulation\WORKSPACE_CONTEXT.md`。

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
