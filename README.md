# ASV VLA

这是一个面向 UE5 S2 正弦跟踪场景的模块化 VLA 原型。最终在线边界只有：

```text
UE5 JPEG + 任务指令 + 本船 UEASVState
  -> 图像+指令感知 -> 任务相关 Entities -> 跨帧速度 -> ego 对齐
  -> CUDA 视觉/Qwen/策略 -> 20x2 期望位移轨迹
  -> 安全门 -> desired_x/desired_y -> UE5 运动学执行
```

UE5 的 `/ue/entities` 仍可随图像一起发送，但只进入 recorder 和 PC 离线监督；它不
进入在线感知、实体张量或策略。专家轨迹同样只用于离线标签。VLA 不输出左右推力，
底层推力控制器可在独立工程中调参而不改变本仓库接口。

## 当前活动面

仓库只保留四个 ROS 2 包：

- `asv_jetson_interfaces`：UE5/VLA 消息
- `asv_ue_bridge`：TCP JSON、JPEG、本船状态和运动学 setpoint bridge
- `asv_vla`：图像实体感知、tracker、视觉/Qwen/策略、安全门和采集器
- `asv_bringup`：在线闭环、离线采集、回放 launch

旧的 CPU/ONNX、语言 `.npy` stub、旧硬件控制 ROS 包和历史训练配置已经从活动树移除。
可恢复副本位于本机 `/tmp/asv_vla_cleanup_20260802/`，不属于运行时。

PC 训练数据和模型不提交进 Git，唯一活动目录是：

```text
C:\Users\LIU\Documents\jetson_ws\pc_datasets
```

其中保留 near S2 原始帧、冻结特征、seed42 policy、图像校准器和 Qwen3-Embedding-0.6B。

## Jetson 在线启动

先启动 Jetson：

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/setup.bash
colcon build --merge-install --symlink-install --packages-select asv_vla asv_bringup asv_ue_bridge asv_jetson_interfaces
source install/setup.bash
ros2 launch asv_bringup vla_closed_loop.launch.py \
  model_path:=/home/jetson/jetson_asv_ws/models/policy_sine_near_image_color_seed42.pt \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/image_entity_color_calibrated_v1.npz \
  language_device:=cuda language_release_after_encode:=true \
  policy_device:=cuda visual_device:=cuda \
  execution_address:=192.168.137.1 execution_port:=8081
```

实测 Orin Nano 8 GB 统一内存下常驻 Qwen 与其他 CUDA 模型并发会 OOM，因此默认
`language_release_after_encode:=true`（Qwen 首次 CUDA 编码后释放权重，仍为真实
Qwen embedding，不得改回 `.npy` 或 CPU）；launch 已对 CUDA 模型错峰启动。

然后启动 UE5，项目文件必须是 UnrealEditor 的第一个参数：

```powershell
& "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe" `
  "D:\Unreal Projects\VLA\VLA.uproject" /Game/Main_Map -game -SceneAuto `
  -Slot=FINAL-S2-230908 -Layout=L7 -Motion=S2 -Seed=230908 `
  -MaxRuntimeSeconds=35 -SceneExecPort=8081 -YawFixWholeRun `
  -ResX=1280 -ResY=720 -windowed
```

验收时应看到：

```text
image_entity_perception ... device=cuda
LANGUAGE_READY_VALID ... device=cuda
visual_encoder ... device=cuda
POLICY_READY backend=torch_cuda device=cuda
PERCEPTION_TRACE ... source=image_perception
POLICY_TRACE ... ego_valid=true
SCENE_EXEC_APPLY ...
SCENE_UE_COMPLETE ...
```

## 数据采集与训练

训练数据采集单独运行，不与在线 VLA 同时启动：

```bash
ros2 launch asv_bringup collect.launch.py \
  slot_id:=L7 layout_id:=L7 motion_state:=S2 scene_seed:=230908 \
  execution_address:=192.168.137.1 execution_port:=8081
```

采集器保存 JPEG、`UEASVState`、UE Entities 和身份元数据。PC 训练时 Entities 只作为
监督标签，速度由相邻图像实体和时间戳计算；策略训练输入是图像推断的 Entities、
语言 embedding 和真实 ego。

## 本地验证

```bash
cd /mnt/c/Users/LIU/Documents/jetson_ws/asv_vla
PYTHONPATH=src/asv_vla pytest -q
git diff --check
```

不允许并行启动第二套 bridge、expert、policy 或 safety gate。CUDA/模型/身份/ego 任一
项失败，都必须输出 `valid=false` 并 hold。

详细边界见 [ARCHITECTURE.md](ARCHITECTURE.md)、[TODO.md](TODO.md) 和
[docs/demo_runbook.md](docs/demo_runbook.md)。
