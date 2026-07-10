# Jetson ASV ROS2 工作空间

基于 ROS2 的 无人船感知与决策系统工作空间。

## 功能简介

- 接收低频感知原始数据，输出高频决策
- 基于 micro-ROS agent 实现与下位机串口通信
- 基于 TCP/IP 与 UE5 通信实现控制指令回传与数据解析

## 实现细节

- ArUco占位图像处理相关内容
- 最小二乘拟合预测器实现低频数据的高频预测

## 使用方式


```
source /opt/ros/humble/setup.bash
source /microros_ws/install/setup.bash
colcon build
source install/setup.bash
ros2 launch asv_bringup fullsystem.launch.py
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -v6
```

## 测试环境

- jetson orin nano 8GB
- ROS2 humble
- micro-ROS agent humble