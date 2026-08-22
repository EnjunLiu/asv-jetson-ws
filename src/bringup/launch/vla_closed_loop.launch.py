"""VLA 闭环启动文件（UE5 仿真）。

流程（无重复发布器，使用真实 Qwen 语言编码器）：

    UE5 -> bridge -> /ue/camera_frame + /ue/asv_state
                        |
                        v
              perception (image + language) -> /vla/entities
              Qwen CUDA encoder -> /vla/language_embedding
                        |
                        v
              decision (PyTorch, CUDA + safety) -> /control/desired_displacement
                        |
                        v
              bridge -> UE5 kinematic execution
                        |
                        v
              UE5 (kinematic execution)

启动文件明确不启动控制管理器、分配器或 ESP32。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    workspace_models_dir = os.path.abspath(
        os.path.join(
            get_package_share_directory("bringup"),
            "..",
            "..",
            "..",
            "..",
            "models",
        )
    )
    ue_bridge_config = os.path.join(
        get_package_share_directory("bridge"),
        "config",
        "ue_bridge.yaml",
    )
    language_node = Node(
        package="vla",
        executable="language",
        name="language",
        output="screen",
        parameters=[{
            "model_path": ParameterValue(
                LaunchConfiguration("language_model_path"),
                value_type=str,
            ),
            "device": ParameterValue(
                LaunchConfiguration("language_device"),
                value_type=str,
            ),
            "model_id": ParameterValue(
                LaunchConfiguration("language_model_id"),
                value_type=str,
            ),
            "task_description": ParameterValue(
                LaunchConfiguration("task_text"), value_type=str
            ),
            "release_model_after_encode": ParameterValue(
                LaunchConfiguration("language_release_after_encode"),
                value_type=bool,
            ),
            "use_sim_time": ParameterValue(
                LaunchConfiguration("use_sim_time"),
                value_type=bool,
            ),
        }],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_ue_bridge", default_value="true"),
            DeclareLaunchArgument(
                "models_dir",
                default_value=workspace_models_dir,
                description="Directory containing the deployment model artifacts.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution([
                    LaunchConfiguration("models_dir"),
                    "policy.pt",
                ]),
            ),
            DeclareLaunchArgument("policy_device", default_value="cuda"),
            DeclareLaunchArgument(
                "perception_model_path",
                default_value=PathJoinSubstitution([
                    LaunchConfiguration("models_dir"),
                    "perception.npz",
                ]),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "execution_address", default_value=""
            ),
            DeclareLaunchArgument("execution_port", default_value="8081"),
            DeclareLaunchArgument(
                "language_model_path",
                default_value=PathJoinSubstitution([
                    LaunchConfiguration("models_dir"),
                    "Qwen3-Embedding-0.6B",
                ]),
            ),
            DeclareLaunchArgument("language_device", default_value="cuda"),
            DeclareLaunchArgument(
                "language_model_id", default_value="Qwen/Qwen3-Embedding-0.6B"
            ),
            DeclareLaunchArgument(
                "language_release_after_encode", default_value="true"
            ),
            DeclareLaunchArgument(
                "task_text", default_value="跟随红色目标船，保持3米距离"
            ),
            # Qwen 首次 CUDA 编码并释放模型后，再创建其他 CUDA 进程。
            DeclareLaunchArgument(
                "perception_start_delay_sec", default_value="45.0"
            ),
            DeclareLaunchArgument(
                "policy_start_delay_sec", default_value="50.0"
            ),
            DeclareLaunchArgument("visual_device", default_value="cuda"),
            # ── TCP bridge（运动学输出）──
            Node(
                package="bridge",
                executable="bridge_node",
                name="bridge_node",
                output="screen",
                parameters=[
                    ue_bridge_config,
                    {
                        "outbound_command_mode": "kinematic",
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                        "execution_address": ParameterValue(
                            LaunchConfiguration("execution_address"),
                            value_type=str,
                        ),
                        "execution_port": ParameterValue(
                            LaunchConfiguration("execution_port"),
                            value_type=int,
                        ),
                    },
                ],
                condition=IfCondition(
                    LaunchConfiguration("start_ue_bridge")
                ),
                respawn=True,
                respawn_delay=2.0,
            ),
            # ── 感知与内部时序跟踪 ──
            TimerAction(
                period=LaunchConfiguration("perception_start_delay_sec"),
                actions=[
                    Node(
                        package="vla",
                        executable="perception",
                        name="perception",
                        output="screen",
                        parameters=[{
                            "model_path": LaunchConfiguration(
                                "perception_model_path"
                            ),
                            "device": ParameterValue(
                                LaunchConfiguration("visual_device"),
                                value_type=str,
                            ),
                            "use_sim_time": ParameterValue(
                                LaunchConfiguration("use_sim_time"),
                                value_type=bool,
                            ),
                        }],
                    )
                ],
            ),
            # ── 语言嵌入（真实 Qwen CUDA）──
            language_node,
            # ── 决策推理与安全检查（JetPack PyTorch、CUDA）──
            TimerAction(
                period=LaunchConfiguration("policy_start_delay_sec"),
                actions=[
                    Node(
                        package="vla",
                        executable="decision",
                        name="decision",
                        output="screen",
                        parameters=[
                            {
                                "model_path": LaunchConfiguration(
                                    "model_path"
                                ),
                                "device": LaunchConfiguration(
                                    "policy_device"
                                ),
                                "language_release_after_encode": ParameterValue(
                                    LaunchConfiguration(
                                        "language_release_after_encode"
                                    ),
                                    value_type=bool,
                                ),
                            },
                            {
                                "use_sim_time": ParameterValue(
                                    LaunchConfiguration("use_sim_time"),
                                    value_type=bool,
                                ),
                            },
                        ],
                    )
                ],
            ),
        ]
    )
