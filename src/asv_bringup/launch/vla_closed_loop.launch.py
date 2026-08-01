"""Day 19/20: VLA closed-loop launch (UE5 simulation).

Pipeline (no duplicate publishers, no stub stack):

    UE5 -> bridge -> /ue/camera_frame + /ue/entities
                        |
                        v
              visual_encoder -> /vla/visual_features
              task_entity_tensor -> /vla/task_features
              language_stub   -> /vla/language_embedding
                        |
                        v
              vla_policy (ONNX, CPU) -> /vla/policy_trajectory
                        |
                        v
              safety_gate -> /vla/selected_trajectory
                        |
                        v
              trajectory_controller -> /decision/output
                        |
                        v
              decision_setpoint_adapter -> /ue/kinematic_setpoint
                        |
                        v
              UE5 (kinematic execution)

The launch intentionally starts NO control manager, allocator, or ESP32.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_ue_bridge", default_value="true"),
            DeclareLaunchArgument(
                "model_path",
                default_value="/home/jetson/jetson_asv_ws/models/policy.onnx",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "execution_address", default_value=""
            ),
            # ── TCP bridge (kinematic outbound) ──
            Node(
                package="asv_ue_bridge",
                executable="ue_object_deliverer_bridge_node",
                name="ue_object_deliverer_bridge_node",
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
                    },
                ],
                condition=IfCondition(
                    LaunchConfiguration("start_ue_bridge")
                ),
                respawn=True,
                respawn_delay=2.0,
            ),
            # ── Language stub (zero embedding; Qwen excluded to save memory) ──
            Node(
                package="asv_vla",
                executable="language_stub",
                name="language_stub",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Visual encoder (MobileNet, CUDA) ──
            Node(
                package="asv_vla",
                executable="visual_encoder",
                name="visual_encoder",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Task entity tensor ──
            Node(
                package="asv_vla",
                executable="task_entity_tensor",
                name="task_entity_tensor",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── VLA policy inference (ONNX, CPU) ──
            Node(
                package="asv_vla",
                executable="vla_policy",
                name="vla_policy",
                output="screen",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                    },
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                    },
                ],
            ),
            # ── Safety gate (sole publisher of /vla/selected_trajectory) ──
            Node(
                package="asv_vla",
                executable="safety_gate",
                name="safety_gate",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Trajectory controller (prefix execution -> desired_x/y) ──
            Node(
                package="asv_vla",
                executable="trajectory_controller",
                name="trajectory_controller",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Adapter: /decision/output -> /ue/kinematic_setpoint ──
            Node(
                package="asv_vla",
                executable="decision_setpoint_adapter",
                name="decision_setpoint_adapter",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
        ]
    )
