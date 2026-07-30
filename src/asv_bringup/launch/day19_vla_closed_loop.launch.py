"""Day 19: VLA closed-loop launch (UE5 simulation).

mode:=vla     — visual encoder + entity tensor + policy + safety + control
mode:=legacy  — stub stack (Day 1 fail-closed)

The learned policy node is started only when a checkpoint path is provided.
Language is a stub embedding (Qwen excluded to stay within 8 GB memory).
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
                "checkpoint_path",
                default_value="/tmp/best_day16.pt",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            # ── TCP bridge ──
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
                    },
                ],
                condition=IfCondition(
                    LaunchConfiguration("start_ue_bridge")
                ),
                respawn=True,
                respawn_delay=2.0,
            ),
            # ── Stub language encoder (256-dim zero → safe, no OOM) ──
            Node(
                package="asv_vla",
                executable="stub_stack",
                name="language_stub",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
                remappings=[
                    ("/vla/selected_trajectory", "/vla/stub_trajectory"),
                ],
            ),
            # ── Visual encoder ──
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
            # ── VLA policy inference ──
            Node(
                package="asv_vla",
                executable="vla_policy",
                name="vla_policy",
                output="screen",
                parameters=[
                    {
                        "checkpoint_path": LaunchConfiguration(
                            "checkpoint_path"
                        ),
                    },
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                    },
                ],
            ),
            # ── Safety gate ──
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
            # ── Trajectory controller ──
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
        ]
    )
