import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    output_root = LaunchConfiguration("output_root")
    task_text = LaunchConfiguration("task_text")
    max_frames = LaunchConfiguration("max_frames")
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "output_root",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                "artifacts",
                "day8_episode",
            ]),
        ),
        DeclareLaunchArgument(
            "task_text",
            default_value="follow the red boat",
        ),
        DeclareLaunchArgument("max_frames", default_value="50"),

        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[
                ue_bridge_config,
                {"use_sim_time": True},
            ],
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="asv_vla",
            executable="record_episode",
            name="day8_episode_recorder",
            output="screen",
            parameters=[{
                "output_root": ParameterValue(
                    output_root, value_type=str
                ),
                "task_text": ParameterValue(task_text, value_type=str),
                "max_frames": ParameterValue(max_frames, value_type=int),
                "sync_cache_size": 64,
                "use_sim_time": False,
            }],
        ),
    ])
