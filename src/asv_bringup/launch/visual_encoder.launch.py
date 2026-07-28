import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    run_id = LaunchConfiguration("run_id")
    python_executable = LaunchConfiguration("python_executable")
    device = LaunchConfiguration("device")
    start_ue_bridge = LaunchConfiguration("start_ue_bridge")
    use_sim_time = LaunchConfiguration("use_sim_time")
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument("run_id", default_value="visual-encoder"),
        DeclareLaunchArgument(
            "python_executable",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                ".venv",
                "bin",
                "python",
            ]),
            description="Python environment containing Jetson torch/torchvision.",
        ),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument(
            "start_ue_bridge",
            default_value="true",
            description="Start the UE5 ObjectDeliverer TCP bridge on port 8080.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[
                ue_bridge_config,
                {
                    "use_sim_time": ParameterValue(
                        use_sim_time, value_type=bool
                    )
                },
            ],
            condition=IfCondition(start_ue_bridge),
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="asv_vla",
            executable="visual_encoder",
            name="visual_encoder",
            output="screen",
            prefix=[python_executable],
            parameters=[{
                "run_id": ParameterValue(run_id, value_type=str),
                "device": ParameterValue(device, value_type=str),
                "image_width": 1280,
                "image_height": 720,
                "horizontal_fov_deg": 90.0,
                "camera_mount_x_m": 0.42,
                "camera_mount_y_m": 0.0,
                "camera_mount_z_m": 0.20,
                "camera_pitch_deg": -5.0,
                "target_crop_size_px": 224,
                "entity_wait_sec": 0.25,
                "sync_cache_size": 16,
                "use_sim_time": ParameterValue(
                    use_sim_time, value_type=bool
                ),
            }],
        ),
    ])
