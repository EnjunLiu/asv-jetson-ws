import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    action = LaunchConfiguration("action")
    target_attribute = LaunchConfiguration("target_attribute")
    distance_bucket = LaunchConfiguration("distance_bucket")
    max_speed_mps = LaunchConfiguration("max_speed_mps")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    source_timeout_sec = LaunchConfiguration("source_timeout_sec")
    max_step_m = LaunchConfiguration("max_step_m")
    start_ue_bridge = LaunchConfiguration("start_ue_bridge")
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument("action", default_value="follow"),
        DeclareLaunchArgument(
            "target_attribute",
            default_value="color:red",
        ),
        DeclareLaunchArgument("distance_bucket", default_value="3m"),
        DeclareLaunchArgument("max_speed_mps", default_value="1.5"),
        DeclareLaunchArgument("publish_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("source_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("max_step_m", default_value="0.35"),
        DeclareLaunchArgument("start_ue_bridge", default_value="true"),

        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[
                ue_bridge_config,
                {
                    "outbound_command_mode": "kinematic",
                    "use_sim_time": True,
                },
            ],
            condition=IfCondition(start_ue_bridge),
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="asv_vla",
            executable="expert_trajectory",
            name="expert_trajectory",
            output="screen",
            parameters=[{
                "run_id": "day11-expert-kinematic",
                "action": ParameterValue(action, value_type=str),
                "target_attribute": ParameterValue(
                    target_attribute,
                    value_type=str,
                ),
                "distance_bucket": ParameterValue(
                    distance_bucket,
                    value_type=str,
                ),
                "max_speed_mps": ParameterValue(
                    max_speed_mps,
                    value_type=float,
                ),
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="expert_kinematic_executor",
            name="expert_kinematic_executor",
            output="screen",
            parameters=[{
                "publish_rate_hz": ParameterValue(
                    publish_rate_hz,
                    value_type=float,
                ),
                "source_timeout_sec": ParameterValue(
                    source_timeout_sec,
                    value_type=float,
                ),
                "max_step_m": ParameterValue(
                    max_step_m,
                    value_type=float,
                ),
                "use_sim_time": False,
            }],
        ),
    ])
