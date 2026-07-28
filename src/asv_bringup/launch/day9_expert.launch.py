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
    start_probe = LaunchConfiguration("start_probe")

    return LaunchDescription([
        DeclareLaunchArgument("action", default_value="follow"),
        DeclareLaunchArgument(
            "target_attribute", default_value="color:red"
        ),
        DeclareLaunchArgument("distance_bucket", default_value="3m"),
        DeclareLaunchArgument("max_speed_mps", default_value="1.5"),
        DeclareLaunchArgument("start_probe", default_value="true"),

        Node(
            package="asv_vla",
            executable="expert_trajectory",
            name="expert_trajectory",
            output="screen",
            parameters=[{
                "run_id": "day9-expert",
                "action": ParameterValue(action, value_type=str),
                "target_attribute": ParameterValue(
                    target_attribute, value_type=str
                ),
                "distance_bucket": ParameterValue(
                    distance_bucket, value_type=str
                ),
                "max_speed_mps": ParameterValue(
                    max_speed_mps, value_type=float
                ),
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="expert_trajectory_probe",
            name="expert_trajectory_probe",
            output="screen",
            parameters=[{"use_sim_time": False}],
            condition=IfCondition(start_probe),
        ),
    ])
