"""Day 17 launch: safety gate node without UE5 or the learned policy.

The safety gate subscribes to ``/vla/policy_trajectory`` and is the sole
publisher of ``/vla/selected_trajectory``.  This launch does NOT start any
TCP bridge, UE5 receiver, or control actuators.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("max_step_m", default_value="0.3"),
            DeclareLaunchArgument("max_total_displacement_m", default_value="10.0"),
            DeclareLaunchArgument("max_curvature", default_value="2.0"),
            DeclareLaunchArgument("stale_timeout_sec", default_value="1.0"),
            DeclareLaunchArgument("estop_timeout_sec", default_value="2.0"),
            DeclareLaunchArgument("collision_margin_m", default_value="1.0"),
            Node(
                package="asv_vla",
                executable="safety_gate",
                name="safety_gate",
                output="screen",
                parameters=[
                    {
                        "max_step_m": ParameterValue(
                            LaunchConfiguration("max_step_m"),
                            value_type=float,
                        ),
                        "max_total_displacement_m": ParameterValue(
                            LaunchConfiguration("max_total_displacement_m"),
                            value_type=float,
                        ),
                        "max_curvature": ParameterValue(
                            LaunchConfiguration("max_curvature"),
                            value_type=float,
                        ),
                        "stale_timeout_sec": ParameterValue(
                            LaunchConfiguration("stale_timeout_sec"),
                            value_type=float,
                        ),
                        "estop_timeout_sec": ParameterValue(
                            LaunchConfiguration("estop_timeout_sec"),
                            value_type=float,
                        ),
                        "collision_margin_m": ParameterValue(
                            LaunchConfiguration("collision_margin_m"),
                            value_type=float,
                        ),
                        "use_sim_time": False,
                    }
                ],
            ),
        ]
    )
