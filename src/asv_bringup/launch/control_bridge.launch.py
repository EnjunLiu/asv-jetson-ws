"""Day 18: safety gate + trajectory control bridge (no UE5, no learned policy)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="asv_vla",
                executable="safety_gate",
                name="safety_gate",
                output="screen",
                parameters=[{"use_sim_time": False}],
            ),
            Node(
                package="asv_vla",
                executable="trajectory_controller",
                name="trajectory_controller",
                output="screen",
                parameters=[{"use_sim_time": False}],
            ),
        ]
    )
