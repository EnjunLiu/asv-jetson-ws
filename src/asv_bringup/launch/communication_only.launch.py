from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription([
        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[bridge_config, {"use_sim_time": True}],
            respawn=True,
            respawn_delay=2.0,
        ),
    ])