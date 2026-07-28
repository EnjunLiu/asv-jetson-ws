import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    run_id = LaunchConfiguration("run_id")
    start_ue_bridge = LaunchConfiguration("start_ue_bridge")
    use_sim_time = LaunchConfiguration("use_sim_time")
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument("run_id", default_value="task-entity-tensor"),
        DeclareLaunchArgument("start_ue_bridge", default_value="true"),
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
            executable="task_entity_tensor",
            name="task_entity_tensor",
            output="screen",
            parameters=[{
                "run_id": ParameterValue(run_id, value_type=str),
                "max_entities": 16,
                "risk_horizon_sec": 4.0,
                "risk_radius_m": 3.0,
                "use_sim_time": ParameterValue(
                    use_sim_time, value_type=bool
                ),
            }],
        ),
    ])
