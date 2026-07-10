from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def config(package_name, file_name):
    return os.path.join(
        get_package_share_directory(package_name),
        "config",
        file_name,
    )


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[
                config("asv_ue_bridge", "ue_bridge.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_perception",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[
                config("asv_perception", "perception.yaml"),
                {"mode": "ground_truth_passthrough"},
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_planning",
            executable="state_predictor_node",
            name="state_predictor_node",
            output="screen",
            parameters=[
                config("asv_planning", "planning.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_planning",
            executable="decision_node",
            name="decision_node",
            output="screen",
            parameters=[
                config("asv_planning", "planning.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_control_manager",
            executable="control_input_mux_node",
            name="control_input_mux_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_control_manager",
            executable="safety_supervisor_node",
            name="safety_supervisor_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_control_manager",
            executable="thruster_allocator_node",
            name="thruster_allocator_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_control_manager",
            executable="system_monitor_node",
            name="system_monitor_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="asv_tools",
            executable="fake_esp32_wrench_node",
            name="fake_esp32_wrench_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
    ])
