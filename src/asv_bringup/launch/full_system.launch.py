from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def config(package_name, file_name):
    return os.path.join(
        get_package_share_directory(package_name),
        "config",
        file_name,
    )


def generate_launch_description():
    start_agent = LaunchConfiguration("start_agent")
    serial_dev = LaunchConfiguration("serial_dev")
    baudrate = LaunchConfiguration("baudrate")
    start_param_manager = LaunchConfiguration("start_param_manager")
    perception_mode = LaunchConfiguration("perception_mode")

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_agent",
            default_value="false",
            description="Start micro_ros_agent in this launch file.",
        ),
        DeclareLaunchArgument("serial_dev", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baudrate", default_value="115200"),
        DeclareLaunchArgument(
            "start_param_manager",
            default_value="false",
            description="Apply the YAML controller parameters to ESP32 once.",
        ),
        DeclareLaunchArgument(
            "perception_mode",
            default_value="aruco",
            description="Use 2 Hz Jetson ArUco perception; ground truth remains available for tests.",
        ),

        Node(
            package="micro_ros_agent",
            executable="micro_ros_agent",
            name="micro_ros_agent",
            output="screen",
            arguments=[
                "serial",
                "--dev", serial_dev,
                "--baudrate", baudrate,
                "-v6",
            ],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(start_agent),
            respawn=True,
            respawn_delay=2.0,
        ),

        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[
                config("asv_ue_bridge", "ue_bridge.yaml"),
                {"use_sim_time": True},
            ],
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="asv_perception",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[
                config("asv_perception", "perception.yaml"),
                {"mode": ParameterValue(perception_mode, value_type=str)},
                {"use_sim_time": True},
            ],
            respawn=True,
            respawn_delay=2.0,
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
            respawn=True,
            respawn_delay=2.0,
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
            respawn=True,
            respawn_delay=2.0,
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
            respawn=True,
            respawn_delay=2.0,
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
            respawn=True,
            respawn_delay=2.0,
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
            respawn=True,
            respawn_delay=2.0,
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
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="asv_control_manager",
            executable="esp32_param_manager_node",
            name="esp32_param_manager_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": True},
            ],
            condition=IfCondition(start_param_manager),
        ),
    ])
