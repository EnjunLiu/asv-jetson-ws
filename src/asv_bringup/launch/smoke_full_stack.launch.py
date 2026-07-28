from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
    run_id = LaunchConfiguration("run_id")
    jetson_git_sha = LaunchConfiguration("jetson_git_sha")
    run_id_parameter = ParameterValue(run_id, value_type=str)
    jetson_git_sha_parameter = ParameterValue(jetson_git_sha, value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument("run_id", default_value="day1-smoke"),
        DeclareLaunchArgument(
            "jetson_git_sha",
            default_value="unknown-unset",
            description="Exact Jetson repository commit under test.",
        ),

        Node(
            package="asv_vla",
            executable="smoke_inputs",
            name="smoke_inputs",
            output="screen",
            parameters=[{
                "run_id": run_id_parameter,
                "scene_seed": 1,
                "jetson_git_sha": jetson_git_sha_parameter,
                "esp32_git_sha": "not-connected",
                "config_sha256": "day1-smoke-v2",
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="stub_stack",
            output="screen",
            parameters=[{
                "run_id": run_id_parameter,
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_control_manager",
            executable="control_input_mux_node",
            name="control_input_mux_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": False},
            ],
        ),
        Node(
            package="asv_tools",
            executable="fake_esp32_wrench_node",
            name="fake_esp32_wrench_node",
            output="screen",
            parameters=[{
                "force": 0.0,
                "moment": 0.0,
                "valid": False,
                "rate_hz": 10.0,
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_control_manager",
            executable="safety_supervisor_node",
            name="safety_supervisor_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": False},
            ],
        ),
        Node(
            package="asv_control_manager",
            executable="thruster_allocator_node",
            name="thruster_allocator_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": False},
            ],
        ),
        Node(
            package="asv_control_manager",
            executable="system_monitor_node",
            name="system_monitor_node",
            output="screen",
            parameters=[
                config("asv_control_manager", "control_manager.yaml"),
                {"use_sim_time": False},
            ],
        ),
    ])
