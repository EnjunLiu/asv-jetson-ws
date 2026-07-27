import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def config(package_name, file_name):
    return os.path.join(
        get_package_share_directory(package_name),
        "config",
        file_name,
    )


def generate_launch_description():
    run_id = LaunchConfiguration("run_id")
    model_path = LaunchConfiguration("model_path")
    device = LaunchConfiguration("device")
    output_dim = LaunchConfiguration("output_dim")
    max_chars = LaunchConfiguration("max_chars")

    run_id_parameter = ParameterValue(run_id, value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument("run_id", default_value="language-embedding"),
        DeclareLaunchArgument(
            "model_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                "models",
                "Qwen3-Embedding-0.6B",
            ]),
        ),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("output_dim", default_value="256"),
        DeclareLaunchArgument("max_chars", default_value="512"),

        Node(
            package="asv_vla",
            executable="smoke_inputs",
            name="smoke_inputs",
            output="screen",
            parameters=[{
                "run_id": run_id_parameter,
                "scene_seed": 1,
                "jetson_git_sha": "language-embedding-working-tree",
                "esp32_git_sha": "not-connected",
                "config_sha256": "language-embedding",
                "language_model_id": "Qwen/Qwen3-Embedding-0.6B",
                "policy_model_id": "stub:none",
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="language_encoder",
            name="language_encoder",
            output="screen",
            additional_env={
                "USE_TF": "0",
                "USE_FLAX": "0",
                "USE_TORCH": "1",
            },
            parameters=[{
                "run_id": run_id_parameter,
                "model_path": ParameterValue(model_path, value_type=str),
                "model_id": "Qwen/Qwen3-Embedding-0.6B",
                "device": ParameterValue(device, value_type=str),
                "output_dim": ParameterValue(output_dim, value_type=int),
                "max_chars": ParameterValue(max_chars, value_type=int),
                "cache_size": 32,
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="stub_stack_without_language",
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
