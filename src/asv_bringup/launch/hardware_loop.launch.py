"""VLA closed loop with the hardware (ESP32) control chain.

Runs the full VLA pipeline (bridge -> encoders -> policy -> safety gate ->
trajectory controller) and feeds ``/decision/output`` into the control
manager chain instead of the simulation setpoint adapter:

    /decision/output -> control_input_mux -> /control/control_input
      -> [ESP32 firmware | fake_esp32_wrench] -> /control/asv_wrench
      -> safety_supervisor -> /control/safe_wrench
      -> thruster_allocator -> /ue/thruster_command
      -> ue_object_deliverer_bridge_node (thruster mode) -> UE5

``use_fake_esp32=true`` (default) substitutes the fake wrench node when no
ESP32 is connected; switch to false once the real firmware is online on
/dev/ttyUSB0 (see docs/esp32_interface.md).  The protected control-chain
topics and nodes are unchanged; nothing in full_system.launch.py is
modified.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_ue_bridge", default_value="true"),
            DeclareLaunchArgument(
                "model_path",
                default_value="/home/jetson/jetson_asv_ws/models/policy.onnx",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_fake_esp32", default_value="true"),
            # ── TCP bridge in thruster mode (hardware command channel) ──
            Node(
                package="asv_ue_bridge",
                executable="ue_object_deliverer_bridge_node",
                name="ue_object_deliverer_bridge_node",
                output="screen",
                parameters=[
                    ue_bridge_config,
                    {
                        "outbound_command_mode": "thruster",
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                    },
                ],
                condition=IfCondition(
                    LaunchConfiguration("start_ue_bridge")
                ),
                respawn=True,
                respawn_delay=2.0,
            ),
            # ── Language stub (selectable instruction embedding) ──
            Node(
                package="asv_vla",
                executable="language_stub",
                name="language_stub",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Visual encoder (MobileNet, CUDA) ──
            Node(
                package="asv_vla",
                executable="visual_encoder",
                name="visual_encoder",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Task entity tensor ──
            Node(
                package="asv_vla",
                executable="task_entity_tensor",
                name="task_entity_tensor",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── VLA policy inference (ONNX, CPU) ──
            Node(
                package="asv_vla",
                executable="vla_policy",
                name="vla_policy",
                output="screen",
                parameters=[
                    {"model_path": LaunchConfiguration("model_path")},
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                    },
                ],
            ),
            # ── Safety gate (sole publisher of /vla/selected_trajectory) ──
            Node(
                package="asv_vla",
                executable="safety_gate",
                name="safety_gate",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Trajectory controller ──
            Node(
                package="asv_vla",
                executable="trajectory_controller",
                name="trajectory_controller",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            # ── Hardware chain (protected topics, unchanged semantics) ──
            Node(
                package="asv_control_manager",
                executable="control_input_mux_node",
                name="control_input_mux_node",
                output="screen",
            ),
            Node(
                package="asv_tools",
                executable="fake_esp32_wrench_node",
                name="fake_esp32_wrench_node",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("use_fake_esp32")
                ),
            ),
            Node(
                package="asv_control_manager",
                executable="safety_supervisor_node",
                name="safety_supervisor_node",
                output="screen",
            ),
            Node(
                package="asv_control_manager",
                executable="thruster_allocator_node",
                name="thruster_allocator_node",
                output="screen",
            ),
            Node(
                package="asv_control_manager",
                executable="system_monitor_node",
                name="system_monitor_node",
                output="screen",
                condition=UnlessCondition(
                    LaunchConfiguration("use_fake_esp32")
                ),
            ),
        ]
    )
