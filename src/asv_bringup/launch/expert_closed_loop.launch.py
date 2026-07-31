"""Day 19/20: closed loop with the deterministic expert as the controller.

The project spec sanctions the expert as the control-path reference when
the learned policy is unstable (its per-frame outputs oscillate under the
dynamic UE5 water simulation).  The full pipeline is otherwise identical to
vla_closed_loop.launch.py: bridge -> visual/task/language encoders ->
expert -> expert_policy_bridge -> safety_gate -> trajectory_controller ->
decision_setpoint_adapter -> UE5.  The safety gate is never bypassed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            # ── TCP bridge (kinematic outbound) ──
            Node(
                package="asv_ue_bridge",
                executable="ue_object_deliverer_bridge_node",
                name="ue_object_deliverer_bridge_node",
                output="screen",
                parameters=[
                    ue_bridge_config,
                    {
                        "outbound_command_mode": "kinematic",
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
            # ── Language stub (pre-computed instruction embedding) ──
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
            # ── Deterministic expert (follow red at 3 m standoff) ──
            Node(
                package="asv_vla",
                executable="expert_trajectory",
                name="expert_trajectory",
                output="screen",
                parameters=[{
                    "action": "follow",
                    "target_attribute": "color:red",
                    "distance_bucket": "3m",
                }],
            ),
            # ── Expert -> /vla/policy_trajectory bridge ──
            Node(
                package="asv_vla",
                executable="expert_policy_bridge",
                name="expert_policy_bridge",
                output="screen",
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
            # ── Trajectory controller (prefix execution -> desired_x/y) ──
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
            # ── Adapter: /decision/output -> /ue/kinematic_setpoint ──
            Node(
                package="asv_vla",
                executable="decision_setpoint_adapter",
                name="decision_setpoint_adapter",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
        ]
    )
