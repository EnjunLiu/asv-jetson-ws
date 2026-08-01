"""VLA closed-loop launch (UE5 simulation).

Pipeline (no duplicate publishers, one selectable language backend):

    UE5 -> bridge -> /ue/camera_frame + /ue/asv_state
                        |
                        v
              image_perception (image only) -> /vla/perceived_entities
              temporal_tracker -> /vla/tracked_entities
              visual_encoder -> /vla/visual_features
              task_entity_tensor -> /vla/task_features
              language_backend -> /vla/language_embedding
                        |
                        v
              vla_policy (ONNX, CPU) -> /vla/policy_trajectory
                        |
                        v
              safety_gate -> /vla/selected_trajectory
                        |
                        v
              trajectory_controller -> /decision/output
                        |
                        v
              decision_setpoint_adapter -> /ue/kinematic_setpoint
                        |
                        v
              UE5 (kinematic execution)

The launch intentionally starts NO control manager, allocator, or ESP32.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )
    staged_language = PythonExpression(
        [
            "'",
            LaunchConfiguration("language_backend"),
            "' == 'qwen' and '",
            LaunchConfiguration("language_release_after_encode"),
            "' == 'true'",
        ]
    )

    visual_parameters = [{
        "entities_topic": "/vla/tracked_entities",
        "device": ParameterValue(
            LaunchConfiguration("visual_device"), value_type=str
        ),
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        ),
    }]
    visual_encoder_node = Node(
        package="asv_vla",
        executable="visual_encoder",
        name="visual_encoder",
        output="screen",
        parameters=visual_parameters,
        condition=UnlessCondition(staged_language),
    )
    staged_visual_encoder = Node(
        package="asv_vla",
        executable="visual_encoder",
        name="visual_encoder",
        output="screen",
        parameters=visual_parameters,
    )

    language_qwen_node = Node(
        package="asv_vla",
        executable="language_qwen",
        name="language_qwen",
        output="screen",
        parameters=[{
            "model_path": ParameterValue(
                LaunchConfiguration("language_model_path"),
                value_type=str,
            ),
            "device": ParameterValue(
                LaunchConfiguration("language_device"),
                value_type=str,
            ),
            "model_id": ParameterValue(
                LaunchConfiguration("language_model_id"),
                value_type=str,
            ),
            "release_model_after_encode": ParameterValue(
                LaunchConfiguration("language_release_after_encode"),
                value_type=bool,
            ),
            "use_sim_time": ParameterValue(
                LaunchConfiguration("use_sim_time"),
                value_type=bool,
            ),
        }],
        condition=IfCondition(
            PythonExpression(
                ["'", LaunchConfiguration("language_backend"), "' == 'qwen'"]
            )
        ),
    )
    staged_visual_action = TimerAction(
        # The delay is an explicit bounded startup guard for Jetson unified
        # memory; the Qwen node still fails closed if CUDA/model startup fails.
        period=LaunchConfiguration("language_staging_delay_sec"),
        actions=[staged_visual_encoder],
        condition=IfCondition(staged_language),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_ue_bridge", default_value="true"),
            DeclareLaunchArgument(
                "model_path",
                default_value=(
                    "/home/jetson/jetson_asv_ws/models/"
                    "policy_sine_near_image_color_seed42.onnx"
                ),
            ),
            DeclareLaunchArgument(
                "perception_model_path",
                default_value=(
                    "/home/jetson/jetson_asv_ws/models/"
                    "image_entity_color_calibrated_v1.npz"
                ),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "execution_address", default_value=""
            ),
            DeclareLaunchArgument("execution_port", default_value="8081"),
            DeclareLaunchArgument(
                "embedding_path",
                default_value=(
                    "/home/jetson/jetson_asv_ws/models/"
                    "demo_instruction_embedding.npy"
                ),
            ),
            DeclareLaunchArgument("active_embedding", default_value=""),
            DeclareLaunchArgument("language_backend", default_value="stub"),
            DeclareLaunchArgument(
                "language_model_path",
                default_value=(
                    "/home/jetson/jetson_asv_ws/models/"
                    "Qwen3-Embedding-0.6B"
                ),
            ),
            DeclareLaunchArgument("language_device", default_value="cuda"),
            DeclareLaunchArgument(
                "language_model_id", default_value="Qwen3-Embedding-0.6B"
            ),
            DeclareLaunchArgument(
                "language_release_after_encode", default_value="false"
            ),
            DeclareLaunchArgument(
                "language_staging_delay_sec", default_value="30.0"
            ),
            DeclareLaunchArgument(
                "task_text", default_value="跟随红色目标船，保持3米距离"
            ),
            DeclareLaunchArgument("visual_device", default_value="cuda"),
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
                        "execution_address": ParameterValue(
                            LaunchConfiguration("execution_address"),
                            value_type=str,
                        ),
                        "execution_port": ParameterValue(
                            LaunchConfiguration("execution_port"),
                            value_type=int,
                        ),
                    },
                ],
                condition=IfCondition(
                    LaunchConfiguration("start_ue_bridge")
                ),
                respawn=True,
                respawn_delay=2.0,
            ),
            # ── Image-only perception (UE truth is not an input) ──
            Node(
                package="asv_vla",
                executable="image_entity_perception",
                name="image_entity_perception",
                output="screen",
                parameters=[{
                    "model_path": LaunchConfiguration("perception_model_path"),
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"), value_type=bool
                    ),
                }],
            ),
            Node(
                package="asv_vla",
                executable="temporal_entity_tracker",
                name="temporal_entity_tracker",
                output="screen",
                parameters=[{
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"), value_type=bool
                    ),
                }],
            ),
            # ── Language embedding (stub default, Qwen CUDA selectable) ──
            Node(
                package="asv_vla",
                executable="language_stub",
                name="language_stub",
                output="screen",
                parameters=[{
                    "embedding_path": ParameterValue(
                        LaunchConfiguration("embedding_path"),
                        value_type=str,
                    ),
                    "active_embedding": ParameterValue(
                        LaunchConfiguration("active_embedding"),
                        value_type=str,
                    ),
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
                condition=UnlessCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("language_backend"),
                            "' == 'qwen'",
                        ]
                    )
                ),
            ),
            Node(
                package="asv_vla",
                executable="task_instruction",
                name="task_instruction",
                output="screen",
                parameters=[{
                    "task_text": ParameterValue(
                        LaunchConfiguration("task_text"),
                        value_type=str,
                    ),
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                }],
            ),
            language_qwen_node,
            staged_visual_action,
            # ── Visual encoder (MobileNet, CUDA) ──
            visual_encoder_node,
            # ── Task entity tensor ──
            Node(
                package="asv_vla",
                executable="task_entity_tensor",
                name="task_entity_tensor",
                output="screen",
                parameters=[{
                    "entities_topic": "/vla/tracked_entities",
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
                    {
                        "model_path": LaunchConfiguration("model_path"),
                    },
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
                    "entities_topic": "/vla/tracked_entities",
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
