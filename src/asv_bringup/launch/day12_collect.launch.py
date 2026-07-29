"""Single-owner Day 12 collection launch.

This launch owns TCP port 8080, the expert rollout, the UE5 kinematic
setpoint, and the episode recorder.  Start it before pressing Play in UE5.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    slot_id = LaunchConfiguration("slot_id")
    layout_id = LaunchConfiguration("layout_id")
    motion_state = LaunchConfiguration("motion_state")
    scene_seed = LaunchConfiguration("scene_seed")
    max_frames = LaunchConfiguration("max_frames")
    task_text = LaunchConfiguration("task_text")
    action = LaunchConfiguration("action")
    target_attribute = LaunchConfiguration("target_attribute")
    distance_bucket = LaunchConfiguration("distance_bucket")
    max_speed_mps = LaunchConfiguration("max_speed_mps")
    output_root = LaunchConfiguration("output_root")
    ue_bridge_config = os.path.join(
        get_package_share_directory("asv_ue_bridge"),
        "config",
        "ue_bridge.yaml",
    )

    recorder = Node(
        package="asv_vla",
        executable="record_episode",
        name="day12_episode_recorder",
        output="screen",
        parameters=[{
            "output_root": ParameterValue(output_root, value_type=str),
            "task_text": ParameterValue(task_text, value_type=str),
            "max_frames": ParameterValue(max_frames, value_type=int),
            "exit_on_complete": True,
            "execution_mode": "ue5_kinematic_expert_v1",
            "collection_slot": ParameterValue(slot_id, value_type=str),
            "layout_id": ParameterValue(layout_id, value_type=str),
            "motion_state": ParameterValue(
                motion_state, value_type=str
            ),
            "expected_scene_seed": ParameterValue(
                scene_seed, value_type=int
            ),
            "sync_cache_size": 64,
            "use_sim_time": False,
        }],
    )
    expert = Node(
        package="asv_vla",
        executable="expert_trajectory",
        name="expert_trajectory",
        output="screen",
        parameters=[{
            "run_id": "day12-expert-kinematic",
            "action": ParameterValue(action, value_type=str),
            "target_attribute": ParameterValue(
                target_attribute, value_type=str
            ),
            "distance_bucket": ParameterValue(
                distance_bucket, value_type=str
            ),
            "max_speed_mps": ParameterValue(
                max_speed_mps, value_type=float
            ),
            "use_sim_time": False,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("slot_id"),
        DeclareLaunchArgument("layout_id"),
        DeclareLaunchArgument("motion_state", default_value="S0"),
        DeclareLaunchArgument("scene_seed"),
        DeclareLaunchArgument("max_frames", default_value="100"),
        DeclareLaunchArgument(
            "task_text",
            default_value="day12 counterbalanced multimodal scene",
        ),
        DeclareLaunchArgument("action", default_value="follow"),
        DeclareLaunchArgument(
            "target_attribute",
            default_value="color:red",
        ),
        DeclareLaunchArgument("distance_bucket", default_value="3m"),
        DeclareLaunchArgument("max_speed_mps", default_value="0.15"),
        DeclareLaunchArgument(
            "output_root",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                "artifacts",
                "day8_episode",
            ]),
        ),
        Node(
            package="asv_ue_bridge",
            executable="ue_object_deliverer_bridge_node",
            name="ue_object_deliverer_bridge_node",
            output="screen",
            parameters=[
                ue_bridge_config,
                {
                    "outbound_command_mode": "kinematic",
                    "use_sim_time": True,
                },
            ],
            respawn=True,
            respawn_delay=2.0,
        ),
        expert,
        Node(
            package="asv_vla",
            executable="expert_kinematic_executor",
            name="expert_kinematic_executor",
            output="screen",
            parameters=[{
                "publish_rate_hz": 5.0,
                "source_timeout_sec": 0.5,
                "max_step_m": 0.35,
                "use_sim_time": False,
            }],
        ),
        recorder,
        RegisterEventHandler(
            OnProcessExit(
                target_action=recorder,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason="Day 12 recorder process exited"
                        )
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=expert,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason="Day 12 expert process exited"
                        )
                    )
                ],
            )
        ),
    ])
