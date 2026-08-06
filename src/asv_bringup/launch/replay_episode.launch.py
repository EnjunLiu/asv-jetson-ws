from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    episode_dir = LaunchConfiguration("episode_dir")
    python_executable = LaunchConfiguration("python_executable")
    device = LaunchConfiguration("device")
    perception_model = LaunchConfiguration("perception_model")
    replay_rate_hz = LaunchConfiguration("replay_rate_hz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "episode_dir",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                "artifacts",
                "day8_episode",
                "latest",
            ]),
        ),
        DeclareLaunchArgument(
            "python_executable",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                ".venv",
                "bin",
                "python",
            ]),
        ),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument(
            "perception_model",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "jetson_asv_ws",
                "models",
                "perception_image_conditioned.npz",
            ]),
        ),
        DeclareLaunchArgument("replay_rate_hz", default_value="2.0"),

        Node(
            package="asv_vla",
            executable="replay_episode",
            name="episode_replay",
            output="screen",
            parameters=[{
                "episode_dir": ParameterValue(
                    episode_dir, value_type=str
                ),
                "rate_hz": ParameterValue(
                    replay_rate_hz, value_type=float
                ),
                "start_delay_sec": 5.0,
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="image_entity_perception",
            name="image_entity_perception",
            output="screen",
            parameters=[{
                "model_path": ParameterValue(perception_model, value_type=str),
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="temporal_entity_tracker",
            name="temporal_entity_tracker",
            output="screen",
            parameters=[{"use_sim_time": False}],
        ),
        Node(
            package="asv_vla",
            executable="visual_encoder",
            name="visual_encoder",
            output="screen",
            prefix=[python_executable],
            parameters=[{
                "run_id": "replay",
                "entities_topic": "/vla/tracked_entities",
                "device": ParameterValue(device, value_type=str),
                "image_width": 1280,
                "image_height": 720,
                "horizontal_fov_deg": 90.0,
                "camera_mount_x_m": 0.42,
                "camera_mount_y_m": 0.0,
                "camera_mount_z_m": 0.20,
                "camera_pitch_deg": -5.0,
                "target_crop_size_px": 224,
                "entity_wait_sec": 0.4,
                "sync_cache_size": 32,
                "use_sim_time": False,
            }],
        ),
        Node(
            package="asv_vla",
            executable="task_entity_tensor",
            name="task_entity_tensor",
            output="screen",
            parameters=[{
                "run_id": "replay",
                "max_entities": 16,
                "risk_horizon_sec": 4.0,
                "risk_radius_m": 3.0,
                "entities_topic": "/vla/tracked_entities",
                "use_sim_time": False,
            }],
        ),
    ])
