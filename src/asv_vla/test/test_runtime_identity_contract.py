"""Pure-Python guards for the runtime identity propagation contract.

These tests intentionally inspect source and interface text instead of
importing generated ROS message classes, so they remain runnable before a
ROS interface build has happened.
"""

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
INTERFACES = REPOSITORY / "src/asv_jetson_interfaces/msg"
VLA = REPOSITORY / "src/asv_vla/asv_vla"
LAUNCH = REPOSITORY / "src/asv_bringup/launch/vla_closed_loop.launch.py"
HARDWARE_LAUNCH = REPOSITORY / "src/asv_bringup/launch/hardware_loop.launch.py"
REPLAY_LAUNCH = REPOSITORY / "src/asv_bringup/launch/replay_episode.launch.py"
PERCEPTION_NODE = REPOSITORY / "src/asv_vla/asv_vla/image_entity_perception_node.py"
BENCHMARK = REPOSITORY / "training/benchmark_onnx.py"
COLLECT_LAUNCH = REPOSITORY / "src/asv_bringup/launch/collect.launch.py"
MANIFEST = REPOSITORY / "models/manifest.yaml"
README = REPOSITORY / "README.md"


def _fields(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_decision_and_selected_messages_carry_source_identity() -> None:
    decision_fields = _fields(INTERFACES / "DecisionOutput.msg")
    assert decision_fields[-4:] == [
        "string run_id",
        "int64 scene_seed",
        "uint64 source_frame_index",
        "string source_model_version",
    ]

    selected_fields = _fields(INTERFACES / "SelectedTrajectory.msg")
    assert "int64 scene_seed" in selected_fields
    assert "uint64 frame_index" in selected_fields


def test_closed_loop_launch_exposes_runtime_selection_parameters() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    for argument in (
        "embedding_path",
        "active_embedding",
        "execution_port",
        "visual_device",
    ):
        assert (
            f'DeclareLaunchArgument("{argument}"' in source
            or f'"{argument}",' in source
        )
        assert f'LaunchConfiguration("{argument}")' in source
    assert 'DeclareLaunchArgument("visual_device", default_value="cuda")' in source
    assert "/home/jetson/jetson_asv_ws/models/" in source
    assert "demo_instruction_embedding.npy" in source
    assert "zero embedding" not in source.lower()
    assert "day 19" not in source.lower()


def test_closed_loop_uses_final_near_image_color_policy_candidate() -> None:
    launch = LAUNCH.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "policy_sine_near_image_color_seed42.onnx" in launch
    assert "image_entity_color_calibrated_v1.npz" in launch
    assert "default_value=\"/home/jetson/jetson_asv_ws/models/policy.onnx\"" not in launch
    assert "path: models/image_entity_color_calibrated_v1.npz" in manifest
    assert "model_id: image_entity_color_calibrated_v1" in manifest
    assert "path: models/policy_sine_near_image_color_seed42.onnx" in manifest
    assert "model_id: policy_sine_near_image_color_seed42" in manifest
    assert "artifact_sha256: not_available_in_source_checkout" in manifest
    assert "source_sha256: not_available_in_source_checkout" in manifest
    assert "validation_gate_passed: true" in manifest
    assert "deployment_status: final_near_s2_demo" in manifest
    assert "mode: online_qwen_cuda_staged" in manifest
    assert "online_qwen_runtime: true" in manifest
    assert "policy_sine_near_image_color_seed42.onnx" in readme
    assert "image_entity_color_calibrated_v1.npz" in readme
    assert "policy_image_seed17.onnx" not in readme
    assert "policy.onnx" not in readme
    assert "policy_sine_near_image_color_seed42.onnx" in HARDWARE_LAUNCH.read_text(
        encoding="utf-8"
    )
    assert "image_entity_color_calibrated_v1.npz" in REPLAY_LAUNCH.read_text(
        encoding="utf-8"
    )
    assert "image_entity_color_calibrated_v1.npz" in PERCEPTION_NODE.read_text(
        encoding="utf-8"
    )
    assert "policy_sine_near_image_color_seed42.onnx" in BENCHMARK.read_text(
        encoding="utf-8"
    )


def test_collect_launch_exposes_kinematic_executor_address_and_port() -> None:
    source = COLLECT_LAUNCH.read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument("execution_address", default_value="")' in source
    assert 'DeclareLaunchArgument("execution_port", default_value="8081")' in source
    assert 'LaunchConfiguration("execution_address")' in source
    assert 'LaunchConfiguration("execution_port")' in source
    assert '"execution_address": ParameterValue(' in source
    assert '"execution_port": ParameterValue(' in source


def test_adapter_removes_fabricated_identity_and_fails_closed() -> None:
    source = (VLA / "decision_setpoint_adapter.py").read_text(encoding="utf-8")
    assert '"decision-adapter"' not in source
    assert '"trajectory_controller_v1"' not in source
    assert "def _identity_complete" in source
    assert "scene_seed > 0" in source
    assert "source_frame_index >= 0" in source
    assert "executable = bool(msg.valid) and identity_complete" in source
    assert "out.valid = executable" in source
    assert "out.hold_position = not executable" in source
    assert 'out.reason = "IDENTITY_MISSING"' in source


def test_identity_is_copied_and_mixed_frames_stop_before_inference() -> None:
    controller = (VLA / "trajectory_controller_node.py").read_text(encoding="utf-8")
    assert "output.run_id = str(message.run_id)" in controller
    assert "output.source_frame_index = int(message.frame_index)" in controller
    assert "output.source_model_version = str(message.model_version)" in controller

    policy = (VLA / "vla_policy_node.py").read_text(encoding="utf-8")
    assert 'msg.reason = "IDENTITY_MISMATCH"' in policy
    assert "self._pub.publish(msg)" in policy
    assert "self._recent_trajectories.clear()" in policy

    for name, fields in (
        ("expert_policy_bridge.py", ("out.scene_seed", "out.frame_index")),
        ("safety_gate_node.py", ("output.scene_seed", "output.frame_index")),
    ):
        source = (VLA / name).read_text(encoding="utf-8")
        for field in fields:
            assert field in source
