"""Pure-Python guards for the runtime identity propagation contract.

These tests intentionally inspect source and interface text instead of
importing generated ROS message classes, so they remain runnable before a
ROS interface build has happened.
"""

import ast
from pathlib import Path
from types import SimpleNamespace


REPOSITORY = Path(__file__).resolve().parents[3]
INTERFACES = REPOSITORY / "src/asv_jetson_interfaces/msg"
VLA = REPOSITORY / "src/asv_vla/asv_vla"
POLICY = VLA / "vla_policy_node.py"
LAUNCH = REPOSITORY / "src/asv_bringup/launch/vla_closed_loop.launch.py"
REPLAY_LAUNCH = REPOSITORY / "src/asv_bringup/launch/replay_episode.launch.py"
PERCEPTION_NODE = REPOSITORY / "src/asv_vla/asv_vla/image_entity_perception_node.py"
COLLECT_LAUNCH = REPOSITORY / "src/asv_bringup/launch/collect.launch.py"
MANIFEST = REPOSITORY / "models/manifest.yaml"
README = REPOSITORY / "README.md"
DEMO_RUNBOOK = REPOSITORY / "docs/demo_runbook.md"
TRAINING_README = REPOSITORY / "training/README.md"


def _fields(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_identity_guards():
    policy = ast.parse(POLICY.read_text(encoding="utf-8"), filename=str(POLICY))
    wanted_functions = {
        "_identity_tuple",
        "identity_mismatch_reason",
        "task_features_identity_reason",
    }
    nodes = [
        node
        for node in policy.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions
    ]
    namespace = {"Any": object, "FrameKey": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(POLICY), "exec"), namespace)
    return (
        namespace["identity_mismatch_reason"],
        namespace["task_features_identity_reason"],
    )


identity_mismatch_reason, task_features_identity_reason = _load_identity_guards()


def _language(instruction: str, *, run_id: str = "language-qwen", stamp_us: int = 900):
    return SimpleNamespace(
        run_id=run_id,
        stamp_us=stamp_us,
        instruction=instruction,
        valid=True,
    )


def _features(
    instruction: str,
    *,
    run_id: str = "scene-run",
    scene_seed: int = 42,
    frame_index: int = 0,
    stamp_us: int = 100,
    instruction_id: str = "",
):
    return SimpleNamespace(
        run_id=run_id,
        scene_seed=scene_seed,
        frame_index=frame_index,
        stamp_us=stamp_us,
        instruction=instruction,
        instruction_id=instruction_id,
    )


def test_decision_messages_carry_source_identity() -> None:
    decision_fields = _fields(INTERFACES / "DecisionOutput.msg")
    assert decision_fields[-4:] == [
        "string run_id",
        "int64 scene_seed",
        "uint64 source_frame_index",
        "string source_model_version",
    ]

    point_fields = _fields(INTERFACES / "DecisionPoint.msg")
    assert "int64 scene_seed" in point_fields
    assert "uint64 frame_index" in point_fields
    assert "float32 desired_x" in point_fields
    assert "float32 desired_y" in point_fields


def test_closed_loop_launch_exposes_runtime_selection_parameters() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    for argument in ("execution_port", "visual_device"):
        assert (
            f'DeclareLaunchArgument("{argument}"' in source
            or f'"{argument}",' in source
        )
        assert f'LaunchConfiguration("{argument}")' in source
    assert 'DeclareLaunchArgument("visual_device", default_value="cuda")' in source
    assert "/home/jetson/jetson_asv_ws/models/" in source
    assert "demo_instruction_embedding.npy" not in source
    assert 'executable="language_qwen"' in source
    assert 'executable="language_stub"' not in source
    assert 'executable="expert_trajectory"' not in source
    assert 'executable="expert_kinematic_executor"' not in source
    assert "zero embedding" not in source.lower()
    assert "day 19" not in source.lower()


def test_language_identity_uses_task_text_not_encoder_or_frame_stamp() -> None:
    language = _language("follow red target", run_id="language-qwen", stamp_us=900)
    features = _features("follow red target", stamp_us=100)
    assert identity_mismatch_reason(language, features) is None
    assert (
        identity_mismatch_reason(language, _features("follow blue target"))
        == "IDENTITY_MISMATCH"
    )
    assert (
        identity_mismatch_reason(
            language, _features("follow red target", instruction_id="other")
        )
        is None
    )


def test_task_features_identity_is_complete_and_monotonic() -> None:
    first = _features("follow red target", frame_index=10)
    assert task_features_identity_reason(first) is None
    assert (
        task_features_identity_reason(
            _features("follow red target", frame_index=12),
            ("scene-run", 42, 10),
        )
        is None
    )
    assert (
        task_features_identity_reason(
            _features("follow red target", frame_index=9),
            ("scene-run", 42, 10),
        )
        == "IDENTITY_MISMATCH"
    )
    assert (
        task_features_identity_reason(
            _features("follow red target", run_id="", frame_index=11)
        )
        == "IDENTITY_MISMATCH"
    )
    assert (
        task_features_identity_reason(
            _features("follow red target", scene_seed=0, stamp_us=100)
        )
        == "IDENTITY_MISMATCH"
    )
    assert (
        task_features_identity_reason(
            _features("follow red target", scene_seed=42, stamp_us=0)
        )
        == "IDENTITY_MISMATCH"
    )
    assert task_features_identity_reason(
        _features("follow red target", run_id="new-run", frame_index=0),
        ("scene-run", 42, 10),
    ) is None


def test_closed_loop_uses_current_single_point_cuda_artifacts() -> None:
    launch = LAUNCH.read_text(encoding="utf-8")
    replay_launch = REPLAY_LAUNCH.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    demo_runbook = DEMO_RUNBOOK.read_text(encoding="utf-8")
    training_readme = TRAINING_README.read_text(encoding="utf-8")

    assert "policy_single_point.pt" in launch
    assert "perception_image_conditioned.npz" in launch
    assert "perception_image_conditioned.npz" in replay_launch
    assert "default_value=\"/home/jetson/jetson_asv_ws/models/policy.onnx\"" not in launch
    assert "path: models/perception_image_conditioned.npz" in manifest
    assert "source_path: models/perception_image_conditioned.npz" in manifest
    assert "model_id: image_entity_ridge_language_v3" in manifest
    assert "path: models/policy_single_point.pt" in manifest
    assert "source_path: models/policy_single_point.pt" in manifest
    assert "model_id: policy_single_point" in manifest
    assert (
        "artifact_sha256: "
        "a1e7451642c51b879e8b9ce1d7037567c2057d534bcb547c483716188ceb5e6e"
    ) in manifest
    assert (
        "source_sha256: "
        "f2dc38a141a3f230b2ddf55cef26841f00812bbd350f28aa84c84f5d5d1e2483"
    ) in manifest
    assert "deployment_status: selected_for_current_closed_loop" in manifest
    assert "report: pc_datasets/reports/closed_loop_20260805/single_point_policy_dominant" in manifest
    assert "shared_entity_trajectory_tolerance_cm: 5" in manifest
    assert "TRACK-SYNTH-RED3" in manifest
    assert "mode: online_qwen_cuda_release_after_encode" in manifest
    assert "online_qwen_runtime: true" in manifest
    assert "release_model_after_encode: true" in manifest
    assert "first_instruction_encoding: real_qwen_cuda" in manifest
    assert "qwen_weight_resident_after_encode: false" in manifest
    assert "post_encode_embedding_online: true" in manifest
    assert "cached_embedding_file: false" in manifest
    assert "cpu_fallback: false" in manifest
    assert "policy_single_point.pt" in readme
    assert "perception_image_conditioned.npz" in readme
    assert "policy_single_point.pt" in demo_runbook
    assert "perception_image_conditioned.npz" in demo_runbook
    assert "policy_single_point.pt" in training_readme
    assert "perception_image_conditioned.npz" in training_readme
    assert "policy_image_seed17.onnx" not in readme
    assert "policy.onnx" not in readme
    old_active_artifacts = (
        "policy_single_point_v3_full_seed17.pt",
        "perception_image_conditioned_130_v1.npz",
    )
    for source in (launch, replay_launch, manifest, readme):
        for artifact in old_active_artifacts:
            assert artifact not in source
    assert "device={self.device}" in PERCEPTION_NODE.read_text(encoding="utf-8")


def test_online_perception_boundary_excludes_privileged_entities_and_cpu_policy() -> None:
    perception_source = PERCEPTION_NODE.read_text(encoding="utf-8")
    launch_source = LAUNCH.read_text(encoding="utf-8")
    policy_source = (VLA / "vla_policy_node.py").read_text(encoding="utf-8")

    assert '"/ue/entities"' not in perception_source
    assert '"/ue/entities"' not in launch_source
    assert 'self.declare_parameter("backend"' not in policy_source
    assert "onnxruntime" not in policy_source
    assert "TorchPolicyRunner.load" in policy_source


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
    assert 'self._publish_fail_closed(ent, identity_reason)' in policy
    assert 'message.reason = str(reason)' in policy
    assert "identity_mismatch_reason(self._language, ent)" in policy
    assert "self._pending_actions" in policy
    assert "self._previous_action_identity" in policy
    assert "self._previous_action if previous_action_valid else None" in policy
    assert "self._recent_actions" not in policy
    assert "self._frame_sync.clear()" in policy
    assert "self._clear_control_history()" in policy

    safety_gate = (VLA / "safety_gate_node.py").read_text(encoding="utf-8")
    for field in ("output.scene_seed", "output.frame_index"):
        assert field in safety_gate
