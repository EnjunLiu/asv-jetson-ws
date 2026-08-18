"""Pure-Python guards for the runtime identity propagation contract.

These tests intentionally inspect source and interface text instead of
importing generated ROS message classes, so they remain runnable before a
ROS interface build has happened.
"""

import ast
from pathlib import Path
from types import SimpleNamespace


REPOSITORY = Path(__file__).resolve().parents[3]
INTERFACES = REPOSITORY / "src/interfaces/msg"
VLA = REPOSITORY / "src/vla/vla"
POLICY = VLA / "decision.py"
LAUNCH = REPOSITORY / "src/bringup/launch/vla_closed_loop.launch.py"
PERCEPTION_NODE = REPOSITORY / "src/vla/vla/perception_node.py"
MANIFEST = REPOSITORY / "models/manifest.yaml"
README = REPOSITORY / "README.md"


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
        "entity_features_identity_reason",
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
        namespace["entity_features_identity_reason"],
    )


identity_mismatch_reason, entity_features_identity_reason = _load_identity_guards()


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


def test_displacement_message_carries_source_identity() -> None:
    fields = _fields(INTERFACES / "DesiredDisplacement.msg")
    for field in (
        "string run_id",
        "int64 scene_seed",
        "uint64 frame_index",
        "string source",
        "float32 desired_x",
        "float32 desired_y",
        "bool safe_stop",
        "string reason",
    ):
        assert field in fields


def test_closed_loop_launch_exposes_runtime_selection_parameters() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    for argument in ("execution_port", "visual_device"):
        assert (
            f'DeclareLaunchArgument("{argument}"' in source
            or f'"{argument}",' in source
        )
        assert f'LaunchConfiguration("{argument}")' in source
    assert 'DeclareLaunchArgument("visual_device", default_value="cuda")' in source
    assert '"models_dir"' in source
    assert "demo_instruction_embedding.npy" not in source
    assert 'executable="language"' in source
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


def test_entity_features_identity_is_complete_and_monotonic() -> None:
    first = _features("follow red target", frame_index=10)
    assert entity_features_identity_reason(first) is None
    assert (
        entity_features_identity_reason(
            _features("follow red target", frame_index=12),
            ("scene-run", 42, 10),
        )
        is None
    )
    assert (
        entity_features_identity_reason(
            _features("follow red target", frame_index=9),
            ("scene-run", 42, 10),
        )
        == "IDENTITY_MISMATCH"
    )
    assert (
        entity_features_identity_reason(
            _features("follow red target", run_id="", frame_index=11)
        )
        == "IDENTITY_MISMATCH"
    )
    assert (
        entity_features_identity_reason(
            _features("follow red target", scene_seed=0, stamp_us=100)
        )
        == "IDENTITY_MISMATCH"
    )
    assert (
        entity_features_identity_reason(
            _features("follow red target", scene_seed=42, stamp_us=0)
        )
        == "IDENTITY_MISMATCH"
    )
    assert entity_features_identity_reason(
        _features("follow red target", run_id="new-run", frame_index=0),
        ("scene-run", 42, 10),
    ) is None


def test_runtime_uses_current_single_point_cuda_artifacts() -> None:
    launch = LAUNCH.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "policy_single_point.pt" in launch
    assert "perception_image_conditioned.npz" in launch
    assert "default_value=\"models/policy.onnx\"" not in launch
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
    assert "deployment_status: selected_for_deployment" in manifest
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
    assert "policy_image_seed17.onnx" not in readme
    assert "policy.onnx" not in readme
    assert "device={self.device}" in PERCEPTION_NODE.read_text(encoding="utf-8")


def test_online_perception_boundary_excludes_privileged_entities_and_cpu_policy() -> None:
    perception_source = PERCEPTION_NODE.read_text(encoding="utf-8")
    launch_source = LAUNCH.read_text(encoding="utf-8")
    policy_source = POLICY.read_text(encoding="utf-8")

    assert '"/ue/entities"' not in perception_source
    assert '"/ue/entities"' not in launch_source
    assert 'self.declare_parameter("backend"' not in policy_source
    assert "onnxruntime" not in policy_source
    assert "TorchPolicyRunner.load" in policy_source


def test_bridge_consumes_final_displacement_directly() -> None:
    bridge = (REPOSITORY / "src/bridge/src/bridge_node.cpp").read_text(
        encoding="utf-8"
    )
    assert "DesiredDisplacement" in bridge
    assert '"/control/desired_displacement"' in bridge
    assert "!command.safe_stop" in bridge
    assert "command.frame_index" in bridge
    assert "command.source" in bridge
    assert not (VLA / "ue_setpoint_adapter.py").exists()


def test_identity_is_copied_and_mixed_frames_stop_before_inference() -> None:
    policy = POLICY.read_text(encoding="utf-8")
    assert 'self._publish_fail_closed(ent, identity_reason)' in policy
    assert 'message.reason = str(reason)' in policy
    assert "identity_mismatch_reason(self._language, ent)" in policy
    assert "self._previous_action" in policy
    assert "self._recent_actions" not in policy
    assert "self._frame_sync.clear()" in policy
    assert "self._clear_control_history()" in policy

    for field in ("message.scene_seed", "message.frame_index"):
        assert field in policy


def test_setup_points_merged_nodes_at_algorithm_modules() -> None:
    setup = (REPOSITORY / "src/vla/setup.py").read_text(encoding="utf-8")
    assert '"decision = vla.decision_node:main"' in setup
    assert '"temporal_entity_tracker =' not in setup
    assert '"safety_gate =' not in setup
