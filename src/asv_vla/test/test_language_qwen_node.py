"""Focused contract tests for the online Qwen language ROS adapter."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
NODE = REPOSITORY / "src/asv_vla/asv_vla/language_qwen_node.py"
ENCODER = REPOSITORY / "src/asv_vla/asv_vla/language_encoder.py"
SETUP = REPOSITORY / "src/asv_vla/setup.py"
LAUNCH = REPOSITORY / "src/asv_bringup/launch/vla_closed_loop.launch.py"


def _load_ros_independent_helpers() -> dict[str, Any]:
    """Extract helper/state code so tests run before a ROS interface build."""

    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    names = {
        "EMBEDDING_DIM",
        "DEFAULT_MODEL_ID",
        "_bounded_detail",
        "_zero_embedding",
        "LanguageEmbeddingState",
        "_state_payload",
        "_embedding_tuple",
    }
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))
            and any(
                getattr(target, "id", None) in names
                for target in getattr(node, "targets", [])
            )
            or isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in names
        )
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "np": np,
        "dataclass": dataclass,
        "field": field,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(NODE), "exec"),
        namespace,
    )
    return namespace


def test_state_payload_preserves_instruction_model_and_cache_metadata():
    namespace = _load_ros_independent_helpers()
    state = namespace["LanguageEmbeddingState"](
        instruction="follow the red boat",
        embedding=tuple(np.linspace(0.0, 1.0, 256, dtype=np.float32)),
        model_id="qwen:test",
        cached=True,
        valid=True,
        detail="x" * 400,
    )

    payload = namespace["_state_payload"](state, run_id="run-7", stamp_us=123)

    assert payload["stamp_us"] == 123
    assert payload["run_id"] == "run-7"
    assert payload["instruction"] == "follow the red boat"
    assert payload["model_id"] == "qwen:test"
    assert payload["embedding_dim"] == 256
    assert len(payload["embedding"]) == 256
    assert payload["cached"] is True
    assert payload["valid"] is True
    assert len(payload["detail"]) == 240


def test_embedding_helper_enforces_fixed_finite_contract():
    namespace = _load_ros_independent_helpers()
    vector = namespace["_embedding_tuple"](np.ones(256, dtype=np.float32))
    assert isinstance(vector, tuple)
    assert len(vector) == 256

    for invalid in (
        np.ones(255, dtype=np.float32),
        np.full(256, np.nan, dtype=np.float32),
    ):
        try:
            namespace["_embedding_tuple"](invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding was accepted")


def test_qwen_node_contract_is_fail_closed_and_cuda_explicit():
    source = NODE.read_text(encoding="utf-8")
    encoder_source = ENCODER.read_text(encoding="utf-8")
    assert "USVLanguageEncoder" in source
    assert 'device=self.device' in source
    assert "LanguageEncoderError" in source
    assert "MODEL_UNAVAILABLE" in source
    assert "ModuleStatus.ERROR" in source
    assert '"/task/text"' in source
    assert 'String, "/task/text", self.on_task, LANGUAGE_QOS' in source
    assert '"/vla/language_embedding"' in source
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "cached=bool(result.cached)" in source
    assert "embedding_dim" in source
    assert '"release_model_after_encode", False' in source
    assert "_released_state" in source
    assert "gc.collect()" in source
    assert "torch.cuda.empty_cache()" in source
    assert "MODEL_RELEASED_AFTER_FIRST_ENCODE" in source
    assert "LANGUAGE_READY_VALID" in source
    assert "batch_size=1" in encoder_source
    assert "CUDA_MEMORY_ERROR" in encoder_source


def test_first_task_trace_is_bounded_and_includes_instruction():
    source = NODE.read_text(encoding="utf-8")
    assert "LANGUAGE_TASK_TRACE_LIMIT = 1" in source
    assert "_task_trace_count" in source
    assert "_trace_first_task" in source
    assert "LANGUAGE_TASK_RECEIVED" in source
    assert "instruction=" in source
    assert "if not instruction:" in source


def test_launch_selects_exactly_one_language_backend_and_qwen_is_default():
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument("language_backend", default_value="qwen")' in launch
    assert 'executable="language_stub"' in launch
    assert 'executable="language_qwen"' in launch
    assert "UnlessCondition" in launch
    assert "IfCondition" in launch
    assert 'LaunchConfiguration("language_backend")' in launch
    assert 'LaunchConfiguration("language_model_path")' in launch
    assert 'LaunchConfiguration("language_device")' in launch
    assert (
        'DeclareLaunchArgument(\n                "language_release_after_encode", '
        'default_value="true"\n            )'
    ) in launch
    assert 'LaunchConfiguration("language_release_after_encode")' in launch
    assert "TimerAction" in launch
    assert (
        'DeclareLaunchArgument(\n                "language_staging_delay_sec", '
        'default_value="20.0"\n            )'
    ) in launch


def test_setup_registers_online_qwen_entrypoint():
    setup = SETUP.read_text(encoding="utf-8")
    assert '"language_qwen = asv_vla.language_qwen_node:main"' in setup
