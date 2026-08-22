"""在线 Qwen 语言 ROS 适配器的重点合同测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
NODE = REPOSITORY / "src/vla/vla/language_node.py"
ALGORITHM = REPOSITORY / "src/vla/vla/language.py"
SETUP = REPOSITORY / "src/vla/setup.py"
LAUNCH = REPOSITORY / "src/bringup/launch/vla_closed_loop.launch.py"


def _load_algorithm():
    import sys

    sys.path.insert(0, str(REPOSITORY / "src/vla"))
    from vla import language

    return language


def test_state_payload_preserves_instruction_model_and_cache_metadata():
    algorithm = _load_algorithm()
    state = algorithm.LanguageEmbeddingState(
        instruction="follow the red boat",
        embedding=tuple(np.linspace(0.0, 1.0, 256, dtype=np.float32)),
        model_id="qwen:test",
        cached=True,
        valid=True,
        detail="x" * 400,
    )

    payload = algorithm.state_payload(state, run_id="run-7", stamp_us=123)

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
    import numpy as np

    algorithm = _load_algorithm()
    vector = algorithm.embedding_tuple(np.ones(256, dtype=np.float32))
    assert isinstance(vector, tuple)
    assert len(vector) == 256

    for invalid in (
        np.ones(255, dtype=np.float32),
        np.full(256, np.nan, dtype=np.float32),
    ):
        try:
            algorithm.embedding_tuple(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding was accepted")


def test_qwen_node_contract_is_fail_closed_and_cuda_explicit():
    source = NODE.read_text(encoding="utf-8")
    encoder_source = ALGORITHM.read_text(encoding="utf-8")
    assert 'DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"' in encoder_source
    assert "USVLanguageEncoder" in source
    assert 'device=self.device' in source
    assert "LanguageEncoderError" in source
    assert "MODEL_UNAVAILABLE" in source
    assert '"/system/module_status"' not in source
    assert '"/task/text"' in source
    assert 'String, "/task/text", self.on_task, LANGUAGE_QOS' in source
    assert '"/vla/language_embedding"' in source
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "cached=bool(result.cached)" in source
    assert "embedding_dim" in encoder_source
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


def test_launch_uses_real_qwen_cuda_without_cached_stub_backend():
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'executable="language"' in launch
    assert 'executable="language_stub"' not in launch
    assert "demo_instruction_embedding.npy" not in launch
    assert 'LaunchConfiguration("language_model_path")' in launch
    assert 'LaunchConfiguration("language_device")' in launch
    assert '"language_model_id", default_value="Qwen/Qwen3-Embedding-0.6B"' in launch
    assert (
            'DeclareLaunchArgument(\n                "language_release_after_encode", '
            'default_value="true"\n            )'
    ) in launch
    assert '"perception_start_delay_sec", default_value="45.0"' in launch
    assert '"policy_start_delay_sec", default_value="50.0"' in launch
    assert 'period=LaunchConfiguration("perception_start_delay_sec")' in launch
    assert 'period=LaunchConfiguration("policy_start_delay_sec")' in launch
    assert 'LaunchConfiguration("language_release_after_encode")' in launch


def test_setup_registers_online_qwen_entrypoint():
    setup = SETUP.read_text(encoding="utf-8")
    assert '"language = vla.language_node:main"' in setup
