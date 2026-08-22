"""打包 CUDA 策略运行时的合同测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest


POLICY_NODE = Path(__file__).resolve().parents[1] / "vla" / "decision.py"


def test_policy_node_defaults_to_explicit_torch_cuda_backend() -> None:
    source = POLICY_NODE.read_text(encoding="utf-8")
    for token in (
        'DEFAULT_POLICY_BACKEND = "torch_cuda"',
        'self.declare_parameter("device", "cuda")',
        "POLICY_READY",
        "TorchPolicyRunner.load",
        "device=policy_device",
        "inputs=task_embedding+structured_entities+ego_dynamics",
    ):
        assert token in source


def _inputs(config: Any) -> dict[str, np.ndarray]:
    return {
        "language": np.zeros((1, config.language_dim), dtype=np.float32),
        "entity_geometry": np.zeros(
            (1, config.entity_count, config.entity_geometry_dim), dtype=np.float32
        ),
        "ego_state": np.zeros(
            (1, config.ego_state_dim), dtype=np.float32
        ),
        "language_valid": np.ones((1,), dtype=bool),
        "entity_geometry_mask": np.ones((1, config.entity_count), dtype=bool),
        "ego_state_valid": np.ones((1,), dtype=bool),
        "policy_input_valid": np.ones((1,), dtype=bool),
    }


def test_runtime_does_not_ramp_each_action_from_zero_without_previous_action() -> None:
    from vla.decision import smooth_policy_displacement

    output = smooth_policy_displacement(
        (0.3, 0.0),
        previous_action=None,
        max_step_m=0.3,
        max_delta_m=0.05,
    )

    assert output == pytest.approx((0.3, 0.0))


def test_cuda_is_required_without_cpu_fallback(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from vla.decision import (  # noqa: E402
        PolicyRuntimeError,
        TorchPolicyRunner,
    )

    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    model_path = tmp_path / "policy.pt"
    with pytest.raises(PolicyRuntimeError, match="CUDA is unavailable"):
        TorchPolicyRunner.load(model_path, device="cuda")


def test_checkpoint_strict_load_and_runtime_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from vla.decision import (  # noqa: E402
        SmallPolicyConfig,
        SmallActionPolicy,
        TorchPolicyRunner,
    )

    config = SmallPolicyConfig()
    model = SmallActionPolicy(config)
    model_path = tmp_path / "policy.pt"
    torch.save(
        {"model_config": asdict(config), "model_state_dict": model.state_dict()},
        model_path,
    )

    runner = TorchPolicyRunner.load(model_path, device="cuda")
    assert next(runner.model.parameters()).device.type == "cuda"
    action, stop_logit, valid_mask = runner.run(_inputs(config))
    assert action.shape == (1, config.action_dim)
    assert stop_logit.shape == (1, 1)
    assert valid_mask.shape == (1,)
    assert np.all(np.isfinite(action))
    assert np.all(np.isfinite(stop_logit))
