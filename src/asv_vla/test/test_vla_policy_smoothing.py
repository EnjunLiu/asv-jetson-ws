"""Pure tests for the direct single-step policy adapter."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest


POLICY = Path(__file__).resolve().parents[1] / "asv_vla" / "vla_policy_node.py"


def _load_adapter():
    tree = ast.parse(POLICY.read_text(encoding="utf-8"), filename=str(POLICY))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "bound_policy_displacement"
    )
    namespace = {
        "ACTION_DIM": 2,
        "POLICY_MAX_STEP_M": 0.15,
        "Sequence": Sequence,
        "math": math,
        "np": np,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(POLICY), "exec"),
        namespace,
    )
    return namespace["bound_policy_displacement"]


bound_policy_displacement = _load_adapter()


def test_direct_action_is_returned_without_horizon_adaptation() -> None:
    output = bound_policy_displacement((0.1, -0.05))
    assert output == pytest.approx((0.1, -0.05))


def test_direct_action_is_norm_clipped() -> None:
    output = bound_policy_displacement((0.3, 0.4))
    assert output is not None
    assert np.linalg.norm(output) == pytest.approx(0.15)
    assert output[0] == pytest.approx(0.09)
    assert output[1] == pytest.approx(0.12)


def test_stop_invalid_and_nonfinite_outputs_fail_closed() -> None:
    assert bound_policy_displacement((0.1, 0.0), safe_stop=True) is None
    assert bound_policy_displacement((0.1, 0.0), valid=False) is None
    assert bound_policy_displacement((float("nan"), 0.0)) is None
    assert bound_policy_displacement((0.1, 0.0, 0.0)) is None


def test_policy_contract_has_no_visual_or_ego_decision_inputs() -> None:
    source = POLICY.read_text(encoding="utf-8")
    assert "TaskEmbedding, \"/vla/language_embedding\"" in source
    assert "TaskFeatures, \"/vla/task_features\"" in source
    assert "VisualFeatures" not in source
    assert "UEASVState" not in source
    assert "POLICY_MODEL_HORIZON" not in source
    for token in ("POLICY_TRACE_LIMIT", "policy_valid=", "lang_valid=", "entity_valid="):
        assert token in source
