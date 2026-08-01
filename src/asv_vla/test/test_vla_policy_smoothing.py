"""Pure tests for online VLA policy trajectory shaping."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Sequence

import numpy as np


POLICY = Path(__file__).resolve().parents[1] / "asv_vla" / "vla_policy_node.py"


def _load_smoother():
    tree = ast.parse(POLICY.read_text(encoding="utf-8"), filename=str(POLICY))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "smooth_policy_trajectory"
    )
    namespace = {
        "ACTION_DIM": 2,
        "HORIZON": 20,
        "POLICY_MAX_STEP_M": 0.3,
        "Sequence": Sequence,
        "math": math,
        "np": np,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(POLICY), "exec"),
        namespace,
    )
    return namespace["smooth_policy_trajectory"]


smooth_policy_trajectory = _load_smoother()


def _curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    bc = c - b
    denominator = float(np.linalg.norm(ab) * np.linalg.norm(bc))
    if denominator == 0.0:
        return 0.0
    return abs(float(ab[0] * bc[1] - ab[1] * bc[0])) / denominator


def test_sharp_finite_policy_is_a_bounded_straight_cumulative_horizon() -> None:
    raw = np.zeros((20, 2), dtype=np.float32)
    raw[:-1] = np.asarray(
        [(index * 0.1, 1.2 if index % 2 else -1.2) for index in range(19)],
        dtype=np.float32,
    )
    raw[0] = (0.2, -0.1)
    raw[-1] = (8.0, -6.0)

    shaped = smooth_policy_trajectory(raw.reshape(-1))
    assert shaped is not None
    points = np.asarray(shaped, dtype=np.float64).reshape(20, 2)
    assert np.all(np.isfinite(points))

    increments = np.diff(np.vstack((np.zeros((1, 2)), points)), axis=0)
    assert np.allclose(increments[0], raw[0])
    assert np.allclose(increments, increments[0])
    assert np.all(np.linalg.norm(increments, axis=1) <= 0.3 + 1.0e-9)
    assert np.linalg.norm(points[-1]) <= 20 * 0.3 + 1.0e-9
    assert np.allclose(points[0], points[-1] / 20.0)
    assert all(
        _curvature(points[index - 1], points[index], points[index + 1])
        < 1.0e-10
        for index in range(1, 19)
    )


def test_first_waypoint_is_clipped_but_not_diluted() -> None:
    raw = np.zeros((20, 2), dtype=np.float32)
    raw[0] = (0.6, 0.0)
    shaped = smooth_policy_trajectory(raw)
    assert shaped is not None
    points = np.asarray(shaped, dtype=np.float64).reshape(20, 2)
    increments = np.diff(np.vstack((np.zeros((1, 2)), points)), axis=0)
    assert np.allclose(increments[0], (0.3, 0.0))
    assert np.allclose(increments, (0.3, 0.0))
    assert np.allclose(points[-1], (6.0, 0.0))


def test_stop_invalid_and_nonfinite_policy_outputs_are_not_shaped() -> None:
    raw = np.tile(np.asarray((1.0, 0.0), dtype=np.float32), 20)
    assert smooth_policy_trajectory(raw, safe_stop=True) is None
    assert smooth_policy_trajectory(raw, valid=False) is None
    raw[3] = np.nan
    assert smooth_policy_trajectory(raw) is None


def test_policy_trace_is_bounded_and_carries_modality_guard_fields() -> None:
    source = POLICY.read_text(encoding="utf-8")
    for token in (
        "POLICY_TRACE_LIMIT",
        "_policy_trace_count",
        "POLICY_TRACE",
        "policy_valid=",
        "stop=",
        "lang_valid=",
        "vis_valid=",
        "ent_valid=",
        "ego_valid=",
        "guard_result=",
        "guard_reason=",
    ):
        assert token in source
