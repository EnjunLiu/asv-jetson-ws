"""Pure tests for the direct single-step policy adapter."""

from __future__ import annotations

import ast
import copy
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
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


def _load_gate_runtime():
    tree = ast.parse(POLICY.read_text(encoding="utf-8"), filename=str(POLICY))
    policy_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VLAPolicyNode"
    )
    methods = {
        "_audit_action",
        "_maybe_policy_audit",
        "_record_guard_outcome",
        "_record_fail_closed",
        "_clear_previous_action",
        "_clear_control_history",
        "_remember_previous_action",
        "_on_gate_result",
    }
    policy_class = copy.deepcopy(policy_class)
    policy_class.body = [
        node
        for node in policy_class.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    pending = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_PendingAction"
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_identity_tuple", "bound_policy_displacement"}
    ]
    future_annotations = ast.parse(
        "from __future__ import annotations"
    ).body[0]
    namespace = {
        "ACTION_DIM": 2,
        "POLICY_MAX_STEP_M": 0.15,
        "Sequence": Sequence,
        "math": math,
        "np": np,
        "OrderedDict": OrderedDict,
        "dataclass": dataclass,
        "Node": object,
        "POLICY_AUDIT_PERIOD": 100,
        "GUARD_POLICY_DRIVEN": "policy_driven",
        "GUARD_BACKSTOP": "backstop",
        "GUARD_HOLD": "deadband_hold",
        "GUARD_FAIL_CLOSED": "fail_closed",
    }
    nodes = [future_annotations, pending, *helpers, policy_class]
    ast.fix_missing_locations(policy_class)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(POLICY), "exec"),
        namespace,
    )
    return namespace["VLAPolicyNode"], namespace["_PendingAction"]


PolicyNode, PendingAction = _load_gate_runtime()


def _policy_state():
    node = PolicyNode.__new__(PolicyNode)
    node._active_run = ("run-a", 7)
    node._last_gate_frame_index = -1
    node._last_inferred_frame_index = -1
    node._pending_actions = OrderedDict()
    node._previous_action = np.zeros(2, dtype=np.float32)
    node._previous_action_valid = False
    node._previous_action_identity = None
    node._smooth_max_step_m = 0.15
    node._policy_audit_events = 0
    node._policy_driven_count = 0
    node._backstop_count = 0
    node._hold_count = 0
    node._fail_closed_count = 0
    node._policy_stop_count = 0
    return node


def _point(
    frame_index: int,
    *,
    stamp_us: int,
    desired_x: float = 0.04,
    desired_y: float = -0.03,
    valid: bool = True,
    safe_stop: bool = False,
):
    return SimpleNamespace(
        run_id="run-a",
        scene_seed=7,
        frame_index=frame_index,
        stamp_us=stamp_us,
        desired_x=desired_x,
        desired_y=desired_y,
        valid=valid,
        safe_stop=safe_stop,
    )


def _queue(node, frame_index: int, stamp_us: int):
    node._pending_actions[("run-a", 7, frame_index)] = PendingAction(
        stamp_us=stamp_us,
        action=(0.1, 0.0),
    )


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


def test_previous_action_commits_only_after_current_point_passes_gate() -> None:
    node = _policy_state()
    _queue(node, frame_index=4, stamp_us=500)

    assert not node._previous_action_valid
    node._on_gate_result(_point(4, stamp_us=500, desired_x=0.04, desired_y=-0.03))

    assert node._previous_action_valid
    assert node._previous_action_identity == ("run-a", 7, 4)
    assert tuple(node._previous_action) == pytest.approx((0.04, -0.03))


@pytest.mark.parametrize("safe_stop,valid", [(True, False), (False, False)])
def test_gate_hold_or_invalid_clears_previous_action(
    safe_stop: bool, valid: bool
) -> None:
    node = _policy_state()
    _queue(node, frame_index=4, stamp_us=500)
    node._on_gate_result(_point(4, stamp_us=500))
    assert node._previous_action_valid

    _queue(node, frame_index=5, stamp_us=501)
    node._on_gate_result(
        _point(5, stamp_us=501, safe_stop=safe_stop, valid=valid)
    )

    assert not node._previous_action_valid
    assert node._previous_action_identity is None
    assert tuple(node._previous_action) == pytest.approx((0.0, 0.0))


def test_gate_frame_break_clears_history_and_next_contiguous_gate_recovers() -> None:
    node = _policy_state()
    _queue(node, frame_index=4, stamp_us=500)
    node._on_gate_result(_point(4, stamp_us=500))

    _queue(node, frame_index=6, stamp_us=502)
    node._on_gate_result(_point(6, stamp_us=502))
    assert not node._previous_action_valid
    assert node._previous_action_identity is None

    _queue(node, frame_index=7, stamp_us=503)
    node._on_gate_result(_point(7, stamp_us=503, desired_x=0.02, desired_y=0.01))
    assert node._previous_action_valid
    assert node._previous_action_identity == ("run-a", 7, 7)
    assert tuple(node._previous_action) == pytest.approx((0.02, 0.01))


def test_policy_contract_has_no_visual_or_ego_decision_inputs() -> None:
    source = POLICY.read_text(encoding="utf-8")
    assert "TaskEmbedding, \"/vla/language_embedding\"" in source
    assert "TaskFeatures, \"/vla/task_features\"" in source
    assert "VisualFeatures" not in source
    assert "UEASVState" not in source
    assert "POLICY_MODEL_HORIZON" not in source
    for token in ("POLICY_TRACE_LIMIT", "policy_valid=", "lang_valid=", "entity_valid="):
        assert token in source


def test_policy_audit_contract_includes_action_stages_and_counters() -> None:
    source = POLICY.read_text(encoding="utf-8")
    for token in (
        "POLICY_AUDIT",
        "raw_dx=",
        "raw_dy=",
        "guarded_dx=",
        "guarded_dy=",
        "final_dx=",
        "final_dy=",
        "guard_reason=",
        "policy_driven=",
        "backstop=",
        "hold=",
        "fail_closed=",
        "policy_stop=",
        "trigger=\"shutdown\"",
    ):
        assert token in source


def test_policy_audit_counters_are_bounded_and_keep_the_existing_api() -> None:
    class Logger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    node = _policy_state()
    logger = Logger()
    node.get_logger = lambda: logger

    assert node._audit_action((0.1, -0.2)) == ("0.100000", "-0.200000")
    assert node._audit_action((float("nan"), 0.0)) == ("nan", "nan")
    node._record_guard_outcome(
        "policy_driven",
        raw_action=(0.01, 0.02),
        guarded_action=(0.01, 0.02),
        final_action=(0.01, 0.02),
    )
    node._record_guard_outcome("backstop")
    node._record_guard_outcome("deadband_hold")
    node._record_fail_closed(reason="POLICY_STOP")
    assert node._policy_driven_count == 1
    assert node._backstop_count == 1
    assert node._hold_count == 1
    assert node._fail_closed_count == 1
    assert node._policy_stop_count == 1
    assert node._policy_audit_events == 4
    assert logger.messages == []

    for _ in range(95):
        node._record_guard_outcome("policy_driven")
    node._record_guard_outcome(
        "policy_driven",
        raw_action=(0.03, -0.04),
        guarded_action=(0.05, -0.06),
        final_action=(0.02, -0.01),
    )
    assert node._policy_audit_events == 100
    assert len(logger.messages) == 1
    assert "POLICY_AUDIT" in logger.messages[0]
    assert "policy_driven=97" in logger.messages[0]
    assert "backstop=1" in logger.messages[0]
    assert "hold=1" in logger.messages[0]
    assert "fail_closed=1" in logger.messages[0]
    assert "policy_stop=1" in logger.messages[0]
    assert "guard_reason=policy_driven" in logger.messages[0]
    assert "raw_dx=0.030000 raw_dy=-0.040000" in logger.messages[0]
    assert "guarded_dx=0.050000 guarded_dy=-0.060000" in logger.messages[0]
    assert "final_dx=0.020000 final_dy=-0.010000" in logger.messages[0]
