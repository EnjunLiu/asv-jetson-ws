from pathlib import Path
from types import SimpleNamespace

import pytest

from vla.decision import (
    COLLISION_RISK,
    ESTOP,
    INVALID_MODALITY,
    NONFINITE,
    PASS,
    POLICY_STOP,
    SPEED_LIMIT,
    STALE_INPUT,
    SafetyGateConfig,
    _Entity,
    _check_collision,
    _check_kinematics,
    evaluate_safety_gate,
    limit_displacement_rate,
)
from vla.decision import DT_SEC, FRAME_ID, MAX_DISPLACEMENT_M


def test_safety_gate_module_owns_ros_node_entrypoint() -> None:
    source = (Path(__file__).parents[1] / "vla" / "decision.py").read_text(
        encoding="utf-8"
    )
    assert "def evaluate_safety_gate" in source
    assert "allow_truth_entities" not in source
    assert "UNTRUSTED_ENTITY_SOURCE" not in source


def _gate_input(**overrides):
    values = dict(
        stamp_us=1000,
        run_id="test_run",
        frame_id=FRAME_ID,
        model_version="policy",
        dt=DT_SEC,
        desired_x=0.05,
        desired_y=0.0,
        safe_stop=False,
        valid=True,
        reason="test",
        entity_valid=True,
        entities=None,
        last_valid_stamp_us=0,
        time_since_last_valid_sec=0.0,
    )
    values.update(overrides)
    return values


def test_single_point_passes_kinematics():
    assert _check_kinematics(0.1, 0.0, SafetyGateConfig()) is None


def test_single_point_speed_limit_is_norm_based():
    config = SafetyGateConfig(max_step_m=0.15)
    assert _check_kinematics(0.12, 0.1, config) == SPEED_LIMIT


def test_displacement_reversal_is_rate_limited():
    result = evaluate_safety_gate(**_gate_input(desired_x=-MAX_DISPLACEMENT_M))
    limited = limit_displacement_rate(
        result,
        (MAX_DISPLACEMENT_M, 0.0),
        max_delta_m=MAX_DISPLACEMENT_M,
    )
    assert limited.valid
    assert limited.detail == "RATE_LIMITED"
    assert limited.desired_x == pytest.approx(0.0)


def test_invalid_displacement_is_not_rate_limited():
    result = evaluate_safety_gate(**_gate_input(valid=False))
    assert limit_displacement_rate(result, (0.1, 0.0)) == result


def test_nonfinite_action_is_rejected():
    result = evaluate_safety_gate(**_gate_input(desired_x=float("nan")))
    assert result.reason == NONFINITE
    assert not result.valid
    assert result.safe_stop
    assert (result.desired_x, result.desired_y) == (0.0, 0.0)


def test_invalid_modality_is_zero_invalid_stop():
    result = evaluate_safety_gate(**_gate_input(entity_valid=False))
    assert result.reason == INVALID_MODALITY
    assert not result.valid
    assert result.safe_stop
    assert (result.desired_x, result.desired_y) == (0.0, 0.0)


def test_safety_gate_has_no_ego_or_global_visual_contract():
    with pytest.raises(TypeError):
        evaluate_safety_gate(**_gate_input(ego_valid=False))
    with pytest.raises(TypeError):
        evaluate_safety_gate(**_gate_input(visual_valid=False))


def test_policy_stop_is_never_valid():
    result = evaluate_safety_gate(**_gate_input(safe_stop=True))
    assert result.reason == POLICY_STOP
    assert not result.valid
    assert result.safe_stop
    assert (result.desired_x, result.desired_y) == (0.0, 0.0)


def test_stale_stamp_is_rejected():
    result = evaluate_safety_gate(**_gate_input(last_valid_stamp_us=1000))
    assert result.reason == STALE_INPUT
    assert not result.valid
    assert (result.desired_x, result.desired_y) == (0.0, 0.0)


def test_wall_clock_expiry_and_estop_are_fail_closed():
    stale = evaluate_safety_gate(
        **_gate_input(time_since_last_valid_sec=1.1)
    )
    assert stale.reason == STALE_INPUT
    assert not stale.valid
    estop = evaluate_safety_gate(
        **_gate_input(time_since_last_valid_sec=2.1)
    )
    assert estop.reason == ESTOP
    assert not estop.valid
    assert estop.safe_stop


def test_collision_checks_only_current_setpoint_at_dt():
    entity = _Entity("e1", 0.8, 0.0, 0.0, 0.0)
    assert (
        _check_collision(0.0, 0.0, [entity], SafetyGateConfig()) is None
    )
    assert (
        _check_collision(0.4, 0.0, [entity], SafetyGateConfig())
        == COLLISION_RISK
    )


def test_moving_entity_is_extrapolated_for_one_dt():
    entity = _Entity("e1", 1.0, 0.0, -1.0, 0.0)
    assert _check_collision(0.0, 0.0, [entity], SafetyGateConfig()) is None


def test_pass_returns_same_single_point():
    result = evaluate_safety_gate(**_gate_input())
    assert result.reason == PASS
    assert result.valid
    assert not result.safe_stop
    assert result.desired_x == pytest.approx(0.05)
    assert result.desired_y == pytest.approx(0.0)


def test_step_over_limit_is_rejected():
    result = evaluate_safety_gate(
        **_gate_input(desired_x=MAX_DISPLACEMENT_M + 0.01)
    )
    assert result.reason == SPEED_LIMIT
    assert not result.valid
    assert (result.desired_x, result.desired_y) == (0.0, 0.0)


def test_config_rejects_invalid_limits():
    with pytest.raises(ValueError):
        SafetyGateConfig(max_step_m=-0.1)
    with pytest.raises(ValueError):
        SafetyGateConfig(collision_margin_m=0.0)
    with pytest.raises(ValueError):
        SafetyGateConfig(stale_timeout_sec=2.0, estop_timeout_sec=1.0)
