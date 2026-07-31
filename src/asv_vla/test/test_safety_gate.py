"""Day 17 safety gate contract tests."""

from __future__ import annotations

import math
import pytest

from asv_vla.safety_gate import (
    COLLISION_RISK,
    CURVATURE_LIMIT,
    ESTOP,
    INVALID_MODALITY,
    INVALID_SHAPE,
    NONFINITE,
    PASS,
    POLICY_STOP,
    SPEED_LIMIT,
    STALE_INPUT,
    SafetyGateConfig,
    SafetyGateResult,
    _Entity,
    _check_collision,
    _check_kinematics,
    _check_modality_and_shape,
    _deceleration_trajectory,
    _three_point_curvature,
    evaluate_safety_gate,
)

from asv_vla.trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, HORIZON

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _modality_input(**overrides) -> dict:
    """Return kwargs accepted by _check_modality_and_shape."""
    defaults = dict(
        stamp_us=1000,
        run_id="test_run",
        frame_id=FRAME_ID,
        dt=DT_SEC,
        horizon=HORIZON,
        delta_p_xy=(0.0,) * (HORIZON * ACTION_DIM),
        policy_valid=True,
        language_valid=True,
        visual_valid=True,
        entity_valid=True,
        ego_valid=True,
        last_valid_stamp_us=0,
    )
    defaults.update(overrides)
    return defaults


def _gate_input(**overrides) -> dict:
    """Return kwargs accepted by evaluate_safety_gate."""
    defaults = dict(
        stamp_us=1000,
        run_id="test_run",
        frame_id=FRAME_ID,
        model_version="policy_v1",
        dt=DT_SEC,
        horizon=HORIZON,
        delta_p_xy=(0.0,) * (HORIZON * ACTION_DIM),
        safe_stop=False,
        valid=True,
        reason="test",
        language_valid=True,
        visual_valid=True,
        entity_valid=True,
        ego_valid=True,
        last_valid_stamp_us=0,
    )
    defaults.update(overrides)
    return defaults


def _step_trajectory(dx: float, dy: float) -> tuple[float, ...]:
    """Build a trajectory of 20 constant steps."""
    result: list[float] = []
    for i in range(HORIZON):
        result.extend(((i + 1) * dx, (i + 1) * dy))
    return tuple(result)


# ---------------------------------------------------------------------------
# Phase 1: modality & shape
# ---------------------------------------------------------------------------


class TestModalityAndShape:
    def test_valid_passes(self) -> None:
        assert _check_modality_and_shape(**_modality_input(), config=SafetyGateConfig()) is None

    def test_stale_stamp_rejected(self) -> None:
        inp = _modality_input(stamp_us=500, last_valid_stamp_us=500)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == STALE_INPUT

    def test_zero_stamp_rejected(self) -> None:
        inp = _modality_input(stamp_us=0)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == STALE_INPUT

    def test_invalid_language_rejected(self) -> None:
        inp = _modality_input(language_valid=False)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_MODALITY

    def test_invalid_visual_rejected(self) -> None:
        inp = _modality_input(visual_valid=False)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_MODALITY

    def test_empty_run_id_rejected(self) -> None:
        inp = _modality_input(run_id="")
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_MODALITY

    def test_wrong_frame_id_rejected(self) -> None:
        inp = _modality_input(frame_id="odom")
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_MODALITY

    def test_wrong_dt_rejected(self) -> None:
        inp = _modality_input(dt=0.5)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_SHAPE

    def test_wrong_horizon_rejected(self) -> None:
        inp = _modality_input(horizon=10)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_SHAPE

    def test_wrong_shape_rejected(self) -> None:
        inp = _modality_input(delta_p_xy=(0.0, 0.0))
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_SHAPE

    def test_nan_rejected(self) -> None:
        values = [0.0] * (HORIZON * ACTION_DIM)
        values[5] = float("nan")
        inp = _modality_input(delta_p_xy=values)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == NONFINITE

    def test_inf_rejected(self) -> None:
        values = [0.0] * (HORIZON * ACTION_DIM)
        values[10] = float("inf")
        inp = _modality_input(delta_p_xy=values)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == NONFINITE

    def test_policy_invalid_rejected(self) -> None:
        inp = _modality_input(policy_valid=False)
        assert _check_modality_and_shape(**inp, config=SafetyGateConfig()) == INVALID_MODALITY


# ---------------------------------------------------------------------------
# Phase 2: kinematics
# ---------------------------------------------------------------------------


class TestKinematics:
    def test_normal_trajectory_passes(self) -> None:
        traj = _step_trajectory(0.1, 0.0)
        assert _check_kinematics(traj, SafetyGateConfig()) is None

    def test_excessive_step_rejected(self) -> None:
        config = SafetyGateConfig(max_step_m=0.3)
        traj = _step_trajectory(0.35, 0.0)
        assert _check_kinematics(traj, config) == SPEED_LIMIT

    def test_zero_trajectory_passes(self) -> None:
        traj = (0.0,) * (HORIZON * ACTION_DIM)
        assert _check_kinematics(traj, SafetyGateConfig()) is None

    def test_curvature_rejected(self) -> None:
        config = SafetyGateConfig(max_step_m=2.0, max_curvature=0.1)
        # Tight zigzag: small lateral jumps, curvature >> 0.1.
        values: list[float] = []
        for i in range(HORIZON):
            x = i * 0.05
            y = 0.5 if i % 2 == 0 else -0.5
            values.extend((x, y))
        assert _check_kinematics(tuple(values), config) == CURVATURE_LIMIT


class TestCurvature:
    def test_straight_line_zero(self) -> None:
        assert _three_point_curvature((0, 0), (1, 0), (2, 0)) == pytest.approx(0.0)

    def test_right_angle(self) -> None:
        c = _three_point_curvature((0, 0), (1, 0), (1, 1))
        assert c > 0.5

    def test_degenerate_points_zero(self) -> None:
        assert _three_point_curvature((0, 0), (0, 0), (1, 1)) == 0.0


# ---------------------------------------------------------------------------
# Phase 3: collision
# ---------------------------------------------------------------------------


class TestCollision:
    def test_no_entities_passes(self) -> None:
        traj = _step_trajectory(0.1, 0.0)
        assert _check_collision(traj, [], SafetyGateConfig()) is None

    def test_far_entity_passes(self) -> None:
        traj = _step_trajectory(0.1, 0.0)
        entities = [_Entity("e1", 10.0, 0.0, 0.0, 0.0)]
        assert _check_collision(traj, entities, SafetyGateConfig()) is None

    def test_close_static_entity_rejected(self) -> None:
        traj = _step_trajectory(0.4, 0.0)
        # Margin is 0.5 m: an executed waypoint at 0.4 m with the entity at
        # 0.2 m leaves 0.2 m clearance -> rejected.
        entities = [_Entity("e1", 0.2, 0.0, 0.0, 0.0)]
        assert _check_collision(traj, entities, SafetyGateConfig()) == COLLISION_RISK
        # Beyond the margin from every executed waypoint is accepted.
        entities = [_Entity("e1", 1.5, 0.0, 0.0, 0.0)]
        assert _check_collision(traj, entities, SafetyGateConfig()) is None

    def test_moving_entity_extrapolated(self) -> None:
        traj = _step_trajectory(0.5, 0.0)
        # Entity moving away fast enough to clear.
        entities = [_Entity("e1", 0.5, 0.0, -10.0, 0.0)]
        assert _check_collision(traj, entities, SafetyGateConfig()) is None


# ---------------------------------------------------------------------------
# Full gate
# ---------------------------------------------------------------------------


class TestFullGate:
    def test_pass_returns_same_trajectory(self) -> None:
        traj = _step_trajectory(0.1, 0.0)
        result = evaluate_safety_gate(**_gate_input(delta_p_xy=traj))
        assert result.reason == PASS
        assert result.valid is True
        assert result.safe_stop is False
        assert result.delta_p_xy == traj

    def test_policy_stop_produces_zero_trajectory(self) -> None:
        result = evaluate_safety_gate(**_gate_input(safe_stop=True))
        assert result.reason == POLICY_STOP
        assert result.safe_stop is True
        assert result.valid is True
        assert all(v == 0.0 for v in result.delta_p_xy)

    def test_nan_produces_invalid_rejection(self) -> None:
        values = [0.0] * (HORIZON * ACTION_DIM)
        values[3] = float("nan")
        result = evaluate_safety_gate(**_gate_input(delta_p_xy=values))
        assert result.reason == NONFINITE
        assert result.valid is False

    def test_estop_on_timeout(self) -> None:
        result = evaluate_safety_gate(
            **_gate_input(),
            time_since_last_valid_sec=3.0,
        )
        assert result.reason == ESTOP
        assert result.valid is False
        assert result.safe_stop is True

    def test_speed_limit_rejection(self) -> None:
        config = SafetyGateConfig(max_step_m=0.3)
        traj = _step_trajectory(0.35, 0.0)
        result = evaluate_safety_gate(**_gate_input(delta_p_xy=traj), config=config)
        assert result.reason == SPEED_LIMIT
        assert result.valid is False

    def test_deceleration_nonzero_trajectory(self) -> None:
        healthy = _step_trajectory(0.1, 0.0)
        decel = _deceleration_trajectory(healthy, SafetyGateConfig())
        assert len(decel) == HORIZON * ACTION_DIM
        # Should be non-zero but smaller than original.
        assert any(abs(v) > 1e-6 for v in decel)

    def test_deceleration_none_healthy(self) -> None:
        decel = _deceleration_trajectory(None, SafetyGateConfig())
        assert all(v == 0.0 for v in decel)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_negative_max_step_raises(self) -> None:
        with pytest.raises(ValueError):
            SafetyGateConfig(max_step_m=-0.1)

    def test_zero_collision_margin_raises(self) -> None:
        with pytest.raises(ValueError):
            SafetyGateConfig(collision_margin_m=0.0)
