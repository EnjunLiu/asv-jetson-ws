import pytest

from asv_vla.trajectory_controller import (
    MAX_DESIRED_M,
    STALE_THRESHOLD_SEC,
    point_to_command,
)
from asv_vla.trajectory_contract import DT_SEC


def _command(**overrides):
    values = dict(
        desired_x=0.1,
        desired_y=0.05,
        safe_stop=False,
        valid=True,
        reason="PASS",
        stamp_us=1000,
        dt=DT_SEC,
    )
    values.update(overrides)
    return point_to_command(**values)


def test_single_point_is_executed_without_trajectory_parsing():
    command = _command()
    assert command.valid
    assert command.desired_x == pytest.approx(0.1)
    assert command.desired_y == pytest.approx(0.05)


def test_displacement_norm_is_limited():
    command = _command(desired_x=MAX_DESIRED_M, desired_y=0.01)
    assert not command.valid
    assert command.detail == "DISPLACEMENT_LIMIT"


def test_invalid_stop_is_zero_and_not_valid():
    command = _command(safe_stop=True, reason="POLICY_STOP")
    assert not command.valid
    assert (command.desired_x, command.desired_y) == (0.0, 0.0)


def test_invalid_input_is_zero():
    command = _command(valid=False, reason="NONFINITE")
    assert not command.valid
    assert (command.desired_x, command.desired_y) == (0.0, 0.0)


def test_nonfinite_and_wrong_dt_are_rejected():
    assert not _command(desired_x=float("nan")).valid
    assert _command(dt=0.1).detail == "INVALID_DT"


def test_duplicate_and_stale_points_are_rejected():
    assert _command(last_executed_stamp_us=1000).detail == "DUPLICATE_FRAME"
    assert _command(
        time_since_last_valid_sec=STALE_THRESHOLD_SEC + 0.1
    ).detail == "STALE_DISPLACEMENT"


def test_reversal_is_rate_limited():
    command = _command(
        desired_x=-MAX_DESIRED_M,
        desired_y=0.0,
        previous_desired=(MAX_DESIRED_M, 0.0),
        stamp_us=2000,
    )
    assert command.valid
    assert command.detail == "EXECUTE_DISPLACEMENT_RATE_LIMITED"
    assert command.desired_x == pytest.approx(0.0)
    assert command.desired_y == pytest.approx(0.0)


def test_valid_zero_displacement_is_an_ordinary_hold():
    command = _command(desired_x=0.0, desired_y=0.0)
    assert command.valid
    assert not command.detail.startswith("STOP")
