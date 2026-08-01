"""VLA runtime trajectory control bridge tests."""

from __future__ import annotations

import math
import pytest

from asv_vla.trajectory_controller import (
    MAX_DESIRED_M,
    STALE_THRESHOLD_SEC,
    ControlCommand,
    trajectory_to_command,
)
from asv_vla.trajectory_contract import ACTION_DIM, HORIZON


def _step_trajectory(dx: float, dy: float) -> list[float]:
    result: list[float] = []
    for i in range(HORIZON):
        result.extend(((i + 1) * dx, (i + 1) * dy))
    return result


class TestNormalFollow:
    def test_follow_outputs_prefix(self) -> None:
        traj = _step_trajectory(0.1, 0.05)
        cmd = trajectory_to_command(
            traj, safe_stop=False, valid=True, reason="PASS", stamp_us=1000
        )
        assert cmd.valid is True
        assert cmd.desired_x == pytest.approx(0.1)  # cumulative waypoint 0
        assert cmd.desired_y == pytest.approx(0.05)

    def test_clip_excessive_displacement(self) -> None:
        traj = _step_trajectory(5.0, 0.0)
        cmd = trajectory_to_command(
            traj, safe_stop=False, valid=True, reason="PASS", stamp_us=1000
        )
        assert cmd.valid is True
        assert abs(cmd.desired_x) <= MAX_DESIRED_M


class TestStopAndReject:
    def test_safe_stop_invalid(self) -> None:
        cmd = trajectory_to_command(
            _step_trajectory(0.1, 0.0),
            safe_stop=True,
            valid=True,
            reason="POLICY_STOP",
            stamp_us=1000,
        )
        assert cmd.valid is False
        assert "STOP" in cmd.detail

    def test_invalid_trajectory_rejected(self) -> None:
        cmd = trajectory_to_command(
            _step_trajectory(0.1, 0.0),
            safe_stop=False,
            valid=False,
            reason="SPEED_LIMIT",
            stamp_us=1000,
        )
        assert cmd.valid is False
        assert "SPEED_LIMIT" in cmd.detail


class TestDuplicateAndStale:
    def test_duplicate_frame_rejected(self) -> None:
        cmd = trajectory_to_command(
            _step_trajectory(0.1, 0.0),
            safe_stop=False,
            valid=True,
            reason="PASS",
            stamp_us=1000,
            last_executed_stamp_us=1000,
        )
        assert cmd.valid is False
        assert "DUPLICATE" in cmd.detail

    def test_stale_rejected(self) -> None:
        cmd = trajectory_to_command(
            _step_trajectory(0.1, 0.0),
            safe_stop=False,
            valid=True,
            reason="PASS",
            stamp_us=2000,
            time_since_last_valid_sec=STALE_THRESHOLD_SEC + 0.1,
        )
        assert cmd.valid is False
        assert "STALE" in cmd.detail


class TestBadShape:
    def test_nan_rejected(self) -> None:
        values = [0.0] * (HORIZON * ACTION_DIM)
        values[0] = float("nan")
        cmd = trajectory_to_command(
            values, safe_stop=False, valid=True, reason="PASS", stamp_us=1000
        )
        assert cmd.valid is False

    def test_wrong_length_rejected(self) -> None:
        cmd = trajectory_to_command(
            [0.0, 0.0], safe_stop=False, valid=True, reason="PASS", stamp_us=1000
        )
        assert cmd.valid is False


class TestZeroTrajectory:
    def test_zero_still_valid(self) -> None:
        cmd = trajectory_to_command(
            [0.0] * (HORIZON * ACTION_DIM),
            safe_stop=False,
            valid=True,
            reason="PASS",
            stamp_us=1000,
        )
        assert cmd.valid is True
        assert cmd.desired_x == 0.0
        assert cmd.desired_y == 0.0
