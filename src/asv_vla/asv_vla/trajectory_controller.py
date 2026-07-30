"""Day 18 trajectory control bridge.

Rolling execution of the *prefix* (0.2–0.5 s) of the safe trajectory.
Only publishes ``desired_x`` / ``desired_y`` — never thruster values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .trajectory_contract import ACTION_DIM, DT_SEC, FLOAT_TOLERANCE, HORIZON

EXECUTE_WAYPOINTS = 2  # execute 0.4 s of the 4 s trajectory
STALE_THRESHOLD_SEC = 0.6  # trajectory older than this → invalid hold
MAX_DESIRED_M = 3.0  # maximum single-step desired displacement


@dataclass(frozen=True)
class ControlCommand:
    desired_x: float
    desired_y: float
    valid: bool
    detail: str


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def trajectory_to_command(
    delta_p_xy: Sequence[float],
    *,
    safe_stop: bool,
    valid: bool,
    reason: str,
    stamp_us: int,
    last_executed_stamp_us: int = 0,
    time_since_last_valid_sec: float = 0.0,
) -> ControlCommand:
    """Convert a safety-gated trajectory to a single-step control command.

    Only the first ``EXECUTE_WAYPOINTS`` waypoints of the prefix are
    consumed.  The controller does *not* walk the full trajectory.
    """

    # Duplicate frame → no new command.
    if stamp_us <= last_executed_stamp_us:
        return ControlCommand(0.0, 0.0, False, "DUPLICATE_FRAME")

    # Safety gate rejected → propagate invalid.
    if not valid:
        return ControlCommand(0.0, 0.0, False, f"REJECTED:{reason}")

    # STOP from policy or safety gate.
    if safe_stop:
        return ControlCommand(0.0, 0.0, False, f"STOP:{reason}")

    # Stale trajectory.
    if time_since_last_valid_sec > STALE_THRESHOLD_SEC:
        return ControlCommand(0.0, 0.0, False, "STALE_TRAJECTORY")

    # Validate shape.
    values = tuple(float(v) for v in delta_p_xy)
    if len(values) != HORIZON * ACTION_DIM:
        return ControlCommand(0.0, 0.0, False, "INVALID_SHAPE")
    if not all(math.isfinite(v) for v in values):
        return ControlCommand(0.0, 0.0, False, "NONFINITE")

    # Extract first waypoint's displacement from origin.
    # The trajectory stores cumulative displacements; waypoint 0 = first step.
    first_x = values[0]
    first_y = values[1]

    if EXECUTE_WAYPOINTS >= 2 and len(values) >= 2 * ACTION_DIM:
        # Average the first two steps for smoother motion.
        second_x = values[2]
        second_y = values[3]
        step_x = (first_x + second_x) / 2.0
        step_y = (first_y + second_y) / 2.0
    else:
        step_x = first_x
        step_y = first_y

    step_x = _clip(step_x, MAX_DESIRED_M)
    step_y = _clip(step_y, MAX_DESIRED_M)

    if not (math.isfinite(step_x) and math.isfinite(step_y)):
        return ControlCommand(0.0, 0.0, False, "NONFINITE_CONTROL")

    return ControlCommand(step_x, step_y, True, "EXEC_PREFIX")
