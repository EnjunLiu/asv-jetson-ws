"""Online body-frame displacement controller.

The policy and safety gate exchange one ``(desired_x, desired_y)`` command
per frame.  This module validates the command, applies the fixed ``dt`` and
displacement envelope, and never consumes a predicted action sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .trajectory_contract import (
    DT_SEC,
    FLOAT_TOLERANCE,
    MAX_DISPLACEMENT_M,
)

STALE_THRESHOLD_SEC = 1.0
# A reversal can change the requested displacement by 2 * MAX_DISPLACEMENT_M.
# Limiting that change to one envelope per control period prevents a single
# noisy frame from flipping the kinematic setpoint instantaneously.
MAX_RATE_MPS = MAX_DISPLACEMENT_M / DT_SEC
# Compatibility name for callers that already imported the controller limit.
MAX_DESIRED_M = MAX_DISPLACEMENT_M


@dataclass(frozen=True)
class ControlCommand:
    desired_x: float
    desired_y: float
    valid: bool
    detail: str


def _zero(detail: str) -> ControlCommand:
    return ControlCommand(0.0, 0.0, False, detail)


def _bounded_vector(x: float, y: float, limit: float) -> tuple[float, float] | None:
    norm = math.hypot(x, y)
    if not math.isfinite(norm) or norm > limit + FLOAT_TOLERANCE:
        return None
    return x, y


def point_to_command(
    desired_x: float,
    desired_y: float,
    *,
    safe_stop: bool,
    valid: bool,
    reason: str,
    stamp_us: int,
    dt: float = DT_SEC,
    last_executed_stamp_us: int = 0,
    time_since_last_valid_sec: float = 0.0,
    previous_desired: tuple[float, float] | None = None,
    max_displacement_m: float = MAX_DISPLACEMENT_M,
    max_rate_mps: float = MAX_RATE_MPS,
) -> ControlCommand:
    """Validate and rate-limit one body-frame displacement command.

    Invalid, stale, and safe-stop inputs are always non-executable zero
    commands.  A valid zero displacement is allowed only when ``safe_stop``
    is false, because it represents an ordinary position hold at standoff.
    """

    if int(stamp_us) <= int(last_executed_stamp_us):
        return _zero("DUPLICATE_FRAME")
    if safe_stop:
        return _zero(f"STOP:{reason}")
    if not valid:
        return _zero(f"REJECTED:{reason}")
    if not math.isfinite(float(dt)) or abs(float(dt) - DT_SEC) > FLOAT_TOLERANCE:
        return _zero("INVALID_DT")
    if not math.isfinite(float(time_since_last_valid_sec)):
        return _zero("NONFINITE_AGE")
    if time_since_last_valid_sec > STALE_THRESHOLD_SEC:
        return _zero("STALE_DISPLACEMENT")

    try:
        x = float(desired_x)
        y = float(desired_y)
    except (TypeError, ValueError):
        return _zero("INVALID_DISPLACEMENT")
    if not (math.isfinite(x) and math.isfinite(y)):
        return _zero("NONFINITE")

    try:
        limit = float(max_displacement_m)
        rate = float(max_rate_mps)
    except (TypeError, ValueError):
        return _zero("INVALID_LIMIT")
    if not (math.isfinite(limit) and limit > 0.0 and math.isfinite(rate) and rate > 0.0):
        return _zero("INVALID_LIMIT")
    bounded = _bounded_vector(x, y, limit)
    if bounded is None:
        return _zero("DISPLACEMENT_LIMIT")
    x, y = bounded

    detail = "EXECUTE_DISPLACEMENT"
    if previous_desired is not None:
        try:
            previous_x = float(previous_desired[0])
            previous_y = float(previous_desired[1])
        except (TypeError, ValueError, IndexError):
            return _zero("INVALID_PREVIOUS_DISPLACEMENT")
        if not (math.isfinite(previous_x) and math.isfinite(previous_y)):
            return _zero("NONFINITE_PREVIOUS_DISPLACEMENT")
        max_delta = rate * float(dt)
        delta_x = x - previous_x
        delta_y = y - previous_y
        delta_norm = math.hypot(delta_x, delta_y)
        if delta_norm > max_delta + FLOAT_TOLERANCE:
            scale = max_delta / delta_norm
            x = previous_x + delta_x * scale
            y = previous_y + delta_y * scale
            bounded = _bounded_vector(x, y, limit)
            if bounded is None:
                return _zero("RATE_LIMIT_CONTROL_UNREACHABLE")
            x, y = bounded
            detail = "EXECUTE_DISPLACEMENT_RATE_LIMITED"

    return ControlCommand(float(x), float(y), True, detail)
