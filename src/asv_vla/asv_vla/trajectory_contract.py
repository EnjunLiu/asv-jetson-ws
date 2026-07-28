from __future__ import annotations

import math
from typing import Protocol, Sequence


HORIZON = 20
ACTION_DIM = 2
DT_SEC = 0.2
FRAME_ID = "base_link"
SAFE_STOP_MODEL_VERSION = "stub:none"
FLOAT_TOLERANCE = 1.0e-6


class SelectedTrajectoryLike(Protocol):
    stamp_us: int
    run_id: str
    frame_id: str
    model_version: str
    dt: float
    horizon: int
    delta_p_xy: Sequence[float]
    safe_stop: bool
    valid: bool


def finite_zero(value: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isfinite(value) and abs(value) <= tolerance


def is_day1_safe_stop(message: SelectedTrajectoryLike) -> bool:
    """Validate the executable-neutral Day 1 trajectory container.

    ``valid`` means that the trajectory message itself is well formed. The
    downstream Day 1 controller still publishes ``DecisionOutput.valid=false``;
    a valid all-zero displacement must never be treated as a position-hold
    command.
    """

    return (
        message.stamp_us > 0
        and bool(message.run_id)
        and message.frame_id == FRAME_ID
        and message.model_version == SAFE_STOP_MODEL_VERSION
        and message.horizon == HORIZON
        and math.isfinite(message.dt)
        and abs(message.dt - DT_SEC) <= FLOAT_TOLERANCE
        and len(message.delta_p_xy) == HORIZON * ACTION_DIM
        and all(finite_zero(value) for value in message.delta_p_xy)
        and message.safe_stop
        and message.valid
    )
