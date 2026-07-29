"""Pure validation for UE5-only receding-horizon expert execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from .trajectory_contract import ACTION_DIM, FRAME_ID


DEFAULT_MAX_STEP_M = 0.35
ZERO_TOLERANCE_M = 1.0e-6


class ExpertTrajectoryLike(Protocol):
    run_id: str
    frame_id: str
    model_version: str
    dt: float
    horizon: int
    delta_p_xy: Sequence[float]
    safe_stop: bool
    valid: bool
    detail: str


@dataclass(frozen=True)
class KinematicStep:
    delta_x_m: float
    delta_y_m: float
    step_dt: float
    hold_position: bool
    valid: bool
    reason: str


def invalid_hold(reason: str, *, step_dt: float = 0.0) -> KinematicStep:
    """Return a non-executable command that UE5 must interpret as hold."""

    return KinematicStep(
        delta_x_m=0.0,
        delta_y_m=0.0,
        step_dt=step_dt if math.isfinite(step_dt) and step_dt > 0.0 else 0.0,
        hold_position=True,
        valid=False,
        reason=reason,
    )


def first_step_from_expert(
    source: ExpertTrajectoryLike,
    *,
    max_step_m: float = DEFAULT_MAX_STEP_M,
) -> KinematicStep:
    """Extract only the first point of the latest expert trajectory.

    ``delta_p_xy`` contains cumulative body-frame waypoints from the current
    planning origin. Receding-horizon execution therefore consumes waypoint 0
    once, waits for a newer source frame, and replans instead of walking all
    20 points open loop.
    """

    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive and finite")

    step_dt = float(source.dt)
    if not source.valid:
        return invalid_hold(
            f"INVALID_EXPERT:{source.detail}",
            step_dt=step_dt,
        )
    if not str(source.run_id).strip():
        return invalid_hold("INVALID_RUN_ID", step_dt=step_dt)
    if source.frame_id != FRAME_ID:
        return invalid_hold(
            f"INVALID_FRAME:expected={FRAME_ID};got={source.frame_id}",
            step_dt=step_dt,
        )
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        return invalid_hold("INVALID_DT")
    if int(source.horizon) <= 0:
        return invalid_hold("INVALID_HORIZON", step_dt=step_dt)

    values = tuple(float(value) for value in source.delta_p_xy)
    expected_values = int(source.horizon) * ACTION_DIM
    if len(values) != expected_values:
        return invalid_hold(
            f"INVALID_SHAPE:expected={expected_values};got={len(values)}",
            step_dt=step_dt,
        )
    if not all(math.isfinite(value) for value in values):
        return invalid_hold("NONFINITE_TRAJECTORY", step_dt=step_dt)

    delta_x_m, delta_y_m = values[:ACTION_DIM]
    step_norm_m = math.hypot(delta_x_m, delta_y_m)
    if step_norm_m > max_step_m + ZERO_TOLERANCE_M:
        return invalid_hold(
            f"STEP_LIMIT:{step_norm_m:.6f}>{max_step_m:.6f}",
            step_dt=step_dt,
        )

    all_zero = all(abs(value) <= ZERO_TOLERANCE_M for value in values)
    if source.safe_stop:
        if not all_zero:
            return invalid_hold(
                "INVALID_SAFE_STOP_NONZERO_TRAJECTORY",
                step_dt=step_dt,
            )
        return KinematicStep(
            delta_x_m=0.0,
            delta_y_m=0.0,
            step_dt=step_dt,
            hold_position=True,
            valid=True,
            reason=f"SAFE_STOP:{source.detail}",
        )

    hold_position = step_norm_m <= ZERO_TOLERANCE_M
    return KinematicStep(
        delta_x_m=0.0 if hold_position else delta_x_m,
        delta_y_m=0.0 if hold_position else delta_y_m,
        step_dt=step_dt,
        hold_position=hold_position,
        valid=True,
        reason=(
            f"EXPERT_FIRST_POINT:{source.model_version}"
            if not hold_position
            else "EXPERT_ZERO_FIRST_POINT_HOLD"
        ),
    )
