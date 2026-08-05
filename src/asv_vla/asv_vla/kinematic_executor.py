"""Pure validation for one UE5 body-frame expert action per source frame."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .trajectory_contract import FRAME_ID


DEFAULT_MAX_STEP_M = 0.35
ZERO_TOLERANCE_M = 1.0e-6


class ExpertActionLike(Protocol):
    stamp_us: int
    run_id: str
    scene_seed: int
    frame_index: int
    frame_id: str
    model_version: str
    dt: float
    desired_x: float
    desired_y: float
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


def expert_source_identity(
    source: ExpertActionLike,
) -> tuple[str, int, int, int]:
    """Return the complete source-frame identity used for de-duplication."""

    return (
        str(source.run_id),
        int(source.scene_seed),
        int(source.frame_index),
        int(source.stamp_us),
    )


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
    source: ExpertActionLike,
    *,
    max_step_m: float = DEFAULT_MAX_STEP_M,
) -> KinematicStep:
    """Validate and expose the single body-frame action for UE5 execution."""

    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive and finite")

    try:
        step_dt = float(source.dt)
    except (AttributeError, TypeError, ValueError):
        return invalid_hold("INVALID_DT")

    if not bool(getattr(source, "valid", False)):
        return invalid_hold(
            f"INVALID_EXPERT:{getattr(source, 'detail', '')}",
            step_dt=step_dt,
        )
    if not str(getattr(source, "run_id", "")).strip():
        return invalid_hold("INVALID_RUN_ID", step_dt=step_dt)
    if getattr(source, "frame_id", None) != FRAME_ID:
        return invalid_hold(
            f"INVALID_FRAME:expected={FRAME_ID};got={getattr(source, 'frame_id', None)}",
            step_dt=step_dt,
        )
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        return invalid_hold("INVALID_DT")
    if not hasattr(source, "safe_stop"):
        return invalid_hold(
            "INVALID_ACTION_FIELDS:missing safe_stop",
            step_dt=step_dt,
        )

    try:
        desired_x = float(source.desired_x)
        desired_y = float(source.desired_y)
    except (AttributeError, TypeError, ValueError):
        return invalid_hold(
            "INVALID_ACTION_FIELDS:expected desired_x and desired_y",
            step_dt=step_dt,
        )
    if not math.isfinite(desired_x) or not math.isfinite(desired_y):
        return invalid_hold("NONFINITE_ACTION", step_dt=step_dt)

    action_norm_m = math.hypot(desired_x, desired_y)
    if bool(source.safe_stop):
        if action_norm_m > ZERO_TOLERANCE_M:
            return invalid_hold(
                "INVALID_SAFE_STOP_NONZERO_ACTION",
                step_dt=step_dt,
            )
        return KinematicStep(
            delta_x_m=0.0,
            delta_y_m=0.0,
            step_dt=step_dt,
            hold_position=True,
            valid=True,
            reason=f"SAFE_STOP:{getattr(source, 'detail', '')}",
        )

    if action_norm_m > max_step_m + ZERO_TOLERANCE_M:
        return invalid_hold(
            f"STEP_LIMIT:{action_norm_m:.6f}>{max_step_m:.6f}",
            step_dt=step_dt,
        )

    hold_position = action_norm_m <= ZERO_TOLERANCE_M
    return KinematicStep(
        delta_x_m=0.0 if hold_position else desired_x,
        delta_y_m=0.0 if hold_position else desired_y,
        step_dt=step_dt,
        hold_position=hold_position,
        valid=True,
        reason=(
            f"EXPERT_ACTION:{source.model_version}"
            if not hold_position
            else "EXPERT_ZERO_ACTION_HOLD"
        ),
    )
