"""Deterministic safety gate for one online body-frame displacement.

The gate is the only component between the CUDA policy and the kinematic
controller.  It validates the current ``(desired_x, desired_y)`` command,
checks its one-step collision envelope, and fails closed on every rejection.
The model's offline [20, 2] output is not accepted at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .trajectory_contract import (
    DT_SEC,
    FLOAT_TOLERANCE,
    FRAME_ID,
    MAX_DISPLACEMENT_M,
)

PASS = "PASS"
POLICY_STOP = "POLICY_STOP"
STALE_INPUT = "STALE_INPUT"
INVALID_MODALITY = "INVALID_MODALITY"
INVALID_SHAPE = "INVALID_SHAPE"
NONFINITE = "NONFINITE"
SPEED_LIMIT = "SPEED_LIMIT"
COLLISION_RISK = "COLLISION_RISK"
CONTROL_UNREACHABLE = "CONTROL_UNREACHABLE"
ESTOP = "ESTOP"

REJECTION_CODES = frozenset(
    {
        STALE_INPUT,
        INVALID_MODALITY,
        INVALID_SHAPE,
        NONFINITE,
        SPEED_LIMIT,
        COLLISION_RISK,
        CONTROL_UNREACHABLE,
        ESTOP,
    }
)

DEFAULT_MAX_STEP_M = MAX_DISPLACEMENT_M
DEFAULT_MAX_SPEED_MPS = DEFAULT_MAX_STEP_M / DT_SEC
DEFAULT_STALE_TIMEOUT_SEC = 1.0
DEFAULT_ESTOP_TIMEOUT_SEC = 2.0
DEFAULT_COLLISION_MARGIN_M = 0.5


@dataclass(frozen=True)
class SafetyGateConfig:
    """Limits for one body-frame command evaluated over ``dt``."""

    max_step_m: float = DEFAULT_MAX_STEP_M
    stale_timeout_sec: float = DEFAULT_STALE_TIMEOUT_SEC
    estop_timeout_sec: float = DEFAULT_ESTOP_TIMEOUT_SEC
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_step_m) or self.max_step_m <= 0.0:
            raise ValueError("max_step_m must be positive and finite")
        if not math.isfinite(self.stale_timeout_sec) or self.stale_timeout_sec <= 0.0:
            raise ValueError("stale_timeout_sec must be positive and finite")
        if not math.isfinite(self.estop_timeout_sec) or self.estop_timeout_sec <= 0.0:
            raise ValueError("estop_timeout_sec must be positive and finite")
        if self.estop_timeout_sec < self.stale_timeout_sec:
            raise ValueError("estop_timeout_sec must be >= stale_timeout_sec")
        if not math.isfinite(self.collision_margin_m) or self.collision_margin_m <= 0.0:
            raise ValueError("collision_margin_m must be positive and finite")


@dataclass(frozen=True)
class SafetyGateResult:
    desired_x: float
    desired_y: float
    safe_stop: bool
    valid: bool
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class _Entity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_vx: float
    relative_vy: float


def _reject(code: str, detail: str = "") -> SafetyGateResult:
    """Return a non-executable zero command for every rejected input."""

    return SafetyGateResult(
        desired_x=0.0,
        desired_y=0.0,
        safe_stop=True,
        valid=False,
        reason=code,
        detail=detail,
    )


def _check_modality_and_shape(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    desired_x: float,
    desired_y: float,
    policy_valid: bool,
    language_valid: bool,
    entity_valid: bool,
    last_valid_stamp_us: int,
) -> str | None:
    """Return a rejection code before any command is executed."""

    if int(stamp_us) <= 0 or int(stamp_us) <= int(last_valid_stamp_us):
        return STALE_INPUT
    if not all((language_valid, entity_valid)):
        return INVALID_MODALITY
    if not policy_valid:
        return INVALID_MODALITY
    if not str(run_id).strip() or not str(model_version).strip():
        return INVALID_MODALITY
    if frame_id != FRAME_ID:
        return INVALID_MODALITY
    if not math.isfinite(float(dt)) or abs(float(dt) - DT_SEC) > FLOAT_TOLERANCE:
        return INVALID_SHAPE
    try:
        x = float(desired_x)
        y = float(desired_y)
    except (TypeError, ValueError):
        return INVALID_SHAPE
    if not (math.isfinite(x) and math.isfinite(y)):
        return NONFINITE
    return None


def _check_kinematics(
    desired_x: float,
    desired_y: float,
    config: SafetyGateConfig,
) -> str | None:
    """Check the norm of the one-step displacement."""

    norm = math.hypot(float(desired_x), float(desired_y))
    if not math.isfinite(norm):
        return NONFINITE
    if norm > config.max_step_m + FLOAT_TOLERANCE:
        return SPEED_LIMIT
    return None


def _check_entity_finiteness(entities: Sequence[_Entity]) -> str | None:
    for entity in entities:
        values = (
            entity.relative_x,
            entity.relative_y,
            entity.relative_vx,
            entity.relative_vy,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return NONFINITE
    return None


def _check_collision(
    desired_x: float,
    desired_y: float,
    entities: Sequence[_Entity],
    config: SafetyGateConfig,
    *,
    dt: float = DT_SEC,
) -> str | None:
    """Check the one executed setpoint against constant-velocity entities."""

    for entity in entities:
        predicted_x = float(entity.relative_x) + float(entity.relative_vx) * float(dt)
        predicted_y = float(entity.relative_y) + float(entity.relative_vy) * float(dt)
        distance = math.hypot(
            float(desired_x) - predicted_x,
            float(desired_y) - predicted_y,
        )
        if not math.isfinite(distance):
            return NONFINITE
        if distance < config.collision_margin_m:
            return COLLISION_RISK
    return None


def evaluate_safety_gate(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    desired_x: float,
    desired_y: float,
    safe_stop: bool,
    valid: bool,
    reason: str,
    language_valid: bool = True,
    entity_valid: bool = True,
    entities: Sequence[_Entity] | None = None,
    last_valid_stamp_us: int = 0,
    time_since_last_valid_sec: float = 0.0,
    config: SafetyGateConfig | None = None,
) -> SafetyGateResult:
    """Evaluate one body-frame displacement through the safety gate."""

    cfg = config or SafetyGateConfig()
    rejection = _check_modality_and_shape(
        stamp_us=stamp_us,
        run_id=run_id,
        frame_id=frame_id,
        model_version=model_version,
        dt=dt,
        desired_x=desired_x,
        desired_y=desired_y,
        policy_valid=valid,
        language_valid=language_valid,
        entity_valid=entity_valid,
        last_valid_stamp_us=last_valid_stamp_us,
    )
    if rejection is not None:
        return _reject(rejection, reason)

    # A stop is an invalid hold marker, never a valid zero action.
    if safe_stop:
        return _reject(POLICY_STOP, reason or "policy requested stop")

    try:
        age = float(time_since_last_valid_sec)
    except (TypeError, ValueError):
        return _reject(NONFINITE, "invalid policy age")
    if not math.isfinite(age) or age < 0.0:
        return _reject(NONFINITE, "invalid policy age")
    if age > cfg.estop_timeout_sec:
        return _reject(ESTOP, f"no valid policy for {age:.2f}s")
    if age > cfg.stale_timeout_sec:
        return _reject(STALE_INPUT, f"last valid policy is {age:.2f}s old")

    rejection = _check_kinematics(desired_x, desired_y, cfg)
    if rejection is not None:
        return _reject(rejection)

    checked_entities = tuple(entities or ())
    rejection = _check_entity_finiteness(checked_entities)
    if rejection is not None:
        return _reject(rejection)
    if checked_entities:
        rejection = _check_collision(
            desired_x,
            desired_y,
            checked_entities,
            cfg,
            dt=float(dt),
        )
        if rejection is not None:
            return _reject(rejection)

    return SafetyGateResult(
        desired_x=float(desired_x),
        desired_y=float(desired_y),
        safe_stop=False,
        valid=True,
        reason=PASS,
    )
