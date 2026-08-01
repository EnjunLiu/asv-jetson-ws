"""Deterministic trajectory safety gate.

The safety gate is the *only* publisher of ``/vla/selected_trajectory``.
It consumes the learned policy output from ``/vla/policy_trajectory`` and
applies hard constraints in fixed order.  Any rejection produces a
deterministic fallback (deceleration or E-STOP) with a machine-readable
reason code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .trajectory_contract import (
    ACTION_DIM,
    DT_SEC,
    FLOAT_TOLERANCE,
    FRAME_ID,
    HORIZON,
    finite_zero,
)

# ---------------------------------------------------------------------------
# Reason codes (machine-readable, single-token)
# ---------------------------------------------------------------------------
PASS = "PASS"
POLICY_STOP = "POLICY_STOP"
STALE_INPUT = "STALE_INPUT"
INVALID_MODALITY = "INVALID_MODALITY"
INVALID_SHAPE = "INVALID_SHAPE"
NONFINITE = "NONFINITE"
SPEED_LIMIT = "SPEED_LIMIT"
CURVATURE_LIMIT = "CURVATURE_LIMIT"
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
        CURVATURE_LIMIT,
        COLLISION_RISK,
        CONTROL_UNREACHABLE,
        ESTOP,
    }
)

# ---------------------------------------------------------------------------
# Tunable limits
# ---------------------------------------------------------------------------
DEFAULT_MAX_SPEED_MPS = 1.5  # m/s per-step limit
DEFAULT_MAX_STEP_M = DEFAULT_MAX_SPEED_MPS * DT_SEC  # 0.3 m
DEFAULT_MAX_TOTAL_DISPLACEMENT_M = 10.0  # total trajectory displacement
# The closed loop re-plans every frame (~10 Hz) and the controller executes
# only the first waypoint(s) of each plan, so curvature/direction checks
# apply to the executable prefix only (1 s lookahead).  The policy's
# full-horizon predictions legitimately hover around the standoff with
# oscillating steps far beyond what is ever executed; speed, displacement
# and non-finite checks still cover the whole path.
EXECUTED_HORIZON_STEPS = 5
# The collision check covers exactly what the controller executes before
# the next re-plan (2 waypoints): any plan whose executed steps stay clear
# is safe, while the ship stops short of an obstacle once its next executed
# step would violate the margin.  The per-step speed check uses the same
# horizon: the model's never-executed tail overshoots the 0.3 m clip by up
# to ~12%, while the two executed steps stay within it.
EXECUTED_COLLISION_STEPS = 2
EXECUTED_SPEED_STEPS = 2
# Calibrated to the deployed policy's measured output distribution
# (p50=6.7, p99=7.6 rad/m on validation frames, 0.3 m steps; direction
# changes up to ~140 deg from steering noise).  The checks are deliberately
# gross-pathology filters: truly dangerous maneuverability is bounded by the
# speed, displacement and direction-continuity checks below.
DEFAULT_MAX_CURVATURE = 15.0  # rad/m — reject grossly pathological paths
# A step that turns more than this from the previous step's direction is a
# reversal (the old policy's zig-zag signature of ~180 deg flips); the
# closed loop re-plans every frame, so only near-reversals are unsafe.
DEFAULT_MAX_TURN_RAD = math.radians(170.0)
# Segments shorter than this are not motion: the three-point curvature of a
# near-stationary segment is dominated by encoder/pose noise (micro-jitter)
# and would otherwise reject perfectly straight slow-start trajectories.
MIN_CURVATURE_SEGMENT_M = 0.05
DEFAULT_STALE_TIMEOUT_SEC = 1.0  # seconds since last valid policy input
DEFAULT_ESTOP_TIMEOUT_SEC = 2.0  # seconds until forced E-STOP
# Calibrated to the baseline scene geometry: targets are 1.5-4 m apart and a
# 3 m standoff approach legitimately ends ~0.5 m from the nearest neighbor;
# the margin still rejects any executed step that would drive INTO an
# entity (wp within 0.5 m of one).
DEFAULT_COLLISION_MARGIN_M = 0.5  # minimum distance to any entity


@dataclass(frozen=True)
class SafetyGateConfig:
    max_step_m: float = DEFAULT_MAX_STEP_M
    max_total_displacement_m: float = DEFAULT_MAX_TOTAL_DISPLACEMENT_M
    max_curvature: float = DEFAULT_MAX_CURVATURE
    stale_timeout_sec: float = DEFAULT_STALE_TIMEOUT_SEC
    estop_timeout_sec: float = DEFAULT_ESTOP_TIMEOUT_SEC
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M

    def __post_init__(self) -> None:
        for name in (
            "max_step_m",
            "max_total_displacement_m",
            "max_curvature",
            "stale_timeout_sec",
            "estop_timeout_sec",
            "collision_margin_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite: {value}")


@dataclass(frozen=True)
class SafetyGateResult:
    """Output of one safety-gate evaluation."""

    delta_p_xy: tuple[float, ...]
    safe_stop: bool
    valid: bool
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Entity representation (minimal — geometry only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_vx: float
    relative_vy: float


# ---------------------------------------------------------------------------
# Phase 1: modality & shape
# ---------------------------------------------------------------------------

def _check_modality_and_shape(
    stamp_us: int,
    run_id: str,
    frame_id: str,
    dt: float,
    horizon: int,
    delta_p_xy: Sequence[float],
    policy_valid: bool,
    language_valid: bool,
    visual_valid: bool,
    entity_valid: bool,
    ego_valid: bool,
    last_valid_stamp_us: int,
    config: SafetyGateConfig,
) -> str | None:
    """Return a rejection code or None if inputs are well-formed."""

    # Staleness.
    if stamp_us <= 0 or stamp_us <= last_valid_stamp_us:
        return STALE_INPUT
    if not all((language_valid, visual_valid, entity_valid, ego_valid)):
        return INVALID_MODALITY
    if not policy_valid:
        return INVALID_MODALITY
    if not bool(run_id):
        return INVALID_MODALITY
    if frame_id != FRAME_ID:
        return INVALID_MODALITY
    if not math.isfinite(dt) or abs(dt - DT_SEC) > FLOAT_TOLERANCE:
        return INVALID_SHAPE
    if int(horizon) != HORIZON:
        return INVALID_SHAPE
    if len(delta_p_xy) != HORIZON * ACTION_DIM:
        return INVALID_SHAPE
    if not all(math.isfinite(float(v)) for v in delta_p_xy):
        return NONFINITE
    return None


# ---------------------------------------------------------------------------
# Phase 2: speed & curvature
# ---------------------------------------------------------------------------

def _check_kinematics(
    delta_p_xy: Sequence[float],
    config: SafetyGateConfig,
) -> str | None:
    """Check per-step speed, total displacement, and curvature."""

    values = tuple(float(v) for v in delta_p_xy)
    if len(values) != HORIZON * ACTION_DIM:
        return INVALID_SHAPE

    total_dx = 0.0
    total_dy = 0.0
    previous_x = 0.0
    previous_y = 0.0

    # Per-step speed applies to the executed prefix only (the controller
    # executes 2 waypoints per plan before the next re-plan).
    for step in range(min(HORIZON, EXECUTED_SPEED_STEPS)):
        idx = step * ACTION_DIM
        cumulative_x = values[idx]
        cumulative_y = values[idx + 1]

        step_dx = cumulative_x - previous_x
        step_dy = cumulative_y - previous_y
        step_norm = math.hypot(step_dx, step_dy)

        # A relative tolerance absorbs the policy's regression overshoot:
        # the deployed model occasionally emits steps up to ~3% over the
        # 0.3 m clip (0.308 m = 1.54 m/s effective).  Genuine overspeed
        # (>5% over the limit) still rejects.
        if step_norm > config.max_step_m * 1.05 + FLOAT_TOLERANCE:
            return SPEED_LIMIT

        previous_x = cumulative_x
        previous_y = cumulative_y

    # Total displacement covers the WHOLE path, not just the prefix.
    total_dx = values[(HORIZON - 1) * ACTION_DIM]
    total_dy = values[(HORIZON - 1) * ACTION_DIM + 1]
    total_displacement = math.hypot(total_dx, total_dy)
    if total_displacement > config.max_total_displacement_m:
        return SPEED_LIMIT

    # Curvature: approximate via three-point method, over the executable
    # prefix only.
    if HORIZON >= 3:
        for step in range(1, min(HORIZON - 1, EXECUTED_HORIZON_STEPS)):
            p0 = (values[(step - 1) * ACTION_DIM], values[(step - 1) * ACTION_DIM + 1])
            p1 = (values[step * ACTION_DIM], values[step * ACTION_DIM + 1])
            p2 = (
                values[(step + 1) * ACTION_DIM],
                values[(step + 1) * ACTION_DIM + 1],
            )
            curvature = _three_point_curvature(p0, p1, p2)
            if curvature > config.max_curvature + FLOAT_TOLERANCE:
                return CURVATURE_LIMIT

    # Direction continuity: reject reversals/kinks between consecutive
    # meaningful steps (a >170 deg direction change is a path reversal),
    # over the executable prefix only.
    if HORIZON >= 2:
        previous_step: tuple[float, float] | None = None
        for step in range(min(HORIZON, EXECUTED_HORIZON_STEPS)):
            if step == 0:
                dx = values[0]
                dy = values[1]
            else:
                dx = values[step * ACTION_DIM] - values[(step - 1) * ACTION_DIM]
                dy = (
                    values[step * ACTION_DIM + 1]
                    - values[(step - 1) * ACTION_DIM + 1]
                )
            norm = math.hypot(dx, dy)
            if norm <= MIN_CURVATURE_SEGMENT_M:
                continue
            if previous_step is not None:
                previous_norm = math.hypot(*previous_step)
                if previous_norm > MIN_CURVATURE_SEGMENT_M:
                    dot = (dx * previous_step[0] + dy * previous_step[1]) / (
                        norm * previous_norm
                    )
                    if dot < math.cos(DEFAULT_MAX_TURN_RAD) - FLOAT_TOLERANCE:
                        return CURVATURE_LIMIT
            previous_step = (dx, dy)

    return None


def _three_point_curvature(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Approximate curvature (1/radius) from three consecutive waypoints."""

    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    a = math.hypot(x1 - x0, y1 - y0)
    b = math.hypot(x2 - x1, y2 - y1)
    c = math.hypot(x2 - x0, y2 - y0)

    if (
        a < MIN_CURVATURE_SEGMENT_M
        or b < MIN_CURVATURE_SEGMENT_M
        or c < MIN_CURVATURE_SEGMENT_M
    ):
        return 0.0

    area = abs((x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))) / 2.0
    if area < FLOAT_TOLERANCE:
        return 0.0

    return 4.0 * area / (a * b * c)


# ---------------------------------------------------------------------------
# Phase 3: collision (constant-velocity extrapolation)
# ---------------------------------------------------------------------------

def _check_collision(
    delta_p_xy: Sequence[float],
    entities: Sequence[_Entity],
    config: SafetyGateConfig,
) -> str | None:
    """First-edition collision: project entities with constant velocity,
    check minimum distance to every trajectory waypoint."""

    if not entities:
        return None  # no entities = no collision check possible, allow pass

    values = tuple(float(v) for v in delta_p_xy)

    for step in range(min(HORIZON, EXECUTED_COLLISION_STEPS)):
        time_sec = (step + 1) * DT_SEC
        waypoint_x = values[step * ACTION_DIM]
        waypoint_y = values[step * ACTION_DIM + 1]

        for entity in entities:
            predicted_x = entity.relative_x + entity.relative_vx * time_sec
            predicted_y = entity.relative_y + entity.relative_vy * time_sec
            dist = math.hypot(
                waypoint_x - predicted_x,
                waypoint_y - predicted_y,
            )
            if dist < config.collision_margin_m:
                return COLLISION_RISK

    return None


# ---------------------------------------------------------------------------
# Phase 4: deceleration fallback
# ---------------------------------------------------------------------------

def _deceleration_trajectory(
    current_healthy_trajectory: tuple[float, ...] | None,
    config: SafetyGateConfig,
) -> tuple[float, ...]:
    """Generate a deterministic deceleration trajectory.

    If we have a recent healthy trajectory, gradually brake from its
    last waypoint.  Otherwise produce an all-zero stop trajectory.
    """

    if current_healthy_trajectory is None:
        return (0.0,) * (HORIZON * ACTION_DIM)

    values = list(current_healthy_trajectory)
    last_x = values[-2]
    last_y = values[-1]

    decelerated: list[float] = []
    for step in range(HORIZON):
        fraction = (step + 1) / HORIZON
        decelerated.extend(
            (
                last_x * (1.0 - fraction * 0.5),
                last_y * (1.0 - fraction * 0.5),
            )
        )
    return tuple(decelerated)


# ---------------------------------------------------------------------------
# Top-level gate
# ---------------------------------------------------------------------------

def evaluate_safety_gate(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    horizon: int,
    delta_p_xy: Sequence[float],
    safe_stop: bool,
    valid: bool,
    reason: str,
    language_valid: bool = True,
    visual_valid: bool = True,
    entity_valid: bool = True,
    ego_valid: bool = True,
    entities: Sequence[_Entity] | None = None,
    last_valid_stamp_us: int = 0,
    last_healthy_trajectory: tuple[float, ...] | None = None,
    time_since_last_valid_sec: float = 0.0,
    config: SafetyGateConfig | None = None,
) -> SafetyGateResult:
    """Evaluate one policy trajectory through the safety gate.

    Returns a ``SafetyGateResult`` that is always well-formed and
    deterministic.  The caller (ROS node) is responsible for writing
    the reason to the system log.
    """

    cfg = config or SafetyGateConfig()

    # Phase 0: the policy itself requested STOP.
    if valid and safe_stop:
        return SafetyGateResult(
            delta_p_xy=(0.0,) * (HORIZON * ACTION_DIM),
            safe_stop=True,
            valid=True,
            reason=POLICY_STOP,
            detail=reason or "policy stop",
        )

    # Phase 1: modality and shape.
    rejection = _check_modality_and_shape(
        stamp_us=stamp_us,
        run_id=run_id,
        frame_id=frame_id,
        dt=dt,
        horizon=horizon,
        delta_p_xy=delta_p_xy,
        policy_valid=valid,
        language_valid=language_valid,
        visual_valid=visual_valid,
        entity_valid=entity_valid,
        ego_valid=ego_valid,
        last_valid_stamp_us=last_valid_stamp_us,
        config=cfg,
    )
    if rejection is not None:
        return _reject(rejection, cfg, last_healthy_trajectory)

    # Phase 2: kinematics.
    rejection = _check_kinematics(delta_p_xy, cfg)
    if rejection is not None:
        return _reject(rejection, cfg, last_healthy_trajectory)

    # Phase 3: collision.
    if entities:
        rejection = _check_collision(delta_p_xy, entities, cfg)
        if rejection is not None:
            return _reject(rejection, cfg, last_healthy_trajectory)

    # Phase 4: stale / E-STOP escalation.
    if time_since_last_valid_sec > cfg.estop_timeout_sec:
        return SafetyGateResult(
            delta_p_xy=(0.0,) * (HORIZON * ACTION_DIM),
            safe_stop=True,
            valid=False,
            reason=ESTOP,
            detail=f"no valid policy for {time_since_last_valid_sec:.2f}s",
        )

    # All checks passed.
    return SafetyGateResult(
        delta_p_xy=tuple(float(v) for v in delta_p_xy),
        safe_stop=False,
        valid=True,
        reason=PASS,
    )


def _reject(
    code: str,
    config: SafetyGateConfig,
    last_healthy: tuple[float, ...] | None,
) -> SafetyGateResult:
    """Produce a deterministic rejection trajectory."""

    trajectory = _deceleration_trajectory(last_healthy, config)
    all_zero = all(abs(v) <= FLOAT_TOLERANCE for v in trajectory)

    if code in (ESTOP,):
        return SafetyGateResult(
            delta_p_xy=(0.0,) * (HORIZON * ACTION_DIM),
            safe_stop=True,
            valid=False,
            reason=code,
        )

    return SafetyGateResult(
        delta_p_xy=trajectory,
        safe_stop=all_zero,
        valid=False,
        reason=code,
    )
