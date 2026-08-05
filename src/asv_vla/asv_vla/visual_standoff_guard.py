"""Image-only standoff guidance from the normalized ``TaskFeatures`` tensor.

This module is intentionally ROS-free.  It consumes only the entity IDs,
mask, normalized geometry, and optional tracker velocity already present in a
``TaskFeatures``-shaped object.  UE truth entities are neither imported nor
accepted as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import numpy as np


FEATURE_DIM = 16
POSITION_SCALE_M = 20.0
VELOCITY_SCALE_MPS = 5.0
DEFAULT_STANDOFF_M = 3.0
DEFAULT_GUARD_MAX_STEP_M = 0.15
# Hold tolerance.  The verified closed-loop envelope (2.4-3.3 m for 3 m)
# is ~+-0.45 m around the standoff; a too-narrow 0.15 m deadband lets pipeline
# latency push the ASV past the reliable image-detection range.
DEFAULT_DEADBAND_M = 0.45
DEFAULT_PREDICTION_HORIZON_SEC = 0.2
MAX_TARGET_DISTANCE_M = 20.0
MAX_TARGET_SPEED_MPS = 5.0
TARGET_IDS = ("target_red", "target_blue", "target_left", "target_right")

# Backstop semantics: the learned policy drives the executed point; the
# deterministic radial step only replaces a policy action that points away
# from the target.  ``BACKSTOP_DOT_THRESHOLD`` = 0.0 means an angle greater
# than 90 degrees is corrected.
BACKSTOP_DOT_THRESHOLD = 0.0
# Retained as a keyword-compatible parameter; action magnitude alone no
# longer activates the backstop.
BACKSTOP_ZERO_STEP_M = 1.0e-3

# Guard outcomes returned by ``apply_standoff_guard``.
GUARD_PASS_THROUGH = "pass_through"  # non-FOLLOW task, action unchanged
GUARD_FAIL_CLOSED = "fail_closed"  # FOLLOW, target missing/OOD -> safe stop
GUARD_BACKSTOP = "backstop"  # FOLLOW, policy step unsafe -> radial replacement
GUARD_POLICY_DRIVEN = "policy_driven"  # FOLLOW, policy step kept
GUARD_HOLD = "deadband_hold"  # FOLLOW, standoff deadband -> zero hold

_DISTANCE_RE = re.compile(
    r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(?:m|米|meters?|metres?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TargetObservation:
    """Finite image/tracker geometry in the vehicle ``base_link`` frame."""

    entity_id: str
    relative_x: float
    relative_y: float
    relative_velocity_x: float = 0.0
    relative_velocity_y: float = 0.0
    velocity_valid: bool = True

    @property
    def distance_m(self) -> float:
        return math.hypot(self.relative_x, self.relative_y)


def _instruction_text(task_features: Any, instruction: str | None) -> str:
    if instruction is not None:
        return str(instruction).strip()
    return str(getattr(task_features, "instruction", "")).strip()


def is_follow_instruction(instruction: str) -> bool:
    """Return whether a task asks for FOLLOW-like standoff behavior."""

    text = str(instruction).strip().casefold()
    if not text or any(token in text for token in ("stop", "停止", "停船")):
        return False
    return any(
        token in text
        for token in ("follow", "track", "跟随", "跟踪", "追踪", "保持", "靠近")
    )


def target_id_from_instruction(instruction: str) -> str | None:
    """Map color/bearing words to the canonical task-tensor entity ID."""

    text = str(instruction).strip().casefold()
    if not text:
        return None
    for target_id in TARGET_IDS:
        if target_id in text:
            return target_id
    if any(token in text for token in ("red", "红", "赤")):
        return "target_red"
    if any(token in text for token in ("blue", "蓝")):
        return "target_blue"
    if any(token in text for token in ("left", "左")):
        return "target_left"
    if any(token in text for token in ("right", "右")):
        return "target_right"
    return None


def desired_standoff_from_instruction(
    instruction: str,
    *,
    default_m: float = DEFAULT_STANDOFF_M,
) -> float:
    """Parse ``3m``/``3米``/``10 meters`` with a safe 3 m default."""

    default = float(default_m)
    if not math.isfinite(default) or default <= 0.0:
        raise ValueError("default standoff must be finite and positive")
    match = _DISTANCE_RE.search(str(instruction))
    if match is None:
        return default
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0.0 or value > MAX_TARGET_DISTANCE_M:
        return default
    return value


def _feature_row(task_features: Any, index: int) -> np.ndarray | None:
    try:
        feature_dim = int(getattr(task_features, "feature_dim", FEATURE_DIM))
        max_entities = int(
            getattr(task_features, "max_entities", len(getattr(task_features, "entity_ids", [])))
        )
        values = np.asarray(task_features.features, dtype=np.float64).reshape(
            max_entities, feature_dim
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if feature_dim < 5 or index < 0 or index >= max_entities:
        return None
    return values[index]


def extract_target_observation(
    task_features: Any,
    instruction: str | None = None,
) -> TargetObservation | None:
    """Extract the instruction-selected, visible target from image features."""

    text = _instruction_text(task_features, instruction)
    target_id = target_id_from_instruction(text)
    if target_id is None or not bool(getattr(task_features, "valid", False)):
        return None
    try:
        entity_ids = [str(value).strip() for value in task_features.entity_ids]
        masks = [bool(value) for value in task_features.mask]
        index = entity_ids.index(target_id)
    except (AttributeError, ValueError, TypeError):
        return None
    if index >= len(masks) or not masks[index]:
        return None
    row = _feature_row(task_features, index)
    if row is None or not np.all(np.isfinite(row[:5])):
        return None
    relative_x = float(row[0] * POSITION_SCALE_M)
    relative_y = float(row[1] * POSITION_SCALE_M)
    relative_velocity_x = float(row[3] * VELOCITY_SCALE_MPS)
    relative_velocity_y = float(row[4] * VELOCITY_SCALE_MPS)
    distance = math.hypot(relative_x, relative_y)
    if not (
        math.isfinite(distance)
        and 0.05 <= relative_x <= MAX_TARGET_DISTANCE_M
        and abs(relative_y) <= MAX_TARGET_DISTANCE_M
        and distance <= MAX_TARGET_DISTANCE_M
        and abs(relative_velocity_x) <= MAX_TARGET_SPEED_MPS
        and abs(relative_velocity_y) <= MAX_TARGET_SPEED_MPS
    ):
        return None
    return TargetObservation(
        entity_id=target_id,
        relative_x=relative_x,
        relative_y=relative_y,
        relative_velocity_x=relative_velocity_x,
        relative_velocity_y=relative_velocity_y,
        velocity_valid=True,
    )


def compute_standoff_step(
    observation: TargetObservation,
    desired_standoff_m: float = DEFAULT_STANDOFF_M,
    *,
    guard_max_step_m: float = DEFAULT_GUARD_MAX_STEP_M,
    deadband_m: float = DEFAULT_DEADBAND_M,
    prediction_horizon_sec: float = DEFAULT_PREDICTION_HORIZON_SEC,
) -> tuple[float, float] | None:
    """Return one bounded radial step toward/away from the predicted target."""

    desired = float(desired_standoff_m)
    max_step = float(guard_max_step_m)
    deadband = float(deadband_m)
    horizon = float(prediction_horizon_sec)
    if not (
        math.isfinite(desired)
        and desired > 0.0
        and math.isfinite(max_step)
        and max_step >= 0.0
        and math.isfinite(deadband)
        and deadband >= 0.0
        and math.isfinite(horizon)
        and horizon >= 0.0
    ):
        return None
    values = (
        float(observation.relative_x),
        float(observation.relative_y),
        float(observation.relative_velocity_x),
        float(observation.relative_velocity_y),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    predicted_x = values[0]
    predicted_y = values[1]
    if observation.velocity_valid:
        predicted_x += values[2] * horizon
        predicted_y += values[3] * horizon
    distance = math.hypot(predicted_x, predicted_y)
    if not (
        0.05 <= predicted_x <= MAX_TARGET_DISTANCE_M
        and abs(predicted_y) <= MAX_TARGET_DISTANCE_M
        and 0.05 <= distance <= MAX_TARGET_DISTANCE_M
    ):
        return None
    error = distance - desired
    if abs(error) <= deadband:
        return 0.0, 0.0
    step_norm = min(abs(error), max_step)
    sign = 1.0 if error > 0.0 else -1.0
    return (
        float(sign * step_norm * predicted_x / distance),
        float(sign * step_norm * predicted_y / distance),
    )


def apply_standoff_guard(
    displacement: Sequence[float] | np.ndarray,
    task_features: Any,
    *,
    dot_threshold: float = BACKSTOP_DOT_THRESHOLD,
    zero_step_m: float = BACKSTOP_ZERO_STEP_M,
    deadband_m: float = DEFAULT_DEADBAND_M,
) -> tuple[tuple[float, ...] | None, str]:
    """Apply the learned-policy-first standoff backstop.

    Returns ``(desired_displacement, reason)`` where ``reason`` is one of
    ``GUARD_PASS_THROUGH`` / ``GUARD_FAIL_CLOSED`` / ``GUARD_BACKSTOP`` /
    ``GUARD_POLICY_DRIVEN`` / ``GUARD_HOLD``.

    - Non-FOLLOW tasks retain the finite policy displacement unchanged.
    - A FOLLOW task with a missing/OOD target returns ``(None, GUARD_FAIL_CLOSED)``
      so the caller can publish a safe stop.
    - A FOLLOW task with a visible target keeps the policy's direct action
      (the learned policy drives the executed command) unless the action
      points away from the target (``dot(action, target_dir) < dot_threshold``,
      i.e. > 90 deg). Only then is it replaced by the deterministic radial
      standoff action (``GUARD_BACKSTOP``).
    """

    # ``zero_step_m`` is retained for callers of the previous interface.  A
    # small but directionally correct learned action must now pass through.
    _ = zero_step_m
    try:
        values = np.asarray(displacement, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, GUARD_FAIL_CLOSED
    if values.size != 2 or not np.all(np.isfinite(values)):
        return None, GUARD_FAIL_CLOSED
    instruction = _instruction_text(task_features, None)
    if not is_follow_instruction(instruction):
        return tuple(float(value) for value in values), GUARD_PASS_THROUGH
    observation = extract_target_observation(task_features, instruction)
    if observation is None:
        return None, GUARD_FAIL_CLOSED
    desired = desired_standoff_from_instruction(instruction)
    try:
        base_deadband = float(deadband_m)
    except (TypeError, ValueError):
        return None, GUARD_FAIL_CLOSED
    if not math.isfinite(base_deadband) or base_deadband < 0.0:
        return None, GUARD_FAIL_CLOSED
    error = observation.distance_m - desired
    effective_deadband = base_deadband
    step = compute_standoff_step(
        observation,
        desired,
        deadband_m=effective_deadband,
    )
    if step is None:
        return None, GUARD_FAIL_CLOSED

    # The hold rule is an intentional safety behavior: within the standoff
    # deadband, no approach or retreat action is allowed through.
    if abs(error) <= effective_deadband:
        values = values.copy()
        values[:] = (0.0, 0.0)
        return tuple(float(value) for value in values), GUARD_HOLD

    first_step = values
    target_dir = (observation.relative_x, observation.relative_y)
    target_norm = math.hypot(*target_dir)
    if target_norm <= 0.0:
        return None, GUARD_FAIL_CLOSED
    target_dir = (target_dir[0] / target_norm, target_dir[1] / target_norm)
    dot_value = first_step[0] * target_dir[0] + first_step[1] * target_dir[1]
    if dot_value < dot_threshold:
        values = values.copy()
        values[:] = step
        return tuple(float(value) for value in values), GUARD_BACKSTOP
    return tuple(float(value) for value in values), GUARD_POLICY_DRIVEN
