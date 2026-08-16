"""Deterministic safety gate for one online body-frame displacement.

The gate is the only component between the CUDA policy and the kinematic
controller.  It validates the current ``(desired_x, desired_y)`` command,
checks its one-step collision envelope, and fails closed on every rejection.
The model's offline [20, 2] output is not accepted at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

try:
    import rclpy
    from rclpy.node import Node
    from asv_jetson_interfaces.msg import DesiredDisplacement, EntityArray
except ModuleNotFoundError:  # Keep pure gate tests runnable outside ROS.
    rclpy = None
    Node = object
    DesiredDisplacement = EntityArray = object

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


def limit_displacement_rate(
    result: SafetyGateResult,
    previous: tuple[float, float] | None,
    *,
    max_delta_m: float = DEFAULT_MAX_STEP_M,
) -> SafetyGateResult:
    """Limit the change between consecutive executable displacements."""

    if not result.valid or previous is None:
        return result
    if not math.isfinite(max_delta_m) or max_delta_m <= 0.0:
        return _reject(CONTROL_UNREACHABLE, "invalid rate limit")
    previous_x, previous_y = map(float, previous)
    if not (math.isfinite(previous_x) and math.isfinite(previous_y)):
        return _reject(CONTROL_UNREACHABLE, "invalid previous displacement")
    delta_x = float(result.desired_x) - previous_x
    delta_y = float(result.desired_y) - previous_y
    delta_norm = math.hypot(delta_x, delta_y)
    if delta_norm <= max_delta_m + FLOAT_TOLERANCE:
        return result
    scale = max_delta_m / delta_norm
    return SafetyGateResult(
        desired_x=previous_x + delta_x * scale,
        desired_y=previous_y + delta_y * scale,
        safe_stop=False,
        valid=True,
        reason=result.reason,
        detail="RATE_LIMITED",
    )


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


SAFETY_GATE_MODEL_VERSION = "safety_gate"


class SafetyGateNode(Node):
    """Publish only validated one-frame body-frame displacements."""

    def __init__(self, *, config: SafetyGateConfig | None = None) -> None:
        super().__init__("safety_gate")
        self._config = config or SafetyGateConfig()
        self.declare_parameter("max_step_m", self._config.max_step_m)
        self.declare_parameter("stale_timeout_sec", self._config.stale_timeout_sec)
        self.declare_parameter("estop_timeout_sec", self._config.estop_timeout_sec)
        self.declare_parameter("collision_margin_m", self._config.collision_margin_m)
        self.entities_topic = str(
            self.declare_parameter("entities_topic", "/vla/tracked_entities")
            .get_parameter_value()
            .string_value
        )
        self.allow_truth_entities = bool(
            self.declare_parameter("allow_truth_entities", False)
            .get_parameter_value()
            .bool_value
        )

        self._policy_sub = self.create_subscription(
            DesiredDisplacement,
            "/vla/policy_displacement",
            self._on_policy,
            10,
        )
        self._entity_sub = self.create_subscription(
            EntityArray, self.entities_topic, self._on_entities, 10
        )
        self._selected_pub = self.create_publisher(
            DesiredDisplacement, "/control/desired_displacement", 10
        )

        now = time.monotonic()
        self._last_policy_arrival = now
        self._last_valid_policy_arrival = now
        self._last_valid_policy_stamp_us = 0
        self._has_valid_policy = False
        self._last_run_id = ""
        self._latest_entities: list[_Entity] = []
        self._latest_entities_valid = False
        self._last_policy_message: DesiredDisplacement | None = None
        self._previous_desired: tuple[float, float] | None = None
        self._timeout_reason = ""
        self._rejection_log: dict[str, int] = {}
        self.create_timer(DT_SEC, self._on_timeout)

    def _read_config(self) -> SafetyGateConfig:
        return SafetyGateConfig(
            max_step_m=float(
                self.get_parameter("max_step_m").get_parameter_value().double_value
            ),
            stale_timeout_sec=float(
                self.get_parameter("stale_timeout_sec")
                .get_parameter_value()
                .double_value
            ),
            estop_timeout_sec=float(
                self.get_parameter("estop_timeout_sec")
                .get_parameter_value()
                .double_value
            ),
            collision_margin_m=float(
                self.get_parameter("collision_margin_m")
                .get_parameter_value()
                .double_value
            ),
        )

    def _on_entities(self, message: Any) -> None:
        if (
            not self.allow_truth_entities
            and str(message.source) not in {"image_perception", "temporal_tracker"}
        ):
            return
        if not bool(message.valid):
            self._latest_entities = []
            self._latest_entities_valid = False
            return

        entities: list[_Entity] = []
        for entity in message.entities:
            if not (bool(entity.valid) and bool(entity.visible)):
                continue
            values = (
                float(entity.relative_x),
                float(entity.relative_y),
                float(entity.relative_velocity_x),
                float(entity.relative_velocity_y),
            )
            if not all(math.isfinite(value) for value in values):
                self._latest_entities = []
                self._latest_entities_valid = False
                return
            entities.append(
                _Entity(
                    entity_id=str(entity.entity_id),
                    relative_x=values[0],
                    relative_y=values[1],
                    relative_vx=values[2],
                    relative_vy=values[3],
                )
            )
        self._latest_entities = entities
        self._latest_entities_valid = True

    def _publish_result(
        self, source: DesiredDisplacement, result: SafetyGateResult
    ) -> None:
        output = DesiredDisplacement()
        output.stamp_us = int(source.stamp_us)
        output.run_id = str(source.run_id)
        output.scene_seed = int(source.scene_seed)
        output.frame_index = int(source.frame_index)
        output.frame_id = FRAME_ID
        output.source = SAFETY_GATE_MODEL_VERSION
        output.step_dt = DT_SEC
        output.desired_x = float(result.desired_x)
        output.desired_y = float(result.desired_y)
        output.safe_stop = bool(result.safe_stop)
        output.valid = bool(result.valid)
        output.reason = (
            f"{result.reason}:{result.detail}" if result.detail else result.reason
        )
        self._selected_pub.publish(output)

    def _on_policy(self, message: DesiredDisplacement) -> None:
        now = time.monotonic()
        if str(message.run_id) != self._last_run_id:
            self._last_run_id = str(message.run_id)
            self._last_valid_policy_stamp_us = 0
            self._has_valid_policy = False
            self._previous_desired = None
            self._last_valid_policy_arrival = now
            self._timeout_reason = ""

        time_since_last_valid = (
            now - self._last_valid_policy_arrival if self._has_valid_policy else 0.0
        )
        self._last_policy_arrival = now
        config = self._read_config()
        result = evaluate_safety_gate(
            stamp_us=int(message.stamp_us),
            run_id=str(message.run_id),
            frame_id=str(message.frame_id),
            model_version=str(message.source),
            dt=float(message.step_dt),
            desired_x=float(message.desired_x),
            desired_y=float(message.desired_y),
            safe_stop=bool(message.safe_stop),
            valid=bool(message.valid),
            reason=str(message.reason),
            entity_valid=self._latest_entities_valid,
            entities=self._latest_entities,
            last_valid_stamp_us=self._last_valid_policy_stamp_us,
            time_since_last_valid_sec=time_since_last_valid,
            config=config,
        )

        result = limit_displacement_rate(
            result,
            self._previous_desired,
            max_delta_m=config.max_step_m,
        )
        if result.valid:
            self._last_valid_policy_stamp_us = int(message.stamp_us)
            self._last_valid_policy_arrival = now
            self._has_valid_policy = True
            self._timeout_reason = ""
            self._previous_desired = (result.desired_x, result.desired_y)
        else:
            self._previous_desired = None

        self._last_policy_message = message
        if result.reason != PASS:
            self._rejection_log[result.reason] = (
                self._rejection_log.get(result.reason, 0) + 1
            )
            self.get_logger().info(f"safety gate rejected: reason={result.reason}")
        if result.reason == ESTOP:
            self.get_logger().error(f"E-STOP engaged: {result.detail}")
        self._publish_result(message, result)

    def _on_timeout(self) -> None:
        source = self._last_policy_message
        if source is None or not self._has_valid_policy:
            return
        age = time.monotonic() - self._last_policy_arrival
        config = self._read_config()
        if age <= config.stale_timeout_sec:
            return
        code = ESTOP if age > config.estop_timeout_sec else STALE_INPUT
        if code == self._timeout_reason:
            return
        self._timeout_reason = code
        self._previous_desired = None
        self._publish_result(
            source,
            SafetyGateResult(
                desired_x=0.0,
                desired_y=0.0,
                safe_stop=True,
                valid=False,
                reason=code,
                detail=f"policy stream silent for {age:.2f}s",
            ),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
