"""ROS wrapper for the single-point online safety gate."""

from __future__ import annotations

import math
import time
from typing import Any

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import DecisionPoint, UEEntityArray

from .safety_gate import (
    ESTOP,
    PASS,
    POLICY_STOP,
    STALE_INPUT,
    SafetyGateConfig,
    SafetyGateResult,
    _Entity,
    evaluate_safety_gate,
)
from .trajectory_contract import DT_SEC, FRAME_ID

SAFETY_GATE_MODEL_VERSION = "safety_gate_v1"


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
            DecisionPoint, "/vla/policy_point", self._on_policy, 10
        )
        self._entity_sub = self.create_subscription(
            UEEntityArray, self.entities_topic, self._on_entities, 10
        )
        self._selected_pub = self.create_publisher(
            DecisionPoint, "/vla/selected_point", 10
        )

        now = time.monotonic()
        self._last_policy_arrival = now
        self._last_valid_policy_arrival = now
        self._last_valid_policy_stamp_us = 0
        self._has_valid_policy = False
        self._last_run_id = ""
        self._latest_entities: list[_Entity] = []
        self._latest_entities_valid = False
        self._last_policy_message: DecisionPoint | None = None
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

    def _publish_result(self, source: DecisionPoint, result: SafetyGateResult) -> None:
        output = DecisionPoint()
        output.stamp_us = int(source.stamp_us)
        output.run_id = str(source.run_id)
        output.scene_seed = int(source.scene_seed)
        output.frame_index = int(source.frame_index)
        output.frame_id = FRAME_ID
        output.model_version = SAFETY_GATE_MODEL_VERSION
        output.dt = DT_SEC
        output.desired_x = float(result.desired_x)
        output.desired_y = float(result.desired_y)
        output.safe_stop = bool(result.safe_stop)
        output.valid = bool(result.valid)
        output.reason = (
            f"{result.reason}:{result.detail}" if result.detail else result.reason
        )
        self._selected_pub.publish(output)

    def _on_policy(self, message: DecisionPoint) -> None:
        now = time.monotonic()
        if str(message.run_id) != self._last_run_id:
            self._last_run_id = str(message.run_id)
            self._last_valid_policy_stamp_us = 0
            self._has_valid_policy = False
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
            model_version=str(message.model_version),
            dt=float(message.dt),
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

        if result.valid:
            self._last_valid_policy_stamp_us = int(message.stamp_us)
            self._last_valid_policy_arrival = now
            self._has_valid_policy = True
            self._timeout_reason = ""

        self._last_policy_message = message
        if result.reason != PASS:
            self._rejection_log[result.reason] = (
                self._rejection_log.get(result.reason, 0) + 1
            )
            self.get_logger().info(
                f"safety gate rejected: reason={result.reason}"
            )
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
