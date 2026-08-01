"""ROS safety gate node.

The safety gate is the *only* publisher of ``/vla/selected_trajectory``.
It consumes ``/vla/policy_trajectory`` and publishes a validated,
deterministic trajectory after hard-constraint checks.
"""

from __future__ import annotations

import time
from typing import Any

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import SelectedTrajectory

from .safety_gate import (
    ESTOP,
    PASS,
    POLICY_STOP,
    SafetyGateConfig,
    SafetyGateResult,
    _Entity,
    evaluate_safety_gate,
)
from .trajectory_contract import (
    ACTION_DIM,
    DT_SEC,
    FRAME_ID,
    HORIZON,
    SAFE_STOP_MODEL_VERSION,
)

SAFETY_GATE_MODEL_VERSION = "safety_gate_v1"


class SafetyGateNode(Node):
    """ROS node wrapping the deterministic trajectory safety gate."""

    def __init__(self, *, config: SafetyGateConfig | None = None) -> None:
        super().__init__("safety_gate")
        self._config = config or SafetyGateConfig()

        # Declare tunable parameters.
        self.declare_parameter("max_step_m", self._config.max_step_m)
        self.declare_parameter(
            "max_total_displacement_m", self._config.max_total_displacement_m
        )
        self.declare_parameter("max_curvature", self._config.max_curvature)
        self.declare_parameter("stale_timeout_sec", self._config.stale_timeout_sec)
        self.declare_parameter("estop_timeout_sec", self._config.estop_timeout_sec)
        self.declare_parameter("collision_margin_m", self._config.collision_margin_m)

        # Subscribers.
        self._policy_sub = self.create_subscription(
            SelectedTrajectory,
            "/vla/policy_trajectory",
            self._on_policy,
            10,
        )
        # Entity subscription for collision checking.
        from asv_jetson_interfaces.msg import UEEntityArray

        self._entity_sub = self.create_subscription(
            UEEntityArray,
            "/ue/entities",
            self._on_entities,
            10,
        )

        # Sole publisher of selected trajectory.
        self._selected_pub = self.create_publisher(
            SelectedTrajectory, "/vla/selected_trajectory", 10
        )

        # State.
        self._last_valid_policy_stamp_us = 0
        self._last_policy_arrival = time.monotonic()
        self._last_healthy_trajectory: tuple[float, ...] | None = None
        self._latest_entities: list[_Entity] = []
        self._rejection_log: dict[str, int] = {}
        self._has_valid_policy = False
        self._last_run_id = ""

    def _read_config(self) -> SafetyGateConfig:
        return SafetyGateConfig(
            max_step_m=float(
                self.get_parameter("max_step_m").get_parameter_value().double_value
            ),
            max_total_displacement_m=float(
                self.get_parameter("max_total_displacement_m")
                .get_parameter_value()
                .double_value
            ),
            max_curvature=float(
                self.get_parameter("max_curvature")
                .get_parameter_value()
                .double_value
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
        if not message.valid:
            return
        entities: list[_Entity] = []
        for entity in message.entities:
            if not (entity.valid and entity.visible):
                continue
            entities.append(
                _Entity(
                    entity_id=str(entity.entity_id),
                    relative_x=float(entity.relative_x),
                    relative_y=float(entity.relative_y),
                    relative_vx=float(entity.relative_velocity_x),
                    relative_vy=float(entity.relative_velocity_y),
                )
            )
        self._latest_entities = entities

    def _on_policy(self, message: SelectedTrajectory) -> None:
        now = time.monotonic()
        # The UE5 frame counter restarts on every simulation launch; reset
        # the stamp baseline when the run changes so the new run's stamps
        # are not judged stale against the previous run.
        if str(message.run_id) != self._last_run_id:
            self._last_run_id = str(message.run_id)
            self._last_valid_policy_stamp_us = 0
            self._has_valid_policy = False
            self._last_policy_arrival = now

        # First valid input must not E-STOP just because the node started
        # seconds ago; E-STOP applies only after a valid stream goes stale.
        time_since_last = (
            now - self._last_policy_arrival if self._has_valid_policy else 0.0
        )

        config = self._read_config()

        result = evaluate_safety_gate(
            stamp_us=int(message.stamp_us),
            run_id=str(message.run_id),
            frame_id=str(message.frame_id),
            model_version=str(message.model_version),
            dt=float(message.dt),
            horizon=int(message.horizon),
            delta_p_xy=tuple(float(v) for v in message.delta_p_xy),
            safe_stop=bool(message.safe_stop),
            valid=bool(message.valid),
            reason=str(message.reason),
            entities=self._latest_entities or None,
            last_valid_stamp_us=self._last_valid_policy_stamp_us,
            last_healthy_trajectory=self._last_healthy_trajectory,
            time_since_last_valid_sec=time_since_last,
            config=config,
        )

        # Update state.
        if result.valid:
            self._last_valid_policy_stamp_us = int(message.stamp_us)
            self._last_policy_arrival = now
            self._has_valid_policy = True
        if result.valid and result.reason == PASS:
            self._last_healthy_trajectory = result.delta_p_xy

        # Track rejection statistics.
        if result.reason != PASS:
            self._rejection_log[result.reason] = (
                self._rejection_log.get(result.reason, 0) + 1
            )

        # Publish.
        output = SelectedTrajectory()
        output.stamp_us = int(message.stamp_us)
        output.run_id = str(message.run_id)
        output.scene_seed = int(message.scene_seed)
        output.frame_index = int(message.frame_index)
        output.frame_id = FRAME_ID
        output.model_version = SAFETY_GATE_MODEL_VERSION
        output.dt = DT_SEC
        output.horizon = HORIZON
        output.delta_p_xy = list(result.delta_p_xy)
        output.safe_stop = result.safe_stop
        output.valid = result.valid
        output.reason = f"{result.reason}:{result.detail}" if result.detail else result.reason

        self._selected_pub.publish(output)

        # Log rejections at INFO.
        if result.reason not in (PASS, POLICY_STOP):
            self.get_logger().info(
                f"safety gate rejected: reason={result.reason}"
            )
        if result.reason == ESTOP:
            self.get_logger().error(
                f"E-STOP engaged: {result.detail}"
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
