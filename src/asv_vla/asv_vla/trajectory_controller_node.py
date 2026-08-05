"""ROS bridge from gated body-frame displacement to DecisionOutput."""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import DecisionOutput, DecisionPoint

from .trajectory_controller import (
    STALE_THRESHOLD_SEC,
    point_to_command,
)
from .trajectory_contract import DT_SEC


class TrajectoryControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_controller")
        self._sub = self.create_subscription(
            DecisionPoint,
            "/vla/selected_point",
            self._on_point,
            10,
        )
        self._pub = self.create_publisher(DecisionOutput, "/decision/output", 10)
        self._last_executed_stamp_us = 0
        self._last_point_arrival = time.monotonic()
        self._last_run_id = ""
        self._previous_desired: tuple[float, float] | None = None
        self._last_point: DecisionPoint | None = None
        self._timeout_published = False
        self.create_timer(DT_SEC, self._on_timeout)

    def _publish(self, message: DecisionPoint, *, desired_x: float, desired_y: float, valid: bool) -> None:
        output = DecisionOutput()
        output.stamp_us = int(message.stamp_us)
        output.desired_x = float(desired_x)
        output.desired_y = float(desired_y)
        output.valid = bool(valid)
        output.run_id = str(message.run_id)
        output.scene_seed = int(message.scene_seed)
        output.source_frame_index = int(message.frame_index)
        output.source_model_version = str(message.model_version)
        self._pub.publish(output)

    def _on_point(self, message: DecisionPoint) -> None:
        now = time.monotonic()
        if str(message.run_id) != self._last_run_id:
            self._last_run_id = str(message.run_id)
            self._last_executed_stamp_us = 0
            self._previous_desired = None

        time_since_last = now - self._last_point_arrival
        command = point_to_command(
            desired_x=float(message.desired_x),
            desired_y=float(message.desired_y),
            safe_stop=bool(message.safe_stop),
            valid=bool(message.valid),
            reason=str(message.reason),
            stamp_us=int(message.stamp_us),
            dt=float(message.dt),
            last_executed_stamp_us=self._last_executed_stamp_us,
            time_since_last_valid_sec=time_since_last,
            previous_desired=self._previous_desired,
        )
        self._last_point_arrival = now
        self._last_point = message
        self._timeout_published = False

        if command.valid:
            self._last_executed_stamp_us = int(message.stamp_us)
            self._previous_desired = (command.desired_x, command.desired_y)
        else:
            self._previous_desired = None

        self._publish(
            message,
            desired_x=command.desired_x,
            desired_y=command.desired_y,
            valid=command.valid,
        )
        if not command.valid:
            self.get_logger().info(f"control bridge: {command.detail}")

    def _on_timeout(self) -> None:
        if self._last_point is None or self._timeout_published:
            return
        if time.monotonic() - self._last_point_arrival <= STALE_THRESHOLD_SEC:
            return
        self._timeout_published = True
        self._previous_desired = None
        self._publish(
            self._last_point,
            desired_x=0.0,
            desired_y=0.0,
            valid=False,
        )
        self.get_logger().warning("control bridge: STALE_DISPLACEMENT")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
