"""ROS trajectory control bridge node.

Consumes ``/vla/selected_trajectory`` (from the safety gate) and publishes
``/decision/output`` with ``desired_x`` / ``desired_y``.  Never publishes
thruster values.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import DecisionOutput, SelectedTrajectory

from .trajectory_controller import trajectory_to_command


class TrajectoryControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_controller")

        self._sub = self.create_subscription(
            SelectedTrajectory,
            "/vla/selected_trajectory",
            self._on_trajectory,
            10,
        )
        self._pub = self.create_publisher(
            DecisionOutput, "/decision/output", 10
        )

        self._last_executed_stamp_us = 0
        self._last_trajectory_arrival = time.monotonic()
        self._has_valid_trajectory = False
        self._last_run_id = ""

    def _on_trajectory(self, message: SelectedTrajectory) -> None:
        now = time.monotonic()
        # The UE5 frame counter restarts on every simulation launch; reset
        # the executed-stamp baseline when the run changes.
        if str(message.run_id) != self._last_run_id:
            self._last_run_id = str(message.run_id)
            self._last_executed_stamp_us = 0
            self._has_valid_trajectory = False
            self._last_trajectory_arrival = now

        # First valid trajectory must not be stale just because the node
        # started seconds ago; staleness applies after a stream is lost.
        time_since = (
            now - self._last_trajectory_arrival if self._has_valid_trajectory else 0.0
        )

        command = trajectory_to_command(
            delta_p_xy=message.delta_p_xy,
            safe_stop=bool(message.safe_stop),
            valid=bool(message.valid),
            reason=str(message.reason),
            stamp_us=int(message.stamp_us),
            last_executed_stamp_us=self._last_executed_stamp_us,
            time_since_last_valid_sec=time_since,
        )

        if command.valid:
            self._last_executed_stamp_us = int(message.stamp_us)
            self._last_trajectory_arrival = now
            self._has_valid_trajectory = True

        output = DecisionOutput()
        output.stamp_us = int(message.stamp_us)
        output.desired_x = float(command.desired_x)
        output.desired_y = float(command.desired_y)
        output.valid = command.valid
        output.run_id = str(message.run_id)
        output.scene_seed = int(message.scene_seed)
        output.source_frame_index = int(message.frame_index)
        output.source_model_version = str(message.model_version)

        self._pub.publish(output)

        if not command.valid:
            self.get_logger().info(
                f"control bridge: {command.detail}"
            )


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
