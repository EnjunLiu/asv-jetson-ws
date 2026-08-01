"""Adapter: /decision/output (DecisionOutput) -> /ue/kinematic_setpoint.

The trajectory controller publishes ``desired_x / desired_y`` on
``/decision/output`` (the existing control boundary).  The UE5 kinematic
execution path consumes single-step setpoints on ``/ue/kinematic_setpoint``.
This node bridges the two without touching the legacy control chain.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import DecisionOutput, UEKinematicSetpoint

from .trajectory_contract import DT_SEC


class DecisionSetpointAdapter(Node):
    def __init__(self) -> None:
        super().__init__("decision_setpoint_adapter")
        self._sub = self.create_subscription(
            DecisionOutput, "/decision/output", self._on_decision, 10
        )
        self._pub = self.create_publisher(
            UEKinematicSetpoint, "/ue/kinematic_setpoint", 10
        )
        self._sequence = 0

    def _on_decision(self, msg: DecisionOutput) -> None:
        out = UEKinematicSetpoint()
        out.stamp_us = int(msg.stamp_us)
        out.source_stamp_us = int(msg.stamp_us)
        out.run_id = "decision-adapter"
        out.scene_seed = 0
        out.source_frame_index = 0
        out.sequence = self._sequence
        self._sequence += 1
        out.frame_id = "base_link"
        out.source_model_version = "trajectory_controller_v1"
        out.step_dt = DT_SEC
        out.delta_x_m = float(msg.desired_x)
        out.delta_y_m = float(msg.desired_y)
        out.hold_position = False
        out.valid = bool(msg.valid)
        out.reason = (
            "DECISION_VALID" if msg.valid else "DECISION_INVALID"
        )
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionSetpointAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
