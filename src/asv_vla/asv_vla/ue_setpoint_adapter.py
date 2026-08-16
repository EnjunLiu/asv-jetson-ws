"""Convert the final safe displacement into the UE5 execution contract."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import DesiredDisplacement, UESetpoint


def _identity_complete(message: DesiredDisplacement) -> bool:
    try:
        scene_seed = int(message.scene_seed)
        frame_index = int(message.frame_index)
    except (TypeError, ValueError, AttributeError):
        return False
    return bool(
        str(message.run_id).strip()
        and str(message.source).strip()
        and scene_seed > 0
        and frame_index >= 0
    )


class UESetpointAdapter(Node):
    def __init__(self) -> None:
        super().__init__("ue_setpoint_adapter")
        self._sub = self.create_subscription(
            DesiredDisplacement,
            "/control/desired_displacement",
            self._on_displacement,
            10,
        )
        self._pub = self.create_publisher(
            UESetpoint, "/ue/kinematic_setpoint", 10
        )
        self._sequence = 0

    def _on_displacement(self, msg: DesiredDisplacement) -> None:
        identity_complete = _identity_complete(msg)
        executable = bool(msg.valid) and not bool(msg.safe_stop) and identity_complete
        out = UESetpoint()
        out.stamp_us = int(msg.stamp_us)
        out.source_stamp_us = int(msg.stamp_us)
        out.run_id = str(msg.run_id)
        out.scene_seed = int(msg.scene_seed)
        out.source_frame_index = int(msg.frame_index)
        out.sequence = self._sequence
        self._sequence += 1
        out.frame_id = str(msg.frame_id)
        out.source_model_version = str(msg.source)
        out.step_dt = float(msg.step_dt)
        out.delta_x_m = float(msg.desired_x) if executable else 0.0
        out.delta_y_m = float(msg.desired_y) if executable else 0.0
        out.hold_position = not executable
        out.valid = executable
        if not identity_complete:
            out.reason = "IDENTITY_MISSING"
        else:
            out.reason = str(msg.reason)
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UESetpointAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
