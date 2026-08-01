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


def _identity_complete(message: DecisionOutput) -> bool:
    """Return whether a decision carries an attributable source identity.

    Scene seeds are positive by contract, while frame index zero is valid for
    the first frame in a run.  ``getattr`` keeps this adapter fail-closed if
    an old generated message type is accidentally sourced before the
    interface package is rebuilt.
    """

    if not all(
        hasattr(message, field)
        for field in (
            "run_id",
            "scene_seed",
            "source_frame_index",
            "source_model_version",
        )
    ):
        return False
    try:
        scene_seed = int(message.scene_seed)
        source_frame_index = int(message.source_frame_index)
    except (TypeError, ValueError, AttributeError):
        return False
    return bool(
        str(message.run_id).strip()
        and str(message.source_model_version).strip()
        and scene_seed > 0
        and source_frame_index >= 0
    )


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
        identity_complete = _identity_complete(msg)
        executable = bool(msg.valid) and identity_complete
        out = UEKinematicSetpoint()
        out.stamp_us = int(msg.stamp_us)
        out.source_stamp_us = int(msg.stamp_us)
        out.run_id = str(getattr(msg, "run_id", ""))
        out.scene_seed = int(getattr(msg, "scene_seed", 0))
        out.source_frame_index = int(getattr(msg, "source_frame_index", 0))
        out.sequence = self._sequence
        self._sequence += 1
        out.frame_id = "base_link"
        out.source_model_version = str(
            getattr(msg, "source_model_version", "")
        )
        out.step_dt = DT_SEC
        out.delta_x_m = float(msg.desired_x) if executable else 0.0
        out.delta_y_m = float(msg.desired_y) if executable else 0.0
        out.hold_position = not executable
        out.valid = executable
        if not identity_complete:
            out.reason = "IDENTITY_MISSING"
        else:
            out.reason = "DECISION_VALID" if msg.valid else "DECISION_INVALID"
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
