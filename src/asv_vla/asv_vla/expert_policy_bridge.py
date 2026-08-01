"""Bridge: /vla/expert_trajectory (ExpertTrajectory) -> /vla/policy_trajectory.

Deterministic-expert fallback for the closed loop.  The project spec
sanctions the expert as the control-path reference when the learned policy
is unstable (its per-frame outputs oscillate under the dynamic UE5 water
simulation).  The safety gate and every downstream node is unchanged: they
consume /vla/policy_trajectory as before.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import ExpertTrajectory, SelectedTrajectory


class ExpertPolicyBridge(Node):
    def __init__(self) -> None:
        super().__init__("expert_policy_bridge")
        self._sub = self.create_subscription(
            ExpertTrajectory, "/vla/expert_trajectory", self._on_expert, 10
        )
        self._pub = self.create_publisher(
            SelectedTrajectory, "/vla/policy_trajectory", 10
        )

    def _on_expert(self, msg: ExpertTrajectory) -> None:
        out = SelectedTrajectory()
        out.stamp_us = int(msg.stamp_us)
        out.run_id = str(msg.run_id)
        out.scene_seed = int(msg.scene_seed)
        out.frame_index = int(msg.frame_index)
        out.frame_id = str(msg.frame_id)
        out.model_version = f"expert:{msg.model_version}"
        out.dt = float(msg.dt)
        out.horizon = int(msg.horizon)
        out.delta_p_xy = [float(v) for v in msg.delta_p_xy]
        out.safe_stop = bool(msg.safe_stop)
        out.valid = bool(msg.valid)
        if not out.valid:
            out.reason = "EXPERT_INVALID"
        elif out.safe_stop:
            out.reason = "EXPERT_STOP"
        else:
            out.reason = "EXPERT_INFERRED"
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExpertPolicyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
