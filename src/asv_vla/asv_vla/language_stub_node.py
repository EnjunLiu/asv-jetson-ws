"""Minimal language stub publishing a fixed 256-dim embedding."""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from asv_jetson_interfaces.msg import TaskEmbedding

EMBEDDING_DIM = 256

LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class LanguageStubNode(Node):
    def __init__(self) -> None:
        super().__init__("language_stub")
        self._pub = self.create_publisher(
            TaskEmbedding, "/vla/language_embedding", LANG_QOS
        )
        self._timer = self.create_timer(1.0, self._publish)
        self._seq = 0

    def _publish(self) -> None:
        msg = TaskEmbedding()
        msg.stamp_us = self.get_clock().now().nanoseconds // 1000
        msg.embedding = [0.0] * EMBEDDING_DIM
        msg.model_id = "stub:zero"
        msg.cached = True
        msg.valid = True
        self._pub.publish(msg)
        self._seq += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LanguageStubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
