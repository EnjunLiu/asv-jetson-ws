"""订阅语言和实体消息并发布期望位移。"""

from __future__ import annotations

import rclpy

from .decision import DecisionNode


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
