"""Publish the active task instruction for the closed-loop VLA graph.

The task text is a latched (transient-local) ``std_msgs/String`` so language
backends can start after this node and still receive the current instruction.
An empty parameter is rejected and never published as a valid task.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


DEFAULT_TASK_TEXT = "跟随红色目标船，保持3米距离"
TASK_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _validate_task_text(value: object) -> str:
    """Normalize a task parameter and reject empty instructions."""

    text = str(value).strip()
    if not text:
        raise ValueError("task_text must be non-empty")
    return text


class TaskInstructionNode(Node):
    """Publish one deterministic task string and refresh it for late joiners."""

    def __init__(self) -> None:
        super().__init__("task_instruction")
        self._pub = self.create_publisher(String, "/task/text", TASK_QOS)
        self._task_text = ""
        self.last_error = ""

        raw_text = self.declare_parameter(
            "task_text", DEFAULT_TASK_TEXT
        ).value
        try:
            self._task_text = _validate_task_text(raw_text)
        except ValueError as exc:
            self.last_error = str(exc)
            self.get_logger().error(self.last_error)
            return

        self.get_logger().info(f"publishing task text: {self._task_text}")
        self._publish()
        self._timer = self.create_timer(1.0, self._publish)

    def _publish(self) -> None:
        if not self._task_text:
            # Never send an empty String: subscribers must keep their
            # fail-closed/no-task state until a valid parameter is supplied.
            return
        message = String()
        message.data = self._task_text
        self._pub.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskInstructionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
