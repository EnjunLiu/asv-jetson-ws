"""Minimal language stub publishing a fixed 256-dim embedding.

Loads a pre-computed instruction embedding (npy, 256-D) if present, so the
policy receives the exact embedding it was trained with for the demo
instruction ("follow red 3 m").  Falls back to a zero embedding when the
file is missing so the launch still starts.
"""

from pathlib import Path

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
        self.declare_parameter(
            "embedding_path", "models/demo_instruction_embedding.npy"
        )
        self._pub = self.create_publisher(
            TaskEmbedding, "/vla/language_embedding", LANG_QOS
        )
        path = Path(
            str(self.get_parameter("embedding_path").get_parameter_value().string_value)
        ).expanduser().resolve()
        self._embedding = self._load_embedding(path)
        self._timer = self.create_timer(1.0, self._publish)
        self._seq = 0

    def _load_embedding(self, path: Path) -> list[float]:
        if not path.is_file():
            self.get_logger().warn(
                f"embedding file missing ({path}); using zero embedding"
            )
            return [0.0] * EMBEDDING_DIM
        try:
            array = np.load(path, allow_pickle=False).reshape(-1)
            if array.size != EMBEDDING_DIM or not np.all(np.isfinite(array)):
                raise ValueError("expected a finite 256-D embedding")
            self.get_logger().info(
                f"language stub loaded instruction embedding from {path}"
            )
            return [float(v) for v in array]
        except Exception as exc:
            self.get_logger().error(
                f"failed to load embedding from {path}: {exc}; using zeros"
            )
            return [0.0] * EMBEDDING_DIM

    def _publish(self) -> None:
        msg = TaskEmbedding()
        msg.stamp_us = self.get_clock().now().nanoseconds // 1000
        msg.embedding = list(self._embedding)
        msg.model_id = (
            "stub:file" if any(v != 0.0 for v in self._embedding) else "stub:zero"
        )
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
