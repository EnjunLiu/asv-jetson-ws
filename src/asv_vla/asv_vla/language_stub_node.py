"""Language stub publishing a selectable pre-computed instruction embedding.

Loads a pre-computed instruction embedding (npy, 256-D) so the policy
receives the exact embedding it was trained with for the demo instruction
(e.g. "follow red 3 m").  The active embedding can be switched at runtime
through the ``active_embedding`` parameter (``ros2 param set
/language_stub active_embedding <path>``), which lets a demo switch between
"follow the red boat" and "follow the blue boat" without restarting.
Falls back to a zero embedding when the file is missing so the launch still
starts.
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
        self.declare_parameter("active_embedding", "")
        self._pub = self.create_publisher(
            TaskEmbedding, "/vla/language_embedding", LANG_QOS
        )
        self._embedding = [0.0] * EMBEDDING_DIM
        self._model_id = "stub:zero"
        self._load_embedding(Path(self._active_path()))
        self._timer = self.create_timer(1.0, self._publish)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._seq = 0

    def _active_path(self) -> str:
        override = (
            str(self.get_parameter("active_embedding").get_parameter_value().string_value)
            .strip()
        )
        if override:
            return override
        return str(
            self.get_parameter("embedding_path").get_parameter_value().string_value
        )

    def _load_embedding(self, path: Path) -> None:
        if not path.is_file():
            self.get_logger().warn(
                f"embedding file missing ({path}); using zero embedding"
            )
            self._embedding = [0.0] * EMBEDDING_DIM
            self._model_id = "stub:zero"
            return
        try:
            array = np.load(path, allow_pickle=False).reshape(-1)
            if array.size != EMBEDDING_DIM or not np.all(np.isfinite(array)):
                raise ValueError("expected a finite 256-D embedding")
            self._embedding = [float(v) for v in array]
            self._model_id = f"stub:file:{path.stem}"
            self.get_logger().info(
                f"language stub loaded instruction embedding from {path} "
                f"(model_id={self._model_id})"
            )
        except Exception as exc:
            self.get_logger().error(
                f"failed to load embedding from {path}: {exc}; using zeros"
            )
            self._embedding = [0.0] * EMBEDDING_DIM
            self._model_id = "stub:zero"

    def _on_set_parameters(self, changes) -> list:
        for change in changes:
            if change.name == "active_embedding":
                path = Path(str(change.value)).expanduser().resolve()
                self._load_embedding(path)
                # Publish the new embedding immediately.
                self._publish()
        return []

    def _publish(self) -> None:
        msg = TaskEmbedding()
        msg.stamp_us = self.get_clock().now().nanoseconds // 1000
        msg.embedding = list(self._embedding)
        msg.model_id = self._model_id
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
