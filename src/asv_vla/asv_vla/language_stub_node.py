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
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from asv_jetson_interfaces.msg import TaskEmbedding

EMBEDDING_DIM = 256

LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _read_embedding(path: Path) -> tuple[list[float], str] | None:
    """Read and validate an embedding without mutating node state."""

    if not path.is_file():
        return None
    try:
        array = np.load(path, allow_pickle=False).reshape(-1)
        if array.size != EMBEDDING_DIM or not np.all(np.isfinite(array)):
            return None
    except Exception:
        return None
    return [float(v) for v in array], f"stub:file:{path.stem}"


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
        loaded = _read_embedding(path)
        if loaded is None:
            if not path.is_file():
                detail = f"embedding file missing ({path})"
            else:
                detail = f"failed to load valid embedding from {path}"
            self.get_logger().warn(
                f"{detail}; using zero embedding"
            )
            self._embedding = [0.0] * EMBEDDING_DIM
            self._model_id = "stub:zero"
            return
        self._embedding, self._model_id = loaded
        self.get_logger().info(
            f"language stub loaded instruction embedding from {path} "
            f"(model_id={self._model_id})"
        )

    def _on_set_parameters(self, changes) -> SetParametersResult:
        pending: tuple[list[float], str] | None = None
        pending_path: Path | None = None
        for change in changes:
            if change.name == "active_embedding":
                requested = str(change.value).strip()
                if requested:
                    path = Path(requested).expanduser().resolve()
                else:
                    path = Path(
                        str(
                            self.get_parameter("embedding_path")
                            .get_parameter_value()
                            .string_value
                        )
                    ).expanduser().resolve()
                loaded = _read_embedding(path)
                if loaded is None:
                    return SetParametersResult(
                        successful=False,
                        reason=f"invalid embedding file: {path}",
                    )
                pending = loaded
                pending_path = path
        if pending is not None and pending_path is not None:
            self._embedding, self._model_id = pending
            self.get_logger().info(
                f"language stub loaded instruction embedding from {pending_path} "
                f"(model_id={self._model_id})"
            )
            # Publish the new embedding immediately.
            self._publish()
        return SetParametersResult(successful=True, reason="embedding updated")

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
