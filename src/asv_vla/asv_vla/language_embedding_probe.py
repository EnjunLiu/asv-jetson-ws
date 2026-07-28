from __future__ import annotations

import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from asv_jetson_interfaces.msg import TaskEmbedding


LANGUAGE_DIM = 256
REQUIRED_MESSAGES = 10
TIMEOUT_SEC = 180.0
REPEAT_TOLERANCE = 1.0e-5
NORM_TOLERANCE = 1.0e-4

LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class LanguageEmbeddingProbe(Node):
    def __init__(self) -> None:
        super().__init__("language_embedding_probe")
        self.started = time.monotonic()
        self.messages = []
        self.done = False
        self.passed = False
        self.subscription = self.create_subscription(
            TaskEmbedding,
            "/vla/language_embedding",
            self.on_embedding,
            LATCHED_QOS,
        )
        self.create_timer(0.1, self.evaluate)

    def on_embedding(self, message: TaskEmbedding) -> None:
        if self.done:
            return
        self.messages.append(message)

    @staticmethod
    def _finite(values) -> bool:
        return all(math.isfinite(value) for value in values)

    @staticmethod
    def _norm(values) -> float:
        return math.sqrt(sum(value * value for value in values))

    @staticmethod
    def _max_abs_diff(left, right) -> float:
        return max(abs(a - b) for a, b in zip(left, right))

    def evaluate(self) -> None:
        if len(self.messages) >= REQUIRED_MESSAGES:
            sample = self.messages[:REQUIRED_MESSAGES]
            first = sample[0]
            checks = {
                "all_valid": all(message.valid for message in sample),
                "dimension_field": all(
                    message.embedding_dim == LANGUAGE_DIM for message in sample
                ),
                "vector_shape": all(
                    len(message.embedding) == LANGUAGE_DIM for message in sample
                ),
                "finite": all(
                    self._finite(message.embedding) for message in sample
                ),
                "normalized": all(
                    abs(self._norm(message.embedding) - 1.0)
                    <= NORM_TOLERANCE
                    for message in sample
                ),
                "real_model": all(
                    message.model_id
                    and not message.model_id.startswith("stub:")
                    for message in sample
                ),
                "same_instruction": all(
                    message.instruction == first.instruction for message in sample
                ),
                "repeat_deterministic": all(
                    self._max_abs_diff(first.embedding, message.embedding)
                    <= REPEAT_TOLERANCE
                    for message in sample[1:]
                ),
                "cache_hit_observed": any(
                    message.cached for message in sample[1:]
                ),
            }
            for name, passed in checks.items():
                if passed:
                    self.get_logger().info(f"{name}=PASS")
                else:
                    self.get_logger().error(f"{name}=FAIL")
            self.passed = all(checks.values())
            marker = (
                "LANGUAGE_EMBEDDING_PASS"
                if self.passed
                else "LANGUAGE_EMBEDDING_FAIL"
            )
            self.get_logger().info(marker)
            self.done = True
            return

        if time.monotonic() - self.started > TIMEOUT_SEC:
            self.get_logger().error(
                "LANGUAGE_EMBEDDING_FAIL "
                f"received={len(self.messages)} required={REQUIRED_MESSAGES}"
            )
            self.passed = False
            self.done = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LanguageEmbeddingProbe()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        exit_code = 0 if node.passed else 1
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
