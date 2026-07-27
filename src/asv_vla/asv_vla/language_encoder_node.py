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

from asv_jetson_interfaces.msg import ModuleStatus, TaskEmbedding

from .language_encoder import (
    DEFAULT_TASK_DESCRIPTION,
    EmptyInstructionError,
    InstructionTooLongError,
    LanguageEncoderError,
    USVLanguageEncoder,
)


LANGUAGE_DIM = 256
RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000


class LanguageEncoderNode(Node):
    def __init__(self) -> None:
        super().__init__("language_encoder")
        self.run_id = (
            self.declare_parameter("run_id", "language-embedding")
            .get_parameter_value()
            .string_value
        )
        self.model_path = (
            self.declare_parameter(
                "model_path", "models/Qwen3-Embedding-0.6B"
            )
            .get_parameter_value()
            .string_value
        )
        self.model_id = (
            self.declare_parameter(
                "model_id", "Qwen/Qwen3-Embedding-0.6B"
            )
            .get_parameter_value()
            .string_value
        )
        self.device = (
            self.declare_parameter("device", "cuda")
            .get_parameter_value()
            .string_value
        )
        self.output_dim = (
            self.declare_parameter("output_dim", LANGUAGE_DIM)
            .get_parameter_value()
            .integer_value
        )
        self.max_chars = (
            self.declare_parameter("max_chars", 512)
            .get_parameter_value()
            .integer_value
        )
        self.cache_size = (
            self.declare_parameter("cache_size", 32)
            .get_parameter_value()
            .integer_value
        )
        self.task_description = (
            self.declare_parameter(
                "task_description", DEFAULT_TASK_DESCRIPTION
            )
            .get_parameter_value()
            .string_value
        )

        self.publisher = self.create_publisher(
            TaskEmbedding, "/vla/language_embedding", LATCHED_QOS
        )
        self.subscription = self.create_subscription(
            String, "/task/text", self.on_text, LATCHED_QOS
        )
        self.status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", RELIABLE_QOS
        )
        self.create_timer(1.0, self.publish_status)

        self.encoder = None
        self.module_state = ModuleStatus.STARTING
        self.input_ready = False
        self.output_valid = False
        self.detail = "loading frozen language encoder"
        try:
            self.encoder = USVLanguageEncoder(
                self.model_path,
                output_dim=self.output_dim,
                max_chars=self.max_chars,
                task_description=self.task_description,
                device=self.device,
                cache_size=self.cache_size,
            )
        except Exception as exc:
            self.module_state = ModuleStatus.ERROR
            self.detail = f"MODEL_LOAD_ERROR:{type(exc).__name__}:{exc}"
            self.get_logger().error(self.detail)
        else:
            self.module_state = ModuleStatus.READY
            self.detail = (
                f"ready model={self.model_id} device={self.device} "
                f"output_dim={self.output_dim}"
            )
            self.get_logger().info(self.detail)

    def _new_message(self, instruction: str) -> TaskEmbedding:
        message = TaskEmbedding()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.instruction = instruction
        message.model_id = self.model_id
        message.embedding_dim = self.output_dim
        message.embedding = [0.0] * self.output_dim
        message.cached = False
        message.valid = False
        message.detail = "UNINITIALIZED"
        return message

    def _publish_invalid(
        self, message: TaskEmbedding, detail: str, *, input_ready: bool
    ) -> None:
        message.embedding = [0.0] * self.output_dim
        message.cached = False
        message.valid = False
        message.detail = detail
        self.publisher.publish(message)
        self.input_ready = input_ready
        self.output_valid = False
        if self.encoder is None:
            self.module_state = ModuleStatus.ERROR
        else:
            self.module_state = ModuleStatus.DEGRADED
        self.detail = detail
        self.get_logger().warning(detail)

    def on_text(self, task: String) -> None:
        instruction = task.data
        message = self._new_message(instruction)

        if self.encoder is None:
            self._publish_invalid(
                message,
                "MODEL_UNAVAILABLE: language encoder failed to load",
                input_ready=bool(instruction.strip()),
            )
            return

        try:
            result = self.encoder.encode_with_metadata(instruction)
        except EmptyInstructionError as exc:
            self._publish_invalid(
                message,
                f"EMPTY_INSTRUCTION:{exc}",
                input_ready=False,
            )
            return
        except InstructionTooLongError as exc:
            self._publish_invalid(
                message,
                f"INSTRUCTION_TOO_LONG:{exc}",
                input_ready=True,
            )
            return
        except LanguageEncoderError as exc:
            self._publish_invalid(
                message,
                f"ENCODE_ERROR:{exc}",
                input_ready=True,
            )
            return
        except Exception as exc:
            self._publish_invalid(
                message,
                f"UNEXPECTED_ENCODE_ERROR:{type(exc).__name__}:{exc}",
                input_ready=True,
            )
            return

        message.embedding = result.embedding.tolist()
        message.cached = result.cached
        message.valid = True
        message.detail = "OK:CACHE_HIT" if result.cached else "OK:INFERRED"
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = True
        self.module_state = ModuleStatus.READY
        self.detail = message.detail

    def publish_status(self) -> None:
        message = ModuleStatus()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.module_name = self.get_name()
        message.state = self.module_state
        message.alive = True
        message.input_ready = self.input_ready
        message.output_valid = self.output_valid
        message.detail = self.detail
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LanguageEncoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
