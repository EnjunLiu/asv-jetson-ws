"""订阅 ``/task/text`` 并发布 ``/vla/language_embedding``。"""

from __future__ import annotations

import gc
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from interfaces.msg import TaskEmbedding

from .language import (
    DEFAULT_TASK_DESCRIPTION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PATH,
    EMBEDDING_DIM,
    LanguageEmbeddingState,
    LanguageEncoderError,
    USVLanguageEncoder,
    _bounded_detail,
    embedding_tuple,
    state_payload,
)

LANGUAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# 保留旧适配器使用的短名称，供诊断和启动合同检查使用。
LANG_QOS = LANGUAGE_QOS
# 首条真实任务到达时输出一次有界诊断。
LANGUAGE_TASK_TRACE_LIMIT = 1


class LanguageNode(Node):

    def __init__(self) -> None:
        super().__init__("language")

        self.run_id = str(self.declare_parameter("run_id", "language-qwen").value)
        self.model_path = str(
            self.declare_parameter("model_path", DEFAULT_MODEL_PATH).value
        )
        self.device = str(self.declare_parameter("device", "cuda").value).strip()
        self.model_id = str(
            self.declare_parameter("model_id", DEFAULT_MODEL_ID).value
        ).strip() or DEFAULT_MODEL_ID
        self.task_description = str(
            self.declare_parameter(
                "task_description", DEFAULT_TASK_DESCRIPTION
            ).value
        )
        self.max_chars = int(self.declare_parameter("max_chars", 512).value)
        self.cache_size = int(self.declare_parameter("cache_size", 32).value)
        self.release_model_after_encode = bool(
            self.declare_parameter("release_model_after_encode", False).value
        )
        self.publish_period_sec = max(
            0.1, float(self.declare_parameter("publish_period_sec", 1.0).value)
        )

        self._pub = self.create_publisher(
            TaskEmbedding, "/vla/language_embedding", LANGUAGE_QOS
        )
        self._subscription = self.create_subscription(
            String, "/task/text", self.on_task, LANGUAGE_QOS
        )
        self._state = LanguageEmbeddingState(
            model_id=self.model_id,
            detail="STARTING_LANGUAGE_ENCODER",
        )
        self._encoder: USVLanguageEncoder | None = None
        self._released_state: LanguageEmbeddingState | None = None
        self._instruction = ""
        self._task_trace_count = 0
        # 模型构造前输出 STARTING；Jetson 统一内存上的构造可能耗时。
        self._publish_current()

        # 启动失败时保持节点存活以便观察诊断；不改设备，也不重试 CPU，
        # 后续任务继续保持无效/停止。
        try:
            self._encoder = USVLanguageEncoder(
                self.model_path,
                output_dim=EMBEDDING_DIM,
                max_chars=self.max_chars,
                task_description=self.task_description,
                device=self.device,
                cache_size=self.cache_size,
            )
        except Exception as exc:
            self._set_invalid(
                "",
                f"MODEL_UNAVAILABLE:{type(exc).__name__}:{exc}",
            )
            self.get_logger().error(self._state.detail)
        else:
            ready_detail = (
                f"READY model={self.model_id};device={self.device};"
                f"cache_size={self.cache_size}"
            )
            self.get_logger().info(ready_detail)

        self._timer = self.create_timer(
            self.publish_period_sec, self._publish_current
        )
        # 启动参数是无头模式入口，立即编码；/task/text 仍可接收后续任务。
        if self.task_description.strip():
            self.on_task(String(data=self.task_description))
        else:
            self._publish_current()

    def _set_invalid(self, instruction: str, detail: object) -> None:
        self._instruction = str(instruction).strip()
        self._state = LanguageEmbeddingState(
            instruction=self._instruction,
            model_id=self.model_id,
            cached=False,
            valid=False,
            detail=_bounded_detail(detail),
        )

    def _release_encoder_model(self) -> str:
        """首次编码后释放模型引用和 CUDA 分配块。"""

        encoder = self._encoder
        self._encoder = None
        try:
            del encoder
            gc.collect()
            if self.device.startswith("cuda"):
                # 这里只做清理；CUDA 清理不可用时不重试 CPU，也不改设备。
                import torch

                torch.cuda.empty_cache()
            return "MODEL_RELEASED"
        except Exception as exc:
            self.get_logger().warn(
                _bounded_detail(f"MODEL_RELEASE_CLEANUP_ERROR:{exc}")
            )
            return f"MODEL_RELEASE_CLEANUP_ERROR:{type(exc).__name__}"

    def _trace_first_task(self, instruction: str) -> None:
        """只记录订阅器收到的首条非空任务。"""

        if self._task_trace_count >= LANGUAGE_TASK_TRACE_LIMIT:
            return
        self._task_trace_count += 1
        self.get_logger().info(
            "LANGUAGE_TASK_RECEIVED "
            f"instruction={_bounded_detail(instruction)}"
        )

    def on_task(self, message: String) -> None:
        """编码一条指令并立即发布以降低延迟。"""

        instruction = str(getattr(message, "data", "")).strip()
        if not instruction:
            self._set_invalid("", "EMPTY_INSTRUCTION")
            self._publish_current()
            return

        self._trace_first_task(instruction)

        if self._encoder is None:
            if self._released_state is not None:
                if instruction == self._released_state.instruction:
                    self._state = LanguageEmbeddingState(
                        instruction=self._released_state.instruction,
                        embedding=self._released_state.embedding,
                        model_id=self._released_state.model_id,
                        cached=True,
                        valid=True,
                        detail=(
                            f"{self._released_state.detail};"
                            "cache=hit;model_released"
                        ),
                    )
                    self._publish_current()
                    return
                self._set_invalid(
                    instruction,
                    "MODEL_RELEASED_AFTER_FIRST_ENCODE",
                )
                self._publish_current()
                return
            self._set_invalid(instruction, "MODEL_UNAVAILABLE")
            self._publish_current()
            return

        try:
            result = self._encoder.encode_with_metadata(instruction)
            embedding = embedding_tuple(result.embedding)
        except Exception as exc:
            error_kind = (
                "LANGUAGE_ENCODER_ERROR"
                if isinstance(exc, LanguageEncoderError)
                else type(exc).__name__
            )
            self._set_invalid(
                instruction,
                f"ENCODE_ERROR:{error_kind}:{exc}",
            )
            self.get_logger().error(self._state.detail)
        else:
            self._instruction = instruction
            self._state = LanguageEmbeddingState(
                instruction=instruction,
                embedding=embedding,
                model_id=self.model_id,
                cached=bool(result.cached),
                valid=True,
                detail=(
                    f"OK model={self.model_id};device={self.device};"
                    f"cache={'hit' if result.cached else 'miss'}"
                ),
            )
            if self.release_model_after_encode:
                self._released_state = self._state
                # 释放模型前先发布有效嵌入；瞬态本地发布器会在清理和感知节点延迟期间保留它。
                self._publish_current()
                release_detail = self._release_encoder_model()
                self._state = LanguageEmbeddingState(
                    instruction=self._state.instruction,
                    embedding=self._state.embedding,
                    model_id=self._state.model_id,
                    cached=self._state.cached,
                    valid=self._state.valid,
                    detail=f"{self._state.detail};{release_detail}",
                )
                self._released_state = self._state
                self.get_logger().info(
                    "LANGUAGE_READY_VALID "
                    f"instruction={instruction};release_model=true"
                )

        self._publish_current()

    def _publish_current(self) -> None:
        message = TaskEmbedding()
        fields = state_payload(
            self._state,
            run_id=self.run_id,
            stamp_us=self.get_clock().now().nanoseconds // 1000,
        )
        for field_name, value in fields.items():
            setattr(message, field_name, value)
        self._pub.publish(message)

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LanguageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
