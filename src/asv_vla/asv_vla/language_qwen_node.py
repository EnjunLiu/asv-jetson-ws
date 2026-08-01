"""ROS 2 adapter for the live Qwen language encoder.

The node deliberately treats CUDA as a requested execution contract. If the
model cannot be loaded on the requested device, it stays alive and publishes
an invalid embedding plus an ``ERROR`` module status; it never silently
retries on CPU. The model remains resident by default so new task text is
encoded online rather than read from a cached ``.npy`` file.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    LanguageEncoderError,
    USVLanguageEncoder,
)


EMBEDDING_DIM = 256
DEFAULT_MODEL_PATH = "/home/jetson/jetson_asv_ws/models/Qwen3-Embedding-0.6B"
DEFAULT_MODEL_ID = "Qwen3-Embedding-0.6B"

LANGUAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# Keep the short name used by the existing stub/policy adapters available to
# diagnostics and launch-level contract checks.
LANG_QOS = LANGUAGE_QOS
RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
# Emit one bounded diagnostic when the first real task reaches this node.
LANGUAGE_TASK_TRACE_LIMIT = 1


def _bounded_detail(detail: object, limit: int = 240) -> str:
    """Keep status/message diagnostics bounded and single-line."""

    text = " ".join(str(detail).split())
    return text[:limit]


def _zero_embedding() -> tuple[float, ...]:
    return (0.0,) * EMBEDDING_DIM


@dataclass(frozen=True)
class LanguageEmbeddingState:
    """ROS-independent state used to construct each published embedding."""

    instruction: str = ""
    embedding: tuple[float, ...] = field(default_factory=_zero_embedding)
    model_id: str = DEFAULT_MODEL_ID
    cached: bool = False
    valid: bool = False
    detail: str = "WAITING_FOR_INSTRUCTION"


def _state_payload(
    state: LanguageEmbeddingState, *, run_id: str, stamp_us: int
) -> dict[str, Any]:
    """Return the exact ``TaskEmbedding`` fields without requiring ROS types."""

    return {
        "stamp_us": int(stamp_us),
        "run_id": str(run_id),
        "instruction": str(state.instruction),
        "model_id": str(state.model_id),
        "embedding_dim": EMBEDDING_DIM,
        "embedding": list(state.embedding),
        "cached": bool(state.cached),
        "valid": bool(state.valid),
        "detail": _bounded_detail(state.detail),
    }


def _embedding_tuple(values: object) -> tuple[float, ...]:
    """Validate and copy an encoder result into the fixed ROS contract."""

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != EMBEDDING_DIM or not np.all(np.isfinite(array)):
        raise ValueError(
            f"encoder returned an invalid {array.size}-value embedding"
        )
    return tuple(float(value) for value in array)


class LanguageQwenNode(Node):
    """Publish Qwen ``TaskEmbedding`` messages from ``/task/text``."""

    def __init__(self) -> None:
        super().__init__("language_qwen")

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
        self._status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", LANGUAGE_QOS
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
        self.module_state = ModuleStatus.STARTING
        self.input_ready = False
        self.output_valid = False
        self._status_detail = "STARTING_LANGUAGE_ENCODER"
        self._task_trace_count = 0
        # Emit STARTING before model construction, which may take a bounded
        # but non-trivial amount of time on Jetson unified memory.
        self._publish_current()
        self._publish_status()

        # Keep the node alive on startup failures so diagnostics can be
        # observed. This exception path does not alter ``self.device`` or
        # retry on CPU; subsequent task messages remain invalid/hold.
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
            self.module_state = ModuleStatus.ERROR
            self._status_detail = self._state.detail
            self.get_logger().error(self._status_detail)
        else:
            self.module_state = ModuleStatus.READY
            self._status_detail = (
                f"READY model={self.model_id};device={self.device};"
                f"cache_size={self.cache_size}"
            )
            self.get_logger().info(self._status_detail)

        self._timer = self.create_timer(
            self.publish_period_sec, self._publish_current
        )
        self._status_timer = self.create_timer(1.0, self._publish_status)
        # Publish an initial invalid/empty state so transient-local subscribers
        # observe a fail-closed value before the first instruction arrives.
        self._publish_current()
        self._publish_status()

    def _set_invalid(self, instruction: str, detail: object) -> None:
        self._instruction = str(instruction).strip()
        self._state = LanguageEmbeddingState(
            instruction=self._instruction,
            model_id=self.model_id,
            cached=False,
            valid=False,
            detail=_bounded_detail(detail),
        )
        self.input_ready = bool(self._instruction)
        self.output_valid = False

    def _release_encoder_model(self) -> str:
        """Release model references and CUDA allocator blocks after first encode."""

        encoder = self._encoder
        self._encoder = None
        try:
            del encoder
            gc.collect()
            if self.device.startswith("cuda"):
                # This is cleanup only: there is deliberately no CPU retry or
                # device reassignment if CUDA cleanup is unavailable.
                import torch

                torch.cuda.empty_cache()
            return "MODEL_RELEASED"
        except Exception as exc:
            self.get_logger().warn(
                _bounded_detail(f"MODEL_RELEASE_CLEANUP_ERROR:{exc}")
            )
            return f"MODEL_RELEASE_CLEANUP_ERROR:{type(exc).__name__}"

    def _trace_first_task(self, instruction: str) -> None:
        """Record only the first non-empty task received by the subscriber."""

        if self._task_trace_count >= LANGUAGE_TASK_TRACE_LIMIT:
            return
        self._task_trace_count += 1
        self.get_logger().info(
            "LANGUAGE_TASK_RECEIVED "
            f"instruction={_bounded_detail(instruction)}"
        )

    def on_task(self, message: String) -> None:
        """Encode one instruction and publish immediately for low latency."""

        instruction = str(getattr(message, "data", "")).strip()
        if not instruction:
            self._set_invalid("", "EMPTY_INSTRUCTION")
            self.module_state = (
                ModuleStatus.ERROR
                if self._encoder is None
                else ModuleStatus.DEGRADED
            )
            self._status_detail = self._state.detail
            self._publish_current()
            self._publish_status()
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
                    self.module_state = ModuleStatus.READY
                    self.input_ready = True
                    self.output_valid = True
                    self._status_detail = self._state.detail
                    self._publish_current()
                    self._publish_status()
                    return
                self._set_invalid(
                    instruction,
                    "MODEL_RELEASED_AFTER_FIRST_ENCODE",
                )
                self.module_state = ModuleStatus.DEGRADED
                self._status_detail = self._state.detail
                self._publish_current()
                self._publish_status()
                return
            self._set_invalid(instruction, "MODEL_UNAVAILABLE")
            self.module_state = ModuleStatus.ERROR
            self._status_detail = self._state.detail
            self._publish_current()
            self._publish_status()
            return

        try:
            result = self._encoder.encode_with_metadata(instruction)
            embedding = _embedding_tuple(result.embedding)
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
            self.module_state = ModuleStatus.DEGRADED
            self._status_detail = self._state.detail
            self.get_logger().error(self._status_detail)
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
            self.module_state = ModuleStatus.READY
            self.input_ready = True
            self.output_valid = True
            self._status_detail = self._state.detail
            if self.release_model_after_encode:
                self._released_state = self._state
                # Deliver one valid embedding before dropping the model. The
                # transient-local publisher retains this value while cleanup
                # runs and while the staged visual node is still delayed.
                self._publish_current()
                self._publish_status()
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
                self._status_detail = self._state.detail
                self.get_logger().info(
                    "LANGUAGE_READY_VALID "
                    f"instruction={instruction};release_model=true"
                )

        self._publish_current()
        self._publish_status()

    def _publish_current(self) -> None:
        message = TaskEmbedding()
        fields = _state_payload(
            self._state,
            run_id=self.run_id,
            stamp_us=self.get_clock().now().nanoseconds // 1000,
        )
        for field_name, value in fields.items():
            setattr(message, field_name, value)
        self._pub.publish(message)

    def _publish_status(self) -> None:
        status = ModuleStatus()
        status.stamp_us = self.get_clock().now().nanoseconds // 1000
        status.run_id = self.run_id
        status.module_name = "language_qwen"
        status.state = self.module_state
        status.alive = True
        status.input_ready = self.input_ready
        status.output_valid = self.output_valid
        status.detail = _bounded_detail(self._status_detail)
        self._status_pub.publish(status)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LanguageQwenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
