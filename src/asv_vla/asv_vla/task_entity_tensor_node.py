from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    ModuleStatus,
    TaskFeatures,
    UEEntityArray,
)

from .task_entity_tensor import (
    BACKEND_ID,
    DEFAULT_RISK_HORIZON_SEC,
    DEFAULT_RISK_RADIUS_M,
    FEATURE_DIM,
    MAX_ENTITIES,
    TaskEntityTensorError,
    build_entity_tensor,
)


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)


def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000


class TaskEntityTensorNode(Node):
    def __init__(self) -> None:
        super().__init__("task_entity_tensor")
        self._last_stamp_us = 0
        self.status_run_id = (
            self.declare_parameter("run_id", "task-entity-tensor")
            .get_parameter_value()
            .string_value
        )
        self.max_entities = (
            self.declare_parameter("max_entities", MAX_ENTITIES)
            .get_parameter_value()
            .integer_value
        )
        self.risk_horizon_sec = (
            self.declare_parameter(
                "risk_horizon_sec", DEFAULT_RISK_HORIZON_SEC
            )
            .get_parameter_value()
            .double_value
        )
        self.risk_radius_m = (
            self.declare_parameter(
                "risk_radius_m", DEFAULT_RISK_RADIUS_M
            )
            .get_parameter_value()
            .double_value
        )
        self.entities_topic = str(
            self.declare_parameter("entities_topic", "/vla/tracked_entities")
            .get_parameter_value()
            .string_value
        )
        self.allow_truth_entities = bool(
            self.declare_parameter("allow_truth_entities", False)
            .get_parameter_value()
            .bool_value
        )
        if self.max_entities != MAX_ENTITIES:
            raise ValueError(
                f"max_entities must remain fixed at {MAX_ENTITIES}"
            )
        if self.risk_horizon_sec <= 0.0:
            raise ValueError("risk_horizon_sec must be positive")
        if self.risk_radius_m <= 0.0:
            raise ValueError("risk_radius_m must be positive")

        self.publisher = self.create_publisher(
            TaskFeatures, "/vla/task_features", RELIABLE_QOS
        )
        self.status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            UEEntityArray, self.entities_topic, self.on_entities, RELIABLE_QOS
        )
        self.create_timer(1.0, self.publish_status)

        self.module_state = ModuleStatus.READY
        self.input_ready = False
        self.output_valid = False
        self.detail = (
            f"waiting for {self.entities_topic}; backend={BACKEND_ID} "
            f"shape={self.max_entities}x{FEATURE_DIM}"
        )

    def _new_message(self, source: UEEntityArray) -> TaskFeatures:
        message = TaskFeatures()
        # The UE5 simulation clock can step backwards in headless runs
        # (frame-rate dependent time accumulation); monotonicise the stamp
        # so downstream staleness checks (safety gate) see a strict order.
        if source.stamp_us > self._last_stamp_us:
            self._last_stamp_us = int(source.stamp_us)
        else:
            self._last_stamp_us += 1
        message.stamp_us = self._last_stamp_us
        message.run_id = source.run_id
        message.scene_seed = source.scene_seed
        message.frame_index = source.frame_index
        message.frame_id = source.frame_id
        message.backend = BACKEND_ID
        message.max_entities = self.max_entities
        message.feature_dim = FEATURE_DIM
        message.entity_count = 0
        message.entity_ids = [""] * self.max_entities
        message.features = [0.0] * (self.max_entities * FEATURE_DIM)
        message.mask = [False] * self.max_entities
        message.valid = False
        message.instruction_id = str(source.instruction_id)
        message.instruction = str(source.instruction)
        message.detail = "UNINITIALIZED"
        return message

    def _publish_invalid(
        self,
        source: UEEntityArray,
        detail: str,
        *,
        input_ready: bool,
    ) -> None:
        message = self._new_message(source)
        message.detail = detail
        self.publisher.publish(message)
        self.input_ready = input_ready
        self.output_valid = False
        self.module_state = ModuleStatus.DEGRADED
        self.detail = detail
        self.get_logger().warning(detail)

    def on_entities(self, source: UEEntityArray) -> None:
        if (
            not self.allow_truth_entities
            and str(source.source) not in {"image_perception", "temporal_tracker"}
        ):
            self._publish_invalid(
                source,
                f"UNTRUSTED_ENTITY_SOURCE:{source.source!r}",
                input_ready=False,
            )
            return
        if not source.valid:
            self._publish_invalid(
                source,
                f"INVALID_SOURCE:{source.detail}",
                input_ready=False,
            )
            return
        if not source.run_id.strip():
            self._publish_invalid(
                source,
                "INVALID_RUN_ID: run_id is empty",
                input_ready=True,
            )
            return
        if source.frame_id != "base_link":
            self._publish_invalid(
                source,
                f"INVALID_FRAME: expected base_link, got "
                f"{source.frame_id!r}",
                input_ready=True,
            )
            return

        try:
            result = build_entity_tensor(
                source.entities,
                max_entities=self.max_entities,
                risk_horizon_sec=self.risk_horizon_sec,
                risk_radius_m=self.risk_radius_m,
            )
        except (TaskEntityTensorError, ValueError) as exc:
            self._publish_invalid(
                source,
                f"{type(exc).__name__.upper()}:{exc}",
                input_ready=True,
            )
            return
        except Exception as exc:
            self._publish_invalid(
                source,
                f"UNEXPECTED_ENTITY_TENSOR_ERROR:"
                f"{type(exc).__name__}:{exc}",
                input_ready=True,
            )
            return

        message = self._new_message(source)
        message.entity_count = result.entity_count
        message.entity_ids = list(result.entity_ids)
        message.features = result.features.reshape(-1).tolist()
        message.mask = result.mask.tolist()
        message.valid = True
        message.detail = (
            f"OK:selected={result.entity_count};"
            f"targets={result.target_count};"
            f"risks={result.risk_count};"
            f"dropped={result.dropped_count}"
        )
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = True
        self.module_state = ModuleStatus.READY
        self.detail = message.detail

    def publish_status(self) -> None:
        message = ModuleStatus()
        message.stamp_us = now_us(self)
        message.run_id = self.status_run_id
        message.module_name = self.get_name()
        message.state = self.module_state
        message.alive = True
        message.input_ready = self.input_ready
        message.output_valid = self.output_valid
        message.detail = self.detail
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskEntityTensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
