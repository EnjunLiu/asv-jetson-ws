"""ROS 2 wrapper for deterministic Day 9 expert labels."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    ExpertTrajectory,
    ModuleStatus,
    UEEntityArray,
)

from .expert_trajectory import (
    DEFAULT_MAX_SPEED_MPS,
    MODEL_VERSION,
    ExpertTrajectoryError,
    generate_expert_trajectory,
    task_from_labels,
)
from .trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, HORIZON


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
)


def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000


class ExpertTrajectoryNode(Node):
    def __init__(self) -> None:
        super().__init__("expert_trajectory")
        self.status_run_id = (
            self.declare_parameter("run_id", "day9-expert")
            .get_parameter_value()
            .string_value
        )
        action = (
            self.declare_parameter("action", "follow")
            .get_parameter_value()
            .string_value
        )
        target_attribute = (
            self.declare_parameter("target_attribute", "color:red")
            .get_parameter_value()
            .string_value
        )
        distance_bucket = (
            self.declare_parameter("distance_bucket", "3m")
            .get_parameter_value()
            .string_value
        )
        self.max_speed_mps = (
            self.declare_parameter(
                "max_speed_mps", DEFAULT_MAX_SPEED_MPS
            )
            .get_parameter_value()
            .double_value
        )
        self.task = task_from_labels(
            action, target_attribute, distance_bucket
        )

        self.publisher = self.create_publisher(
            ExpertTrajectory, "/vla/expert_trajectory", RELIABLE_QOS
        )
        self.status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            UEEntityArray, "/ue/entities", self.on_entities, RELIABLE_QOS
        )
        self.create_timer(1.0, self.publish_status)

        self.module_state = ModuleStatus.READY
        self.input_ready = False
        self.output_valid = False
        self.detail = (
            f"waiting for /ue/entities; action={self.task.action};"
            f"target={self.task.target_attribute};"
            f"distance_m={self.task.desired_distance_m:.1f};"
            f"model={MODEL_VERSION}"
        )
        self.get_logger().info(self.detail)

    def _message(self, source: UEEntityArray) -> ExpertTrajectory:
        message = ExpertTrajectory()
        message.stamp_us = source.stamp_us
        message.run_id = source.run_id
        message.scene_seed = source.scene_seed
        message.frame_index = source.frame_index
        message.frame_id = FRAME_ID
        message.model_version = MODEL_VERSION
        message.action = self.task.action
        message.target_attribute = self.task.target_attribute
        message.desired_distance_m = self.task.desired_distance_m
        message.selected_entity_id = ""
        message.dt = DT_SEC
        message.horizon = HORIZON
        message.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
        message.safe_stop = True
        message.valid = False
        message.detail = "UNINITIALIZED"
        return message

    def _publish_invalid(
        self,
        source: UEEntityArray,
        detail: str,
        *,
        input_ready: bool,
    ) -> None:
        message = self._message(source)
        message.detail = detail
        self.publisher.publish(message)
        self.input_ready = input_ready
        self.output_valid = False
        self.module_state = ModuleStatus.DEGRADED
        self.detail = detail
        self.get_logger().warning(detail)

    def on_entities(self, source: UEEntityArray) -> None:
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
        if source.frame_id != FRAME_ID:
            self._publish_invalid(
                source,
                f"INVALID_FRAME: expected {FRAME_ID}, got "
                f"{source.frame_id!r}",
                input_ready=True,
            )
            return

        try:
            result = generate_expert_trajectory(
                self.task,
                source.entities,
                max_speed_mps=self.max_speed_mps,
            )
        except (ExpertTrajectoryError, ValueError) as exc:
            self._publish_invalid(
                source,
                f"{type(exc).__name__.upper()}:{exc}",
                input_ready=True,
            )
            return
        except Exception as exc:
            self._publish_invalid(
                source,
                f"UNEXPECTED_EXPERT_ERROR:{type(exc).__name__}:{exc}",
                input_ready=True,
            )
            return

        message = self._message(source)
        message.selected_entity_id = result.selected_entity_id
        message.delta_p_xy = list(result.delta_p_xy)
        message.safe_stop = result.safe_stop
        message.valid = True
        message.detail = result.detail
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = True
        self.module_state = ModuleStatus.READY
        self.detail = result.detail

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
    node = ExpertTrajectoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
