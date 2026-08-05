"""Publish one UE5-only displacement from each fresh expert action."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    ExpertTrajectory,
    ModuleStatus,
    UEKinematicSetpoint,
)

from .kinematic_executor import (
    DEFAULT_MAX_STEP_M,
    expert_source_identity,
    first_step_from_expert,
    invalid_hold,
)
from .trajectory_contract import FRAME_ID


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
)


def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000


class ExpertKinematicExecutorNode(Node):
    """Rate-owning adapter from one expert action to one UE5 setpoint."""

    def __init__(self) -> None:
        super().__init__("expert_kinematic_executor")
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 5.0).value
        )
        self.source_timeout_sec = float(
            self.declare_parameter("source_timeout_sec", 0.5).value
        )
        self.max_step_m = float(
            self.declare_parameter(
                "max_step_m", DEFAULT_MAX_STEP_M
            ).value
        )
        self.start_delay_sec = float(
            self.declare_parameter("start_delay_sec", 0.0).value
        )
        if not math.isfinite(self.publish_rate_hz) or self.publish_rate_hz <= 0:
            raise ValueError("publish_rate_hz must be positive and finite")
        if not math.isfinite(self.start_delay_sec) or self.start_delay_sec < 0:
            raise ValueError("start_delay_sec must be non-negative and finite")
        if (
            not math.isfinite(self.source_timeout_sec)
            or self.source_timeout_sec <= 0
        ):
            raise ValueError("source_timeout_sec must be positive and finite")
        if not math.isfinite(self.max_step_m) or self.max_step_m <= 0:
            raise ValueError("max_step_m must be positive and finite")

        self.publisher = self.create_publisher(
            UEKinematicSetpoint,
            "/ue/kinematic_setpoint",
            RELIABLE_QOS,
        )
        self.status_pub = self.create_publisher(
            ModuleStatus,
            "/system/module_status",
            RELIABLE_QOS,
        )
        self.subscription = self.create_subscription(
            ExpertTrajectory,
            "/vla/expert_trajectory",
            self.on_expert,
            RELIABLE_QOS,
        )
        self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_latest_once,
        )
        self.create_timer(1.0, self.publish_status)

        self.latest: ExpertTrajectory | None = None
        self.start_monotonic = time.monotonic()
        self.latest_received_monotonic = 0.0
        self.last_source_identity: tuple[str, int, int, int] | None = None
        self.stale_reported_identity: tuple[str, int, int, int] | None = None
        self.sequence = 0
        self.module_state = ModuleStatus.READY
        self.input_ready = False
        self.output_valid = False
        self.detail = "waiting for /vla/expert_trajectory"
        self.get_logger().info(
            "UE5 kinematic mode: "
            f"rate={self.publish_rate_hz:.3f} Hz "
            f"timeout={self.source_timeout_sec:.3f} s "
            f"max_step={self.max_step_m:.3f} m"
        )

    @staticmethod
    def identity(source: ExpertTrajectory) -> tuple[str, int, int, int]:
        return expert_source_identity(source)

    def on_expert(self, source: ExpertTrajectory) -> None:
        self.latest = source
        self.latest_received_monotonic = time.monotonic()
        self.input_ready = True

    def make_message(
        self,
        source: ExpertTrajectory,
        step,
    ) -> UEKinematicSetpoint:
        message = UEKinematicSetpoint()
        message.stamp_us = now_us(self)
        message.source_stamp_us = source.stamp_us
        message.run_id = source.run_id
        message.scene_seed = source.scene_seed
        message.source_frame_index = source.frame_index
        message.sequence = self.sequence
        message.frame_id = FRAME_ID
        message.source_model_version = source.model_version
        message.step_dt = step.step_dt
        message.delta_x_m = step.delta_x_m
        message.delta_y_m = step.delta_y_m
        message.hold_position = step.hold_position
        message.valid = step.valid
        message.reason = step.reason
        return message

    def publish_step(self, source: ExpertTrajectory, step) -> None:
        message = self.make_message(source, step)
        self.publisher.publish(message)
        self.sequence += 1
        self.output_valid = step.valid
        self.module_state = (
            ModuleStatus.READY if step.valid else ModuleStatus.DEGRADED
        )
        self.detail = (
            f"{step.reason};source_frame={source.frame_index};"
            f"sequence={message.sequence};"
            f"dx={step.delta_x_m:.3f};dy={step.delta_y_m:.3f}"
        )

    def publish_latest_once(self) -> None:
        # Collection-time start delay: keeps the ASV stationary for
        # start_delay_sec while the UE5 targets hold their spawn positions
        # (matching -SineDelay), so the recorded frames include the static
        # initial geometry the online loop sees during its own startup.
        if time.monotonic() - self.start_monotonic < self.start_delay_sec:
            return

        source = self.latest
        if source is None:
            return

        identity = self.identity(source)
        age_sec = time.monotonic() - self.latest_received_monotonic
        if age_sec > self.source_timeout_sec:
            if self.stale_reported_identity == identity:
                return
            self.publish_step(
                source,
                invalid_hold(
                    f"STALE_EXPERT:{age_sec:.3f}s",
                    step_dt=float(source.dt),
                ),
            )
            self.stale_reported_identity = identity
            return

        if self.last_source_identity == identity:
            return

        step = first_step_from_expert(
            source,
            max_step_m=self.max_step_m,
        )
        self.publish_step(source, step)
        self.last_source_identity = identity
        self.stale_reported_identity = None

    def publish_status(self) -> None:
        message = ModuleStatus()
        message.stamp_us = now_us(self)
        message.run_id = self.latest.run_id if self.latest else ""
        message.module_name = self.get_name()
        message.state = self.module_state
        message.alive = True
        message.input_ready = self.input_ready
        message.output_valid = self.output_valid
        message.detail = self.detail
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExpertKinematicExecutorNode()
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
