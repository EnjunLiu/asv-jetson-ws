"""ROS acceptance probe for the Day 9 expert-label node."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    ExpertTrajectory,
    UEEntity,
    UEEntityArray,
)

from .expert_trajectory import MODEL_VERSION
from .trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, HORIZON


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
)


class ExpertTrajectoryProbe(Node):
    def __init__(self) -> None:
        super().__init__("expert_trajectory_probe")
        self.publisher = self.create_publisher(
            UEEntityArray, "/ue/entities", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            ExpertTrajectory,
            "/vla/expert_trajectory",
            self.on_trajectory,
            RELIABLE_QOS,
        )
        self.started_at = time.monotonic()
        self.last_publish_at = 0.0
        self.phase = 0
        self.errors: list[str] = []
        self.exit_code: int | None = None
        self.create_timer(0.1, self.on_timer)

    @staticmethod
    def _source(
        frame_index: int,
        *,
        valid: bool,
    ) -> UEEntityArray:
        source = UEEntityArray()
        source.stamp_us = 9_000_000 + frame_index
        source.run_id = "day9-probe"
        source.scene_seed = 9009
        source.frame_index = frame_index
        source.frame_id = FRAME_ID
        source.valid = valid
        source.detail = "ok" if valid else "synthetic invalid source"
        if valid:
            target = UEEntity()
            target.entity_id = "red_target"
            target.class_name = "boat"
            target.color = "red"
            target.is_target = True
            target.visible = True
            target.relative_x = 8.0
            target.relative_y = 2.0
            target.relative_z = 0.0
            target.relative_velocity_x = 0.2
            target.relative_velocity_y = 0.0
            target.relative_velocity_z = 0.0
            target.valid = True
            source.entities.append(target)
        return source

    def add_error(self, error: str) -> None:
        if error not in self.errors:
            self.errors.append(error)

    def on_timer(self) -> None:
        if self.exit_code is not None:
            return
        now = time.monotonic()
        if now - self.started_at > 15.0:
            self.finish(["timeout"])
            return
        if self.publisher.get_subscription_count() < 1:
            return
        if now - self.last_publish_at < 0.25:
            return
        if self.phase == 0:
            self.publisher.publish(self._source(101, valid=True))
        elif self.phase == 1:
            self.publisher.publish(self._source(102, valid=False))
        self.last_publish_at = now

    def _check_identity(
        self,
        message: ExpertTrajectory,
        frame_index: int,
    ) -> None:
        expected_stamp = 9_000_000 + frame_index
        if (
            message.stamp_us != expected_stamp
            or message.run_id != "day9-probe"
            or message.scene_seed != 9009
            or message.frame_index != frame_index
            or message.frame_id != FRAME_ID
        ):
            self.add_error(f"frame identity mismatch for {frame_index}")
        if (
            message.model_version != MODEL_VERSION
            or not math.isclose(
                message.dt, DT_SEC, rel_tol=0.0, abs_tol=1.0e-6
            )
            or message.horizon != HORIZON
            or len(message.delta_p_xy) != HORIZON * ACTION_DIM
        ):
            self.add_error(f"trajectory contract mismatch for {frame_index}")
        if not all(math.isfinite(value) for value in message.delta_p_xy):
            self.add_error(f"NaN/Inf for {frame_index}")

    def on_trajectory(self, message: ExpertTrajectory) -> None:
        if self.exit_code is not None:
            return
        if message.frame_index == 101 and self.phase == 0:
            self._check_identity(message, 101)
            if (
                not message.valid
                or message.safe_stop
                or message.action != "follow"
                or message.target_attribute != "color:red"
                or not math.isclose(
                    message.desired_distance_m, 3.0, abs_tol=1.0e-6
                )
                or message.selected_entity_id != "red_target"
                or message.delta_p_xy[-2] <= 0.0
                or message.delta_p_xy[-1] <= 0.0
            ):
                self.add_error(
                    f"valid FOLLOW label mismatch: {message.detail}"
                )
            self.phase = 1
            self.last_publish_at = 0.0
            return
        if message.frame_index == 102 and self.phase == 1:
            self._check_identity(message, 102)
            if (
                message.valid
                or not message.safe_stop
                or any(message.delta_p_xy)
                or "INVALID_SOURCE" not in message.detail
            ):
                self.add_error(
                    f"invalid-source fail-closed mismatch: {message.detail}"
                )
            self.phase = 2
            self.finish([])

    def finish(self, extra_errors: list[str]) -> None:
        if self.exit_code is not None:
            return
        for error in extra_errors:
            self.add_error(error)
        if self.errors:
            self.exit_code = 1
            print(
                "DAY9_EXPERT_ROS_FAIL " + "; ".join(self.errors),
                flush=True,
            )
        else:
            self.exit_code = 0
            print(
                "DAY9_EXPERT_ROS_PASS "
                f"shape={HORIZON}x{ACTION_DIM} "
                "target=red_target invalid_source=fail_closed "
                "topic=/vla/expert_trajectory",
                flush=True,
            )


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ExpertTrajectoryProbe()
    try:
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        if node.exit_code is None:
            node.exit_code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return node.exit_code if node.exit_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
