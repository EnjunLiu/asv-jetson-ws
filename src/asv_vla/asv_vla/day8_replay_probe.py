"""Acceptance probe for synchronized Day 8 multimodal ROS replay."""

from __future__ import annotations

import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool

from asv_jetson_interfaces.msg import (
    DecisionOutput,
    SelectedTrajectory,
    TaskFeatures,
    VisualFeatures,
)

from .episode import load_episode_records
from .trajectory_contract import is_day1_safe_stop


TOKEN_COUNT = 2
VISUAL_DIM = 576
MAX_ENTITIES = 16
ENTITY_DIM = 16


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
)
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def feature_key(message) -> tuple[str, int, int, int]:
    return (
        str(message.run_id),
        int(message.scene_seed),
        int(message.frame_index),
        int(message.stamp_us),
    )


class Day8ReplayProbe(Node):
    def __init__(self) -> None:
        super().__init__("day8_replay_probe")
        episode_dir = Path(
            self.declare_parameter(
                "episode_dir",
                str(
                    Path.home()
                    / "jetson_asv_ws"
                    / "artifacts"
                    / "day8_episode"
                    / "latest"
                ),
            )
            .get_parameter_value()
            .string_value
        ).expanduser()
        self.min_frames = int(
            self.declare_parameter("min_frames", 20)
            .get_parameter_value()
            .integer_value
        )
        self.timeout_sec = (
            self.declare_parameter("timeout_sec", 120.0)
            .get_parameter_value()
            .double_value
        )
        self.grace_sec = (
            self.declare_parameter("completion_grace_sec", 5.0)
            .get_parameter_value()
            .double_value
        )
        if self.min_frames < 1:
            raise ValueError("min_frames must be positive")

        self.expected_keys = {
            (
                record["run_id"],
                record["scene_seed"],
                record["frame_index"],
                record["stamp_us"],
            )
            for record in load_episode_records(episode_dir)
        }
        self.visual_keys: set[tuple[str, int, int, int]] = set()
        self.task_keys: set[tuple[str, int, int, int]] = set()
        self.errors: list[str] = []
        self.safe_stop_count = 0
        self.invalid_zero_count = 0
        self.replay_completed_at: float | None = None
        self.started_at = time.monotonic()
        self.exit_code: int | None = None

        self.create_subscription(
            VisualFeatures,
            "/vla/visual_features",
            self.on_visual,
            RELIABLE_QOS,
        )
        self.create_subscription(
            TaskFeatures,
            "/vla/task_features",
            self.on_task,
            RELIABLE_QOS,
        )
        self.create_subscription(
            SelectedTrajectory,
            "/vla/selected_trajectory",
            self.on_selected,
            RELIABLE_QOS,
        )
        self.create_subscription(
            DecisionOutput,
            "/decision/output",
            self.on_decision,
            RELIABLE_QOS,
        )
        self.create_subscription(
            Bool,
            "/day8/replay_complete",
            self.on_complete,
            LATCHED_QOS,
        )
        self.create_timer(0.2, self.on_timer)
        self.get_logger().info(
            f"DAY8_REPLAY_PROBE_READY expected={len(self.expected_keys)} "
            f"minimum={self.min_frames}"
        )

    def add_error(self, error: str) -> None:
        if error not in self.errors:
            self.errors.append(error)

    def on_visual(self, message: VisualFeatures) -> None:
        key = feature_key(message)
        if key not in self.expected_keys:
            self.add_error(f"unexpected visual frame key={key}")
            return
        if not message.valid:
            self.add_error(
                f"invalid visual frame={message.frame_index}:{message.detail}"
            )
            return
        if (
            message.token_count != TOKEN_COUNT
            or message.feature_dim != VISUAL_DIM
            or len(message.feature) != TOKEN_COUNT * VISUAL_DIM
            or list(message.mask) != [True] * TOKEN_COUNT
        ):
            self.add_error(
                f"visual shape/mask mismatch frame={message.frame_index}"
            )
            return
        if not all(math.isfinite(value) for value in message.feature):
            self.add_error(
                f"visual NaN/Inf frame={message.frame_index}"
            )
            return
        self.visual_keys.add(key)

    def on_task(self, message: TaskFeatures) -> None:
        key = feature_key(message)
        if key not in self.expected_keys:
            self.add_error(f"unexpected task frame key={key}")
            return
        if not message.valid:
            self.add_error(
                f"invalid task frame={message.frame_index}:{message.detail}"
            )
            return
        if (
            message.max_entities != MAX_ENTITIES
            or message.feature_dim != ENTITY_DIM
            or len(message.features) != MAX_ENTITIES * ENTITY_DIM
            or len(message.mask) != MAX_ENTITIES
            or len(message.entity_ids) != MAX_ENTITIES
        ):
            self.add_error(
                f"task shape/mask mismatch frame={message.frame_index}"
            )
            return
        if not all(math.isfinite(value) for value in message.features):
            self.add_error(f"task NaN/Inf frame={message.frame_index}")
            return
        self.task_keys.add(key)

    def on_selected(self, message: SelectedTrajectory) -> None:
        if is_day1_safe_stop(message):
            self.safe_stop_count += 1
        else:
            self.add_error("selected trajectory violated safe-stop contract")

    def on_decision(self, message: DecisionOutput) -> None:
        if (
            not message.valid
            and math.isclose(message.desired_x, 0.0, abs_tol=1e-9)
            and math.isclose(message.desired_y, 0.0, abs_tol=1e-9)
        ):
            self.invalid_zero_count += 1
        else:
            self.add_error("DecisionOutput is not invalid zero")

    def on_complete(self, message: Bool) -> None:
        if message.data and self.replay_completed_at is None:
            self.replay_completed_at = time.monotonic()

    def on_timer(self) -> None:
        now = time.monotonic()
        if now - self.started_at > self.timeout_sec:
            self.finish(["probe timeout"])
            return
        if (
            self.replay_completed_at is not None
            and now - self.replay_completed_at >= self.grace_sec
        ):
            self.finish([])

    def finish(self, extra_errors: list[str]) -> None:
        if self.exit_code is not None:
            return
        for error in extra_errors:
            self.add_error(error)
        matched = self.visual_keys & self.task_keys
        if len(matched) < self.min_frames:
            self.add_error(
                f"matched feature frames={len(matched)} "
                f"is below minimum={self.min_frames}"
            )
        if self.safe_stop_count < 1:
            self.add_error("no valid safe-stop SelectedTrajectory received")
        if self.invalid_zero_count < 1:
            self.add_error("no invalid-zero DecisionOutput received")

        if self.errors:
            self.exit_code = 1
            print("DAY8_REPLAY_FAIL " + "; ".join(self.errors), flush=True)
        else:
            self.exit_code = 0
            print(
                "DAY8_REPLAY_PASS "
                f"matched_frames={len(matched)} "
                f"visual_shape={TOKEN_COUNT}x{VISUAL_DIM} "
                f"task_shape={MAX_ENTITIES}x{ENTITY_DIM} "
                f"safe_stop_count={self.safe_stop_count} "
                f"invalid_zero_count={self.invalid_zero_count}",
                flush=True,
            )


def main(args=None) -> int:
    rclpy.init(args=args)
    node = Day8ReplayProbe()
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
