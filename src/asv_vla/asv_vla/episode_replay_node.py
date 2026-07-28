"""Replay a validated Day 8 episode onto the original UE5 ROS topics."""

from __future__ import annotations

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
from std_msgs.msg import Bool, String

from asv_jetson_interfaces.msg import (
    CameraFrame,
    UEASVState,
    UEEntity,
    UEEntityArray,
)

from .episode import EpisodeError, evaluate_episode, load_episode_records


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
)
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=4,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class EpisodeReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("day8_episode_replay")
        self.episode_dir = Path(
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
        self.rate_hz = (
            self.declare_parameter("rate_hz", 2.0)
            .get_parameter_value()
            .double_value
        )
        self.start_delay_sec = (
            self.declare_parameter("start_delay_sec", 5.0)
            .get_parameter_value()
            .double_value
        )
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if self.start_delay_sec < 0.0:
            raise ValueError("start_delay_sec must not be negative")

        report = evaluate_episode(
            self.episode_dir, min_frames=1, write_report=True
        )
        if not report["passed"]:
            raise EpisodeError(
                "episode quality gate failed: " + "; ".join(report["errors"])
            )
        self.records = load_episode_records(self.episode_dir)
        self.index = 0
        self.started_at = time.monotonic()
        self.completed = False

        self.task_pub = self.create_publisher(String, "/task/text", LATCHED_QOS)
        self.state_pub = self.create_publisher(
            UEASVState, "/ue/asv_state", RELIABLE_QOS
        )
        self.entity_pub = self.create_publisher(
            UEEntityArray, "/ue/entities", RELIABLE_QOS
        )
        self.camera_pub = self.create_publisher(
            CameraFrame, "/ue/camera_frame", SENSOR_QOS
        )
        self.complete_pub = self.create_publisher(
            Bool, "/day8/replay_complete", LATCHED_QOS
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.get_logger().info(
            "DAY8_REPLAY_READY "
            f"episode={self.episode_dir.resolve()} "
            f"frames={len(self.records)} rate_hz={self.rate_hz:.3f} "
            f"start_delay_sec={self.start_delay_sec:.3f}"
        )

    @staticmethod
    def _state(record: dict) -> UEASVState:
        source = record["ego"]
        message = UEASVState()
        message.stamp_us = record["stamp_us"]
        message.run_id = record["run_id"]
        message.scene_seed = record["scene_seed"]
        message.frame_index = record["frame_index"]
        message.simulation_time = source["simulation_time_s"]
        (
            message.position_x,
            message.position_y,
            message.position_z,
        ) = source["position_m"]
        message.roll, message.pitch, message.yaw = source["rpy_ue_rad"]
        message.surge_velocity = source["surge_velocity_mps"]
        message.yaw_rate = source["yaw_rate_radps"]
        message.valid = source["valid"]
        return message

    @staticmethod
    def _entities(record: dict) -> UEEntityArray:
        source = record["entities"]
        message = UEEntityArray()
        message.stamp_us = record["stamp_us"]
        message.run_id = record["run_id"]
        message.scene_seed = record["scene_seed"]
        message.frame_index = record["frame_index"]
        message.frame_id = source["frame_id"]
        message.valid = source["valid"]
        message.detail = "ok; Day 8 episode replay"
        for item in source["items"]:
            entity = UEEntity()
            entity.entity_id = item["entity_id"]
            entity.class_name = item["class_name"]
            entity.color = item["color"]
            entity.is_target = item["is_target"]
            entity.visible = item["visible"]
            (
                entity.relative_x,
                entity.relative_y,
                entity.relative_z,
            ) = item["relative_position_m"]
            (
                entity.relative_velocity_x,
                entity.relative_velocity_y,
                entity.relative_velocity_z,
            ) = item["relative_velocity_mps"]
            entity.valid = item["valid"]
            message.entities.append(entity)
        return message

    def _camera(self, record: dict) -> CameraFrame:
        source = record["camera"]
        message = CameraFrame()
        message.stamp_us = record["stamp_us"]
        message.run_id = record["run_id"]
        message.scene_seed = record["scene_seed"]
        message.frame_index = record["frame_index"]
        message.encoding = source["encoding"]
        message.data = list(
            (self.episode_dir / source["image_path"]).read_bytes()
        )
        message.valid = source["valid"]
        return message

    def on_timer(self) -> None:
        if self.completed:
            return
        if time.monotonic() - self.started_at < self.start_delay_sec:
            return
        if self.index >= len(self.records):
            self.completed = True
            complete = Bool()
            complete.data = True
            self.complete_pub.publish(complete)
            self.get_logger().info(
                f"DAY8_REPLAY_COMPLETE frames={len(self.records)}"
            )
            return

        record = self.records[self.index]
        if self.index == 0:
            task = String()
            task.data = record["task"]["text"]
            self.task_pub.publish(task)

        # Entities precede the camera so the visual encoder can exact-match
        # without hitting its short frame wait timeout.
        self.entity_pub.publish(self._entities(record))
        self.state_pub.publish(self._state(record))
        self.camera_pub.publish(self._camera(record))
        self.index += 1
        if self.index == 1 or self.index % 10 == 0:
            self.get_logger().info(
                f"DAY8_REPLAYED frame={record['frame_index']} "
                f"count={self.index}/{len(self.records)}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeReplayNode()
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
