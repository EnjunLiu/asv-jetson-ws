"""ROS adapter for the geometry-only temporal entity tracker."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import UEEntity, UEEntityArray

from .temporal_entity_tracker import (
    FrameMetadata,
    GeometryObservation,
    TemporalEntityTracker,
    TemporalEntityTrackerError,
)


RELIABLE_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


class TemporalEntityTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("temporal_entity_tracker")
        self.tracker = TemporalEntityTracker(
            ttl_frames=int(self.declare_parameter("ttl_frames", 2).value),
            ttl_sec=float(self.declare_parameter("ttl_sec", 0.5).value),
            velocity_filter=str(
                self.declare_parameter("velocity_filter", "ema").value
            ),
            alpha=float(self.declare_parameter("alpha", 0.6).value),
            beta=float(self.declare_parameter("beta", 0.85).value),
        )
        self.publisher = self.create_publisher(
            UEEntityArray, "/vla/tracked_entities", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            UEEntityArray,
            "/vla/perceived_entities",
            self.on_entities,
            RELIABLE_QOS,
        )

    def on_entities(self, source: UEEntityArray) -> None:
        output = UEEntityArray()
        output.stamp_us = int(source.stamp_us)
        output.run_id = str(source.run_id)
        output.scene_seed = int(source.scene_seed)
        output.frame_index = int(source.frame_index)
        output.frame_id = "base_link"
        output.source = "temporal_tracker"
        output.instruction_id = str(source.instruction_id)
        output.instruction = str(source.instruction)
        if not source.valid or not source.run_id.strip():
            output.valid = False
            output.detail = f"INVALID_SOURCE:{source.detail}"
            self.publisher.publish(output)
            return
        try:
            frame = FrameMetadata(
                run_id=source.run_id,
                scene_seed=source.scene_seed,
                frame_index=source.frame_index,
                stamp_us=source.stamp_us,
            )
            observations = [
                GeometryObservation(
                    entity_id=entity.entity_id,
                    relative_x=entity.relative_x,
                    relative_y=entity.relative_y,
                    relative_z=entity.relative_z,
                    class_name=entity.class_name,
                    color=entity.color,
                    is_target=entity.is_target,
                    visible=entity.visible,
                    bbox=(
                        entity.bbox_x_min,
                        entity.bbox_y_min,
                        entity.bbox_x_max,
                        entity.bbox_y_max,
                    )
                    if entity.bbox_valid
                    else None,
                    confidence=entity.confidence,
                    run_id=source.run_id,
                    scene_seed=source.scene_seed,
                    frame_index=source.frame_index,
                    stamp_us=source.stamp_us,
                )
                for entity in source.entities
                if entity.valid and entity.visible
            ]
            tracked = self.tracker.update(observations, frame=frame)
        except (TemporalEntityTrackerError, ValueError) as exc:
            output.valid = False
            output.detail = f"TRACKER_ERROR:{type(exc).__name__}:{exc}"
            self.publisher.publish(output)
            return

        for item in tracked:
            message = UEEntity()
            for field, value in item.as_ue_entity_kwargs().items():
                setattr(message, field, value)
            output.entities.append(message)
        output.valid = True
        output.detail = f"OK:tracked={len(output.entities)}"
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TemporalEntityTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
