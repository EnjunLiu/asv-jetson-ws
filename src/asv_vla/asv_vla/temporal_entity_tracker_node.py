"""ROS adapter for the geometry-only temporal entity tracker."""

from __future__ import annotations

from dataclasses import replace
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import UEEntity, UEEntityArray

from .temporal_entity_tracker import (
    FrameMetadata,
    GeometryObservation,
    TemporalEntityTracker,
    TemporalEntityTrackerError,
    TrackedEntity,
)


RELIABLE_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
DEFAULT_DROPOUT_HOLD_FRAMES = 30
DEFAULT_DROPOUT_HOLD_SEC = 3.0


class _DropoutRecovery:
    """Bounded, identity-scoped recovery for short perception dropouts."""

    def __init__(
        self,
        *,
        dropout_hold_frames: int = DEFAULT_DROPOUT_HOLD_FRAMES,
        dropout_hold_sec: float = DEFAULT_DROPOUT_HOLD_SEC,
    ) -> None:
        if int(dropout_hold_frames) < 0:
            raise ValueError("dropout_hold_frames must be non-negative")
        if not math.isfinite(float(dropout_hold_sec)) or float(
            dropout_hold_sec
        ) <= 0.0:
            raise ValueError("dropout_hold_sec must be finite and positive")
        self.dropout_hold_frames = int(dropout_hold_frames)
        self.dropout_hold_sec = float(dropout_hold_sec)
        self._identity: tuple[str, int] | None = None
        self._tracks: dict[str, TrackedEntity] = {}
        self.last_predicted_ids: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str, int] | None:
        return self._identity

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._identity = None
        self._tracks.clear()
        self.last_predicted_ids = ()

    def update(
        self,
        observed: tuple[TrackedEntity, ...],
        *,
        frame: FrameMetadata,
    ) -> tuple[TrackedEntity, ...]:
        if not isinstance(frame, FrameMetadata):
            raise TypeError("frame must be FrameMetadata")
        if self._identity != frame.identity:
            self.reset()
            self._identity = frame.identity

        observed_ids = set()
        for item in observed:
            if (
                item.run_id != frame.run_id
                or item.scene_seed != frame.scene_seed
            ):
                raise TemporalEntityTrackerError(
                    "tracked entities must share the current run and scene"
                )
            observed_ids.add(item.entity_id)
            self._tracks[item.entity_id] = item

        predicted: list[TrackedEntity] = []
        expired: list[str] = []
        for entity_id in sorted(self._tracks):
            if entity_id in observed_ids:
                continue
            item = self._tracks[entity_id]
            frame_gap = frame.frame_index - item.frame_index
            elapsed_sec = (frame.stamp_us - item.stamp_us) / 1.0e6
            within_window = (
                frame_gap > 0
                and elapsed_sec >= 0.0
                and frame_gap <= self.dropout_hold_frames
                and elapsed_sec <= self.dropout_hold_sec
            )
            if not within_window:
                if (
                    frame_gap > self.dropout_hold_frames
                    or elapsed_sec > self.dropout_hold_sec
                ):
                    expired.append(entity_id)
                continue

            velocity = (
                item.velocity if item.velocity_valid else (0.0, 0.0, 0.0)
            )
            predicted_position = tuple(
                position + component * elapsed_sec
                for position, component in zip(item.position, velocity)
            )
            predicted.append(
                replace(
                    item,
                    relative_x=predicted_position[0],
                    relative_y=predicted_position[1],
                    relative_z=predicted_position[2],
                    frame_index=frame.frame_index,
                    stamp_us=frame.stamp_us,
                    frame_gap=frame_gap,
                    source="temporal_tracker",
                )
            )

        for entity_id in expired:
            self._tracks.pop(entity_id, None)
        self.last_predicted_ids = tuple(item.entity_id for item in predicted)
        return tuple(observed) + tuple(predicted)


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
        self.dropout_recovery = _DropoutRecovery(
            dropout_hold_frames=int(
                self.declare_parameter(
                    "dropout_hold_frames", DEFAULT_DROPOUT_HOLD_FRAMES
                ).value
            ),
            dropout_hold_sec=float(
                self.declare_parameter(
                    "dropout_hold_sec", DEFAULT_DROPOUT_HOLD_SEC
                ).value
            ),
        )
        self._message_count = 0
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
        self._message_count += 1
        self._source_entity_count = getattr(self, "_source_entity_count", 0)
        self._source_entity_count = sum(
            1
            for entity in source.entities
            if entity.valid and entity.visible
        )
        if self._message_count % 50 == 0:
            self.get_logger().info(
                f"TRACK_TRACE frame_index={int(source.frame_index)} "
                f"count={self._message_count} source_valid={source.valid} "
                f"source_entities={len(source.entities)} "
                f"visible_source={self._source_entity_count}"
            )
        output.stamp_us = int(source.stamp_us)
        output.run_id = str(source.run_id)
        output.scene_seed = int(source.scene_seed)
        output.frame_index = int(source.frame_index)
        output.frame_id = "base_link"
        output.source = "temporal_tracker"
        output.instruction_id = str(source.instruction_id)
        output.instruction = str(source.instruction)
        if not source.valid or not source.run_id.strip():
            self.tracker.reset()
            self.dropout_recovery.reset()
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
            tracked = self.dropout_recovery.update(tracked, frame=frame)
        except (TemporalEntityTrackerError, ValueError) as exc:
            self.tracker.reset()
            self.dropout_recovery.reset()
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
        predicted_ids = self.dropout_recovery.last_predicted_ids
        predicted_detail = ",".join(predicted_ids) if predicted_ids else "none"
        output.detail = (
            f"OK:tracked={len(output.entities)};"
            f"dropout_hold={len(predicted_ids)};"
            f"predicted_ids={predicted_detail}"
        )
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
