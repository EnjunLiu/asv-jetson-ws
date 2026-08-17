"""Pure temporal tracking for geometry-only entity observations.

The UE bridge can provide positions and semantics without a velocity field.
This module keeps that transport boundary honest: velocity is zero and marked
invalid until two time-ordered observations of the same entity are available.
No ROS imports are required, so the tracker can be tested independently of a
running graph and its records can be adapted to the ``Entity``
message shape by callers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from interfaces.msg import Entity, EntityArray


Position3 = tuple[float, float, float]
BBox4 = tuple[float, float, float, float]
VelocityFilter = Literal["none", "ema", "alpha_beta"]


class TemporalEntityTrackerError(ValueError):
    """Raised when a geometry frame cannot satisfy the tracker contract."""


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Identity and timing attached to one observation frame."""

    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int

    def __post_init__(self) -> None:
        run_id = str(self.run_id).strip()
        if not run_id:
            raise TemporalEntityTrackerError("run_id must not be empty")
        scene_seed = int(self.scene_seed)
        frame_index = int(self.frame_index)
        stamp_us = int(self.stamp_us)
        if frame_index < 0:
            raise TemporalEntityTrackerError("frame_index must be non-negative")
        if stamp_us < 0:
            raise TemporalEntityTrackerError("stamp_us must be non-negative")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "scene_seed", scene_seed)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "stamp_us", stamp_us)

    @property
    def identity(self) -> tuple[str, int]:
        return self.run_id, self.scene_seed


@dataclass(frozen=True, slots=True)
class GeometryObservation:
    """One entity observation containing geometry and semantic metadata."""

    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    class_name: str = ""
    color: str = ""
    is_target: bool = False
    visible: bool = True
    bbox: BBox4 | None = None
    confidence: float = 1.0
    run_id: str = ""
    scene_seed: int = 0
    frame_index: int = 0
    stamp_us: int = 0

    def __post_init__(self) -> None:
        entity_id = str(self.entity_id).strip()
        if not entity_id:
            raise TemporalEntityTrackerError("entity_id must not be empty")
        values = (
            float(self.relative_x),
            float(self.relative_y),
            float(self.relative_z),
        )
        if not all(math.isfinite(value) for value in values):
            raise TemporalEntityTrackerError(
                f"entity {entity_id!r} position must be finite"
            )
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise TemporalEntityTrackerError(
                f"entity {entity_id!r} confidence must be in [0, 1]"
            )
        bbox = None if self.bbox is None else tuple(float(v) for v in self.bbox)
        if bbox is not None and (
            len(bbox) != 4 or not all(math.isfinite(value) for value in bbox)
        ):
            raise TemporalEntityTrackerError(
                f"entity {entity_id!r} bbox must contain four finite values"
            )
        metadata = FrameMetadata(
            run_id=self.run_id,
            scene_seed=self.scene_seed,
            frame_index=self.frame_index,
            stamp_us=self.stamp_us,
        )
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "relative_x", values[0])
        object.__setattr__(self, "relative_y", values[1])
        object.__setattr__(self, "relative_z", values[2])
        object.__setattr__(self, "class_name", str(self.class_name))
        object.__setattr__(self, "color", str(self.color))
        object.__setattr__(self, "is_target", bool(self.is_target))
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "run_id", metadata.run_id)
        object.__setattr__(self, "scene_seed", metadata.scene_seed)
        object.__setattr__(self, "frame_index", metadata.frame_index)
        object.__setattr__(self, "stamp_us", metadata.stamp_us)

    @property
    def position(self) -> Position3:
        return self.relative_x, self.relative_y, self.relative_z

    @property
    def metadata(self) -> FrameMetadata:
        return FrameMetadata(
            self.run_id, self.scene_seed, self.frame_index, self.stamp_us
        )


@dataclass(frozen=True, slots=True)
class TrackedEntity:
    """Current geometry plus an explicitly validity-gated velocity estimate."""

    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    relative_velocity_x: float
    relative_velocity_y: float
    relative_velocity_z: float
    velocity_valid: bool
    class_name: str
    color: str
    is_target: bool
    visible: bool
    bbox: BBox4 | None
    confidence: float
    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int
    frame_gap: int = 0
    valid: bool = True
    source: str = "temporal_tracker"

    @property
    def position(self) -> Position3:
        return self.relative_x, self.relative_y, self.relative_z

    @property
    def velocity(self) -> Position3:
        return (
            self.relative_velocity_x,
            self.relative_velocity_y,
            self.relative_velocity_z,
        )

    def as_entity_kwargs(self) -> dict[str, object]:
        """Return fields accepted by the ``Entity`` message."""

        bbox = self.bbox or (0.0, 0.0, 0.0, 0.0)
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "color": self.color,
            "is_target": self.is_target,
            "visible": self.visible,
            "relative_x": self.relative_x,
            "relative_y": self.relative_y,
            "relative_z": self.relative_z,
            "relative_velocity_x": self.relative_velocity_x,
            "relative_velocity_y": self.relative_velocity_y,
            "relative_velocity_z": self.relative_velocity_z,
            "valid": self.valid,
            "source": self.source,
            "bbox_x_min": bbox[0],
            "bbox_y_min": bbox[1],
            "bbox_x_max": bbox[2],
            "bbox_y_max": bbox[3],
            "bbox_valid": self.bbox is not None,
            "confidence": self.confidence,
            "velocity_valid": self.velocity_valid,
        }

@dataclass(slots=True)
class _TrackState:
    position: Position3
    filter_position: Position3
    frame_index: int
    stamp_us: int
    velocity: Position3 = (0.0, 0.0, 0.0)
    velocity_valid: bool = False


class TemporalEntityTracker:
    """Track geometry observations and estimate velocity by finite difference."""

    def __init__(
        self,
        *,
        ttl_frames: int = 2,
        ttl_sec: float | None = None,
        velocity_filter: VelocityFilter = "none",
        alpha: float = 1.0,
        beta: float = 0.85,
    ) -> None:
        if ttl_frames < 0:
            raise ValueError("ttl_frames must be non-negative")
        if ttl_sec is not None and ttl_sec <= 0.0:
            raise ValueError("ttl_sec must be positive when provided")
        if velocity_filter not in {"none", "ema", "alpha_beta"}:
            raise ValueError("velocity_filter must be none, ema, or alpha_beta")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 < beta <= 2.0:
            raise ValueError("beta must be in (0, 2]")
        self.ttl_frames = int(ttl_frames)
        self.ttl_sec = None if ttl_sec is None else float(ttl_sec)
        self.velocity_filter = velocity_filter
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._tracks: dict[str, _TrackState] = {}
        self._identity: tuple[str, int] | None = None
        self._last_frame_index: int | None = None
        self._last_stamp_us: int | None = None

    @property
    def identity(self) -> tuple[str, int] | None:
        return self._identity

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._identity = None
        self._last_frame_index = None
        self._last_stamp_us = None

    def update(
        self,
        observations: Iterable[GeometryObservation],
        *,
        frame: FrameMetadata | None = None,
    ) -> tuple[TrackedEntity, ...]:
        """Consume one frame and return records for entities seen in it.

        A frame with no observations can be represented by passing ``frame``;
        this advances TTL bookkeeping without fabricating entity positions.
        Frames with a non-increasing index are ignored.  A timestamp regression
        is accepted as a geometry update but invalidates velocity for that
        frame, rather than guessing a time interval.
        """

        items = tuple(observations)
        metadata = self._metadata_for(items, frame)
        if any(item.metadata != metadata for item in items):
            raise TemporalEntityTrackerError(
                "all observations in a frame must share run/scene/frame/stamp"
            )
        ids = [item.entity_id for item in items]
        if len(ids) != len(set(ids)):
            raise TemporalEntityTrackerError("duplicate entity_id in frame")

        if self._identity != metadata.identity:
            self._tracks.clear()
            self._identity = metadata.identity
            self._last_frame_index = None
            self._last_stamp_us = None

        if (
            self._last_frame_index is not None
            and metadata.frame_index <= self._last_frame_index
        ):
            return ()

        monotonic_stamp = (
            self._last_stamp_us is None
            or metadata.stamp_us > self._last_stamp_us
        )
        self._expire_tracks(metadata)
        records = tuple(
            self._record_for(item, metadata, monotonic_stamp) for item in items
        )
        self._last_frame_index = metadata.frame_index
        self._last_stamp_us = metadata.stamp_us
        return records

    # Explicit alias makes the intended frame-processing operation discoverable.
    process_frame = update

    def _metadata_for(
        self,
        items: Sequence[GeometryObservation],
        frame: FrameMetadata | None,
    ) -> FrameMetadata:
        if frame is not None and not isinstance(frame, FrameMetadata):
            raise TypeError("frame must be FrameMetadata")
        if items:
            metadata = items[0].metadata
            if frame is not None and frame != metadata:
                raise TemporalEntityTrackerError(
                    "explicit frame metadata does not match observation"
                )
            return metadata
        if frame is None:
            raise TemporalEntityTrackerError(
                "an empty frame requires explicit FrameMetadata"
            )
        return frame

    def _expire_tracks(self, metadata: FrameMetadata) -> None:
        expired = []
        for entity_id, state in self._tracks.items():
            frame_gap = metadata.frame_index - state.frame_index
            time_gap = (metadata.stamp_us - state.stamp_us) / 1.0e6
            too_many_frames = frame_gap > self.ttl_frames
            too_long = self.ttl_sec is not None and time_gap > self.ttl_sec
            if too_many_frames or (time_gap >= 0.0 and too_long):
                expired.append(entity_id)
        for entity_id in expired:
            del self._tracks[entity_id]

    def _record_for(
        self,
        item: GeometryObservation,
        metadata: FrameMetadata,
        monotonic_stamp: bool,
    ) -> TrackedEntity:
        state = self._tracks.get(item.entity_id)
        velocity = (0.0, 0.0, 0.0)
        velocity_valid = False
        frame_gap = 0
        if state is not None:
            frame_gap = metadata.frame_index - state.frame_index
            dt_sec = (metadata.stamp_us - state.stamp_us) / 1.0e6
            if monotonic_stamp and frame_gap > 0 and dt_sec > 0.0:
                raw_velocity = tuple(
                    (current - previous) / dt_sec
                    for current, previous in zip(item.position, state.position)
                )
                velocity, filter_position = self._filter_velocity(
                    state, raw_velocity, dt_sec, item
                )
                velocity_valid = all(math.isfinite(value) for value in velocity)
                if not velocity_valid:
                    velocity = (0.0, 0.0, 0.0)
                    filter_position = item.position
            else:
                filter_position = item.position
        else:
            filter_position = item.position

        self._tracks[item.entity_id] = _TrackState(
            position=item.position,
            filter_position=filter_position,
            frame_index=metadata.frame_index,
            stamp_us=metadata.stamp_us,
            velocity=velocity,
            velocity_valid=velocity_valid,
        )
        return TrackedEntity(
            entity_id=item.entity_id,
            relative_x=item.relative_x,
            relative_y=item.relative_y,
            relative_z=item.relative_z,
            relative_velocity_x=velocity[0],
            relative_velocity_y=velocity[1],
            relative_velocity_z=velocity[2],
            velocity_valid=velocity_valid,
            class_name=item.class_name,
            color=item.color,
            is_target=item.is_target,
            visible=item.visible,
            bbox=item.bbox,
            confidence=item.confidence,
            run_id=metadata.run_id,
            scene_seed=metadata.scene_seed,
            frame_index=metadata.frame_index,
            stamp_us=metadata.stamp_us,
            frame_gap=frame_gap,
        )

    def _filter_velocity(
        self,
        state: _TrackState,
        raw_velocity: Position3,
        dt_sec: float,
        item: GeometryObservation,
    ) -> tuple[Position3, Position3]:
        if not state.velocity_valid or self.velocity_filter == "none":
            return raw_velocity, item.position
        if self.velocity_filter == "ema":
            return (
                tuple(
                    self.alpha * raw + (1.0 - self.alpha) * previous
                    for raw, previous in zip(raw_velocity, state.velocity)
                ),
                item.position,
            )

        # Alpha-beta: the position residual corrects the prior velocity.  The
        # reported position remains the actual geometry observation.
        residual = tuple(
            current - (previous + velocity * dt_sec)
            for current, previous, velocity in zip(
                item.position, state.filter_position, state.velocity
            )
        )
        predicted_position = tuple(
            previous + velocity * dt_sec
            for previous, velocity in zip(state.filter_position, state.velocity)
        )
        corrected_position = tuple(
            predicted + self.alpha * error
            for predicted, error in zip(predicted_position, residual)
        )
        return (
            tuple(
                velocity + self.beta * error / dt_sec
                for velocity, error in zip(state.velocity, residual)
            ),
            corrected_position,
        )


__all__ = [
    "FrameMetadata",
    "GeometryObservation",
    "TemporalEntityTracker",
    "TemporalEntityTrackerError",
    "TrackedEntity",
]


RELIABLE_QOS = (
    QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    if QoSProfile is not None
    else None
)
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
            if item.run_id != frame.run_id or item.scene_seed != frame.scene_seed:
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

            velocity = item.velocity if item.velocity_valid else (0.0, 0.0, 0.0)
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
            EntityArray, "/vla/tracked_entities", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            EntityArray,
            "/vla/perceived_entities",
            self.on_entities,
            RELIABLE_QOS,
        )

    def on_entities(self, source: EntityArray) -> None:
        output = EntityArray()
        self._message_count += 1
        self._source_entity_count = getattr(self, "_source_entity_count", 0)
        self._source_entity_count = sum(
            1 for entity in source.entities if entity.valid and entity.visible
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
            message = Entity()
            for field, value in item.as_entity_kwargs().items():
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
