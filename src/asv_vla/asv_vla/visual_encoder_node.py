from __future__ import annotations

from collections import OrderedDict
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    CameraFrame,
    ModuleStatus,
    UEEntityArray,
    VisualFeatures,
)

from .visual_encoder import (
    BACKBONE_ID,
    DEFAULT_LOW_LIGHT_BRIGHTNESS,
    DEFAULT_LOW_LIGHT_CONTRAST,
    DEFAULT_LOW_LIGHT_GAMMA,
    FEATURE_DIM,
    TOKEN_COUNT,
    CameraProfile,
    FrozenMobileNetEncoder,
    InvalidImageError,
    TargetProjectionError,
    VisualEncoderError,
    decode_camera_image,
    enhance_low_light_image,
    make_target_crop,
)
from .task_entity_tensor import MAX_ENTITIES, build_entity_tensor


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


# Keep visual/entity matching aligned with the policy cache.  The Jetson
# visual encoder can lag the tracked-entity stream by more than six frames
# during CUDA work; rejecting that gap starves the policy with INVALID_MODALITY.
DEFAULT_SYNC_FRAME_TOLERANCE = 12
MAX_SYNC_FRAME_TOLERANCE = 12
ALIGNMENT_DIAGNOSTIC_LIMIT = 8
FrameKey = tuple[str, int, int]


class FrameSyncCache:
    """Bounded, same-run/scene cache for camera/entity frame pairing."""

    def __init__(
        self,
        *,
        cache_size: int,
        frame_tolerance: int,
        ttl_sec: float,
    ) -> None:
        if int(cache_size) <= 0:
            raise ValueError("cache_size must be positive")
        if not 0 <= int(frame_tolerance) <= MAX_SYNC_FRAME_TOLERANCE:
            raise ValueError(
                "frame_tolerance must be between 0 and "
                f"{MAX_SYNC_FRAME_TOLERANCE}"
            )
        if not math.isfinite(float(ttl_sec)) or float(ttl_sec) <= 0.0:
            raise ValueError("ttl_sec must be finite and positive")
        self.cache_size = int(cache_size)
        self.frame_tolerance = int(frame_tolerance)
        self.ttl_sec = float(ttl_sec)
        self.frames: OrderedDict[FrameKey, tuple[object, float]] = OrderedDict()
        self.entities: OrderedDict[FrameKey, tuple[object, float]] = OrderedDict()

    @staticmethod
    def key_for(message: object) -> FrameKey:
        return (
            str(message.run_id),
            int(message.scene_seed),
            int(message.frame_index),
        )

    def put_frame(
        self, frame: object, received_at: float | None = None
    ) -> tuple[object, ...]:
        return self._put(self.frames, frame, received_at)

    def put_entities(
        self, entities: object, received_at: float | None = None
    ) -> None:
        self._put(self.entities, entities, received_at)

    def _put(
        self,
        cache: OrderedDict[FrameKey, tuple[object, float]],
        message: object,
        received_at: float | None,
    ) -> tuple[object, ...]:
        key = self.key_for(message)
        cache[key] = (
            message,
            time.monotonic() if received_at is None else float(received_at),
        )
        cache.move_to_end(key)
        evicted: list[object] = []
        while len(cache) > self.cache_size:
            _, (old_message, _) = cache.popitem(last=False)
            evicted.append(old_message)
        return tuple(evicted)

    def match_for_frame(
        self, key: FrameKey, now: float | None = None
    ) -> tuple[object, object, int] | None:
        frame_entry = self.frames.get(key)
        if frame_entry is None:
            return None
        current = time.monotonic() if now is None else float(now)
        if not self._fresh(self.frames, key, current):
            self.frames.pop(key, None)
            return None
        entity_key = self._nearest_key(self.entities, key, current)
        if entity_key is None:
            return None
        frame, _ = self.frames.pop(key)
        entities, _ = self.entities.pop(entity_key)
        return frame, entities, int(entity_key[2]) - int(key[2])

    def match_for_entities(
        self, key: FrameKey, now: float | None = None
    ) -> tuple[object, object, int] | None:
        entity_entry = self.entities.get(key)
        if entity_entry is None:
            return None
        current = time.monotonic() if now is None else float(now)
        if not self._fresh(self.entities, key, current):
            self.entities.pop(key, None)
            return None
        frame_key = self._nearest_key(self.frames, key, current)
        if frame_key is None:
            return None
        frame, _ = self.frames.pop(frame_key)
        entities, _ = self.entities.pop(key)
        return frame, entities, int(key[2]) - int(frame_key[2])

    def _nearest_key(
        self,
        cache: OrderedDict[FrameKey, tuple[object, float]],
        key: FrameKey,
        current: float,
    ) -> FrameKey | None:
        exact = cache.get(key)
        if exact is not None:
            if self._fresh(cache, key, current):
                return key
            cache.pop(key, None)

        candidates: list[tuple[int, int, FrameKey]] = []
        for candidate, (_, received_at) in cache.items():
            if candidate[0] != key[0] or candidate[1] != key[1]:
                continue
            if current - received_at >= self.ttl_sec:
                continue
            distance = abs(int(candidate[2]) - int(key[2]))
            if distance <= self.frame_tolerance:
                candidates.append((distance, int(candidate[2]), candidate))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    def _fresh(
        self,
        cache: OrderedDict[FrameKey, tuple[object, float]],
        key: FrameKey,
        current: float,
    ) -> bool:
        entry = cache.get(key)
        return entry is not None and current - entry[1] < self.ttl_sec

    def expire(self, now: float | None = None) -> tuple[object, ...]:
        current = time.monotonic() if now is None else float(now)
        expired_frames: list[object] = []
        for cache, is_frame_cache in (
            (self.frames, True),
            (self.entities, False),
        ):
            for key, (_, received_at) in tuple(cache.items()):
                if current - received_at < self.ttl_sec:
                    continue
                message, _ = cache.pop(key)
                if is_frame_cache:
                    expired_frames.append(message)
        return tuple(expired_frames)


def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000


class VisualEncoderNode(Node):
    def __init__(self) -> None:
        super().__init__("visual_encoder")
        self.status_run_id = (
            self.declare_parameter("run_id", "visual-encoder")
            .get_parameter_value()
            .string_value
        )
        device = (
            self.declare_parameter("device", "cuda")
            .get_parameter_value()
            .string_value
        )
        self.entity_wait_sec = (
            self.declare_parameter("entity_wait_sec", 0.25)
            .get_parameter_value()
            .double_value
        )
        self.sync_frame_tolerance = int(
            self.declare_parameter(
                "sync_frame_tolerance", DEFAULT_SYNC_FRAME_TOLERANCE
            )
            .get_parameter_value()
            .integer_value
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
        self.image_preprocess_enabled = bool(
            self.declare_parameter("image_preprocess_enabled", False).value
        )
        self.image_preprocess_gamma = float(
            self.declare_parameter(
                "image_preprocess_gamma", DEFAULT_LOW_LIGHT_GAMMA
            ).value
        )
        self.image_preprocess_brightness = float(
            self.declare_parameter(
                "image_preprocess_brightness", DEFAULT_LOW_LIGHT_BRIGHTNESS
            ).value
        )
        self.image_preprocess_contrast = float(
            self.declare_parameter(
                "image_preprocess_contrast", DEFAULT_LOW_LIGHT_CONTRAST
            ).value
        )
        cache_size = (
            self.declare_parameter("sync_cache_size", 16)
            .get_parameter_value()
            .integer_value
        )
        if self.entity_wait_sec <= 0.0:
            raise ValueError("entity_wait_sec must be positive")
        if not 0 <= self.sync_frame_tolerance <= MAX_SYNC_FRAME_TOLERANCE:
            raise ValueError(
                "sync_frame_tolerance must be between 0 and "
                f"{MAX_SYNC_FRAME_TOLERANCE}"
            )
        if cache_size <= 0:
            raise ValueError("sync_cache_size must be positive")
        self.cache_size = int(cache_size)

        self.profile = CameraProfile(
            width=self.declare_parameter(
                "image_width", 1280
            ).get_parameter_value().integer_value,
            height=self.declare_parameter(
                "image_height", 720
            ).get_parameter_value().integer_value,
            horizontal_fov_deg=self.declare_parameter(
                "horizontal_fov_deg", 90.0
            ).get_parameter_value().double_value,
            mount_x_m=self.declare_parameter(
                "camera_mount_x_m", 0.42
            ).get_parameter_value().double_value,
            mount_y_m=self.declare_parameter(
                "camera_mount_y_m", 0.0
            ).get_parameter_value().double_value,
            mount_z_m=self.declare_parameter(
                "camera_mount_z_m", 0.20
            ).get_parameter_value().double_value,
            pitch_deg=self.declare_parameter(
                "camera_pitch_deg", -5.0
            ).get_parameter_value().double_value,
            crop_size_px=self.declare_parameter(
                "target_crop_size_px", 224
            ).get_parameter_value().integer_value,
        )

        self.publisher = self.create_publisher(
            VisualFeatures, "/vla/visual_features", RELIABLE_QOS
        )
        self.status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", RELIABLE_QOS
        )
        self.entity_subscription = self.create_subscription(
            UEEntityArray, self.entities_topic, self.on_entities, RELIABLE_QOS
        )
        self.frame_subscription = self.create_subscription(
            CameraFrame, "/ue/camera_frame", self.on_frame, SENSOR_QOS
        )
        self.create_timer(0.05, self.flush_expired_frames)
        self.create_timer(1.0, self.publish_status)

        self._sync_cache = FrameSyncCache(
            cache_size=self.cache_size,
            frame_tolerance=self.sync_frame_tolerance,
            ttl_sec=self.entity_wait_sec,
        )
        self.entities = self._sync_cache.entities
        self.frames = self._sync_cache.frames
        self._process_count = 0
        self._encode_ms = -1.0
        self._alignment_diagnostics: list[int] = []
        self._alignment_log_count = 0
        self.encoder = None
        self.module_state = ModuleStatus.STARTING
        self.input_ready = False
        self.output_valid = False
        self.detail = "loading frozen MobileNetV3-small visual encoder"
        try:
            self.encoder = FrozenMobileNetEncoder(device=device)
        except Exception as exc:
            self.module_state = ModuleStatus.ERROR
            self.detail = f"MODEL_LOAD_ERROR:{type(exc).__name__}:{exc}"
            self.get_logger().error(self.detail)
        else:
            self.module_state = ModuleStatus.READY
            self.detail = (
                f"ready backbone={BACKBONE_ID} device={device} "
                f"tokens={1 + MAX_ENTITIES} (1 global + "
                f"{MAX_ENTITIES} entity slots)x{FEATURE_DIM}; "
                f"low_light={self.image_preprocess_enabled} "
                f"gamma={self.image_preprocess_gamma:.3f} "
                f"brightness={self.image_preprocess_brightness:.3f} "
                f"contrast={self.image_preprocess_contrast:.3f}"
            )
            self.get_logger().info(
                f"{self.detail}; entities_topic={self.entities_topic}"
            )

    @staticmethod
    def _key(run_id: str, scene_seed: int, frame_index: int) -> FrameKey:
        return str(run_id), int(scene_seed), int(frame_index)

    def _new_message(
        self, source: object, token_count: int = TOKEN_COUNT
    ) -> VisualFeatures:
        message = VisualFeatures()
        message.stamp_us = source.stamp_us
        message.run_id = source.run_id
        message.scene_seed = source.scene_seed
        message.frame_index = source.frame_index
        message.backbone = BACKBONE_ID
        message.token_count = token_count
        message.feature_dim = FEATURE_DIM
        message.feature = [0.0] * (token_count * FEATURE_DIM)
        message.mask = [False] * token_count
        message.source_received = True
        message.valid = False
        message.detail = "UNINITIALIZED"
        return message

    def _publish_invalid(
        self,
        source: object,
        detail: str,
        *,
        input_ready: bool,
    ) -> None:
        message = self._new_message(source)
        message.detail = detail
        self.publisher.publish(message)
        self.input_ready = input_ready
        self.output_valid = False
        self.module_state = (
            ModuleStatus.ERROR
            if self.encoder is None
            else ModuleStatus.DEGRADED
        )
        self.detail = detail
        self.get_logger().warning(detail)

    def _publish_evicted_frames(self, frames: tuple[object, ...]) -> None:
        for frame in frames:
            self._publish_invalid(
                frame,
                "SYNC_CACHE_EVICTED: matching entity frame did not arrive",
                input_ready=True,
            )

    def _record_alignment(self, frame: CameraFrame, entities: UEEntityArray) -> int:
        delta = int(entities.frame_index) - int(frame.frame_index)
        if delta == 0:
            return 0
        if len(self._alignment_diagnostics) < ALIGNMENT_DIAGNOSTIC_LIMIT:
            self._alignment_diagnostics.append(delta)
        if self._alignment_log_count < ALIGNMENT_DIAGNOSTIC_LIMIT:
            self.get_logger().warning(
                f"SYNC_NEAR_MATCH run_id={frame.run_id!r} "
                f"scene_seed={int(frame.scene_seed)} "
                f"camera_frame={int(frame.frame_index)} "
                f"entity_frame={int(entities.frame_index)} "
                f"frame_delta={delta}"
            )
            self._alignment_log_count += 1
        return delta

    def _process_pair(
        self, pair: tuple[object, object, int]
    ) -> None:
        frame, entities, frame_delta = pair
        self._record_alignment(frame, entities)
        self._process(
            frame,
            entities,
            frame_delta=frame_delta,
        )

    def on_entities(self, entities: UEEntityArray) -> None:
        if (
            not self.allow_truth_entities
            and str(entities.source) not in {"image_perception", "temporal_tracker"}
        ):
            self.get_logger().warning(
                "IGNORE_UNTRUSTED_ENTITY_SOURCE:" + str(entities.source)
            )
            return
        key = self._key(
            entities.run_id, entities.scene_seed, entities.frame_index
        )
        self._sync_cache.put_entities(entities)
        pair = self._sync_cache.match_for_entities(key)
        if pair is not None:
            self._process_pair(pair)

    def on_frame(self, frame: CameraFrame) -> None:
        if not frame.valid or not frame.data:
            self._publish_invalid(
                frame,
                "INVALID_FRAME: camera valid=false or JPEG payload is empty",
                input_ready=False,
            )
            return
        key = self._key(frame.run_id, frame.scene_seed, frame.frame_index)
        self._publish_evicted_frames(self._sync_cache.put_frame(frame))
        pair = self._sync_cache.match_for_frame(key)
        if pair is not None:
            self._process_pair(pair)

    def flush_expired_frames(self) -> None:
        for frame in self._sync_cache.expire():
            self._publish_invalid(
                frame,
                "ENTITY_FRAME_TIMEOUT: no same-run/scene entity within "
                f"frame_tolerance={self.sync_frame_tolerance}",
                input_ready=True,
            )

    def _process(
        self,
        frame: CameraFrame,
        entities: UEEntityArray,
        *,
        frame_delta: int = 0,
    ) -> None:
        if self.encoder is None:
            self._publish_invalid(
                entities,
                "MODEL_UNAVAILABLE: visual encoder failed to load",
                input_ready=True,
            )
            return
        if not entities.valid:
            self._publish_invalid(
                entities,
                f"INVALID_ENTITIES:{entities.detail}",
                input_ready=True,
            )
            return
        if entities.frame_id != "base_link":
            self._publish_invalid(
                entities,
                f"INVALID_ENTITY_FRAME: expected base_link, got "
                f"{entities.frame_id!r}",
                input_ready=True,
            )
            return
        if frame.run_id != entities.run_id:
            self._publish_invalid(
                entities,
                "RUN_ID_MISMATCH: camera and entities differ",
                input_ready=True,
            )
            return
        if frame.scene_seed != entities.scene_seed:
            self._publish_invalid(
                entities,
                "SCENE_SEED_MISMATCH: camera and entities differ",
                input_ready=True,
            )
            return

        try:
            image = enhance_low_light_image(
                decode_camera_image(frame.data, frame.encoding),
                enabled=self.image_preprocess_enabled,
                gamma=self.image_preprocess_gamma,
                brightness=self.image_preprocess_brightness,
                contrast=self.image_preprocess_contrast,
            )
            # Full slot layout matching the training cache:
            # [global, slot0, slot1, ..., slot15]; unprojectable slots are
            # zero features with mask=false.  A single-crop payload is out
            # of distribution for the policy and makes it thrash.
            order = build_entity_tensor(entities.entities)
            entity_by_id = {
                str(entity.entity_id): entity for entity in entities.entities
            }
            crops: list[tuple[int, object]] = []
            projected: list[str] = []
            for slot, entity_id in enumerate(order.entity_ids):
                if not order.mask[slot] or not entity_id:
                    continue
                entity = entity_by_id.get(entity_id)
                if entity is None:
                    continue
                try:
                    crop, _ = make_target_crop(image, entity, self.profile)
                except (TargetProjectionError, InvalidImageError):
                    continue
                crops.append((slot, crop))
                projected.append(entity_id)

            batch_images = [image] + [crop for _, crop in crops]
            _encode_start = time.monotonic()
            encoded = self.encoder.encode_images(batch_images)
            self._encode_ms = (time.monotonic() - _encode_start) * 1000.0
        except (
            InvalidImageError,
            TargetProjectionError,
            VisualEncoderError,
        ) as exc:
            self._publish_invalid(
                entities,
                f"{type(exc).__name__.upper()}:{exc}",
                input_ready=True,
            )
            return
        except Exception as exc:
            self._publish_invalid(
                entities,
                f"UNEXPECTED_VISUAL_ERROR:{type(exc).__name__}:{exc}",
                input_ready=True,
            )
            return

        # Assemble [global] + 16 slot tokens, zero-filled.
        token_count = 1 + MAX_ENTITIES
        features = np.zeros((token_count, FEATURE_DIM), dtype=np.float32)
        features[0] = encoded[0] if len(encoded) else 0.0
        for index, (slot, _) in enumerate(crops):
            features[1 + slot] = encoded[1 + index]
        mask = np.zeros(token_count, dtype=bool)
        mask[0] = True
        for _, (slot, _) in enumerate(crops):
            mask[1 + slot] = True

        if not np.all(np.isfinite(features)):
            self._publish_invalid(
                entities,
                "NONFINITE_VISUAL_FEATURES",
                input_ready=True,
            )
            return

        message = self._new_message(entities, token_count=token_count)
        message.feature = features.reshape(-1).tolist()
        message.mask = [bool(v) for v in mask]
        message.valid = True
        sync_detail = (
            "exact" if frame_delta == 0 else f"near;frame_delta={frame_delta}"
        )
        message.detail = (
            f"OK:tokens={token_count};crops={','.join(projected) or 'none'};"
            f"sync={sync_detail}"
        )
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = True
        self.module_state = ModuleStatus.READY
        self.detail = message.detail
        self._process_count += 1
        if self._process_count % 50 == 0:
            self.get_logger().info(
                f"VIS_TRACE frame_index={int(frame.frame_index)} "
                f"count={self._process_count} crops={len(crops)} "
                f"encode_ms={getattr(self, '_encode_ms', -1.0):.0f} "
                f"cache_entities={len(self.entities)} cache_frames={len(self.frames)}"
            )

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
    node = VisualEncoderNode()
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
