from __future__ import annotations

from collections import OrderedDict
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
    FEATURE_DIM,
    TOKEN_COUNT,
    CameraProfile,
    FrozenMobileNetEncoder,
    InvalidImageError,
    TargetProjectionError,
    VisualEncoderError,
    decode_camera_image,
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
        cache_size = (
            self.declare_parameter("sync_cache_size", 16)
            .get_parameter_value()
            .integer_value
        )
        if self.entity_wait_sec <= 0.0:
            raise ValueError("entity_wait_sec must be positive")
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

        self.entities: OrderedDict[
            tuple[str, int], UEEntityArray
        ] = OrderedDict()
        self.frames: OrderedDict[
            tuple[str, int], tuple[CameraFrame, float]
        ] = OrderedDict()
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
                f"{MAX_ENTITIES} entity slots)x{FEATURE_DIM}"
            )
            self.get_logger().info(
                f"{self.detail}; entities_topic={self.entities_topic}"
            )

    @staticmethod
    def _key(run_id: str, frame_index: int) -> tuple[str, int]:
        return str(run_id), int(frame_index)

    def _new_message(
        self, frame: CameraFrame, token_count: int = TOKEN_COUNT
    ) -> VisualFeatures:
        message = VisualFeatures()
        message.stamp_us = frame.stamp_us
        message.run_id = frame.run_id
        message.scene_seed = frame.scene_seed
        message.frame_index = frame.frame_index
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
        frame: CameraFrame,
        detail: str,
        *,
        input_ready: bool,
    ) -> None:
        message = self._new_message(frame)
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

    def _trim_cache(self) -> None:
        while len(self.entities) > self.cache_size:
            self.entities.popitem(last=False)
        while len(self.frames) > self.cache_size:
            _, (frame, _) = self.frames.popitem(last=False)
            self._publish_invalid(
                frame,
                "SYNC_CACHE_EVICTED: matching entity frame did not arrive",
                input_ready=True,
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
        key = self._key(entities.run_id, entities.frame_index)
        self.entities[key] = entities
        self.entities.move_to_end(key)
        self._process_if_ready(key)
        self._trim_cache()

    def on_frame(self, frame: CameraFrame) -> None:
        if not frame.valid or not frame.data:
            self._publish_invalid(
                frame,
                "INVALID_FRAME: camera valid=false or JPEG payload is empty",
                input_ready=False,
            )
            return
        key = self._key(frame.run_id, frame.frame_index)
        self.frames[key] = (frame, time.monotonic())
        self.frames.move_to_end(key)
        self._process_if_ready(key)
        self._trim_cache()

    def flush_expired_frames(self) -> None:
        current = time.monotonic()
        expired = [
            key
            for key, (_, received_at) in self.frames.items()
            if current - received_at >= self.entity_wait_sec
        ]
        for key in expired:
            frame, _ = self.frames.pop(key)
            self._publish_invalid(
                frame,
                "ENTITY_FRAME_TIMEOUT: no exact run_id/frame_index match",
                input_ready=True,
            )

    def _process_if_ready(self, key: tuple[str, int]) -> None:
        frame_entry = self.frames.get(key)
        entities = self.entities.get(key)
        if frame_entry is None or entities is None:
            return
        frame, _ = self.frames.pop(key)
        self.entities.pop(key, None)
        self._process(frame, entities)

    def _process(
        self,
        frame: CameraFrame,
        entities: UEEntityArray,
    ) -> None:
        if self.encoder is None:
            self._publish_invalid(
                frame,
                "MODEL_UNAVAILABLE: visual encoder failed to load",
                input_ready=True,
            )
            return
        if not entities.valid:
            self._publish_invalid(
                frame,
                f"INVALID_ENTITIES:{entities.detail}",
                input_ready=True,
            )
            return
        if entities.frame_id != "base_link":
            self._publish_invalid(
                frame,
                f"INVALID_ENTITY_FRAME: expected base_link, got "
                f"{entities.frame_id!r}",
                input_ready=True,
            )
            return
        if frame.scene_seed != entities.scene_seed:
            self._publish_invalid(
                frame,
                "SCENE_SEED_MISMATCH: camera and entities differ",
                input_ready=True,
            )
            return

        try:
            image = decode_camera_image(frame.data, frame.encoding)
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
            encoded = self.encoder.encode_images(batch_images)
        except (
            InvalidImageError,
            TargetProjectionError,
            VisualEncoderError,
        ) as exc:
            self._publish_invalid(
                frame,
                f"{type(exc).__name__.upper()}:{exc}",
                input_ready=True,
            )
            return
        except Exception as exc:
            self._publish_invalid(
                frame,
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
                frame,
                "NONFINITE_VISUAL_FEATURES",
                input_ready=True,
            )
            return

        message = self._new_message(frame, token_count=token_count)
        message.feature = features.reshape(-1).tolist()
        message.mask = [bool(v) for v in mask]
        message.valid = True
        message.detail = (
            f"OK:tokens={token_count};crops={','.join(projected) or 'none'}"
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
