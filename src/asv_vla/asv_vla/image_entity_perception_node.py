"""Image-only entity perception node.

The node consumes camera JPEGs and emits geometry/semantics on
``/vla/perceived_entities``.  It deliberately has no subscription to
``/ue/entities``; that topic is reserved for the recorder and offline labels.
Velocity is always zero with ``velocity_valid=false`` until the temporal
tracker observes a second frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from asv_jetson_interfaces.msg import CameraFrame, ModuleStatus, UEEntity, UEEntityArray

from .image_entity_perception import (
    ImageEntityModel,
    ImageEntityPerceptionError,
)
from .visual_encoder import CameraProfile, decode_camera_image, project_target_to_pixel


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


class ImageEntityPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("image_entity_perception")
        default_model = str(
            Path.home() / "jetson_asv_ws" / "models" / "image_entity_perception_v1.npz"
        )
        model_path = str(
            self.declare_parameter("model_path", default_model)
            .get_parameter_value()
            .string_value
        )
        self.confidence_threshold = float(
            self.declare_parameter("confidence_threshold", 0.55)
            .get_parameter_value()
            .double_value
        )
        self.profile = CameraProfile(
            width=int(self.declare_parameter("image_width", 1280).value),
            height=int(self.declare_parameter("image_height", 720).value),
            horizontal_fov_deg=float(
                self.declare_parameter("horizontal_fov_deg", 90.0).value
            ),
            mount_x_m=float(self.declare_parameter("camera_mount_x_m", 0.42).value),
            mount_y_m=float(self.declare_parameter("camera_mount_y_m", 0.0).value),
            mount_z_m=float(self.declare_parameter("camera_mount_z_m", 0.20).value),
            pitch_deg=float(self.declare_parameter("camera_pitch_deg", -5.0).value),
        )
        self.publisher = self.create_publisher(
            UEEntityArray, "/vla/perceived_entities", RELIABLE_QOS
        )
        self.status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", RELIABLE_QOS
        )
        self.create_subscription(CameraFrame, "/ue/camera_frame", self.on_frame, SENSOR_QOS)
        self.create_subscription(String, "/task/text", self.on_task, RELIABLE_QOS)
        self.task_text = ""
        self.model: ImageEntityModel | None = None
        self.detail = "loading image-only perception model"
        self.module_state = ModuleStatus.STARTING
        self.input_ready = False
        self.output_valid = False
        try:
            self.model = ImageEntityModel.load(model_path)
        except ImageEntityPerceptionError as exc:
            self.detail = f"MODEL_LOAD_ERROR:{exc}"
            self.module_state = ModuleStatus.ERROR
            self.get_logger().error(self.detail)
        else:
            self.detail = f"ready model={self.model.model_version} path={model_path}"
            self.module_state = ModuleStatus.READY
            self.get_logger().info(self.detail)
        self.create_timer(1.0, self.publish_status)

    def on_task(self, message: String) -> None:
        self.task_text = str(message.data).strip()

    @staticmethod
    def _new_array(frame: CameraFrame) -> UEEntityArray:
        message = UEEntityArray()
        message.stamp_us = int(frame.stamp_us)
        message.run_id = str(frame.run_id)
        message.scene_seed = int(frame.scene_seed)
        message.frame_index = int(frame.frame_index)
        message.frame_id = "base_link"
        message.valid = False
        message.source = "image_perception"
        message.instruction = ""
        message.instruction_id = ""
        message.detail = "UNINITIALIZED"
        return message

    def on_frame(self, frame: CameraFrame) -> None:
        message = self._new_array(frame)
        if self.model is None:
            message.detail = self.detail
            self.publisher.publish(message)
            self.output_valid = False
            return
        if not frame.valid or not frame.data:
            message.detail = "INVALID_CAMERA_FRAME"
            self.publisher.publish(message)
            self.output_valid = False
            return
        try:
            image = decode_camera_image(frame.data, frame.encoding)
            predictions = self.model.predict(image)
        except (ImageEntityPerceptionError, ValueError) as exc:
            message.detail = f"PERCEPTION_ERROR:{type(exc).__name__}:{exc}"
            self.publisher.publish(message)
            self.output_valid = False
            return

        for prediction in predictions:
            entity = UEEntity()
            entity.entity_id = prediction.entity_id
            entity.class_name = "boat"
            entity.color = {
                "target_red": "red",
                "target_blue": "blue",
                "target_left": "white",
                "target_right": "white",
            }[prediction.entity_id]
            entity.is_target = True
            entity.relative_x = prediction.relative_x
            entity.relative_y = prediction.relative_y
            entity.relative_z = prediction.relative_z
            entity.relative_velocity_x = 0.0
            entity.relative_velocity_y = 0.0
            entity.relative_velocity_z = 0.0
            entity.velocity_valid = False
            entity.source = "image_perception"
            entity.confidence = float(prediction.confidence)
            entity.valid = all(
                math.isfinite(value)
                for value in (
                    prediction.relative_x,
                    prediction.relative_y,
                    prediction.relative_z,
                )
            )
            entity.visible = False
            try:
                pixel_x, pixel_y, depth = project_target_to_pixel(
                    prediction.relative_x,
                    prediction.relative_y,
                    prediction.relative_z,
                    self.profile,
                )
                # The model has no privileged UE bounding box.  This is a
                # conservative calibrated box around the image-derived
                # centre, used only for diagnostics/crops.
                half_w = max(8.0, min(96.0, 1600.0 / max(depth, 1.0)))
                half_h = max(6.0, min(64.0, half_w * 0.45))
                entity.bbox_x_min = float(max(0.0, pixel_x - half_w))
                entity.bbox_y_min = float(max(0.0, pixel_y - half_h))
                entity.bbox_x_max = float(min(self.profile.width - 1.0, pixel_x + half_w))
                entity.bbox_y_max = float(min(self.profile.height - 1.0, pixel_y + half_h))
                entity.bbox_valid = True
                entity.visible = (
                    prediction.confidence >= self.confidence_threshold
                    and entity.bbox_x_max > entity.bbox_x_min
                    and entity.bbox_y_max > entity.bbox_y_min
                )
            except (ValueError, ArithmeticError):
                entity.bbox_valid = False
            message.entities.append(entity)

        message.valid = True
        message.source = "image_perception"
        message.instruction = self.task_text
        message.detail = (
            f"OK:image_only;entities={len(message.entities)};"
            f"model={self.model.model_version}"
        )
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = True
        self.module_state = ModuleStatus.READY
        self.detail = message.detail

    def publish_status(self) -> None:
        message = ModuleStatus()
        message.stamp_us = self.get_clock().now().nanoseconds // 1000
        message.run_id = "image-entity-perception"
        message.module_name = self.get_name()
        message.state = self.module_state
        message.alive = True
        message.input_ready = self.input_ready
        message.output_valid = self.output_valid
        message.detail = self.detail
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImageEntityPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
