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
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from asv_jetson_interfaces.msg import CameraFrame, ModuleStatus, UEEntity, UEEntityArray

from .image_entity_perception import (
    ImageEntityModel,
    ImageEntityPerceptionError,
    TaskSpec,
    parse_task_instruction,
    select_task_entities,
)
from .visual_encoder import (
    CameraProfile,
    TargetProjectionError,
    decode_camera_image,
    project_target_to_pixel,
)


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
TASK_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

PERCEPTION_TRACE_LIMIT = 5


def _format_perception_trace(
    *,
    run_id: str,
    frame_index: int,
    sample_index: int,
    model_version: str,
    entity: object,
) -> str:
    """Format one bounded red-target diagnostic line without changing data."""

    return (
        "PERCEPTION_TRACE "
        f"run_id={run_id} frame_index={int(frame_index)} "
        f"sample={int(sample_index)}/{PERCEPTION_TRACE_LIMIT} "
        f"model={model_version} target_red "
        f"relative_x={float(getattr(entity, 'relative_x', float('nan'))):.6f} "
        f"relative_y={float(getattr(entity, 'relative_y', float('nan'))):.6f} "
        f"visible={bool(getattr(entity, 'visible', False))} "
        f"confidence={float(getattr(entity, 'confidence', 0.0)):.6f} "
        f"bbox_valid={bool(getattr(entity, 'bbox_valid', False))}"
    )


def _apply_prediction_projection(
    entity: UEEntity,
    prediction: object,
    profile: CameraProfile,
    confidence_threshold: float,
) -> None:
    """Attach a conservative bbox without letting one bad projection escape."""

    entity.bbox_x_min = 0.0
    entity.bbox_y_min = 0.0
    entity.bbox_x_max = 0.0
    entity.bbox_y_max = 0.0
    entity.bbox_valid = False
    entity.visible = False
    if not entity.valid:
        return
    try:
        pixel_x, pixel_y, depth = project_target_to_pixel(
            entity.relative_x,
            entity.relative_y,
            entity.relative_z,
            profile,
        )
        # The model has no privileged UE bounding box.  This is a
        # conservative calibrated box around the image-derived centre, used
        # only for diagnostics/crops.
        half_w = max(8.0, min(96.0, 1600.0 / max(depth, 1.0)))
        half_h = max(6.0, min(64.0, half_w * 0.45))
        entity.bbox_x_min = float(max(0.0, pixel_x - half_w))
        entity.bbox_y_min = float(max(0.0, pixel_y - half_h))
        entity.bbox_x_max = float(min(profile.width - 1.0, pixel_x + half_w))
        entity.bbox_y_max = float(min(profile.height - 1.0, pixel_y + half_h))
        entity.bbox_valid = (
            entity.bbox_x_max > entity.bbox_x_min
            and entity.bbox_y_max > entity.bbox_y_min
        )
        entity.visible = (
            bool(prediction.visible)
            and entity.confidence >= confidence_threshold
            and entity.bbox_valid
        )
    except (TargetProjectionError, ValueError, ArithmeticError):
        # Keep the finite model geometry for temporal tracking, but fail closed
        # for image-space validity and visibility.
        return


class ImageEntityPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("image_entity_perception")
        self.device = str(
            self.declare_parameter("device", "cuda").value
        ).strip() or "cuda"
        default_model = str(
            Path.home()
            / "jetson_asv_ws"
            / "models"
            / "image_entity_color_calibrated_v1.npz"
        )
        model_path = str(
            self.declare_parameter("model_path", default_model)
            .get_parameter_value()
            .string_value
        )
        self.confidence_threshold = float(
            self.declare_parameter("confidence_threshold", 0.3)
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
        self.create_subscription(String, "/task/text", self.on_task, TASK_QOS)
        self.task_text = ""
        self.task_spec: TaskSpec = parse_task_instruction("")
        self._trace_run_id = ""
        self._trace_count = 0
        self.model: ImageEntityModel | None = None
        self.detail = (
            "loading image-only perception model;"
            f"device={self.device}"
        )
        self.module_state = ModuleStatus.STARTING
        self.input_ready = False
        self.output_valid = False
        try:
            self.model = ImageEntityModel.load(model_path)
            self.model.validate_device(self.device)
        except ImageEntityPerceptionError as exc:
            self.model = None
            self.detail = f"MODEL_LOAD_ERROR:device={self.device}:{exc}"
            self.module_state = ModuleStatus.ERROR
            self.get_logger().error(self.detail)
        else:
            self.detail = (
                f"ready model={self.model.model_version};"
                f"device={self.device};path={model_path}"
            )
            self.module_state = ModuleStatus.READY
            self.get_logger().info(self.detail)
        self.create_timer(1.0, self.publish_status)

    def on_task(self, message: String) -> None:
        self.task_text = str(message.data).strip()
        self.task_spec = parse_task_instruction(self.task_text)

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
        task_spec = getattr(
            self, "task_spec", parse_task_instruction(getattr(self, "task_text", ""))
        )
        task_text = str(getattr(self, "task_text", ""))
        message.instruction = task_text
        message.instruction_id = task_spec.instruction_id
        run_id = str(frame.run_id)
        if run_id != self._trace_run_id:
            self._trace_run_id = run_id
            self._trace_count = 0
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
            predictions = self.model.predict(
                image,
                task=task_spec,
                device=str(getattr(self, "device", "numpy")),
            )
        except (ImageEntityPerceptionError, ValueError) as exc:
            message.detail = f"PERCEPTION_ERROR:{type(exc).__name__}:{exc}"
            self.publisher.publish(message)
            self.output_valid = False
            return

        selected_ids = {
            prediction.entity_id
            for prediction in select_task_entities(predictions, task_spec)
        }
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
            entity.is_target = prediction.entity_id in selected_ids
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
            _apply_prediction_projection(
                entity,
                prediction,
                self.profile,
                self.confidence_threshold,
            )
            # A perception prediction can be geometrically valid while still
            # being irrelevant to the active task. Only selected entities are
            # visible/target-bearing in the online task stream.
            entity.visible = bool(entity.is_target and entity.visible)
            message.entities.append(entity)

        message.valid = bool(task_spec.valid)
        message.source = "image_perception"
        message.detail = (
            f"OK:image+instruction;task={task_spec.instruction_id};"
            f"entities={len(message.entities)};"
            f"model={self.model.model_version}"
        )
        red_entity = next(
            (
                entity
                for entity in message.entities
                if str(entity.entity_id) == "target_red"
            ),
            None,
        )
        if (
            message.valid
            and red_entity is not None
            and bool(red_entity.valid)
            and bool(red_entity.is_target)
            and self._trace_count < PERCEPTION_TRACE_LIMIT
        ):
            self._trace_count += 1
            self.get_logger().info(
                _format_perception_trace(
                    run_id=run_id,
                    frame_index=int(frame.frame_index),
                    sample_index=self._trace_count,
                    model_version=str(self.model.model_version),
                    entity=red_entity,
                )
            )
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = bool(message.valid)
        self.module_state = (
            ModuleStatus.READY if message.valid else ModuleStatus.DEGRADED
        )
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
