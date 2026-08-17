"""Image-only entity perception node.

The node consumes camera JPEGs and emits geometry/semantics on
``/vla/perceived_entities``.  It deliberately has no subscription to
``/ue/entities``; that topic is reserved for the recorder and offline labels.
Velocity is always zero with ``velocity_valid=false`` until the temporal
tracker observes a second frame.
"""

from __future__ import annotations

import inspect
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
from std_msgs.msg import String

from interfaces.msg import (
    CameraFrame,
    TaskEmbedding,
    Entity,
    EntityArray,
)

from .image_entity_perception import (
    COLOR_CALIBRATED_MODEL_VERSION,
    COLOR_CALIBRATED_MODEL_VERSION_V2,
    ImageEntityModel,
    ImageEntityPerceptionError,
    LANGUAGE_EMBEDDING_DIM,
    LOW_LIGHT_PREPROCESS_BRIGHTNESS,
    LOW_LIGHT_PREPROCESS_CONTRAST,
    LOW_LIGHT_PREPROCESS_CONTRACT,
    LOW_LIGHT_PREPROCESS_ENABLED,
    LOW_LIGHT_PREPROCESS_GAMMA,
    TaskSpec,
    parse_task_instruction,
    select_task_entities,
    validate_task_embedding,
)
from .visual_encoder import (
    CameraProfile,
    InvalidImageError,
    TargetProjectionError,
    decode_camera_image,
    enhance_low_light_image,
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


def _predict_with_color_reference(
    model: ImageEntityModel,
    feature_image: object,
    color_image: object,
    *,
    task: TaskSpec,
    device: str,
    task_embedding: object | None,
):
    """Run inference with separate feature and original-RGB inputs.

    The deployed ``ImageEntityModel`` exposes ``color_image`` explicitly.
    Non-calibrated test doubles may keep their older signature; a calibrated
    model without this contract fails closed instead of masking colors on
    enhanced RGB.
    """

    predict = model.predict
    try:
        parameters = inspect.signature(predict).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_color_reference = any(
        parameter.name == "color_image"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    kwargs = {"task": task, "device": device}
    supports_task_embedding = any(
        parameter.name == "task_embedding"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_task_embedding:
        kwargs["task_embedding"] = task_embedding
    elif model.model_version not in {
        "image_entity_ridge_v1",
        "image_entity_ridge_v2",
        "image_entity_color_calibrated_v1",
    }:
        raise ImageEntityPerceptionError(
            "MODEL_INPUT_CONTRACT_MISMATCH: model does not accept task_embedding"
        )
    if supports_color_reference:
        kwargs["color_image"] = color_image
    elif model.model_version in (
        COLOR_CALIBRATED_MODEL_VERSION,
        COLOR_CALIBRATED_MODEL_VERSION_V2,
    ):
        raise ImageEntityPerceptionError(
            "calibrated perception model lacks original-RGB color contract"
        )
    return predict(feature_image, **kwargs)


def _format_perception_trace(
    *,
    run_id: str,
    frame_index: int,
    sample_index: int,
    model_version: str,
    entity: object,
) -> str:
    """Format one bounded target diagnostic line without changing data."""

    return (
        "PERCEPTION_TRACE "
        f"run_id={run_id} frame_index={int(frame_index)} "
        f"sample={int(sample_index)}/{PERCEPTION_TRACE_LIMIT} "
        f"model={model_version} target={str(getattr(entity, 'entity_id', ''))} "
        f"relative_x={float(getattr(entity, 'relative_x', float('nan'))):.6f} "
        f"relative_y={float(getattr(entity, 'relative_y', float('nan'))):.6f} "
        f"visible={bool(getattr(entity, 'visible', False))} "
        f"confidence={float(getattr(entity, 'confidence', 0.0)):.6f} "
        f"bbox_valid={bool(getattr(entity, 'bbox_valid', False))}"
    )


def _apply_prediction_projection(
    entity: Entity,
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
            / "perception_image_conditioned_130_v1.npz"
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
        self.allow_legacy_image_only = bool(
            self.declare_parameter("allow_legacy_image_only", False).value
        )
        self.image_preprocess_enabled = bool(
            self.declare_parameter(
                "image_preprocess_enabled", LOW_LIGHT_PREPROCESS_ENABLED
            ).value
        )
        self.image_preprocess_gamma = float(
            self.declare_parameter(
                "image_preprocess_gamma", LOW_LIGHT_PREPROCESS_GAMMA
            ).value
        )
        self.image_preprocess_brightness = float(
            self.declare_parameter(
                "image_preprocess_brightness", LOW_LIGHT_PREPROCESS_BRIGHTNESS
            ).value
        )
        self.image_preprocess_contrast = float(
            self.declare_parameter(
                "image_preprocess_contrast", LOW_LIGHT_PREPROCESS_CONTRAST
            ).value
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
            EntityArray, "/vla/perceived_entities", RELIABLE_QOS
        )
        self.create_subscription(CameraFrame, "/ue/camera_frame", self.on_frame, SENSOR_QOS)
        self.create_subscription(String, "/task/text", self.on_task, TASK_QOS)
        self.create_subscription(
            TaskEmbedding,
            "/vla/language_embedding",
            self.on_embedding,
            TASK_QOS,
        )
        self.task_text = ""
        self.task_spec: TaskSpec = parse_task_instruction("")
        self.task_embedding = None
        self.embedding_model_id = ""
        self.embedding_detail = "WAITING_FOR_TASK_EMBEDDING"
        self._trace_run_id = ""
        self._trace_count = 0
        self._frame_count = 0
        self.model: ImageEntityModel | None = None
        self.detail = (
            "loading image-only perception model;"
            f"device={self.device}"
        )
        try:
            self.model = ImageEntityModel.load(
                model_path, allow_legacy=self.allow_legacy_image_only
            )
            self.model.validate_device(self.device)
        except ImageEntityPerceptionError as exc:
            self.model = None
            self.detail = f"MODEL_LOAD_ERROR:device={self.device}:{exc}"
            self.get_logger().error(self.detail)
        else:
            self.detail = (
                f"ready model={self.model.model_version};"
                f"device={self.device};path={model_path};"
                f"input={self.model.input_contract};"
                f"language_model={self.model.language_model_id or 'legacy'};"
                f"legacy_image_only={self.model.model_version in {'image_entity_ridge_v1', 'image_entity_ridge_v2', 'image_entity_color_calibrated_v1'}};"
                f"preprocess={LOW_LIGHT_PREPROCESS_CONTRACT};"
                f"enabled={self.image_preprocess_enabled};"
                f"gamma={self.image_preprocess_gamma:.3f};"
                f"brightness={self.image_preprocess_brightness:.3f};"
                f"contrast={self.image_preprocess_contrast:.3f}"
            )
            self.get_logger().info(self.detail)

    def on_task(self, message: String) -> None:
        next_text = str(message.data).strip()
        if next_text != self.task_text:
            self.task_embedding = None
            self.embedding_model_id = ""
            self.embedding_detail = "WAITING_FOR_TASK_EMBEDDING"
        self.task_text = next_text
        self.task_spec = parse_task_instruction(self.task_text)

    def on_embedding(self, message: TaskEmbedding) -> None:
        """Accept only a valid embedding for the current instruction/model."""

        instruction = str(getattr(message, "instruction", "")).strip()
        if not bool(getattr(message, "valid", False)):
            self.task_embedding = None
            self.embedding_model_id = ""
            self.embedding_detail = "INVALID_TASK_EMBEDDING"
            return
        if self.model is None:
            self.task_embedding = None
            self.embedding_detail = "MODEL_UNAVAILABLE"
            return
        if self.model.model_version in {
            "image_entity_ridge_v1",
            "image_entity_ridge_v2",
            "image_entity_color_calibrated_v1",
        }:
            self.embedding_detail = "LEGACY_IMAGE_ONLY_MODE_IGNORES_TASK_EMBEDDING"
            return
        if instruction and self.task_text and instruction != self.task_text:
            self.task_embedding = None
            self.embedding_model_id = ""
            self.embedding_detail = (
                "TASK_EMBEDDING_INSTRUCTION_MISMATCH:"
                f"text={self.task_text!r};embedding={instruction!r}"
            )
            return
        model_id = str(getattr(message, "model_id", "")).strip()
        if model_id != self.model.language_model_id:
            self.task_embedding = None
            self.embedding_model_id = ""
            self.embedding_detail = (
                "TASK_EMBEDDING_MODEL_ID_MISMATCH:"
                f"expected={self.model.language_model_id};got={model_id}"
            )
            return
        embedding_dim = int(getattr(message, "embedding_dim", 0))
        try:
            embedding = validate_task_embedding(
                getattr(message, "embedding", ()),
                expected_dim=self.model.task_embedding_dim,
            )
        except ImageEntityPerceptionError as exc:
            self.task_embedding = None
            self.embedding_model_id = ""
            self.embedding_detail = f"TASK_EMBEDDING_ERROR:{exc}"
            return
        if embedding_dim != self.model.task_embedding_dim:
            self.task_embedding = None
            self.embedding_model_id = ""
            self.embedding_detail = (
                "TASK_EMBEDDING_DIM_MISMATCH:"
                f"expected={self.model.task_embedding_dim};got={embedding_dim}"
            )
            return
        if instruction and not self.task_text:
            self.task_text = instruction
            self.task_spec = parse_task_instruction(instruction)
        self.task_embedding = embedding
        self.embedding_model_id = model_id
        self.embedding_detail = "VALID_TASK_EMBEDDING"

    @staticmethod
    def _new_array(frame: CameraFrame) -> EntityArray:
        message = EntityArray()
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
            return
        if (
            self.model.model_version not in {
                "image_entity_ridge_v1",
                "image_entity_ridge_v2",
                "image_entity_color_calibrated_v1",
            }
            and self.task_embedding is None
        ):
            message.detail = f"MISSING_TASK_EMBEDDING:{self.embedding_detail}"
            self.publisher.publish(message)
            return
        if not frame.valid or not frame.data:
            message.detail = "INVALID_CAMERA_FRAME"
            self.publisher.publish(message)
            return
        try:
            color_image = decode_camera_image(frame.data, frame.encoding)
            feature_image = enhance_low_light_image(
                color_image,
                enabled=getattr(
                    self, "image_preprocess_enabled", LOW_LIGHT_PREPROCESS_ENABLED
                ),
                gamma=getattr(
                    self, "image_preprocess_gamma", LOW_LIGHT_PREPROCESS_GAMMA
                ),
                brightness=getattr(
                    self,
                    "image_preprocess_brightness",
                    LOW_LIGHT_PREPROCESS_BRIGHTNESS,
                ),
                contrast=getattr(
                    self, "image_preprocess_contrast", LOW_LIGHT_PREPROCESS_CONTRAST
                ),
            )
            _predict_start = time.monotonic()
            predictions = _predict_with_color_reference(
                self.model,
                feature_image,
                color_image,
                task=task_spec,
                device=str(getattr(self, "device", "numpy")),
                task_embedding=getattr(self, "task_embedding", None),
            )
            _predict_ms = (time.monotonic() - _predict_start) * 1000.0
        except (ImageEntityPerceptionError, InvalidImageError, ValueError) as exc:
            message.detail = f"PERCEPTION_ERROR:{type(exc).__name__}:{exc}"
            self.publisher.publish(message)
            return

        selected_ids = {
            prediction.entity_id
            for prediction in select_task_entities(predictions, task_spec)
        }
        for prediction in predictions:
            entity = Entity()
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
        input_mode = (
            "image-only-legacy"
            if self.model.model_version
            in {
                "image_entity_ridge_v1",
                "image_entity_ridge_v2",
                "image_entity_color_calibrated_v1",
            }
            else "image+task_embedding"
        )
        message.detail = (
            f"OK:{input_mode};task={task_spec.instruction_id};"
            f"entities={len(message.entities)};"
            f"model={self.model.model_version};"
            "velocity_output=false;velocity_source=temporal_entity_tracker"
        )
        target_entity = next(
            (
                entity
                for entity in message.entities
                if bool(entity.is_target)
            ),
            None,
        )
        if (
            message.valid
            and target_entity is not None
            and bool(target_entity.valid)
            and self._trace_count < PERCEPTION_TRACE_LIMIT
        ):
            self._trace_count += 1
            self.get_logger().info(
                _format_perception_trace(
                    run_id=run_id,
                    frame_index=int(frame.frame_index),
                    sample_index=self._trace_count,
                    model_version=str(self.model.model_version),
                    entity=target_entity,
                )
            )
        self.publisher.publish(message)
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        if self._frame_count % 50 == 0:
            self.get_logger().info(
                f"PERCEPTION_PERF_TRACE frame_index={int(frame.frame_index)} "
                f"count={self._frame_count} predict_ms={_predict_ms:.1f} "
                f"entities={len(message.entities)} valid={message.valid}"
            )


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
