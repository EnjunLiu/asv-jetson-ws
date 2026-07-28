from __future__ import annotations

from io import BytesIO
import math
import time

import rclpy
from PIL import Image, ImageDraw
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    CameraFrame,
    UEEntity,
    UEEntityArray,
    VisualFeatures,
)

from .visual_encoder import FEATURE_DIM, TOKEN_COUNT


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


def synthetic_jpeg() -> bytes:
    image = Image.new("RGB", (1280, 720), (20, 45, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle((550, 390, 730, 570), fill=(210, 35, 35))
    draw.ellipse((605, 440, 675, 510), fill=(245, 220, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class VisualEncoderProbe(Node):
    def __init__(self) -> None:
        super().__init__("visual_encoder_probe")
        self.camera_pub = self.create_publisher(
            CameraFrame, "/ue/camera_frame", SENSOR_QOS
        )
        self.entity_pub = self.create_publisher(
            UEEntityArray, "/ue/entities", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            VisualFeatures,
            "/vla/visual_features",
            self.on_features,
            RELIABLE_QOS,
        )
        self.jpeg = synthetic_jpeg()
        self.started_at = time.monotonic()
        self.last_publish_at = 0.0
        self.phase = "WAIT_DISCOVERY"
        self.first_valid = None
        self.success = False
        self.failure = ""
        self.create_timer(0.1, self.tick)

    @staticmethod
    def make_frame(
        run_id: str,
        frame_index: int,
        *,
        valid: bool,
        data: bytes,
    ) -> CameraFrame:
        message = CameraFrame()
        message.stamp_us = 1_000_000 + frame_index
        message.run_id = run_id
        message.scene_seed = 12345
        message.frame_index = frame_index
        message.encoding = "jpeg"
        message.data = list(data)
        message.valid = valid
        return message

    @staticmethod
    def make_entities(run_id: str, frame_index: int) -> UEEntityArray:
        target = UEEntity()
        target.entity_id = "target_01"
        target.class_name = "boat"
        target.color = "red"
        target.is_target = True
        target.visible = True
        target.relative_x = 1.5
        target.relative_y = 0.0
        target.relative_z = -0.10554275
        target.relative_velocity_x = 0.0
        target.relative_velocity_y = 0.0
        target.relative_velocity_z = 0.0
        target.valid = True

        message = UEEntityArray()
        message.stamp_us = 1_000_000 + frame_index
        message.run_id = run_id
        message.scene_seed = 12345
        message.frame_index = frame_index
        message.frame_id = "base_link"
        message.entities = [target]
        message.valid = True
        message.detail = "probe"
        return message

    def fail(self, detail: str) -> None:
        if self.failure:
            return
        self.failure = detail
        self.get_logger().error(f"VISUAL_ENCODER_PROBE_FAIL:{detail}")

    @staticmethod
    def validate_fixed_contract(message: VisualFeatures) -> str | None:
        if message.token_count != TOKEN_COUNT:
            return f"token_count={message.token_count}"
        if message.feature_dim != FEATURE_DIM:
            return f"feature_dim={message.feature_dim}"
        if len(message.feature) != TOKEN_COUNT * FEATURE_DIM:
            return f"feature_length={len(message.feature)}"
        if len(message.mask) != TOKEN_COUNT:
            return f"mask_length={len(message.mask)}"
        if any(not math.isfinite(value) for value in message.feature):
            return "nonfinite feature"
        return None

    def on_features(self, message: VisualFeatures) -> None:
        contract_error = self.validate_fixed_contract(message)
        if contract_error:
            self.fail(contract_error)
            return

        if message.run_id == "day6-invalid":
            if message.valid:
                self.fail("empty invalid camera was promoted to valid")
                return
            if any(message.mask) or any(value != 0.0 for value in message.feature):
                self.fail("invalid camera did not produce zero/masked output")
                return
            self.phase = "VALID_1"
            self.last_publish_at = 0.0
            return

        if message.run_id == "day6-valid-1":
            if not message.valid or list(message.mask) != [True, True]:
                self.fail(f"first valid frame rejected: {message.detail}")
                return
            values = list(message.feature)
            for token_index in range(TOKEN_COUNT):
                start = token_index * FEATURE_DIM
                token = values[start:start + FEATURE_DIM]
                norm = math.sqrt(sum(value * value for value in token))
                if not math.isclose(norm, 1.0, abs_tol=1.0e-5):
                    self.fail(f"token {token_index} norm={norm}")
                    return
            self.first_valid = values
            self.phase = "VALID_2"
            self.last_publish_at = 0.0
            return

        if message.run_id == "day6-valid-2":
            if not message.valid:
                self.fail(f"second valid frame rejected: {message.detail}")
                return
            current = list(message.feature)
            max_difference = max(
                abs(left - right)
                for left, right in zip(self.first_valid, current)
            )
            if max_difference > 1.0e-6:
                self.fail(
                    f"repeated image changed; max_difference={max_difference}"
                )
                return
            self.success = True
            print(
                "VISUAL_ENCODER_PASS "
                f"tokens={TOKEN_COUNT}x{FEATURE_DIM} "
                f"deterministic_max_diff={max_difference:.9f}",
                flush=True,
            )

    def publish_valid_pair(self, run_id: str, frame_index: int) -> None:
        self.entity_pub.publish(self.make_entities(run_id, frame_index))
        self.camera_pub.publish(
            self.make_frame(
                run_id,
                frame_index,
                valid=True,
                data=self.jpeg,
            )
        )

    def tick(self) -> None:
        now = time.monotonic()
        if now - self.started_at > 45.0:
            self.fail(f"timeout in phase {self.phase}")
            return
        if (
            self.camera_pub.get_subscription_count() == 0
            or self.entity_pub.get_subscription_count() == 0
            or self.count_publishers("/vla/visual_features") == 0
        ):
            return
        if now - self.last_publish_at < 0.5:
            return
        self.last_publish_at = now

        if self.phase == "WAIT_DISCOVERY":
            self.phase = "INVALID"
        if self.phase == "INVALID":
            self.camera_pub.publish(
                self.make_frame(
                    "day6-invalid", 0, valid=False, data=b""
                )
            )
        elif self.phase == "VALID_1":
            self.publish_valid_pair("day6-valid-1", 1)
        elif self.phase == "VALID_2":
            self.publish_valid_pair("day6-valid-2", 2)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualEncoderProbe()
    try:
        while rclpy.ok() and not node.success and not node.failure:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        success = node.success
        failure = node.failure
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not success:
        raise SystemExit(f"VISUAL_ENCODER_PROBE_FAIL:{failure or 'unknown'}")


if __name__ == "__main__":
    main()
