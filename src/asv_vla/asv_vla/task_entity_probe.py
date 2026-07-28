from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from asv_jetson_interfaces.msg import (
    TaskFeatures,
    UEEntity,
    UEEntityArray,
)

from .task_entity_tensor import FEATURE_DIM, MAX_ENTITIES


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)


def make_entity(
    entity_id: str,
    x: float,
    *,
    y: float = 0.0,
    vx: float = 0.0,
    color: str = "",
    is_target: bool = False,
    visible: bool = True,
) -> UEEntity:
    entity = UEEntity()
    entity.entity_id = entity_id
    entity.class_name = "boat"
    entity.color = color
    entity.is_target = is_target
    entity.visible = visible
    entity.relative_x = x
    entity.relative_y = y
    entity.relative_z = 0.0
    entity.relative_velocity_x = vx
    entity.relative_velocity_y = 0.0
    entity.relative_velocity_z = 0.0
    entity.valid = True
    return entity


class TaskEntityProbe(Node):
    def __init__(self) -> None:
        super().__init__("task_entity_probe")
        self.publisher = self.create_publisher(
            UEEntityArray, "/ue/entities", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            TaskFeatures,
            "/vla/task_features",
            self.on_features,
            RELIABLE_QOS,
        )
        self.phase = "WAIT_DISCOVERY"
        self.started_at = time.monotonic()
        self.last_publish_at = 0.0
        self.success = False
        self.failure = ""
        self.create_timer(0.1, self.tick)

    @staticmethod
    def make_source(
        run_id: str,
        frame_index: int,
        *,
        valid: bool,
        entities: list[UEEntity],
    ) -> UEEntityArray:
        source = UEEntityArray()
        source.stamp_us = 2_000_000 + frame_index
        source.run_id = run_id
        source.scene_seed = 12345
        source.frame_index = frame_index
        source.frame_id = "base_link"
        source.entities = entities
        source.valid = valid
        source.detail = "probe" if valid else "probe-invalid"
        return source

    def fail(self, detail: str) -> None:
        if self.failure:
            return
        self.failure = detail
        self.get_logger().error(f"TASK_ENTITY_PROBE_FAIL:{detail}")

    @staticmethod
    def contract_error(message: TaskFeatures) -> str | None:
        if message.max_entities != MAX_ENTITIES:
            return f"max_entities={message.max_entities}"
        if message.feature_dim != FEATURE_DIM:
            return f"feature_dim={message.feature_dim}"
        if len(message.features) != MAX_ENTITIES * FEATURE_DIM:
            return f"feature_length={len(message.features)}"
        if len(message.mask) != MAX_ENTITIES:
            return f"mask_length={len(message.mask)}"
        if len(message.entity_ids) != MAX_ENTITIES:
            return f"entity_ids_length={len(message.entity_ids)}"
        if any(not math.isfinite(value) for value in message.features):
            return "nonfinite feature"
        return None

    def on_features(self, message: TaskFeatures) -> None:
        error = self.contract_error(message)
        if error:
            self.fail(error)
            return

        if message.run_id == "day7-invalid":
            if message.valid:
                self.fail("invalid source was promoted to valid")
                return
            if any(message.mask) or any(message.features):
                self.fail("invalid source did not produce fixed zero output")
                return
            self.phase = "VALID"
            self.last_publish_at = 0.0
            return

        if message.run_id != "day7-valid":
            return
        if not message.valid:
            self.fail(f"valid source rejected: {message.detail}")
            return
        if message.entity_count != MAX_ENTITIES:
            self.fail(f"entity_count={message.entity_count}")
            return
        if not all(message.mask):
            self.fail("selected rows are not fully masked true")
            return
        if message.entity_ids[0] != "far_target":
            self.fail(
                f"target was not retained first: {message.entity_ids[0]!r}"
            )
            return
        if message.entity_ids[1] != "closing_risk":
            self.fail(
                f"risk was not retained second: {message.entity_ids[1]!r}"
            )
            return
        if "hidden_nearest" in message.entity_ids:
            self.fail("hidden entity was retained")
            return

        target_offset = 0
        risk_offset = FEATURE_DIM
        if message.features[target_offset + 12] != 1.0:
            self.fail("target flag missing from first row")
            return
        if message.features[target_offset + 14] != 1.0:
            self.fail("red color flag missing from target row")
            return
        if message.features[risk_offset + 13] != 1.0:
            self.fail("risk flag missing from second row")
            return

        self.success = True
        print(
            "TASK_ENTITY_TENSOR_PASS "
            f"shape={MAX_ENTITIES}x{FEATURE_DIM} "
            f"first={message.entity_ids[0]} "
            f"second={message.entity_ids[1]}",
            flush=True,
        )

    def publish_invalid(self) -> None:
        self.publisher.publish(
            self.make_source(
                "day7-invalid",
                0,
                valid=False,
                entities=[],
            )
        )

    def publish_valid(self) -> None:
        entities = [
            make_entity(f"normal_{index:02d}", 1.0 + index)
            for index in range(20)
        ]
        entities.extend([
            make_entity(
                "far_target",
                100.0,
                color="red",
                is_target=True,
            ),
            make_entity(
                "closing_risk",
                8.0,
                y=1.0,
                vx=-2.0,
            ),
            make_entity("hidden_nearest", 0.1, visible=False),
        ])
        self.publisher.publish(
            self.make_source(
                "day7-valid",
                1,
                valid=True,
                entities=entities,
            )
        )

    def tick(self) -> None:
        now = time.monotonic()
        if now - self.started_at > 30.0:
            self.fail(f"timeout in phase {self.phase}")
            return
        if (
            self.publisher.get_subscription_count() == 0
            or self.count_publishers("/vla/task_features") == 0
        ):
            return
        if now - self.last_publish_at < 0.5:
            return
        self.last_publish_at = now
        if self.phase == "WAIT_DISCOVERY":
            self.phase = "INVALID"
        if self.phase == "INVALID":
            self.publish_invalid()
        elif self.phase == "VALID":
            self.publish_valid()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskEntityProbe()
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
        raise SystemExit(
            f"TASK_ENTITY_PROBE_FAIL:{failure or 'unknown'}"
        )


if __name__ == "__main__":
    main()
