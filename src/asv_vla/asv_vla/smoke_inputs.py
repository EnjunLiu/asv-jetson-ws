from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, String

from asv_jetson_interfaces.msg import (
    CameraFrame,
    RunContext,
    UEASVState,
    WorldState,
)


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class SmokeInputs(Node):
    def __init__(self) -> None:
        super().__init__("smoke_inputs")
        self.run_id = (
            self.declare_parameter("run_id", "day1-smoke").get_parameter_value().string_value
        )
        self.scene_seed = (
            self.declare_parameter("scene_seed", 1).get_parameter_value().integer_value
        )
        self.jetson_git_sha = (
            self.declare_parameter("jetson_git_sha", "unknown")
            .get_parameter_value()
            .string_value
        )
        self.esp32_git_sha = (
            self.declare_parameter("esp32_git_sha", "unknown")
            .get_parameter_value()
            .string_value
        )
        self.config_sha256 = (
            self.declare_parameter("config_sha256", "day1-placeholder")
            .get_parameter_value()
            .string_value
        )

        self.task_pub = self.create_publisher(String, "/task/text", LATCHED_QOS)
        self.context_pub = self.create_publisher(
            RunContext, "/system/run_context", LATCHED_QOS
        )
        self.connected_pub = self.create_publisher(
            Bool, "/ue/connected", LATCHED_QOS
        )
        self.camera_pub = self.create_publisher(
            CameraFrame, "/ue/camera_frame", SENSOR_QOS
        )
        self.world_pub = self.create_publisher(
            WorldState, "/perception/world_state", RELIABLE_QOS
        )
        self.state_pub = self.create_publisher(
            UEASVState, "/ue/asv_state", RELIABLE_QOS
        )

        self.create_timer(1.0, self.publish_inputs)
        self.publish_inputs()

    def now_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def publish_inputs(self) -> None:
        stamp_us = self.now_us()

        task = String()
        task.data = "跟随红色目标船，保持5米"
        self.task_pub.publish(task)

        context = RunContext()
        context.stamp_us = stamp_us
        context.run_id = self.run_id
        context.scene_seed = self.scene_seed
        context.jetson_git_sha = self.jetson_git_sha
        context.esp32_git_sha = self.esp32_git_sha
        context.config_sha256 = self.config_sha256
        context.language_model_id = "stub:none"
        context.policy_model_id = "stub:none"
        self.context_pub.publish(context)

        connected = Bool()
        connected.data = False
        self.connected_pub.publish(connected)

        camera = CameraFrame()
        camera.stamp_us = stamp_us
        camera.encoding = "none"
        camera.data = []
        camera.valid = False
        self.camera_pub.publish(camera)

        world = WorldState()
        world.stamp_us = stamp_us
        world.relative_x = 0.0
        world.relative_y = 0.0
        world.relative_z = 0.0
        world.confidence = 0.0
        world.tracking_id = 0
        world.valid = False
        self.world_pub.publish(world)

        state = UEASVState()
        state.stamp_us = stamp_us
        state.simulation_time = 0.0
        state.position_x = 0.0
        state.position_y = 0.0
        state.position_z = 0.0
        state.roll = 0.0
        state.pitch = 0.0
        state.yaw = 0.0
        state.surge_velocity = 0.0
        state.yaw_rate = 0.0
        state.valid = False
        self.state_pub.publish(state)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SmokeInputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
