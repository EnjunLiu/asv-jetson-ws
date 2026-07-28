from __future__ import annotations

from typing import Iterable

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from asv_jetson_interfaces.msg import (
    CameraFrame,
    DecisionOutput,
    ModuleStatus,
    SelectedTrajectory,
    TaskEmbedding,
    TaskFeatures,
    VisualFeatures,
    WorldState,
)

from .trajectory_contract import (
    ACTION_DIM,
    DT_SEC,
    FRAME_ID,
    HORIZON,
    SAFE_STOP_MODEL_VERSION,
    is_day1_safe_stop,
)


LANGUAGE_DIM = 256
VISUAL_TOKEN_COUNT = 2
VISUAL_DIM = 576
MAX_ENTITIES = 16
ENTITY_FEATURE_DIM = 16
POLICY_PERIOD_SEC = 0.5
CONTROL_PERIOD_SEC = 0.1

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


def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000


class StatusNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self.run_id = (
            self.declare_parameter("run_id", "day1-smoke").get_parameter_value().string_value
        )
        self.module_state = ModuleStatus.STARTING
        self.input_ready = False
        self.output_valid = False
        self.detail = "starting"
        self.status_pub = self.create_publisher(
            ModuleStatus, "/system/module_status", RELIABLE_QOS
        )
        self.create_timer(1.0, self.publish_status)

    def publish_status(self) -> None:
        message = ModuleStatus()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.module_name = self.get_name()
        message.state = self.module_state
        message.alive = True
        message.input_ready = self.input_ready
        message.output_valid = self.output_valid
        message.detail = self.detail
        self.status_pub.publish(message)


class LanguageEncoderStub(StatusNode):
    def __init__(self) -> None:
        super().__init__("language_encoder_stub")
        self.publisher = self.create_publisher(
            TaskEmbedding, "/vla/language_embedding", LATCHED_QOS
        )
        self.subscription = self.create_subscription(
            String, "/task/text", self.on_text, LATCHED_QOS
        )
        self.module_state = ModuleStatus.DEGRADED
        self.detail = "waiting for task text; embedding backend is a stub"

    def on_text(self, task: String) -> None:
        message = TaskEmbedding()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.instruction = task.data
        message.model_id = "stub:none"
        message.embedding_dim = LANGUAGE_DIM
        message.embedding = [0.0] * LANGUAGE_DIM
        message.cached = True
        message.valid = False
        message.detail = "Day 1 placeholder; replace with the frozen text encoder"
        self.publisher.publish(message)
        self.input_ready = bool(task.data.strip())
        self.output_valid = False
        self.detail = "task text received; placeholder embedding published"


class VisualEncoderStub(StatusNode):
    def __init__(self) -> None:
        super().__init__("visual_encoder_stub")
        self.publisher = self.create_publisher(
            VisualFeatures, "/vla/visual_features", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            CameraFrame, "/ue/camera_frame", self.on_frame, SENSOR_QOS
        )
        self.module_state = ModuleStatus.DEGRADED
        self.detail = "waiting for camera frame; visual backend is a stub"

    def on_frame(self, frame: CameraFrame) -> None:
        message = VisualFeatures()
        message.stamp_us = now_us(self)
        message.run_id = frame.run_id or self.run_id
        message.scene_seed = frame.scene_seed
        message.frame_index = frame.frame_index
        message.backbone = "stub:none"
        message.token_count = VISUAL_TOKEN_COUNT
        message.feature_dim = VISUAL_DIM
        message.feature = [0.0] * (VISUAL_TOKEN_COUNT * VISUAL_DIM)
        message.mask = [False] * VISUAL_TOKEN_COUNT
        message.source_received = True
        message.valid = False
        message.detail = "Day 1 placeholder; image validity is not promoted"
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = False
        self.detail = (
            "camera message received; placeholder visual features published; "
            f"source_valid={frame.valid}"
        )


class TaskFeatureBuilderStub(StatusNode):
    def __init__(self) -> None:
        super().__init__("task_feature_builder_stub")
        self.publisher = self.create_publisher(
            TaskFeatures, "/vla/task_features", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            WorldState, "/perception/world_state", self.on_world_state, RELIABLE_QOS
        )
        self.module_state = ModuleStatus.DEGRADED
        self.detail = "waiting for world state; task feature backend is a stub"

    def on_world_state(self, world_state: WorldState) -> None:
        message = TaskFeatures()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.scene_seed = 0
        message.frame_index = 0
        message.frame_id = "base_link"
        message.backend = "stub_v0"
        message.max_entities = MAX_ENTITIES
        message.feature_dim = ENTITY_FEATURE_DIM
        message.entity_count = 0
        message.entity_ids = [""] * MAX_ENTITIES
        message.features = [0.0] * (MAX_ENTITIES * ENTITY_FEATURE_DIM)
        message.mask = [False] * MAX_ENTITIES
        message.valid = False
        message.detail = "Day 1 placeholder; UE5 truth is not promoted to a valid feature"
        self.publisher.publish(message)
        self.input_ready = True
        self.output_valid = False
        self.detail = (
            "world-state message received; placeholder task features published; "
            f"source_valid={world_state.valid}"
        )


class TrajectoryPolicyStub(StatusNode):
    """Stub policy that publishes a single safe-stop trajectory directly.

    In the production implementation this will be replaced by the VLA
    trajectory policy that maps language, vision, and task features to a
    4 s (20-step × 0.2 s) 2-D displacement trajectory in the body frame.
    """

    def __init__(self) -> None:
        super().__init__("trajectory_policy_stub")
        self.have_language = False
        self.have_visual = False
        self.have_task_features = False
        self.language_valid = False
        self.visual_valid = False
        self.task_features_valid = False
        self.publisher = self.create_publisher(
            SelectedTrajectory, "/vla/selected_trajectory", RELIABLE_QOS
        )
        self.language_sub = self.create_subscription(
            TaskEmbedding,
            "/vla/language_embedding",
            self.on_language,
            LATCHED_QOS,
        )
        self.visual_sub = self.create_subscription(
            VisualFeatures, "/vla/visual_features", self.on_visual, RELIABLE_QOS
        )
        self.task_sub = self.create_subscription(
            TaskFeatures, "/vla/task_features", self.on_task_features, RELIABLE_QOS
        )
        self.create_timer(POLICY_PERIOD_SEC, self.publish_trajectory)
        self.module_state = ModuleStatus.DEGRADED
        self.detail = "policy backend is a stub; safe stop only"

    def on_language(self, message: TaskEmbedding) -> None:
        self.have_language = True
        self.language_valid = message.valid

    def on_visual(self, message: VisualFeatures) -> None:
        self.have_visual = True
        self.visual_valid = message.valid

    def on_task_features(self, message: TaskFeatures) -> None:
        self.have_task_features = True
        self.task_features_valid = message.valid

    def publish_trajectory(self) -> None:
        message = SelectedTrajectory()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.frame_id = FRAME_ID
        message.model_version = SAFE_STOP_MODEL_VERSION
        message.dt = DT_SEC
        message.horizon = HORIZON
        message.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
        message.safe_stop = True
        message.valid = True
        message.reason = "Day 1 placeholder; stub policy outputs safe stop"
        self.publisher.publish(message)
        self.input_ready = (
            self.have_language and self.have_visual and self.have_task_features
        )
        self.output_valid = True
        if self.language_valid and self.visual_valid and self.task_features_valid:
            self.detail = "all inputs valid; Day 1 policy still forces safe stop"
        else:
            self.detail = "one or more inputs invalid; fail-closed safe stop published"


class TrajectoryControllerStub(StatusNode):
    def __init__(self) -> None:
        super().__init__("trajectory_controller_stub")
        self.have_selected = False
        self.publisher = self.create_publisher(
            DecisionOutput, "/decision/output", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            SelectedTrajectory,
            "/vla/selected_trajectory",
            self.on_selected,
            RELIABLE_QOS,
        )
        self.create_timer(CONTROL_PERIOD_SEC, self.publish_invalid_zero)
        self.module_state = ModuleStatus.SAFE_STOP
        self.detail = "safe stop is mapped to an invalid zero DecisionOutput"

    def on_selected(self, message: SelectedTrajectory) -> None:
        self.have_selected = is_day1_safe_stop(message)

    def publish_invalid_zero(self) -> None:
        message = DecisionOutput()
        message.stamp_us = now_us(self)
        message.desired_x = 0.0
        message.desired_y = 0.0
        message.valid = False
        self.publisher.publish(message)
        self.input_ready = self.have_selected
        self.output_valid = False


def destroy_nodes(nodes: Iterable[Node]) -> None:
    for node in nodes:
        node.destroy_node()


def run_stack(include_language: bool, args=None) -> None:
    rclpy.init(args=args)
    nodes = []
    if include_language:
        nodes.append(LanguageEncoderStub())
    nodes.extend([
        VisualEncoderStub(),
        TaskFeatureBuilderStub(),
        TrajectoryPolicyStub(),
        TrajectoryControllerStub(),
    ])
    executor = MultiThreadedExecutor(num_threads=4)
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        destroy_nodes(nodes)
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None) -> None:
    run_stack(include_language=True, args=args)


def main_without_language(args=None) -> None:
    run_stack(include_language=False, args=args)


def main_safety_tail(args=None) -> None:
    """Run only language fallback, safe-stop policy, and controller.

    Day 8 uses the real visual and task-entity encoders, so their Day 1 stubs
    must not be started as duplicate publishers.
    """

    rclpy.init(args=args)
    nodes = [
        LanguageEncoderStub(),
        TrajectoryPolicyStub(),
        TrajectoryControllerStub(),
    ]
    executor = MultiThreadedExecutor(num_threads=3)
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        destroy_nodes(nodes)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
