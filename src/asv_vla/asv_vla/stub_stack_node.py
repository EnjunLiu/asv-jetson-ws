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
    TrajectoryCandidates,
    VisualFeatures,
    WorldModelEvaluation,
    WorldState,
)


NUM_MODES = 6
HORIZON = 20
ACTION_DIM = 2
STOP_INDEX = 5
LANGUAGE_DIM = 256
VISUAL_DIM = 128
MAX_ENTITIES = 16
ENTITY_FEATURE_DIM = 12
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
        message.detail = "Stub decision output; replace with learned policy"
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
        message.run_id = self.run_id
        message.feature_dim = VISUAL_DIM
        message.feature = [0.0] * VISUAL_DIM
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
        message.frame_id = "base_link"
        message.backend = "stub_v0"
        message.max_entities = MAX_ENTITIES
        message.feature_dim = ENTITY_FEATURE_DIM
        message.entity_count = 0
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
    def __init__(self) -> None:
        super().__init__("trajectory_policy_stub")
        self.have_language = False
        self.have_visual = False
        self.have_task_features = False
        self.publisher = self.create_publisher(
            TrajectoryCandidates, "/vla/candidate_trajectories", RELIABLE_QOS
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
        self.create_timer(POLICY_PERIOD_SEC, self.publish_candidates)
        self.module_state = ModuleStatus.DEGRADED
        self.detail = "policy backend is a stub; zero candidates only"

    def on_language(self, _: TaskEmbedding) -> None:
        self.have_language = True

    def on_visual(self, _: VisualFeatures) -> None:
        self.have_visual = True

    def on_task_features(self, _: TaskFeatures) -> None:
        self.have_task_features = True

    def publish_candidates(self) -> None:
        message = TrajectoryCandidates()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.num_modes = NUM_MODES
        message.horizon = HORIZON
        message.dt = 0.2
        message.xy = [0.0] * (NUM_MODES * HORIZON * ACTION_DIM)
        message.logits = [0.0] * NUM_MODES
        message.logits[STOP_INDEX] = 1.0
        message.valid = False
        message.detail = "Day 1 placeholder; all six candidates are zero displacement"
        self.publisher.publish(message)
        self.input_ready = (
            self.have_language and self.have_visual and self.have_task_features
        )
        self.output_valid = False


class WorldModelStub(StatusNode):
    def __init__(self) -> None:
        super().__init__("world_model_stub")
        self.publisher = self.create_publisher(
            WorldModelEvaluation, "/vla/world_evaluation", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            TrajectoryCandidates,
            "/vla/candidate_trajectories",
            self.on_candidates,
            RELIABLE_QOS,
        )
        self.module_state = ModuleStatus.DEGRADED
        self.detail = "world model backend is a stub"

    def on_candidates(self, candidates: TrajectoryCandidates) -> None:
        message = WorldModelEvaluation()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.feasible = [False] * NUM_MODES
        message.total_cost = [1.0e6] * NUM_MODES
        message.rejection_reason = ["Day 1 placeholder"] * NUM_MODES
        message.valid = False
        message.detail = "invalid evaluation forces the selector into safe stop"
        self.publisher.publish(message)
        self.input_ready = len(candidates.xy) == NUM_MODES * HORIZON * ACTION_DIM
        self.output_valid = False


class TrajectorySelectorStub(StatusNode):
    def __init__(self) -> None:
        super().__init__("trajectory_selector_stub")
        self.have_candidates = False
        self.have_evaluation = False
        self.publisher = self.create_publisher(
            SelectedTrajectory, "/vla/selected_trajectory", RELIABLE_QOS
        )
        self.candidates_sub = self.create_subscription(
            TrajectoryCandidates,
            "/vla/candidate_trajectories",
            self.on_candidates,
            RELIABLE_QOS,
        )
        self.evaluation_sub = self.create_subscription(
            WorldModelEvaluation,
            "/vla/world_evaluation",
            self.on_evaluation,
            RELIABLE_QOS,
        )
        self.create_timer(CONTROL_PERIOD_SEC, self.publish_safe_stop)
        self.module_state = ModuleStatus.SAFE_STOP
        self.detail = "Day 1 selector is forced to safe stop"

    def on_candidates(self, _: TrajectoryCandidates) -> None:
        self.have_candidates = True

    def on_evaluation(self, _: WorldModelEvaluation) -> None:
        self.have_evaluation = True

    def publish_safe_stop(self) -> None:
        message = SelectedTrajectory()
        message.stamp_us = now_us(self)
        message.run_id = self.run_id
        message.selected_index = STOP_INDEX
        message.horizon = HORIZON
        message.dt = 0.2
        message.xy = [0.0] * (HORIZON * ACTION_DIM)
        message.safe_stop = True
        message.valid = True
        message.reason = "Day 1 forced safe stop"
        self.publisher.publish(message)
        self.input_ready = self.have_candidates and self.have_evaluation
        self.output_valid = True


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

    def on_selected(self, _: SelectedTrajectory) -> None:
        self.have_selected = True

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
        WorldModelStub(),
        TrajectorySelectorStub(),
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
        rclpy.shutdown()


def main(args=None) -> None:
    run_stack(include_language=True, args=args)


def main_without_language(args=None) -> None:
    run_stack(include_language=False, args=args)


if __name__ == "__main__":
    main()
