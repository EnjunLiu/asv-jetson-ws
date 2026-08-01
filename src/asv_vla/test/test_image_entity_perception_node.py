from __future__ import annotations

from io import BytesIO
import importlib
from pathlib import Path
import sys
import types

from PIL import Image


def _load_node_module():
    """Load the ROS adapter with tiny local stubs on non-ROS test hosts."""

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_qos = types.ModuleType("rclpy.qos")
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    interfaces = types.ModuleType("asv_jetson_interfaces")
    interfaces_msg = types.ModuleType("asv_jetson_interfaces.msg")

    class FakeNode:
        def __init__(self, *_args, **_kwargs):
            pass

    class FakeQoS:
        def __init__(self, **_kwargs):
            pass

    class FakeString:
        data = ""

    class FakeModuleStatus:
        STARTING = "starting"
        ERROR = "error"
        READY = "ready"

    class FakeUEEntity:
        def __init__(self):
            self.entity_id = ""
            self.relative_x = 0.0
            self.relative_y = 0.0
            self.relative_z = 0.0
            self.relative_velocity_x = 0.0
            self.relative_velocity_y = 0.0
            self.relative_velocity_z = 0.0
            self.velocity_valid = False
            self.bbox_x_min = 0.0
            self.bbox_y_min = 0.0
            self.bbox_x_max = 0.0
            self.bbox_y_max = 0.0
            self.bbox_valid = False
            self.visible = False
            self.valid = False

    class FakeUEEntityArray:
        def __init__(self):
            self.entities = []

    rclpy_node.Node = FakeNode
    rclpy_qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
    rclpy_qos.DurabilityPolicy = types.SimpleNamespace(
        TRANSIENT_LOCAL="transient_local"
    )
    rclpy_qos.QoSProfile = FakeQoS
    rclpy_qos.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE="reliable", BEST_EFFORT="best_effort"
    )
    std_msgs_msg.String = FakeString
    interfaces_msg.CameraFrame = type("CameraFrame", (), {})
    interfaces_msg.ModuleStatus = FakeModuleStatus
    interfaces_msg.UEEntity = FakeUEEntity
    interfaces_msg.UEEntityArray = FakeUEEntityArray
    std_msgs.msg = std_msgs_msg
    interfaces.msg = interfaces_msg
    rclpy.node = rclpy_node
    rclpy.qos = rclpy_qos
    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.node": rclpy_node,
            "rclpy.qos": rclpy_qos,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
            "asv_jetson_interfaces": interfaces,
            "asv_jetson_interfaces.msg": interfaces_msg,
        }
    )
    return importlib.import_module("asv_vla.image_entity_perception_node")


class _Prediction:
    def __init__(self, entity_id, *, x, y, visible):
        self.entity_id = entity_id
        self.relative_x = x
        self.relative_y = y
        self.relative_z = 0.0
        self.visible = visible
        self.confidence = 0.9


class _Model:
    model_version = "test_model"

    def predict(self, _image):
        return (
            _Prediction("target_red", x=5.0, y=100.0, visible=True),
            _Prediction("target_blue", x=5.0, y=0.0, visible=False),
        )


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _jpeg_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (1280, 720), (10, 20, 30)).save(stream, format="JPEG")
    return stream.getvalue()


def test_out_of_image_projection_does_not_kill_on_frame_and_fails_closed():
    module = _load_node_module()
    node = object.__new__(module.ImageEntityPerceptionNode)
    node.model = _Model()
    node.profile = module.CameraProfile()
    node.confidence_threshold = 0.55
    node.publisher = _Publisher()
    node.task_text = ""
    node.input_ready = False
    node.output_valid = False
    node.module_state = module.ModuleStatus.STARTING
    node.detail = ""
    node._trace_run_id = ""
    node._trace_count = 0
    node.trace_logs = []
    node.get_logger = lambda: types.SimpleNamespace(
        info=node.trace_logs.append
    )

    frame = types.SimpleNamespace(
        stamp_us=100,
        run_id="RUN",
        scene_seed=1,
        frame_index=0,
        valid=True,
        data=_jpeg_bytes(),
        encoding="jpeg",
    )
    node.on_frame(frame)

    message = node.publisher.messages[-1]
    assert message.valid is True
    assert len(message.entities) == 2
    out_of_image, hidden = message.entities
    assert out_of_image.valid is True
    assert out_of_image.bbox_valid is False
    assert out_of_image.visible is False
    assert hidden.bbox_valid is True
    assert hidden.visible is False
    assert node.output_valid is True
    assert len(node.trace_logs) == 1
    assert "PERCEPTION_TRACE" in node.trace_logs[0]
    assert "relative_x=5.000000" in node.trace_logs[0]
    assert "relative_y=100.000000" in node.trace_logs[0]
    assert "visible=False" in node.trace_logs[0]
    assert "bbox_valid=False" in node.trace_logs[0]


def test_perception_trace_formatter_keeps_required_red_diagnostics():
    module = _load_node_module()
    entity = types.SimpleNamespace(
        relative_x=3.25,
        relative_y=-0.75,
        visible=True,
        confidence=0.91,
        bbox_valid=True,
    )
    trace = module._format_perception_trace(
        run_id="RUN-TRACE",
        frame_index=7,
        sample_index=2,
        model_version="image_entity_ridge_v2",
        entity=entity,
    )
    assert trace.startswith("PERCEPTION_TRACE ")
    for token in (
        "run_id=RUN-TRACE",
        "frame_index=7",
        "model=image_entity_ridge_v2",
        "relative_x=3.250000",
        "relative_y=-0.750000",
        "visible=True",
        "confidence=0.910000",
        "bbox_valid=True",
    ):
        assert token in trace


def test_task_text_subscription_uses_transient_local_qos():
    node_source = (
        Path(__file__).resolve().parents[1]
        / "asv_vla"
        / "image_entity_perception_node.py"
    ).read_text(encoding="utf-8")
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in node_source
    assert 'String, "/task/text", self.on_task, TASK_QOS' in node_source
