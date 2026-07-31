"""ROS 2 recorder for synchronized UE5 FrameRecord v1 episodes."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import threading
import time
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, String

from asv_jetson_interfaces.msg import CameraFrame, UEASVState, UEEntityArray

from .episode import (
    EpisodeError,
    evaluate_episode,
    frame_key,
    make_manifest,
    validate_run_id_path,
    write_episode_frame,
    write_json_atomic,
)


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
)
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=4,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class EpisodeRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("episode_recorder")
        self.output_root = Path(
            self.declare_parameter(
                "output_root",
                str(Path.home() / "jetson_asv_ws" / "artifacts" / "day8_episode"),
            )
            .get_parameter_value()
            .string_value
        ).expanduser()
        self.task_text = (
            self.declare_parameter("task_text", "follow the red boat")
            .get_parameter_value()
            .string_value
            .strip()
        )
        self.execution_mode = (
            self.declare_parameter("execution_mode", "observation_only")
            .get_parameter_value()
            .string_value
            .strip()
        )
        self.collection_slot = (
            self.declare_parameter("collection_slot", "")
            .get_parameter_value()
            .string_value
            .strip()
        )
        self.layout_id = (
            self.declare_parameter("layout_id", "")
            .get_parameter_value()
            .string_value
            .strip()
        )
        self.motion_state = (
            self.declare_parameter("motion_state", "")
            .get_parameter_value()
            .string_value
            .strip()
        )
        self.expected_scene_seed = int(
            self.declare_parameter("expected_scene_seed", -1)
            .get_parameter_value()
            .integer_value
        )
        self.max_frames = int(
            self.declare_parameter("max_frames", 50)
            .get_parameter_value()
            .integer_value
        )
        self.exit_on_complete = bool(
            self.declare_parameter("exit_on_complete", False)
            .get_parameter_value()
            .bool_value
        )
        self.cache_size = int(
            self.declare_parameter("sync_cache_size", 64)
            .get_parameter_value()
            .integer_value
        )
        if not self.task_text:
            raise ValueError("task_text must be non-empty")
        if self.execution_mode not in {
            "observation_only",
            "ue5_kinematic_expert_v1",
            "legacy_thruster",
        }:
            raise ValueError(
                "execution_mode must be observation_only, "
                "ue5_kinematic_expert_v1 or legacy_thruster"
            )
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.cache_size < 4:
            raise ValueError("sync_cache_size must be at least 4")
        collection_values = (
            self.collection_slot,
            self.layout_id,
            self.motion_state,
        )
        if any(collection_values) and not all(collection_values):
            raise ValueError(
                "collection_slot, layout_id and motion_state must either "
                "all be set or all be empty"
            )

        self.task_pub = self.create_publisher(String, "/task/text", LATCHED_QOS)
        self.complete_pub = self.create_publisher(
            Bool, "/episode/recording_complete", LATCHED_QOS
        )
        self.create_subscription(
            UEASVState, "/ue/asv_state", self.on_state, RELIABLE_QOS
        )
        self.create_subscription(
            UEEntityArray, "/ue/entities", self.on_entities, RELIABLE_QOS
        )
        self.create_subscription(
            CameraFrame, "/ue/camera_frame", self.on_camera, SENSOR_QOS
        )
        self.create_timer(1.0, self.publish_task)

        self.states: OrderedDict[tuple[str, int, int, int], UEASVState] = (
            OrderedDict()
        )
        self.cameras: OrderedDict[tuple[str, int, int, int], CameraFrame] = (
            OrderedDict()
        )
        self.entities: OrderedDict[
            tuple[str, int, int, int], UEEntityArray
        ] = OrderedDict()
        self.episode_dir: Path | None = None
        self.run_id = ""
        self.scene_seed = 0
        self.frame_indices: list[int] = []
        self.stamp_values: list[int] = []
        self.started_monotonic = time.monotonic()
        self.finished = False
        self.finalized = False
        self.exit_timer = None
        self.exit_requested = threading.Event()
        self.cache_evictions = 0
        self.invalid_drops = 0
        self.get_logger().info(
            "EPISODE_RECORDER_READY waiting for synchronized "
            "/ue/asv_state + /ue/entities + /ue/camera_frame; "
            f"target_frames={self.max_frames}"
        )
        self.publish_task()

    def publish_task(self) -> None:
        message = String()
        message.data = self.task_text
        self.task_pub.publish(message)

    def on_state(self, message: UEASVState) -> None:
        self._store(self.states, message)

    def on_entities(self, message: UEEntityArray) -> None:
        self._store(self.entities, message)

    def on_camera(self, message: CameraFrame) -> None:
        self._store(self.cameras, message)

    def _store(self, cache: OrderedDict, message: Any) -> None:
        if self.finished:
            return
        key = frame_key(message)
        cache[key] = message
        cache.move_to_end(key)
        self._record_if_complete(key)
        self._trim_caches()

    def _trim_caches(self) -> None:
        for cache in (self.states, self.cameras, self.entities):
            while len(cache) > self.cache_size:
                cache.popitem(last=False)
                self.cache_evictions += 1

    def _create_episode(self, key: tuple[str, int, int, int]) -> None:
        run_id, scene_seed, _, _ = key
        validate_run_id_path(run_id)
        if (
            self.expected_scene_seed >= 0
            and scene_seed != self.expected_scene_seed
        ):
            raise EpisodeError(
                f"Scene_Seed mismatch: expected "
                f"{self.expected_scene_seed}, got {scene_seed}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        episode_dir = self.output_root / run_id
        if episode_dir.exists():
            raise EpisodeError(
                f"refusing to overwrite existing episode: {episode_dir}"
            )
        (episode_dir / "frames").mkdir(parents=True)
        (episode_dir / "camera").mkdir()
        self.episode_dir = episode_dir
        self.run_id = run_id
        self.scene_seed = scene_seed

        temporary_link = self.output_root / ".latest.tmp"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(episode_dir.name, target_is_directory=True)
        os.replace(temporary_link, self.output_root / "latest")
        self.get_logger().info(
            f"EPISODE_RECORDING_STARTED episode={episode_dir}"
        )

    def _record_if_complete(self, key: tuple[str, int, int, int]) -> None:
        state = self.states.get(key)
        camera = self.cameras.get(key)
        entities = self.entities.get(key)
        if state is None or camera is None or entities is None:
            return
        self.states.pop(key, None)
        self.cameras.pop(key, None)
        self.entities.pop(key, None)

        if not (
            state.valid
            and camera.valid
            and bool(camera.data)
            and entities.valid
            and entities.frame_id == "base_link"
        ):
            self.invalid_drops += 1
            self.get_logger().warning(
                f"EPISODE_DROP_INVALID_FRAME key={key} "
                f"ego={state.valid} camera={camera.valid} "
                f"entities={entities.valid}:{entities.detail}"
            )
            return

        try:
            if self.episode_dir is None:
                self._create_episode(key)
            if key[0] != self.run_id or key[1] != self.scene_seed:
                raise EpisodeError(
                    "Run_ID or Scene_Seed changed within the active episode"
                )
            record_path = write_episode_frame(
                self.episode_dir,
                task_text=self.task_text,
                task_stamp_us=0,
                state=state,
                camera=camera,
                entities=entities,
            )
        except Exception as exc:
            self.finished = True
            self.get_logger().error(
                f"EPISODE_RECORDING_FAILED:{type(exc).__name__}:{exc}"
            )
            self.finalize("failed")
            return

        self.frame_indices.append(int(camera.frame_index))
        self.stamp_values.append(int(camera.stamp_us))
        count = len(self.frame_indices)
        if count == 1 or count % 10 == 0:
            self.get_logger().info(
                f"EPISODE_RECORDED frame={camera.frame_index} "
                f"count={count}/{self.max_frames} path={record_path.name}"
            )
        if count >= self.max_frames:
            self.finished = True
            self.finalize("complete")

    def finalize(self, status: str) -> None:
        if self.finalized:
            return
        self.finalized = True
        if self.episode_dir is None:
            self.get_logger().warning(
                "EPISODE_RECORDING_EMPTY no synchronized frame was written"
            )
            return

        manifest = make_manifest(
            run_id=self.run_id,
            scene_seed=self.scene_seed,
            task_text=self.task_text,
            frame_indices=self.frame_indices,
            stamp_values=self.stamp_values,
            status=status,
            execution_mode=self.execution_mode,
            collection_slot=self.collection_slot,
            layout_id=self.layout_id,
            motion_state=self.motion_state,
        )
        manifest["cache_evictions"] = self.cache_evictions
        manifest["invalid_drops"] = self.invalid_drops
        manifest["recording_wall_time_s"] = round(
            time.monotonic() - self.started_monotonic, 3
        )
        write_json_atomic(self.episode_dir / "manifest.json", manifest)
        report = evaluate_episode(
            self.episode_dir,
            min_frames=self.max_frames if status == "complete" else 1,
            write_report=True,
        )
        complete = Bool()
        complete.data = bool(status == "complete" and report["passed"])
        self.complete_pub.publish(complete)
        marker = (
            "EPISODE_RECORDING_COMPLETE"
            if complete.data
            else "EPISODE_RECORDING_INCOMPLETE"
        )
        self.get_logger().info(
            f"{marker} episode={self.episode_dir} "
            f"frames={report['frame_count']} gaps={report['frame_gaps']} "
            f"quality_pass={report['passed']}"
        )
        if complete.data and self.exit_on_complete:
            self.get_logger().info(
                "EPISODE_RECORDER_EXIT requested after successful recording"
            )
            self.exit_timer = threading.Timer(
                0.25, self._shutdown_after_complete
            )
            self.exit_timer.daemon = True
            self.exit_timer.start()

    def _shutdown_after_complete(self) -> None:
        self.exit_requested.set()

    def destroy_node(self) -> bool:
        if not self.finalized:
            self.finalize("interrupted")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeRecorderNode()
    try:
        while rclpy.ok() and not node.exit_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
