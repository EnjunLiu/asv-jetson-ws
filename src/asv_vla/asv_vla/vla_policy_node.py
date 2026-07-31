"""Day 19 VLA policy inference node (ONNX, CPU).

Subscribes to encoder topics, runs the frozen ONNX policy, and publishes
one trajectory per frame to ``/vla/policy_trajectory``.

Key fixes vs the earlier PyTorch version:
- Uses ONNX Runtime on CPU (no CUDA OOM with the visual encoder).
- Pads entity tokens from the 2-token visual encoder output to the
  model's required 16-entity layout.
- Propagates modality ``valid`` flags into the policy input mask.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from asv_jetson_interfaces.msg import (
    SelectedTrajectory,
    TaskEmbedding,
    TaskFeatures,
    VisualFeatures,
)

from .trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, HORIZON

POLICY_MODEL_VERSION = "vla_onnx_cpu_v1"

# Model contract (frozen at export time).
ENTITY_COUNT = 16
LANGUAGE_DIM = 256
VISUAL_DIM = 576
ENTITY_GEOMETRY_DIM = 16
EGO_DIM = 2

# Maximum staleness for each modality (seconds).
STALE_SEC = 1.0

LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class VLAPolicyNode(Node):
    """Subscribes to encoder topics and publishes one trajectory per frame."""

    def __init__(self, model_path: str = "") -> None:
        super().__init__("vla_policy")

        self.declare_parameter("model_path", model_path)
        self.declare_parameter("checkpoint_path", "")  # deprecated alias

        # Latest encoder messages.
        self._language: TaskEmbedding | None = None
        self._visual: VisualFeatures | None = None
        self._entities: TaskFeatures | None = None
        self._language_stamp = 0.0
        self._visual_stamp = 0.0
        self._entities_stamp = 0.0

        # Subscribers.
        self._lang_sub = self.create_subscription(
            TaskEmbedding, "/vla/language_embedding", self._on_language, LANG_QOS
        )
        self._vis_sub = self.create_subscription(
            VisualFeatures, "/vla/visual_features", self._on_visual, 10
        )
        self._ent_sub = self.create_subscription(
            TaskFeatures, "/vla/task_features", self._on_entities, 10
        )

        # Publisher.
        self._pub = self.create_publisher(
            SelectedTrajectory, "/vla/policy_trajectory", 10
        )

        # Resolve model path: model_path takes precedence, then checkpoint_path.
        model_path = (
            str(self.get_parameter("model_path").get_parameter_value().string_value)
            or model_path
        )
        if not model_path:
            model_path = str(
                self.get_parameter("checkpoint_path")
                .get_parameter_value()
                .string_value
            )
        self._session = self._load_session(model_path) if model_path else None
        if self._session is not None:
            self.get_logger().info(f"VLA policy ONNX loaded from {model_path}")
        else:
            self.get_logger().warn("no ONNX model — publishing safe stop only")

        self._frame_seq = 0
        # Temporal smoothing: the raw policy output oscillates frame to
        # frame (the model is sharply sensitive to small camera changes);
        # the executed trajectory is the mean of the last 5 valid raw
        # outputs.  A STOP prediction clears the window and is published
        # as-is (fail-closed priority).
        self._smooth_window = 5
        self._recent_trajectories: deque[np.ndarray] = deque(maxlen=5)

    def _load_session(self, path: str) -> Any:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def _on_language(self, msg: TaskEmbedding) -> None:
        self._language = msg
        self._language_stamp = time.monotonic()

    def _on_visual(self, msg: VisualFeatures) -> None:
        self._visual = msg
        self._visual_stamp = time.monotonic()

    def _on_entities(self, msg: TaskFeatures) -> None:
        self._entities = msg
        self._entities_stamp = time.monotonic()
        # Entities arrive last (after visual), trigger inference.
        self._maybe_infer()

    def _maybe_infer(self) -> None:
        now = time.monotonic()
        lang = self._language
        vis = self._visual
        ent = self._entities

        if lang is None or vis is None or ent is None:
            return
        if any(
            now - t > STALE_SEC
            for t in (self._language_stamp, self._visual_stamp, self._entities_stamp)
        ):
            return

        msg = SelectedTrajectory()
        msg.stamp_us = int(ent.stamp_us)
        msg.run_id = str(ent.run_id)
        msg.frame_id = FRAME_ID
        msg.model_version = POLICY_MODEL_VERSION
        msg.dt = DT_SEC
        msg.horizon = HORIZON

        if self._session is None:
            msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
            msg.safe_stop = True
            msg.valid = True
            msg.reason = "NO_MODEL_LOADED"
            self._pub.publish(msg)
            return

        try:
            inputs = self._build_inputs(lang, vis, ent)
        except (ValueError, IndexError) as exc:
            msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
            msg.safe_stop = True
            msg.valid = False
            msg.reason = f"INPUT_ERROR:{exc}"
            self._pub.publish(msg)
            return

        try:
            outputs = self._session.run(None, inputs)
            traj, stop_logit, valid_mask = outputs
        except Exception as exc:
            msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
            msg.safe_stop = True
            msg.valid = False
            msg.reason = f"INFERENCE_ERROR:{exc}"
            self._pub.publish(msg)
            return

        traj = np.asarray(traj, dtype=np.float32).reshape(-1)
        stop = float(np.asarray(stop_logit).reshape(-1)[0])
        valid = bool(np.asarray(valid_mask).reshape(-1)[0])

        safe_stop = stop > 0.0
        if safe_stop or not valid:
            self._recent_trajectories.clear()
            msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
            msg.safe_stop = True
            msg.valid = bool(valid and lang.valid and vis.valid and ent.valid)
            msg.reason = "POLICY_STOP" if safe_stop else "POLICY_INFERRED"
            self._pub.publish(msg)
            self._frame_seq += 1
            return

        self._recent_trajectories.append(traj)
        if len(self._recent_trajectories) > 1:
            traj = np.mean(
                np.stack(list(self._recent_trajectories)), axis=0
            )

        msg.delta_p_xy = [float(v) for v in traj[: HORIZON * ACTION_DIM]]
        msg.safe_stop = False
        msg.valid = bool(valid and lang.valid and vis.valid and ent.valid)
        msg.reason = "POLICY_INFERRED"

        self._pub.publish(msg)
        self._frame_seq += 1

    def _build_inputs(
        self,
        lang: TaskEmbedding,
        vis: VisualFeatures,
        ent: TaskFeatures,
    ) -> dict[str, np.ndarray]:
        """Build ONNX inputs, padding entities to the frozen 16-entity layout."""

        # Language [256].
        lang_arr = np.array(lang.embedding, dtype=np.float32).reshape(1, LANGUAGE_DIM)

        # Visual: token 0 = global, tokens 1..16 = per-slot entity crops in
        # the task-tensor slot order (the encoder emits the full 17-token
        # layout with zero-filled slots).  Legacy 2-token senders fall back
        # to placing their single crop in slot 0.
        vf = np.array(vis.feature, dtype=np.float32)
        vis_dim = int(vis.feature_dim)
        global_token = vf[:vis_dim].reshape(1, VISUAL_DIM)
        entity_visual = np.zeros((1, ENTITY_COUNT, VISUAL_DIM), dtype=np.float32)
        entity_visual_mask = np.zeros((1, ENTITY_COUNT), dtype=bool)

        tok_count = int(vis.token_count)
        vis_mask = np.asarray(vis.mask, dtype=bool).reshape(-1)
        if tok_count >= 1 + ENTITY_COUNT and len(vf) >= vis_dim * (1 + ENTITY_COUNT):
            # Full per-slot layout.
            for slot in range(ENTITY_COUNT):
                start = vis_dim * (1 + slot)
                entity_visual[0, slot] = vf[start : start + vis_dim]
                entity_visual_mask[0, slot] = bool(
                    vis_mask[1 + slot] if len(vis_mask) > 1 + slot else vis.valid
                )
        elif tok_count >= 2 and len(vf) >= vis_dim * 2:
            # Legacy 2-token fallback: single crop in slot 0.
            crop = vf[vis_dim : vis_dim * 2]
            entity_visual[0, 0] = crop
            entity_visual_mask[0, 0] = bool(vis.valid)

        # Entity geometry: pad to 16 rows.
        ent_feat = np.array(ent.features, dtype=np.float32).reshape(
            int(ent.max_entities), int(ent.feature_dim)
        )
        entity_geometry = np.zeros(
            (1, ENTITY_COUNT, ENTITY_GEOMETRY_DIM), dtype=np.float32
        )
        entity_geometry_mask = np.zeros((1, ENTITY_COUNT), dtype=bool)
        n = min(int(ent.entity_count), ENTITY_COUNT)
        entity_geometry[0, :n] = ent_feat[:n, :ENTITY_GEOMETRY_DIM]
        # Zero out color truth columns 14/15 (policy must not see UE5 truth).
        entity_geometry[0, :n, 14] = 0.0
        entity_geometry[0, :n, 15] = 0.0
        ent_mask = np.array(ent.mask, dtype=bool).reshape(-1)
        entity_geometry_mask[0, :n] = ent_mask[:n]

        # Ego: stationary placeholder (no live ego topic in this launch).
        ego = np.zeros((1, EGO_DIM), dtype=np.float32)
        ego_valid = bool(ent.valid)

        # Global validity.
        language_valid = bool(lang.valid)
        global_visual_mask = bool(vis.valid)
        policy_input_valid = language_valid and global_visual_mask and ego_valid

        return {
            "language": lang_arr,
            "global_visual": global_token,
            "entity_visual": entity_visual,
            "entity_geometry": entity_geometry,
            "ego": ego,
            "language_valid": np.array([language_valid], dtype=bool),
            "global_visual_mask": np.array([global_visual_mask], dtype=bool),
            "entity_visual_mask": entity_visual_mask,
            "entity_geometry_mask": entity_geometry_mask,
            "ego_valid": np.array([ego_valid], dtype=bool),
            "policy_input_valid": np.array([policy_input_valid], dtype=bool),
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
