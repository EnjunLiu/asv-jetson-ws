"""VLA policy inference node (ONNX, CPU).

Subscribes to encoder topics, runs the frozen ONNX policy, and publishes
one trajectory per frame to ``/vla/policy_trajectory``.

Key fixes vs the earlier PyTorch version:
- Uses ONNX Runtime on CPU (no CUDA OOM with the visual encoder).
- Pads entity tokens from the 2-token visual encoder output to the
  model's required 16-entity layout.
- Propagates modality ``valid`` flags into the policy input mask.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
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

# Maximum staleness for the language modality (seconds).
STALE_SEC = 1.0

# The visual encoder can lag the entity stream by many frames while it is
# running on the Jetson.  Keep enough keyed entries for that delay, but never
# let the policy node retain an unbounded stream of feature tensors.
SYNC_CACHE_SIZE = 256
SYNC_CACHE_TTL_SEC = 5.0
# Do not publish one invalid marker for every high-rate entity frame while the
# CUDA encoder is catching up.  Emit a fail-closed marker only after the
# synchronized stream has actually been quiet for this interval.
SYNC_FAIL_PUBLISH_PERIOD_SEC = 1.0

LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


FrameKey = tuple[str, int, int]


@dataclass(frozen=True)
class _SyncEntry:
    message: Any
    received_at: float


class FrameSyncCache:
    """Bounded, exact-identity cache for visual and task features.

    A policy input is executable only when both modalities are present under
    the same ``(run_id, scene_seed, frame_index)`` key.  The cache deliberately
    does not fall back to the latest message from either stream: an unmatched
    frame is a fail-closed condition, while a later callback can still find
    the counterpart that was delayed by the visual encoder.
    """

    def __init__(
        self,
        *,
        cache_size: int = SYNC_CACHE_SIZE,
        ttl_sec: float = SYNC_CACHE_TTL_SEC,
    ) -> None:
        if int(cache_size) <= 0:
            raise ValueError("cache_size must be positive")
        if not math.isfinite(float(ttl_sec)) or float(ttl_sec) <= 0.0:
            raise ValueError("ttl_sec must be finite and positive")
        self.cache_size = int(cache_size)
        self.ttl_sec = float(ttl_sec)
        self._visual: OrderedDict[FrameKey, _SyncEntry] = OrderedDict()
        self._entities: OrderedDict[FrameKey, _SyncEntry] = OrderedDict()
        self._active_run: tuple[str, int] | None = None

    @staticmethod
    def key_for(message: Any) -> FrameKey:
        return (
            str(message.run_id),
            int(message.scene_seed),
            int(message.frame_index),
        )

    @property
    def active_run(self) -> tuple[str, int] | None:
        return self._active_run

    @property
    def visual_size(self) -> int:
        return len(self._visual)

    @property
    def entity_size(self) -> int:
        return len(self._entities)

    def keys(self) -> tuple[FrameKey, ...]:
        """Return a snapshot of keys currently retained by either cache."""

        return tuple(dict.fromkeys((*self._visual.keys(), *self._entities.keys())))

    def clear(self) -> None:
        self._visual.clear()
        self._entities.clear()

    def _select_run(self, key: FrameKey) -> bool:
        run = (key[0], key[1])
        if self._active_run is None:
            self._active_run = run
            return False
        if run == self._active_run:
            return False
        # A new run/scene is a hard boundary.  Do not allow a prior run's
        # delayed visual tensor to pair with the new task stream.
        self.clear()
        self._active_run = run
        return True

    def _put(
        self,
        cache: OrderedDict[FrameKey, _SyncEntry],
        key: FrameKey,
        message: Any,
        received_at: float,
    ) -> None:
        cache[key] = _SyncEntry(message=message, received_at=float(received_at))
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def put_visual(
        self, message: Any, received_at: float | None = None
    ) -> tuple[FrameKey, bool]:
        key = self.key_for(message)
        switched = self._select_run(key)
        self._put(
            self._visual,
            key,
            message,
            time.monotonic() if received_at is None else received_at,
        )
        return key, switched

    def put_entities(
        self, message: Any, received_at: float | None = None
    ) -> tuple[FrameKey, bool]:
        key = self.key_for(message)
        switched = self._select_run(key)
        self._put(
            self._entities,
            key,
            message,
            time.monotonic() if received_at is None else received_at,
        )
        return key, switched

    def expire(self, now: float | None = None) -> int:
        current = time.monotonic() if now is None else float(now)
        removed = 0
        for cache in (self._visual, self._entities):
            for key, entry in tuple(cache.items()):
                if current - entry.received_at > self.ttl_sec:
                    del cache[key]
                    removed += 1
        return removed

    def match(
        self, key: FrameKey, now: float | None = None
    ) -> tuple[tuple[Any, Any] | None, str]:
        """Peek an exact pair and return ``(pair, status)``.

        ``status`` is ``MATCH``, ``NO_MATCH`` or ``STALE``.  Fresh pairs stay
        cached until :meth:`consume` so a temporarily unavailable language
        embedding does not discard a correctly synchronized frame.
        """

        current = time.monotonic() if now is None else float(now)
        if self._active_run is not None and (key[0], key[1]) != self._active_run:
            return None, "RUN_MISMATCH"

        visual = self._visual.get(key)
        entities = self._entities.get(key)
        if visual is None or entities is None:
            return None, "NO_MATCH"

        visual_stale = current - visual.received_at > self.ttl_sec
        entities_stale = current - entities.received_at > self.ttl_sec
        if visual_stale or entities_stale:
            if visual_stale:
                self._visual.pop(key, None)
            if entities_stale:
                self._entities.pop(key, None)
            return None, "STALE"

        return (visual.message, entities.message), "MATCH"

    def entity_for(self, key: FrameKey) -> Any | None:
        entry = self._entities.get(key)
        return entry.message if entry is not None else None

    def consume(self, key: FrameKey) -> None:
        self._visual.pop(key, None)
        self._entities.pop(key, None)


class VLAPolicyNode(Node):
    """Subscribes to encoder topics and publishes one trajectory per frame."""

    def __init__(self, model_path: str = "") -> None:
        super().__init__("vla_policy")

        self.declare_parameter("model_path", model_path)
        self.declare_parameter("checkpoint_path", "")  # deprecated alias

        sync_cache_size = int(
            self.declare_parameter("sync_cache_size", SYNC_CACHE_SIZE)
            .get_parameter_value()
            .integer_value
        )
        sync_cache_ttl_sec = float(
            self.declare_parameter("sync_cache_ttl_sec", SYNC_CACHE_TTL_SEC)
            .get_parameter_value()
            .double_value
        )
        self._frame_sync = FrameSyncCache(
            cache_size=sync_cache_size,
            ttl_sec=sync_cache_ttl_sec,
        )

        # Latest messages are retained for diagnostics and language freshness;
        # visual/entity inference always uses the exact keyed cache below.
        self._language: TaskEmbedding | None = None
        self._visual: VisualFeatures | None = None
        self._entities: TaskFeatures | None = None
        self._language_stamp = 0.0
        self._visual_stamp = 0.0
        self._entities_stamp = 0.0
        # Monotonic output stamp: the UE5 simulation clock can step
        # backwards headless; keep the published stamp strictly increasing
        # so downstream staleness checks never see a regression.
        self._last_out_stamp_us = 0
        self._last_inference_time = 0.0
        self._last_sync_fail_time = 0.0

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
        now = time.monotonic()
        self._language_stamp = now
        # A synchronized pair may have arrived before the transient-local
        # language embedding.  Retry only keys that are already in the
        # bounded cache; an unmatched entity remains fail-closed without
        # producing a burst of duplicate stop markers.
        for key in self._frame_sync.keys():
            self._maybe_infer(key, trigger="language", now=now)

    def _on_visual(self, msg: VisualFeatures) -> None:
        self._visual = msg
        now = time.monotonic()
        self._visual_stamp = now
        _, switched = self._frame_sync.put_visual(msg, received_at=now)
        if switched:
            self._recent_trajectories.clear()
        self._maybe_infer(
            self._frame_sync.key_for(msg), trigger="visual", now=now
        )

    def _on_entities(self, msg: TaskFeatures) -> None:
        self._entities = msg
        now = time.monotonic()
        self._entities_stamp = now
        key, switched = self._frame_sync.put_entities(msg, received_at=now)
        if switched:
            self._recent_trajectories.clear()
        # Trigger inference for this exact task frame.  If the visual tensor
        # is still being encoded, retain the key and retry when the matching
        # visual callback eventually arrives.  A throttled fail-closed marker
        # is emitted only if no synchronized inference has arrived for a full
        # interval; this avoids replacing good commands with one stop marker
        # per entity frame on the normal high-rate/slow-encoder path.
        self._maybe_infer(key, trigger="entities", now=now)

    def _new_output(self, ent: TaskFeatures) -> SelectedTrajectory:
        """Create an output carrying the task frame identity and monotonic stamp."""

        msg = SelectedTrajectory()
        if int(ent.stamp_us) > self._last_out_stamp_us:
            self._last_out_stamp_us = int(ent.stamp_us)
        else:
            self._last_out_stamp_us += 1
        msg.stamp_us = self._last_out_stamp_us
        if self._frame_seq < 20:
            self.get_logger().info(
                f"stamp trace: out={msg.stamp_us} ent={int(ent.stamp_us)}"
            )
        msg.run_id = str(ent.run_id)
        msg.scene_seed = int(ent.scene_seed)
        msg.frame_index = int(ent.frame_index)
        msg.frame_id = FRAME_ID
        msg.model_version = POLICY_MODEL_VERSION
        msg.dt = DT_SEC
        msg.horizon = HORIZON
        return msg

    def _publish_fail_closed(self, ent: TaskFeatures, reason: str) -> None:
        self._last_sync_fail_time = time.monotonic()
        msg = self._new_output(ent)
        msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
        msg.safe_stop = True
        msg.valid = False
        # Keep the old literal available for source-level contract checks;
        # normal mixed-frame input now fails as SYNC_NO_MATCH instead.
        if reason == "IDENTITY_MISMATCH":
            msg.reason = "IDENTITY_MISMATCH"
        else:
            msg.reason = reason
        self._recent_trajectories.clear()
        self._pub.publish(msg)
        self._frame_seq += 1

    def _maybe_infer(
        self,
        key: FrameKey | None = None,
        *,
        trigger: str = "entities",
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else float(now)
        if key is None:
            if self._entities is None:
                return
            key = self._frame_sync.key_for(self._entities)

        # Never substitute the latest visual/entity message.  The cache
        # returns a pair only for an exact identity match.
        ent_for_key = self._frame_sync.entity_for(key)
        pair, sync_status = self._frame_sync.match(key, now=current)
        if pair is None:
            if (
                trigger == "entities"
                and ent_for_key is not None
                and sync_status in {"STALE", "RUN_MISMATCH"}
                and (
                    current - self._last_inference_time
                    >= SYNC_FAIL_PUBLISH_PERIOD_SEC
                )
                and (
                    current - self._last_sync_fail_time
                    >= SYNC_FAIL_PUBLISH_PERIOD_SEC
                )
            ):
                reason = {
                    "STALE": "SYNC_STALE",
                    "RUN_MISMATCH": "SYNC_RUN_MISMATCH",
                }.get(sync_status, "SYNC_STALE")
                self._publish_fail_closed(ent_for_key, reason)
            return

        vis, ent = pair
        lang = self._language
        if lang is None or current - self._language_stamp > STALE_SEC:
            # Retain the synchronized pair.  The language callback retries it
            # after the transient-local embedding arrives.
            return

        # Consume only after all freshness/identity preconditions pass.  A
        # failed model/input/inference path below cannot accidentally replay
        # the same frame on every callback.
        self._frame_sync.consume(key)
        self._last_inference_time = current

        msg = self._new_output(ent)

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
