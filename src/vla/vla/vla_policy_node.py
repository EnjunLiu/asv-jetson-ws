"""CUDA decision-head node for ASV displacement commands.

The perception chain owns image understanding and temporal velocity:

    camera -> image perception -> temporal tracker -> EntityFeatures

This node consumes only ``TaskEmbedding`` and the structured ``EntityFeatures``
message.  It publishes one bounded body-frame displacement for the next
control interval.  There is no trajectory horizon, global visual token,
entity crop token, or ego-state input at this boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from interfaces.msg import DesiredDisplacement, EntityFeatures, TaskEmbedding

from .trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, MAX_DISPLACEMENT_M
from .visual_standoff_guard import (
    GUARD_BACKSTOP,
    GUARD_FAIL_CLOSED,
    GUARD_HOLD,
    GUARD_PASS_THROUGH,
    GUARD_POLICY_DRIVEN,
    apply_standoff_guard,
)


DEFAULT_POLICY_BACKEND = "torch_cuda"
POLICY_MODEL_VERSION = "vla_torch_cuda_action_history"
ENTITY_COUNT = 16
LANGUAGE_DIM = 256
ENTITY_GEOMETRY_DIM = 16
STALE_SEC = 1.0
MIN_INFERENCE_INTERVAL_SEC = 0.2
SYNC_CACHE_SIZE = 256
SYNC_CACHE_TTL_SEC = 5.0
SYNC_FAIL_PUBLISH_PERIOD_SEC = 1.0
POLICY_MAX_STEP_M = MAX_DISPLACEMENT_M
POLICY_MAX_ACTION_DELTA_M = 0.05
POLICY_TRACE_LIMIT = 5
POLICY_AUDIT_PERIOD = 100

LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

FrameKey = tuple[str, int, int]
IDENTITY_FIELDS = ("run_id", "scene_seed", "frame_index")


@dataclass(frozen=True)
class _PendingAction:
    stamp_us: int
    action: tuple[float, float]


@dataclass(frozen=True)
class _SyncEntry:
    message: Any
    received_at: float


def _identity_tuple(message: Any) -> tuple[str, int, int] | None:
    """Return complete frame identity, or None when it cannot be trusted."""

    try:
        run_id = str(getattr(message, "run_id")).strip()
        scene_seed = int(getattr(message, "scene_seed"))
        frame_index = int(getattr(message, "frame_index"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not run_id or scene_seed <= 0 or frame_index < 0:
        return None
    return run_id, scene_seed, frame_index


def identity_mismatch_reason(
    language: Any,
    entities: Any,
) -> str | None:
    """Validate task identity without treating language as a camera frame.

    ``TaskEmbedding`` is task-level: its ``run_id`` identifies the encoder
    provenance and its ``stamp_us`` is publication time.  Neither is compared
    with the camera-frame identity.  The task text, and ``instruction_id``
    when both message interfaces provide it, are the synchronization keys.
    """

    language_instruction = str(getattr(language, "instruction", "")).strip()
    entity_instruction = str(getattr(entities, "instruction", "")).strip()
    if not language_instruction or not entity_instruction:
        return "IDENTITY_MISMATCH"
    if language_instruction != entity_instruction:
        return "IDENTITY_MISMATCH"
    language_instruction_id = str(
        getattr(language, "instruction_id", "")
    ).strip()
    entity_instruction_id = str(getattr(entities, "instruction_id", "")).strip()
    if (
        language_instruction_id
        and entity_instruction_id
        and language_instruction_id != entity_instruction_id
    ):
        return "IDENTITY_MISMATCH"
    return None


def entity_features_identity_reason(
    message: Any,
    previous_identity: FrameKey | None = None,
) -> str | None:
    """Validate EntityFeatures identity and monotonic same-run frame order."""

    identity = _identity_tuple(message)
    if identity is None:
        return "IDENTITY_MISMATCH"
    try:
        stamp_us = int(getattr(message, "stamp_us"))
    except (AttributeError, TypeError, ValueError):
        return "IDENTITY_MISMATCH"
    if stamp_us <= 0:
        return "IDENTITY_MISMATCH"
    if previous_identity is not None:
        if (
            identity[:2] == previous_identity[:2]
            and identity[2] <= previous_identity[2]
        ):
            return "IDENTITY_MISMATCH"
    return None


def bound_policy_displacement(
    displacement: Sequence[float] | np.ndarray,
    *,
    safe_stop: bool = False,
    valid: bool = True,
    max_step_m: float = POLICY_MAX_STEP_M,
) -> tuple[float, float] | None:
    """Validate and norm-bound one direct ``[desired_x, desired_y]`` action."""

    if safe_stop or not valid:
        return None
    try:
        maximum = float(max_step_m)
        values = np.asarray(displacement, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(maximum) or maximum < 0.0:
        return None
    if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
        return None
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm):
        return None
    if norm > maximum and norm > 0.0:
        values = values * (maximum / norm)
    return float(values[0]), float(values[1])


def smooth_policy_displacement(
    displacement: Sequence[float] | np.ndarray,
    *,
    previous_action: Sequence[float] | np.ndarray | None = None,
    max_step_m: float = POLICY_MAX_STEP_M,
    max_delta_m: float = POLICY_MAX_ACTION_DELTA_M,
) -> tuple[float, float] | None:
    """Apply a bounded per-frame action change around the previous command.

    A missing previous command starts from zero, so the first action uses the
    same delta bound as later frames while preserving the policy direction.
    """

    current = bound_policy_displacement(displacement, max_step_m=max_step_m)
    if current is None:
        return None
    try:
        maximum_delta = float(max_delta_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(maximum_delta) or maximum_delta < 0.0:
        return None
    if previous_action is None:
        previous = np.zeros(ACTION_DIM, dtype=np.float64)
    else:
        try:
            previous = np.asarray(previous_action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if previous.size != ACTION_DIM or not np.all(np.isfinite(previous)):
            return None
    current_array = np.asarray(current, dtype=np.float64)
    delta = current_array - previous
    delta_norm = float(np.linalg.norm(delta))
    if not math.isfinite(delta_norm):
        return None
    if delta_norm > maximum_delta and delta_norm > 0.0:
        current_array = previous + delta * (maximum_delta / delta_norm)
    return bound_policy_displacement(current_array, max_step_m=max_step_m)


class FrameSyncCache:
    """Bounded, scene-isolated cache for structured entity frames."""

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
    def entity_size(self) -> int:
        return len(self._entities)

    def keys(self) -> tuple[FrameKey, ...]:
        return tuple(self._entities.keys())

    def clear(self) -> None:
        self._entities.clear()

    def put_entities(
        self, message: Any, received_at: float | None = None
    ) -> tuple[FrameKey, bool]:
        key = self.key_for(message)
        run = (key[0], key[1])
        switched = self._active_run is not None and run != self._active_run
        if switched:
            self.clear()
        self._active_run = run
        self._entities[key] = _SyncEntry(
            message=message,
            received_at=time.monotonic() if received_at is None else float(received_at),
        )
        self._entities.move_to_end(key)
        while len(self._entities) > self.cache_size:
            self._entities.popitem(last=False)
        return key, switched

    def entity_for(self, key: FrameKey) -> Any | None:
        entry = self._entities.get(key)
        return entry.message if entry is not None else None

    def expire(self, now: float | None = None) -> int:
        current = time.monotonic() if now is None else float(now)
        removed = 0
        for key, entry in tuple(self._entities.items()):
            if current - entry.received_at > self.ttl_sec:
                del self._entities[key]
                removed += 1
        return removed

    def consume(self, key: FrameKey) -> None:
        self._entities.pop(key, None)


class VLAPolicyNode(Node):
    """Run the language-conditioned decision head at the camera cadence."""

    def __init__(self, model_path: str = "") -> None:
        super().__init__("vla_policy")
        self.declare_parameter("model_path", model_path)
        self.declare_parameter("device", "cuda")
        self._language_released = bool(
            self.declare_parameter("language_release_after_encode", False).value
        )
        self._backend = DEFAULT_POLICY_BACKEND
        policy_device = str(self.get_parameter("device").value).strip() or "cuda"
        self._frame_sync = FrameSyncCache(
            cache_size=int(self.declare_parameter("sync_cache_size", SYNC_CACHE_SIZE).value),
            ttl_sec=float(self.declare_parameter("sync_cache_ttl_sec", SYNC_CACHE_TTL_SEC).value),
        )
        self._language: TaskEmbedding | None = None
        self._language_stamp = 0.0
        self._language_task_key: str | None = None
        self._entities: EntityFeatures | None = None
        self._last_out_stamp_us = 0
        self._last_inference_time = 0.0
        self._last_sync_fail_time = 0.0
        self._frame_seq = 0
        self._policy_trace_count = 0
        self._policy_audit_events = 0
        self._policy_driven_count = 0
        self._backstop_count = 0
        self._hold_count = 0
        self._fail_closed_count = 0
        self._policy_stop_count = 0
        self._policy_audit_shutdown_logged = False
        self._last_audit_guard_reason = "none"
        self._last_audit_raw_dx, self._last_audit_raw_dy = "nan", "nan"
        self._last_audit_guarded_dx, self._last_audit_guarded_dy = "nan", "nan"
        self._last_audit_final_dx, self._last_audit_final_dy = "nan", "nan"
        self._inference_count = 0
        self._active_run: tuple[str, int] | None = None
        self._retired_runs: set[tuple[str, int]] = set()
        self._last_entity_identity: FrameKey | None = None
        self._last_entity_frame_index = -1
        self._last_inferred_frame_index = -1
        self._last_gate_frame_index = -1
        self._pending_actions: OrderedDict[FrameKey, _PendingAction] = OrderedDict()
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_action_valid = False
        self._previous_action_identity: FrameKey | None = None

        self._lang_sub = self.create_subscription(
            TaskEmbedding, "/vla/language_embedding", self._on_language, LANG_QOS
        )
        self._ent_sub = self.create_subscription(
            EntityFeatures, "/vla/entity_features", self._on_entities, 10
        )
        self._gate_sub = self.create_subscription(
            DesiredDisplacement,
            "/control/desired_displacement",
            self._on_gate_result,
            10,
        )
        self._pub = self.create_publisher(
            DesiredDisplacement, "/vla/policy_displacement", 10
        )
        self.create_timer(1.0, self._expire_cache)

        self._torch_runner = None
        self._policy_load_error = ""
        self._model_version = POLICY_MODEL_VERSION
        try:
            from .policy_model import DEFAULT_POLICY_MODEL_PATH, TorchPolicyRunner

            requested_path = str(self.get_parameter("model_path").value) or model_path
            self._torch_runner = TorchPolicyRunner.load(
                requested_path or DEFAULT_POLICY_MODEL_PATH,
                device=policy_device,
            )
            self.get_logger().info(
                f"POLICY_READY backend={self._backend} device={policy_device} "
                f"model={requested_path or DEFAULT_POLICY_MODEL_PATH} "
                "inputs=task_embedding+structured_entities+previous_action "
                "output=[desired_x,desired_y]"
            )
        except Exception as exc:
            self._policy_load_error = f"MODEL_LOAD_ERROR:{exc}"
            self.get_logger().error(
                f"POLICY_LOAD_ERROR backend={self._backend} error={self._policy_load_error}"
            )

        self._smooth_max_step_m = float(
            self.declare_parameter(
                "smoothing_max_step_m", POLICY_MAX_STEP_M
            ).value
        )
        self._smooth_max_delta_m = float(
            self.declare_parameter(
                "smoothing_max_delta_m", POLICY_MAX_ACTION_DELTA_M
            ).value
        )

    def _clear_previous_action(self) -> None:
        self._previous_action.fill(0.0)
        self._previous_action_valid = False
        self._previous_action_identity = None

    def _clear_control_history(self) -> None:
        self._pending_actions.clear()
        self._last_gate_frame_index = -1
        self._clear_previous_action()

    def _remember_previous_action(
        self, action: Sequence[float], *, identity: FrameKey
    ) -> None:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
            self._clear_previous_action()
            return
        bounded = bound_policy_displacement(
            values, max_step_m=self._smooth_max_step_m
        )
        if bounded is None:
            self._clear_previous_action()
            return
        self._previous_action[:] = bounded
        self._previous_action_valid = True
        self._previous_action_identity = identity

    def _on_gate_result(self, message: DesiredDisplacement) -> None:
        """Commit only the gate result for the current pending control frame."""

        identity = _identity_tuple(message)
        if identity is None or self._active_run is None:
            return
        if identity[:2] != self._active_run:
            return
        frame_index = identity[2]
        if (
            self._last_gate_frame_index >= 0
            and frame_index <= self._last_gate_frame_index
        ):
            self._clear_previous_action()
            return
        # A delayed result must not retroactively become the history for a
        # newer control frame that has already been inferred.
        if (
            self._last_inferred_frame_index >= 0
            and frame_index < self._last_inferred_frame_index
        ):
            self._pending_actions.pop(identity, None)
            self._clear_previous_action()
            return
        pending = self._pending_actions.pop(identity, None)
        if pending is None:
            self._last_gate_frame_index = frame_index
            self._clear_previous_action()
            return
        try:
            stamp_matches = int(message.stamp_us) == pending.stamp_us
        except (AttributeError, TypeError, ValueError):
            stamp_matches = False
        if not stamp_matches:
            self._last_gate_frame_index = frame_index
            self._clear_previous_action()
            return
        self._last_gate_frame_index = frame_index
        if not bool(message.valid) or bool(message.safe_stop):
            self._clear_previous_action()
            return
        self._remember_previous_action(
            (float(message.desired_x), float(message.desired_y)),
            identity=identity,
        )

    def _expire_cache(self) -> None:
        self._frame_sync.expire()

    @staticmethod
    def _audit_action(action: Sequence[float] | np.ndarray | None) -> tuple[str, str]:
        if action is None:
            return "nan", "nan"
        try:
            values = np.asarray(action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return "nan", "nan"
        if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
            return "nan", "nan"
        return f"{float(values[0]):.6f}", f"{float(values[1]):.6f}"

    def _maybe_policy_audit(self, *, force: bool = False, trigger: str = "periodic") -> None:
        events = int(self._policy_audit_events)
        if not force and (events == 0 or events % POLICY_AUDIT_PERIOD != 0):
            return
        self.get_logger().info(
            "POLICY_AUDIT "
            f"trigger={trigger} events={events} "
            f"policy_driven={int(self._policy_driven_count)} "
            f"backstop={int(self._backstop_count)} "
            f"hold={int(self._hold_count)} "
            f"fail_closed={int(self._fail_closed_count)} "
            f"policy_stop={int(self._policy_stop_count)} "
            f"guard_reason={self._last_audit_guard_reason} "
            f"raw_dx={self._last_audit_raw_dx} raw_dy={self._last_audit_raw_dy} "
            f"guarded_dx={self._last_audit_guarded_dx} "
            f"guarded_dy={self._last_audit_guarded_dy} "
            f"final_dx={self._last_audit_final_dx} "
            f"final_dy={self._last_audit_final_dy}"
        )

    def _record_guard_outcome(
        self,
        guard_reason: str,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
        guarded_action: Sequence[float] | np.ndarray | None = None,
        final_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if guard_reason == GUARD_POLICY_DRIVEN:
            self._policy_driven_count += 1
        elif guard_reason == GUARD_BACKSTOP:
            self._backstop_count += 1
        elif guard_reason == GUARD_HOLD:
            self._hold_count += 1
        self._last_audit_guard_reason = str(guard_reason)
        self._last_audit_raw_dx, self._last_audit_raw_dy = self._audit_action(
            raw_action
        )
        self._last_audit_guarded_dx, self._last_audit_guarded_dy = self._audit_action(
            guarded_action
        )
        self._last_audit_final_dx, self._last_audit_final_dy = self._audit_action(
            final_action
        )
        self._policy_audit_events += 1
        self._maybe_policy_audit()

    def _record_fail_closed(
        self,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
        reason: str = "",
    ) -> None:
        self._fail_closed_count += 1
        if str(reason) == "POLICY_STOP":
            self._policy_stop_count += 1
        self._last_audit_guard_reason = GUARD_FAIL_CLOSED
        self._last_audit_raw_dx, self._last_audit_raw_dy = self._audit_action(
            raw_action
        )
        self._last_audit_guarded_dx, self._last_audit_guarded_dy = "nan", "nan"
        self._last_audit_final_dx, self._last_audit_final_dy = "0.000000", "0.000000"
        self._policy_audit_events += 1
        self._maybe_policy_audit()

    def _trace_policy_decision(
        self,
        ent: EntityFeatures,
        *,
        policy_valid: bool,
        stop: bool,
        lang_valid: bool,
        ent_valid: bool,
        guard_result: str,
        guard_reason: str,
        raw_action: Sequence[float] | np.ndarray | None = None,
        guarded_action: Sequence[float] | np.ndarray | None = None,
        final_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if self._policy_trace_count >= POLICY_TRACE_LIMIT:
            return
        self._policy_trace_count += 1
        raw_dx, raw_dy = self._audit_action(raw_action)
        guarded_dx, guarded_dy = self._audit_action(guarded_action)
        final_dx, final_dy = self._audit_action(final_action)
        self.get_logger().info(
            "POLICY_TRACE "
            f"sample={self._policy_trace_count}/{POLICY_TRACE_LIMIT} "
            f"run_id={ent.run_id} scene_seed={int(ent.scene_seed)} "
            f"frame_index={int(ent.frame_index)} policy_valid={bool(policy_valid)} "
            f"stop={bool(stop)} lang_valid={bool(lang_valid)} "
            f"entity_valid={bool(ent_valid)} guard_result={guard_result} "
            f"guard_reason={guard_reason} "
            f"raw_dx={raw_dx} raw_dy={raw_dy} "
            f"guarded_dx={guarded_dx} guarded_dy={guarded_dy} "
            f"final_dx={final_dx} final_dy={final_dy} "
            f"policy_driven={int(self._policy_driven_count)} "
            f"backstop={int(self._backstop_count)} "
            f"hold={int(self._hold_count)} "
            f"fail_closed={int(self._fail_closed_count)} "
            f"policy_stop={int(self._policy_stop_count)}"
        )

    def _on_language(self, message: TaskEmbedding) -> None:
        task_key = str(getattr(message, "instruction", "")).strip()
        if self._language_task_key != task_key:
            # A new instruction invalidates both queued frames and gate
            # history. The next matching EntityFeatures frame starts cold.
            self._frame_sync.clear()
            self._last_entity_identity = None
            self._last_entity_frame_index = -1
            self._last_inferred_frame_index = -1
            self._clear_control_history()
        self._language_task_key = task_key
        self._language = message
        self._language_stamp = time.monotonic()
        for key in self._frame_sync.keys():
            self._maybe_infer(key, trigger="language")

    def _on_entities(self, message: EntityFeatures) -> None:
        self._entities = message
        now = time.monotonic()
        identity = _identity_tuple(message)
        continuity_reason = entity_features_identity_reason(
            message, self._last_entity_identity
        )
        if continuity_reason is not None:
            self._frame_sync.clear()
            self._clear_control_history()
            self._last_entity_identity = identity
            self._last_entity_frame_index = (
                identity[2] if identity is not None else -1
            )
            self._last_inferred_frame_index = -1
            self._publish_fail_closed(message, continuity_reason)
            return
        if identity is None:
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return

        run = identity[:2]
        if self._active_run is not None and run != self._active_run:
            if run in self._retired_runs:
                self._frame_sync.clear()
                self._clear_control_history()
                self._publish_fail_closed(message, "IDENTITY_MISMATCH")
                return
            self._retired_runs.add(self._active_run)

        if (
            self._language is not None
            and identity_mismatch_reason(self._language, message) is not None
        ):
            self._frame_sync.clear()
            self._clear_control_history()
            self._last_entity_identity = identity
            self._last_entity_frame_index = identity[2]
            self._last_inferred_frame_index = -1
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return

        key, switched = self._frame_sync.put_entities(message, received_at=now)
        if switched or self._active_run != run:
            self._active_run = run
            self._last_entity_frame_index = -1
            self._last_inferred_frame_index = -1
            self._last_entity_identity = None
            self._clear_control_history()
        elif (
            self._last_entity_frame_index >= 0
            and identity[2] <= self._last_entity_frame_index
        ):
            self._clear_control_history()
        self._last_entity_identity = identity
        self._last_entity_frame_index = identity[2]
        self._inference_count += 1
        if self._inference_count <= 5 or self._inference_count % 100 == 0:
            self.get_logger().info(
                f"ENT_IN_TRACE count={self._inference_count} "
                f"frame_index={int(message.frame_index)} valid={bool(message.valid)} "
                f"detail={str(message.detail)[:80]}"
            )
        self._maybe_infer(key, trigger="entities", now=now)

    def _new_output(self, ent: EntityFeatures) -> DesiredDisplacement:
        message = DesiredDisplacement()
        stamp = int(ent.stamp_us)
        self._last_out_stamp_us = max(stamp, self._last_out_stamp_us + 1)
        message.stamp_us = self._last_out_stamp_us
        message.run_id = str(ent.run_id)
        message.scene_seed = int(ent.scene_seed)
        message.frame_index = int(ent.frame_index)
        message.frame_id = FRAME_ID
        message.source = self._model_version
        message.step_dt = DT_SEC
        return message

    def _publish_fail_closed(
        self,
        ent: EntityFeatures,
        reason: str,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self._last_sync_fail_time = time.monotonic()
        self._record_fail_closed(raw_action=raw_action, reason=reason)
        self._clear_control_history()
        message = self._new_output(ent)
        message.desired_x = 0.0
        message.desired_y = 0.0
        message.safe_stop = True
        message.valid = False
        message.reason = str(reason)
        self._pub.publish(message)
        self._frame_seq += 1

    def _discard_old_pending_actions(self, identity: FrameKey) -> None:
        discarded_pending = False
        for pending_identity in tuple(self._pending_actions):
            if (
                pending_identity[:2] == identity[:2]
                and pending_identity[2] < identity[2]
            ):
                self._pending_actions.pop(pending_identity, None)
                discarded_pending = True
        # A discarded pending action means its gate result was not received;
        # a frame gap alone does not invalidate a committed action.
        if discarded_pending:
            self._clear_previous_action()

    def _maybe_infer(
        self,
        key: FrameKey,
        *,
        trigger: str,
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else float(now)
        if current - self._last_inference_time < MIN_INFERENCE_INTERVAL_SEC:
            return
        ent = self._frame_sync.entity_for(key)
        if ent is None:
            return
        if self._language is None:
            return
        if (
            current - self._language_stamp > STALE_SEC
            and not self._language_released
        ):
            return
        identity_reason = identity_mismatch_reason(self._language, ent)
        if identity_reason is not None:
            self._frame_sync.consume(key)
            self._clear_control_history()
            self._last_inference_time = current
            self._last_inferred_frame_index = int(ent.frame_index)
            self._publish_fail_closed(ent, identity_reason)
            return
        self._discard_old_pending_actions(key)
        self._frame_sync.consume(key)
        self._last_inference_time = current
        message = self._new_output(ent)
        if (
            self._last_inferred_frame_index >= 0
            and int(ent.frame_index) < self._last_inferred_frame_index
        ):
            self._clear_previous_action()
        self._last_inferred_frame_index = int(ent.frame_index)

        if self._torch_runner is None:
            self._publish_fail_closed(ent, self._policy_load_error or "NO_MODEL_LOADED")
            return

        try:
            previous_action_valid = bool(
                self._previous_action_valid
                and self._previous_action_identity is not None
                and self._previous_action_identity[:2] == key[:2]
                and self._previous_action_identity[2] < key[2]
            )
            inputs = self._build_inputs(
                self._language,
                ent,
                previous_action=self._previous_action,
                previous_action_valid=previous_action_valid,
            )
            action, stop_logit, valid_mask = self._torch_runner.run(inputs)
            action = np.asarray(action, dtype=np.float32)
            stop_logit = np.asarray(stop_logit, dtype=np.float32)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if (
                action.shape != (1, ACTION_DIM)
                or stop_logit.shape != (1, 1)
                or valid_mask.shape != (1,)
                or not np.all(np.isfinite(action))
                or not np.all(np.isfinite(stop_logit))
            ):
                raise ValueError(
                    f"invalid direct policy shapes action={action.shape} "
                    f"stop_logit={stop_logit.shape} valid_mask={valid_mask.shape}"
                )
        except Exception as exc:
            self._publish_fail_closed(ent, f"INFERENCE_ERROR:{exc}")
            return

        policy_valid = bool(valid_mask[0]) and bool(self._language.valid) and bool(ent.valid)
        stop = float(stop_logit[0, 0]) >= 0.0
        if stop or not policy_valid:
            self._publish_fail_closed(ent, "POLICY_STOP" if stop else "POLICY_INVALID")
            self._trace_policy_decision(
                ent,
                policy_valid=policy_valid,
                stop=stop,
                lang_valid=bool(self._language.valid),
                ent_valid=bool(ent.valid),
                guard_result="skipped",
                guard_reason="POLICY_STOP" if stop else "POLICY_INVALID",
                raw_action=action[0],
                final_action=(0.0, 0.0),
            )
            return

        displacement = bound_policy_displacement(
            action[0], valid=True, max_step_m=self._smooth_max_step_m
        )
        if displacement is None:
            self._publish_fail_closed(ent, "POLICY_ACTION_INVALID")
            return
        guarded, guard_reason = apply_standoff_guard(displacement, ent)
        if guarded is None:
            self._publish_fail_closed(
                ent, "VISUAL_TARGET_MISSING", raw_action=action[0]
            )
            self._trace_policy_decision(
                ent,
                policy_valid=policy_valid,
                stop=False,
                lang_valid=bool(self._language.valid),
                ent_valid=bool(ent.valid),
                guard_result=GUARD_FAIL_CLOSED,
                guard_reason=guard_reason,
                raw_action=action[0],
                final_action=(0.0, 0.0),
            )
            return

        current_action = np.asarray(guarded, dtype=np.float32)
        shaped = smooth_policy_displacement(
            current_action,
            previous_action=(
                self._previous_action if previous_action_valid else None
            ),
            max_step_m=self._smooth_max_step_m,
            max_delta_m=self._smooth_max_delta_m,
        )
        if shaped is None:
            self._publish_fail_closed(ent, "POLICY_SMOOTHING_INVALID")
            return

        self._record_guard_outcome(
            guard_reason,
            raw_action=action[0],
            guarded_action=guarded,
            final_action=shaped,
        )
        message.desired_x = float(shaped[0])
        message.desired_y = float(shaped[1])
        message.safe_stop = False
        message.valid = True
        message.reason = "POLICY_INFERRED_SMOOTHED"
        self._trace_policy_decision(
            ent,
            policy_valid=True,
            stop=False,
            lang_valid=bool(self._language.valid),
            ent_valid=bool(ent.valid),
            guard_result=guard_reason,
            guard_reason=guard_reason,
            raw_action=action[0],
            guarded_action=guarded,
            final_action=shaped,
        )
        self._pending_actions[key] = _PendingAction(
            stamp_us=int(message.stamp_us),
            action=(float(shaped[0]), float(shaped[1])),
        )
        self._pending_actions.move_to_end(key)
        while len(self._pending_actions) > SYNC_CACHE_SIZE:
            self._pending_actions.popitem(last=False)
        self._pub.publish(message)
        self._frame_seq += 1

    @staticmethod
    def _build_inputs(
        language: TaskEmbedding,
        entities: EntityFeatures,
        *,
        previous_action: Sequence[float] | np.ndarray | None = None,
        previous_action_valid: bool = False,
    ) -> dict[str, np.ndarray]:
        """Build the decision-head contract without image or ego fields."""

        identity_reason = identity_mismatch_reason(language, entities)
        if identity_reason is not None:
            raise ValueError(identity_reason)
        embedding = np.asarray(language.embedding, dtype=np.float32).reshape(-1)
        if embedding.size != LANGUAGE_DIM or not np.all(np.isfinite(embedding)):
            raise ValueError("task embedding must be finite float32[256]")
        if int(entities.max_entities) != ENTITY_COUNT:
            raise ValueError("EntityFeatures max_entities does not match policy contract")
        if int(entities.feature_dim) != ENTITY_GEOMETRY_DIM:
            raise ValueError("EntityFeatures feature_dim does not match policy contract")
        geometry = np.asarray(entities.features, dtype=np.float32).reshape(
            ENTITY_COUNT, ENTITY_GEOMETRY_DIM
        )
        mask = np.asarray(entities.mask, dtype=bool).reshape(ENTITY_COUNT)
        if not np.all(np.isfinite(geometry[mask])):
            raise ValueError("active structured entity features are non-finite")
        if not str(language.instruction).strip() == str(entities.instruction).strip() and str(entities.instruction).strip():
            raise ValueError("language/entity instruction mismatch")
        previous = np.zeros(ACTION_DIM, dtype=np.float32)
        if bool(previous_action_valid):
            if previous_action is None:
                raise ValueError("previous_action is missing while marked valid")
            previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
            if previous.size != ACTION_DIM or not np.all(np.isfinite(previous)):
                raise ValueError("previous action must be finite float32[2]")
        policy_valid = bool(language.valid) and bool(entities.valid) and bool(np.any(mask))
        return {
            "language": embedding.reshape(1, LANGUAGE_DIM),
            "entity_geometry": geometry.reshape(1, ENTITY_COUNT, ENTITY_GEOMETRY_DIM),
            "previous_action": previous.reshape(1, ACTION_DIM),
            "language_valid": np.asarray([bool(language.valid)], dtype=bool),
            "entity_geometry_mask": mask.reshape(1, ENTITY_COUNT),
            "previous_action_valid": np.asarray(
                [bool(previous_action_valid)], dtype=bool
            ),
            "policy_input_valid": np.asarray([policy_valid], dtype=bool),
        }

    def destroy_node(self) -> bool:
        if not self._policy_audit_shutdown_logged:
            self._policy_audit_shutdown_logged = True
            self._maybe_policy_audit(force=True, trigger="shutdown")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
