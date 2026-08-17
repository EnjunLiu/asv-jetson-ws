from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from interfaces.msg import EntityArray, EntityFeatures


MAX_ENTITIES = 16
FEATURE_DIM = 16
BACKEND_ID = "deterministic_entity_tensor"

# Each row is:
# [x, y, z, vx, vy, vz, planar_distance, bearing_sin, bearing_cos,
#  closing_speed, time_to_cpa, cpa_distance, is_target, is_risk,
#  color_red, color_blue]

# Continuous values are normalized and clipped to the ranges below.
POSITION_SCALE_M = 20.0
HEIGHT_SCALE_M = 5.0
VELOCITY_SCALE_MPS = 5.0
DEFAULT_RISK_HORIZON_SEC = 4.0
DEFAULT_RISK_RADIUS_M = 3.0


class EntityFeaturesError(RuntimeError):
    """Raised when entity data cannot be converted into valid features."""


@dataclass(frozen=True)
class EntityMetrics:
    distance_m: float
    bearing_sin: float
    bearing_cos: float
    closing_speed_mps: float
    time_to_cpa_sec: float
    cpa_distance_m: float
    is_risk: bool


@dataclass(frozen=True)
class EntityFeaturesResult:
    features: np.ndarray
    mask: np.ndarray
    entity_ids: tuple[str, ...]
    entity_count: int
    target_count: int
    risk_count: int
    dropped_count: int


def _clip(value: float, scale: float, low: float = -1.0) -> float:
    return float(np.clip(value / scale, low, 1.0))


def compute_entity_metrics(
    entity: Any,
    *,
    risk_horizon_sec: float = DEFAULT_RISK_HORIZON_SEC,
    risk_radius_m: float = DEFAULT_RISK_RADIUS_M,
) -> EntityMetrics:
    if risk_horizon_sec <= 0.0:
        raise ValueError("risk_horizon_sec must be positive")
    if risk_radius_m <= 0.0:
        raise ValueError("risk_radius_m must be positive")

    x = float(entity.relative_x)
    y = float(entity.relative_y)
    vx = float(entity.relative_velocity_x)
    vy = float(entity.relative_velocity_y)
    distance = math.hypot(x, y)
    if distance > 1.0e-9:
        bearing_sin = y / distance
        bearing_cos = x / distance
        closing_speed = -(x * vx + y * vy) / distance
    else:
        bearing_sin = 0.0
        bearing_cos = 1.0
        closing_speed = 0.0

    velocity_squared = vx * vx + vy * vy
    if velocity_squared > 1.0e-12:
        raw_time_to_cpa = -(x * vx + y * vy) / velocity_squared
        time_to_cpa = min(max(raw_time_to_cpa, 0.0), risk_horizon_sec)
    else:
        raw_time_to_cpa = math.inf
        time_to_cpa = risk_horizon_sec

    cpa_x = x + vx * time_to_cpa
    cpa_y = y + vy * time_to_cpa
    cpa_distance = math.hypot(cpa_x, cpa_y)
    is_risk = (
        closing_speed > 0.0
        and 0.0 < raw_time_to_cpa <= risk_horizon_sec
        and cpa_distance <= risk_radius_m
    )
    return EntityMetrics(
        distance_m=distance,
        bearing_sin=bearing_sin,
        bearing_cos=bearing_cos,
        closing_speed_mps=closing_speed,
        time_to_cpa_sec=time_to_cpa,
        cpa_distance_m=cpa_distance,
        is_risk=is_risk,
    )


def _validate_visible_entity(entity: Any) -> str:
    entity_id = str(entity.entity_id).strip()
    if not entity_id:
        raise EntityFeaturesError(
            "a valid visible entity has an empty entity_id"
        )
    values = (
        float(entity.relative_x),
        float(entity.relative_y),
        float(entity.relative_z),
        float(entity.relative_velocity_x),
        float(entity.relative_velocity_y),
        float(entity.relative_velocity_z),
    )
    if not all(math.isfinite(value) for value in values):
        raise EntityFeaturesError(
            f"entity {entity_id!r} contains NaN or Inf"
        )
    return entity_id


def _entity_row(
    candidate: tuple[Any, str, EntityMetrics],
    *,
    risk_horizon_sec: float,
) -> np.ndarray:
    entity, _, metrics = candidate
    color = str(entity.color).strip().casefold()
    is_red = color in {"red", "红", "红色"}
    is_blue = color in {"blue", "蓝", "蓝色"}
    return np.asarray(
        (
            _clip(float(entity.relative_x), POSITION_SCALE_M),
            _clip(float(entity.relative_y), POSITION_SCALE_M),
            _clip(float(entity.relative_z), HEIGHT_SCALE_M),
            _clip(float(entity.relative_velocity_x), VELOCITY_SCALE_MPS),
            _clip(float(entity.relative_velocity_y), VELOCITY_SCALE_MPS),
            _clip(float(entity.relative_velocity_z), VELOCITY_SCALE_MPS),
            _clip(metrics.distance_m, POSITION_SCALE_M, low=0.0),
            metrics.bearing_sin,
            metrics.bearing_cos,
            _clip(metrics.closing_speed_mps, VELOCITY_SCALE_MPS),
            _clip(metrics.time_to_cpa_sec, risk_horizon_sec, low=0.0),
            _clip(metrics.cpa_distance_m, POSITION_SCALE_M, low=0.0),
            1.0 if bool(entity.is_target) else 0.0,
            1.0 if metrics.is_risk else 0.0,
            1.0 if is_red else 0.0,
            1.0 if is_blue else 0.0,
        ),
        dtype=np.float32,
    )


def _candidate_sort_key(
    candidate: tuple[Any, str, EntityMetrics],
) -> tuple[Any, ...]:
    entity, entity_id, metrics = candidate
    if bool(entity.is_target):
        return 0, metrics.distance_m, entity_id
    if metrics.is_risk:
        return (
            1,
            metrics.cpa_distance_m,
            metrics.time_to_cpa_sec,
            metrics.distance_m,
            entity_id,
        )
    return 2, metrics.distance_m, entity_id


def build_entity_features(
    entities: Iterable[Any],
    *,
    max_entities: int = MAX_ENTITIES,
    risk_horizon_sec: float = DEFAULT_RISK_HORIZON_SEC,
    risk_radius_m: float = DEFAULT_RISK_RADIUS_M,
) -> EntityFeaturesResult:
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    if risk_horizon_sec <= 0.0:
        raise ValueError("risk_horizon_sec must be positive")
    if risk_radius_m <= 0.0:
        raise ValueError("risk_radius_m must be positive")

    candidates = []
    seen_ids = set()
    for entity in entities:
        if not bool(entity.valid) or not bool(entity.visible):
            continue
        entity_id = _validate_visible_entity(entity)
        if entity_id in seen_ids:
            raise EntityFeaturesError(
                f"duplicate valid visible entity_id {entity_id!r}"
            )
        seen_ids.add(entity_id)
        metrics = compute_entity_metrics(
            entity,
            risk_horizon_sec=risk_horizon_sec,
            risk_radius_m=risk_radius_m,
        )
        candidates.append((entity, entity_id, metrics))

    selected = sorted(candidates, key=_candidate_sort_key)[:max_entities]

    features = np.zeros((max_entities, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_entities, dtype=np.bool_)
    entity_ids = [""] * max_entities
    for index, candidate in enumerate(selected):
        features[index] = _entity_row(
            candidate,
            risk_horizon_sec=risk_horizon_sec,
        )
        mask[index] = True
        entity_ids[index] = candidate[1]

    if not np.all(np.isfinite(features)):
        raise EntityFeaturesError("entity features contain NaN or Inf")

    selected_targets = sum(
        bool(entity.is_target) for entity, _, _ in selected
    )
    selected_risks = sum(metrics.is_risk for _, _, metrics in selected)
    return EntityFeaturesResult(
        features=features,
        mask=mask,
        entity_ids=tuple(entity_ids),
        entity_count=len(selected),
        target_count=selected_targets,
        risk_count=selected_risks,
        dropped_count=max(0, len(candidates) - len(selected)),
    )


RELIABLE_QOS = (
    QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    if QoSProfile is not None
    else None
)


class EntityFeaturesNode(Node):
    def __init__(self) -> None:
        super().__init__("entity_features")
        self._last_stamp_us = 0
        self.max_entities = self.declare_parameter(
            "max_entities", MAX_ENTITIES
        ).value
        self.risk_horizon_sec = self.declare_parameter(
            "risk_horizon_sec", DEFAULT_RISK_HORIZON_SEC
        ).value
        self.risk_radius_m = self.declare_parameter(
            "risk_radius_m", DEFAULT_RISK_RADIUS_M
        ).value
        self.entities_topic = self.declare_parameter(
            "entities_topic", "/vla/tracked_entities"
        ).value
        if self.max_entities != MAX_ENTITIES:
            raise ValueError(f"max_entities must remain fixed at {MAX_ENTITIES}")
        if self.risk_horizon_sec <= 0.0:
            raise ValueError("risk_horizon_sec must be positive")
        if self.risk_radius_m <= 0.0:
            raise ValueError("risk_radius_m must be positive")

        self.publisher = self.create_publisher(
            EntityFeatures, "/vla/entity_features", RELIABLE_QOS
        )
        self.subscription = self.create_subscription(
            EntityArray, self.entities_topic, self.on_entities, RELIABLE_QOS
        )

    def _new_message(self, source: EntityArray) -> EntityFeatures:
        message = EntityFeatures()
        self._last_stamp_us = max(int(source.stamp_us), self._last_stamp_us + 1)
        message.stamp_us = self._last_stamp_us
        message.run_id = source.run_id
        message.scene_seed = source.scene_seed
        message.frame_index = source.frame_index
        message.frame_id = source.frame_id
        message.backend = BACKEND_ID
        message.max_entities = self.max_entities
        message.feature_dim = FEATURE_DIM
        message.entity_count = 0
        message.entity_ids = [""] * self.max_entities
        message.features = [0.0] * (self.max_entities * FEATURE_DIM)
        message.mask = [False] * self.max_entities
        message.valid = False
        message.instruction_id = str(source.instruction_id)
        message.instruction = str(source.instruction)
        message.detail = "UNINITIALIZED"
        return message

    def _publish_invalid(self, source: EntityArray, detail: str) -> None:
        message = self._new_message(source)
        message.detail = detail
        self.publisher.publish(message)
        self.get_logger().warning(detail)

    def on_entities(self, source: EntityArray) -> None:
        if not source.valid:
            self._publish_invalid(source, f"INVALID_SOURCE:{source.detail}")
            return
        if not source.run_id.strip():
            self._publish_invalid(source, "INVALID_RUN_ID: run_id is empty")
            return
        if source.frame_id != "base_link":
            self._publish_invalid(
                source,
                f"INVALID_FRAME: expected base_link, got {source.frame_id!r}",
            )
            return

        try:
            result = build_entity_features(
                source.entities,
                max_entities=self.max_entities,
                risk_horizon_sec=self.risk_horizon_sec,
                risk_radius_m=self.risk_radius_m,
            )
        except (EntityFeaturesError, ValueError) as exc:
            self._publish_invalid(source, f"{type(exc).__name__.upper()}:{exc}")
            return
        except Exception as exc:
            self._publish_invalid(
                source,
                f"UNEXPECTED_ENTITY_TENSOR_ERROR:{type(exc).__name__}:{exc}",
            )
            return

        message = self._new_message(source)
        message.entity_count = result.entity_count
        message.entity_ids = list(result.entity_ids)
        message.features = result.features.reshape(-1).tolist()
        message.mask = result.mask.tolist()
        message.valid = True
        message.detail = (
            f"OK:selected={result.entity_count};"
            f"targets={result.target_count};"
            f"risks={result.risk_count};"
            f"dropped={result.dropped_count}"
        )
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EntityFeaturesNode()
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
