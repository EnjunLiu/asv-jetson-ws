from dataclasses import dataclass
import numpy as np
import pytest

from vla.vla_policy_node import (
    ENTITY_COUNT,
    ENTITY_GEOMETRY_DIM,
    EntityFeaturesError,
    build_entity_features,
    compute_entity_metrics,
)

MAX_ENTITIES = ENTITY_COUNT
FEATURE_DIM = ENTITY_GEOMETRY_DIM


@dataclass
class FakeEntity:
    entity_id: str
    relative_x: float
    relative_y: float = 0.0
    relative_z: float = 0.0
    relative_velocity_x: float = 0.0
    relative_velocity_y: float = 0.0
    relative_velocity_z: float = 0.0
    color: str = ""
    is_target: bool = False
    visible: bool = True
    valid: bool = True


def test_empty_input_has_fixed_zero_shape():
    result = build_entity_features([])

    assert result.features.shape == (MAX_ENTITIES, FEATURE_DIM)
    assert result.features.dtype == np.float32
    assert result.mask.shape == (MAX_ENTITIES,)
    assert not result.mask.any()
    assert not result.features.any()
    assert result.entity_count == 0
    assert result.entity_ids == ("",) * MAX_ENTITIES


def test_target_is_retained_before_nearer_normal_entities():
    entities = [
        FakeEntity(f"normal_{index:02d}", 1.0 + index)
        for index in range(MAX_ENTITIES + 4)
    ]
    entities.append(
        FakeEntity("far_target", 100.0, color="red", is_target=True)
    )
    result = build_entity_features(entities)

    assert result.entity_ids[0] == "far_target"
    assert result.features[0, 12] == 1.0
    assert result.features[0, 14] == 1.0
    assert result.target_count == 1
    assert result.entity_count == MAX_ENTITIES
    assert result.dropped_count == 5


def test_collision_risk_is_retained_before_nearer_normal_entities():
    entities = [
        FakeEntity(f"normal_{index:02d}", 1.0 + index)
        for index in range(MAX_ENTITIES + 2)
    ]
    entities.append(
        FakeEntity(
            "closing_risk",
            8.0,
            relative_y=1.0,
            relative_velocity_x=-2.0,
        )
    )
    result = build_entity_features(entities)

    assert result.entity_ids[0] == "closing_risk"
    assert result.features[0, 13] == 1.0
    assert result.risk_count == 1


def test_target_precedes_risk_and_normal_order_is_nearest_first():
    result = build_entity_features([
        FakeEntity("normal_far", 6.0),
        FakeEntity("target", 10.0, is_target=True),
        FakeEntity(
            "risk",
            8.0,
            relative_velocity_x=-2.0,
        ),
        FakeEntity("normal_near", 2.0),
    ])

    assert result.entity_ids[:4] == (
        "target",
        "risk",
        "normal_near",
        "normal_far",
    )


def test_ties_are_broken_by_entity_id():
    result = build_entity_features([
        FakeEntity("b", 2.0),
        FakeEntity("a", 2.0),
    ])
    assert result.entity_ids[:2] == ("a", "b")


def test_invalid_and_invisible_entities_are_filtered():
    result = build_entity_features([
        FakeEntity("invalid", 1.0, valid=False),
        FakeEntity("hidden", 1.0, visible=False),
        FakeEntity("valid", 2.0),
    ])

    assert result.entity_ids[0] == "valid"
    assert result.entity_count == 1
    assert result.mask.tolist() == [True] + [False] * (MAX_ENTITIES - 1)


def test_nonfinite_valid_entity_fails_closed():
    with pytest.raises(EntityFeaturesError):
        build_entity_features([FakeEntity("bad", np.nan)])


def test_duplicate_visible_entity_id_fails_closed():
    with pytest.raises(EntityFeaturesError):
        build_entity_features([
            FakeEntity("duplicate", 1.0),
            FakeEntity("duplicate", 2.0),
        ])


def test_features_are_finite_bounded_and_encode_ros_bearing_and_color():
    result = build_entity_features([
        FakeEntity(
            "blue_left",
            1_000.0,
            relative_y=1_000.0,
            relative_z=-100.0,
            relative_velocity_x=-100.0,
            relative_velocity_y=100.0,
            relative_velocity_z=-100.0,
            color="blue",
        )
    ])
    row = result.features[0]

    assert np.all(np.isfinite(row))
    assert np.all(row >= -1.0)
    assert np.all(row <= 1.0)
    assert row[0] == 1.0
    assert row[1] == 1.0
    assert row[2] == -1.0
    assert row[7] == pytest.approx(2.0 ** -0.5)
    assert row[8] == pytest.approx(2.0 ** -0.5)
    assert row[15] == 1.0


def test_cpa_metrics_identify_approaching_risk_only():
    approaching = compute_entity_metrics(
        FakeEntity("approaching", 8.0, relative_velocity_x=-2.0)
    )
    receding = compute_entity_metrics(
        FakeEntity("receding", 8.0, relative_velocity_x=2.0)
    )

    assert approaching.is_risk
    assert approaching.time_to_cpa_sec == pytest.approx(4.0)
    assert approaching.cpa_distance_m == pytest.approx(0.0)
    assert not receding.is_risk
