from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL import ImageDraw

from asv_vla.image_entity_perception import (
    BASE_FEATURE_DIM,
    COLOR_CALIBRATION_WIDTH,
    ENTITY_COUNT,
    FEATURE_DIM,
    LEGACY_MODEL_VERSION,
    OUTPUT_DIM,
    ImageEntityModel,
    calibrated_red_geometry,
    extract_image_features,
    save_model,
)


def test_image_features_are_fixed_and_finite() -> None:
    image = Image.new("RGB", (1280, 720), (12, 30, 60))
    features = extract_image_features(image)
    assert features.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(features))


def test_image_features_accept_rgb_arrays_and_include_moments() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[8:16, 10:20] = (220, 20, 20)
    features = extract_image_features(image)
    assert features.shape == (FEATURE_DIM,)
    assert FEATURE_DIM > BASE_FEATURE_DIM
    assert np.all(np.isfinite(features))


def test_legacy_model_round_trip_keeps_v1_feature_contract(tmp_path: Path) -> None:
    path = tmp_path / "legacy_perception.npz"
    save_model(
        path,
        feature_mean=np.zeros(BASE_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(BASE_FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((BASE_FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=np.ones(OUTPUT_DIM, dtype=np.float32),
        model_version=LEGACY_MODEL_VERSION,
    )
    model = ImageEntityModel.load(path)
    assert model.model_version == LEGACY_MODEL_VERSION
    assert len(model.predict(np.zeros((24, 32, 3), dtype=np.uint8))) == ENTITY_COUNT


def test_model_round_trip_and_no_velocity_output(tmp_path: Path) -> None:
    path = tmp_path / "perception.npz"
    save_model(
        path,
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=np.ones(OUTPUT_DIM, dtype=np.float32),
        metadata={"velocity_output": False},
    )
    model = ImageEntityModel.load(path)
    predictions = model.predict(Image.new("RGB", (1280, 720), (40, 50, 60)))
    assert len(predictions) == ENTITY_COUNT
    assert all(prediction.visible for prediction in predictions)
    assert all(
        np.isfinite(
            [prediction.relative_x, prediction.relative_y, prediction.relative_z]
        ).all()
        for prediction in predictions
    )


def test_calibrated_red_geometry_uses_image_centroid_sign_only() -> None:
    left = Image.new("RGB", (1280, 720), (20, 30, 40))
    ImageDraw.Draw(left).rectangle((180, 300, 300, 390), fill=(220, 20, 20))
    right = Image.new("RGB", (1280, 720), (20, 30, 40))
    ImageDraw.Draw(right).rectangle((980, 300, 1100, 390), fill=(220, 20, 20))

    left_valid, left_x, left_y, left_area, left_centroid = calibrated_red_geometry(left)
    right_valid, right_x, right_y, right_area, right_centroid = calibrated_red_geometry(right)

    assert left_valid and right_valid
    assert left_area == pytest.approx(right_area, rel=0.05)
    assert left_x == pytest.approx(right_x, rel=0.05)
    assert left_y > 0.0
    assert right_y < 0.0
    assert left_centroid[0] < COLOR_CALIBRATION_WIDTH / 2.0
    assert right_centroid[0] > COLOR_CALIBRATION_WIDTH / 2.0


def test_calibrated_red_geometry_fails_closed_without_red_component() -> None:
    valid, relative_x, relative_y, area, centroid = calibrated_red_geometry(
        Image.new("RGB", (1280, 720), (20, 30, 40))
    )
    assert not valid
    assert np.isnan(relative_x)
    assert np.isnan(relative_y)
    assert area == 0.0
    assert np.isnan(centroid[0])
    assert np.isnan(centroid[1])
