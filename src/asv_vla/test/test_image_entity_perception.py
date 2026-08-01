from pathlib import Path

import numpy as np
from PIL import Image

from asv_vla.image_entity_perception import (
    BASE_FEATURE_DIM,
    ENTITY_COUNT,
    FEATURE_DIM,
    LEGACY_MODEL_VERSION,
    OUTPUT_DIM,
    ImageEntityModel,
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
